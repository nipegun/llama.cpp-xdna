#!/usr/bin/env -S PYTHONDONTWRITEBYTECODE=1 python3

"""Mixture-of-experts overlay (xdna2-moe 1) for GGML_OP_MUL_MAT_ID on XDNA2.

Multiplies C chunks of 1 to 24 rows (1 to 3 groups of 8 rows) by their chosen expert matrices in one command, across
all 16 input channels of the shims and all 32 cores (8 columns x 4 core rows), with 32-bit float sums and the weights
decoded in the cores from bfp16, Q4_K, Q5_K or Q6_K, optionally fusing the two segments of an expert (gate and up)
with SwiGLU in core 0 before emitting Y.

  moe_experts.py --dev npu2 --xclbin-path <file> --insts-path <file> [--compute 0|1] [--gemv-os 0|1]
"""

import argparse
import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
  Buffer,
  CascadeFlow,
  CompileTime,
  ExternalFunction,
  In,
  ObjectFifo,
  Out,
  Program,
  Runtime,
  Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from aie.utils import config
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

cCols = 8
cChannels = 2
cStreams = cCols * cChannels
cObjectBytes = 9216
cCoreBytes = 4608
cOutputs = 32
cTileK = 256
cMaxRowGroups = 3
cSums = cMaxRowGroups * 256  # 768 floats: up to 3 groups of 8 rows by 4 groups of 8 outputs
cRow = 32                    # 32 floats per emitted row of a block
cSegments = 2
cParamsWords = 16
cKeptX = 7 * cCoreBytes      # 32256 bytes: 6 objects of X (27648 bytes) + 1 object for the chunk table
cStage = 64 + 2048 + 2048 + 1792  # 5952 bytes: counters, factors, bases and carry
cChain = 1664                # floats of gate kept in core 0 for SwiGLU (up to 52 rows of 32 outputs)
cKernel = Path(__file__).resolve().parents[2] / "kernels" / "xdna2-moe.cc"
dFormats = {
  "bfp16": (0, 2304),
  "q4k": (1, 1152),
  "q5k": (2, 1408),
  "q6k": (3, 1696),
  "q8_0": (4, 2176),
  "q5_1": (5, 1536),
}


@iron.jit
def fMoeExperts(
  pIn: In,
  pOut: Out,
  *,
  pK: CompileTime[int] = 2048,
  pN: CompileTime[int] = 512,
  pFormat: CompileTime[str] = "q6k",
  pCompute: CompileTime[bool] = True,
  pGemvOs: CompileTime[bool] = True,
):
  vTilesCore = pK // cTileK // 2
  vBlocks = pN // cStreams // cOutputs
  vObjectsW = (vBlocks * 4 * vTilesCore * dFormats[pFormat][1] + cCoreBytes - 1) // cCoreBytes
  vObjectsX = (vTilesCore * 2304 + cCoreBytes - 1) // cCoreBytes
  vObjectsStream = 1 + vObjectsX + vObjectsW
  vOutFloats = cStreams * vBlocks * cRow

  vTypeObject = np.ndarray[(cObjectBytes,), np.dtype[np.uint8]]
  vTypeCore = np.ndarray[(cCoreBytes,), np.dtype[np.uint8]]
  vTypeSums = np.ndarray[(cSums,), np.dtype[np.float32]]
  vTypeRow = np.ndarray[(cRow,), np.dtype[np.float32]]
  vTypeKeptX = np.ndarray[(cKeptX,), np.dtype[np.uint8]]
  vTypeStage = np.ndarray[(cStage,), np.dtype[np.uint8]]
  vTypeChain = np.ndarray[(cChain,), np.dtype[np.float32]]
  vTypeParams = np.ndarray[(cParamsWords,), np.dtype[np.int32]]
  vTypeIn = np.ndarray[(cStreams, vObjectsStream * cObjectBytes), np.dtype[np.uint8]]
  vTypeOut = np.ndarray[(vOutFloats,), np.dtype[np.float32]]

  def fKernel(pName, pDefine, lArgs, lFlags=()):
    return ExternalFunction(
      pName,
      source_file=str(cKernel),
      arg_types=lArgs,
      include_dirs=[config.cxx_header_path()],
      compile_flags=[f"-DXDNA2_MOE_{pDefine}"] + list(lFlags),
    )

  vKernelParams = fKernel("xdna2_moe_params", "PARAMS", [vTypeCore, vTypeParams, vTypeKeptX, vTypeStage])
  vKernelChunk = fKernel("xdna2_moe_chunk", "CHUNK", [vTypeKeptX, vTypeStage, vTypeParams])
  vKernelX = fKernel("xdna2_moe_x", "X", [vTypeCore, vTypeKeptX, vTypeStage])
  vKernelGemv = fKernel(
    "xdna2_moe_gemv",
    "GEMV",
    [vTypeKeptX, vTypeCore, vTypeSums, vTypeStage, vTypeParams, np.int32, np.int32, np.int32],
    ([] if pCompute else ["-DXDNA2_MOE_NO_COMPUTE"]) + (["-Os"] if pGemvOs else []),
  )
  vKernelSend = fKernel("xdna2_moe_send", "SEND", [vTypeSums, vTypeStage])
  vKernelFinish = fKernel("xdna2_moe_finish", "FINISH", [vTypeSums, vTypeStage, vTypeChain, vTypeParams])
  vKernelEmit = fKernel("xdna2_moe_emit", "EMIT", [vTypeRow, vTypeSums, vTypeChain, vTypeStage])

  def fCore0(
    pInW,
    pOutY,
    pKeptX,
    pStage,
    pChain,
    pParams,
    pSums,
    pParamsKernel,
    pChunkKernel,
    pXKernel,
    pGemvKernel,
    pFinishKernel,
    pEmitKernel,
  ):
    vElemW = pInW.acquire(1)
    pParamsKernel(vElemW, pParams, pKeptX, pStage)
    pInW.release(1)
    for _ in range_(pParams[0]):
      pChunkKernel(pKeptX, pStage, pParams)
      for _ in range_(pParams[12]):
        vElemW = pInW.acquire(1)
        pXKernel(vElemW, pKeptX, pStage)
        pInW.release(1)
      for vSegment in range(cSegments):
        for _ in range_(pParams[4 + 3 * vSegment]):
          vElemW = pInW.acquire(1)
          pGemvKernel(
            pKeptX,
            vElemW,
            pSums,
            pStage,
            pParams,
            pParams[5 + 3 * vSegment],
            pParams[1],
            pParams[3 + 3 * vSegment],
          )
          for _ in range_(pParams[13]):
            pFinishKernel(pSums, pStage, pChain, pParams)
            for _ in range_(pParams[14]):
              vElemY = pOutY.acquire(1)
              pEmitKernel(vElemY, pSums, pChain, pStage)
              pOutY.release(1)
            pGemvKernel(
              pKeptX,
              vElemW,
              pSums,
              pStage,
              pParams,
              pParams[5 + 3 * vSegment],
              pParams[1],
              pParams[3 + 3 * vSegment],
            )
          pInW.release(1)

  def fCore1(
    pInW,
    pKeptX,
    pStage,
    pParams,
    pSums,
    pParamsKernel,
    pChunkKernel,
    pXKernel,
    pGemvKernel,
    pSendKernel,
  ):
    vElemW = pInW.acquire(1)
    pParamsKernel(vElemW, pParams, pKeptX, pStage)
    pInW.release(1)
    for _ in range_(pParams[0]):
      pChunkKernel(pKeptX, pStage, pParams)
      for _ in range_(pParams[12]):
        vElemW = pInW.acquire(1)
        pXKernel(vElemW, pKeptX, pStage)
        pInW.release(1)
      for vSegment in range(cSegments):
        for _ in range_(pParams[4 + 3 * vSegment]):
          vElemW = pInW.acquire(1)
          pGemvKernel(
            pKeptX,
            vElemW,
            pSums,
            pStage,
            pParams,
            pParams[5 + 3 * vSegment],
            pParams[1],
            pParams[3 + 3 * vSegment],
          )
          for _ in range_(pParams[13]):
            pSendKernel(pSums, pStage)
            pGemvKernel(
              pKeptX,
              vElemW,
              pSums,
              pStage,
              pParams,
              pParams[5 + 3 * vSegment],
              pParams[1],
              pParams[3 + 3 * vSegment],
            )
          pInW.release(1)

  lFifosW = []
  lFifosY = []
  lWorkers = []
  for vCol in range(cCols):
    for vChannel in range(cChannels):
      vStream = vCol * cChannels + vChannel
      vFifoW = ObjectFifo(vTypeObject, name=f"W_{vStream}", depth=2)
      vFifoY = ObjectFifo(vTypeRow, name=f"Y_{vStream}", depth=2)
      lFifosW.append(vFifoW)
      lFifosY.append(vFifoY)
      lHalves = vFifoW.cons().split(
        [0, cCoreBytes],
        tile=Tile(vCol, 1),
        obj_types=[vTypeCore, vTypeCore],
        names=[f"W_{vStream}_0", f"W_{vStream}_1"],
        depths=[2, 2],
      )
      vRow0 = 2 + 2 * vChannel
      lCore0 = [
        Buffer(vTypeKeptX, name=f"x_{vStream}_0"),
        Buffer(vTypeStage, name=f"stage_{vStream}_0"),
        Buffer(vTypeChain, name=f"chain_{vStream}_0"),
        Buffer(vTypeParams, name=f"params_{vStream}_0"),
        Buffer(vTypeSums, name=f"sums_{vStream}_0"),
      ]
      lCore1 = [
        Buffer(vTypeKeptX, name=f"x_{vStream}_1"),
        Buffer(vTypeStage, name=f"stage_{vStream}_1"),
        Buffer(vTypeParams, name=f"params_{vStream}_1"),
        Buffer(vTypeSums, name=f"sums_{vStream}_1"),
      ]
      vWorker0 = Worker(
        fCore0,
        [lHalves[0].cons(), vFifoY.prod()]
        + lCore0
        + [vKernelParams, vKernelChunk, vKernelX, vKernelGemv, vKernelFinish, vKernelEmit],
        tile=Tile(vCol, vRow0),
        stack_size=0xF00,
      )
      vWorker1 = Worker(
        fCore1,
        [lHalves[1].cons()] + lCore1 + [vKernelParams, vKernelChunk, vKernelX, vKernelGemv, vKernelSend],
        tile=Tile(vCol, vRow0 + 1),
        stack_size=0xF00,
      )
      CascadeFlow(vWorker1, vWorker0)
      lWorkers += [vWorker0, vWorker1]

  def fLinear(pTotal, pOffset, pCount):
    return TensorAccessPattern((pTotal,), pOffset, [1, 1, 1, pCount], [0, 0, 0, 1])

  def fSequence(pBufferIn, pBufferOut, lHandlesW, lHandlesY):
    vTotalIn = cStreams * vObjectsStream * cObjectBytes
    for vStream in range(cStreams):
      lHandlesY[vStream].drain(
        pBufferOut,
        tap=fLinear(vOutFloats, vStream * vBlocks * cRow, vBlocks * cRow),
        wait=True,
      )
    for vStream in range(cStreams):
      lHandlesW[vStream].fill(
        pBufferIn,
        tap=fLinear(vTotalIn, vStream * vObjectsStream * cObjectBytes, vObjectsStream * cObjectBytes),
      )

  vRuntime = Runtime(
    fSequence,
    [
      vTypeIn,
      vTypeOut,
      [vFifo.prod(tile=Tile(vStream // cChannels, 0)) for vStream, vFifo in enumerate(lFifosW)],
      [vFifo.cons(tile=Tile(vStream // cChannels, 0)) for vStream, vFifo in enumerate(lFifosY)],
    ],
  )
  return Program(iron.get_current_device(), vRuntime, workers=lWorkers).resolve_program()


def fMakeArgparser():
  vParser = argparse.ArgumentParser(prog="Mixture-of-experts overlay for XDNA2")
  add_compile_args(vParser, dev_choices=("npu2",), with_emit_mlir=True)
  vParser.add_argument("-K", type=int, default=2048)
  vParser.add_argument("-N", type=int, default=512)
  vParser.add_argument("--format", choices=sorted(dFormats), default="q6k")
  vParser.add_argument("--compute", type=int, choices=[0, 1], default=1)
  vParser.add_argument("--gemv-os", type=int, choices=[0, 1], default=1)
  return vParser


def fValidate(pOpts):
  if pOpts.K < 2 * cTileK or pOpts.K % (2 * cTileK) != 0:
    sys.exit(f"K must be a positive multiple of {2 * cTileK}")
  if pOpts.N < cStreams * cOutputs or pOpts.N % (cStreams * cOutputs) != 0:
    sys.exit(f"N must be a positive multiple of {cStreams * cOutputs}")


def fCompileKwargs(pOpts):
  return dict(pK=pOpts.K, pN=pOpts.N, pFormat=pOpts.format, pCompute=bool(pOpts.compute), pGemvOs=bool(pOpts.gemv_os))


def fRunAndVerify(pOpts):
  sys.exit("this design is only compiled here: it is run with llama-xdna-run or llama-xdna-gemv")


def fMain():
  vOpts = fMakeArgparser().parse_args()
  run_design_cli(
    fMoeExperts,
    vOpts,
    compile_kwargs=fCompileKwargs,
    run_and_verify=fRunAndVerify,
    device=lambda pOpts: from_name(pOpts.dev, n_cols=None),
    validate=fValidate,
  )


if __name__ == "__main__":
  try:
    fMain()
  except Exception as vError:
    print("error:", repr(vError))
    sys.exit(1)
  finally:
    pass
