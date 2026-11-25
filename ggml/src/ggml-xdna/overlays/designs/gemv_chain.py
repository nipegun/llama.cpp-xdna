#!/usr/bin/env -S PYTHONDONTWRITEBYTECODE=1 python3

"""Chained overlay for generating tokens (xdna2-gemv 9, step 2 of the plan to run everything on the NPU).

Like gemv_stream.py, Y = X x W^T for a few rows, the weights in up to 4 segments of outputs in bfp16, Q4_K, Q5_K or
Q6_K, through the sixteen input channels of the shims in objects of 9216 bytes, and all 32 cores. What changes:

  - The two cores fed by a channel share the work along K: the memory tile of the column gives the first half of
    every object (4608 bytes) to the core of the even row (2 or 4, core 0 of the pair) and the second half to the
    core above it (core 1). Core 0 makes the first half of the tiles of K of every block of 32 outputs (4 groups of
    8) and core 1 the second; core 1 sends its sums over the cascade (row 3 to 2, 5 to 4), and core 0 adds them,
    finishes them and sends the block of Y to the shim. Every core keeps only its half of X, so nothing is sent twice.
  - The parameters come in the first object of every command (the same in both halves), so no runtime parameters,
    no barriers and no writes of registers: a table of 5 rows of 32 integers, one row for each of up to 4 phases and
    the last one with the number of phases.
  - A command runs its phases one after the other: the objects of the prologue of a phase (the residual of its
    epilogue, and X in bfp16 or in floats, or the rows of an RMS norm and its weights), then its blocks.
  - Version 6: the pieces of the blocks of a segment follow each other in the objects with no padding between blocks,
    so a core loops over the objects of a segment, and a block can end inside one: the kernel stops there and says
    so in the parameters, the core finishes (core 0) or sends (core 1) the block in a loop of 0 or 1 turns, and the
    kernel goes on in the same object. Core 0 sums a block in a buffer of its own and writes Y when it finishes it.
    X takes 7 objects of a core (32256 bytes; 3 before), with the buffer of the epilogues cut to 1664 floats.
  - Version 7: an object of Y is one row of a block, its 32 outputs (128 bytes), and core 0 sends as many as the phase
    has rows: with one row Y is the outputs in their order, and the X of a phase reads it in line. Before, an object of
    Y was the whole block with room for 8 rows (1 KB), 7 of every 8 floats sent for nothing with one row.
  - Version 8: a phase can run the recurrence of a layer of linear attention (the gated delta net of ggml), one head
    of 128 values per core: the states of the heads come through W, 9 rows to the half of an object of a core, and go
    back through Y, row by row (core 1 hands its rows to core 0 over the cascade), then the outputs. The RMS norm of a
    prologue takes 1 / K from the host (item 25), which took the division of floats out of the cores. --test gdn
    builds the command of 32 heads, with the states in a third buffer.
  - Version 9: a phase of a layer of linear attention (item 26 = 1): its prologue (mode 3) brings the inputs of the
    pair (the head of keys s and the value heads s and s + 16) to X[1] to X[3], core 1 convolves the 512 channels of
    the pair, takes their SiLU, the L2 norms of q and k and the gates of both heads, and sends to core 0 what its head
    needs over the cascade (xdna2_chain_conv and xdna2_chain_conv_recv, in a loop of 0 or 1 turns after the prologue),
    and the recurrence then ends every head with its gated norm. --test la builds the command of 32 heads, head s on
    core 0 and head s + 16 on core 1 of stream s, with the states in a third buffer.

The kernel and the layout of every object are described in ggml/src/ggml-xdna/kernels/xdna2-gemv-chain.cc.

  gemv_chain.py --dev npu2 --xclbin-path <file> --insts-path <file> [--test single|ffn] [-K <k> -N <n>]
                [--format <format>] [--compute 0]

The overlay is the same for every size and test: only the instruction stream of the test command changes. With
--test single (the default), one phase of one row, X in floats and one segment of N outputs in one format; host
buffers In [16 streams][objects of 9216 bytes] and Out [16 streams][blocks of 32 outputs][rows][32 outputs] in floats
(with one row, the outputs in their order). With --test ffn, the three phases of a feed-forward block of Qwen3.5 4B in Q4_K (see cFfn):
4096 to 2560 plus a residual, the RMS norm and 2 x 9216 with SwiGLU, and 9216 to 2560 plus h, every phase taking its
X from the Y of the one before through Out, which holds the Y of the three phases one after the other (the second as
[segment][stream][block]). --compute 0 builds the kernel that only counts the pieces, to time the path of the data.
The data and the check of gemv_chain_check.py are those of version 5 (a block per object): the overlay is checked with
llama-xdna-gemv, whose instruction streams for these tests are those of this script.
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
    TaskGroup,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile, from_name
from aie.utils import config
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

cCols = 8          # columns of the array
cChannels = 2      # input channels of a shim: streams of a column
cStreams = cCols * cChannels
cObjectBytes = 9216  # bytes of an object of a stream
cCoreBytes = 4608    # and of a core: half of it
cOutputs = 32      # outputs of a block
cTileK = 256       # values of K of a tile
cSums = 256        # 32-bit floats of the sums of a block: 4 groups of 8 outputs by 8 rows
cRow = 32          # and of an object of Y: one row of them, [4 groups][8 outputs]
cSegments = 2      # segments of outputs a phase can have (the table keeps room for 4; with more, the host gives
                   # bfp16), which keeps the program of a core under its 16 KB
cPhases = 4        # phases of a command
cParamsColumns = 32
cKeptX = 7 * 4608  # bytes of X a core keeps (its half of K): whole halves of objects, as X in bfp16 comes
cStage = 2304 + 64 + 2048 + 2048 + 1792  # the staging buffer of the kernel: X of a tile, the counters, the factors and
                                         # bases of the sub-blocks of a piece, and a piece carried over
cChain = 1728      # floats of the residual (320), h (320) and the gate (1088) of the epilogues
cChunk = cCoreBytes // 4  # floats of X of a core in an object
cKernel = Path(__file__).resolve().parents[2] / "kernels" / "xdna2-gemv-chain.cc"  # ggml/src/ggml-xdna/kernels
# the recurrence of a layer of linear attention (version 8): one head of 128 values per core, its state of 128 rows of
# 128 floats through W, 9 rows to an object of a core and 15 objects a head (the last one with 2 rows and padding)
cGdnDim = 128
cGdnRows = cCoreBytes // (cGdnDim * 4)            # 9
cGdnObjectsHead = (cGdnDim + cGdnRows - 1) // cGdnRows  # 15
cGdnObjectsItem = 24                              # item of a phase: objects of the state of a core (0: no recurrence)
cGdnHeads = 2 * cStreams                          # heads of the gdn test: one per core
cGdnPad = 1024                                    # floats of room after the states: the last object reads 7 rows past
cLaItem = 26                                      # item of a phase: 1 in a phase of linear attention (version 9)
cLaObjects = 3                                    # and the objects of its prologue
dFormats = {"bfp16": (0, 2304), "q4k": (1, 1152), "q5k": (2, 1408), "q6k": (3, 1696), "q8_0": (4, 2176)}  # number, bytes of a piece


# the feed-forward test: K and N of the three phases (the second has two segments of N, the gate and the up)
cFfn = ((4096, 2560), (2560, 9216), (9216, 2560))


def fFfnLayout():
  # objects of every stream in the three parts of In, and floats of the three Y in Out
  (vK1, vN1), (vK2, vN2), (vK3, vN3) = cFfn
  vPart1 = 1 + 1 + fObjectsX(vK1) + fObjectsPerSegment("q4k", vK1, vN1 // cStreams // cOutputs)
  vPart2 = fObjectsX(vK2) + 2 * fObjectsPerSegment("q4k", vK2, vN2 // cStreams // cOutputs)
  vPart3 = fObjectsPerSegment("q4k", vK3, vN3 // cStreams // cOutputs)
  vY1 = vN1 // cOutputs * cRow
  vY2 = 2 * vN2 // cOutputs * cRow
  vY3 = vN3 // cOutputs * cRow
  return vPart1, vPart2, vPart3, vY1, vY2, vY3


def fTilesPerCore(pK):
  return pK // cTileK // 2


def fObjectsX(pK):
  # objects of X in floats of a core, one row
  return (pK // 2 + cChunk - 1) // cChunk


def fObjectsPerSegment(pFormat, pK, pBlocks):
  # objects of a core for a segment of pBlocks blocks: 4 groups of 8 outputs by its tiles, a piece each, the blocks
  # one after the other
  vBytes = pBlocks * 4 * fTilesPerCore(pK) * dFormats[pFormat][1]
  return (vBytes + cCoreBytes - 1) // cCoreBytes


@iron.jit
def fGemvChain(
    pIn: In,
    pOut: Out,
    *,
    pK: CompileTime[int] = 2560,
    pN: CompileTime[int] = 2560,
    pFormat: CompileTime[str] = "q4k",
    pCompute: CompileTime[bool] = True,
    pGemvOs: CompileTime[bool] = True,
    pTest: CompileTime[str] = "single",
):
  vBlocks = pN // cStreams // cOutputs  # blocks of every stream in the test command
  vObjectsStream = 1 + fObjectsX(pK) + fObjectsPerSegment(pFormat, pK, vBlocks)
  vOutFloats = cStreams * vBlocks * cRow
  if pTest == "ffn":
    vPart1, vPart2, vPart3, vY1, vY2, vY3 = fFfnLayout()
    vObjectsStream = vPart1 + vPart2 + vPart3
    vOutFloats = vY1 + vY2 + vY3 + 2 * cChunk  # and room past the end for the reads of the last chunks
  if pTest == "gdn":
    vObjectsStream = 2                  # the parameters and the inputs of the heads: the states come from their buffer
    vOutFloats = cGdnHeads * cGdnDim    # the outputs of the heads, [head][128]
  if pTest == "la":
    vObjectsStream = 1 + cLaObjects     # the parameters and the prologue: the states come from their buffer
    vOutFloats = cGdnHeads * cGdnDim    # the outputs of the heads after their gated norm, [head][128]
  vTypeState = np.ndarray[(cGdnHeads * cGdnDim * cGdnDim + cGdnPad,), np.dtype[np.float32]]
  vTypeObject = np.ndarray[(cObjectBytes,), np.dtype[np.uint8]]
  vTypeCore = np.ndarray[(cCoreBytes,), np.dtype[np.uint8]]
  vTypeSums = np.ndarray[(cSums,), np.dtype[np.float32]]
  vTypeRow = np.ndarray[(cRow,), np.dtype[np.float32]]
  vTypeKeptX = np.ndarray[(cKeptX,), np.dtype[np.uint8]]
  vTypeStage = np.ndarray[(cStage,), np.dtype[np.uint8]]
  vTypeChain = np.ndarray[(cChain,), np.dtype[np.float32]]
  vTypeParams = np.ndarray[(cPhases + 1, cParamsColumns), np.dtype[np.int32]]
  vTypeIn = np.ndarray[(cStreams, vObjectsStream * cObjectBytes), np.dtype[np.uint8]]
  vTypeOut = np.ndarray[(vOutFloats,), np.dtype[np.float32]]

  def fKernel(pName, pDefine, pArgs, pFlags=()):
    return ExternalFunction(
      pName,
      source_file=str(cKernel),
      arg_types=pArgs,
      include_dirs=[config.cxx_header_path()],
      compile_flags=[f"-DXDNA2_CHAIN_{pDefine}"] + list(pFlags),
    )

  kParams = fKernel("xdna2_chain_params", "PARAMS", [vTypeCore, vTypeParams, vTypeStage])
  kPhase = fKernel("xdna2_chain_phase", "PHASE", [vTypeStage, vTypeParams])
  kPrologue = fKernel("xdna2_chain_prologue", "PROLOGUE",
                      [vTypeCore, vTypeKeptX, vTypeStage, vTypeChain, vTypeParams, np.int32], ["-Os"])
  kGemv = fKernel("xdna2_chain_gemv", "GEMV",
                  [vTypeKeptX, vTypeCore, vTypeSums, vTypeStage, vTypeParams, np.int32, np.int32, np.int32, np.int32],
                  ([] if pCompute else ["-DXDNA2_CHAIN_NO_COMPUTE"]) + (["-Os"] if pGemvOs else []))
  kSend = fKernel("xdna2_chain_send", "SEND", [vTypeSums, vTypeStage])
  kFinish = fKernel("xdna2_chain_finish", "FINISH", [vTypeSums, vTypeStage, vTypeChain, vTypeParams])
  kEmit = fKernel("xdna2_chain_emit", "EMIT", [vTypeRow, vTypeSums, vTypeStage])
  kGdn0 = fKernel("xdna2_chain_gdn0", "GDN0", [vTypeCore, vTypeKeptX, vTypeSums, vTypeStage, vTypeParams],
                  [] if pCompute else ["-DXDNA2_CHAIN_NO_COMPUTE"])
  kGdn1 = fKernel("xdna2_chain_gdn1", "GDN1", [vTypeCore, vTypeKeptX, vTypeSums, vTypeStage, vTypeParams],
                  [] if pCompute else ["-DXDNA2_CHAIN_NO_COMPUTE"])
  kGdnEmit = fKernel("xdna2_chain_gdn_emit", "GDN_EMIT", [vTypeRow, vTypeCore, vTypeSums, vTypeStage])
  kConv = fKernel("xdna2_chain_conv", "CONV", [vTypeKeptX, vTypeParams], [] if pCompute else ["-DXDNA2_CHAIN_NO_COMPUTE"])
  kConvRecv = fKernel("xdna2_chain_conv_recv", "CONV_RECV", [vTypeKeptX, vTypeParams],
                      [] if pCompute else ["-DXDNA2_CHAIN_NO_COMPUTE"])

  # The cores of a pair: the same loops, over the objects of every segment. The kernel stops where a block ends and
  # writes pParams[4][1] = 1: then core 0 finishes the block (core 1 sends it over the cascade) in a loop of that one
  # turn and sends its rows, an object of Y each, and the kernel goes on in the same object. Both cores sum a block in a
  # buffer of their own.
  # Version 8, the recurrence: after its segments (none in a phase of the recurrence) a phase runs the objects of the
  # states of the heads of the core (pParams[phase][24], 0 without it), 9 rows each: the kernel updates the next row and
  # writes in pParams[4][1] how many objects of Y core 0 sends for it (8: its row and that of core 1, which comes over
  # the cascade; 16 after the last row of a head, with the outputs of both cores; 0 for the padding of the last object).
  def fCore0(pInW, pOutY, pKeptX, pStage, pChain, pParams, pSums, pParamsKernel, pPhaseKernel, pPrologueKernel,
             pGemvKernel, pFinishKernel, pEmitKernel, pGdnKernel, pGdnEmitKernel, pConvRecvKernel):
    vElemW = pInW.acquire(1)
    pParamsKernel(vElemW, pParams, pStage)
    pInW.release(1)
    for vPhase in range_(pParams[cPhases, 0]):
      pPhaseKernel(pStage, pParams)
      for _ in range_(pParams[vPhase, 1]):
        vElemW = pInW.acquire(1)
        if pCompute:
          pPrologueKernel(vElemW, pKeptX, pStage, pChain, pParams, 0)
        pInW.release(1)
      for _ in range_(pParams[vPhase, cLaItem]):
        pConvRecvKernel(pKeptX, pParams)
      for vSegment in range(cSegments):
        for _ in range_(pParams[vPhase, 9 + 4 * vSegment]):
          vElemW = pInW.acquire(1)
          pGemvKernel(pKeptX, vElemW, pSums, pStage, pParams, pParams[vPhase, 2], pParams[vPhase, 10 + 4 * vSegment],
                      pParams[vPhase, 3], pParams[vPhase, 8 + 4 * vSegment])
          for _ in range_(pParams[cPhases, 1]):
            pFinishKernel(pSums, pStage, pChain, pParams)
            for _ in range_(pParams[vPhase, 2]):
              vElemY = pOutY.acquire(1)
              pEmitKernel(vElemY, pSums, pStage)
              pOutY.release(1)
            pGemvKernel(pKeptX, vElemW, pSums, pStage, pParams, pParams[vPhase, 2],
                        pParams[vPhase, 10 + 4 * vSegment], pParams[vPhase, 3], pParams[vPhase, 8 + 4 * vSegment])
          pInW.release(1)
      for _ in range_(pParams[vPhase, cGdnObjectsItem]):
        vElemW = pInW.acquire(1)
        for _ in range_(cGdnRows):
          pGdnKernel(vElemW, pKeptX, pSums, pStage, pParams)
          for _ in range_(pParams[cPhases, 1]):
            vElemY = pOutY.acquire(1)
            pGdnEmitKernel(vElemY, vElemW, pSums, pStage)
            pOutY.release(1)
        pInW.release(1)

  def fCore1(pInW, pKeptX, pStage, pChain, pParams, pSums, pParamsKernel, pPhaseKernel, pPrologueKernel, pGemvKernel,
             pSendKernel, pGdnKernel, pConvKernel):
    vElemW = pInW.acquire(1)
    pParamsKernel(vElemW, pParams, pStage)
    pInW.release(1)
    for vPhase in range_(pParams[cPhases, 0]):
      pPhaseKernel(pStage, pParams)
      for _ in range_(pParams[vPhase, 1]):
        vElemW = pInW.acquire(1)
        if pCompute:
          pPrologueKernel(vElemW, pKeptX, pStage, pChain, pParams, 1)
        pInW.release(1)
      for _ in range_(pParams[vPhase, cLaItem]):
        pConvKernel(pKeptX, pParams)
      for vSegment in range(cSegments):
        for _ in range_(pParams[vPhase, 9 + 4 * vSegment]):
          vElemW = pInW.acquire(1)
          pGemvKernel(pKeptX, vElemW, pSums, pStage, pParams, pParams[vPhase, 2], pParams[vPhase, 10 + 4 * vSegment],
                      pParams[vPhase, 3], pParams[vPhase, 8 + 4 * vSegment])
          for _ in range_(pParams[cPhases, 1]):
            pSendKernel(pSums, pStage)
            pGemvKernel(pKeptX, vElemW, pSums, pStage, pParams, pParams[vPhase, 2],
                        pParams[vPhase, 10 + 4 * vSegment], pParams[vPhase, 3], pParams[vPhase, 8 + 4 * vSegment])
          pInW.release(1)
      for _ in range_(pParams[vPhase, cGdnObjectsItem]):
        vElemW = pInW.acquire(1)
        for _ in range_(cGdnRows):
          pGdnKernel(vElemW, pKeptX, pSums, pStage, pParams)
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
      # the memory tile of the column: the first half of every object to core 0, the second half to core 1
      lHalves = vFifoW.cons().split(
        [0, cCoreBytes],
        tile=Tile(vCol, 1),
        obj_types=[vTypeCore, vTypeCore],
        names=[f"W_{vStream}_0", f"W_{vStream}_1"],
        depths=[2, 2],
      )
      vRow0 = 2 + 2 * vChannel
      lCore0 = [Buffer(vTypeKeptX, name=f"x_{vStream}_0"), Buffer(vTypeStage, name=f"stage_{vStream}_0"),
                Buffer(vTypeChain, name=f"chain_{vStream}_0"), Buffer(vTypeParams, name=f"params_{vStream}_0"),
                Buffer(vTypeSums, name=f"sums_{vStream}_0")]
      lCore1 = [Buffer(vTypeKeptX, name=f"x_{vStream}_1"), Buffer(vTypeStage, name=f"stage_{vStream}_1"),
                Buffer(vTypeChain, name=f"chain_{vStream}_1"), Buffer(vTypeParams, name=f"params_{vStream}_1"),
                Buffer(vTypeSums, name=f"sums_{vStream}_1")]
      vWorker0 = Worker(fCore0, [lHalves[0].cons(), vFifoY.prod()] + lCore0 + [kParams, kPhase, kPrologue, kGemv,
                                                                               kFinish, kEmit, kGdn0, kGdnEmit,
                                                                               kConvRecv],
                        tile=Tile(vCol, vRow0), stack_size=0xF00)
      vWorker1 = Worker(fCore1, [lHalves[1].cons()] + lCore1 + [kParams, kPhase, kPrologue, kGemv, kSend, kGdn1,
                                                                kConv],
                        tile=Tile(vCol, vRow0 + 1), stack_size=0xF00)
      CascadeFlow(vWorker1, vWorker0)
      lWorkers += [vWorker0, vWorker1]

  def fLinear(pTotal, pOffset, pCount):
    return TensorAccessPattern((pTotal,), pOffset, [1, 1, 1, pCount], [0, 0, 0, 1])

  def fXFromY(pBase, pK):
    # X of the next phase from the Y of a segment of one row (output o at pBase + o): object m is [chunk m of the first
    # half of K | chunk m of the second half], 1152 floats each, 8 at a time
    return TensorAccessPattern((vOutFloats,), pBase, [fObjectsX(pK), 2, cChunk // 8, 8],
                               [cChunk, pK // 2, 8, 1])

  def fSequence(pBufferIn, pBufferOut, lHandlesW, lHandlesY):
    vTotalIn = cStreams * vObjectsStream * cObjectBytes
    if pTest != "ffn":
      for vStream in range(cStreams):
        lHandlesY[vStream].drain(pBufferOut, tap=fLinear(vOutFloats, vStream * vBlocks * cRow, vBlocks * cRow),
                                 wait=True)
      for vStream in range(cStreams):
        lHandlesW[vStream].fill(pBufferIn, tap=fLinear(vTotalIn, vStream * vObjectsStream * cObjectBytes,
                                                       vObjectsStream * cObjectBytes))
      return
    (vK1, vN1), (vK2, vN2), (vK3, vN3) = cFfn
    vBlocks1 = vN1 // cStreams // cOutputs
    vBlocks2 = vN2 // cStreams // cOutputs
    vBlocks3 = vN3 // cStreams // cOutputs
    lParts = [(0, vPart1), (vPart1, vPart2), (vPart1 + vPart2, vPart3)]
    for vPhase in range(3):
      vGroup = TaskGroup()
      for vStream in range(cStreams):
        if vPhase == 0:
          vTap = fLinear(vOutFloats, vStream * vBlocks1 * cRow, vBlocks1 * cRow)
        elif vPhase == 1:
          # [segment][stream][block]: the gate, then the up
          vTap = TensorAccessPattern((vOutFloats,), vY1 + vStream * vBlocks2 * cRow, [1, 1, 2, vBlocks2 * cRow],
                                     [0, 0, cStreams * vBlocks2 * cRow, 1])
        else:
          vTap = fLinear(vOutFloats, vY1 + vY2 + vStream * vBlocks3 * cRow, vBlocks3 * cRow)
        lHandlesY[vStream].drain(pBufferOut, tap=vTap, wait=True, group=vGroup)
      for vStream in range(cStreams):
        if vPhase == 1:
          lHandlesW[vStream].fill(pBufferOut, tap=fXFromY(0, vK2), group=vGroup)
        elif vPhase == 2:
          lHandlesW[vStream].fill(pBufferOut, tap=fXFromY(vY1 + vY2 // 2, vK3), group=vGroup)
        vFirst, vCount = lParts[vPhase]
        lHandlesW[vStream].fill(pBufferIn, tap=fLinear(vTotalIn, (vStream * vObjectsStream + vFirst) * cObjectBytes,
                                                       vCount * cObjectBytes), group=vGroup)
      vGroup.finish()

  def fSequenceGdn(pBufferIn, pBufferOut, pBufferState, lHandlesW, lHandlesY):
    # One phase of the recurrence over 32 heads, head 2s + c on core c of stream s. The input of a stream: the
    # parameters and the inputs of its two heads (an object), then the objects of the states, [rows 9o to 9o + 8 of
    # head 2s | the same rows of head 2s + 1] for o from 0 to 14 (rows from 128 on are the padding of the last object,
    # read from the head after it). Its Y: row j of head 2s and row j of head 2s + 1 for j from 0 to 127, which go back
    # where they came from, and then the outputs of both heads.
    vTotalIn = cStreams * vObjectsStream * cObjectBytes
    vTotalState = cGdnHeads * cGdnDim * cGdnDim + cGdnPad
    vHeadFloats = cGdnDim * cGdnDim
    for vStream in range(cStreams):
      lHandlesY[vStream].drain(pBufferState, tap=TensorAccessPattern((vTotalState,), 2 * vStream * vHeadFloats,
                                                                     [1, cGdnDim, 2, cGdnDim],
                                                                     [0, cGdnDim, vHeadFloats, 1]), wait=True)
      lHandlesY[vStream].drain(pBufferOut, tap=fLinear(vOutFloats, 2 * vStream * cGdnDim, 2 * cGdnDim), wait=True)
    for vStream in range(cStreams):
      lHandlesW[vStream].fill(pBufferIn, tap=fLinear(vTotalIn, vStream * vObjectsStream * cObjectBytes,
                                                     vObjectsStream * cObjectBytes))
      lHandlesW[vStream].fill(pBufferState, tap=TensorAccessPattern((vTotalState,), 2 * vStream * vHeadFloats,
                                                                    [cGdnObjectsHead, 2, cGdnRows, cGdnDim],
                                                                    [cGdnRows * cGdnDim, vHeadFloats, cGdnDim, 1]))

  def fSequenceLa(pBufferIn, pBufferOut, pBufferState, lHandlesW, lHandlesY):
    # One phase of a layer of linear attention over 32 value heads, head s on core 0 and head s + 16 on core 1 of
    # stream s. The input of a stream: the parameters and the 3 objects of the prologue, then the objects of the
    # states, [rows 9o to 9o + 8 of head s | the same rows of head s + 16] for o from 0 to 14. Its Y: row j of head s
    # and row j of head s + 16 for j from 0 to 127, back where they came from, and then the outputs of both heads after
    # their gated norm, to heads s and s + 16 of Out.
    vTotalIn = cStreams * vObjectsStream * cObjectBytes
    vTotalState = cGdnHeads * cGdnDim * cGdnDim + cGdnPad
    vHeadFloats = cGdnDim * cGdnDim
    vPairState = cStreams * vHeadFloats  # from head s to head s + 16
    for vStream in range(cStreams):
      lHandlesY[vStream].drain(pBufferState, tap=TensorAccessPattern((vTotalState,), vStream * vHeadFloats,
                                                                     [1, cGdnDim, 2, cGdnDim],
                                                                     [0, cGdnDim, vPairState, 1]), wait=True)
      lHandlesY[vStream].drain(pBufferOut, tap=TensorAccessPattern((vOutFloats,), vStream * cGdnDim, [1, 1, 2, cGdnDim],
                                                                   [0, 0, cStreams * cGdnDim, 1]), wait=True)
    for vStream in range(cStreams):
      lHandlesW[vStream].fill(pBufferIn, tap=fLinear(vTotalIn, vStream * vObjectsStream * cObjectBytes,
                                                     vObjectsStream * cObjectBytes))
      lHandlesW[vStream].fill(pBufferState, tap=TensorAccessPattern((vTotalState,), vStream * vHeadFloats,
                                                                    [cGdnObjectsHead, 2, cGdnRows, cGdnDim],
                                                                    [cGdnRows * cGdnDim, vPairState, cGdnDim, 1]))

  if pTest in ("gdn", "la"):
    vRuntime = Runtime(
      fSequenceGdn if pTest == "gdn" else fSequenceLa,
      [
        vTypeIn,
        vTypeOut,
        vTypeState,
        [vFifo.prod(tile=Tile(vStream // cChannels, 0)) for vStream, vFifo in enumerate(lFifosW)],
        [vFifo.cons(tile=Tile(vStream // cChannels, 0)) for vStream, vFifo in enumerate(lFifosY)],
      ],
    )
  else:
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
  vParser = argparse.ArgumentParser(prog="Chained overlay for generating tokens")
  add_compile_args(vParser, dev_choices=("npu2",), with_emit_mlir=True)
  vParser.add_argument("-K", type=int, default=2560)
  vParser.add_argument("-N", type=int, default=2560)
  vParser.add_argument("--format", choices=sorted(dFormats), default="q4k")
  vParser.add_argument("--compute", type=int, choices=[0, 1], default=1)
  vParser.add_argument("--gemv-os", type=int, choices=[0, 1], default=1)
  vParser.add_argument("--test", choices=["single", "ffn", "gdn", "la"], default="single")
  return vParser


def fValidate(pOpts):
  if pOpts.K < 2 * cTileK or pOpts.K % (2 * cTileK) != 0:
    sys.exit(f"K must be a positive multiple of {2 * cTileK}")
  if (pOpts.K // 2 // 8) * 12 > cKeptX:
    sys.exit(f"half of K in bfp16 of one row has to fit in {cKeptX} bytes")
  if pOpts.N < cStreams * cOutputs or pOpts.N % (cStreams * cOutputs) != 0:
    sys.exit(f"N must be a positive multiple of {cStreams * cOutputs}")


def fCompileKwargs(pOpts):
  return dict(pK=pOpts.K, pN=pOpts.N, pFormat=pOpts.format, pCompute=bool(pOpts.compute), pTest=pOpts.test,
              pGemvOs=bool(pOpts.gemv_os))


def fRunAndVerify(pOpts):
  sys.exit("this design is only compiled here: it is run with llama-xdna-run")


def fMain():
  vOpts = fMakeArgparser().parse_args()
  run_design_cli(
    fGemvChain,
    vOpts,
    compile_kwargs=fCompileKwargs,
    run_and_verify=fRunAndVerify,
    # the whole array: without n_cols=None the device is the one-column variant
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
