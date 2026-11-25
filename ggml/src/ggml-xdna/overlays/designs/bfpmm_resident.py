#!/usr/bin/env -S PYTHONDONTWRITEBYTECODE=1 python3

"""Design for the NPU: matrix multiplication C = A x B on the whole array (8 columns x 4 rows of cores) with the
kernel of this project (ggml/src/ggml-xdna/kernels/xdna2-mm-bfp16.cc), with the bands of A kept in the memory
tiles instead of streamed again and again from the host.

A and B are block floating point (bfp16ebs8: 8 values of 8 bits share an exponent, 9 bytes), C 32-bit floats or,
with --c-type bf16, bfloat16 rounded to the nearest: half the bytes; or, with --c-type bfp16, block floating point
too, every 8 rows of one column with one exponent (the kernel built with -DXDNA2_C_BFP16): 9 bytes for 8 values.
Rounded, every core adds up the tiles of K in 32-bit floats in a buffer of its own, and the last tile of K writes
the rounded tile of C; K is then two tiles at least.

Against bfpmm_linear.py the roles of the rows and the columns of cores are swapped:

  A  goes to the eight columns of cores. The host sends every band of m rows once; the memory tile of the column
     keeps it and sends it to its four cores once for every tile of C they produce with it.
  B  goes to the four rows of cores, each tile to the eight cores of its row.

So a pass of the whole B through the array meets 8 bands of A instead of 4, and A is sent once: for 4096 x 4096
x 4096 the host streams 170 MB instead of 453.

Every host buffer holds its bytes in the order of the streams, so every shim channel reads or writes one run of
consecutive bytes (the tiles are in the order of the kernel, described in its source):

  A  [M / m bands][K / k tiles][tile of A]                         band b holds the rows b * m ... of A
  B  [4 rows][N / n / 4][K / k tiles][tile of B transposed]         row r holds the tile columns r, r + 4 ... of B
  C  [8 columns][M / m / 8 blocks][N / n / 4][4 rows][tile of C]

gemm_bfp_linear.py with the layout "columns" writes A and B like that and turns C back into a row-major matrix.

With --b-split 2 every tile of B comes from the host in two halves, through two shim channels, and the memory
tile joins them before it sends the tile to the eight cores of its row: B uses the eight input channels A leaves
free instead of four. A memory tile has six input channels, and the ones that take B already use them all (A, B
and the four tiles of C of the column), so the last row of cores then sends its tiles of C straight to the host,
through the second output channel of the shim:

  B  [4 rows][2 halves][N / n / 4][K / k tiles][half of a tile of B]
  C  [8 columns][M / m / 8 blocks][N / n / 4][rows 0 to 2][tile of C], then
     [8 columns][M / m / 8 blocks][N / n / 4][tile of C of row 3]

M only shapes the instruction stream. K and N are built into the overlay: the memory tiles hold a band of
m x K values and repeat it N / n / 4 times.

  bfpmm_resident.py --dev npu2 -M <M> -K <K> -N <N> [-m 64 -k 128 -n 64] [--compute 0] [--b-depth 2]
                    [--c-type f32|bf16|bfp16] [--kernel 2x2|s3] [--b-split 1|2]
                    --xclbin-path <file> --insts-path <file>

With --compute 0 the cores move the same data and multiply nothing: what is left is the time of the data path.
--kernel s3 builds the cores with kernels/xdna2-mm-bfp16-s3.cc (steps of three sets of sums, another order of the
groups in the tiles, which the manifest says with the name of the kernel) instead of kernels/xdna2-mm-bfp16.cc.
--b-depth is the number of tiles of B a memory tile holds on their way from the host to the cores.

Buffers of the design, in order: A (M x K values, 9 bytes for 8), B (K x N values, 9 bytes for 8) and C (M x N
32-bit floats, bfloat16 or 9 bytes for 8 values).

With --gdn 1 (and C in bfloat16) the cores can also run the recurrence of a gated delta net by chunks of 16 tokens
(kernels/xdna2-gdn-prompt.cc), in the same buffers: a command sets the runtime parameters 2 and 3 (rounds, and
groups of 8 chunks a round) instead of 0 and 1, which it leaves at 0 (and a multiplication leaves 2 and 3 at 0). A
round takes 2 heads of keys with their two value heads each (s and s + 16 in Qwen3.5), each head of keys on two rows
of cores: row r takes the head of keys r / 2 of the round and the half r % 2 of the value indices, and the core of
column x of it the block of 8 value indices 8 (r % 2) + x of both value heads, a state of 8 KB (the 32 cores hold
the 512 KB of the states of the 4 value heads). The band of a column brings the states of the round, one object per
row ([2 heads][8 j][128] f32 in its first 8 KB), and two more objects that its cores keep as their work memory for
the round; B brings every chunk to its row in two objects; C takes the outputs of 8 chunks an object
([2 heads][8 chunks][16 t][8 j] f32 per core, the four rows of a column one after the other), and then the states.

--c-rows 1 (with --kernel s3 and C in bfloat16) builds the last tile of K with -DXDNA2_C_ROWS
(xdna2_mm_bfp16_s3_last_rows): every core writes its tile of C as the rows of the tile, so that the host lays the
tiles out as the rows of C with the descriptors of the shims (manifest line "rows 1", fXdnaGemmPlan). The data
paths of the overlay do not change; its own instruction stream for the test still writes C as it comes, so the
rows are checked with llama-xdna-gemm. Written as the rows of C, a shim writes runs of n values (128 bytes), which
measured 49 % slower than one run per column (512 x 2560 x 9216); so the host lays C out as strips of n columns
instead, [N / n][M][n], where the tile of a core is 8 KB in a row. Interleaving the tiles of a turn in the memory
tile, for runs of 4 n values, is not possible: a join gives every input a slot of its own, to the offset of the next
(ObjectFifoLinkOp in mlir-aie), and walks its output slot by slot.

--gdn-skip leaves out pieces of every chunk, to measure the rest and the data alone: both halves (both), or a
comma separated list of kq and rest (the halves), split, products, gamma, head, decay and update.

--test gdn builds the instruction stream of one such command over --gdn-rounds rounds of --gdn-groups groups, for
an overlay built with -K 768 -N 256 (the memory tiles then hold a band of 6 tiles and send it once, which needs no
rearming), with the buffers and the transfers of fXdnaGdnPlan (ggml/src/ggml-xdna/xdna-gemm.cpp): A [8
columns][rounds][6 tiles], B [2 x rounds heads of keys][chunks][2 tiles], which row r reads for the head of keys
r / 2 of every round, and C [8 columns][rounds][groups + 1][4 rows][8192 bytes].
"""

import argparse
import sys
from pathlib import Path

import aie.iron as iron
import numpy as np
from ml_dtypes import bfloat16
from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Buffer,
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.device import AnyMemTile, Tile
from aie.iron.controlflow import range_
from aie.iron.dataflow.objectfifo import ObjectFifoLink
from aie.iron.device import from_name
from aie.utils import config
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli

cCols = 8      # columns of the array: streams of A
cRows = 4      # rows of core tiles: streams of B
cDepth = 2     # objects of the fifos that do not say otherwise
cInFlight = 4  # row blocks whose transfers are queued at once
cBlock = 8     # values that share an exponent
cPair = 16     # rows, columns and values of K the kernel takes at a time
cMemoryB = (1, 3, 5, 7)  # with --b-split 2, the memory tile that joins the halves of B of every row of cores
cKernels = Path(__file__).resolve().parents[2] / "kernels"  # ggml/src/ggml-xdna/kernels
dKernels = {"2x2": ("xdna2-mm-bfp16.cc", "xdna2_mm_bfp16"), "s3": ("xdna2-mm-bfp16-s3.cc", "xdna2_mm_bfp16_s3")}
cGdnSource = "xdna2-gdn-prompt.cc"
cChunksOut = 8   # with --gdn 1, chunks whose outputs an object of C holds
cGdnBand = 6     # objects of the band of a round: the states of the 4 rows, and 2 of work
cGdnOut = 5      # objects of C a core gives a round with 4 groups: 4 of outputs and the states (one per group + 1)


@iron.jit
def fBfpMmResident(
    pA: In,
    pB: In,
    pC: Out,
    *,
    pM: CompileTime[int] = 512,
    pK: CompileTime[int] = 512,
    pN: CompileTime[int] = 512,
    pTileM: CompileTime[int] = 64,
    pTileK: CompileTime[int] = 128,
    pTileN: CompileTime[int] = 64,
    pCompute: CompileTime[bool] = True,
    pDepthB: CompileTime[int] = 2,
    pTypeC: CompileTime[str] = "f32",
    pKernel: CompileTime[str] = "2x2",
    pSplitB: CompileTime[int] = 1,
    pGdn: CompileTime[bool] = False,
    pTest: CompileTime[str] = "gemm",
    pGdnRounds: CompileTime[int] = 1,
    pGdnGroups: CompileTime[int] = 4,
    pGdnSkip: CompileTime[str] = "none",
    pRowsC: CompileTime[bool] = False,
    pCompB: CompileTime[bool] = False,
):
  vSource, vName = dKernels[pKernel]
  vSplit = pSplitB == 2   # B in two halves, and the tiles of C of the last row of cores straight to the host
  vRowsJoined = cRows - 1 if vSplit else cRows   # rows of cores whose tiles of C the memory tile joins
  cKernelSource = cKernels / vSource
  vBfp16 = pTypeC == "bfp16"
  vBf16 = pTypeC == "bf16" or vBfp16   # C rounded: the sums stay in a buffer of every core
  vDtypeC = v8bfp16ebs8 if vBfp16 else bfloat16 if vBf16 else np.float32
  vFlagsC = ["-DXDNA2_C_BFP16"] if vBfp16 else []
  vBlocks = pM // (pTileM * cCols)        # row blocks: every one is a pass of the whole B through the array
  vTilesK = pK // pTileK                  # tiles every core adds up into one tile of C
  vTilesOut = pN // (pTileN * cRows)      # tiles of C every core produces in one row block
  vBlocksA = pTileM * pTileK // cBlock    # blocks of 8 values of a tile of A
  vBlocksB = pTileK * pTileN // cBlock    # and of a tile of B
  vHalfB = vBlocksB // 2                  # blocks of half a tile of B
  vSumsC = pTileM * pTileN                # sums of a tile of C
  vItemsC = vSumsC // cBlock if pTypeC == "bfp16" else vSumsC  # and the items of its type that hold them
  vItemsAllC = cCols * vBlocks * vTilesOut * cRows * vItemsC    # items of the whole C

  vMatmul = ExternalFunction(
    vName,
    source_file=str(cKernelSource),
    arg_types=[
      np.ndarray[(vBlocksA,), np.dtype[v8bfp16ebs8]],
      np.ndarray[(vBlocksB,), np.dtype[v8bfp16ebs8]],
      np.ndarray[(vSumsC,), np.dtype[np.float32]],
      np.int32,
    ],
    include_dirs=[config.cxx_header_path()],
    compile_flags=[f"-DXDNA2_DIM_M={pTileM}", f"-DXDNA2_DIM_K={pTileK}", f"-DXDNA2_DIM_N={pTileN}"] + vFlagsC,
  )
  # with C rounded, the last tile of K: the same source, built with the other function only
  vMatmulLast = None
  if vBf16:
    vMatmulLast = ExternalFunction(
      vName + ("_last_rows" if pRowsC else "_last"),
      source_file=str(cKernelSource),
      arg_types=[
        np.ndarray[(vBlocksA,), np.dtype[v8bfp16ebs8]],
        np.ndarray[(vBlocksB,), np.dtype[v8bfp16ebs8]],
        np.ndarray[(vSumsC,), np.dtype[np.float32]],
        np.ndarray[(vItemsC,), np.dtype[vDtypeC]],
      ],
      include_dirs=[config.cxx_header_path()],
      compile_flags=[
        f"-DXDNA2_DIM_M={pTileM}", f"-DXDNA2_DIM_K={pTileK}", f"-DXDNA2_DIM_N={pTileN}", "-DXDNA2_LAST_TILE"
      ] + vFlagsC + (["-DXDNA2_C_ROWS"] if pRowsC else []),
    )
  vMatmulComp = None
  if pCompB:
    vMatmulComp = ExternalFunction(
      "xdna2_mm_comp_step",
      source_file=str(cKernelSource),
      arg_types=[
        np.ndarray[(vBlocksA,), np.dtype[v8bfp16ebs8]],
        np.ndarray[(vBlocksB,), np.dtype[v8bfp16ebs8]],
        np.ndarray[(vItemsC,), np.dtype[vDtypeC]],
        np.ndarray[(vSumsC,), np.dtype[np.float32]],
        np.int32,
        np.int32,
        np.int32,
      ],
      include_dirs=[config.cxx_header_path()],
      compile_flags=[
        f"-DXDNA2_DIM_M={pTileM}", f"-DXDNA2_DIM_K={pTileK}", f"-DXDNA2_DIM_N={pTileN}",
        "-DXDNA2_MM_COMP", "-DXDNA2_C_ROWS", "-O2", "-fno-jump-tables"
      ] + vFlagsC,
    )
  # with --gdn 1, the recurrence: one object per function of the same source
  vGdnState = vGdnKq = vGdnRest = vGdnStateOut = None
  if pGdn:
    vTile = np.ndarray[(vBlocksA,), np.dtype[v8bfp16ebs8]]
    vSums = np.ndarray[(vSumsC,), np.dtype[np.float32]]
    vOut = np.ndarray[(vItemsC,), np.dtype[vDtypeC]]

    # --gdn-skip: halves of a chunk left out, to measure what the other and the data take
    dSkip = {"none": [], "both": ["KQ", "REST"]}
    lSkip = [f"-DXDNA2_GDN_SKIP_{vPiece}" for vPiece in dSkip.get(pGdnSkip, pGdnSkip.upper().split(","))]

    def fGdnFunction(pName, pFlag, lArgTypes, pSizeOpt=False):
      return ExternalFunction(
        pName,
        source_file=str(cKernels / cGdnSource),
        arg_types=lArgTypes,
        include_dirs=[config.cxx_header_path()],
        compile_flags=[pFlag] + (["-Os"] if pSizeOpt else []) + lSkip,
      )

    vGdnState = fGdnFunction("xdna2_gdn_state", "-DXDNA2_GDN_STATE", [vTile, vSums, np.int32, np.int32], pCompB)
    vGdnKq = fGdnFunction("xdna2_gdn_kq", "-DXDNA2_GDN_KQ", [vTile, vSums, vTile, vTile], False)
    vGdnRest = fGdnFunction("xdna2_gdn_rest", "-DXDNA2_GDN_REST", [vTile, vSums, vTile, vOut, np.int32], pCompB)
    vGdnStateOut = fGdnFunction("xdna2_gdn_state_out", "-DXDNA2_GDN_STATE_OUT", [vSums, vOut], pCompB)

  # the host buffers, with the shape the walks below index
  vTypeA = np.ndarray[(pM // pTileM, vTilesK * vBlocksA), np.dtype[v8bfp16ebs8]]
  if vSplit:
    vShapeB = (cRows * 2, vTilesOut * vTilesK * vHalfB)
    vTypeC = np.ndarray[(vItemsAllC,), np.dtype[vDtypeC]]
  else:
    vShapeB = (cRows, vTilesOut * vTilesK * vBlocksB)
    vTypeC = np.ndarray[(cCols * vBlocks, vTilesOut * cRows * vItemsC), np.dtype[vDtypeC]]
  vTypeB = np.ndarray[vShapeB, np.dtype[v8bfp16ebs8]]

  vTypeBandA = np.ndarray[(vTilesK * vBlocksA,), np.dtype[v8bfp16ebs8]]
  vTypeTileA = np.ndarray[(vBlocksA,), np.dtype[v8bfp16ebs8]]
  vTypeTileB = np.ndarray[(vBlocksB,), np.dtype[v8bfp16ebs8]]
  vTypeHalfB = np.ndarray[(vHalfB,), np.dtype[v8bfp16ebs8]]
  vTypeTileC = np.ndarray[(vItemsC,), np.dtype[vDtypeC]]
  vTypeColumnC = np.ndarray[(vItemsC * vRowsJoined,), np.dtype[vDtypeC]]
  vTypeSums = np.ndarray[(vSumsC,), np.dtype[np.float32]]
  vTypeRtp = np.ndarray[(8 if pCompB else 4 if pGdn else 2,), np.dtype[np.int32]]

  # split, every transfer has its place: the sixteen input and the sixteen output channels of the shims are all in
  # use, and the placer would put several bands of A in one memory tile (with K = 4096 a band fills 295 of its
  # 512 KB). Row r of cores takes B through the shims of columns 2r and 2r + 1.
  def fMemory(pColumn):
    return Tile(pColumn, 1) if vSplit else AnyMemTile

  def fShim(pColumn):
    return Tile(pColumn, 0) if vSplit else None

  # A: host -> memory tile of the column, a whole band at a time. The memory tile holds one band and sends it
  # once for every tile of C; the cores take it one tile at a time.
  lFifosAL3L2 = []
  lFifosAL2L1 = []
  for vCol in range(cCols):
    vFifoIn = ObjectFifo(vTypeBandA, name=f"A_L3L2_{vCol}", depth=1)
    vFifoOut = ObjectFifo(
      vTypeBandA, name=f"A_L2L1_{vCol}", depth=1, repeat_count=vTilesOut, consumer_obj_type=vTypeTileA
    )
    ObjectFifoLink(vFifoIn.cons(), vFifoOut.prod(), tile=fMemory(vCol))
    lFifosAL3L2.append(vFifoIn)
    lFifosAL2L1.append(vFifoOut)

  # B: host -> memory tile -> the eight cores of a row, a tile at a time; split, the memory tile joins the two
  # halves of every tile, each from a shim channel of its own
  lFifosBL3L2 = []
  lFifosBL2L1 = []
  for vRow in range(cRows):
    if vSplit:
      vFifo = ObjectFifo(vTypeTileB, name=f"B_L2L1_{vRow}", depth=pDepthB)
      lFifosBL3L2.extend(
        vFifo.prod().join(
          [0, vHalfB],
          tile=Tile(cMemoryB[vRow], 1),
          obj_types=[vTypeHalfB] * 2,
          names=[f"B_L3L2_{vRow}_{vHalf}" for vHalf in range(2)],
          depths=[pDepthB] * 2,
        )
      )
      lFifosBL2L1.append(vFifo)
    else:
      vFifo = ObjectFifo(vTypeTileB, name=f"B_L3L2_{vRow}", depth=pDepthB)
      lFifosBL3L2.append(vFifo)
      lFifosBL2L1.append(vFifo.cons().forward(obj_type=vTypeTileB, name=f"B_L2L1_{vRow}", depth=cDepth))

  # C: cores -> memory tile -> host: the tiles of C of a column travel together; split, the last row of cores
  # sends its own straight to the host
  lFifosCL2L3 = []
  lFifosCL1L3 = []
  aFifosCL1L2 = [[] for _ in range(cRows)]
  for vCol in range(cCols):
    vFifoC = ObjectFifo(vTypeColumnC, name=f"C_L2L3_{vCol}", depth=cDepth)
    lFifosCL2L3.append(vFifoC)
    lJoined = vFifoC.prod().join(
      [vItemsC * vRow for vRow in range(vRowsJoined)],
      tile=fMemory(vCol),
      obj_types=[vTypeTileC] * vRowsJoined,
      names=[f"C_L1L2_{vCol}_{vRow}" for vRow in range(vRowsJoined)],
      depths=[1] * vRowsJoined,
    )
    for vRow in range(vRowsJoined):
      aFifosCL1L2[vRow].append(lJoined[vRow])
    if vSplit:
      vFifoDirect = ObjectFifo(vTypeTileC, name=f"C_L1L3_{vCol}", depth=1)
      lFifosCL1L3.append(vFifoDirect)
      aFifosCL1L2[cRows - 1].append(vFifoDirect)

  # the trip counts reach every core as runtime parameters
  aRtps = [
    [Buffer(vTypeRtp, name=f"rtp_{vRow}_{vCol}", use_write_rtp=True) for vCol in range(cCols)]
    for vRow in range(cRows)
  ]
  aBarriers = [[WorkerRuntimeBarrier() for _ in range(cCols)] for _ in range(cRows)]
  # with C in bfloat16, the sums of every core stay in a buffer of its own
  aSums = None
  if vBf16:
    aSums = [[Buffer(vTypeSums, name=f"sums_{vRow}_{vCol}") for vCol in range(cCols)] for vRow in range(cRows)]

  def fCore(pInA, pInB, pOutC, pMatmulKernel, pRtp, pBarrier):
    pBarrier.wait_for_value(1)
    vTripsK = pRtp[0]
    vTripsOut = pRtp[1]
    # the wait leaves the barrier at 1: step it off, so the next command waits for its own values
    pBarrier.release_with_value(1)
    for _ in range_(vTripsOut):
      vElemC = pOutC.acquire(1)
      # the first tile of K starts the sums
      vElemA = pInA.acquire(1)
      vElemB = pInB.acquire(1)
      if pCompute:
        pMatmulKernel(vElemA, vElemB, vElemC, 1)
      pInA.release(1)
      pInB.release(1)
      for _ in range_(vTripsK):
        vElemA = pInA.acquire(1)
        vElemB = pInB.acquire(1)
        if pCompute:
          pMatmulKernel(vElemA, vElemB, vElemC, 0)
        pInA.release(1)
        pInB.release(1)
      pOutC.release(1)

  def fCoreBf16(pInA, pInB, pOutC, pSums, pMatmulKernel, pLastKernel, pRtp, pBarrier):
    pBarrier.wait_for_value(1)
    vTripsMiddle = pRtp[0]
    vTripsOut = pRtp[1]
    pBarrier.release_with_value(1)
    fTilesBf16(pInA, pInB, pOutC, pSums, pMatmulKernel, pLastKernel, vTripsMiddle, vTripsOut)

  # with --gdn 1: the same multiplication, then the rounds of the recurrence (0 of them in a multiplication, and no
  # tile of C in a recurrence)
  def fCoreGdn(pRow, pCol):
    def fCore(pInA, pInB, pOutC, pSums, pMatmulKernel, pLastKernel, pState, pKq, pRest, pStateOut, pRtp, pBarrier):
      pBarrier.wait_for_value(1)
      vTripsMiddle = pRtp[0]
      vTripsOut = pRtp[1]
      vRounds = pRtp[2]
      vGroups = pRtp[3]
      pBarrier.release_with_value(1)
      fTilesBf16(pInA, pInB, pOutC, pSums, pMatmulKernel, pLastKernel, vTripsMiddle, vTripsOut)
      for _ in range_(vRounds):
        # the states of the round, one object per row of the column: this core keeps the one of its row
        for vRowAt in range(cRows):
          vElemA = pInA.acquire(1)
          pState(vElemA, pSums, vRowAt, pRow)
          pInA.release(1)
        # the two objects of work, kept for the round
        lWork = pInA.acquire(2)
        for _ in range_(vGroups):
          vElemC = pOutC.acquire(1)
          # a chunk is two objects of B, one at a time: holding both would take a third buffer
          for _ in range_(cChunksOut):
            vIn = pInB.acquire(1)
            pKq(vIn, pSums, lWork[0], lWork[1])
            pInB.release(1)
            vIn = pInB.acquire(1)
            pRest(vIn, pSums, lWork[1], vElemC, 8 * (pRow % 2) + pCol)
            pInB.release(1)
          pOutC.release(1)
        vElemC = pOutC.acquire(1)
        pStateOut(pSums, vElemC)
        pOutC.release(1)
        pInA.release(2)

    def fCoreComp(pInA, pInB, pOutC, pSums, pMatmulKernel, pLastKernel, pCompKernel, pState, pKq, pRest, pStateOut,
                  pRtp, pBarrier):
      pBarrier.wait_for_value(1)
      vTripsMiddle = pRtp[0]
      vTripsOut = pRtp[1]
      vRounds = pRtp[2]
      vGroups = pRtp[3]
      vTripsOutQ6K = pRtp[4]
      vQuadsK = pRtp[5]
      vTripsOutQ8 = pRtp[6]
      pBarrier.release_with_value(1)
      fTilesBf16(pInA, pInB, pOutC, pSums, pMatmulKernel, pLastKernel, vTripsMiddle, vTripsOut)
      for _ in range_(vTripsOutQ6K):
        vElemC = pOutC.acquire(1)
        for vQuad in range_(vQuadsK):
          for vSubStep in range_(2):
            vElemA = pInA.acquire(1)
            vElemB = pInB.acquire(1)
            if pCompute:
              pCompKernel(vElemA, vElemB, vElemC, pSums, vSubStep, vQuad, vQuadsK)
            pInA.release(1)
            pInB.release(1)
          vElemA = pInA.acquire(1)
          vElemB = pInB.acquire(1)
          if pCompute:
            pCompKernel(vElemA, vElemB, vElemC, pSums, 2, vQuad, vQuadsK)
          pInA.release(1)
          vElemA = pInA.acquire(1)
          if pCompute:
            pCompKernel(vElemA, vElemB, vElemC, pSums, 3, vQuad, vQuadsK)
          pInA.release(1)
          pInB.release(1)
        pOutC.release(1)
      for _ in range_(vTripsOutQ8):
        vElemC = pOutC.acquire(1)
        for vTile in range_(vQuadsK):
          vElemA = pInA.acquire(1)
          vElemB = pInB.acquire(1)
          if pCompute:
            pCompKernel(vElemA, vElemB, vElemC, pSums, 4, vTile, vQuadsK)
          pInA.release(1)
          pInB.release(1)
        pOutC.release(1)
      for _ in range_(vRounds):
        for vRowAt in range(cRows):
          vElemA = pInA.acquire(1)
          pState(vElemA, pSums, vRowAt, pRow)
          pInA.release(1)
        lWork = pInA.acquire(2)
        for _ in range_(vGroups):
          vElemC = pOutC.acquire(1)
          for _ in range_(cChunksOut):
            vIn = pInB.acquire(1)
            pKq(vIn, pSums, lWork[0], lWork[1])
            pInB.release(1)
            vIn = pInB.acquire(1)
            pRest(vIn, pSums, lWork[1], vElemC, 8 * (pRow % 2) + pCol)
            pInB.release(1)
          pOutC.release(1)
        vElemC = pOutC.acquire(1)
        pStateOut(pSums, vElemC)
        pOutC.release(1)
        pInA.release(2)

    return fCoreComp if pCompB else fCore

  def fTilesBf16(pInA, pInB, pOutC, pSums, pMatmulKernel, pLastKernel, pTripsMiddle, pTripsOut):
    for _ in range_(pTripsOut):
      # the first tile of K starts the sums
      vElemA = pInA.acquire(1)
      vElemB = pInB.acquire(1)
      if pCompute:
        pMatmulKernel(vElemA, vElemB, pSums, 1)
      pInA.release(1)
      pInB.release(1)
      for _ in range_(pTripsMiddle):
        vElemA = pInA.acquire(1)
        vElemB = pInB.acquire(1)
        if pCompute:
          pMatmulKernel(vElemA, vElemB, pSums, 0)
        pInA.release(1)
        pInB.release(1)
      # the last one writes the tile of C, rounded: only now it is needed, while the previous one may still be
      # on its way out
      vElemC = pOutC.acquire(1)
      vElemA = pInA.acquire(1)
      vElemB = pInB.acquire(1)
      if pCompute:
        pLastKernel(vElemA, vElemB, pSums, vElemC)
      pInA.release(1)
      pInB.release(1)
      pOutC.release(1)

  def fWorker(pRow, pCol):
    lCommon = [
      lFifosAL2L1[pCol].cons(depth=cDepth), lFifosBL2L1[pRow].cons(depth=cDepth), aFifosCL1L2[pRow][pCol].prod()
    ]
    if pGdn:
      lKernels = [vMatmul, vMatmulLast] + ([vMatmulComp] if pCompB else [])
      lArgs = lCommon + [aSums[pRow][pCol]] + lKernels + [
        vGdnState, vGdnKq, vGdnRest, vGdnStateOut, aRtps[pRow][pCol], aBarriers[pRow][pCol]
      ]
      return Worker(fCoreGdn(pRow, pCol), lArgs, stack_size=0xF00)
    if vBf16:
      lArgs = lCommon + [aSums[pRow][pCol], vMatmul, vMatmulLast, aRtps[pRow][pCol], aBarriers[pRow][pCol]]
      return Worker(fCoreBf16, lArgs, stack_size=0xF00)
    lArgs = lCommon + [vMatmul, aRtps[pRow][pCol], aBarriers[pRow][pCol]]
    return Worker(fCore, lArgs, stack_size=0xF00)

  aWorkers = Worker.grid(cRows, cCols, fWorker)
  lWorkers = [vWorker for lRow in aWorkers for vWorker in lRow]

  def fSequence(pBufferA, pBufferB, pBufferC, lHandlesA, lHandlesB, lHandlesC, lHandlesDirect):
    for vRow in range(cRows):
      for vCol in range(cCols):
        # tiles of K that go through the loop of the core (with C in bfloat16 the last one does not either), and
        # tiles of C
        aRtps[vRow][vCol][0] = vTilesK - (2 if vBf16 else 1)
        aRtps[vRow][vCol][1] = vBlocks * vTilesOut
        if pGdn:
          # no round of the recurrence
          aRtps[vRow][vCol][2] = 0
          aRtps[vRow][vCol][3] = 0
        if pCompB:
          aRtps[vRow][vCol][4] = 0
          aRtps[vRow][vCol][5] = 0
          aRtps[vRow][vCol][6] = 0
        aBarriers[vRow][vCol].set(1)

    # every walk is one run of consecutive bytes of its buffer
    vWalkA = TensorAccessPattern.full((pM // pTileM, vTilesK * vBlocksA))
    vWalkB = TensorAccessPattern.full(vShapeB)
    vRunJoined = vTilesOut * vRowsJoined * vItemsC   # items of C of a column in a row block, through the memory tile
    vRunDirect = vTilesOut * vItemsC                 # and of the last row of cores, straight to the host

    def fWalkC(pColumn, pBlock, pDirect):
      if not vSplit:
        return TensorAccessPattern.full((cCols * vBlocks, vTilesOut * cRows * vItemsC))[pColumn * vBlocks + pBlock]
      vRun = vRunDirect if pDirect else vRunJoined
      vStart = cCols * vBlocks * vRunJoined if pDirect else 0
      return TensorAccessPattern(
        (vItemsAllC,), vStart + (pColumn * vBlocks + pBlock) * vRun, [1, 1, 1, vRun], [0, 0, 0, 1]
      )

    for vBlock, vGroup in TaskGroup.pipelined(vBlocks, depth=cInFlight):
      for vCol in range(cCols):
        lHandlesC[vCol].drain(pBufferC, tap=fWalkC(vCol, vBlock, False), wait=True, group=vGroup)
        if vSplit:
          lHandlesDirect[vCol].drain(pBufferC, tap=fWalkC(vCol, vBlock, True), wait=True, group=vGroup)
        # the band of A of this column of cores, once: its memory tile repeats it
        lHandlesA[vCol].fill(pBufferA, tap=vWalkA[vBlock * cCols + vCol], group=vGroup)
      for vStream in range(len(lHandlesB)):
        # the part of B of this row of cores (or its half), whole, for every row block
        lHandlesB[vStream].fill(pBufferB, tap=vWalkB[vStream], group=vGroup)

  # --test gdn: one command of the recurrence, --gdn-rounds rounds of --gdn-groups groups of chunks, with the buffers
  # and the transfers of fXdnaGdnPlan (xdna-gemm.cpp): A [column][round][6 objects], B [head of keys][chunk][2
  # objects], which row r reads for the head of keys r / 2 of every round, C [column][round][group + 1][row][8 KB]
  vChunksGdn = pGdnGroups * cChunksOut
  vHeadBlocks = vChunksGdn * 2 * vBlocksB
  vItemsRoundC = (pGdnGroups + 1) * cRows * vItemsC
  vTypeGdnA = np.ndarray[(cCols, pGdnRounds * cGdnBand * vBlocksA), np.dtype[v8bfp16ebs8]]
  vTypeGdnB = np.ndarray[(2 * pGdnRounds * vHeadBlocks,), np.dtype[v8bfp16ebs8]]
  vTypeGdnC = np.ndarray[(cCols, pGdnRounds * vItemsRoundC), np.dtype[vDtypeC]]

  def fSequenceGdn(pBufferA, pBufferB, pBufferC, lHandlesA, lHandlesB, lHandlesC, lHandlesDirect):
    for vRow in range(cRows):
      for vCol in range(cCols):
        aRtps[vRow][vCol][0] = 0
        aRtps[vRow][vCol][1] = 0
        aRtps[vRow][vCol][2] = pGdnRounds
        aRtps[vRow][vCol][3] = pGdnGroups
        if pCompB:
          aRtps[vRow][vCol][4] = 0
          aRtps[vRow][vCol][5] = 0
          aRtps[vRow][vCol][6] = 0
        aBarriers[vRow][vCol].set(1)
    vWalkA = TensorAccessPattern.full((cCols, pGdnRounds * cGdnBand * vBlocksA))
    vWalkC = TensorAccessPattern.full((cCols, pGdnRounds * vItemsRoundC))
    for vCol in range(cCols):
      lHandlesC[vCol].drain(pBufferC, tap=vWalkC[vCol], wait=True)
    for vCol in range(cCols):
      # the bands of all the rounds of the column, one after another: the memory tile takes one at a time
      lHandlesA[vCol].fill(pBufferA, tap=vWalkA[vCol])
    for vRow in range(cRows):
      vWalkB = TensorAccessPattern(
        (2 * pGdnRounds * vHeadBlocks,), (vRow // 2) * vHeadBlocks, [pGdnRounds, 1, 1, vHeadBlocks],
        [2 * vHeadBlocks, 0, 0, 1]
      )
      lHandlesB[vRow].fill(pBufferB, tap=vWalkB)

  lHandles = [
    [vFifo.prod(tile=fShim(vCol)) for vCol, vFifo in enumerate(lFifosAL3L2)],
    [vFifo.prod(tile=fShim(vStream)) for vStream, vFifo in enumerate(lFifosBL3L2)],
    [vFifo.cons(tile=fShim(vCol)) for vCol, vFifo in enumerate(lFifosCL2L3)],
    [vFifo.cons(tile=fShim(vCol)) for vCol, vFifo in enumerate(lFifosCL1L3)],
  ]
  if pTest == "gdn":
    vRuntime = Runtime(fSequenceGdn, [vTypeGdnA, vTypeGdnB, vTypeGdnC] + lHandles)
  else:
    vRuntime = Runtime(fSequence, [vTypeA, vTypeB, vTypeC] + lHandles)
  return Program(iron.get_current_device(), vRuntime, workers=lWorkers).resolve_program()


def fMakeArgparser():
  vParser = argparse.ArgumentParser(prog="NPU matrix multiplication in block floating point with resident bands of A")
  add_compile_args(vParser, dev_choices=("npu2",), with_emit_mlir=True)
  vParser.add_argument("-M", type=int, default=512)
  vParser.add_argument("-K", type=int, default=512)
  vParser.add_argument("-N", type=int, default=512)
  vParser.add_argument("-m", type=int, default=64)
  vParser.add_argument("-k", type=int, default=128)
  vParser.add_argument("-n", type=int, default=64)
  vParser.add_argument("--compute", type=int, choices=[0, 1], default=1)
  vParser.add_argument("--b-depth", type=int, default=2)
  vParser.add_argument("--c-type", choices=["f32", "bf16", "bfp16"], default="f32")
  vParser.add_argument("--kernel", choices=["2x2", "s3"], default="2x2")
  vParser.add_argument("--b-split", type=int, choices=[1, 2], default=1)
  vParser.add_argument("--gdn", type=int, choices=[0, 1], default=0)
  vParser.add_argument("--test", choices=["gemm", "gdn"], default="gemm")
  vParser.add_argument("--gdn-rounds", type=int, default=1)
  vParser.add_argument("--gdn-groups", type=int, default=4)
  vParser.add_argument("--c-rows", type=int, choices=[0, 1], default=0)
  vParser.add_argument("--comp-b", type=int, choices=[0, 1], default=0)
  vParser.add_argument("--gdn-skip", default="none",
                       help="none, both, or a comma separated list of kq, rest, split, products, gamma, head, decay, update")
  return vParser


def fValidate(pOpts):
  if min(pOpts.M, pOpts.K, pOpts.N, pOpts.m, pOpts.k, pOpts.n) < 1:
    sys.exit("every size must be positive")
  if pOpts.m % cPair or pOpts.k % cPair or pOpts.n % cPair:
    sys.exit(f"m, k and n must be multiples of {cPair}")
  if pOpts.M % (pOpts.m * cCols) != 0:
    sys.exit(f"M must be a multiple of m * {cCols}")
  if pOpts.K % pOpts.k != 0:
    sys.exit("K must be a multiple of k")
  if pOpts.N % (pOpts.n * cRows) != 0:
    sys.exit(f"N must be a multiple of n * {cRows}")
  if pOpts.c_type != "f32" and pOpts.K // pOpts.k < 2:
    sys.exit("with C rounded, K must be two tiles at least: the first and the last are different calls")
  if pOpts.N // (pOpts.n * cRows) > 63:
    sys.exit("a memory tile repeats a band 63 times at most (its locks count to 63): N / n / 4 is too large")
  if pOpts.gdn and (pOpts.c_type != "bf16" or pOpts.b_split != 1 or (pOpts.m, pOpts.k, pOpts.n) != (64, 128, 64)):
    sys.exit("the recurrence wants C in bfloat16 (its state goes in the sums), B in one piece and tiles of "
             "64 x 128 x 64")
  if pOpts.c_rows and (pOpts.kernel != "s3" or pOpts.c_type != "bf16" or pOpts.b_split != 1):
    sys.exit("--c-rows 1 wants --kernel s3, C in bfloat16 and B in one piece")
  if pOpts.comp_b and (not pOpts.gdn or not pOpts.c_rows):
    sys.exit("--comp-b 1 wants --gdn 1 and --c-rows 1")
  if pOpts.test == "gdn":
    if not pOpts.gdn:
      sys.exit("--test gdn needs --gdn 1")
    if pOpts.K != cGdnBand * pOpts.k or pOpts.N != pOpts.n * cRows:
      sys.exit(f"--test gdn wants -K {cGdnBand * pOpts.k} -N {pOpts.n * cRows}: a band of {cGdnBand} tiles, sent once")
    if pOpts.gdn_rounds < 1 or pOpts.gdn_groups < 1:
      sys.exit("--gdn-rounds and --gdn-groups must be positive")


def fCompileKwargs(pOpts):
  return dict(
    pM=pOpts.M,
    pK=pOpts.K,
    pN=pOpts.N,
    pTileM=pOpts.m,
    pTileK=pOpts.k,
    pTileN=pOpts.n,
    pCompute=bool(pOpts.compute),
    pDepthB=pOpts.b_depth,
    pTypeC=pOpts.c_type,
    pKernel=pOpts.kernel,
    pSplitB=pOpts.b_split,
    pGdn=bool(pOpts.gdn),
    pTest=pOpts.test,
    pGdnRounds=pOpts.gdn_rounds,
    pGdnGroups=pOpts.gdn_groups,
    pGdnSkip=pOpts.gdn_skip,
    pRowsC=bool(pOpts.c_rows),
    pCompB=bool(pOpts.comp_b),
  )


def fRunAndVerify(pOpts):
  sys.exit("this design is only compiled here: it is run with llama-xdna-run")


def fMain():
  vOpts = fMakeArgparser().parse_args()
  run_design_cli(
    fBfpMmResident,
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
