#!/usr/bin/env -S PYTHONDONTWRITEBYTECODE=1 python3

"""Writes the manifest of an overlay built with mlir-aie: where its placer put every stream and every core,
which is what the host needs to drive it (ggml/src/ggml-xdna/xdna-overlay.h describes the file).

  overlay_manifest.py <project folder> <layout> <m> <k> <n> [<K> <N>] > <manifest>

<project folder> is the .prj folder of the build: the placement is read in its input_with_addresses.mlir.
<layout> is "rows" for bfpmm_linear.py and "columns" for bfpmm_resident.py.

With the layout "columns" the overlay is built for one K and one N. Given <K> and <N>, the manifest says so and
the overlay only serves them. Without them, the manifest describes the memory tiles that keep the bands of A, so
that the host can set the size of the band and its repeats before every multiplication: the overlay then serves
any K up to the one it was built for, and any N.

The manifest also says whether C comes back in 32-bit floats, in bfloat16 or in block floating point
(bfpmm_resident.py --c-type), from the type of the buffers of C in the memory tiles, which kernel the cores
run (bfpmm_resident.py --kernel), which sets the order of the groups in the tiles, and whether every tile of B
comes in two halves and the last row of cores sends its tiles of C straight to the host (bfpmm_resident.py
--b-split 2): the streams B_L3L2_<row>_<half> are then the streams B 2 * row + half of the manifest, and
C_L1L3_<column> its streams D. And whether the cores can also run the recurrence of a gated delta net by chunks
(bfpmm_resident.py --gdn 1: four runtime parameters a core instead of two), from its kernel, and whether the cores
write every tile of C as its rows (bfpmm_resident.py --c-rows 1, the function xdna2_mm_bfp16_s3_last_rows: the
line "rows 1").
"""

import re
import sys
from pathlib import Path

cColumns = 8      # of the array of this NPU
cRows = 6
cMemoryRows = 1
cKernels = ("xdna2_mm_bfp16_s3", "xdna2_mm_bfp16")  # the longer name first: the other one is the start of it


def fManifest(pProject, pLayout, pTileM, pTileK, pTileN, pFixedK, pFixedN):
  if pLayout not in ("rows", "columns"):
    raise RuntimeError("the layout is rows or columns")
  vText = (pProject / "input_with_addresses.mlir").read_text()

  # the parameters of every core say which tile the placer gave to the core of every row and column of the design
  dCores = {}
  for vMatch in re.finditer(r"%rtp_(\d+)_(\d+) = aie\.buffer\(%tile_(\d+)_(\d+)\) \{address = (\d+)", vText):
    vRow, vColumn, vTileColumn, vTileRow, vAddress = (int(vValue) for vValue in vMatch.groups())
    dCores[(vRow, vColumn)] = [vTileColumn, vTileRow, vAddress]
  # the lock that starts a core is the only one of its tile without a name
  dLocks = {}
  for vMatch in re.finditer(r"%lock_(\d+)_(\d+) = aie\.lock\(%tile_(\d+)_(\d+), (\d+)\)", vText):
    vTileColumn, vTileRow, vLock = int(vMatch.group(3)), int(vMatch.group(4)), int(vMatch.group(5))
    if (vTileColumn, vTileRow) in dLocks:
      raise RuntimeError(f"the tile ({vTileColumn}, {vTileRow}) has more than one lock without a name")
    dLocks[(vTileColumn, vTileRow)] = vLock
  # the streams between the host and the array; split, the halves of B and the tiles of C of the last row of cores
  dStreams = {}
  vSplit = 1
  vPattern = (
    r"aie\.shim_dma_allocation @([ABC])_(L\dL\d)_(\d+)(?:_(\d+))?__shim_alloc\(%shim_noc_tile_(\d+)_0, "
    r"(MM2S|S2MM), (\d+)\)"
  )
  for vMatch in re.finditer(vPattern, vText):
    vName, vLevels, vNumber, vHalf, vColumn, vDirection, vChannel = vMatch.groups()
    if (vName == "C") != (vDirection == "S2MM"):
      raise RuntimeError(f"the stream {vName} {vNumber} goes the wrong way")
    vNumber = int(vNumber)
    if vHalf is not None:
      if vName != "B" or int(vHalf) > 1:
        raise RuntimeError(f"the stream {vMatch.group(0)} is not understood")
      vSplit = 2
      vNumber = vNumber * 2 + int(vHalf)
    if vName == "C" and vLevels == "L1L3":
      vName = "D"
    dStreams[(vName, vNumber)] = (int(vColumn), int(vChannel))
  if (vSplit == 2) != any(vName == "D" for vName, _ in dStreams):
    raise RuntimeError("B comes in halves but no core sends C straight, or the other way round")

  if not dCores or not dStreams:
    raise RuntimeError("no cores or no streams found: is this the project of one of the matrix multiplications?")
  vMatch = re.search(r"%C_L2L3_0__buf0 = aie\.buffer\(.*?: memref<\d+x(bf16|f32|!aiex\.bfp)", vText)
  if not vMatch:
    raise RuntimeError("the buffers of C in the memory tiles are not found")
  vOutput = {"bf16": "bf16", "f32": "f32", "!aiex.bfp": "bfp16"}[vMatch.group(1)]
  vKernel = next((vName for vName in cKernels if re.search(rf"func\.func private @{vName}\(", vText)), None)
  vGdn = 1 if re.search(r"func\.func private @xdna2_gdn_kq\(", vText) else 0
  vRowsC = 1 if re.search(r"func\.func private @xdna2_mm_bfp16_s3_last_rows\(", vText) else 0
  vCompB = 1 if re.search(r"func\.func private @xdna2_mm_comp_step\(", vText) else 0
  if vKernel is None:
    raise RuntimeError("the kernel of the cores is not one of " + ", ".join(cKernels))

  lLines = [
    "xdna2-overlay 1",
    f"array {cColumns} {cRows} {cMemoryRows}",
    f"kernel {vKernel}",
    f"layout {pLayout}",
    f"tile {pTileM} {pTileK} {pTileN}",
    f"fixed {pFixedK} {pFixedN}",
    f"output {vOutput}",
    f"split {vSplit}",
  ]
  if vGdn:
    lLines.append("gdn 1")
  if vRowsC:
    lLines.append("rows 1")
  if vCompB:
    lLines.append("comp 1")
  for (vRow, vColumn), (vTileColumn, vTileRow, vAddress) in sorted(dCores.items()):
    if (vTileColumn, vTileRow) not in dLocks:
      raise RuntimeError(f"the tile ({vTileColumn}, {vTileRow}) has no lock to start its core")
    lLines.append(f"core {vRow} {vColumn} {vTileColumn} {vTileRow} {vAddress} {dLocks[(vTileColumn, vTileRow)]}")
  for (vName, vNumber), (vColumn, vChannel) in sorted(dStreams.items()):
    lLines.append(f"stream {vName} {vNumber} {vColumn} {vChannel}")
  if pLayout == "columns" and pFixedK == 0 and pFixedN == 0:
    lLines.extend(fBands(vText))
  return "\n".join(lLines) + "\n"


def fBands(pText):
  """The lines of the manifest that describe the memory tile that keeps the band of A of every stream."""
  # the buffer of the band: which memory tile, and how many values of 9 bytes for 8 it holds
  dBuffers = {}
  vPattern = r"%A_L3L2_(\d+)__cons__buf0 = aie\.buffer\(%mem_tile_(\d+)_(\d+)\).*?: memref<(\d+)x!aiex\.bfp"
  for vMatch in re.finditer(vPattern, pText):
    vStream, vColumn, vRow, vBlocks = (int(vValue) for vValue in vMatch.groups())
    dBuffers[vStream] = (vColumn, vRow, vBlocks * 9 // 4)
  # its two locks
  dLocks = {}
  for vMatch in re.finditer(r"%A_L3L2_(\d+)__cons__(prod|cons)lock0 = aie\.lock\(%mem_tile_\d+_\d+, (\d+)\)", pText):
    dLocks[(int(vMatch.group(1)), vMatch.group(2))] = int(vMatch.group(3))
  # the channel and the buffer descriptor that receive it and the ones that send it: every dma_start of a memory
  # tile is followed by its buffer descriptors
  dEnds = {}
  vDirection, vChannel = None, None
  for vLine in pText.splitlines():
    vMatch = re.search(r"aie\.dma_start\((S2MM|MM2S), (\d+),", vLine)
    if vMatch:
      vDirection, vChannel = vMatch.group(1), int(vMatch.group(2))
      continue
    vMatch = re.search(r"aie\.dma_bd\(%A_L3L2_(\d+)__cons__buf0 .*\{bd_id = (\d+) : i32, next_bd_id = (\d+) : i32\}", vLine)
    if vMatch:
      vStream, vDescriptor, vNext = (int(vValue) for vValue in vMatch.groups())
      if vDescriptor != vNext:
        raise RuntimeError(f"the band of the stream {vStream} of A has more than one buffer descriptor per channel")
      dEnds[(vStream, vDirection)] = (vChannel, vDescriptor)

  lLines = []
  sCapacities = {vWords for _, _, vWords in dBuffers.values()}
  if len(sCapacities) != 1:
    raise RuntimeError("the bands of A do not all have the same size")
  lLines.append(f"capacity {sCapacities.pop()}")
  for vStream in sorted(dBuffers):
    vColumn, vRow, _ = dBuffers[vStream]
    vChannelIn, vDescriptorIn = dEnds[(vStream, "S2MM")]
    vChannelOut, vDescriptorOut = dEnds[(vStream, "MM2S")]
    lLines.append(
      f"band {vStream} {vColumn} {vRow} {vChannelIn} {vDescriptorIn} {vChannelOut} {vDescriptorOut} "
      f"{dLocks[(vStream, 'prod')]} {dLocks[(vStream, 'cons')]}"
    )
  return lLines


def fMain():
  if len(sys.argv) not in (6, 8):
    raise RuntimeError("usage: overlay_manifest.py <project folder> <layout> <m> <k> <n> [<K> <N>]")
  vFixedK, vFixedN = (int(sys.argv[6]), int(sys.argv[7])) if len(sys.argv) == 8 else (0, 0)
  sys.stdout.write(
    fManifest(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), vFixedK, vFixedN)
  )


if __name__ == "__main__":
  try:
    fMain()
    vOk = True
  except Exception as vError:
    print("error:", repr(vError), file=sys.stderr)
    vOk = False
  finally:
    pass
  sys.exit(0 if vOk else 1)
