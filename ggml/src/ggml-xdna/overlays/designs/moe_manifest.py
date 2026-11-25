#!/usr/bin/env -S PYTHONDONTWRITEBYTECODE=1 python3

"""Writes the manifest of a build of the mixture-of-experts overlay (moe_experts.py), which the host reads to drive it
(ggml/src/ggml-xdna/xdna-moe.h, XdnaMoeOverlay):

  moe_manifest.py <project folder> <manifest>
"""

import re
import sys
from pathlib import Path

cColumns = 8
cRows = 6
cMemoryRows = 1
cKernel = "xdna2_moe_gemv"
cObjectBytes = 9216
cCoreBytes = 4608
cRowsMax = 24
cChunksMax = 1136
cSegments = 2


def fManifest(pProject):
  vText = (pProject / "input_with_addresses.mlir").read_text()
  if not re.search(rf"func\.func private @{cKernel}\(", vText):
    raise RuntimeError(f"the kernel {cKernel} is not in the build: is this a build of moe_experts.py?")
  if "cascade" not in vText:
    raise RuntimeError("the build has no cascade between the cores of a pair")
  dStreams = {}
  vPattern = r"aie\.shim_dma_allocation @([WY])_(\d+)__shim_alloc\(%shim_noc_tile_(\d+)_0, (MM2S|S2MM), (\d+)\)"
  for vMatch in re.finditer(vPattern, vText):
    vName, vNumber, vColumn, vDirection, vChannel = vMatch.groups()
    if (vName == "Y") != (vDirection == "S2MM"):
      raise RuntimeError(f"the stream {vName} {vNumber} goes the wrong way")
    dStreams[(vName, int(vNumber))] = (int(vColumn), int(vChannel))
  vStreams = len([vKey for vKey in dStreams if vKey[0] == "W"])
  if vStreams != 16 or any((vName, vStream) not in dStreams for vName in "WY" for vStream in range(vStreams)):
    raise RuntimeError("the streams W and Y do not match")
  vMatchX = re.search(rf"func\.func private @{cKernel}\(memref<(\d+)xui8>", vText)
  vMatchChain = re.search(r"func\.func private @xdna2_moe_finish\([^)]*memref<(\d+)xf32>", vText)
  if not vMatchX or not vMatchChain:
    raise RuntimeError("the buffers of X or of the gate of the cores are not found")
  vKeptX = int(vMatchX.group(1))
  vRoomX = vKeptX - cCoreBytes
  vChainFloats = int(vMatchChain.group(1))
  lLines = [
    "xdna2-moe 1",
    f"array {cColumns} {cRows} {cMemoryRows}",
    f"kernel {cKernel}",
    f"object {cObjectBytes} {cCoreBytes}",
    f"x {vKeptX} {vRoomX}",
    f"rows {cRowsMax}",
    f"chunks {cChunksMax}",
    f"segments {cSegments}",
    f"chain {vChainFloats}",
    "formats bfp16 q4k q5k q6k q8_0 q5_1",
  ]
  for (vName, vNumber), (vColumn, vChannel) in sorted(dStreams.items()):
    lLines.append(f"stream {vName} {vNumber} {vColumn} {vChannel}")
  return "\n".join(lLines) + "\n"


def fMain():
  if len(sys.argv) != 3:
    raise RuntimeError("usage: moe_manifest.py <project folder> <manifest>")
  vManifest = fManifest(Path(sys.argv[1]))
  Path(sys.argv[2]).write_text(vManifest)
  print(f"written {sys.argv[2]}")


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
