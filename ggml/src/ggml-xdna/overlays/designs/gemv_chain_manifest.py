#!/usr/bin/env -S PYTHONDONTWRITEBYTECODE=1 python3

"""Writes the manifest of a build of the chained overlay for generating tokens (gemv_chain.py), which the host reads
to drive it (ggml/src/ggml-xdna/xdna-chain.h, XdnaChainOverlay):

  gemv_chain_manifest.py <project folder> <manifest>

<project folder> is the .prj folder of the build: the placement is read in its input_with_addresses.mlir. The
manifest says the version (xdna2-gemv 9: the blocks of a segment one after the other in the objects, with no padding
between them, an object of Y for every row of a block, phases of the recurrence of a layer of linear attention, and
phases of a whole layer of linear attention, with its convolution and its gated norm), the array, the kernel, the bytes
of an object of a stream and of a core, the bytes of X a core keeps,
the segments of a phase and the phases of a command the cores run, the formats, and the shim tile and channel of the
stream of weights W and of the stream of sums Y of every channel.
"""

import re
import sys
from pathlib import Path

cColumns = 8
cRows = 6
cMemoryRows = 1
cKernel = "xdna2_chain_gemv"
cObjectBytes = 9216
cCoreBytes = 4608
cSegments = 2      # as gemv_chain.py runs them
cPhases = 4


def fManifest(pProject):
  vText = (pProject / "input_with_addresses.mlir").read_text()
  if not re.search(rf"func\.func private @{cKernel}\(", vText):
    raise RuntimeError(f"the kernel {cKernel} is not in the build: is this a build of gemv_chain.py?")
  if "cascade" not in vText:
    raise RuntimeError("the build has no cascade between the cores of a pair")
  if not re.search(r"func\.func private @xdna2_chain_conv\(", vText):
    raise RuntimeError("the build has no phase of linear attention (version 9): is it a build of an older gemv_chain.py?")
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
  vMatch = re.search(rf"func\.func private @{cKernel}\(memref<(\d+)xui8>", vText)
  if not vMatch:
    raise RuntimeError("the buffer of X of the cores is not found")
  vKeptX = int(vMatch.group(1))
  lLines = [
    "xdna2-gemv 9",
    f"array {cColumns} {cRows} {cMemoryRows}",
    f"kernel {cKernel}",
    f"object {cObjectBytes} {cCoreBytes}",
    f"x {vKeptX}",
    f"segments {cSegments}",
    f"phases {cPhases}",
    "formats bfp16 q4k q5k q6k",
  ]
  for (vName, vNumber), (vColumn, vChannel) in sorted(dStreams.items()):
    lLines.append(f"stream {vName} {vNumber} {vColumn} {vChannel}")
  return "\n".join(lLines) + "\n"


def fMain():
  if len(sys.argv) != 3:
    raise RuntimeError("usage: gemv_chain_manifest.py <project folder> <manifest>")
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
