#!/usr/bin/env -S PYTHONDONTWRITEBYTECODE=1 python3

"""Moves the buffers of every core of a build of bfpmm_resident.py to other addresses of its data memory, to test
how the banks of the memory change the speed (mlir-aie 1.4.5 cannot pin the buffers of an object fifo to a bank).

  core_banks.py <input_with_addresses.mlir> <output.mlir>

The data memory of a core of AIE2P is 4 banks of 16 KB (AIETargetModel.h, getNumBanks). While the core reads one
tile of A and one of B, its DMA writes the next ones; mlir-aie puts the two tiles of B in the same bank, and the
second one in the bank of the first tile of A. Here every tile of B has a bank of its own, which nothing that the
DMA writes at the same time shares (B is what the kernel reads most: every block 8 times per call):

  bank 0  stack, B 0, first part of C
  bank 1  rest of C, B 1, runtime parameters, data of the core, first part of A 0
  bank 2  rest of A 0, A 1
  bank 3  sums

The output is given to aiecc with the kernel objects next to it, which builds the overlay from it as it is.
"""

import re
import sys
from pathlib import Path

cStack = 3840
cBank = 16384  # bytes of a bank of the data memory of a core
# the new address of every buffer of a core, by the start of its name
dAddresses = {
  "B_L2L1_": (3840, 21248),   # buf0, buf1
  "C_L1L2_": (13056,),
  "rtp_": (30464,),
  "core_data_": (30496,),  # aligned to 256 bits, as aiecc asks
  "A_L2L1_": (30528, 39744),
  "sums_": (49152,),
}
# the runtime parameters: two words, or four with --gdn 1, which still end before the data of the core
dSizes = {"B_L2L1_": 9216, "C_L1L2_": 8192, "rtp_": 16, "core_data_": 8, "A_L2L1_": 9216, "sums_": 16384}
cMemory = 65536


def fMain():
  if len(sys.argv) != 3:
    raise RuntimeError("usage: core_banks.py <input_with_addresses.mlir> <output.mlir>")
  vText = Path(sys.argv[1]).read_text()
  vPattern = re.compile(
    r'(%(\w+) = aie\.buffer\(%tile_(\d+)_(\d+)\) \{address = )(\d+)( : i32.*?mem_bank = )(\d+)( : i32.*?sym_name = "(\w+)")'
  )
  dSeen = {}
  vMoved = 0

  def fReplace(pMatch):
    nonlocal vMoved
    vName = pMatch.group(9)
    vTile = (int(pMatch.group(3)), int(pMatch.group(4)))
    if vTile[1] < 2:
      return pMatch.group(0)  # a memory tile
    vPrefix = next((vKey for vKey in dAddresses if vName.startswith(vKey)), None)
    if vPrefix is None:
      raise RuntimeError(f"a buffer of a core that is not known: {vName}")
    vIndex = 0
    vBuffer = re.search(r"__buf(\d+)$", vName)
    if vBuffer:
      vIndex = int(vBuffer.group(1))
    vAddress = dAddresses[vPrefix][vIndex]
    dSeen.setdefault(vTile, []).append((vAddress, vAddress + dSizes[vPrefix], vName))
    vMoved += 1
    # the bank it starts in, as aiecc records it
    return pMatch.group(1) + str(vAddress) + pMatch.group(6) + str(vAddress // cBank) + pMatch.group(8)

  vNew = vPattern.sub(fReplace, vText)
  # every core: the same buffers, inside the memory, after the stack, none on top of another
  for vTile, lBuffers in sorted(dSeen.items()):
    lBuffers.sort()
    if len(lBuffers) != 8:
      raise RuntimeError(f"the core {vTile} has {len(lBuffers)} buffers, not 8")
    vEnd = cStack
    for vStart, vStop, vName in lBuffers:
      if vStart < vEnd or vStop > cMemory:
        raise RuntimeError(f"the buffer {vName} of the core {vTile} overlaps or does not fit")
      vEnd = vStop
  Path(sys.argv[2]).write_text(vNew)
  print(f"moved {vMoved} buffers of {len(dSeen)} cores")


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
