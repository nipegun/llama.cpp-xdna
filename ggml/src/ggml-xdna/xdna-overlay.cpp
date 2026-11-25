#include "xdna-overlay.h"

#include <fstream>
#include <sstream>

namespace {

const uint32_t cManifestVersion = 1;
const uint32_t cMaxCoreRows     = 16;
const uint32_t cMaxCoreColumns  = 64;
const uint32_t cMaxStreams      = 64;

// puts a stream at its number, growing the list
bool fSetStream(std::vector<XdnaOverlayStream> & aStreams, uint32_t pNumber, uint32_t pColumn, uint32_t pChannel) {
  if (pNumber >= cMaxStreams || pColumn > 255 || pChannel > 255) {
    return false;
  }
  if (aStreams.size() <= pNumber) {
    aStreams.resize(pNumber + 1, XdnaOverlayStream{255, 255});
  }
  aStreams[pNumber] = {uint8_t(pColumn), uint8_t(pChannel)};
  return true;
}

bool fComplete(const std::vector<XdnaOverlayStream> & aStreams) {
  for (const XdnaOverlayStream & vStream : aStreams) {
    if (vStream.vColumn == 255) {
      return false;
    }
  }
  return !aStreams.empty();
}

}  // namespace

bool XdnaOverlay::mLoad(const std::string & pPath, std::string & pError) {
  std::ifstream vFile(pPath);
  if (!vFile) {
    pError = "cannot open the manifest " + pPath;
    return false;
  }
  *this = XdnaOverlay();

  // the cores come in any order: keep them with their place until all are read
  struct XdnaPlacedCore {
    uint32_t         vRow;
    uint32_t         vColumn;
    XdnaOverlayCore vCore;
  };
  std::vector<XdnaPlacedCore> aPlaced;
  std::vector<bool>            aBandSeen;

  bool        vHasVersion = false;
  std::string vLine;
  uint32_t    vLineNumber = 0;
  while (std::getline(vFile, vLine)) {
    vLineNumber++;
    std::istringstream vWords(vLine);
    std::string        vKey;
    if (!(vWords >> vKey) || vKey[0] == '#') {
      continue;
    }
    bool vOk = true;
    if (vKey == "xdna2-overlay") {
      uint32_t vVersion = 0;
      vOk = bool(vWords >> vVersion) && vVersion == cManifestVersion;
      vHasVersion = vOk;
    } else if (vKey == "array") {
      vOk = bool(vWords >> vColumns >> vRows >> vMemoryRows);
    } else if (vKey == "kernel") {
      vOk = bool(vWords >> vKernel);
    } else if (vKey == "layout") {
      std::string vName;
      vOk = bool(vWords >> vName) && (vName == "rows" || vName == "columns");
      vLayout = vName == "columns" ? XdnaGemmLayout::cColumns : XdnaGemmLayout::cRows;
    } else if (vKey == "tile") {
      vOk = bool(vWords >> vTileM >> vTileK >> vTileN);
    } else if (vKey == "fixed") {
      vOk = bool(vWords >> vFixedK >> vFixedN);
    } else if (vKey == "output") {
      std::string vName;
      vOk = bool(vWords >> vName) && (vName == "f32" || vName == "bf16" || vName == "bfp16");
      vOutput = vName == "bf16"  ? XdnaGemmOutput::cBfloat16 :
                vName == "bfp16" ? XdnaGemmOutput::cBfp16 :
                                   XdnaGemmOutput::cFloat;
    } else if (vKey == "split") {
      vOk = bool(vWords >> vSplit) && (vSplit == 1 || vSplit == 2);
    } else if (vKey == "gdn") {
      vOk = bool(vWords >> vGdn) && vGdn <= 1;
    } else if (vKey == "rows") {
      vOk = bool(vWords >> vRowsC) && vRowsC <= 1;
    } else if (vKey == "comp") {
      vOk = bool(vWords >> vCompB) && vCompB <= 1;
    } else if (vKey == "core") {
      uint32_t vRow = 0, vColumn = 0, vTileColumn = 0, vTileRow = 0, vAddress = 0, vLockNumber = 0;
      vOk = bool(vWords >> vRow >> vColumn >> vTileColumn >> vTileRow >> vAddress >> vLockNumber) &&
            vRow < cMaxCoreRows && vColumn < cMaxCoreColumns && vTileColumn <= 255 && vTileRow <= 255;
      if (vOk) {
        aPlaced.push_back({vRow, vColumn, {uint8_t(vTileColumn), uint8_t(vTileRow), vAddress, vLockNumber}});
      }
    } else if (vKey == "stream") {
      std::string vName;
      uint32_t    vNumber = 0, vColumn = 0, vChannel = 0;
      vOk = bool(vWords >> vName >> vNumber >> vColumn >> vChannel);
      if (vOk && vName == "A") {
        vOk = fSetStream(aStreamsA, vNumber, vColumn, vChannel);
      } else if (vOk && vName == "B") {
        vOk = fSetStream(aStreamsB, vNumber, vColumn, vChannel);
      } else if (vOk && vName == "C") {
        vOk = fSetStream(aStreamsC, vNumber, vColumn, vChannel);
      } else if (vOk && vName == "D") {
        vOk = fSetStream(aStreamsD, vNumber, vColumn, vChannel);
      } else {
        vOk = false;
      }
    } else if (vKey == "capacity") {
      vOk = bool(vWords >> vBandWords);
    } else if (vKey == "band") {
      uint32_t aValues[9] = {};
      for (uint32_t i = 0; vOk && i < 9; i++) {
        vOk = bool(vWords >> aValues[i]) && aValues[i] <= 255;
      }
      vOk = vOk && aValues[0] < cMaxStreams;
      if (vOk) {
        if (aBands.size() <= aValues[0]) {
          aBands.resize(aValues[0] + 1);
          aBandSeen.resize(aValues[0] + 1, false);
        }
        XdnaOverlayBand & vBand = aBands[aValues[0]];
        vBand.vTileColumn    = uint8_t(aValues[1]);
        vBand.vTileRow       = uint8_t(aValues[2]);
        vBand.vChannelIn     = uint8_t(aValues[3]);
        vBand.vDescriptorIn  = uint8_t(aValues[4]);
        vBand.vChannelOut    = uint8_t(aValues[5]);
        vBand.vDescriptorOut = uint8_t(aValues[6]);
        vBand.vLockFree      = uint8_t(aValues[7]);
        vBand.vLockFull      = uint8_t(aValues[8]);
        aBandSeen[aValues[0]] = true;
      }
    } else {
      vOk = false;
    }
    if (!vOk) {
      pError = pPath + ": line " + std::to_string(vLineNumber) + " is not understood";
      return false;
    }
  }
  for (bool vSeen : aBandSeen) {
    if (!vSeen) {
      pError = pPath + ": the band of a stream of A is missing in the manifest";
      return false;
    }
  }
  if (!aBands.empty() && (aBands.size() != aStreamsA.size() || vBandWords == 0)) {
    pError = pPath + ": the bands of the manifest do not match its streams of A";
    return false;
  }

  if (!vHasVersion || vColumns == 0 || vRows == 0 || vTileM == 0 || vTileK == 0 || vTileN == 0) {
    pError = pPath + ": the manifest is not complete";
    return false;
  }
  for (const XdnaPlacedCore & vPlaced : aPlaced) {
    vCoreRows    = vPlaced.vRow + 1 > vCoreRows ? vPlaced.vRow + 1 : vCoreRows;
    vCoreColumns = vPlaced.vColumn + 1 > vCoreColumns ? vPlaced.vColumn + 1 : vCoreColumns;
  }
  if (aPlaced.empty() || aPlaced.size() != size_t(vCoreRows) * vCoreColumns) {
    pError = pPath + ": the cores of the manifest do not fill a grid";
    return false;
  }
  aCores.assign(aPlaced.size(), XdnaOverlayCore());
  std::vector<bool> aSeen(aPlaced.size(), false);
  for (const XdnaPlacedCore & vPlaced : aPlaced) {
    const size_t vAt = size_t(vPlaced.vRow) * vCoreColumns + vPlaced.vColumn;
    if (aSeen[vAt]) {
      pError = pPath + ": a core appears twice in the manifest";
      return false;
    }
    aSeen[vAt] = true;
    aCores[vAt] = vPlaced.vCore;
  }
  if (!fComplete(aStreamsA) || !fComplete(aStreamsB) || !fComplete(aStreamsC) ||
      (vSplit == 2 && !fComplete(aStreamsD))) {
    pError = pPath + ": a stream is missing in the manifest";
    return false;
  }
  // the halves of B are joined in the memory tiles of the layout "columns", and only there go the streams D
  if ((vSplit == 2 && (vLayout != XdnaGemmLayout::cColumns || aStreamsD.size() != vCoreColumns ||
                       aStreamsB.size() != size_t(vCoreRows) * vSplit || vCoreRows < 2)) ||
      (vSplit == 1 && !aStreamsD.empty())) {
    pError = pPath + ": the streams of the manifest do not match its split";
    return false;
  }
  // the rows of C come from the tiles that a memory tile joins, as the bytes of C a sum takes
  if (vRowsC == 1 && (vSplit != 1 || vLayout != XdnaGemmLayout::cColumns || vOutput == XdnaGemmOutput::cBfp16)) {
    pError = pPath + ": rows of C need C in bf16 or f32, the layout columns and no split";
    return false;
  }
  return true;
}
