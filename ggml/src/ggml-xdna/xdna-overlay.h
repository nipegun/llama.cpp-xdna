#pragma once

// What the host has to know about an overlay to drive it: where the placer of mlir-aie put every stream and
// every core. It is read from the manifest that is written next to the overlay when it is built, a text file:
//
//   xdna2-overlay 1
//   array <columns> <rows> <rows of memory tiles>
//   kernel <name>
//   layout rows|columns
//   tile <m> <k> <n>
//   fixed <K> <N>
//   output f32|bf16|bfp16                                                                              (f32 if absent)
//   split 1|2                                                                                            (1 if absent)
//   gdn 0|1                                                                                              (0 if absent)
//   core <row> <column> <tile column> <tile row> <address of its parameters> <lock that starts it>     (one per core)
//   stream A|B|C|D <number> <shim column> <channel>                                                    (one per stream)
//   capacity <32-bit words>
//   band <stream of A> <tile column> <tile row> <channel in> <descriptor in> <channel out> <descriptor out>
//        <lock free> <lock full>                                                                       (one per stream of A)
//
// "layout" says which operand goes where: with "rows", A goes to the rows of cores and B to the columns; with
// "columns", A goes to the columns of cores, kept in their memory tiles, and B to the rows. "fixed" holds the K
// and the N an overlay is built for, or 0 when it serves any.
//
// "output" says how C comes back. With f32 every sum is a 32-bit float, and the cores add up the tiles of K in
// the tile of C they send. With bf16 every sum is rounded to bfloat16, to the nearest and ties to even: half the
// bytes. The cores then add up in 32-bit floats in a buffer of their own, write the rounded tile with the last
// tile of K, and their first and last tiles of K are different calls: K has to be two tiles at least. With bfp16
// C is rounded to block floating point, 9 bytes for 8 values, every 8 rows of one column with one exponent: 8
// tokens of one output, of a similar size (the cores compute C transposed in every 8 x 8 for that), so that in
// the buffer every 8 x 8 of the tile is 8 blocks, one per column, of its 8 rows. The same, otherwise, as bf16.
//
// "split 2", with the layout "columns", says that every tile of B comes from the host in two halves, each through a
// stream of its own, which the memory tile of the row of cores joins: the streams B 2r and 2r + 1 carry the first
// and the second half of the tiles of the row r. B then has as many streams as A. The memory tiles that join them
// lack an input for the tile of C of one core, so the last row of cores sends its tiles of C straight to the host,
// through the streams D, one per column of cores. Without it, a stream B carries the whole tiles of a row of
// cores and the memory tile of a column joins the tiles of C of all its cores.
//
// "gdn 1" says that the cores can also run the recurrence of a gated delta net by chunks (xdna-gemm.h,
// fXdnaGdnPlan): they have four runtime parameters instead of two, and a multiplication sets the last two to 0.
//
// llama.cpp-xdna: "rows 1", with C in bf16 or f32, the layout "columns" and no split, says that the cores write
// every tile of C as rows of the tile, one after the other (kernel xdna2_mm_bfp16_s3_last_rows), and not by groups
// of 8 x 8: the shims then lay the tiles out in the buffer of C as strips of the width of a tile, [padded N / tile
// N][padded M][tile N], every tile 8 KB in a row, so that the host reads every strip in order (see fXdnaGemmPlan).
//
// "band" and "capacity" describe, for the layout "columns", the memory tile that keeps the band of A of every
// stream: the channel and the buffer descriptor that receive it from the host, the ones that send it to the
// cores, the lock that counts the sends still free and the one that counts the sends still owed, and how many
// words the band can have. With them the host sets the size of the band and how many times it is sent before
// every multiplication, so the overlay serves any K that fits and any N.

#include <cstdint>
#include <string>
#include <vector>

enum class XdnaGemmLayout {
  cRows,     // A to the rows of cores, B to the columns
  cColumns,  // A to the columns of cores, B to the rows
};

// how the sums of C come back
enum class XdnaGemmOutput {
  cFloat,     // 32-bit floats
  cBfloat16,  // rounded to bfloat16, to the nearest and ties to even
  cBfp16,     // rounded to block floating point, every 8 rows of an output with one exponent
};

// one core of the overlay
struct XdnaOverlayCore {
  uint8_t  vTileColumn = 0;
  uint8_t  vTileRow    = 0;
  uint32_t vParameters = 0;  // address, in the data memory of the tile, of its runtime parameters (two, or four)
  uint32_t vLock       = 0;  // lock that lets it start once the parameters are written
};

// one stream between the host and the array: a channel of a shim tile
struct XdnaOverlayStream {
  uint8_t vColumn  = 0;
  uint8_t vChannel = 0;
};

// the memory tile that keeps the band of A of one stream and repeats it to the cores
struct XdnaOverlayBand {
  uint8_t  vTileColumn    = 0;
  uint8_t  vTileRow       = 0;
  uint8_t  vChannelIn     = 0;  // from the host
  uint8_t  vDescriptorIn  = 0;
  uint8_t  vChannelOut    = 0;  // to the cores
  uint8_t  vDescriptorOut = 0;
  uint8_t  vLockFree      = 0;  // sends the band can still get before a new one may come in
  uint8_t  vLockFull      = 0;  // sends of the band still owed to the cores
};

struct XdnaOverlay {
  uint32_t        vColumns    = 0;  // of the array
  uint32_t        vRows       = 0;
  uint32_t        vMemoryRows = 0;
  std::string     vKernel;
  XdnaGemmLayout vLayout     = XdnaGemmLayout::cRows;
  uint32_t        vTileM      = 0;  // rows of A, values of K and columns of B that a core multiplies in one call
  uint32_t        vTileK      = 0;
  uint32_t        vTileN      = 0;
  uint32_t        vFixedK     = 0;  // sizes the overlay is built for; 0: any
  uint32_t        vFixedN     = 0;
  XdnaGemmOutput vOutput     = XdnaGemmOutput::cFloat;
  uint32_t        vSplit      = 1;  // pieces of every tile of B; with 2, the last row of cores sends C straight
  uint32_t        vGdn        = 0;  // 1: the cores can also run the recurrence of a gated delta net
  uint32_t        vRowsC      = 0;  // llama.cpp-xdna, 1: the buffer of C holds strips of C ("rows 1")
  uint32_t        vCoreRows    = 0;  // rows and columns of cores
  uint32_t        vCoreColumns = 0;
  std::vector<XdnaOverlayCore>   aCores;     // by rows of cores: [row * vCoreColumns + column]
  std::vector<XdnaOverlayStream> aStreamsA;  // host to array
  std::vector<XdnaOverlayStream> aStreamsB;  // host to array
  std::vector<XdnaOverlayStream> aStreamsC;  // array to host, through the memory tiles
  std::vector<XdnaOverlayStream> aStreamsD;  // array to host, the tiles of C of the last row of cores (split 2)
  std::vector<XdnaOverlayBand>   aBands;         // per stream of A, when their memory tiles can be set; else empty
  uint32_t                        vBandWords = 0;  // 32-bit words a band can have

  // Reads a manifest. False, with the reason in pError, when it cannot be read or is not complete.
  bool mLoad(const std::string & pPath, std::string & pError);
};
