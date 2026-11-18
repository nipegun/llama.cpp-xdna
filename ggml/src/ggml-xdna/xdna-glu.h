#pragma once

// SwiGLU on the host, as the CPU backend of ggml computes it (ggml_vec_swiglu_f32 of ggml-cpu/vec.cpp): silu(g) * u,
// silu(x) being x / (1 + exp(-x)), with the vector exp of ggml (adapted from the optimized routines of Arm) for the
// elements of a row that a vector of the CPU backend takes and expf for the rest, so that both give the same floats.
// The variant of the CPU backend of a machine follows its features, and so does this: AVX-512 (F and DQ, 16 floats
// at a time), AVX2 with FMA (8), or none (every element with expf; the CPU backend would use SSE2 there, which can
// differ in the last bit).

#include <cstdint>

// pOut[i] = silu(pGate[i]) * pUp[i] for pCount consecutive elements of a row of pRowLength, the first of them at
// the column pFirst of the row.
void fXdnaSwiglu(const float * pGate, const float * pUp, float * pOut, uint32_t pCount, uint64_t pFirst,
                  uint64_t pRowLength);

// The same for 8 rows by 8 columns given as bfloat16, 8 rows of 8 (an 8 x 8 of the C of the overlay of the prompts),
// row r into pOut + r * pStride: the columns have to be among those a vector of the CPU backend takes (below the
// last pRowLength % width of a row), and are then computed by vectors of the width of the CPU backend.
void fXdnaSwigluBlock(const uint16_t * pGate, const uint16_t * pUp, float * pOut, uint64_t pStride);

// whether every column from pFirst to pFirst + 8 of a row of pRowLength goes by vectors (fXdnaSwigluBlock)
bool fXdnaSwigluBlockFits(uint64_t pFirst, uint64_t pRowLength);

// llama.cpp-xdna: for the attention of a batch on the overlay of the prompts (fXdnaGemmPackAttnP, xdna-gemm.h). A run
// of pCount scores of a row of S as the NPU gives them (bfloat16), each plus the value of the mask for its key when
// pMask is given (halves of IEEE, as the KQ mask of llama.cpp: 0 or minus infinity), by vectors of AVX-512 or AVX2 when
// the machine has them:
// - fXdnaScoresMax: the largest of them (minus infinity when all are masked)
float fXdnaScoresMax(const uint16_t * pScores, const uint16_t * pMask, uint32_t pCount);
// - fXdnaScoresExp: pOut[i] = exp(score + mask - pShift), with the vector exp of ggml (expf for what no vector takes),
//   0 for a masked one, and their sum
float fXdnaScoresExp(const uint16_t * pScores, const uint16_t * pMask, float pShift, float * pOut, uint32_t pCount);
// llama.cpp-xdna: pCount halves of IEEE (the F16 cache of ggml) as floats, with F16C when the machine has it
void fXdnaHalvesToFloats(const uint16_t * pHalves, float * pFloats, uint32_t pCount);

// llama.cpp-xdna: the scales of RMS_NORM for pCount rows of pLength floats, pStride floats apart, as the CPU backend
// computes them (ggml_compute_forward_rms_norm_f32): the squares in float added up in double, in order, and then
// 1 / sqrt(sum / length + eps) in float, so that they are the same floats. With AVX, 8 rows at a time, one in every
// lane of two vectors of doubles; every row still adds its squares in order.
void fXdnaRmsScales(const float * pRows, uint64_t pCount, uint64_t pLength, uint64_t pStride, float pEps,
                     float * pScales);

// llama.cpp-xdna: a token of the short convolution of Qwen3.5 by tokens, as the CPU backend computes SSM_CONV by
// tokens (ggml_compute_forward_ssm_conv_f32_by_tokens, ops.cpp): for every one of the pCount channels i, the 4 taps
// aTaps[k][i] (the oldest first, the token last) by their weights pWeights[k * pWeightStride + i] in 4 fused
// multiply-adds from 0, in that order, and with pSilu its SiLU as ggml_vec_silu_f32, into pOut[i]. Every value goes
// through the lane of a vector, as on the CPU backend when its channels are a multiple of its vectors (16 floats with
// AVX-512, 8 with AVX2), so that both give the same floats.
void fXdnaConvToken(const float * const aTaps[4], const float * pWeights, uint64_t pWeightStride, float * pOut,
                     uint32_t pCount, bool pSilu);
// The same with its SiLU for the rows pFirstRow to pEndRow (3 or more) of a strip of C of the overlay of the prompts,
// pWidth bfloat16 a row (the 4 taps of row t are its rows t - 3 to t, converted in the registers), the outputs of row t
// at pOut + t * pOutStride, and with pStream past the cache; one call for every strip, not for every row. False, with
// nothing written, when pCount is not a multiple of the vectors of the machine (16 floats with AVX-512, 8 with AVX2),
// it has neither, or pFirstRow is less than 3
bool fXdnaConvRowsBf16(const uint16_t * pStrip, uint64_t pWidth, uint64_t pFirstRow, uint64_t pEndRow,
                        const float * pWeights, uint64_t pWeightStride, float * pOut, uint64_t pOutStride,
                        uint32_t pCount, bool pStream);

// llama.cpp-xdna: the SiLU of pCount floats as ggml_vec_silu_f32, every value through the lane of a vector (the same
// floats as the CPU backend for rows that are a multiple of its vectors)
void fXdnaSilu(const float * pIn, float * pOut, uint32_t pCount);
