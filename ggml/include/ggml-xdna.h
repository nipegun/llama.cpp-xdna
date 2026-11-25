#pragma once

// XDNA backend of llama.cpp-xdna: the NPU of AMD Ryzen AI processors, through the amdxdna driver of Linux.
//
// It is an accelerator device: it takes the multiplications of a model by its matrices of weights and leaves
// every other operation to the processor. What it does and how is described in doc/XDNA2.md.
//
// It reads these environment variables when it is registered:
//
//   GGML_XDNA_DISABLE      any value but 0: the backend registers no device
//   GGML_XDNA_DEVICE       device node of the NPU; /dev/accel/accel0 if not set
//   GGML_XDNA_OVERLAY_DIR  folder of the overlays (xdna2-gemm-bfp16.pdi and xdna2-gemm-bfp16.manifest); the
//                           folder of the program if not set
//   GGML_XDNA_MIN_BATCH    rows a multiplication needs for the backend to take it; 72 if not set. With fewer, the
//                           processor is faster (the generation of tokens, small prompts)
//   GGML_XDNA_EMULATE      any value but 0: the processor does what the NPU would do, with the same numbers.
//                           For development: slow, and the NPU is not opened

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_xdna_reg(void);

#ifdef  __cplusplus
}
#endif
