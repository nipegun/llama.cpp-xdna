// llama.cpp-xdna: minimal Qwen3-TTS audio generation and speaker-cloning implementation of libmtmd (mtmd.h and
// mtmd-helper.h), ported from tools/mtmd/ of upstream llama.cpp.
// Vision, video, streaming transcription and unrelated TTS architectures are excluded (mtmd_support_vision() is always
// false, preserving the SSRF/media-URL mitigations of llama-server).

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "mtmd.h"
#include "mtmd-helper.h"
#include "models/models.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"
#include "llama-cpp.h"
#include "../src/llama-ext.h"

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#define MA_API static
#include "miniaudio/miniaudio.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#define MTMD_UNSUPPORTED_MSG "only Qwen3-TTS audio generation is included in llama.cpp-xdna"

static ggml_log_callback g_default_log_cb = [](ggml_log_level, const char * text, void *) {
    fputs(text, stderr);
    fflush(stderr);
};

static ggml_log_callback g_log_callback  = g_default_log_cb;
static void *            g_log_user_data = nullptr;

static void mtmd_log(ggml_log_level level, const char * fmt, ...) {
    if (!fmt) {
        return;
    }
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    char buf[256];
    const int n = vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (n >= 0 && (size_t) n < sizeof(buf)) {
        g_log_callback(level, buf, g_log_user_data);
    } else if (n >= 0) {
        std::vector<char> dyn((size_t) n + 1);
        vsnprintf(dyn.data(), dyn.size(), fmt, ap2);
        g_log_callback(level, dyn.data(), g_log_user_data);
    }
    va_end(ap2);
}

#define LOG_DBG(...) mtmd_log(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_INF(...) mtmd_log(GGML_LOG_LEVEL_INFO,  __VA_ARGS__)
#define LOG_WRN(...) mtmd_log(GGML_LOG_LEVEL_WARN,  __VA_ARGS__)
#define LOG_ERR(...) mtmd_log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

static void mtmd_stub_log_error(const char * func) {
    LOG_ERR("%s: " MTMD_UNSUPPORTED_MSG "\n", func);
}

std::string mtmd_format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    const int size = vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    if (size < 0) {
        va_end(ap2);
        return "";
    }
    std::vector<char> buf((size_t) size + 1);
    vsnprintf(buf.data(), buf.size(), fmt, ap2);
    va_end(ap2);
    return std::string(buf.data(), (size_t) size);
}

//
// Backends and scheduler (with XDNA ACCEL selection)
//

static void mtmd_backend_set_n_threads(ggml_backend_t backend, int n_threads) {
    if (!backend) {
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        auto fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (fn) {
            fn(backend, std::max(1, n_threads));
        }
    }
}

static std::shared_ptr<ggml_threadpool> mtmd_threadpool_init(int n_threads) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg || n_threads < 2) {
        return nullptr;
    }
    auto * new_fn  = (decltype(ggml_threadpool_new) *)  ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
    auto * free_fn = (decltype(ggml_threadpool_free) *) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
    if (!new_fn || !free_fn) {
        return nullptr;
    }
    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);
    tpp.poll = 0;
    ggml_threadpool_t tp = new_fn(&tpp);
    if (!tp) {
        throw std::runtime_error("failed to create threadpool");
    }
    return std::shared_ptr<ggml_threadpool>(tp, free_fn);
}

static ggml_backend_dev_t mtmd_select_device(bool use_gpu, ggml_backend_dev_t device) {
    if (!use_gpu) {
        return nullptr;
    }
    if (device != nullptr) {
        return device;
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    // llama.cpp-xdna: use the XDNA NPU accelerator for mmproj (code_predictor and code2wav) when available
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_ACCEL);
    }
    return dev;
}

struct mtmd_backends {
    ggml_backend_t backend     = nullptr;
    ggml_backend_t backend_cpu = nullptr;

    std::vector<ggml_backend_t>             ptrs;
    std::vector<ggml_backend_buffer_type_t> bufts;
    std::shared_ptr<ggml_threadpool>        threadpool;
    ggml_backend_sched_ptr                  sched;

    mtmd_backends(ggml_backend_dev_t device, int n_threads, int max_nodes,
                  ggml_backend_sched_eval_callback cb_eval, void * cb_eval_user_data) {
        threadpool = mtmd_threadpool_init(n_threads);

        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!backend_cpu) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        if (threadpool) {
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
            auto * set_tp = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
            if (set_tp) {
                set_tp(backend_cpu, threadpool.get());
            }
        }
        if (device != nullptr) {
            backend = ggml_backend_dev_init(device, nullptr);
            if (!backend) {
                ggml_backend_free(backend_cpu);
                throw std::runtime_error(mtmd_format("mtmd_backends: failed to initialize \"%s\" backend",
                                                     ggml_backend_dev_name(device)));
            }
            mtmd_backend_set_n_threads(backend, n_threads);
        }

        if (backend) {
            LOG_INF("mtmd_backends: using %s backend\n", ggml_backend_name(backend));
            ptrs.push_back(backend);
            bufts.push_back(ggml_backend_get_default_buffer_type(backend));
        } else {
            backend = backend_cpu;
            LOG_INF("mtmd_backends: using CPU backend\n");
        }

        ptrs.push_back(backend_cpu);
        bufts.push_back(ggml_backend_get_default_buffer_type(backend_cpu));

        sched.reset(ggml_backend_sched_new(ptrs.data(), bufts.data(), ptrs.size(), max_nodes, false, true));
        if (cb_eval != nullptr) {
            ggml_backend_sched_set_eval_callback(sched.get(), cb_eval, cb_eval_user_data);
        }
    }

    ~mtmd_backends() {
        sched.reset();
        ggml_backend_free(backend);
        if (backend != backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }

    std::map<ggml_backend_dev_t, size_t> get_mem_compute() const {
        std::map<ggml_backend_dev_t, size_t> res;
        for (ggml_backend_t b : ptrs) {
            res[ggml_backend_get_device(b)] += ggml_backend_sched_get_buffer_size(sched.get(), b);
        }
        return res;
    }
};

//
// Graph base
//

mtmd_graph::mtmd_graph(mtmd_subcontext * ctx, const mtmd_image_f32 & img) :
        model(ctx->model),
        hparams(model.hparams),
        proj_type(ctx->model.proj_type),
        img(img),
        n_embd(hparams.n_embd),
        n_head(hparams.n_head),
        n_head_kv(hparams.n_head_kv),
        d_head(hparams.n_embd_head > 0 ? hparams.n_embd_head : (n_head > 0 ? n_embd / n_head : 0)),
        n_layer(hparams.n_layer),
        n_mmproj_embd(ctx->model.n_mmproj_embd()),
        eps(hparams.eps),
        kq_scale(d_head > 0 ? 1.0f / sqrtf((float) d_head) : 0.0f),
        flash_attn_type(ctx->flash_attn_type) {
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    ctx0_ptr.reset(ggml_init(params));
    ctx0 = ctx0_ptr.get();
    gf = ggml_new_graph_custom(ctx0, ctx->max_nodes, false);
}

mtmd_graph::mtmd_graph(const mtmd_graph & parent) :
        model(parent.model),
        hparams(parent.hparams),
        proj_type(parent.proj_type),
        img(parent.img),
        n_embd(parent.n_embd),
        n_head(parent.n_head),
        n_head_kv(parent.n_head_kv),
        d_head(parent.d_head),
        n_layer(parent.n_layer),
        n_mmproj_embd(parent.n_mmproj_embd),
        eps(parent.eps),
        kq_scale(parent.kq_scale),
        flash_attn_type(parent.flash_attn_type),
        ctx0(parent.ctx0),
        gf(parent.gf) {}

void mtmd_graph::cb(ggml_tensor * cur, const char * name, int il) const {
    if (il >= 0) {
        ggml_format_name(cur, "%s-%d", name, il);
    } else {
        ggml_set_name(cur, name);
    }
}

ggml_tensor * mtmd_graph::build_mm(ggml_tensor * w, ggml_tensor * x) const {
    return ggml_mul_mat(ctx0, w, x);
}

ggml_tensor * mtmd_graph::build_inp_raw(int channels) {
    ggml_tensor * inp_raw = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, img.nx(), img.ny(), channels, n_batch);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);
    return inp_raw;
}

ggml_tensor * mtmd_graph::build_attn(
        ggml_tensor * wo,
        ggml_tensor * wo_b,
        ggml_tensor * q_cur,
        ggml_tensor * k_cur,
        ggml_tensor * v_cur,
        ggml_tensor * kq_mask,
        float scale,
        int il) const {
    ggml_build_forward_order(gf, q_cur);
    ggml_build_forward_order(gf, k_cur);
    ggml_build_forward_order(gf, v_cur);

    ggml_tensor * q = ggml_permute(ctx0, q_cur, 0, 2, 1, 3);
    ggml_tensor * k = ggml_permute(ctx0, k_cur, 0, 2, 1, 3);
    ggml_tensor * cur;

    if (flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED) {
        ggml_tensor * v = ggml_permute(ctx0, v_cur, 0, 2, 1, 3);
        k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        GGML_ASSERT(!kq_mask || kq_mask->type == GGML_TYPE_F16);
        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
    } else {
        ggml_tensor * v = ggml_permute(ctx0, v_cur, 1, 2, 0, 3);
        v = ggml_cont(ctx0, v);
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, scale, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0] * cur->ne[1], cur->ne[2] * cur->ne[3]);
    }

    cb(cur, "kqv_out", il);

    if (wo) {
        cur = build_mm(wo, cur);
    }
    if (wo_b) {
        cur = ggml_add(ctx0, cur, wo_b);
    }
    return cur;
}

static std::unique_ptr<mtmd_graph> mtmd_get_graph_builder(
        mtmd_subcontext * ctx,
        const mtmd_image_f32_batch & imgs,
        const mtmd_subctx_encode_params * params = nullptr) {
    const mtmd_image_f32 & img = imgs.entries[0];
    switch (ctx->model.proj_type) {
        case PROJECTOR_TYPE_QWEN3TTS_SPKENC:
            return std::make_unique<mtmd_graph_qwen3tts_spkenc>(ctx, img);
        case PROJECTOR_TYPE_QWEN3TTS_GEN: {
            const mtmd_subctx_gen_process proc = params ? params->gen_process : MTMD_SUBCTX_GEN_PROCESS_GEN_CODE;
            const int   tk = params ? params->top_k : 50;
            const float tp = params ? params->top_p : 1.0f;
            return std::make_unique<mtmd_graph_qwen3tts_gen>(ctx, img, proc, tk, tp);
        }
        default:
            throw std::runtime_error("unsupported projector type in mtmd_get_graph_builder");
    }
}

//
// Subcontext
//

mtmd_subcontext::mtmd_subcontext(const mtmd_model & model, mtmd_backends & backends, mtmd_flash_attn_type fa_type, bool do_warmup)
    : model(model), backends(backends), max_nodes(model.max_nodes), flash_attn_type(fa_type) {
    if (flash_attn_type == MTMD_FLASH_ATTN_TYPE_AUTO && !model.flash_attn_auto) {
        flash_attn_type = MTMD_FLASH_ATTN_TYPE_DISABLED;
    }
    buf_compute_meta.resize((size_t) max_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(max_nodes, false));

    mtmd_image_f32_batch batch;
    mtmd_image_f32 img;
    if (model.modality == MTMD_MODALITY_GEN_AUDIO) {
        img.set_size({model.n_mmproj_embd(), 1}, false, true);
    } else {
        if (model.hparams.n_mel_bins <= 0 || model.hparams.n_mel_bins > 256) {
            throw std::runtime_error(mtmd_format("invalid n_mel_bins (%d), must be in [1, 256]", model.hparams.n_mel_bins));
        }
        img.set_size({model.hparams.warmup_audio_size, model.hparams.n_mel_bins}, false, true);
    }
    batch.entries.push_back(std::move(img));

    if (do_warmup) {
        warmup(batch);
    }
}

void mtmd_subcontext::warmup(const mtmd_image_f32_batch & batch) {
    auto reserve_once = [&]() -> bool {
        ggml_cgraph * gf = mtmd_get_graph_builder(this, batch)->build();
        ggml_backend_sched_reserve(backends.sched.get(), gf);
        ggml_backend_t backend_cpu = ggml_backend_sched_get_backend(
                backends.sched.get(), ggml_backend_sched_get_n_backends(backends.sched.get()) - 1);
        const bool is_host_accel = ggml_backend_dev_type(ggml_backend_get_device(backends.backend)) == GGML_BACKEND_DEVICE_TYPE_ACCEL;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->op == GGML_OP_FLASH_ATTN_EXT &&
                !ggml_backend_supports_op(backends.backend, node) &&
                !(is_host_accel && ggml_backend_supports_op(backend_cpu, node))) {
                return false;
            }
        }
        return true;
    };

    if (flash_attn_type == MTMD_FLASH_ATTN_TYPE_AUTO) {
        flash_attn_type = MTMD_FLASH_ATTN_TYPE_ENABLED;
        if (!reserve_once()) {
            flash_attn_type = MTMD_FLASH_ATTN_TYPE_DISABLED;
            reserve_once();
        }
    } else {
        reserve_once();
    }
    is_allocated = true;
}

static void copy_gen_state_out(ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_nbytes(src) == ggml_nbytes(dst));
    if (src->type == dst->type && ggml_are_same_shape(src, dst) && ggml_is_contiguous(src)) {
        ggml_backend_tensor_copy(src, dst);
        return;
    }
    std::vector<uint8_t> tmp(ggml_nbytes(src));
    ggml_backend_tensor_get(src, tmp.data(), 0, tmp.size());
    ggml_backend_tensor_set(dst, tmp.data(), 0, tmp.size());
}

mtmd_subctx_gen_state * mtmd_subcontext::gen_state_init(uint32_t seed) {
    auto st = std::make_unique<mtmd_subctx_gen_state>();
    st->rng.seed(seed == UINT32_MAX ? std::random_device{}() : seed);

    if (model.proj_type != PROJECTOR_TYPE_QWEN3TTS_GEN) {
        return st.release();
    }
    const auto slots = list_c2w_state_slots(model.hparams, model);
    if (slots.empty()) {
        return st.release();
    }

    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * slots.size(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    st->ctx_data.reset(ggml_init(params));
    for (const auto & slot : slots) {
        ggml_tensor * t = ggml_new_tensor_2d(st->ctx_data.get(), GGML_TYPE_F32, slot.ne0, slot.ne1);
        ggml_format_name(t, "gen_state_%s", slot.name.c_str());
        st->slots.push_back(t);
    }
    st->buf.reset(ggml_backend_alloc_ctx_tensors(st->ctx_data.get(), backends.backend));
    if (!st->buf) {
        LOG_ERR("%s: failed to allocate gen state\n", __func__);
        return nullptr;
    }
    ggml_backend_buffer_clear(st->buf.get(), 0);
    return st.release();
}

bool mtmd_subcontext::encode(mtmd_subctx_encode_params * params) {
    const mtmd_image_f32_batch & imgs = *params->imgs;
    if (imgs.entries.size() != 1) {
        LOG_ERR("%s: expected batch size 1, got %zu\n", __func__, imgs.entries.size());
        return false;
    }

    if (!is_allocated) {
        warmup(imgs);
    }

    const bool needs_gen_state = params->gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_CODE ||
                                 params->gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_WAV;
    if (needs_gen_state && params->gen_state == nullptr) {
        LOG_ERR("%s: audio gen requires a gen state\n", __func__);
        return false;
    }

    ggml_backend_sched_reset(backends.sched.get());
    ggml_cgraph * gf = mtmd_get_graph_builder(this, imgs, params)->build();
    ggml_tensor * embeddings = params->out_embd ? ggml_graph_node(gf, -1) : nullptr;

    if (!ggml_backend_sched_alloc_graph(backends.sched.get(), gf)) {
        LOG_ERR("%s: failed to allocate compute graph\n", __func__);
        return false;
    }

    auto get_inp_tensor = [&gf](const char * name) -> ggml_tensor * {
        ggml_tensor * inp = ggml_graph_get_tensor(gf, name);
        if (!inp || !(inp->flags & GGML_TENSOR_FLAG_INPUT)) {
            throw std::runtime_error(std::string("missing or non-input tensor: ") + name);
        }
        return inp;
    };
    auto set_input_f32 = [&get_inp_tensor](const char * name, const std::vector<float> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_F32 && ggml_nelements(cur) == (int64_t) values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };
    auto set_input_i32 = [&get_inp_tensor](const char * name, const std::vector<int32_t> & values) {
        ggml_tensor * cur = get_inp_tensor(name);
        GGML_ASSERT(cur->type == GGML_TYPE_I32 && ggml_nelements(cur) == (int64_t) values.size());
        ggml_backend_tensor_set(cur, values.data(), 0, ggml_nbytes(cur));
    };

    if (params->gen_process != MTMD_SUBCTX_GEN_PROCESS_GEN_WAV &&
        ggml_graph_get_tensor(gf, "inp_raw") != nullptr) {
        const auto & mel_inp = imgs.entries[0];
        const auto & buf = mel_inp.get_ro_buf();
        GGML_ASSERT((size_t) mel_inp.nx() * (size_t) mel_inp.ny() == buf.size());
        set_input_f32("inp_raw", buf);
    }

    if (model.proj_type == PROJECTOR_TYPE_QWEN3TTS_GEN) {
        if (params->gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_WAV) {
            if (!params->codes) {
                LOG_ERR("%s: missing codes for GEN_WAV\n", __func__);
                return false;
            }
            const int64_t n_codes    = model.n_acoustic_codebooks() + 1;
            const int64_t n_frames_w = model.hparams.wav_tfm_swa;
            const int64_t n_frames   = (int64_t) params->codes->size() / n_codes;
            if (n_frames <= 0 || n_frames > n_frames_w) {
                LOG_ERR("%s: invalid frame count %lld (window %lld)\n", __func__, (long long) n_frames, (long long) n_frames_w);
                return false;
            }

            const int64_t vocab_first = model.c2w.quant_first_cb_w->ne[1];
            const int64_t vocab_rest  = model.c2w.quant_rest_cb_w->ne[1];
            for (int64_t f = 0; f < n_frames; f++) {
                for (int64_t g = 0; g < n_codes; g++) {
                    const int32_t c = (*params->codes)[(size_t) (f * n_codes + g)];
                    const int64_t vocab = (g == 0) ? vocab_first : vocab_rest;
                    if (c < 0 || (int64_t) c >= vocab) {
                        LOG_ERR("%s: code out of range (frame %lld, group %lld, code %d, vocab %lld)\n",
                                __func__, (long long) f, (long long) g, c, (long long) vocab);
                        return false;
                    }
                }
            }

            std::vector<int32_t> codes((size_t) (n_frames_w * n_codes), 0);
            for (int64_t f = 0; f < n_frames; f++) {
                for (int64_t g = 0; g < n_codes; g++) {
                    codes[(size_t) (g * n_frames_w + f)] = (*params->codes)[(size_t) (f * n_codes + g)];
                }
            }
            set_input_i32("inp_codes", codes);

            const auto & slots = params->gen_state->slots;
            size_t i = 0;
            for (const auto & slot : list_c2w_state_slots(model.hparams, model)) {
                GGML_ASSERT(i < slots.size());
                if (slot.in_place) {
                    i++;
                    continue;
                }
                ggml_backend_tensor_copy(slots[i++], get_inp_tensor(("state_in_" + slot.name).c_str()));
            }
        } else {
            const int64_t vocab0 = model.gen_code_out_embd_w->ne[1];
            if (params->code0 < 0 || (int64_t) params->code0 >= vocab0) {
                LOG_ERR("%s: code0 out of range (%d, vocab %lld)\n", __func__, params->code0, (long long) vocab0);
                return false;
            }
            std::vector<int32_t> code0 = { params->code0 };
            set_input_i32("inp_code0", code0);

            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            const int64_t n_acoustic = model.n_acoustic_codebooks();
            for (int64_t g = 0; g < n_acoustic; g++) {
                std::vector<float> r = { dist(params->gen_state->rng) };
                set_input_f32(("inp_rand_" + std::to_string(g)).c_str(), r);
            }
        }
    }

    mtmd_backend_set_n_threads(backends.backend_cpu, params->n_threads);
    if (backends.backend != backends.backend_cpu) {
        mtmd_backend_set_n_threads(backends.backend, params->n_threads);
    }

    const ggml_status status = ggml_backend_sched_graph_compute(backends.sched.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        LOG_ERR("%s: ggml_backend_sched_graph_compute failed with error %d\n", __func__, status);
        return false;
    }

    if (embeddings != nullptr) {
        params->out_embd->resize((size_t) ggml_nelements(embeddings));
        ggml_backend_tensor_get(embeddings, params->out_embd->data(), 0, ggml_nbytes(embeddings));
    }

    if (params->out_codes != nullptr) {
        ggml_tensor * codes = ggml_graph_get_tensor(gf, "out_codes");
        if (codes != nullptr) {
            params->out_codes->resize((size_t) ggml_nelements(codes));
            ggml_backend_tensor_get(codes, params->out_codes->data(), 0, ggml_nbytes(codes));
        }
    }

    if (params->out_audio != nullptr) {
        ggml_tensor * audio = ggml_graph_get_tensor(gf, "out_audio");
        if (!audio) {
            LOG_ERR("%s: graph has no out_audio tensor\n", __func__);
            return false;
        }
        auto & out_audio = *params->out_audio;
        out_audio.resize((size_t) ggml_nelements(audio));
        ggml_backend_tensor_get(audio, out_audio.data(), 0, ggml_nbytes(audio));

        const int64_t n_codes    = params->codes ? model.n_acoustic_codebooks() + 1 : 0;
        const int64_t n_frames_w = model.hparams.wav_tfm_swa;
        const int64_t n_frames   = params->codes ? (int64_t) params->codes->size() / n_codes : n_frames_w;
        if (n_frames < n_frames_w) {
            const size_t hop = out_audio.size() / (size_t) n_frames_w;
            out_audio.resize((size_t) n_frames * hop);
        }
    }

    if (params->gen_process == MTMD_SUBCTX_GEN_PROCESS_GEN_WAV) {
        const auto & slots = params->gen_state->slots;
        size_t i = 0;
        for (const auto & slot : list_c2w_state_slots(model.hparams, model)) {
            if (slot.in_place) {
                i++;
                continue;
            }
            ggml_tensor * t = ggml_graph_get_tensor(gf, ("state_out_" + slot.name).c_str());
            if (!t || i >= slots.size()) {
                LOG_ERR("%s: missing state_out_%s\n", __func__, slot.name.c_str());
                return false;
            }
            copy_gen_state_out(t, slots[i++]);
        }
    }

    return true;
}

//
// Slaney Mel Spectrogram Preprocessor for Qwen3-TTS Speaker Encoder (24 kHz, 128 mel bins)
//

struct mtmd_audio_cache {
    std::vector<float> sin_vals;
    std::vector<float> cos_vals;
    std::vector<float> hann_window;
    std::vector<float> filters;

    void fill_sin_cos_table(uint32_t n) {
        sin_vals.resize(n);
        cos_vals.resize(n);
        for (uint32_t i = 0; i < n; i++) {
            double theta = (2 * M_PI * i) / n;
            sin_vals[i] = sinf(theta);
            cos_vals[i] = cosf(theta);
        }
    }

    void fill_hann_window(uint32_t length, bool periodic) {
        hann_window.resize(length);
        int offset = periodic ? 0 : -1;
        for (uint32_t i = 0; i < length; i++) {
            hann_window[i] = 0.5 * (1.0 - cosf((2.0 * M_PI * i) / (length + offset)));
        }
    }

    void fill_mel_filterbank_matrix(int64_t n_mel, int64_t n_fft, int sample_rate) {
        const double fmin = 0.0;
        const double fmax = 0.5 * sample_rate;
        const double min_log_hz  = 1000.0;
        const double lin_slope   = 3.0 / 200.0;
        const double min_log_mel = min_log_hz * lin_slope;
        const double log_step    = std::log(6.4) / 27.0;
        auto hz_to_mel = [&](double f_hz) -> double {
            return (f_hz < min_log_hz) ? f_hz * lin_slope : min_log_mel + std::log(f_hz / min_log_hz) / log_step;
        };
        auto mel_to_hz = [&](double m) -> double {
            return (m < min_log_mel) ? m / lin_slope : min_log_hz * std::exp((m - min_log_mel) * log_step);
        };

        const double bin_hz_step = double(sample_rate) / double(n_fft);
        const double m_lo = hz_to_mel(fmin);
        const double m_hi = hz_to_mel(fmax);
        std::vector<double> hz_pts((size_t) n_mel + 2);
        for (int64_t i = 0; i < n_mel + 2; ++i) {
            const double m = m_lo + (m_hi - m_lo) * (double(i) / double(n_mel + 1));
            hz_pts[(size_t) i] = mel_to_hz(m);
        }

        const int64_t n_fft_bins = n_fft / 2 + 1;
        filters.assign((size_t) n_mel * (size_t) n_fft_bins, 0.0f);
        for (int64_t m = 0; m < n_mel; ++m) {
            const double f_left   = hz_pts[(size_t) m];
            const double f_center = hz_pts[(size_t) m + 1];
            const double f_right  = hz_pts[(size_t) m + 2];
            const double denom_l  = std::max(1e-30, f_center - f_left);
            const double denom_r  = std::max(1e-30, f_right - f_center);
            const double enorm    = 2.0 / std::max(1e-30, f_right - f_left); // Slaney area norm
            for (int64_t k = 0; k < n_fft_bins; ++k) {
                const double f = double(k) * bin_hz_step;
                double w = 0.0;
                if (f >= f_left && f <= f_center) {
                    w = (f - f_left) / denom_l;
                } else if (f > f_center && f <= f_right) {
                    w = (f_right - f) / denom_r;
                }
                filters[(size_t) m * (size_t) n_fft_bins + (size_t) k] = (float) (w * enorm);
            }
        }
    }
};

struct mtmd_qwen3tts_spk_preproc {
    mtmd_hparams     hparams;
    mtmd_audio_cache cache;
    int              n_threads = 4;
    ggml_backend_sched_t sched = nullptr;

    explicit mtmd_qwen3tts_spk_preproc(const mtmd_hparams & hp) : hparams(hp) {
        cache.fill_sin_cos_table((uint32_t) hparams.audio_n_fft);
        cache.fill_hann_window((uint32_t) hparams.audio_window_len, true);
        cache.fill_mel_filterbank_matrix(hparams.n_mel_bins, hparams.audio_n_fft, hparams.audio_sample_rate);
    }

    bool compute_mel(const float * samples_in, size_t n_samples_in, mtmd_image_f32 & out_entry) const {
        const int pad = (hparams.audio_n_fft - hparams.audio_hop_len) / 2; // 384
        if (n_samples_in < (size_t) pad + 1) {
            LOG_ERR("mtmd_qwen3tts_spk_preproc: speaker audio clip too short (%zu samples)\n", n_samples_in);
            return false;
        }
        std::vector<float> padded(n_samples_in + 2 * (size_t) pad, 0.0f);
        for (int i = 0; i < pad; i++) {
            padded[(size_t) i] = samples_in[pad - i];
        }
        std::copy(samples_in, samples_in + n_samples_in, padded.begin() + pad);
        for (int i = 0; i < pad; i++) {
            padded[n_samples_in + (size_t) pad + (size_t) i] = samples_in[n_samples_in - 2 - (size_t) i];
        }

        const int64_t n_fft_bins = 1 + (hparams.audio_n_fft / 2);
        const int64_t frame_size = (n_fft_bins - 1) * 2;
        const int64_t hop        = hparams.audio_hop_len;
        const int64_t n_mel      = hparams.n_mel_bins;
        if ((int64_t) padded.size() < frame_size) {
            return false;
        }
        const int64_t n_frames = ((int64_t) padded.size() - frame_size) / hop + 1;
        if (n_frames <= 0 || n_frames > INT32_MAX) {
            return false;
        }

        ggml_backend_t backend_cpu = ggml_backend_sched_get_backend(sched, ggml_backend_sched_get_n_backends(sched) - 1);
        mtmd_backend_set_n_threads(backend_cpu, n_threads);

        std::vector<float> dft((size_t) (2 * n_fft_bins * frame_size));
        for (int64_t k = 0; k < n_fft_bins; k++) {
            for (int64_t j = 0; j < frame_size; j++) {
                const int64_t q = (j * k) % frame_size;
                dft[(size_t) (k * frame_size + j)]                = cache.hann_window[(size_t) j] * cache.cos_vals[(size_t) q];
                dft[(size_t) ((k + n_fft_bins) * frame_size + j)] = cache.hann_window[(size_t) j] * cache.sin_vals[(size_t) q];
            }
        }

        std::vector<float> mel_out((size_t) n_mel * (size_t) n_frames, 0.0f);
        const int64_t n_frames_max = 8192;
        for (int64_t off = 0; off < n_frames; off += n_frames_max) {
            const int64_t n_cur = std::min(n_frames_max, n_frames - off);
            ggml_init_params params = {
                /*.mem_size   =*/ ggml_tensor_overhead() * 32 + ggml_graph_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr ctx_ptr(ggml_init(params));
            ggml_context * ctx0 = ctx_ptr.get();

            ggml_tensor * inp_samples = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, (n_cur - 1) * hop + frame_size);
            ggml_tensor * inp_dft     = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, frame_size, 2 * n_fft_bins);
            ggml_tensor * inp_filters = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_fft_bins, n_mel);
            ggml_set_input(inp_samples);
            ggml_set_input(inp_dft);
            ggml_set_input(inp_filters);

            ggml_tensor * cur = ggml_cont(ctx0, ggml_view_2d(ctx0, inp_samples, frame_size, n_cur, (size_t) hop * sizeof(float), 0));
            cur = ggml_mul_mat(ctx0, inp_dft, cur);
            ggml_prec_set_src(cur, GGML_PREC_F32, 1);

            cur = ggml_sqr(ctx0, cur);
            cur = ggml_add(ctx0,
                    ggml_view_2d(ctx0, cur, n_fft_bins, n_cur, cur->nb[1], 0),
                    ggml_view_2d(ctx0, cur, n_fft_bins, n_cur, cur->nb[1], (size_t) n_fft_bins * sizeof(float)));
            cur = ggml_sqrt(ctx0, cur); // magnitude

            cur = ggml_mul_mat(ctx0, cur, inp_filters);
            ggml_prec_set_src(cur, GGML_PREC_F32, 1);
            cur = ggml_clamp(ctx0, cur, 1e-5f, FLT_MAX);
            cur = ggml_log(ctx0, cur); // natural log
            ggml_set_output(cur);

            ggml_cgraph * gf = ggml_new_graph(ctx0);
            ggml_build_forward_expand(gf, cur);

            ggml_backend_sched_reset(sched);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                return false;
            }
            ggml_backend_tensor_set(inp_samples, padded.data() + off * hop, 0, ggml_nbytes(inp_samples));
            ggml_backend_tensor_set(inp_dft,     dft.data(),                0, ggml_nbytes(inp_dft));
            ggml_backend_tensor_set(inp_filters, cache.filters.data(),      0, ggml_nbytes(inp_filters));

            if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
                return false;
            }
            for (int64_t j = 0; j < n_mel; j++) {
                ggml_backend_tensor_get(cur, mel_out.data() + j * n_frames + off, (size_t) j * cur->nb[1], (size_t) n_cur * sizeof(float));
            }
        }

        out_entry.set_size({(int) n_frames, (int) n_mel}, false, true);
        out_entry.cpy_buf(mel_out);
        return true;
    }
};

//
// Minimal Qwen3-TTS mmproj GGUF loader (with strict exception validation)
//

struct mtmd_models_pair {
    std::unique_ptr<mtmd_model> a;
    std::unique_ptr<mtmd_model> gen_a;
};

struct mtmd_qwen3tts_loader {
    ggml_context_ptr ctx_meta;
    gguf_context_ptr ctx_gguf;
    std::string      fname;
    bool has_vision    = false;
    bool has_audio     = false;
    bool has_gen_audio = false;

    mtmd_qwen3tts_loader(const char * path, bool /*skip_tensors*/) : fname(path) {
        ggml_context * meta = nullptr;
        gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
        ctx_gguf.reset(gguf_init_from_file(path, params));
        if (!ctx_gguf) {
            throw std::runtime_error(mtmd_format("failed to load mmproj model from %s", path));
        }
        ctx_meta.reset(meta);

        get_bool("clip.has_vision_encoder",    has_vision,    false);
        get_bool("clip.has_audio_encoder",     has_audio,     false);
        get_bool("clip.has_gen_audio_encoder", has_gen_audio, false);
    }

    void get_bool(const std::string & key, bool & out, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("missing GGUF key: " + key);
            return;
        }
        if (gguf_get_kv_type(ctx_gguf.get(), i) != GGUF_TYPE_BOOL) {
            throw std::runtime_error("expected bool for GGUF key: " + key);
        }
        out = gguf_get_val_bool(ctx_gguf.get(), i);
    }

    void get_u32(const std::string & key, int32_t & out, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("missing GGUF key: " + key);
            return;
        }
        if (gguf_get_kv_type(ctx_gguf.get(), i) != GGUF_TYPE_UINT32) {
            throw std::runtime_error("expected uint32 for GGUF key: " + key);
        }
        const uint32_t val = gguf_get_val_u32(ctx_gguf.get(), i);
        if (val > (uint32_t) INT32_MAX) {
            throw std::runtime_error("GGUF key exceeds INT32_MAX: " + key);
        }
        out = (int32_t) val;
    }

    void get_f32(const std::string & key, float & out, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("missing GGUF key: " + key);
            return;
        }
        if (gguf_get_kv_type(ctx_gguf.get(), i) != GGUF_TYPE_FLOAT32) {
            throw std::runtime_error("expected float32 for GGUF key: " + key);
        }
        out = gguf_get_val_f32(ctx_gguf.get(), i);
    }

    void get_string(const std::string & key, std::string & out, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) throw std::runtime_error("missing GGUF key: " + key);
            return;
        }
        if (gguf_get_kv_type(ctx_gguf.get(), i) != GGUF_TYPE_STRING) {
            throw std::runtime_error("expected string for GGUF key: " + key);
        }
        out = gguf_get_val_str(ctx_gguf.get(), i);
    }

    std::unique_ptr<mtmd_model> load_modality(
            mtmd_modality modality,
            ggml_backend_buffer_type_t buft,
            bool no_alloc,
            mtmd_progress_callback progress_cb,
            void * progress_ud) const {
        auto model = std::make_unique<mtmd_model>();
        model->modality = modality;
        auto & hp = model->hparams;

        const bool is_audio = modality == MTMD_MODALITY_AUDIO;
        const char * kv_prefix = is_audio ? "audio" : "gen.audio";

        std::string proj_str;
        get_string("clip.projector_type", proj_str, false);
        if (proj_str.empty()) {
            get_string(is_audio ? "clip.audio.projector_type" : "clip.gen.audio.projector_type", proj_str, true);
        }
        if (is_audio && proj_str == "qwen3tts_spkenc") {
            model->proj_type = PROJECTOR_TYPE_QWEN3TTS_SPKENC;
        } else if (!is_audio && proj_str == "qwen3tts_gen") {
            model->proj_type = PROJECTOR_TYPE_QWEN3TTS_GEN;
        } else {
            throw std::runtime_error("unsupported mmproj projector_type '" + proj_str + "' (" MTMD_UNSUPPORTED_MSG ")");
        }

        get_u32(mtmd_format("clip.%s.embedding_length",             kv_prefix), hp.n_embd);
        get_u32(mtmd_format("clip.%s.attention.head_count",         kv_prefix), hp.n_head);
        get_u32(mtmd_format("clip.%s.attention.head_dim",           kv_prefix), hp.n_embd_head, false);
        get_u32(mtmd_format("clip.%s.feed_forward_length",          kv_prefix), hp.n_ff);
        get_u32(mtmd_format("clip.%s.block_count",                  kv_prefix), hp.n_layer);
        get_u32(mtmd_format("clip.%s.projection_dim",               kv_prefix), hp.projection_dim);
        get_f32(mtmd_format("clip.%s.attention.layer_norm_epsilon", kv_prefix), hp.eps);
        hp.n_head_kv = hp.n_head;
        get_u32(mtmd_format("clip.%s.attention.head_count_kv",      kv_prefix), hp.n_head_kv, false);

        if (hp.n_embd <= 0 || hp.n_head <= 0 || hp.n_head_kv <= 0 || hp.n_layer <= 0 || hp.n_layer > 64) {
            throw std::runtime_error("invalid mmproj hyperparameters in " + fname);
        }

        if (is_audio) {
            get_u32("clip.audio.num_mel_bins", hp.n_mel_bins);
            if (hp.n_mel_bins <= 0 || hp.n_mel_bins > 256 || hp.n_layer != 3) {
                throw std::runtime_error(mtmd_format("invalid qwen3tts_spkenc params: n_mel_bins=%d n_layer=%d", hp.n_mel_bins, hp.n_layer));
            }
            hp.audio_sample_rate = 24000;
            hp.audio_n_fft       = 1024;
            hp.audio_window_len  = 1024;
            hp.audio_hop_len     = 256;
        } else {
            get_string("clip.gen.audio.model_variant", hp.gen_model_variant, false);
            hp.rope_theta           = 1000000.0f;
            hp.wav_tfm_n_layer      = 8;
            hp.wav_tfm_n_embd       = 512;
            hp.wav_tfm_n_ff         = 1024;
            hp.wav_tfm_n_head       = 16;
            hp.wav_tfm_n_head_kv    = 16;
            hp.wav_tfm_eps          = 1e-5f;
            hp.wav_tfm_rope_theta   = 10000.0f;
            hp.wav_upsample_n_block = 2;
            hp.wav_dac_n_block      = 4;
            hp.wav_dac_n_res        = 3;
            hp.wav_tfm_swa          = 72;
        }

        const int n_gguf_tensors = gguf_get_n_tensors(ctx_gguf.get());
        const size_t data_base = gguf_get_data_offset(ctx_gguf.get());
        std::map<std::string, size_t> file_offsets;
        for (int i = 0; i < n_gguf_tensors; i++) {
            file_offsets[gguf_get_tensor_name(ctx_gguf.get(), i)] = data_base + gguf_get_tensor_offset(ctx_gguf.get(), i);
        }

        struct load_entry {
            ggml_tensor * dst;
            size_t        offset;
            size_t        nbytes;
        };
        std::vector<load_entry> to_load;

        ggml_init_params ctx_params = {
            /*.mem_size   =*/ (size_t) (n_gguf_tensors + 128) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        model->ctx_data.reset(ggml_init(ctx_params));

        const ggml_backend_dev_t buft_dev = ggml_backend_buft_get_device(buft);
        const bool use_conv_2d = buft_dev != nullptr && ggml_backend_dev_type(buft_dev) != GGML_BACKEND_DEVICE_TYPE_CPU;

        auto add_conv1d_2d = [&](ggml_tensor * w) {
            if (!use_conv_2d || !w || w->ne[2] <= 1) {
                return;
            }
            ggml_tensor * w2d = ggml_new_tensor_2d(model->ctx_data.get(), GGML_TYPE_F32, w->ne[0] * w->ne[1], w->ne[2]);
            ggml_format_name(w2d, "%s.2d", w->name);
            model->conv1d_w_2d[w] = w2d;
        };

        auto add_conv1d_tr_2d = [&](ggml_tensor * w) {
            if (!use_conv_2d || !w) {
                return;
            }
            ggml_tensor * wt = ggml_new_tensor_2d(model->ctx_data.get(), GGML_TYPE_F32, w->ne[2], w->ne[0] * w->ne[1]);
            ggml_format_name(wt, "%s.tr2d", w->name);
            model->conv1d_tr_w_2d[w] = wt;
        };

        auto get_tensor = [&](const std::string & name, bool required = true) -> ggml_tensor * {
            ggml_tensor * meta_t = ggml_get_tensor(ctx_meta.get(), name.c_str());
            if (!meta_t) {
                if (required) {
                    throw std::runtime_error("missing mmproj tensor: " + name);
                }
                return nullptr;
            }
            ggml_tensor * cur = ggml_dup_tensor(model->ctx_data.get(), meta_t);
            ggml_set_name(cur, name.c_str());
            to_load.push_back({cur, file_offsets.at(name), ggml_nbytes(cur)});
            return cur;
        };

        if (model->proj_type == PROJECTOR_TYPE_QWEN3TTS_SPKENC) {
            model->conv1d_1_w = get_tensor("a.conv1d.0.weight");
            model->conv1d_1_b = get_tensor("a.conv1d.0.bias");
            model->layers.resize((size_t) hp.n_layer);
            for (int il = 0; il < hp.n_layer; il++) {
                auto & layer = model->layers[(size_t) il];
                const int bid = il + 1;
                layer.conv_pw1_w = get_tensor(mtmd_format("a.blk.%d.conv_pw1.weight", bid));
                layer.conv_pw1_b = get_tensor(mtmd_format("a.blk.%d.conv_pw1.bias",   bid));
                layer.conv_pw2_w = get_tensor(mtmd_format("a.blk.%d.conv_pw2.weight", bid));
                layer.conv_pw2_b = get_tensor(mtmd_format("a.blk.%d.conv_pw2.bias",   bid));
                layer.se_conv1_w = get_tensor(mtmd_format("a.blk.%d.se_conv1.weight", bid));
                layer.se_conv1_b = get_tensor(mtmd_format("a.blk.%d.se_conv1.bias",   bid));
                layer.se_conv2_w = get_tensor(mtmd_format("a.blk.%d.se_conv2.weight", bid));
                layer.se_conv2_b = get_tensor(mtmd_format("a.blk.%d.se_conv2.bias",   bid));
                layer.res2_conv_w.resize(7);
                layer.res2_conv_b.resize(7);
                for (int xid = 0; xid < 7; xid++) {
                    layer.res2_conv_w[(size_t) xid] = get_tensor(mtmd_format("a.blk.%d.res2.%d.weight", bid, xid));
                    layer.res2_conv_b[(size_t) xid] = get_tensor(mtmd_format("a.blk.%d.res2.%d.bias",   bid, xid));
                }
            }
            model->conv_out_w     = get_tensor("a.conv_out.weight");
            model->conv_out_b     = get_tensor("a.conv_out.bias");
            model->spk_asp_attn_w = get_tensor("a.asp_attn.weight");
            model->spk_asp_attn_b = get_tensor("a.asp_attn.bias");
            model->spk_asp_tdnn_w = get_tensor("a.asp_tdnn.weight");
            model->spk_asp_tdnn_b = get_tensor("a.asp_tdnn.bias");
            model->mm_fc_w        = get_tensor("mm.a.fc.weight");
            model->mm_fc_b        = get_tensor("mm.a.fc.bias");
        } else {
            model->layers.resize((size_t) hp.n_layer);
            for (int il = 0; il < hp.n_layer; il++) {
                auto & layer = model->layers[(size_t) il];
                layer.q_w       = get_tensor(mtmd_format("a.gen.code.blk.%d.attn_q.weight",      il));
                layer.k_w       = get_tensor(mtmd_format("a.gen.code.blk.%d.attn_k.weight",      il));
                layer.v_w       = get_tensor(mtmd_format("a.gen.code.blk.%d.attn_v.weight",      il));
                layer.o_w       = get_tensor(mtmd_format("a.gen.code.blk.%d.attn_out.weight",    il));
                layer.q_norm    = get_tensor(mtmd_format("a.gen.code.blk.%d.attn_q_norm.weight", il));
                layer.k_norm    = get_tensor(mtmd_format("a.gen.code.blk.%d.attn_k_norm.weight", il));
                layer.ln_1_w    = get_tensor(mtmd_format("a.gen.code.blk.%d.ln1.weight",         il));
                layer.ln_2_w    = get_tensor(mtmd_format("a.gen.code.blk.%d.ln2.weight",         il));
                layer.ff_gate_w = get_tensor(mtmd_format("a.gen.code.blk.%d.ffn_gate.weight",    il));
                layer.ff_up_w   = get_tensor(mtmd_format("a.gen.code.blk.%d.ffn_up.weight",      il));
                layer.ff_down_w = get_tensor(mtmd_format("a.gen.code.blk.%d.ffn_down.weight",    il));
                if (layer.q_w->ne[1] % hp.n_head != 0 || (layer.q_w->ne[1] / hp.n_head) % 2 != 0) {
                    throw std::runtime_error("invalid code_predictor head geometry");
                }
            }
            model->gen_code_proj_in_w  = get_tensor("a.gen.code.proj_in.weight", false);
            model->gen_code_proj_in_b  = get_tensor("a.gen.code.proj_in.bias",   false);
            model->gen_code_embd_w     = get_tensor("a.gen.code.embd.weight");
            model->gen_code_out_embd_w = get_tensor("a.gen.code.out_embd.weight");
            model->gen_code_norm_w     = get_tensor("a.gen.code.output_norm.weight");

            // llama.cpp-xdna: split a.gen.code.head.weight ([n_embd, vocab, n_acoustic]) into standalone 2D tensors
            ggml_tensor * head_meta = ggml_get_tensor(ctx_meta.get(), "a.gen.code.head.weight");
            if (!head_meta || head_meta->ne[2] < 1 || head_meta->ne[2] > 64 || head_meta->ne[2] != model->gen_code_embd_w->ne[2]) {
                throw std::runtime_error("invalid a.gen.code.head.weight shape");
            }
            const int64_t n_acoustic  = head_meta->ne[2];
            const size_t  head_offset = file_offsets.at("a.gen.code.head.weight");
            const size_t  slice_bytes = head_meta->nb[2];
            model->gen_code_head_2d.resize((size_t) n_acoustic);
            for (int64_t g = 0; g < n_acoustic; g++) {
                ggml_tensor * h2d = ggml_new_tensor_2d(model->ctx_data.get(), head_meta->type, head_meta->ne[0], head_meta->ne[1]);
                ggml_format_name(h2d, "a.gen.code.head.%d.weight", (int) g);
                to_load.push_back({h2d, head_offset + (size_t) g * slice_bytes, slice_bytes});
                model->gen_code_head_2d[(size_t) g] = h2d;
            }

            auto & c2w = model->c2w;
            c2w.quant_first_in_w  = get_tensor("a.gen.wav.quant.first.in_proj.weight");
            c2w.quant_first_out_w = get_tensor("a.gen.wav.quant.first.out_proj.weight");
            c2w.quant_first_cb_w  = get_tensor("a.gen.wav.quant.first.codebook.weight");
            c2w.quant_rest_in_w   = get_tensor("a.gen.wav.quant.rest.in_proj.weight");
            c2w.quant_rest_out_w  = get_tensor("a.gen.wav.quant.rest.out_proj.weight");
            c2w.quant_rest_cb_w   = get_tensor("a.gen.wav.quant.rest.codebook.weight");
            if (c2w.quant_rest_cb_w->ne[2] != n_acoustic) {
                throw std::runtime_error("mismatched acoustic codebook count in code2wav");
            }

            c2w.pre_conv_w        = get_tensor("a.gen.wav.pre_conv.weight");
            c2w.pre_conv_b        = get_tensor("a.gen.wav.pre_conv.bias");
            c2w.tfm_in_proj_w     = get_tensor("a.gen.wav.tfm.in_proj.weight");
            c2w.tfm_in_proj_b     = get_tensor("a.gen.wav.tfm.in_proj.bias");
            c2w.tfm_out_proj_w    = get_tensor("a.gen.wav.tfm.out_proj.weight");
            c2w.tfm_out_proj_b    = get_tensor("a.gen.wav.tfm.out_proj.bias");
            c2w.tfm_output_norm_w = get_tensor("a.gen.wav.tfm.output_norm.weight");

            c2w.tfm_layers.resize((size_t) hp.wav_tfm_n_layer);
            for (int il = 0; il < hp.wav_tfm_n_layer; il++) {
                auto & layer = c2w.tfm_layers[(size_t) il];
                layer.q_w       = get_tensor(mtmd_format("a.gen.wav.tfm.blk.%d.attn_q.weight",   il));
                layer.k_w       = get_tensor(mtmd_format("a.gen.wav.tfm.blk.%d.attn_k.weight",   il));
                layer.v_w       = get_tensor(mtmd_format("a.gen.wav.tfm.blk.%d.attn_v.weight",   il));
                layer.o_w       = get_tensor(mtmd_format("a.gen.wav.tfm.blk.%d.attn_out.weight", il));
                layer.ln_1_w    = get_tensor(mtmd_format("a.gen.wav.tfm.blk.%d.ln1.weight",      il));
                layer.ln_2_w    = get_tensor(mtmd_format("a.gen.wav.tfm.blk.%d.ln2.weight",      il));
                layer.ls_1_w    = get_tensor(mtmd_format("a.gen.wav.tfm.blk.%d.ls1.weight",      il));
                layer.ls_2_w    = get_tensor(mtmd_format("a.gen.wav.tfm.blk.%d.ls2.weight",      il));
                layer.ff_gate_w = get_tensor(mtmd_format("a.gen.wav.tfm.blk.%d.ffn_gate.weight", il));
                layer.ff_up_w   = get_tensor(mtmd_format("a.gen.wav.tfm.blk.%d.ffn_up.weight",   il));
                layer.ff_down_w = get_tensor(mtmd_format("a.gen.wav.tfm.blk.%d.ffn_down.weight", il));
            }

            c2w.upsample.resize((size_t) hp.wav_upsample_n_block);
            for (int il = 0; il < hp.wav_upsample_n_block; il++) {
                auto & up = c2w.upsample[(size_t) il];
                up.conv_w   = get_tensor(mtmd_format("a.gen.wav.up.blk.%d.conv.weight",   il));
                up.conv_b   = get_tensor(mtmd_format("a.gen.wav.up.blk.%d.conv.bias",     il));
                up.dwconv_w = get_tensor(mtmd_format("a.gen.wav.up.blk.%d.dwconv.weight", il));
                up.dwconv_b = get_tensor(mtmd_format("a.gen.wav.up.blk.%d.dwconv.bias",   il));
                up.norm_w   = get_tensor(mtmd_format("a.gen.wav.up.blk.%d.norm.weight",   il));
                up.norm_b   = get_tensor(mtmd_format("a.gen.wav.up.blk.%d.norm.bias",     il));
                up.pw1_w    = get_tensor(mtmd_format("a.gen.wav.up.blk.%d.pw1.weight",    il));
                up.pw1_b    = get_tensor(mtmd_format("a.gen.wav.up.blk.%d.pw1.bias",      il));
                up.pw2_w    = get_tensor(mtmd_format("a.gen.wav.up.blk.%d.pw2.weight",    il));
                up.pw2_b    = get_tensor(mtmd_format("a.gen.wav.up.blk.%d.pw2.bias",      il));
                up.gamma    = get_tensor(mtmd_format("a.gen.wav.up.blk.%d.gamma",         il));
            }

            c2w.dac_entry_w = get_tensor("a.gen.wav.dac.entry.weight");
            c2w.dac_entry_b = get_tensor("a.gen.wav.dac.entry.bias");
            c2w.dac.resize((size_t) hp.wav_dac_n_block);
            for (int il = 0; il < hp.wav_dac_n_block; il++) {
                auto & blk = c2w.dac[(size_t) il];
                blk.snake_alpha = get_tensor(mtmd_format("a.gen.wav.dac.blk.%d.snake.alpha", il));
                blk.snake_beta  = get_tensor(mtmd_format("a.gen.wav.dac.blk.%d.snake.beta",  il));
                blk.conv_w      = get_tensor(mtmd_format("a.gen.wav.dac.blk.%d.conv.weight", il));
                blk.conv_b      = get_tensor(mtmd_format("a.gen.wav.dac.blk.%d.conv.bias",   il));
                if (blk.conv_w->ne[0] < 2 || blk.conv_w->ne[0] % 2 != 0) {
                    throw std::runtime_error("invalid DAC conv kernel width");
                }
                blk.res.resize((size_t) hp.wav_dac_n_res);
                for (int ir = 0; ir < hp.wav_dac_n_res; ir++) {
                    auto & res = blk.res[(size_t) ir];
                    res.act1_alpha = get_tensor(mtmd_format("a.gen.wav.dac.blk.%d.res.%d.act1.alpha",   il, ir));
                    res.act1_beta  = get_tensor(mtmd_format("a.gen.wav.dac.blk.%d.res.%d.act1.beta",    il, ir));
                    res.conv1_w    = get_tensor(mtmd_format("a.gen.wav.dac.blk.%d.res.%d.conv1.weight", il, ir));
                    res.conv1_b    = get_tensor(mtmd_format("a.gen.wav.dac.blk.%d.res.%d.conv1.bias",   il, ir));
                    res.act2_alpha = get_tensor(mtmd_format("a.gen.wav.dac.blk.%d.res.%d.act2.alpha",   il, ir));
                    res.act2_beta  = get_tensor(mtmd_format("a.gen.wav.dac.blk.%d.res.%d.act2.beta",    il, ir));
                    res.conv2_w    = get_tensor(mtmd_format("a.gen.wav.dac.blk.%d.res.%d.conv2.weight", il, ir));
                    res.conv2_b    = get_tensor(mtmd_format("a.gen.wav.dac.blk.%d.res.%d.conv2.bias",   il, ir));
                }
            }
            c2w.dac_post_snake_alpha = get_tensor("a.gen.wav.dac.post_snake.alpha");
            c2w.dac_post_snake_beta  = get_tensor("a.gen.wav.dac.post_snake.beta");
            c2w.dac_post_conv_w      = get_tensor("a.gen.wav.dac.post_conv.weight");
            c2w.dac_post_conv_b      = get_tensor("a.gen.wav.dac.post_conv.bias");

            add_conv1d_2d(c2w.pre_conv_w);
            for (const auto & up : c2w.upsample) {
                add_conv1d_tr_2d(up.conv_w);
            }
            add_conv1d_2d(c2w.dac_entry_w);
            for (const auto & blk : c2w.dac) {
                add_conv1d_tr_2d(blk.conv_w);
                for (const auto & res : blk.res) {
                    add_conv1d_2d(res.conv1_w);
                    add_conv1d_2d(res.conv2_w);
                }
            }
        }

        if (progress_cb) {
            progress_cb(0.0f, progress_ud);
        }

        size_t total_bytes = 0;
        for (const auto & e : to_load) {
            total_bytes += e.nbytes;
        }
        for (const auto & kv : model->conv1d_w_2d) {
            total_bytes += ggml_nbytes(kv.second);
        }
        for (const auto & kv : model->conv1d_tr_w_2d) {
            total_bytes += ggml_nbytes(kv.second);
        }

        if (no_alloc) {
            model->buf.reset(ggml_backend_buft_alloc_buffer(buft, 0));
        } else {
            model->buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(model->ctx_data.get(), buft));
            if (!model->buf) {
                throw std::runtime_error("failed to allocate mmproj weight buffer");
            }
            ggml_backend_buffer_set_usage(model->buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

            std::ifstream fin(fname, std::ios::binary);
            if (!fin) {
                throw std::runtime_error("failed to open mmproj file: " + fname);
            }
            std::vector<uint8_t> read_buf;
            size_t loaded_bytes = 0;
            const bool is_host = ggml_backend_buft_is_host(buft);
            for (const auto & e : to_load) {
                fin.seekg((std::streamoff) e.offset, std::ios::beg);
                if (!fin) {
                    throw std::runtime_error(std::string("failed to seek for tensor ") + e.dst->name);
                }
                if (is_host) {
                    fin.read(reinterpret_cast<char *>(e.dst->data), (std::streamsize) e.nbytes);
                } else {
                    read_buf.resize(e.nbytes);
                    fin.read(reinterpret_cast<char *>(read_buf.data()), (std::streamsize) e.nbytes);
                    ggml_backend_tensor_set(e.dst, read_buf.data(), 0, e.nbytes);
                }
                if (!fin) {
                    throw std::runtime_error(std::string("failed to read tensor ") + e.dst->name);
                }
                loaded_bytes += e.nbytes;
                if (progress_cb && total_bytes > 0) {
                    if (!progress_cb((float) loaded_bytes / (float) total_bytes, progress_ud)) {
                        throw std::runtime_error("mmproj loading cancelled by progress_callback");
                    }
                }
            }

            // llama.cpp-xdna: populate standalone 2D F32 weight tensors for ECAPA-TDNN and code2wav convolutions
            auto tensor_to_f32 = [&](const ggml_tensor * src, std::vector<float> & out) {
                const int64_t ne0  = src->ne[0];
                const int64_t rows = src->ne[1] * src->ne[2] * src->ne[3];
                out.resize((size_t) (ne0 * rows));
                std::vector<uint8_t> raw;
                const char * base = nullptr;
                if (is_host) {
                    base = static_cast<const char *>(src->data);
                } else {
                    raw.resize(ggml_nbytes(src));
                    ggml_backend_tensor_get(src, raw.data(), 0, raw.size());
                    base = reinterpret_cast<const char *>(raw.data());
                }
                const auto * traits = ggml_get_type_traits(src->type);
                const size_t row_bytes = ggml_row_size(src->type, ne0);
                for (int64_t r = 0; r < rows; r++) {
                    const char * src_row = base + (size_t) r * row_bytes;
                    float * dst_row = out.data() + (size_t) r * (size_t) ne0;
                    if (src->type == GGML_TYPE_F32) {
                        std::memcpy(dst_row, src_row, (size_t) ne0 * sizeof(float));
                    } else {
                        traits->to_float(src_row, dst_row, ne0);
                    }
                }
            };

            std::vector<float> tmp_f32;
            std::vector<float> tr_f32;
            for (const auto & kv : model->conv1d_w_2d) {
                tensor_to_f32(kv.first, tmp_f32);
                ggml_backend_tensor_set(kv.second, tmp_f32.data(), 0, tmp_f32.size() * sizeof(float));
            }
            for (const auto & kv : model->conv1d_tr_w_2d) {
                const ggml_tensor * w = kv.first; // [K, OC, IC] -> [IC, K * OC]
                tensor_to_f32(w, tmp_f32);
                const int64_t koc = w->ne[0] * w->ne[1];
                const int64_t ic  = w->ne[2];
                tr_f32.resize((size_t) (ic * koc));
                for (int64_t i = 0; i < ic; i++) {
                    for (int64_t m = 0; m < koc; m++) {
                        tr_f32[(size_t) (m * ic + i)] = tmp_f32[(size_t) (i * koc + m)];
                    }
                }
                ggml_backend_tensor_set(kv.second, tr_f32.data(), 0, tr_f32.size() * sizeof(float));
            }
        }
        model->mem_weights[ggml_backend_buft_get_device(buft)] = total_bytes;
        return model;
    }
};

//
// Internal mtmd types backing mtmd.h
//

struct mtmd_bitmap {
    uint32_t nx = 0;
    uint32_t ny = 1;
    std::string id;
    bool is_audio = true;
    bool is_voice = false;
    std::vector<unsigned char> data;

    mtmd_bitmap(const float * samples, size_t n_samples) : nx((uint32_t) n_samples), ny(1), is_audio(true) {
        if (samples && n_samples > 0) {
            data.resize(n_samples * sizeof(float));
            std::memcpy(data.data(), samples, data.size());
        }
    }

    explicit mtmd_bitmap(const std::string & voice_id) : nx(0), ny(0), id(voice_id), is_audio(false), is_voice(true) {}
};

struct mtmd_image_tokens {
    mtmd_image_f32_batch batch_f32;
    std::string id;
};

struct mtmd_input_chunk {
    mtmd_input_chunk_type type = MTMD_INPUT_CHUNK_TYPE_TEXT;
    std::vector<llama_token> tokens_text;
    std::unique_ptr<mtmd_image_tokens> tokens_audio;
};

struct mtmd_input_chunks {
    std::vector<mtmd_input_chunk> entries;
};

struct mtmd_context {
    int n_threads = 4;
    std::string media_marker;
    const llama_model * text_model = nullptr;

    std::unique_ptr<mtmd_backends>              backends;
    mtmd_models_pair                            models;
    std::unique_ptr<mtmd_subcontext>            ctx_a;
    std::unique_ptr<mtmd_subcontext>            ctx_gen_a;
    std::unique_ptr<mtmd_qwen3tts_spk_preproc>  audio_preproc;
    std::vector<std::string>                    voice_names; // llama.cpp-xdna: built-in <|spk_*|> voices (CustomVoice)

    std::mutex         compute_mutex;
    std::vector<float> out_embd;

    mtmd_context(const char * mmproj_fname, const llama_model * tm, const mtmd_context_params & p, bool no_alloc)
        : n_threads(std::max(1, p.n_threads)),
          media_marker(p.media_marker ? p.media_marker : mtmd_default_marker()),
          text_model(tm) {
        mtmd_qwen3tts_loader loader(mmproj_fname, false);
        if (loader.has_vision || !loader.has_gen_audio) {
            throw std::runtime_error(MTMD_UNSUPPORTED_MSG);
        }

        ggml_backend_dev_t dev = mtmd_select_device(p.use_gpu, p.device);
        ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_buffer_type(dev) : ggml_backend_cpu_buffer_type();

        if (loader.has_audio) {
            models.a = loader.load_modality(MTMD_MODALITY_AUDIO, buft, no_alloc, p.progress_callback, p.progress_callback_user_data);
        }
        models.gen_a = loader.load_modality(MTMD_MODALITY_GEN_AUDIO, buft, no_alloc, p.progress_callback, p.progress_callback_user_data);

        if (text_model != nullptr) {
            const int n_text_embd = llama_model_n_embd_inp(text_model);
            if (models.a && models.a->n_mmproj_embd() != n_text_embd) {
                throw std::runtime_error("speaker encoder projection_dim does not match text model n_embd");
            }
            if (models.gen_a->n_mmproj_embd() != n_text_embd) {
                throw std::runtime_error("code_predictor embedding width does not match text model n_embd");
            }
            // llama.cpp-xdna: discover built-in speaker tokens (<|spk_*|>) in CustomVoice models
            const llama_vocab * vocab = llama_model_get_vocab(text_model);
            if (vocab != nullptr) {
                const int32_t n_tok = llama_vocab_n_tokens(vocab);
                const int32_t start = std::max<int32_t>(0, n_tok - 4096);
                for (int32_t t = start; t < n_tok; t++) {
                    const char * piece = llama_vocab_get_text(vocab, t);
                    if (!piece) {
                        continue;
                    }
                    const std::string s(piece);
                    if (s.size() > 8 && s.compare(0, 6, "<|spk_") == 0 && s.compare(s.size() - 2, 2, "|>") == 0) {
                        std::string vname = s.substr(6, s.size() - 8);
                        for (char & c : vname) {
                            c = (char) std::tolower((unsigned char) c);
                        }
                        voice_names.push_back(std::move(vname));
                    }
                }
                std::sort(voice_names.begin(), voice_names.end());
                voice_names.erase(std::unique(voice_names.begin(), voice_names.end()), voice_names.end());
            }
        }

        const int max_nodes = 8192;
        backends = std::make_unique<mtmd_backends>(dev, n_threads, max_nodes, p.cb_eval, p.cb_eval_user_data);

        mtmd_flash_attn_type fa = MTMD_FLASH_ATTN_TYPE_AUTO;
        if (p.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
            fa = MTMD_FLASH_ATTN_TYPE_DISABLED;
        } else if (p.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            fa = MTMD_FLASH_ATTN_TYPE_ENABLED;
        }

        const bool do_warmup = p.warmup && !no_alloc;
        if (models.a) {
            ctx_a = std::make_unique<mtmd_subcontext>(*models.a, *backends, fa, do_warmup);
            audio_preproc = std::make_unique<mtmd_qwen3tts_spk_preproc>(models.a->hparams);
            audio_preproc->n_threads = n_threads;
            audio_preproc->sched     = backends->sched.get();
        }
        ctx_gen_a = std::make_unique<mtmd_subcontext>(*models.gen_a, *backends, fa, do_warmup);
    }
};

struct mtmd_gen_state {
    mtmd_subcontext *       sctx      = nullptr;
    mtmd_subctx_gen_state * gen_state = nullptr;
    std::vector<int32_t>    out_codes;
    std::vector<float>      out_embd;
    std::vector<float>      out_audio;

    ~mtmd_gen_state() {
        delete gen_state;
    }
};

//
// mtmd.h public API
//

const char * mtmd_default_marker() {
    return "<__media__>";
}

mtmd_context_params mtmd_context_params_default() {
    mtmd_context_params params {
        /* use_gpu           */ true,
        /* device            */ nullptr,
        /* print_timings     */ true,
        /* n_threads         */ 4,
        /* image_marker      */ nullptr,
        /* media_marker      */ mtmd_default_marker(),
        /* flash_attn_type   */ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /* warmup            */ true,
        /* image_min_tokens  */ -1,
        /* image_max_tokens  */ -1,
        /* cb_eval           */ nullptr,
        /* cb_eval_user_data */ nullptr,
        /* batch_max_tokens  */ 1024,
        /* progress_callback */ nullptr,
        /* progress_callback_user_data */ nullptr,
    };
    return params;
}

mtmd_context * mtmd_init_from_file(const char * mmproj_fname, const struct llama_model * text_model, const struct mtmd_context_params ctx_params) {
    try {
        return new mtmd_context(mmproj_fname, text_model, ctx_params, false);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return nullptr;
    }
}

void mtmd_free(mtmd_context * ctx) {
    delete ctx;
}

bool mtmd_decode_use_non_causal(const mtmd_context *, const mtmd_input_chunk *) {
    return false;
}

bool mtmd_decode_use_mrope(const mtmd_context * ctx) {
    return ctx && ctx->text_model &&
           (llama_model_rope_type(ctx->text_model) == LLAMA_ROPE_TYPE_MROPE ||
            llama_model_rope_type(ctx->text_model) == LLAMA_ROPE_TYPE_IMROPE);
}

bool mtmd_support_vision(const mtmd_context *) {
    return false;
}

bool mtmd_support_audio(const mtmd_context * ctx) {
    return ctx && ctx->ctx_a != nullptr;
}

int mtmd_get_audio_sample_rate(const mtmd_context * ctx) {
    return (ctx && ctx->ctx_a) ? ctx->ctx_a->model.hparams.audio_sample_rate : -1;
}

struct mtmd_video_config mtmd_get_video_config(const mtmd_context *, int32_t, int32_t, int32_t, float) {
    return {};
}

const char * mtmd_get_marker(const mtmd_context * ctx) {
    return ctx ? ctx->media_marker.c_str() : mtmd_default_marker();
}

mtmd_bitmap * mtmd_bitmap_init(uint32_t, uint32_t, const unsigned char *) {
    mtmd_stub_log_error(__func__);
    return nullptr;
}

mtmd_bitmap * mtmd_bitmap_init_from_audio(size_t n_samples, const float * data) {
    if (n_samples > UINT32_MAX / sizeof(float)) {
        return nullptr;
    }
    return new mtmd_bitmap(data, n_samples);
}

uint32_t mtmd_bitmap_get_nx(const mtmd_bitmap * bitmap) {
    return bitmap ? bitmap->nx : 0;
}

uint32_t mtmd_bitmap_get_ny(const mtmd_bitmap * bitmap) {
    return bitmap ? bitmap->ny : 0;
}

const unsigned char * mtmd_bitmap_get_data(const mtmd_bitmap * bitmap) {
    return bitmap ? bitmap->data.data() : nullptr;
}

size_t mtmd_bitmap_get_n_bytes(const mtmd_bitmap * bitmap) {
    return bitmap ? bitmap->data.size() : 0;
}

bool mtmd_bitmap_is_audio(const mtmd_bitmap * bitmap) {
    return bitmap ? bitmap->is_audio : false;
}

bool mtmd_bitmap_is_voice(const mtmd_bitmap * bitmap) {
    return bitmap ? bitmap->is_voice : false;
}

void mtmd_bitmap_free(mtmd_bitmap * bitmap) {
    delete bitmap;
}

const char * mtmd_bitmap_get_id(const mtmd_bitmap * bitmap) {
    return bitmap ? bitmap->id.c_str() : "";
}

void mtmd_bitmap_set_id(mtmd_bitmap * bitmap, const char * id) {
    if (bitmap) {
        bitmap->id = id ? id : "";
    }
}

void mtmd_bitmap_set_mergeable(mtmd_bitmap *, bool) {}
void mtmd_bitmap_set_is_video(mtmd_bitmap *, bool) {}

mtmd_bitmap * mtmd_bitmap_init_lazy(const mtmd_context *, const char *, void *, mtmd_bitmap_lazy_callback) {
    mtmd_stub_log_error(__func__);
    return nullptr;
}

mtmd_input_chunks * mtmd_input_chunks_init() {
    return new mtmd_input_chunks;
}

size_t mtmd_input_chunks_size(const mtmd_input_chunks * chunks) {
    return chunks ? chunks->entries.size() : 0;
}

const mtmd_input_chunk * mtmd_input_chunks_get(const mtmd_input_chunks * chunks, size_t idx) {
    return (chunks && idx < chunks->entries.size()) ? &chunks->entries[idx] : nullptr;
}

void mtmd_input_chunks_free(mtmd_input_chunks * chunks) {
    delete chunks;
}

enum mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk * chunk) {
    return chunk ? chunk->type : MTMD_INPUT_CHUNK_TYPE_TEXT;
}

const llama_token * mtmd_input_chunk_get_tokens_text(const mtmd_input_chunk * chunk, size_t * n_tokens_output) {
    if (chunk && chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        if (n_tokens_output) {
            *n_tokens_output = chunk->tokens_text.size();
        }
        return chunk->tokens_text.data();
    }
    if (n_tokens_output) {
        *n_tokens_output = 0;
    }
    return nullptr;
}

const mtmd_image_tokens * mtmd_input_chunk_get_tokens_image(const mtmd_input_chunk * chunk) {
    return chunk ? chunk->tokens_audio.get() : nullptr;
}

size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk * chunk) {
    if (!chunk) {
        return 0;
    }
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        return chunk->tokens_text.size();
    }
    return chunk->tokens_audio ? 1 : 0;
}

const char * mtmd_input_chunk_get_id(const mtmd_input_chunk * chunk) {
    return (chunk && chunk->tokens_audio) ? chunk->tokens_audio->id.c_str() : "";
}

llama_pos mtmd_input_chunk_get_n_pos(const mtmd_input_chunk * chunk) {
    return (llama_pos) mtmd_input_chunk_get_n_tokens(chunk);
}

mtmd_input_chunk * mtmd_input_chunk_copy(const mtmd_input_chunk * chunk) {
    if (!chunk) {
        return nullptr;
    }
    auto * c = new mtmd_input_chunk();
    c->type        = chunk->type;
    c->tokens_text = chunk->tokens_text;
    if (chunk->tokens_audio) {
        c->tokens_audio = std::make_unique<mtmd_image_tokens>(*chunk->tokens_audio);
    }
    return c;
}

void mtmd_input_chunk_free(mtmd_input_chunk * chunk) {
    delete chunk;
}

mtmd_input_chunk * mtmd_input_chunk_get_placeholder(const mtmd_input_chunk *) {
    return nullptr;
}

int32_t mtmd_input_chunk_save(const mtmd_input_chunk *, char *, size_t, size_t * expected_out_len) {
    if (expected_out_len) *expected_out_len = 0;
    return 1;
}

mtmd_input_chunk * mtmd_input_chunk_load(const char *, size_t) {
    return nullptr;
}

size_t mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * t) {
    return t ? 1 : 0;
}

const char * mtmd_image_tokens_get_id(const mtmd_image_tokens * t) {
    return t ? t->id.c_str() : "";
}

llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * t) {
    return t ? 1 : 0;
}

size_t mtmd_image_tokens_get_nx(const mtmd_image_tokens * t) {
    return t ? 1 : 0;
}

size_t mtmd_image_tokens_get_ny(const mtmd_image_tokens * t) {
    return t ? 1 : 0;
}

struct mtmd_decoder_pos mtmd_image_tokens_get_decoder_pos(const mtmd_image_tokens *, llama_pos pos_0, size_t i) {
    const uint32_t p = pos_0 + (llama_pos) i >= 0 ? (uint32_t) (pos_0 + (llama_pos) i) : 0u;
    return { p, p, p, p };
}

int32_t mtmd_tokenize(const mtmd_context * ctx, mtmd_input_chunks * output, const mtmd_input_text * text,
                      const mtmd_bitmap * const * bitmaps, size_t n_bitmaps) {
    if (!ctx || !output || !text || !ctx->ctx_a || !ctx->audio_preproc) {
        mtmd_stub_log_error(__func__);
        return 2;
    }
    output->entries.clear();
    const std::string txt = text->text ? std::string(text->text) : std::string();
    if (n_bitmaps != 1 || txt != ctx->media_marker || !bitmaps || !bitmaps[0] || !bitmaps[0]->is_audio) {
        mtmd_stub_log_error(__func__);
        return 1;
    }
    const mtmd_bitmap * bm = bitmaps[0];
    const size_t n_samples = bm->data.size() / sizeof(float);
    const float * samples  = reinterpret_cast<const float *>(bm->data.data());

    mtmd_image_f32 mel_entry;
    {
        auto * mctx = const_cast<mtmd_context *>(ctx);
        std::lock_guard<std::mutex> lock(mctx->compute_mutex);
        if (!ctx->audio_preproc->compute_mel(samples, n_samples, mel_entry)) {
            return 2;
        }
    }

    mtmd_input_chunk chunk;
    chunk.type = MTMD_INPUT_CHUNK_TYPE_AUDIO;
    chunk.tokens_audio = std::make_unique<mtmd_image_tokens>();
    chunk.tokens_audio->id = bm->id;
    chunk.tokens_audio->batch_f32.is_audio = true;
    chunk.tokens_audio->batch_f32.entries.push_back(std::move(mel_entry));
    output->entries.push_back(std::move(chunk));
    return 0;
}

int32_t mtmd_tokenize_from_parts(const mtmd_context *, mtmd_input_chunks *, const mtmd_input_part * const *, size_t, bool) {
    mtmd_stub_log_error(__func__);
    return 2;
}

int32_t mtmd_encode_chunk(mtmd_context * ctx, const mtmd_input_chunk * chunk) {
    if (!ctx || !chunk) {
        return 1;
    }
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        return 0;
    }
    if (chunk->type != MTMD_INPUT_CHUNK_TYPE_AUDIO || !ctx->ctx_a || !chunk->tokens_audio) {
        return 1;
    }
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    try {
        mtmd_subctx_encode_params params;
        params.n_threads = ctx->n_threads;
        params.imgs      = &chunk->tokens_audio->batch_f32;
        params.out_embd  = &ctx->out_embd;
        return ctx->ctx_a->encode(&params) ? 0 : 1;
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

int32_t mtmd_encode(mtmd_context * ctx, const mtmd_image_tokens * image_tokens) {
    if (!ctx || !image_tokens || !ctx->ctx_a) {
        return 1;
    }
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    try {
        mtmd_subctx_encode_params params;
        params.n_threads = ctx->n_threads;
        params.imgs      = &image_tokens->batch_f32;
        params.out_embd  = &ctx->out_embd;
        return ctx->ctx_a->encode(&params) ? 0 : 1;
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

float * mtmd_get_output_embd(mtmd_context * ctx) {
    return ctx ? ctx->out_embd.data() : nullptr;
}

mtmd_batch * mtmd_batch_init(mtmd_context *) { return nullptr; }
void mtmd_batch_free(mtmd_batch *) {}
int32_t mtmd_batch_add_chunk(mtmd_batch *, const mtmd_input_chunk *) { return 1; }
int32_t mtmd_batch_encode(mtmd_batch *) { return 1; }
float * mtmd_batch_get_output_embd(mtmd_batch *, const mtmd_input_chunk *) { return nullptr; }

void mtmd_log_set(ggml_log_callback log_callback, void * user_data) {
    g_log_callback  = log_callback ? log_callback : g_default_log_cb;
    g_log_user_data = user_data;
}

struct mtmd_caps mtmd_get_cap_from_file(const char * mmproj_fname) {
    mtmd_caps res {};
    try {
        mtmd_qwen3tts_loader loader(mmproj_fname, true);
        if (!loader.has_vision && loader.has_gen_audio) {
            std::string proj_gen;
            loader.get_string("clip.projector_type", proj_gen, false);
            if (proj_gen.empty()) {
                loader.get_string("clip.gen.audio.projector_type", proj_gen, false);
            }
            if (proj_gen == "qwen3tts_gen") {
                res.gen_audio = true;
                res.inp_audio = loader.has_audio;
            }
        }
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
    }
    return res;
}

bool mtmd_support_audio_stream(const mtmd_context *) {
    return false;
}

struct mtmd_audio_stream_params mtmd_audio_stream_params_default() {
    mtmd_audio_stream_params params {};
    params.step_ms  = 320;
    params.prompt   = nullptr;
    params.n_prompt = 0;
    return params;
}

mtmd_audio_stream * mtmd_audio_stream_init(mtmd_context *, struct mtmd_audio_stream_params) {
    mtmd_stub_log_error(__func__);
    return nullptr;
}

void mtmd_audio_stream_free(mtmd_audio_stream *) {}
int32_t mtmd_audio_stream_push(mtmd_audio_stream *, const float *, size_t) { return 1; }
int32_t mtmd_audio_stream_flush(mtmd_audio_stream *, bool) { return 1; }
int32_t mtmd_audio_stream_step(mtmd_audio_stream *, llama_token) { return -2; }
const float * mtmd_audio_stream_get_output_embd(const mtmd_audio_stream *) { return nullptr; }
bool mtmd_audio_stream_is_seq_start(const mtmd_audio_stream *) { return false; }
int64_t mtmd_audio_stream_get_n_rows(const mtmd_audio_stream *) { return 0; }
int64_t mtmd_audio_stream_get_n_rows_flushed(const mtmd_audio_stream *) { return 0; }

struct mtmd_gen_audio_info mtmd_gen_audio_get_info(const mtmd_context * ctx) {
    mtmd_gen_audio_info info {};
    info.model_variant = "";
    if (!ctx || !ctx->ctx_gen_a) {
        info.type = MTMD_GEN_AUDIO_TYPE_NONE;
        return info;
    }
    info.type          = MTMD_GEN_AUDIO_TYPE_QWEN3TTS;
    info.sample_rate   = 24000;
    info.n_voices      = (int32_t) ctx->voice_names.size();
    info.model_variant = ctx->ctx_gen_a->model.hparams.gen_model_variant.c_str();
    return info;
}

size_t mtmd_gen_audio_read_tensor(mtmd_context *, const char *, float *, size_t) {
    return 0;
}

const char * mtmd_gen_audio_voice_name(const mtmd_context * ctx, int32_t idx) {
    if (!ctx || idx < 0 || (size_t) idx >= ctx->voice_names.size()) {
        return nullptr;
    }
    return ctx->voice_names[(size_t) idx].c_str();
}

mtmd_bitmap * mtmd_gen_audio_get_voice(const mtmd_context * ctx, const char * name) {
    if (!ctx || !name || !name[0]) {
        return nullptr;
    }
    std::string lower(name);
    for (char & c : lower) {
        c = (char) std::tolower((unsigned char) c);
    }
    for (const auto & v : ctx->voice_names) {
        if (v == lower) {
            return new mtmd_bitmap(v);
        }
    }
    return nullptr;
}

struct mtmd_gen_state_params mtmd_gen_state_params_default() {
    mtmd_gen_state_params params {};
    params.seed = UINT32_MAX;
    return params;
}

mtmd_gen_state * mtmd_gen_state_init(mtmd_context * ctx, struct mtmd_gen_state_params params) {
    if (!ctx || !ctx->ctx_gen_a) {
        LOG_ERR("%s: model does not support audio generation\n", __func__);
        return nullptr;
    }
    auto st = std::make_unique<mtmd_gen_state>();
    st->sctx      = ctx->ctx_gen_a.get();
    st->gen_state = ctx->ctx_gen_a->gen_state_init(params.seed);
    if (!st->gen_state) {
        return nullptr;
    }
    return st.release();
}

void mtmd_gen_state_free(mtmd_gen_state * state) {
    delete state;
}

void mtmd_gen_state_reset(mtmd_gen_state * state) {
    if (state && state->gen_state) {
        state->gen_state->reset();
    }
}

int32_t mtmd_gen_state_set_ref(mtmd_context *, mtmd_gen_state *) {
    return 1;
}

struct mtmd_gen_inp mtmd_gen_inp_default(const mtmd_context * ctx) {
    mtmd_gen_inp inp {};
    inp.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
    if (ctx && ctx->ctx_gen_a) {
        inp.top_k = 50;
        inp.top_p = 1.0f;
        inp.temp  = 0.9f;
    }
    return inp;
}

int32_t mtmd_gen_audio_process(mtmd_context * ctx, mtmd_gen_state * state, const struct mtmd_gen_inp * inp, struct mtmd_gen_out * out) {
    if (!ctx || !ctx->ctx_gen_a || !state || state->sctx != ctx->ctx_gen_a.get() || !inp || !out) {
        return 1;
    }
    std::lock_guard<std::mutex> lock(ctx->compute_mutex);
    try {
        mtmd_subcontext * sctx = ctx->ctx_gen_a.get();
        *out = {};

        if (inp->type == MTMD_GEN_PROCESS_TYPE_GEN_CODE) {
            const size_t n_embd = (size_t) sctx->model.n_mmproj_embd();
            mtmd_image_f32 hidden_state;
            hidden_state.set_size({(int) n_embd, 1}, false, true);
            hidden_state.cpy_buf(inp->embd ? std::vector<float>(inp->embd, inp->embd + n_embd) : std::vector<float>(n_embd, 0.0f));

            mtmd_image_f32_batch batch;
            batch.is_audio = true;
            batch.entries.push_back(std::move(hidden_state));

            std::vector<float>   out_embd(n_embd);
            std::vector<int32_t> out_codes;

            mtmd_subctx_encode_params params;
            params.imgs        = &batch;
            params.n_threads   = ctx->n_threads;
            params.gen_process = MTMD_SUBCTX_GEN_PROCESS_GEN_CODE;
            params.out_embd    = &out_embd;
            params.out_codes   = &out_codes;
            params.code0       = inp->code0;
            params.top_k       = inp->top_k;
            params.top_p       = inp->top_p;
            params.temp        = inp->temp;
            params.gen_state   = state->gen_state;

            if (!sctx->encode(&params)) {
                LOG_ERR("%s: encode failed (gen_code)\n", __func__);
                return 1;
            }

            state->out_embd  = std::move(out_embd);
            state->out_codes = std::move(out_codes);

            out->embd    = state->out_embd.data();
            out->codes   = state->out_codes.data();
            out->n_codes = state->out_codes.size();
            return 0;
        }

        if (!inp->codes || inp->n_codes == 0) {
            LOG_ERR("%s: codes required for gen_wav\n", __func__);
            return 1;
        }
        std::vector<int32_t> in_codes(inp->codes, inp->codes + inp->n_codes);

        mtmd_image_f32 dummy;
        dummy.set_size({1, 1}, false, true);
        dummy.cpy_buf(std::vector<float>(1, 0.0f));

        mtmd_image_f32_batch batch;
        batch.is_audio = true;
        batch.entries.push_back(std::move(dummy));

        mtmd_subctx_encode_params params;
        params.imgs        = &batch;
        params.n_threads   = ctx->n_threads;
        params.gen_process = MTMD_SUBCTX_GEN_PROCESS_GEN_WAV;
        params.codes       = &in_codes;
        params.out_audio   = &state->out_audio;
        params.gen_state   = state->gen_state;

        if (!sctx->encode(&params)) {
            LOG_ERR("%s: encode failed (code2wav)\n", __func__);
            return 1;
        }

        out->audio     = state->out_audio.data();
        out->n_samples = state->out_audio.size();
        return 0;
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return 1;
    }
}

mtmd_input_chunks * mtmd_test_create_input_chunks() {
    return mtmd_input_chunks_init();
}

struct mtmd_memory_usage mtmd_get_memory_usage(const char * mmproj_fname, struct mtmd_context_params ctx_params) {
    auto saved_cb = g_log_callback;
    auto saved_ud = g_log_user_data;
    ctx_params.progress_callback = nullptr;
    try {
        mtmd_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
        mtmd::context_ptr ctx(new mtmd_context(mmproj_fname, nullptr, ctx_params, true));
        mtmd_log_set(saved_cb, saved_ud);

        std::map<ggml_backend_dev_t, size_t> total_mem = ctx->backends->get_mem_compute();
        for (const auto * m : { ctx->models.a.get(), ctx->models.gen_a.get() }) {
            if (m) {
                for (const auto & kv : m->mem_weights) {
                    total_mem[kv.first] += kv.second;
                }
            }
        }
        mtmd_memory_usage res;
        res.backend_mem_usage = std::move(total_mem);
        res.image_max_tokens  = -1;
        res.use_non_causal    = false;
        return res;
    } catch (const std::exception & e) {
        mtmd_log_set(saved_cb, saved_ud);
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return {};
    }
}

//
// mtmd-helper.h public API
//

static bool is_audio_file_buf(const unsigned char * buf, size_t len) {
    if (len < 12) {
        return false;
    }
    const bool is_wav  = std::memcmp(buf, "RIFF", 4) == 0 && std::memcmp(buf + 8, "WAVE", 4) == 0;
    const bool is_mp3  = std::memcmp(buf, "ID3", 3) == 0 || (buf[0] == 0xFF && (buf[1] & 0xE0) == 0xE0);
    const bool is_flac = std::memcmp(buf, "fLaC", 4) == 0;
    return is_wav || is_mp3 || is_flac;
}

static bool decode_audio_from_buf(const unsigned char * buf_in, size_t len, int target_sample_rate, std::vector<float> & pcmf32_mono) {
    ma_decoder_config decoder_config = ma_decoder_config_init(ma_format_f32, 1, (ma_uint32) target_sample_rate);
    ma_decoder decoder;
    if (ma_decoder_init_memory(buf_in, len, &decoder_config, &decoder) != MA_SUCCESS) {
        return false;
    }
    ma_uint64 frame_count = 0;
    ma_uint64 frames_read = 0;
    if (ma_decoder_get_length_in_pcm_frames(&decoder, &frame_count) != MA_SUCCESS || frame_count == 0) {
        ma_decoder_uninit(&decoder);
        return false;
    }
    pcmf32_mono.resize((size_t) frame_count);
    const ma_result res = ma_decoder_read_pcm_frames(&decoder, pcmf32_mono.data(), frame_count, &frames_read);
    ma_decoder_uninit(&decoder);
    if (res != MA_SUCCESS) {
        return false;
    }
    return true;
}

struct mtmd_helper_video_init_params mtmd_helper_video_init_params_default() {
    return {
        /* fps_target             */ 4.0f,
        /* ffmpeg_bin_dir         */ nullptr,
        /* timestamp_interval_ms  */ 5000,
        /* no_soundtrack          */ false,
    };
}

struct mtmd_helper_init_opt mtmd_helper_init_opt_default() {
    return { mtmd_helper_video_init_params_default() };
}

void mtmd_helper_log_set(ggml_log_callback log_callback, void * user_data) {
    mtmd_log_set(log_callback, user_data);
}

struct mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_buf(
        const mtmd_context * ctx,
        const unsigned char * buf,
        size_t len,
        bool /*placeholder*/,
        struct mtmd_helper_init_opt) {
    if (!ctx || !buf || len == 0 || !mtmd_support_audio(ctx) || !is_audio_file_buf(buf, len)) {
        mtmd_stub_log_error(__func__);
        return { nullptr, nullptr };
    }
    const int sr = mtmd_get_audio_sample_rate(ctx);
    if (sr <= 0) {
        return { nullptr, nullptr };
    }
    std::vector<float> pcm;
    if (!decode_audio_from_buf(buf, len, sr, pcm)) {
        LOG_ERR("%s: failed to decode audio buffer\n", __func__);
        return { nullptr, nullptr };
    }
    return { mtmd_bitmap_init_from_audio(pcm.size(), pcm.data()), nullptr };
}

struct mtmd_helper_bitmap_wrapper mtmd_helper_bitmap_init_from_file(
        const mtmd_context * ctx,
        const char * fname,
        bool placeholder,
        struct mtmd_helper_init_opt opt) {
    if (!fname) {
        return { nullptr, nullptr };
    }
    std::ifstream fin(fname, std::ios::binary | std::ios::ate);
    if (!fin) {
        LOG_ERR("%s: unable to open file %s\n", __func__, fname);
        return { nullptr, nullptr };
    }
    const std::streamsize sz = fin.tellg();
    if (sz <= 0) {
        return { nullptr, nullptr };
    }
    std::vector<unsigned char> buf((size_t) sz);
    fin.seekg(0, std::ios::beg);
    if (!fin.read(reinterpret_cast<char *>(buf.data()), sz)) {
        return { nullptr, nullptr };
    }
    return mtmd_helper_bitmap_init_from_buf(ctx, buf.data(), buf.size(), placeholder, opt);
}

mtmd_helper_resampler * mtmd_helper_resampler_init(int32_t, int32_t) {
    mtmd_stub_log_error(__func__);
    return nullptr;
}
void mtmd_helper_resampler_free(mtmd_helper_resampler *) {}
int64_t mtmd_helper_resampler_process(mtmd_helper_resampler *, const float *, size_t, const float **) {
    return -1;
}

size_t mtmd_helper_get_n_tokens(const mtmd_input_chunks * chunks) {
    size_t n = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        n += mtmd_input_chunk_get_n_tokens(mtmd_input_chunks_get(chunks, i));
    }
    return n;
}

llama_pos mtmd_helper_get_n_pos(const mtmd_input_chunks * chunks) {
    llama_pos n = 0;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        n += mtmd_input_chunk_get_n_pos(mtmd_input_chunks_get(chunks, i));
    }
    return n;
}

void mtmd_helper_image_get_decoder_pos(const mtmd_image_tokens * chunks, llama_pos pos_0, struct mtmd_decoder_pos * out_pos) {
    const size_t n = mtmd_image_tokens_get_n_tokens(chunks);
    for (size_t i = 0; i < n; i++) {
        out_pos[i] = mtmd_image_tokens_get_decoder_pos(chunks, pos_0, i);
    }
}

int32_t mtmd_helper_eval_chunks(mtmd_context *, struct llama_context *, const mtmd_input_chunks *, llama_pos,
        llama_seq_id, int32_t, bool, llama_pos *) {
    mtmd_stub_log_error(__func__);
    return 1;
}

int32_t mtmd_helper_eval_chunk_single(mtmd_context *, struct llama_context *, const mtmd_input_chunk *, llama_pos,
        llama_seq_id, int32_t, bool, llama_pos *) {
    mtmd_stub_log_error(__func__);
    return 1;
}

int32_t mtmd_helper_decode_image_chunk(mtmd_context *, struct llama_context *, const mtmd_input_chunk *, float *,
        llama_pos, llama_seq_id, int32_t, llama_pos *, mtmd_helper_post_decode_callback, void *) {
    return -1;
}

mtmd_helper_video * mtmd_helper_video_init(const struct mtmd_context *, const char *, struct mtmd_helper_video_init_params) {
    mtmd_stub_log_error(__func__);
    return nullptr;
}

mtmd_helper_video * mtmd_helper_video_init_from_buf(const struct mtmd_context *, const unsigned char *, size_t,
        struct mtmd_helper_video_init_params) {
    mtmd_stub_log_error(__func__);
    return nullptr;
}

void mtmd_helper_video_free(mtmd_helper_video *) {}

struct mtmd_helper_video_info mtmd_helper_video_get_info(const mtmd_helper_video *) {
    mtmd_helper_video_info info {};
    info.n_frames = -1;
    return info;
}

int32_t mtmd_helper_video_read_next(mtmd_helper_video *, mtmd_bitmap ** out_bitmap, char ** out_text) {
    if (out_bitmap) *out_bitmap = nullptr;
    if (out_text)   *out_text   = nullptr;
    return -2;
}

struct mtmd_helper_cap mtmd_helper_get_capabilities(const struct llama_model * model, const struct mtmd_context * mctx) {
    mtmd_helper_cap cap = {};
    cap.chat     = true;
    cap.text_gen = true;
    if (!mctx) {
        return cap;
    }
    cap.vision = mtmd_support_vision(mctx);
    cap.audio  = mtmd_support_audio(mctx);
    cap.audio_stream = mtmd_support_audio_stream(mctx);
    const auto gen_type = mtmd_gen_audio_get_info(mctx).type;
    cap.audio_gen         = gen_type != MTMD_GEN_AUDIO_TYPE_NONE;
    cap.audio_gen_spk_ref = gen_type == MTMD_GEN_AUDIO_TYPE_QWEN3TTS;
    cap.text_gen = !cap.audio_gen || (model && llama_model_chat_template(model, nullptr) != nullptr);
    cap.chat     = cap.text_gen && (!model || llama_model_n_tokens_delay(model) == 0);
    return cap;
}

bool mtmd_helper_model_can_chat(const struct llama_context * lctx, const struct mtmd_context * mctx) {
    return mtmd_helper_get_capabilities(lctx ? llama_get_model(lctx) : nullptr, mctx).chat;
}

bool mtmd_helper_support_video(const struct mtmd_context *) {
    return false;
}

//
// Qwen3-TTS end-to-end audio generation helper pipeline
//

namespace {

struct decode_embd_batch {
    int n_pos_per_embd;
    int n_mmproj_embd;
    int32_t n_tokens;
    const float * embd;
    std::vector<llama_pos> pos;
    std::vector<int8_t>    logits;
    llama_seq_id seq_id = 0;
    llama_batch_ext_ptr batch;

    decode_embd_batch(const float * embd, int32_t n_tokens, int n_pos_per_embd, int n_mmproj_embd)
        : n_pos_per_embd(n_pos_per_embd), n_mmproj_embd(n_mmproj_embd), n_tokens(n_tokens), embd(embd) {
        GGML_ASSERT(n_tokens > 0 && n_pos_per_embd > 0 && n_mmproj_embd > 0);
        pos.resize((size_t) n_tokens * (size_t) n_pos_per_embd);
        logits.resize((size_t) n_tokens);
    }

    void set_position_normal(llama_pos pos_0, llama_seq_id sid) {
        seq_id = sid;
        for (int i = 0; i < n_tokens; i++) {
            pos[(size_t) i] = pos_0 + i;
        }
    }

    void set_position_mrope_1d(llama_pos pos_0, llama_seq_id sid) {
        GGML_ASSERT(n_pos_per_embd == 4);
        seq_id = sid;
        const size_t n = (size_t) n_tokens;
        for (int i = 0; i < n_tokens; i++) {
            const size_t idx = (size_t) i;
            pos[idx        ] = pos_0 + i;
            pos[idx + n    ] = pos_0 + i;
            pos[idx + n * 2] = pos_0 + i;
            pos[idx + n * 3] = pos_0 + i;
        }
    }

    llama_batch_ext * render(llama_context * lctx, int offset, int n) {
        GGML_ASSERT(offset >= 0 && n > 0 && offset + n <= n_tokens);
        if (!batch) {
            batch.reset(llama_batch_ext_init(lctx));
        }
        llama_batch_ext_clear(batch.get());
        for (int i = offset; i < offset + n; i++) {
            const llama_embd e = { embd + (size_t) i * (size_t) n_mmproj_embd, 1, (size_t) n_mmproj_embd };
            const int32_t idx = llama_batch_ext_add_embd(batch.get(), seq_id, e);
            GGML_ASSERT(idx >= 0);

            llama_pos p[GGML_MROPE_SECTIONS] = { 0, 0, 0, 0 };
            for (int j = 0; j < n_pos_per_embd; j++) {
                p[j] = pos[(size_t) j * (size_t) n_tokens + (size_t) i];
            }
            llama_batch_ext_set_pos(batch.get(), idx, p);
            if (logits[(size_t) i]) {
                llama_batch_ext_set_output_logits(batch.get(), idx, true);
            }
        }
        return batch.get();
    }
};

static std::string tts_resolve_lang(const std::string & lang) {
    std::string low(lang);
    for (char & c : low) {
        c = (char) std::tolower((unsigned char) c);
    }
    static const std::unordered_map<std::string, std::string> codes = {
        { "zh", "chinese"    },
        { "en", "english"    },
        { "de", "german"     },
        { "it", "italian"    },
        { "pt", "portuguese" },
        { "es", "spanish"    },
        { "ja", "japanese"   },
        { "ko", "korean"     },
        { "fr", "french"     },
        { "ru", "russian"    },
    };
    const auto it = codes.find(low);
    return it != codes.end() ? it->second : low;
}

static llama_token find_special_token(const llama_vocab * vocab, const std::string & piece) {
    const int32_t n = llama_vocab_n_tokens(vocab);
    for (llama_token t = 0; t < n; t++) {
        if (piece == llama_vocab_get_text(vocab, t)) {
            return t;
        }
    }
    return LLAMA_TOKEN_NULL;
}

static void put_bytes(std::vector<char> & buf, const void * p, size_t n) {
    const char * c = static_cast<const char *>(p);
    buf.insert(buf.end(), c, c + n);
}

static void write_wav16_header(std::vector<char> & buf, uint32_t data_sz, int32_t rate) {
    const uint32_t riff_sz   = data_sz == UINT32_MAX ? UINT32_MAX : 36 + data_sz;
    const uint32_t fmt_sz    = 16, byte_rate = (uint32_t) rate * 2;
    const uint16_t fmt = 1, ch = 1, align = 2, bits = 16;
    const uint32_t rate32    = (uint32_t) rate;
    put_bytes(buf, "RIFF", 4); put_bytes(buf, &riff_sz, 4); put_bytes(buf, "WAVE", 4);
    put_bytes(buf, "fmt ", 4); put_bytes(buf, &fmt_sz, 4);
    put_bytes(buf, &fmt, 2); put_bytes(buf, &ch, 2); put_bytes(buf, &rate32, 4);
    put_bytes(buf, &byte_rate, 4); put_bytes(buf, &align, 2); put_bytes(buf, &bits, 2);
    put_bytes(buf, "data", 4); put_bytes(buf, &data_sz, 4);
}

static void append_wav16_pcm(std::vector<char> & buf, const float * pcm, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const int16_t s = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, pcm[i])) * 32767.0f);
        put_bytes(buf, &s, 2);
    }
}

static bool write_wav16(std::vector<char> & buf, const std::vector<float> & pcm, int32_t rate) {
    if (pcm.size() > ((size_t) UINT32_MAX - 36) / 2) {
        return false;
    }
    write_wav16_header(buf, (uint32_t) (pcm.size() * 2), rate);
    append_wav16_pcm(buf, pcm.data(), pcm.size());
    return true;
}

class qwen3tts_gen_audio_pipeline {
public:
    qwen3tts_gen_audio_pipeline(llama_context * lctx, mtmd_context * mctx)
        : lctx(lctx), mctx(mctx), model(llama_get_model(lctx)), vocab(llama_model_get_vocab(model)),
          n_embd(llama_model_n_embd(model)), info(mtmd_gen_audio_get_info(mctx)) {}

    void reset() {
        seq_id = 0;
        pos = 0;
        codes_buf.clear();
        gen_state.reset();
        audio_pcm.clear();
        overlay.clear();
        h_state_buf.clear();
        out_buf.clear();
        prompt_embd_buf.clear();
        prompt_batch.reset();
        n_prompt = 0;
        prompt_pos = 0;
        stream = false;
        pcm_sent = 0;
        wav_header_sent = false;
    }

    int32_t set_input(const mtmd_helper_gen_audio_inp * inp) {
        reset();
        seq_id = inp->seq_id;
        if (!ensure_cache()) {
            return 1;
        }

        const int n_e = n_embd;
        auto row = [&](llama_token t) {
            return std::vector<float>(tok_embd.begin() + (size_t) t * (size_t) n_e,
                                      tok_embd.begin() + (size_t) (t + 1) * (size_t) n_e);
        };
        auto sum_row = [&](llama_token a, llama_token b) {
            std::vector<float> va = row(a), vb = row(b);
            for (int i = 0; i < n_e; i++) va[(size_t) i] += vb[(size_t) i];
            return va;
        };
        auto sum_vec = [&](llama_token a, const std::vector<float> & vb) {
            std::vector<float> va = row(a);
            for (int i = 0; i < n_e; i++) va[(size_t) i] += vb[(size_t) i];
            return va;
        };

        std::string speaker_name;
        std::vector<float> speaker_embd;
        if (inp->speaker_ref) {
            if (inp->speaker_ref->is_voice) {
                speaker_name = inp->speaker_ref->id;
                const llama_token spk_tok = find_special_token(vocab, "<|spk_" + speaker_name + "|>");
                if (spk_tok == LLAMA_TOKEN_NULL) {
                    LOG_ERR("mtmd_helper_gen_audio: unknown built-in speaker '%s'\n", speaker_name.c_str());
                    return 1;
                }
                speaker_embd = row(spk_tok);
            } else {
                if (!encode_speaker(inp->speaker_ref, speaker_embd)) {
                    return 1;
                }
            }
        }

        std::string lang = tts_resolve_lang((inp->lang && inp->lang[0]) ? inp->lang : "auto");
        if ((lang == "chinese" || lang == "auto") && !speaker_name.empty()) {
            if (speaker_name == "eric") {
                lang = "sichuan_dialect";
            } else if (speaker_name == "dylan") {
                lang = "beijing_dialect";
            }
        }
        llama_token c_lang = LLAMA_TOKEN_NULL;
        if (lang != "auto") {
            c_lang = find_special_token(vocab, "<|codec_language_" + lang + "|>");
            if (c_lang == LLAMA_TOKEN_NULL) {
                LOG_ERR("mtmd_helper_gen_audio: unknown language '%s'\n", lang.c_str());
                return 1;
            }
        }

        const std::string full = "<|im_start|>assistant\n" + std::string(inp->prompt, inp->prompt_len) +
                                 "<|im_end|>\n<|im_start|>assistant\n";
        std::vector<llama_token> ids(full.size() + 16);
        int n_ids = llama_tokenize(vocab, full.c_str(), (int32_t) full.size(), ids.data(), (int32_t) ids.size(), false, true);
        if (n_ids < 0) {
            ids.resize((size_t) (-n_ids));
            n_ids = llama_tokenize(vocab, full.c_str(), (int32_t) full.size(), ids.data(), (int32_t) ids.size(), false, true);
        }
        if (n_ids < 8) {
            LOG_ERR("mtmd_helper_gen_audio: tokenization failed\n");
            return 1;
        }
        ids.resize((size_t) n_ids);

        std::vector<std::vector<float>> prompt;
        // llama.cpp-xdna: VoiceDesign and 1.7B-CustomVoice prepend <|im_start|>user\n{instruct}<|im_end|>\n text embeddings
        if (inp->instruct && inp->instruct_len > 0) {
            if (n_embd <= 1024) {
                LOG_WRN("mtmd_helper_gen_audio: 0.6B Qwen3-TTS models do not support --tts-instruct; ignoring instruction\n");
            } else {
                const std::string inst_str = "<|im_start|>user\n" + std::string(inp->instruct, inp->instruct_len) + "<|im_end|>\n";
                std::vector<llama_token> inst_ids(inst_str.size() + 16);
                int n_inst = llama_tokenize(vocab, inst_str.c_str(), (int32_t) inst_str.size(), inst_ids.data(), (int32_t) inst_ids.size(), false, true);
                if (n_inst < 0) {
                    inst_ids.resize((size_t) (-n_inst));
                    n_inst = llama_tokenize(vocab, inst_str.c_str(), (int32_t) inst_str.size(), inst_ids.data(), (int32_t) inst_ids.size(), false, true);
                }
                if (n_inst < 4) {
                    LOG_ERR("mtmd_helper_gen_audio: instruct tokenization failed\n");
                    return 1;
                }
                for (int i = 0; i < n_inst; i++) {
                    prompt.push_back(row(inst_ids[(size_t) i]));
                }
            }
        }
        for (int i = 0; i < 3; i++) prompt.push_back(row(ids[(size_t) i]));
        if (c_lang == LLAMA_TOKEN_NULL) {
            prompt.push_back(sum_row(tts_pad, c_nothink));
            prompt.push_back(sum_row(tts_pad, c_think_b));
            prompt.push_back(sum_row(tts_pad, c_think_e));
        } else {
            prompt.push_back(sum_row(tts_pad, c_think));
            prompt.push_back(sum_row(tts_pad, c_think_b));
            prompt.push_back(sum_row(tts_pad, c_lang));
            prompt.push_back(sum_row(tts_pad, c_think_e));
        }
        if (!speaker_embd.empty()) prompt.push_back(sum_vec(tts_pad, speaker_embd));
        prompt.push_back(sum_row(tts_bos, codec_pad));
        for (int i = 3; i < n_ids - 5; i++) prompt.push_back(sum_row(ids[(size_t) i], codec_pad));
        prompt.push_back(sum_row(tts_eos, codec_pad));
        prompt.push_back(sum_row(tts_pad, codec_bos));

        n_prompt = (int) prompt.size();
        mrope = llama_model_rope_type(model) == LLAMA_ROPE_TYPE_MROPE ||
                llama_model_rope_type(model) == LLAMA_ROPE_TYPE_IMROPE;
        const int n_pos_per_embd = mrope ? 4 : 1;

        prompt_embd_buf.resize((size_t) n_prompt * (size_t) n_e);
        for (int i = 0; i < n_prompt; i++) {
            std::memcpy(prompt_embd_buf.data() + (size_t) i * (size_t) n_e, prompt[(size_t) i].data(), (size_t) n_e * sizeof(float));
        }

        prompt_batch = std::make_unique<decode_embd_batch>(prompt_embd_buf.data(), n_prompt, n_pos_per_embd, n_e);
        if (mrope) prompt_batch->set_position_mrope_1d(0, seq_id);
        else       prompt_batch->set_position_normal  (0, seq_id);
        prompt_pos = 0;

        pos = 0;
        const mtmd_gen_inp def = mtmd_gen_inp_default(mctx);
        top_k    = inp->top_k > 0 ? inp->top_k : def.top_k;
        top_p    = inp->top_p > 0 ? inp->top_p : def.top_p;
        out_type = inp->out_type;
        stream   = inp->stream;

        mtmd_gen_state_params gs_params = mtmd_gen_state_params_default();
        gs_params.seed = inp->seed;
        gen_state.reset(mtmd_gen_state_init(mctx, gs_params));
        if (!gen_state) {
            LOG_ERR("mtmd_helper_gen_audio: failed to create gen state\n");
            return 1;
        }

        overlay = row(tts_pad);
        return 0;
    }

    int32_t step_prompt(int32_t n_batch) {
        GGML_ASSERT(n_batch > 0);
        if (prompt_pos >= n_prompt) {
            return 0;
        }
        const int32_t n_tokens_batch = std::min(n_batch, n_prompt - prompt_pos);
        if (prompt_pos + n_tokens_batch == n_prompt) {
            prompt_batch->logits[(size_t) (prompt_pos + n_tokens_batch - 1)] = 1;
        }
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, prompt_batch->render(lctx, prompt_pos, n_tokens_batch)) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: prompt decode failed\n");
            return -1;
        }
        pos        += n_tokens_batch;
        prompt_pos += n_tokens_batch;
        if (prompt_pos >= n_prompt) {
            prompt_batch.reset();
            prompt_embd_buf.clear();
            return 0;
        }
        return n_prompt - prompt_pos;
    }

    int32_t step_gen(llama_token sampled, const float * h_state_in, const float ** h_state_out, bool * out_stop) {
        if (sampled == LLAMA_TOKEN_NULL) {
            LOG_ERR("mtmd_helper_gen_audio: qwen3tts requires a token sampled from the backbone\n");
            return 1;
        }
        if (sampled == codec_eos || llama_vocab_is_eog(vocab, sampled)) {
            *out_stop    = true;
            *h_state_out = nullptr;
            return 0;
        }

        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type  = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        inp.code0 = sampled - codec_0;
        inp.embd  = const_cast<float *>(h_state_in);
        inp.top_k = top_k;
        inp.top_p = top_p;
        mtmd_gen_out out {};
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: gen_code process failed\n");
            return 1;
        }

        codes_buf.insert(codes_buf.end(), out.codes, out.codes + out.n_codes);
        if (out.n_codes > 0 && codes_buf.size() / out.n_codes >= window_frames) {
            if (!flush_gen_wav()) {
                return 1;
            }
        }

        std::vector<float> fb(out.embd, out.embd + n_embd);
        for (int i = 0; i < n_embd; i++) fb[(size_t) i] += overlay[(size_t) i];

        const int n_pos_per_embd = mrope ? 4 : 1;
        decode_embd_batch batch_embd(fb.data(), 1, n_pos_per_embd, n_embd);
        if (mrope) batch_embd.set_position_mrope_1d(pos, seq_id);
        else       batch_embd.set_position_normal  (pos, seq_id);
        batch_embd.logits[0] = 1;
        pos++;

        if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch_embd.render(lctx, 0, 1)) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: decode failed\n");
            return 1;
        }

        const float * he = llama_get_embeddings_ith(lctx, -1);
        h_state_buf.assign(he, he + n_embd);
        *h_state_out = h_state_buf.data();
        return 0;
    }

    int32_t get_output(int32_t * out_sample_rate, const char ** out_data, size_t * out_data_len, int64_t * out_n_samples) {
        *out_sample_rate = info.sample_rate;
        if (!stream) {
            if (!flush_gen_wav()) {
                return 1;
            }
            if (out_n_samples) {
                *out_n_samples = (int64_t) audio_pcm.size();
            }
            if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
                *out_data     = reinterpret_cast<const char *>(audio_pcm.data());
                *out_data_len = audio_pcm.size() * sizeof(float);
                return 0;
            }
            out_buf.clear();
            if (!write_wav16(out_buf, audio_pcm, info.sample_rate)) {
                LOG_ERR("mtmd_helper_gen_audio: output too large for WAV\n");
                return 1;
            }
            *out_data     = out_buf.data();
            *out_data_len = out_buf.size();
            return 0;
        }

        const size_t n_new = audio_pcm.size() - pcm_sent;
        if (out_n_samples) {
            *out_n_samples = (int64_t) n_new;
        }
        if (out_type == MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM) {
            *out_data     = reinterpret_cast<const char *>(audio_pcm.data() + pcm_sent);
            *out_data_len = n_new * sizeof(float);
            pcm_sent = audio_pcm.size();
            return 0;
        }
        out_buf.clear();
        if (!wav_header_sent) {
            write_wav16_header(out_buf, UINT32_MAX, info.sample_rate);
            wav_header_sent = true;
        }
        append_wav16_pcm(out_buf, audio_pcm.data() + pcm_sent, n_new);
        pcm_sent = audio_pcm.size();
        *out_data     = out_buf.data();
        *out_data_len = out_buf.size();
        return 0;
    }

    int32_t flush() {
        return flush_gen_wav() ? 0 : 1;
    }

private:
    bool encode_speaker(mtmd_bitmap * bitmap, std::vector<float> & out) {
        if (!mtmd_support_audio(mctx)) {
            LOG_ERR("mtmd_helper_gen_audio: mmproj has no speaker encoder\n");
            return false;
        }
        const std::string marker = mtmd_get_marker(mctx);
        mtmd_input_text text { marker.c_str(), marker.size(), false, true };
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        const mtmd_bitmap * bptr = bitmap;
        bool ok = mtmd_tokenize(mctx, chunks, &text, &bptr, 1) == 0;
        if (ok) {
            ok = false;
            for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
                const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
                if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_AUDIO) {
                    continue;
                }
                if (mtmd_encode_chunk(mctx, chunk) != 0) {
                    LOG_ERR("mtmd_helper_gen_audio: speaker encode failed\n");
                    break;
                }
                const float * embd = mtmd_get_output_embd(mctx);
                const size_t  n    = (size_t) llama_model_n_embd_inp(model) * mtmd_input_chunk_get_n_tokens(chunk);
                out.assign(embd, embd + n);
                ok = true;
                break;
            }
        }
        mtmd_input_chunks_free(chunks);
        return ok;
    }

    bool ensure_cache() {
        if (specials_ok) {
            return true;
        }
        codec_0   = find_special_token(vocab, "<|codec_0|>");
        codec_bos = find_special_token(vocab, "<|codec_bos|>");
        codec_eos = find_special_token(vocab, "<|codec_eos_token|>");
        codec_pad = find_special_token(vocab, "<|codec_pad|>");
        c_think   = find_special_token(vocab, "<|codec_think|>");
        c_nothink = find_special_token(vocab, "<|codec_nothink|>");
        c_think_b = find_special_token(vocab, "<|codec_think_bos|>");
        c_think_e = find_special_token(vocab, "<|codec_think_eos|>");
        tts_pad   = find_special_token(vocab, "<tts_pad>");
        tts_bos   = find_special_token(vocab, "<tts_text_bos>");
        tts_eos   = find_special_token(vocab, "<tts_text_eod>");
        for (llama_token t : { codec_0, codec_bos, codec_eos, codec_pad,
                               c_think, c_nothink, c_think_b, c_think_e,
                               tts_pad, tts_bos, tts_eos }) {
            if (t == LLAMA_TOKEN_NULL) {
                LOG_ERR("mtmd_helper_gen_audio: missing required special token in vocab\n");
                return false;
            }
        }
        const uint32_t n_tok_embd = llama_model_get_tok_embd(model, nullptr);
        if (n_tok_embd == 0) {
            LOG_ERR("mtmd_helper_gen_audio: model has no token embeddings\n");
            return false;
        }
        tok_embd.resize(n_tok_embd);
        if (llama_model_get_tok_embd(model, tok_embd.data()) != n_tok_embd) {
            LOG_ERR("mtmd_helper_gen_audio: token embedding copy failed\n");
            return false;
        }
        specials_ok = true;
        return true;
    }

    bool flush_gen_wav() {
        if (codes_buf.empty()) {
            return true;
        }
        mtmd_gen_inp inp = mtmd_gen_inp_default(mctx);
        inp.type    = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        inp.codes   = codes_buf.data();
        inp.n_codes = codes_buf.size();
        mtmd_gen_out out {};
        if (mtmd_gen_audio_process(mctx, gen_state.get(), &inp, &out) != 0) {
            LOG_ERR("mtmd_helper_gen_audio: gen_wav process failed\n");
            return false;
        }
        audio_pcm.insert(audio_pcm.end(), out.audio, out.audio + out.n_samples);
        codes_buf.clear();
        return true;
    }

    llama_context *     lctx;
    mtmd_context *      mctx;
    const llama_model * model;
    const llama_vocab * vocab;
    int                 n_embd;
    mtmd_gen_audio_info info;
    mtmd::gen_state_ptr gen_state;

    bool specials_ok = false;
    llama_token codec_0   = LLAMA_TOKEN_NULL;
    llama_token codec_bos = LLAMA_TOKEN_NULL;
    llama_token codec_eos = LLAMA_TOKEN_NULL;
    llama_token codec_pad = LLAMA_TOKEN_NULL;
    llama_token c_think   = LLAMA_TOKEN_NULL;
    llama_token c_nothink = LLAMA_TOKEN_NULL;
    llama_token c_think_b = LLAMA_TOKEN_NULL;
    llama_token c_think_e = LLAMA_TOKEN_NULL;
    llama_token tts_pad   = LLAMA_TOKEN_NULL;
    llama_token tts_bos   = LLAMA_TOKEN_NULL;
    llama_token tts_eos   = LLAMA_TOKEN_NULL;
    std::vector<float> tok_embd;

    size_t window_frames = 72;

    llama_seq_id seq_id = 0;
    bool mrope = false;
    int pos = 0;
    std::vector<float> prompt_embd_buf;
    std::unique_ptr<decode_embd_batch> prompt_batch;
    int n_prompt = 0;
    int prompt_pos = 0;
    int32_t top_k = 50;
    float   top_p = 1.0f;
    std::vector<int32_t> codes_buf;
    std::vector<float>   audio_pcm;
    std::vector<float>   overlay;
    std::vector<float>   h_state_buf;
    mtmd_helper_gen_audio_outtype out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    std::vector<char> out_buf;
    bool   stream          = false;
    size_t pcm_sent        = 0;
    bool   wav_header_sent = false;
};

} // namespace

struct mtmd_helper_gen_audio {
    std::unique_ptr<qwen3tts_gen_audio_pipeline> pipeline;
};

struct mtmd_helper_gen_audio_inp mtmd_helper_gen_audio_inp_default() {
    mtmd_helper_gen_audio_inp inp {};
    inp.top_k    = 50;
    inp.top_p    = 1.0f;
    inp.temp     = -1.0f;
    inp.seed     = UINT32_MAX;
    inp.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;
    return inp;
}

mtmd_helper_gen_audio * mtmd_helper_gen_audio_init(struct llama_context * lctx, struct mtmd_context * mctx) {
    auto * ctx = new mtmd_helper_gen_audio();
    if (lctx && mctx && mtmd_gen_audio_get_info(mctx).type == MTMD_GEN_AUDIO_TYPE_QWEN3TTS) {
        ctx->pipeline = std::make_unique<qwen3tts_gen_audio_pipeline>(lctx, mctx);
    }
    return ctx;
}

void mtmd_helper_gen_audio_free(mtmd_helper_gen_audio * ctx) {
    delete ctx;
}

void mtmd_helper_gen_audio_reset(mtmd_helper_gen_audio * ctx) {
    if (ctx && ctx->pipeline) {
        ctx->pipeline->reset();
    }
}

int32_t mtmd_helper_gen_audio_set_input(mtmd_helper_gen_audio * ctx, const struct mtmd_helper_gen_audio_inp * inp) {
    if (!ctx || !ctx->pipeline || !inp) {
        LOG_ERR("mtmd_helper_gen_audio: unsupported or missing gen-audio pipeline\n");
        return 1;
    }
    if (inp->n_past > 0) {
        LOG_ERR("mtmd_helper_gen_audio: Qwen3-TTS cannot continue a sequence\n");
        return 1;
    }
    return ctx->pipeline->set_input(inp);
}

int32_t mtmd_helper_gen_audio_step_prompt(mtmd_helper_gen_audio * ctx, int32_t n_batch) {
    return (ctx && ctx->pipeline) ? ctx->pipeline->step_prompt(n_batch) : -1;
}

int32_t mtmd_helper_gen_audio_step_gen(mtmd_helper_gen_audio * ctx, llama_token sampled, const float * h_state_in,
                                       const float ** h_state_out, bool * out_stop) {
    if (!ctx || !ctx->pipeline) {
        if (h_state_out) *h_state_out = nullptr;
        if (out_stop)    *out_stop    = true;
        return 1;
    }
    bool stop = false;
    const int32_t ret = ctx->pipeline->step_gen(sampled, h_state_in, h_state_out, &stop);
    if (out_stop) {
        *out_stop = stop;
    }
    return ret;
}

bool mtmd_helper_gen_audio_next_is_text(mtmd_helper_gen_audio *) {
    return false;
}

const llama_logit_bias * mtmd_helper_gen_audio_get_logit_bias(mtmd_helper_gen_audio *, size_t * n_bias) {
    if (n_bias) *n_bias = 0;
    return nullptr;
}

int32_t mtmd_helper_audio_stream_get_logit_bias(const mtmd_context *, const struct llama_model *, llama_logit_bias *, int32_t) {
    return 0;
}

int32_t mtmd_helper_gen_audio_get_output(mtmd_helper_gen_audio * ctx, int32_t * out_sample_rate, const char ** out_data,
                                         size_t * out_data_len, int64_t * out_n_samples) {
    if (!ctx || !ctx->pipeline) {
        if (out_sample_rate) *out_sample_rate = 0;
        if (out_data)        *out_data        = nullptr;
        if (out_data_len)    *out_data_len    = 0;
        if (out_n_samples)   *out_n_samples   = 0;
        return 1;
    }
    return ctx->pipeline->get_output(out_sample_rate, out_data, out_data_len, out_n_samples);
}

int32_t mtmd_helper_gen_audio_flush(mtmd_helper_gen_audio * ctx) {
    return (ctx && ctx->pipeline) ? ctx->pipeline->flush() : 1;
}

//
// test API of mtmd-helper.h (internals of mtmd_helper_video)
//

mtmd_helper_video_seq::~mtmd_helper_video_seq() {}
void mtmd_helper_video_seq::configure(const mtmd_video_config &, const mtmd_helper_video_src &, float) {}
int32_t mtmd_helper_video_seq::ts_per_n_frames() const { return 0; }
mtmd_bitmap * mtmd_helper_video_seq::soundtrack_slice(int32_t) const { return nullptr; }
mtmd_bitmap * mtmd_helper_video_seq::read_next_frame() { return nullptr; }
int32_t mtmd_helper_video_seq::read_next(mtmd_bitmap ** out_bitmap, char ** out_text) {
    if (out_bitmap) *out_bitmap = nullptr;
    if (out_text)   *out_text   = nullptr;
    return -2;
}
