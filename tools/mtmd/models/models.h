#pragma once

// llama.cpp-xdna: minimal internal definitions for Qwen3-TTS audio generation and speaker encoding
// (PROJECTOR_TYPE_QWEN3TTS_SPKENC and PROJECTOR_TYPE_QWEN3TTS_GEN), ported from tools/mtmd/ of upstream llama.cpp.
// Vision, video and unrelated audio architectures are intentionally excluded.

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-backend.h"
#include "mtmd.h"

#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

std::string mtmd_format(const char * fmt, ...);

enum mtmd_modality {
    MTMD_MODALITY_AUDIO,
    MTMD_MODALITY_GEN_AUDIO,
};

enum projector_type {
    PROJECTOR_TYPE_UNKNOWN,
    PROJECTOR_TYPE_QWEN3TTS_SPKENC,
    PROJECTOR_TYPE_QWEN3TTS_GEN,
};

enum mtmd_flash_attn_type {
    MTMD_FLASH_ATTN_TYPE_AUTO     = -1,
    MTMD_FLASH_ATTN_TYPE_DISABLED = 0,
    MTMD_FLASH_ATTN_TYPE_ENABLED  = 1,
};

enum mtmd_subctx_gen_process {
    MTMD_SUBCTX_GEN_PROCESS_GEN_UNKNOWN,
    MTMD_SUBCTX_GEN_PROCESS_GEN_CODE,
    MTMD_SUBCTX_GEN_PROCESS_GEN_WAV,
};

struct mtmd_image_f32 {
    int nx_ = 0;
    int ny_ = 0;
    bool is_audio = true;
    std::vector<float> buf;

    int nx() const { return nx_; }
    int ny() const { return ny_; }
    bool is_placeholder() const { return buf.empty(); }
    const std::vector<float> & get_ro_buf() const { return buf; }

    void set_size(std::pair<int, int> sz, bool placeholder, bool audio = true) {
        nx_ = sz.first;
        ny_ = sz.second;
        is_audio = audio;
        if (placeholder) {
            buf.clear();
        } else {
            buf.assign((size_t) nx_ * (size_t) ny_, 0.0f);
        }
    }

    void cpy_buf(const std::vector<float> & src) {
        buf = src;
    }
};

struct mtmd_image_f32_batch {
    bool is_audio = true;
    std::vector<mtmd_image_f32> entries;
};

struct mtmd_hparams {
    int32_t patch_size     = 1;
    int32_t n_embd         = 0;
    int32_t n_ff           = 0;
    int32_t projection_dim = 0;
    int32_t n_head         = 0;
    int32_t n_head_kv      = 0;
    int32_t n_embd_head    = 0;
    int32_t n_layer        = 0;
    float   eps            = 1e-6f;
    float   rope_theta     = 1000000.0f;

    int32_t n_mel_bins        = 128;
    int32_t warmup_audio_size = 3000;
    int32_t audio_sample_rate = 24000;
    int32_t audio_n_fft       = 1024;
    int32_t audio_window_len  = 1024;
    int32_t audio_hop_len     = 256;

    std::string gen_model_variant;

    int32_t wav_tfm_n_layer      = 8;
    int32_t wav_tfm_n_embd       = 512;
    int32_t wav_tfm_n_ff         = 1024;
    int32_t wav_tfm_n_head       = 16;
    int32_t wav_tfm_n_head_kv    = 16;
    float   wav_tfm_eps          = 1e-5f;
    float   wav_tfm_rope_theta   = 10000.0f;
    int32_t wav_upsample_n_block = 2;
    int32_t wav_dac_n_block      = 4;
    int32_t wav_dac_n_res        = 3;
    int32_t wav_tfm_swa          = 72;
};

struct mtmd_layer {
    ggml_tensor * ln_1_w = nullptr;
    ggml_tensor * ln_2_w = nullptr;
    ggml_tensor * k_w    = nullptr;
    ggml_tensor * q_w    = nullptr;
    ggml_tensor * v_w    = nullptr;
    ggml_tensor * o_w    = nullptr;
    ggml_tensor * o_b    = nullptr;
    ggml_tensor * k_norm = nullptr;
    ggml_tensor * q_norm = nullptr;

    ggml_tensor * ff_up_w   = nullptr;
    ggml_tensor * ff_gate_w = nullptr;
    ggml_tensor * ff_down_w = nullptr;

    ggml_tensor * ls_1_w = nullptr;
    ggml_tensor * ls_2_w = nullptr;

    ggml_tensor * conv_pw1_w = nullptr;
    ggml_tensor * conv_pw1_b = nullptr;
    ggml_tensor * conv_pw2_w = nullptr;
    ggml_tensor * conv_pw2_b = nullptr;
    ggml_tensor * se_conv1_w = nullptr;
    ggml_tensor * se_conv1_b = nullptr;
    ggml_tensor * se_conv2_w = nullptr;
    ggml_tensor * se_conv2_b = nullptr;
    std::vector<ggml_tensor *> res2_conv_w;
    std::vector<ggml_tensor *> res2_conv_b;
};

struct mtmd_code2wav {
    struct upsample_block {
        ggml_tensor * conv_w   = nullptr;
        ggml_tensor * conv_b   = nullptr;
        ggml_tensor * dwconv_w = nullptr;
        ggml_tensor * dwconv_b = nullptr;
        ggml_tensor * norm_w   = nullptr;
        ggml_tensor * norm_b   = nullptr;
        ggml_tensor * pw1_w    = nullptr;
        ggml_tensor * pw1_b    = nullptr;
        ggml_tensor * pw2_w    = nullptr;
        ggml_tensor * pw2_b    = nullptr;
        ggml_tensor * gamma    = nullptr;
    };

    struct dac_res {
        ggml_tensor * act1_alpha = nullptr;
        ggml_tensor * act1_beta  = nullptr;
        ggml_tensor * conv1_w    = nullptr;
        ggml_tensor * conv1_b    = nullptr;
        ggml_tensor * act2_alpha = nullptr;
        ggml_tensor * act2_beta  = nullptr;
        ggml_tensor * conv2_w    = nullptr;
        ggml_tensor * conv2_b    = nullptr;
    };

    struct dac_block {
        ggml_tensor * snake_alpha = nullptr;
        ggml_tensor * snake_beta  = nullptr;
        ggml_tensor * conv_w      = nullptr;
        ggml_tensor * conv_b      = nullptr;
        std::vector<dac_res> res;
    };

    ggml_tensor * quant_first_in_w  = nullptr;
    ggml_tensor * quant_first_out_w = nullptr;
    ggml_tensor * quant_first_cb_w  = nullptr;
    ggml_tensor * quant_rest_in_w   = nullptr;
    ggml_tensor * quant_rest_out_w  = nullptr;
    ggml_tensor * quant_rest_cb_w   = nullptr;

    ggml_tensor * pre_conv_w = nullptr;
    ggml_tensor * pre_conv_b = nullptr;

    ggml_tensor * tfm_in_proj_w     = nullptr;
    ggml_tensor * tfm_in_proj_b     = nullptr;
    ggml_tensor * tfm_out_proj_w    = nullptr;
    ggml_tensor * tfm_out_proj_b    = nullptr;
    ggml_tensor * tfm_output_norm_w = nullptr;
    std::vector<mtmd_layer> tfm_layers;

    std::vector<upsample_block> upsample;

    ggml_tensor * dac_entry_w = nullptr;
    ggml_tensor * dac_entry_b = nullptr;
    std::vector<dac_block> dac;
    ggml_tensor * dac_post_snake_alpha = nullptr;
    ggml_tensor * dac_post_snake_beta  = nullptr;
    ggml_tensor * dac_post_conv_w      = nullptr;
    ggml_tensor * dac_post_conv_b      = nullptr;
};

struct c2w_state_slot {
    std::string name;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    bool in_place = false;
};

struct mtmd_model {
    mtmd_modality  modality        = MTMD_MODALITY_GEN_AUDIO;
    projector_type proj_type       = PROJECTOR_TYPE_UNKNOWN;
    int            max_nodes       = 8192;
    bool           flash_attn_auto = false;
    mtmd_hparams   hparams;

    ggml_context_ptr        ctx_data;
    ggml_backend_buffer_ptr buf;
    std::map<ggml_backend_dev_t, size_t> mem_weights;

    std::vector<mtmd_layer> layers;

    // speaker encoder (ECAPA-TDNN)
    ggml_tensor * conv1d_1_w     = nullptr;
    ggml_tensor * conv1d_1_b     = nullptr;
    ggml_tensor * conv_out_w     = nullptr;
    ggml_tensor * conv_out_b     = nullptr;
    ggml_tensor * spk_asp_attn_w = nullptr;
    ggml_tensor * spk_asp_attn_b = nullptr;
    ggml_tensor * spk_asp_tdnn_w = nullptr;
    ggml_tensor * spk_asp_tdnn_b = nullptr;
    ggml_tensor * mm_fc_w        = nullptr;
    ggml_tensor * mm_fc_b        = nullptr;

    // code_predictor (qwen3tts_gen)
    ggml_tensor * gen_code_proj_in_w  = nullptr;
    ggml_tensor * gen_code_proj_in_b  = nullptr;
    ggml_tensor * gen_code_embd_w     = nullptr;
    ggml_tensor * gen_code_out_embd_w = nullptr;
    ggml_tensor * gen_code_norm_w     = nullptr;
    // llama.cpp-xdna: 2D slices of a.gen.code.head.weight ([n_embd, vocab] per acoustic codebook) stored as standalone
    // GGML_OP_NONE tensors so XDNA accepts and chains layer 4's output_norm + head_g in xdna2-gemv-chain
    std::vector<ggml_tensor *> gen_code_head_2d;
    // llama.cpp-xdna: standalone 2D weight tensors ([K*IC, OC] for conv1d, [IC, K*OC] for conv_transpose1d) populated
    // when an accelerator/GPU backend is active so XDNA accelerates ECAPA-TDNN and code2wav convolutions
    std::map<const ggml_tensor *, ggml_tensor *> conv1d_w_2d;
    std::map<const ggml_tensor *, ggml_tensor *> conv1d_tr_w_2d;

    mtmd_code2wav c2w;

    int n_mmproj_embd() const {
        if (proj_type == PROJECTOR_TYPE_QWEN3TTS_SPKENC) {
            return (int) mm_fc_w->ne[2];
        }
        return (int) gen_code_out_embd_w->ne[0];
    }

    int64_t n_acoustic_codebooks() const {
        return (int64_t) gen_code_head_2d.size();
    }
};

std::vector<c2w_state_slot> list_c2w_state_slots(const mtmd_hparams & hparams, const mtmd_model & model);

struct mtmd_subctx_gen_state {
    std::mt19937 rng;
    ggml_context_ptr        ctx_data;
    ggml_backend_buffer_ptr buf;
    std::vector<ggml_tensor *> slots;

    void reset() {
        if (buf) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
};

struct mtmd_subctx_encode_params {
    int n_threads = 1;
    const mtmd_image_f32_batch * imgs = nullptr;
    std::vector<float> * out_embd = nullptr;

    mtmd_subctx_gen_process gen_process = MTMD_SUBCTX_GEN_PROCESS_GEN_UNKNOWN;

    int32_t code0 = 0;
    int32_t top_k = 50;
    float   top_p = 1.0f;
    float   temp  = 0.0f;
    std::vector<int32_t> * out_codes = nullptr;

    const std::vector<int32_t> * codes = nullptr;
    std::vector<float> * out_audio = nullptr;

    mtmd_subctx_gen_state * gen_state = nullptr;
};

struct mtmd_backends;

struct mtmd_subcontext {
    const mtmd_model & model;
    mtmd_backends & backends;

    std::vector<uint8_t> buf_compute_meta;
    int max_nodes = 8192;
    mtmd_flash_attn_type flash_attn_type = MTMD_FLASH_ATTN_TYPE_DISABLED;
    bool is_allocated = false;

    mtmd_subcontext(const mtmd_model & model, mtmd_backends & backends, mtmd_flash_attn_type fa_type, bool do_warmup);
    void warmup(const mtmd_image_f32_batch & batch);
    bool encode(mtmd_subctx_encode_params * params);
    mtmd_subctx_gen_state * gen_state_init(uint32_t seed);
};

struct mtmd_graph {
    const mtmd_model & model;
    const mtmd_hparams & hparams;
    projector_type proj_type;
    const mtmd_image_f32 & img;

    const int n_embd;
    const int n_head;
    const int n_head_kv;
    const int d_head;
    const int n_layer;
    const int n_mmproj_embd;
    const float eps;
    const float kq_scale;
    const mtmd_flash_attn_type flash_attn_type;
    int n_batch = 1;

    ggml_context_ptr ctx0_ptr;
    ggml_context * ctx0 = nullptr;
    ggml_cgraph * gf    = nullptr;

    mtmd_graph(mtmd_subcontext * ctx, const mtmd_image_f32 & img);
    mtmd_graph(const mtmd_graph & parent);
    virtual ~mtmd_graph() = default;
    virtual ggml_cgraph * build() = 0;

    void cb(ggml_tensor * cur, const char * name, int il) const;
    ggml_tensor * build_mm(ggml_tensor * w, ggml_tensor * x) const;
    ggml_tensor * build_inp_raw(int channels = 1);
    ggml_tensor * build_attn(
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur,
            ggml_tensor * k_cur,
            ggml_tensor * v_cur,
            ggml_tensor * kq_mask,
            float scale,
            int il) const;
};

struct mtmd_graph_qwen3tts_spkenc : mtmd_graph {
    mtmd_graph_qwen3tts_spkenc(mtmd_subcontext * ctx, const mtmd_image_f32 & img) : mtmd_graph(ctx, img) {}
    ggml_cgraph * build() override;

    ggml_tensor * conv1d_same(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int dilation) const;
    ggml_tensor * res2net(ggml_tensor * x, const mtmd_layer & layer, int dilation, int scale) const;
    ggml_tensor * se_block(ggml_tensor * x, const mtmd_layer & layer) const;
    ggml_tensor * se_res2net_block(ggml_tensor * x, const mtmd_layer & layer, int dilation, int scale) const;
    ggml_tensor * attentive_stats_pool(ggml_tensor * x) const;
};

struct mtmd_graph_qwen3tts_gen : mtmd_graph {
    mtmd_graph_qwen3tts_gen(mtmd_subcontext * ctx, const mtmd_image_f32 & img, mtmd_subctx_gen_process gen_process, int top_k, float top_p)
        : mtmd_graph(ctx, img), gen_process(gen_process), top_k(top_k), top_p(top_p) {}
    ggml_cgraph * build() override;

    mtmd_subctx_gen_process gen_process;
    int   top_k;
    float top_p;

    struct code_gen : mtmd_graph {
        code_gen(const mtmd_graph & parent, int top_k, float top_p)
            : mtmd_graph(parent), top_k(top_k), top_p(top_p) {}
        ggml_cgraph * build() override { GGML_ABORT("call prefill()/step() instead"); }

        int   top_k;
        float top_p;

        ggml_tensor * cache_set(ggml_tensor * cache, int row_idx, ggml_tensor * value) const;
        ggml_tensor * do_sampling(ggml_tensor * logits, ggml_tensor * inp_rand) const;
        ggml_tensor * const_i32(ggml_tensor * anchor, float value) const;
        ggml_tensor * causal_mask_row(int64_t n_kv_pad, int pos) const;
        ggml_tensor * project_in(ggml_tensor * cur) const;

        ggml_tensor * layer_forward(
                ggml_tensor * cur,
                const mtmd_layer & layer,
                ggml_tensor * inp_pos,
                ggml_tensor * kq_mask,
                ggml_tensor *& k_cache_layer,
                ggml_tensor *& v_cache_layer,
                int64_t n_kv_pad,
                int pos,
                int il) const;

        void prefill(
                std::vector<ggml_tensor *> & k_cache,
                std::vector<ggml_tensor *> & v_cache,
                ggml_tensor *& out_code_cache,
                ggml_tensor * h_state,
                ggml_tensor * code0_embd,
                ggml_tensor * inp_rand) const;

        ggml_tensor * step(
                std::vector<ggml_tensor *> & k_cache,
                std::vector<ggml_tensor *> & v_cache,
                ggml_tensor * out_code_cache,
                ggml_tensor * inp_rand,
                int step_idx) const;
    };

    struct code2wav : mtmd_graph {
        code2wav(const mtmd_graph & parent) : mtmd_graph(parent) {}
        ggml_cgraph * build() override { GGML_ABORT("call decode() instead"); }

        std::map<std::string, ggml_tensor *> state_in;
        mutable std::vector<std::pair<std::string, ggml_tensor *>> state_out;

        ggml_tensor * causal_conv1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int dilation, const std::string & state_name) const;
        ggml_tensor * causal_conv1d_dw(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, const std::string & state_name) const;
        ggml_tensor * causal_conv_transpose1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, const std::string & state_name) const;
        ggml_tensor * snake(ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * beta) const;

        ggml_tensor * quant_decode(ggml_tensor * inp_codes) const;
        ggml_tensor * tfm_layer_forward(ggml_tensor * cur, const mtmd_layer & layer, int il) const;
        ggml_tensor * convnext_block(ggml_tensor * x, const mtmd_code2wav::upsample_block & blk, const std::string & state_prefix) const;
        ggml_tensor * dac_res_unit(ggml_tensor * x, const mtmd_code2wav::dac_res & res, int dilation, const std::string & state_name) const;

        bool gelu_erf = false;
        ggml_tensor * decode(ggml_tensor * inp_codes) const;
    };
};
