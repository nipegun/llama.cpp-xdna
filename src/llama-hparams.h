#pragma once

#include "llama.h"

#include <array>
#include <bitset>
#include <cassert>
#include <cmath>

// bump if necessary
#define LLAMA_MAX_LAYERS    512
#define LLAMA_MAX_EXPERTS   512 // llama.cpp-xdna: qwen35moe has 256, qwen4exp has 512
#define LLAMA_MAX_PLE_NGRAM 8   // llama.cpp-xdna: qwen4exp PLE
#define LLAMA_MAX_PLE_HEADS 64  // llama.cpp-xdna: qwen4exp PLE

// llama.cpp-xdna: how the router of a mixture of experts turns its logits into probabilities (ported from upstream)
enum llama_expert_gating_func_type {
    LLAMA_EXPERT_GATING_FUNC_TYPE_NONE           = 0,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX        = 1,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID        = 2,
    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT = 3, // applied to the router weights instead of the logits
};

enum llama_swa_type {
    LLAMA_SWA_TYPE_NONE      = 0,
    LLAMA_SWA_TYPE_STANDARD  = 1,
    LLAMA_SWA_TYPE_CHUNKED   = 2,
    LLAMA_SWA_TYPE_SYMMETRIC = 3,
};

// how the non-causal mask should be constructed with llama_set_causal_attn(ctx, false)
// (e.g. mtmd decoding image tokens)
enum llama_non_causal_type {
    LLAMA_NON_CAUSAL_TYPE_ALL      = 0, // all layers non-causal, SWA still applied (gemma 3, qwen-vl, ...)
    LLAMA_NON_CAUSAL_TYPE_SWA_ONLY = 1, // SWA layers non-causal, dense layers stay causal (gemma 4)
    LLAMA_NON_CAUSAL_TYPE_SWA_FULL = 2, // all layers non-causal, SWA not applied between tokens of the current ubatch (deepseek 4)
    LLAMA_NON_CAUSAL_TYPE_DENSE_ONLY = 3, // dense layers non-causal, SWA layers stay causal (lfm3 d1)
};

// M-RoPE: which input position slot feeds each RoPE section
enum llama_rope_section_order {
    LLAMA_ROPE_SECTION_ORDER_UNSPECIFIED = -1,
    LLAMA_ROPE_SECTION_ORDER_TYXZ        = 0, // default, slot i feeds section i
    LLAMA_ROPE_SECTION_ORDER_ZYXT        = 1, // MiniCPM-V 4.7: time last
};

struct llama_hparams {
    // note: use the `_impl` suffix to avoid name conflict between members and getters
    //       for example: n_embd_out() vs n_embd_out_impl

    bool vocab_only;
    bool rope_finetuned;

    uint32_t n_ctx_train; // context size the model was trained on
    uint32_t n_embd;
    uint32_t n_layer_all;
    uint32_t n_layer_nextn = 0;

    // layer il reads the KV cache of layer kv_source_impl[il] (own cache if equal to il)
    std::array<uint32_t, LLAMA_MAX_LAYERS> kv_source_impl;

    // different head size for full_attention and SWA layers
    uint32_t n_embd_head_k_full; // dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
    uint32_t n_embd_head_v_full; // dimension of values (d_v) aka n_embd_head
    uint32_t n_embd_head_k_swa;
    uint32_t n_embd_head_v_swa;

    // different RoPE dimensions for full_attention and SWA layers
    uint32_t n_rot_full;
    uint32_t n_rot_swa;

    // llama.cpp-xdna: K-only cache marker used by llama_memory_hybrid_idx for the QSA indexer cache
    uint32_t n_embd_head_k_mla_impl = 0;
    uint32_t n_embd_head_v_mla_impl = 0;

    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_kv_arr;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_arr;

    // llama.cpp-xdna: the experts of a mixture of experts (qwen35moe), ported from upstream: how many there are and
    // how many every token uses, the width of a routed expert per layer and of the shared one, and how the weights of
    // the experts of a token are made
    uint32_t n_expert      = 0;
    uint32_t n_expert_used = 0;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_ff_exp_arr;
    uint32_t n_ff_shexp    = 0;
    float    expert_weights_scale = 0.0f;
    bool     expert_weights_norm  = false;
    uint32_t expert_gating_func   = LLAMA_EXPERT_GATING_FUNC_TYPE_NONE;
    // llama.cpp-xdna: flint, ported from upstream: its first n_layer_dense_lead layers have a dense feed-forward
    // block instead of the experts, and its shared expert counts as n_expert_shared experts (one tensor of n_ff_shexp)
    uint32_t n_layer_dense_lead = 0;
    uint32_t n_expert_shared    = 0;

    float f_norm_eps;
    float f_norm_rms_eps;

    float f_attn_logit_softcapping   = 50.0f;

    float    rope_attn_factor = 1.0f;
    float    rope_freq_base_train;
    float    rope_freq_base_train_swa  = 10000.0f;
    float    rope_freq_scale_train;
    float    rope_freq_scale_train_swa = 1.0f;
    float    rope_scaling_alpha        = 0.0f;  // NTK-aware alpha for XDRoPE

    uint32_t n_ctx_orig_yarn;
    float    rope_yarn_log_mul = 0.0f;

    float    yarn_ext_factor  = -1.0f;
    float    yarn_attn_factor =  1.0f;
    float    yarn_beta_fast   = 32.0f;
    float    yarn_beta_slow   =  1.0f;

    std::array<int, 4> rope_sections;

    enum llama_rope_section_order rope_section_order = LLAMA_ROPE_SECTION_ORDER_TYXZ;

    // Per-layer RoPE enable flags (1 = use RoPE, 0 = NoPE)
    // by default, all layers use RoPE (controlled by rope_finetuned)
    std::array<uint32_t, LLAMA_MAX_LAYERS> rope_pattern;

    // Sliding Window Attention (SWA)
    llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;
    // the size of the sliding window (0 - no SWA)
    uint32_t n_swa = 0;

    // see llama_non_causal_type
    // note: for SWA_FULL, older tokens (outside the current ubatch) are still window-clipped
    llama_non_causal_type non_causal_type = LLAMA_NON_CAUSAL_TYPE_ALL;

    // if is_swa_impl[il] == 1, then layer il is SWA
    // if is_swa_impl[il] == 0, then layer il is dense (i.e. non-SWA)
    // by default, all layers are dense
    // note: using uint32_t type for compatibility reason
    std::array<uint32_t, LLAMA_MAX_LAYERS> is_swa_impl;

    // linear attention (gated delta net) layers of the hybrid model
    std::array<uint32_t, LLAMA_MAX_LAYERS> is_recr_impl;

    // gated delta net
    uint32_t ssm_d_conv  = 0;
    uint32_t ssm_d_inner = 0;
    uint32_t ssm_d_state = 0;
    uint32_t ssm_dt_rank = 0;
    uint32_t ssm_n_group = 0;

    // llama.cpp-xdna: qwen4exp QSA indexer, hyper-connections and per-layer n-gram embeddings (PLE)
    uint32_t indexer_n_head            = 0;
    uint32_t indexer_head_size         = 0;
    uint32_t indexer_top_k             = 0;
    uint32_t indexer_kpool             = 0;
    bool     indexer_kpool_select_tail = true;
    uint32_t indexer_kpool_row         = 3;
    bool     indexer_kpool_by_order    = false;

    uint32_t dsv4_hc_mult = 0;
    uint32_t hc_low_rank  = 0;
    std::array<uint32_t, LLAMA_MAX_LAYERS> dsv4_compress_ratios;

    uint32_t    n_embd_per_layer    = 0;
    uint32_t    ple_ngram_size      = 0;
    uint32_t    ple_heads_per_ngram = 0;
    uint32_t    ple_conv_kernel     = 0;
    uint32_t    ple_n_heads         = 0;
    uint32_t    ple_head_dim        = 0;
    uint32_t    ple_eos_token_id    = 0;
    uint32_t    ple_image_token_id  = 0;
    llama_token ple_embd_token      = LLAMA_TOKEN_NULL;
    uint32_t    ple_n_modules       = 1;

    std::bitset<LLAMA_MAX_LAYERS>              is_ple_impl;
    std::array<uint64_t, LLAMA_MAX_PLE_NGRAM>  ple_layer_multipliers;
    std::array<uint32_t, LLAMA_MAX_PLE_HEADS>  ple_head_offsets;
    std::array<uint32_t, LLAMA_MAX_PLE_HEADS>  ple_head_vocab_sizes;

    float f_clamp_kqv      = 0.0f;
    float f_max_alibi_bias = 0.0f;
    float f_logit_scale    = 0.0f;

    // additional scale factors (f_residual_scale, f_embedding_scale and f_attention_scale: granite)
    float f_residual_scale  = 0.0f;
    float f_embedding_scale = 0.0f;
    float f_attention_scale = 0.0f;
    float f_attn_value_scale = 0.0f;

    bool causal_attn   = true;
    bool use_alibi     = false;
    bool attn_soft_cap = false;

    // for Classifiers
    uint32_t n_cls_out = 1;

    // output embedding dimension (0 = use n_embd)
    uint32_t n_embd_out_impl = 0;

    enum llama_pooling_type      pooling_type            = LLAMA_POOLING_TYPE_NONE;
    enum llama_pooling_type      pooling_type_cls        = LLAMA_POOLING_TYPE_UNSPECIFIED; // pooling before the classifier head (RANK)
    enum llama_rope_type         rope_type               = LLAMA_ROPE_TYPE_NONE;
    enum llama_rope_scaling_type rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;

    enum ggml_unary_op           act_cls                 = GGML_UNARY_OP_TANH; // activation of the classifier head (RANK)

    // return true if one of the layers is SWA
    bool is_swa_any() const;

    bool is_swa(uint32_t il) const;

    // whether or not the given layer is recurrent (for hybrid models)
    bool is_recr(uint32_t il) const;

    uint32_t n_head(uint32_t il = 0) const;

    uint32_t n_head_kv(uint32_t il = 0) const;

    uint32_t n_ff(uint32_t il = 0) const;

    // llama.cpp-xdna: the width of a routed expert of layer il
    uint32_t n_ff_exp(uint32_t il = 0) const;

    uint32_t n_gqa(uint32_t il = 0) const;

    uint32_t n_rot(uint32_t il = 0) const;

    // dimension of the input embeddings
    uint32_t n_embd_inp() const;

    // dimension of the encoder input embeddings (no encoder: same as n_embd_inp())
    uint32_t n_embd_inp_enc() const;

    // dimension of output embeddings
    uint32_t n_embd_out() const;

    // dimension of key/value embeddings for each head (per layer)
    uint32_t n_embd_head_k(uint32_t il = 0) const;
    uint32_t n_embd_head_v(uint32_t il = 0) const;

    // dimension of key embeddings across all k-v heads
    uint32_t n_embd_k_gqa(uint32_t il = 0) const;

    // dimension of value embeddings across all k-v heads
    uint32_t n_embd_v_gqa(uint32_t il = 0) const;

    // true if any layer has a different n_embd_v_gqa
    bool is_n_embd_v_gqa_variable() const;

    // return the maximum n_embd_v_gqa across all layers
    uint32_t n_embd_v_gqa_max() const;

    // dimension of the rolling state embeddings (the conv states of the gated delta net)
    uint32_t n_embd_r() const;

    // dimension of the recurrent state embeddings (the states of the gated delta net)
    uint32_t n_embd_s() const;

    uint32_t n_pos_per_embd() const;

    bool is_mla() const;

    bool is_ple(uint32_t il) const;

    uint32_t ple_conv_state() const;

    bool has_kv(uint32_t il) const;

    uint32_t kv_source(uint32_t il) const;

    bool has_rope(uint32_t il) const;

    // number of effective layers (excludes nextn layers)
    uint32_t n_layer() const;

    // note that this function uses different SWA parameters from those in the hparams
    // note: inlined on purpose for performance reasons
    // TODO: think of a better place for this function
    // TODO: pack the SWA params in a struct?
    static bool is_masked_swa(uint32_t n_swa, llama_swa_type swa_type, llama_pos p0, llama_pos p1) {
        assert(p0 >= 0 && p1 >= 0);

        switch (swa_type) {
            case LLAMA_SWA_TYPE_NONE:
                {
                } break;
            case LLAMA_SWA_TYPE_STANDARD:
                {
                    if (p1 - p0 >= (int32_t) n_swa) {
                        return true;
                    }
                } break;
            case LLAMA_SWA_TYPE_CHUNKED:
                {
                    const llama_pos pos_chunk_start = (p1 / n_swa) * n_swa;

                    if (p0 < pos_chunk_start) {
                        return true;
                    }
                } break;
            case LLAMA_SWA_TYPE_SYMMETRIC:
                {
                    const int32_t half_n_swa = (int32_t) n_swa / 2;
                    const int32_t pos_diff = p1 - p0;

                    // Mask if outside the symmetric window
                    if (pos_diff < -half_n_swa || pos_diff > half_n_swa) {
                        return true;
                    }
                } break;
        }

        return false;
    }
};

static_assert(std::is_trivially_copyable<llama_hparams>::value, "llama_hparams must be trivially copyable");
