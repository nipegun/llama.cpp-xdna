#pragma once

#include "llama-model.h"
#include "llama-graph.h"
#include "llama-model-loader.h"

class llama_memory_hybrid_idx_context;

// note: almost all graphs require at least sqrtf, so include cmath globally
#include <cmath>

// ref: https://github.com/ggml-org/llama.cpp/pull/28068
static inline ggml_tensor * build_gdn_l2_norm(ggml_context * ctx, ggml_tensor * x, float eps) {
    const float n = x->ne[0];

    return ggml_scale(ctx, ggml_rms_norm(ctx, x, eps/n), 1.0f/sqrtf(n));
}

//
// base classes
//

struct llm_build_delta_net_base : public llm_graph_context {
    llm_build_delta_net_base(const llm_graph_params & params);

    virtual ~llm_build_delta_net_base() = default;

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_chunking(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il);

    // returns pair of output and new state
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_autoregressive(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                int           il);

    // use the ggml_gated_delta_net fused operator (K=1; state has shape [S_v, S_v, H_v, n_seqs])
    // llama.cpp-xdna: with state_in_place the operator updates s where it is (a cell of the cache) and the new state
    // returned is null (ggml_gated_delta_net_set_state_in_place)
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net_fused(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il,
                       bool   state_in_place = false);

    // choose one of two implementations above based on the number of tokens
    // llama.cpp-xdna: state_in_place only with the fused operator (see build_delta_net_fused)
    std::pair<ggml_tensor *, ggml_tensor *> build_delta_net(
                ggml_tensor * q,
                ggml_tensor * k,
                ggml_tensor * v,
                ggml_tensor * g,
                ggml_tensor * b,
                ggml_tensor * s,
                        int   il,
                       bool   state_in_place = false);

    // read conv state from cache, concat with qkv_mixed, write back (single slot or per-token)
    // qkv_mixed: (qkv_dim, n_seq_tokens, n_seqs); returns conv_input: (kernel_size + n_seq_tokens - 1, channels, n_seqs)
    // llama.cpp-xdna: by_tokens gives the input of the convolution by tokens, as a transposed view (see the
    // definition); conv_states, the states already read from the cache with build_conv_states, or null
    ggml_tensor * build_conv_state(
            llm_graph_input_rs * inp,
            ggml_tensor *        conv_states_all,
            ggml_tensor *        qkv_mixed,
            int64_t              conv_kernel_size,
            int64_t              conv_channels,
            int                  il,
            bool                 by_tokens   = false,
            ggml_tensor *        conv_states = nullptr);

    // llama.cpp-xdna: the states of the convolution of the ubatch, read from the cache: {d_conv - 1, channels, n_seqs}
    ggml_tensor * build_conv_states(
            llm_graph_input_rs * inp,
            ggml_tensor *        conv_states_all,
            int64_t              conv_kernel_size,
            int64_t              conv_channels,
            int                  il);

    // run delta-net attention and write the new recurrent state(s) back to ssm_states_all
    // s: (head_v_dim, head_v_dim, num_v_heads, n_seqs); returns output: (head_v_dim, num_v_heads, n_seq_tokens, n_seqs)
    ggml_tensor * build_recurrent_attn(
            llm_graph_input_rs * inp,
            ggml_tensor *        ssm_states_all,
            ggml_tensor *        q,
            ggml_tensor *        k,
            ggml_tensor *        v,
            ggml_tensor *        g,
            ggml_tensor *        b,
            ggml_tensor *        s,
            int                  il);
};

//
// models
//

struct llama_model_qwen35 : public llama_model_base {
    llama_model_qwen35(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int * sections,
                            int   il);

        // llama.cpp-xdna: conv_states, the states of the short convolution already read (build_conv_states), or null
        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il,
                    ggml_tensor * conv_states = nullptr);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_qwen35moe : public llama_model_base {
    llama_model_qwen35moe(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    // llama.cpp-xdna: Qwen3.5 with a block of experts instead of the dense feed-forward block (ported from upstream,
    // with everything the qwen35 of this fork has for the NPU)
    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);
    private:
        ggml_tensor * build_layer_attn(
        llm_graph_input_attn_kv * inp_attn,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int * sections,
                            int   il);

        // conv_states, the states of the short convolution already read (build_conv_states), or null
        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il,
                    ggml_tensor * conv_states = nullptr);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        // returns pair of qkv, z
        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);
    };

    struct graph_mtp : public llm_graph_context {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

// llama.cpp-xdna: flint (ported from upstream on 2026/11/17): a transformer whose layers with a sliding window have
// RoPE and the others not, with attention sinks, leading dense layers and then a block of experts (sigmoid gating) with
// a shared expert
struct llama_model_flint : public llama_model_base {
    llama_model_flint(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

struct llama_model_granite : public llama_model_base {
    llama_model_granite(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);

    private:
        ggml_tensor * build_attention_layer(
                  ggml_tensor             * cur,
                  ggml_tensor             * inp_pos,
                  llm_graph_input_attn_kv * inp_attn,
            const int64_t                 n_embd_head,
            const int                     il);

        ggml_tensor * build_layer_ffn(
                  ggml_tensor       * cur,
                  ggml_tensor       * inpSA,
            const int                 il);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

// llama.cpp-xdna: Qwen3-TTS talker transformer (ported from upstream on 2026/11/19): dense Qwen3 blocks with
// interleaved M-RoPE, per-head Q/K RMS norms and a 3072-entry codec output head padded with -inf to n_vocab
struct llama_model_qwen3tts : public llama_model_base {
    llama_model_qwen3tts(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        graph(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

// llama.cpp-xdna: qwen4exp (ported from upstream on 2026/11/21): hybrid GDN + QSA block-sparse attention with
// hyper-connections, 512 routed experts + shared expert and per-layer n-gram hash embeddings (PLE)
struct llama_model_qwen4exp : public llama_model_base {
    llama_model_qwen4exp(const struct llama_model_params & params) : llama_model_base(params) {}

    class llm_graph_input_kpool;

    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_build_delta_net_base {
        graph(const llm_graph_params & params);
    protected:
        // the helpers alone, graph_mtp builds its own body
        struct no_build {};
        graph(const llm_graph_params & params, no_build) :
            llm_build_delta_net_base(params) {}

        // HC replaces every layer norm: residual is [n_embd, hc, n_tokens]
        struct hc_mix {
            ggml_tensor * cur;     // the mixed [n_embd, T] stream
            ggml_tensor * inject;  // [hc, T] scatter weights, null when the inject weights are null
        };

        hc_mix build_hc_mix(
                    ggml_tensor * x,
                    ggml_tensor * w_norm,
                    ggml_tensor * w_down,
                    ggml_tensor * w_up,
                    ggml_tensor * w_inject,
                    ggml_tensor * w_down_inject,
                            int   il);

        ggml_tensor * build_hc_combine(
                    ggml_tensor * residual,
                    ggml_tensor * block_out,
                    ggml_tensor * inject,
                            int   il);

        ggml_tensor * build_layer_attn(
              llm_graph_input_attn_kv * inp_attn,
  const llama_memory_hybrid_idx_context * mctx_hyb,
          llm_graph_input_kpool * inp_kpool,
                    ggml_tensor * cur,
                    ggml_tensor * inp_pos,
                            int   il);

        ggml_tensor * build_attn_qsa(
        llm_graph_input_attn_kv * inp,
                    ggml_tensor * q_cur,
                    ggml_tensor * k_cur,
                    ggml_tensor * v_cur,
                    ggml_tensor * sel,
                        int64_t   n_sel,
                          float   kq_scale,
                            int   il);

        llm_graph_input_kpool * build_inp_kpool(const llama_memory_hybrid_idx_context * mctx_hyb);

        ggml_tensor * build_qsa_sel(
  const llama_memory_hybrid_idx_context * mctx_hyb,
          llm_graph_input_kpool * inp_kpool,
                    ggml_tensor * k_raw,
                    ggml_tensor * q_idx,
                    ggml_tensor * inp_pos,
                            int   il);

        ggml_tensor * build_layer_attn_linear(
             llm_graph_input_rs * inp,
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_layer_ffn(
                    ggml_tensor * cur,
                            int   il);

        ggml_tensor * build_norm_gated(
                    ggml_tensor * input,
                    ggml_tensor * weights,
                    ggml_tensor * gate,
                            int   layer);

        std::map<ggml_tensor *, ggml_tensor *> rs_rows;

        ggml_tensor * build_conv_state_at(
             llm_graph_input_rs * inp,
                    ggml_tensor * conv_states_all,
                    ggml_tensor * x,
                        int64_t   state_cols,
                        int64_t   channels,
                            int   il);

        ggml_tensor * build_inp_ple(
  const llama_memory_hybrid_idx_context * mctx_hyb);

        ggml_tensor * build_ple(
             llm_graph_input_rs * inp,
                    ggml_tensor * emb,
                    ggml_tensor * hidden,
                            int   il);

        std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(
                    ggml_tensor * input,
                            int   il);
    };

    struct graph_mtp : public graph {
        graph_mtp(const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
