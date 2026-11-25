#include "models.h"

// llama.cpp-xdna: Qwen3-TTS talker transformer (arch "qwen3tts", e.g. Qwen3-TTS-12Hz-0.6B-Base and 1.7B-Base), ported
// from src/models/qwen3vl.cpp of upstream llama.cpp without the vision/deepstack paths. Q, K and V projections are
// expanded immediately after creation so the XDNA backend groups them in prefill and chains them in generation.

void llama_model_qwen3tts::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // llama.cpp-xdna: validate M-RoPE sections and head geometry with exceptions rather than GGML_ASSERT
    {
        int64_t sect_dims = 0;
        for (int i = 0; i < 4; ++i) {
            if (hparams.rope_sections[i] < 0) {
                throw std::runtime_error("invalid rope.dimension_sections: negative section " + std::to_string(hparams.rope_sections[i]));
            }
            sect_dims += hparams.rope_sections[i];
        }
        if (hparams.rope_sections[0] + hparams.rope_sections[1] + hparams.rope_sections[2] == 0 ||
            sect_dims > (int64_t) hparams.n_embd_head_k_full) {
            throw std::runtime_error("invalid rope.dimension_sections: their sum is " + std::to_string(sect_dims) +
                                     ", the first three must not be 0 and the sum must not be larger than the heads (" +
                                     std::to_string(hparams.n_embd_head_k_full) + ")");
        }
    }
    if (hparams.n_embd_head_v() == 0 ||
        hparams.n_embd_head_v() != hparams.n_embd_head_k() ||
        hparams.n_embd_head_v() != hparams.n_rot()) {
        throw std::runtime_error("invalid qwen3tts head dimensions: key_length (" + std::to_string(hparams.n_embd_head_k()) +
                                 "), value_length (" + std::to_string(hparams.n_embd_head_v()) +
                                 ") and rope.dimension_count (" + std::to_string(hparams.n_rot()) + ") must match and be > 0");
    }

    switch (hparams.n_layer()) {
        case 28: type = hparams.n_embd == 1024 ? LLM_TYPE_0_6B : LLM_TYPE_1_7B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen3tts::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    // [TAG_LLAMA_N_VOCAB_OUT]
    const int64_t n_vocab_out = 3072;
    if (n_vocab < n_vocab_out) {
        throw std::runtime_error("invalid qwen3tts vocab_size: " + std::to_string(n_vocab) + " (must be >= 3072)");
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab_out}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // optional rerank head
    cls_out = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {n_embd, hparams.n_cls_out}, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_k_gqa, n_embd_v_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen3tts::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_qwen3tts::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head,
                    n_embd_head, n_head_kv,
                    n_embd_head, n_head_kv,
                    il, false);

            // llama.cpp-xdna: keep Q, K and V projections consecutive in the graph so XDNA groups them in one command
            // and chains them with the previous layer's FFN during generation
            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_multi(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_multi(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head (codec_head)
    cur = build_lora_mm(model.output, cur, model.output_s);
    // llama.cpp-xdna: expand output projection immediately after result_norm so fFeedForward chains them in layer n_layer - 1
    ggml_build_forward_expand(gf, cur);

    const int64_t n_vocab_in  = model.tok_embd->ne[1];
    const int64_t n_vocab_out = model.output->ne[1];
    if (n_vocab_in > n_vocab_out) {
        // Qwen3-TTS model with codec_head as output
        GGML_ASSERT(model.output_norm);
        const int64_t pad = n_vocab_in - n_vocab_out;

        ggml_tensor * neg_inf = ggml_scale_bias(ctx0,
                ggml_view_1d(ctx0, model.output_norm, 1, 0),
                0.0f, -INFINITY);
        neg_inf = ggml_repeat_4d(ctx0, neg_inf, pad, cur->ne[1], 1, 1);
        cur = ggml_concat(ctx0, neg_inf, cur, 0); // [padded .. n_vocab_out, n_stream]
    } else if (n_vocab_in < n_vocab_out) {
        GGML_ABORT("invalid case");
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
