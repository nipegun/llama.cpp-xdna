#include "models.h"

#include <cmath>
#include <string>

// llama.cpp-xdna: ported from tools/mtmd/models/qwen3tts-gen.cpp of upstream llama.cpp.
// Q, K and V projections in code_gen and code2wav are ordered together (ggml_build_forward_order) and the 15 acoustic
// codebook heads use standalone 2D tensors (model.gen_code_head_2d) so the XDNA backend groups Q/K/V and chains every
// layer of code_gen (including layer 4's output_norm + head) in xdna2-gemv-chain.

// on-device sampling: top-k, top-p, then a random draw
ggml_tensor * mtmd_graph_qwen3tts_gen::code_gen::do_sampling(ggml_tensor * logits, ggml_tensor * inp_rand) const {
    logits = ggml_reshape_1d(ctx0, logits, ggml_nelements(logits));
    const int64_t n_vocab = logits->ne[0];

    // sort a's rows by idx
    auto sort_by = [this](ggml_tensor * a, ggml_tensor * idx) {
        ggml_tensor * a2d = ggml_reshape_2d(ctx0, a, 1, a->ne[0]);
        return ggml_reshape_1d(ctx0, ggml_get_rows(ctx0, a2d, idx), idx->ne[0]);
    };

    ggml_tensor * cur        = logits;
    ggml_tensor * candidates = nullptr; // maps row index back to vocab id

    if (top_k > 0 && top_k < n_vocab) {
        ggml_tensor * idx = ggml_top_k(ctx0, cur, top_k);
        candidates = idx;
        cur        = sort_by(cur, idx);
        cb(cur, "sample_top_k_logits", -1);
    }

    if (top_p < 1.0f) {
        ggml_tensor * sorted_idx    = ggml_argsort(ctx0, cur, GGML_SORT_ORDER_DESC);
        ggml_tensor * sorted_logits = sort_by(cur, sorted_idx);
        candidates = candidates ? sort_by(candidates, sorted_idx) : sorted_idx;

        ggml_tensor * probs = ggml_soft_max(ctx0, sorted_logits);
        ggml_tensor * cdf   = ggml_cumsum(ctx0, probs);

        // keep_mask[i] = 1 once cdf[i] crosses top_p
        ggml_tensor * cdf_scaled = ggml_scale_bias(ctx0, cdf, -1.0f, top_p);
        ggml_tensor * keep_mask  = ggml_step(ctx0, cdf_scaled);
        ggml_tensor * idxf       = ggml_sum(ctx0, keep_mask);
        idxf = ggml_clamp(ctx0, idxf, 0.0f, (float) keep_mask->ne[0] - 1);
        ggml_tensor * ones = ggml_scale_bias(ctx0, idxf, 0.0f, 1.0f);

        // top-p must include the crossing element, so force it to 1
        ggml_tensor * keep_mask_2d = ggml_reshape_2d(ctx0, keep_mask, 1, keep_mask->ne[0]);
        keep_mask_2d = ggml_set_rows(ctx0, keep_mask_2d, ones, ggml_cast(ctx0, idxf, GGML_TYPE_I32));
        keep_mask    = ggml_reshape_1d(ctx0, keep_mask_2d, keep_mask->ne[0]);

        // log(1) = 0 (keep), log(0) = -inf (drop)
        ggml_tensor * bias = ggml_log(ctx0, keep_mask);
        cur = ggml_add(ctx0, sorted_logits, bias);
        cb(cur, "sample_top_p_logits", -1);
    }

    // draw one token: find where the cdf crosses inp_rand
    ggml_tensor * probs  = ggml_soft_max(ctx0, cur);
    ggml_tensor * cumsum = ggml_cumsum(ctx0, probs);

    ggml_tensor * diff       = ggml_sub(ctx0, cumsum, inp_rand);
    ggml_tensor * cross_mask = ggml_step(ctx0, diff);
    ggml_tensor * idxf       = ggml_sum(ctx0, cross_mask);
    ggml_tensor * idx        = ggml_cast(ctx0, ggml_scale_bias(ctx0, idxf, -1.0f, (float) cross_mask->ne[0]), GGML_TYPE_I32);

    if (candidates) {
        ggml_tensor * cand_2d = ggml_reshape_2d(ctx0, candidates, 1, candidates->ne[0]);
        idx = ggml_get_rows(ctx0, cand_2d, idx);
    }
    cb(idx, "sample_token_id", -1);

    return idx;
}

// returns a new cache with row row_idx set to value
ggml_tensor * mtmd_graph_qwen3tts_gen::code_gen::cache_set(ggml_tensor * cache, int row_idx, ggml_tensor * value) const {
    const int64_t n_embd  = cache->ne[0];
    const int64_t n_cache = cache->ne[1];
    GGML_ASSERT(row_idx >= 0 && row_idx < n_cache);

    // append value as the last row, then gather it back into place
    ggml_tensor * value_2d  = ggml_reshape_2d(ctx0, value, n_embd, 1);
    ggml_tensor * cache_ext = ggml_concat(ctx0, cache, value_2d, 1); // [n_embd, n_cache + 1]

    ggml_tensor * idx = const_i32(cache, (float) n_cache);
    if (row_idx > 0) {
        ggml_tensor * prefix = ggml_cast(ctx0, ggml_arange(ctx0, 0.0f, (float) row_idx, 1.0f), GGML_TYPE_I32);
        idx = ggml_concat(ctx0, prefix, idx, 0);
    }
    if (row_idx < n_cache - 1) {
        ggml_tensor * suffix = ggml_cast(ctx0, ggml_arange(ctx0, (float) (row_idx + 1), (float) n_cache, 1.0f), GGML_TYPE_I32);
        idx = ggml_concat(ctx0, idx, suffix, 0);
    }

    ggml_tensor * result = ggml_get_rows(ctx0, cache_ext, idx);
    cb(result, "cache_set_out", -1);
    return result;
}

// builds a const i32 with no host upload: view a tensor, zero it via scale, add value, cast to i32
ggml_tensor * mtmd_graph_qwen3tts_gen::code_gen::const_i32(ggml_tensor * anchor, float value) const {
    ggml_tensor * v = ggml_view_1d(ctx0, anchor, 1, 0);
    if (v->type != GGML_TYPE_F32) {
        v = ggml_cast(ctx0, v, GGML_TYPE_F32);
    }
    return ggml_cast(ctx0, ggml_scale_bias(ctx0, v, 0.0f, value), GGML_TYPE_I32);
}

// causal keep-mask row for a query at position pos, window size n_kv_pad
ggml_tensor * mtmd_graph_qwen3tts_gen::code_gen::causal_mask_row(int64_t n_kv_pad, int pos) const {
    ggml_tensor * ones = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv_pad, n_kv_pad), 1.0f);
    ggml_tensor * keep = ggml_tri(ctx0, ones, GGML_TRI_TYPE_LOWER_DIAG);
    ggml_tensor * row  = ggml_view_1d(ctx0, keep, n_kv_pad, (size_t) pos * keep->nb[1]);
    ggml_tensor * mask = ggml_log(ctx0, row); // 0 = keep, -inf = masked
    mask = ggml_reshape_4d(ctx0, mask, n_kv_pad, 1, 1, 1);
    if (flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED) {
        mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);
    }
    return mask;
}

// talker hidden size -> predictor hidden size (small_to_mtp_projection)
ggml_tensor * mtmd_graph_qwen3tts_gen::code_gen::project_in(ggml_tensor * cur) const {
    if (!model.gen_code_proj_in_w) {
        return cur;
    }
    cur = ggml_mul_mat(ctx0, model.gen_code_proj_in_w, cur);
    if (model.gen_code_proj_in_b) {
        cur = ggml_add(ctx0, cur, model.gen_code_proj_in_b);
    }
    return cur;
}

// one transformer layer at position pos; writes k/v into k_cache_layer/v_cache_layer at row pos
ggml_tensor * mtmd_graph_qwen3tts_gen::code_gen::layer_forward(
        ggml_tensor * cur,
        const mtmd_layer & layer,
        ggml_tensor * inp_pos,
        ggml_tensor * kq_mask,
        ggml_tensor *& k_cache_layer,
        ggml_tensor *& v_cache_layer,
        int64_t n_kv_pad,
        int pos,
        int il) const {
    const int     n_head    = hparams.n_head;
    const int     n_head_kv = hparams.n_head_kv;
    const int64_t d_head    = layer.q_w->ne[1] / n_head; // real head_dim, not n_embd / n_head
    const float   kq_scale  = 1.0f / sqrtf((float) d_head);

    ggml_tensor * residual = cur;

    ggml_tensor * h = ggml_rms_norm(ctx0, cur, hparams.eps);
    h = ggml_mul(ctx0, h, layer.ln_1_w);

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.q_w, h);
    ggml_tensor * k = ggml_mul_mat(ctx0, layer.k_w, h);
    ggml_tensor * v = ggml_mul_mat(ctx0, layer.v_w, h);

    // llama.cpp-xdna: keep Q, K and V projections consecutive without marking GEN_CODE for GEN_WAV
    ggml_build_forward_order(gf, q);
    ggml_build_forward_order(gf, k);
    ggml_build_forward_order(gf, v);

    q = ggml_reshape_3d(ctx0, q, d_head, n_head, 1);
    k = ggml_reshape_3d(ctx0, k, d_head, n_head_kv, 1);

    q = ggml_rms_norm(ctx0, q, hparams.eps);
    q = ggml_mul(ctx0, q, layer.q_norm);
    k = ggml_rms_norm(ctx0, k, hparams.eps);
    k = ggml_mul(ctx0, k, layer.k_norm);

    q = ggml_rope_ext(ctx0, q, inp_pos, nullptr, (int) d_head, GGML_ROPE_TYPE_NEOX, 0,
                      hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx0, k, inp_pos, nullptr, (int) d_head, GGML_ROPE_TYPE_NEOX, 0,
                      hparams.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // write k/v into the cache at row pos, flat layout
    ggml_tensor * k_flat = ggml_reshape_1d(ctx0, k, d_head * n_head_kv);
    k_cache_layer = cache_set(k_cache_layer, pos, k_flat);
    v_cache_layer = cache_set(v_cache_layer, pos, v);

    ggml_tensor * q_cur = ggml_reshape_4d(ctx0, q, d_head, n_head, 1, 1);
    ggml_tensor * k_cur = ggml_reshape_4d(ctx0, k_cache_layer, d_head, n_head_kv, n_kv_pad, 1);
    ggml_tensor * v_cur = ggml_reshape_4d(ctx0, v_cache_layer, d_head, n_head_kv, n_kv_pad, 1);

    ggml_tensor * attn_out = build_attn(layer.o_w, layer.o_b, q_cur, k_cur, v_cur, kq_mask, kq_scale, il);

    cur = ggml_add(ctx0, residual, attn_out);

    ggml_tensor * h2 = ggml_rms_norm(ctx0, cur, hparams.eps);
    h2 = ggml_mul(ctx0, h2, layer.ln_2_w);

    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ff_gate_w, h2);
    ggml_tensor * up   = ggml_mul_mat(ctx0, layer.ff_up_w, h2);
    ggml_tensor * gu   = ggml_swiglu_split(ctx0, gate, up);
    ggml_tensor * down = ggml_mul_mat(ctx0, layer.ff_down_w, gu);

    return ggml_add(ctx0, cur, down);
}

// position 0: hidden bridge, seeds the k/v cache, no sampling
// position 1: embed(code0), sample with lm_head[0], write out_code_cache[1]
void mtmd_graph_qwen3tts_gen::code_gen::prefill(
        std::vector<ggml_tensor *> & k_cache,
        std::vector<ggml_tensor *> & v_cache,
        ggml_tensor *& out_code_cache,
        ggml_tensor * h_state,
        ggml_tensor * code0_embd,
        ggml_tensor * inp_rand) const {
    const int64_t n_kv_pad = k_cache[0]->ne[1];

    {
        ggml_tensor * cur     = project_in(h_state);
        ggml_tensor * kq_mask = causal_mask_row(n_kv_pad, 0);
        ggml_tensor * inp_pos = const_i32(k_cache[0], 0.0f);
        for (size_t il = 0; il < model.layers.size(); il++) {
            cur = layer_forward(cur, model.layers[il], inp_pos, kq_mask, k_cache[il], v_cache[il], n_kv_pad, 0, (int) il);
        }
    }

    {
        ggml_tensor * cur     = project_in(code0_embd);
        ggml_tensor * kq_mask = causal_mask_row(n_kv_pad, 1);
        ggml_tensor * inp_pos = const_i32(k_cache[0], 1.0f);
        for (size_t il = 0; il < model.layers.size(); il++) {
            cur = layer_forward(cur, model.layers[il], inp_pos, kq_mask, k_cache[il], v_cache[il], n_kv_pad, 1, (int) il);
        }

        cur = ggml_rms_norm(ctx0, cur, hparams.eps);
        cur = ggml_mul(ctx0, cur, model.gen_code_norm_w);

        ggml_tensor * logits = ggml_mul_mat(ctx0, model.gen_code_head_2d[0], cur);
        ggml_build_forward_order(gf, logits);

        ggml_tensor * sampled = do_sampling(logits, inp_rand);
        out_code_cache = cache_set(out_code_cache, 1, sampled);
    }
}

ggml_tensor * mtmd_graph_qwen3tts_gen::code_gen::step(
        std::vector<ggml_tensor *> & k_cache,
        std::vector<ggml_tensor *> & v_cache,
        ggml_tensor * out_code_cache,
        ggml_tensor * inp_rand,
        int step_idx) const {
    const int64_t n_acoustic = model.n_acoustic_codebooks();
    GGML_ASSERT(step_idx >= 1 && step_idx < n_acoustic);
    GGML_ASSERT(k_cache.size() == model.layers.size());
    GGML_ASSERT(v_cache.size() == model.layers.size());

    const int64_t n_kv_pad = k_cache[0]->ne[1];
    const int     pos      = step_idx + 1;

    ggml_tensor * code_in = ggml_view_1d(ctx0, out_code_cache, 1, (size_t) step_idx * out_code_cache->nb[1]);

    ggml_tensor * embd_w = model.gen_code_embd_w; // [n_embd_talker, vocab, n_acoustic]
    ggml_tensor * embd_g = ggml_view_2d(ctx0, embd_w, embd_w->ne[0], embd_w->ne[1], embd_w->nb[1],
                                        (size_t) (step_idx - 1) * embd_w->nb[2]);
    ggml_tensor * cur = ggml_get_rows(ctx0, embd_g, code_in);
    cur = ggml_reshape_1d(ctx0, cur, cur->ne[0]);
    cb(cur, "step_embd_in", step_idx);

    cur = project_in(cur);
    cb(cur, "step_proj_in", step_idx);

    ggml_tensor * kq_mask = causal_mask_row(n_kv_pad, pos);
    ggml_tensor * inp_pos = const_i32(k_cache[0], (float) pos);

    for (size_t il = 0; il < model.layers.size(); il++) {
        cur = layer_forward(cur, model.layers[il], inp_pos, kq_mask, k_cache[il], v_cache[il], n_kv_pad, pos, (int) il);
        cb(cur, "step_layer_out", (int) il);
    }

    cur = ggml_rms_norm(ctx0, cur, hparams.eps);
    cur = ggml_mul(ctx0, cur, model.gen_code_norm_w);

    ggml_tensor * logits = ggml_mul_mat(ctx0, model.gen_code_head_2d[(size_t) step_idx], cur);
    ggml_build_forward_order(gf, logits);
    cb(logits, "step_logits", step_idx);

    ggml_tensor * sampled = do_sampling(logits, inp_rand);
    cb(sampled, "step_sampled", step_idx);

    return cache_set(out_code_cache, pos, sampled);
}

ggml_tensor * mtmd_graph_qwen3tts_gen::code2wav::causal_conv1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int dilation, const std::string & state_name) const {
    const int K   = (int) w->ne[0];
    const int pad = (K - 1) * dilation;

    ggml_tensor * x_full = x;
    if (pad > 0) {
        ggml_tensor * left = state_in.at(state_name);
        x_full = ggml_concat(ctx0, left, x, 0);
    }
    ggml_tensor * y = nullptr;
    const auto it2d = model.conv1d_w_2d.find(w);
    if (it2d != model.conv1d_w_2d.end()) {
        // llama.cpp-xdna: express causal conv1d as F32 im2col (or transpose for K=1) + standalone 2D mul_mat for XDNA GEMM
        const int64_t IC    = w->ne[1];
        const int64_t T_out = x_full->ne[0] - pad;
        ggml_tensor * col   = nullptr;
        if (K == 1 && dilation == 1) {
            col = ggml_cont(ctx0, ggml_transpose(ctx0, x_full)); // [IC, T_out]
        } else {
            ggml_tensor * col4d = ggml_im2col(ctx0, w, x_full, 1, 0, 0, 0, dilation, 0, false, GGML_TYPE_F32);
            col = ggml_reshape_2d(ctx0, col4d, (int64_t) K * IC, T_out);
        }
        ggml_tensor * y_t = ggml_mul_mat(ctx0, it2d->second, col); // [OC, T_out]
        y = ggml_cont(ctx0, ggml_transpose(ctx0, y_t));             // [T_out, OC]
    } else {
        y = ggml_conv_1d(ctx0, w, x_full, 1, 0, dilation);
        y = ggml_reshape_2d(ctx0, y, y->ne[0], y->ne[1]);
    }
    if (b) {
        y = ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    if (pad > 0) {
        ggml_tensor * new_left = ggml_cont(ctx0, ggml_view_2d(ctx0, x_full, pad, x_full->ne[1], x_full->nb[1],
                                                              (size_t) (x_full->ne[0] - pad) * x_full->nb[0]));
        state_out.push_back({state_name, new_left});
    }
    return y;
}

ggml_tensor * mtmd_graph_qwen3tts_gen::code2wav::causal_conv1d_dw(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, const std::string & state_name) const {
    const int K   = (int) w->ne[0];
    const int pad = K - 1;

    ggml_tensor * x_full = x;
    if (pad > 0) {
        ggml_tensor * left = state_in.at(state_name);
        x_full = ggml_concat(ctx0, left, x, 0);
    }
    ggml_tensor * y = ggml_conv_1d_dw(ctx0, w, x_full, 1, 0, 1);
    y = ggml_reshape_2d(ctx0, y, y->ne[0], y->ne[1]);
    if (b) {
        y = ggml_add(ctx0, y, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    if (pad > 0) {
        ggml_tensor * new_left = ggml_cont(ctx0, ggml_view_2d(ctx0, x_full, pad, x_full->ne[1], x_full->nb[1],
                                                              (size_t) (x_full->ne[0] - pad) * x_full->nb[0]));
        state_out.push_back({state_name, new_left});
    }
    return y;
}

ggml_tensor * mtmd_graph_qwen3tts_gen::code2wav::causal_conv_transpose1d(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, int stride, const std::string & state_name) const {
    const int     K        = (int) w->ne[0];
    const int     OC       = (int) w->ne[1];
    const int     trim     = K - stride;
    const int64_t emit_len = x->ne[0] * stride;

    const auto it_tr = model.conv1d_tr_w_2d.find(w);
    ggml_tensor * w2 = nullptr;
    if (it_tr != model.conv1d_tr_w_2d.end()) {
        w2 = it_tr->second;
    } else {
        w2 = ggml_reshape_2d(ctx0, w, (int64_t) K * OC, w->ne[2]);
        w2 = ggml_cont(ctx0, ggml_transpose(ctx0, w2));
    }
    ggml_tensor * xt  = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    ggml_tensor * col = ggml_mul_mat(ctx0, w2, xt);
    ggml_tensor * y   = ggml_col2im_1d(ctx0, col, stride, OC, 0);

    ggml_tensor * out = y;
    if (trim > 0) {
        ggml_tensor * tail = state_in.at(state_name);
        ggml_tensor * head = ggml_add(ctx0, ggml_view_2d(ctx0, y, trim, y->ne[1], y->nb[1], 0), tail);
        if (emit_len > trim) {
            ggml_tensor * middle = ggml_view_2d(ctx0, y, emit_len - trim, y->ne[1], y->nb[1], (size_t) trim * y->nb[0]);
            out = ggml_concat(ctx0, head, middle, 0);
        } else {
            out = head;
        }
        ggml_tensor * new_tail = ggml_cont(ctx0, ggml_view_2d(ctx0, y, trim, y->ne[1], y->nb[1], (size_t) emit_len * y->nb[0]));
        state_out.push_back({state_name, new_tail});
    }
    if (b) {
        out = ggml_add(ctx0, out, ggml_reshape_2d(ctx0, b, 1, b->ne[0]));
    }
    return out;
}

ggml_tensor * mtmd_graph_qwen3tts_gen::code2wav::snake(ggml_tensor * x, ggml_tensor * alpha, ggml_tensor * beta) const {
    ggml_tensor * a = ggml_reshape_2d(ctx0, alpha, 1, alpha->ne[0]);
    ggml_tensor * b = ggml_reshape_2d(ctx0, beta,  1, beta->ne[0]);

    ggml_build_forward_order(gf, a);
    ggml_build_forward_order(gf, b);

    ggml_tensor * s = ggml_sin(ctx0, ggml_mul(ctx0, x, a));
    s = ggml_sqr(ctx0, s);
    s = ggml_mul(ctx0, s, b);
    return ggml_add(ctx0, x, s);
}

ggml_tensor * mtmd_graph_qwen3tts_gen::code2wav::quant_decode(ggml_tensor * inp_codes) const {
    const auto & c2w = model.c2w;
    const int64_t T = inp_codes->ne[0];

    auto group_ids = [&](int g) {
        return ggml_view_1d(ctx0, inp_codes, T, (size_t) g * inp_codes->nb[1]);
    };

    ggml_tensor * sem     = ggml_get_rows(ctx0, c2w.quant_first_cb_w, group_ids(0));
    ggml_tensor * sem_out = ggml_mul_mat(ctx0, c2w.quant_first_out_w, sem);

    ggml_tensor * acc = nullptr;
    const int64_t n_acoustic = c2w.quant_rest_cb_w->ne[2];
    for (int g = 1; g <= n_acoustic; g++) {
        ggml_tensor * cb_g  = ggml_view_2d(ctx0, c2w.quant_rest_cb_w, c2w.quant_rest_cb_w->ne[0], c2w.quant_rest_cb_w->ne[1],
                                           c2w.quant_rest_cb_w->nb[1], (size_t) (g - 1) * c2w.quant_rest_cb_w->nb[2]);
        ggml_tensor * embd = ggml_get_rows(ctx0, cb_g, group_ids(g));
        acc = acc ? ggml_add(ctx0, acc, embd) : embd;
    }
    ggml_tensor * ac_out = ggml_mul_mat(ctx0, c2w.quant_rest_out_w, acc);

    ggml_tensor * hidden = ggml_add(ctx0, sem_out, ac_out);
    cb(hidden, "wav_quant_hidden", -1);
    return hidden;
}

ggml_tensor * mtmd_graph_qwen3tts_gen::code2wav::tfm_layer_forward(ggml_tensor * cur, const mtmd_layer & layer, int il) const {
    const int     n_head    = hparams.wav_tfm_n_head;
    const int     n_head_kv = hparams.wav_tfm_n_head_kv;
    const int64_t d_head    = layer.q_w->ne[1] / n_head;
    const float   kq_scale  = 1.0f / sqrtf((float) d_head);
    const int64_t W         = hparams.wav_tfm_swa;
    const int64_t N         = cur->ne[1];
    const int64_t prefix    = W - 1;
    const int64_t total_kv  = prefix + N;

    ggml_tensor * residual = cur;
    ggml_tensor * h = ggml_rms_norm(ctx0, cur, hparams.wav_tfm_eps);
    h = ggml_mul(ctx0, h, layer.ln_1_w);

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.q_w, h);
    ggml_tensor * k = ggml_mul_mat(ctx0, layer.k_w, h);
    ggml_tensor * v = ggml_mul_mat(ctx0, layer.v_w, h);

    // llama.cpp-xdna: keep Q, K and V projections consecutive so XDNA groups them in one GEMM command (N = 72 >= 40)
    ggml_build_forward_order(gf, q);
    ggml_build_forward_order(gf, k);
    ggml_build_forward_order(gf, v);

    q = ggml_reshape_3d(ctx0, q, d_head, n_head, N);
    k = ggml_reshape_3d(ctx0, k, d_head, n_head_kv, N);

    ggml_tensor * base   = ggml_reshape_1d(ctx0, state_in.at("tfm_pos"), 1);
    ggml_tensor * offset = ggml_arange(ctx0, 0.0f, (float) N, 1.0f);
    ggml_tensor * pos    = ggml_cast(ctx0, ggml_add(ctx0, offset, base), GGML_TYPE_I32);

    q = ggml_rope_ext(ctx0, q, pos, nullptr, (int) d_head, GGML_ROPE_TYPE_NEOX, 0,
                      hparams.wav_tfm_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx0, k, pos, nullptr, (int) d_head, GGML_ROPE_TYPE_NEOX, 0,
                      hparams.wav_tfm_rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    if (il == 0) {
        state_out.push_back({"tfm_pos", ggml_scale_bias(ctx0, state_in.at("tfm_pos"), 1.0f, (float) N)});
    }

    ggml_tensor * k_new = ggml_reshape_2d(ctx0, k, d_head * n_head_kv, N);
    ggml_tensor * v_new = ggml_reshape_2d(ctx0, v, d_head * n_head_kv, N);

    ggml_tensor * old_k = state_in.at("tfm_k_" + std::to_string(il));
    ggml_tensor * old_v = state_in.at("tfm_v_" + std::to_string(il));

    ggml_tensor * k_full = ggml_concat(ctx0, old_k, k_new, 1);
    ggml_tensor * v_full = ggml_concat(ctx0, old_v, v_new, 1);

    state_out.push_back({"tfm_k_" + std::to_string(il),
        ggml_cont(ctx0, ggml_view_2d(ctx0, k_full, k_full->ne[0], prefix, k_full->nb[1], (size_t) N * k_full->nb[1]))});
    state_out.push_back({"tfm_v_" + std::to_string(il),
        ggml_cont(ctx0, ggml_view_2d(ctx0, v_full, v_full->ne[0], prefix, v_full->nb[1], (size_t) N * v_full->nb[1]))});

    ggml_tensor * pos_k = ggml_reshape_2d(ctx0, ggml_arange(ctx0, 0.0f, (float) total_kv, 1.0f), total_kv, 1);
    ggml_tensor * pos_q = ggml_reshape_2d(ctx0, ggml_arange(ctx0, (float) prefix, (float) (prefix + N), 1.0f), 1, N);
    ggml_tensor * pos_q_grid = ggml_repeat_4d(ctx0, pos_q, total_kv, N, 1, 1);
    ggml_tensor * diff = ggml_sub(ctx0, pos_q_grid, pos_k);

    ggml_tensor * causal_keep = ggml_step(ctx0, ggml_scale_bias(ctx0, diff, 1.0f, 0.5f));
    ggml_tensor * in_window   = ggml_step(ctx0, ggml_scale_bias(ctx0, diff, -1.0f, (float) W - 0.5f));
    ggml_tensor * keep = ggml_mul(ctx0, causal_keep, in_window);

    ggml_tensor * warm = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_add(ctx0, pos_k, base),
                                                         1.0f, 0.5f - (float) prefix));
    keep = ggml_mul(ctx0, keep, warm);

    ggml_tensor * mask = ggml_reshape_4d(ctx0, ggml_log(ctx0, keep), total_kv, N, 1, 1);
    if (flash_attn_type == MTMD_FLASH_ATTN_TYPE_ENABLED) {
        mask = ggml_cast(ctx0, mask, GGML_TYPE_F16);
    }

    ggml_tensor * q_cur = ggml_reshape_4d(ctx0, q, d_head, n_head, N, 1);
    ggml_tensor * k_cur = ggml_reshape_4d(ctx0, k_full, d_head, n_head_kv, total_kv, 1);
    ggml_tensor * v_cur = ggml_reshape_4d(ctx0, v_full, d_head, n_head_kv, total_kv, 1);

    ggml_tensor * attn_out = build_attn(layer.o_w, layer.o_b, q_cur, k_cur, v_cur, mask, kq_scale, il);
    if (layer.ls_1_w) {
        attn_out = ggml_mul(ctx0, attn_out, layer.ls_1_w);
    }
    cur = ggml_add(ctx0, residual, attn_out);

    ggml_tensor * residual2 = cur;
    ggml_tensor * h2 = ggml_rms_norm(ctx0, cur, hparams.wav_tfm_eps);
    h2 = ggml_mul(ctx0, h2, layer.ln_2_w);

    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ff_gate_w, h2);
    ggml_tensor * up   = ggml_mul_mat(ctx0, layer.ff_up_w, h2);
    ggml_tensor * gu   = ggml_swiglu_split(ctx0, gate, up);
    ggml_tensor * down = ggml_mul_mat(ctx0, layer.ff_down_w, gu);
    if (layer.ls_2_w) {
        down = ggml_mul(ctx0, down, layer.ls_2_w);
    }
    return ggml_add(ctx0, residual2, down);
}

ggml_tensor * mtmd_graph_qwen3tts_gen::code2wav::convnext_block(ggml_tensor * x, const mtmd_code2wav::upsample_block & blk, const std::string & state_prefix) const {
    ggml_tensor * residual = x;

    ggml_tensor * h = causal_conv1d_dw(x, blk.dwconv_w, blk.dwconv_b, state_prefix + "_dwconv");
    ggml_tensor * hc = ggml_cont(ctx0, ggml_transpose(ctx0, h));

    hc = ggml_norm(ctx0, hc, 1e-6f);
    hc = ggml_mul(ctx0, hc, blk.norm_w);
    hc = ggml_add(ctx0, hc, blk.norm_b);

    ggml_tensor * g = ggml_mul_mat(ctx0, blk.pw1_w, hc);
    g = ggml_add(ctx0, g, blk.pw1_b);
    g = gelu_erf ? ggml_gelu_erf(ctx0, g) : ggml_gelu(ctx0, g);
    g = ggml_mul_mat(ctx0, blk.pw2_w, g);
    g = ggml_add(ctx0, g, blk.pw2_b);
    g = ggml_mul(ctx0, g, blk.gamma);

    ggml_tensor * g_t = ggml_cont(ctx0, ggml_transpose(ctx0, g));
    return ggml_add(ctx0, residual, g_t);
}

ggml_tensor * mtmd_graph_qwen3tts_gen::code2wav::dac_res_unit(ggml_tensor * x, const mtmd_code2wav::dac_res & res, int dilation, const std::string & state_name) const {
    ggml_tensor * residual = x;
    ggml_tensor * h = snake(x, res.act1_alpha, res.act1_beta);
    h = causal_conv1d(h, res.conv1_w, res.conv1_b, dilation, state_name);
    h = snake(h, res.act2_alpha, res.act2_beta);
    h = causal_conv1d(h, res.conv2_w, res.conv2_b, 1, "");
    return ggml_add(ctx0, residual, h);
}

ggml_tensor * mtmd_graph_qwen3tts_gen::code2wav::decode(ggml_tensor * inp_codes) const {
    const auto & c2w = model.c2w;

    ggml_tensor * hidden = quant_decode(inp_codes);

    ggml_tensor * x = ggml_cont(ctx0, ggml_transpose(ctx0, hidden));
    x = causal_conv1d(x, c2w.pre_conv_w, c2w.pre_conv_b, 1, "pre_conv");
    cb(x, "wav_pre_conv_out", -1);

    ggml_tensor * cur = ggml_cont(ctx0, ggml_transpose(ctx0, x));
    cur = ggml_mul_mat(ctx0, c2w.tfm_in_proj_w, cur);
    cur = ggml_add(ctx0, cur, c2w.tfm_in_proj_b);

    for (int il = 0; il < hparams.wav_tfm_n_layer; il++) {
        cur = tfm_layer_forward(cur, c2w.tfm_layers[il], il);
    }

    cur = ggml_rms_norm(ctx0, cur, hparams.wav_tfm_eps);
    cur = ggml_mul(ctx0, cur, c2w.tfm_output_norm_w);
    cur = ggml_mul_mat(ctx0, c2w.tfm_out_proj_w, cur);
    cur = ggml_add(ctx0, cur, c2w.tfm_out_proj_b);
    cb(cur, "wav_tfm_out", -1);

    x = ggml_cont(ctx0, ggml_transpose(ctx0, cur));
    for (size_t il = 0; il < c2w.upsample.size(); il++) {
        const auto & up = c2w.upsample[il];
        x = causal_conv_transpose1d(x, up.conv_w, up.conv_b, 2, "");
        x = convnext_block(x, up, "up" + std::to_string(il));
        cb(x, "wav_upsample_out", (int) il);
    }

    static constexpr int DAC_DILATIONS[3] = { 1, 3, 9 };

    x = causal_conv1d(x, c2w.dac_entry_w, c2w.dac_entry_b, 1, "dac_entry");
    cb(x, "wav_dac_entry_out", -1);

    for (size_t il = 0; il < c2w.dac.size(); il++) {
        const auto & blk = c2w.dac[il];
        const int stride = (int) (blk.conv_w->ne[0] / 2);
        const std::string blk_name = "dac" + std::to_string(il);
        x = snake(x, blk.snake_alpha, blk.snake_beta);
        x = causal_conv_transpose1d(x, blk.conv_w, blk.conv_b, stride, blk_name + "_tail");
        for (size_t ir = 0; ir < blk.res.size(); ir++) {
            x = dac_res_unit(x, blk.res[ir], DAC_DILATIONS[ir], blk_name + "_res" + std::to_string(ir));
        }
        cb(x, "wav_dac_block_out", (int) il);
    }

    x = snake(x, c2w.dac_post_snake_alpha, c2w.dac_post_snake_beta);
    x = causal_conv1d(x, c2w.dac_post_conv_w, c2w.dac_post_conv_b, 1, "dac_post_conv");

    x = ggml_clamp(ctx0, x, -1.0f, 1.0f);
    x = ggml_reshape_1d(ctx0, x, x->ne[0]);
    cb(x, "wav_audio_out", -1);
    return x;
}

std::vector<c2w_state_slot> list_c2w_state_slots(const mtmd_hparams & hparams, const mtmd_model & model) {
    const auto & c2w = model.c2w;
    std::vector<c2w_state_slot> slots;

    if (c2w.dac_entry_w == nullptr) {
        return slots;
    }

    slots.push_back({"tfm_pos", 1, 1});

    const int64_t d_head = c2w.tfm_layers[0].q_w->ne[1] / hparams.wav_tfm_n_head;
    const int64_t kv_ch  = d_head * hparams.wav_tfm_n_head_kv;
    const int64_t prefix = hparams.wav_tfm_swa - 1;
    for (int il = 0; il < hparams.wav_tfm_n_layer; il++) {
        slots.push_back({"tfm_k_" + std::to_string(il), kv_ch, prefix});
        slots.push_back({"tfm_v_" + std::to_string(il), kv_ch, prefix});
    }

    if (c2w.pre_conv_w) {
        slots.push_back({"pre_conv", c2w.pre_conv_w->ne[0] - 1, c2w.pre_conv_w->ne[1]});
    }

    for (size_t il = 0; il < c2w.upsample.size(); il++) {
        const auto & up = c2w.upsample[il];
        slots.push_back({"up" + std::to_string(il) + "_dwconv", up.dwconv_w->ne[0] - 1, up.dwconv_w->ne[2]});
    }

    slots.push_back({"dac_entry", c2w.dac_entry_w->ne[0] - 1, c2w.dac_entry_w->ne[1]});

    static constexpr int DAC_DILATIONS[3] = { 1, 3, 9 };
    for (size_t il = 0; il < c2w.dac.size(); il++) {
        const auto & blk = c2w.dac[il];
        const int64_t stride = blk.conv_w->ne[0] / 2;
        const std::string blk_name = "dac" + std::to_string(il);
        slots.push_back({blk_name + "_tail", stride, blk.conv_w->ne[1]});
        for (size_t ir = 0; ir < blk.res.size(); ir++) {
            const auto & res = blk.res[ir];
            slots.push_back({blk_name + "_res" + std::to_string(ir),
                              (res.conv1_w->ne[0] - 1) * DAC_DILATIONS[ir], res.conv1_w->ne[1]});
        }
    }

    slots.push_back({"dac_post_conv", c2w.dac_post_conv_w->ne[0] - 1, c2w.dac_post_conv_w->ne[1]});

    return slots;
}

ggml_cgraph * mtmd_graph_qwen3tts_gen::build() {
    GGML_ASSERT(n_batch == 1);

    int idx;
    switch (gen_process) {
        case MTMD_SUBCTX_GEN_PROCESS_GEN_CODE: idx = 0; break;
        case MTMD_SUBCTX_GEN_PROCESS_GEN_WAV:  idx = 1; break;
        default: GGML_ABORT("unknown gen_process");
    }

    // ---- MTMD_SUBCTX_GEN_PROCESS_GEN_CODE: backbone hidden state -> 16 RVQ codes + next-step embd ----
    ggml_tensor * h_state = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_mmproj_embd);
    ggml_set_name(h_state, "inp_raw");
    ggml_set_input(h_state);

    ggml_tensor * code0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
    ggml_set_name(code0, "inp_code0");
    ggml_set_input(code0);

    ggml_tensor * code0_embd = ggml_get_rows(ctx0, model.gen_code_out_embd_w, code0);
    code0_embd = ggml_reshape_1d(ctx0, code0_embd, code0_embd->ne[0]);
    cb(code0_embd, "code0_embd", -1);

    const int64_t n_acoustic = model.n_acoustic_codebooks();
    const int     n_codes    = (int) n_acoustic + 1;
    const int64_t n_kv_pad   = n_codes;
    const int     n_layer    = (int) model.layers.size();
    const int     n_head     = hparams.n_head;
    const int     n_head_kv  = hparams.n_head_kv;
    const int64_t d_head     = model.layers[0].q_w->ne[1] / n_head;

    std::vector<ggml_tensor *> k_cache(n_layer), v_cache(n_layer);
    for (int il = 0; il < n_layer; il++) {
        k_cache[il] = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_head * n_head_kv, n_kv_pad), 0.0f);
        v_cache[il] = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_head * n_head_kv, n_kv_pad), 0.0f);
    }

    code_gen cg(*this, top_k, top_p);

    ggml_tensor * out_code_cache = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_codes);
    out_code_cache = cg.cache_set(out_code_cache, 0, code0);

    ggml_tensor * inp_rand0 = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(inp_rand0, "inp_rand_0");
    ggml_set_input(inp_rand0);

    cg.prefill(k_cache, v_cache, out_code_cache, h_state, code0_embd, inp_rand0);

    for (int g = 1; g < n_acoustic; g++) {
        ggml_tensor * inp_rand = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
        ggml_set_name(inp_rand, ("inp_rand_" + std::to_string(g)).c_str());
        ggml_set_input(inp_rand);
        out_code_cache = cg.step(k_cache, v_cache, out_code_cache, inp_rand, g);
    }

    ggml_tensor * out_codes = ggml_cont(ctx0, out_code_cache);
    ggml_set_name(out_codes, "out_codes");
    ggml_set_output(out_codes);

    ggml_tensor * out_embd = code0_embd;
    for (int g = 1; g <= n_acoustic; g++) {
        ggml_tensor * code_g = ggml_view_1d(ctx0, out_code_cache, 1, (size_t) g * out_code_cache->nb[1]);

        ggml_tensor * embd_g = ggml_view_2d(ctx0, model.gen_code_embd_w, model.gen_code_embd_w->ne[0], model.gen_code_embd_w->ne[1],
                                            model.gen_code_embd_w->nb[1], (size_t) (g - 1) * model.gen_code_embd_w->nb[2]);
        ggml_tensor * e = ggml_get_rows(ctx0, embd_g, code_g);
        e = ggml_reshape_1d(ctx0, e, e->ne[0]);

        out_embd = ggml_add(ctx0, out_embd, e);
    }
    out_embd = ggml_reshape_2d(ctx0, out_embd, out_embd->ne[0], 1);
    cb(out_embd, "gen_audio_out", -1);

    // ---- MTMD_SUBCTX_GEN_PROCESS_GEN_WAV: 16 RVQ codes -> raw PCM ----
    const int n_frames = hparams.wav_tfm_swa;

    ggml_tensor * inp_codes = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_frames, n_codes);
    ggml_set_name(inp_codes, "inp_codes");
    ggml_set_input(inp_codes);

    code2wav c2w(*this);
    for (const auto & slot : list_c2w_state_slots(hparams, model)) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, slot.ne0, slot.ne1);
        ggml_set_name(t, ("state_in_" + slot.name).c_str());
        ggml_set_input(t);
        c2w.state_in[slot.name] = t;
    }

    ggml_tensor * out_audio = c2w.decode(inp_codes);
    ggml_set_name(out_audio, "out_audio");
    ggml_set_output(out_audio);

    for (auto & slot : c2w.state_out) {
        ggml_set_name(slot.second, ("state_out_" + slot.first).c_str());
        ggml_set_output(slot.second);
    }

    ggml_tensor * outs[2];
    outs[0] = out_codes; outs[1] = out_audio;
    ggml_build_forward_select(gf, outs, 2, idx);
    for (auto & slot : c2w.state_out) {
        outs[0] = out_codes; outs[1] = slot.second;
        ggml_build_forward_select(gf, outs, 2, idx);
    }
    outs[0] = out_embd; outs[1] = out_audio;
    ggml_build_forward_select(gf, outs, 2, idx);

    return gf;
}
