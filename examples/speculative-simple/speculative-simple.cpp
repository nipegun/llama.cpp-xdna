#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // llama.cpp-xdna: optional --runs N, --prompt-cycle and --no-xdna flags for clean CPU vs XDNA MTP benchmarking
    int  cli_runs         = 0;
    bool cli_prompt_cycle = false;
    bool cli_no_xdna      = false;
    std::vector<char *> filtered_argv;
    filtered_argv.reserve(argc);
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            cli_runs = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--prompt-cycle") == 0) {
            cli_prompt_cycle = true;
        } else if (std::strcmp(argv[i], "--no-xdna") == 0) {
            cli_no_xdna = true;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    if (cli_no_xdna) {
#if defined(_WIN32)
        _putenv_s("GGML_XDNA_DISABLE", "1");
#else
        setenv("GGML_XDNA_DISABLE", "1", 1);
#endif
    }
    int    f_argc = (int) filtered_argv.size();
    char ** f_argv = filtered_argv.data();

    common_params params;

    common_init();

    if (!common_params_parse(f_argc, f_argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    const auto output_limits = common_speculative_get_output_limits(
            params.n_batch, params.n_parallel, common_speculative_n_max(&params.speculative));
    params.n_outputs_max = output_limits.total;
    params.n_outputs_max_per_seq = output_limits.per_seq;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model (if any) - this also creates the MTP draft context when MTP speculation is enabled
    common_speculative_init_result_ptr spec_init;

    {
        common_params params_dft = common_base_params_to_speculative(params);

        spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = spec_init->context();
    }

    llama_context * ctx_dft = params.speculative.draft.ctx_dft;

    // check if the context supports partial sequence removal
    const bool use_ckpt_tgt = common_context_can_seq_rm(ctx_tgt) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    const bool use_ckpt_dft = ctx_dft && common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

    if (use_ckpt_tgt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    // Tokenize the prompt
    if (params.prompt.empty() && cli_prompt_cycle) {
        params.prompt = "Explain the architecture of a modern RISC-V out-of-order superscalar processor, including branch prediction, register renaming, reservation stations, and the reorder buffer:";
    }
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);
    if (inp.empty()) {
        inp.push_back(llama_vocab_bos(vocab) != LLAMA_TOKEN_NULL ? llama_vocab_bos(vocab) : 0);
    }

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    llama_seq_id seq_id = 0;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto & params_spec = params.speculative;
    const bool   do_warmup   = getenv("LLAMA_SPEC_NO_WARMUP") == nullptr || atoi(getenv("LLAMA_SPEC_NO_WARMUP")) == 0;
    const int    n_runs      = std::max(1, cli_runs > 0 ? cli_runs : (getenv("LLAMA_SPEC_RUNS") != nullptr ? atoi(getenv("LLAMA_SPEC_RUNS")) : 1));

    static const char * const k_cycle_prompts[5] = {
        "Explain the architecture of a modern RISC-V out-of-order superscalar processor, including branch prediction, register renaming, reservation stations, and the reorder buffer:",
        "Derive the thermodynamic efficiency of an ideal Rankine steam cycle with superheat and regeneration, explaining entropy changes across the turbine, condenser, and feedwater pump:",
        "Describe how Raft consensus guarantees linearizable state machine replication across leader election, log replication, safety invariants, and network partition recovery:",
        "Explain the mechanism, stereochemistry, and kinetic rate law of bimolecular nucleophilic substitution (SN2) versus unimolecular elimination (E1) in polar aprotic solvents:",
        "Prove Euler's totient theorem using Lagrange's theorem on the multiplicative group of integers modulo n, and show how RSA public-key encryption applies it:",
    };
    std::vector<double> run_dec_tps;
    std::vector<double> run_acc_pct;

    // llama.cpp-xdna: run a 1-step warmup pass by default so lazy NPU weight packing (dMoeMatrices / dGemvMatrices,
    // including blk.<n_layer> and head_part) happens outside the timed decode loop, matching llama-bench's methodology
    auto run_once = [&](bool warmup_pass, int run_idx) -> int {
        const std::vector<llama_token> run_inp = (!warmup_pass && cli_prompt_cycle) ?
            common_tokenize(ctx_tgt, k_cycle_prompts[run_idx % 5], true, true) :
            inp;

        llama_memory_clear(llama_get_memory(ctx_tgt), true);
        if (ctx_dft) {
            llama_memory_clear(llama_get_memory(ctx_dft), true);
        }
        llama_perf_context_reset(ctx_tgt);
        if (ctx_dft) {
            llama_perf_context_reset(ctx_dft);
        }

        common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));
        struct common_speculative * spec = common_speculative_init(params.speculative, 1);
        if (spec == nullptr && params.speculative.mtp != 0) {
            LOG_ERR("%s", "failed to initialize speculative decoding\n");
            return 1;
        }

        n_predict = 0;
        n_drafted = 0;
        n_accept  = 0;
        has_eos   = false;

        const size_t prompt_len = warmup_pass ? std::min<size_t>(run_inp.size(), 4) : run_inp.size();
        const int    target_gen = warmup_pass ? 1 : params.n_predict;

        if (!warmup_pass && n_runs > 1) {
            LOG("\n\n=== Run %d / %d ===\n", run_idx + 1, n_runs);
        }

        const auto t_enc_start = ggml_time_us();

        if (prompt_len > 1) {
            common_batch batch_prompt(ctx_tgt);
            for (size_t i = 0; i < prompt_len - 1; ++i) {
                batch_prompt.add(run_inp[i], i, seq_id, false);
            }

            llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_prompt.get());

            if (!common_speculative_process(spec, batch_prompt)) {
                LOG_ERR("%s", "failed to process speculative prompt\n");
                common_speculative_free(spec);
                return 1;
            }
        }

        llama_token id_last = run_inp[prompt_len - 1];
        llama_tokens prompt_tgt(run_inp.begin(), run_inp.begin() + (prompt_len - 1));
        prompt_tgt.reserve(llama_n_ctx(ctx_tgt));
        int n_past = (int) prompt_len - 1;

        common_speculative_begin(spec, seq_id, prompt_tgt);
        llama_synchronize(ctx_tgt);
        if (ctx_dft) {
            llama_synchronize(ctx_dft);
        }

        common_batch batch_tgt(ctx_tgt);
        llama_tokens draft;
        common_prompt_checkpoint ckpt;

        const auto t_enc_end   = ggml_time_us();
        const auto t_dec_start = ggml_time_us();

        while (true) {
            if (draft.empty()) {
                ckpt.update_pos(
                        prompt_tgt.size(),
                        llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                        llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

                if (use_ckpt_dft) {
                    ckpt.update_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }

                int n_draft_max = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
                if (!warmup_pass && target_gen >= 0) {
                    n_draft_max = std::min(n_draft_max, target_gen - n_predict - 1);
                }
                n_draft_max = std::max(n_draft_max, 0);

                if (spec && n_draft_max > 0) {
                    common_speculative_get_draft_params(spec, seq_id) = {
                        /* .drafting   = */ true,
                        /* .n_max      = */ n_draft_max,
                        /* .pos0       = */ n_past,
                        /* .id_last    = */ id_last,
                        /* .prompt     = */ &prompt_tgt,
                        /* .result     = */ &draft,
                    };
                    common_speculative_draft(spec);
                }

                if (!draft.empty()) {
                    if (use_ckpt_tgt) {
                        ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                    }
                }

                if (ctx_dft) {
                    if (use_ckpt_dft) {
                        GGML_ASSERT(ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                    }

                    llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
                }
            } else {
                if (use_ckpt_tgt) {
                    GGML_ASSERT(!ckpt.empty());
                }
            }

            batch_tgt.clear();
            batch_tgt.add(id_last, n_past++, seq_id, true);

            for (size_t i = 0; i < draft.size(); ++i) {
                batch_tgt.add(draft[i], n_past + i, seq_id, true);
            }
            llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch_tgt.get());

            if (!common_speculative_process(spec, batch_tgt)) {
                LOG_ERR("%s", "failed to process speculative batch\n");
                break;
            }

            common_sampler_ptr smpl_save;
            if (use_ckpt_tgt) {
                smpl_save.reset(common_sampler_clone(smpl.get()));
            }

            const size_t n_draft = draft.size();
            auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);
            GGML_ASSERT(ids.size() > 0);

            if (use_ckpt_tgt && ids.size() - 1 < n_draft) {
                LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, n_draft);
                draft = std::move(ids);
                GGML_ASSERT(ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
                if (ctx_dft) {
                    GGML_ASSERT(ckpt.load_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, ckpt.pos_max + 1, -1);
                }
                prompt_tgt.resize(ckpt.n_tokens);
                smpl   = std::move(smpl_save);
                n_past = (int) prompt_tgt.size();
                continue;
            }

            if (spec) {
                common_speculative_accept(spec, seq_id, ids.size() - 1);
            }

            n_past    += ids.size() - 1;
            n_drafted += n_draft;
            n_accept  += ids.size() - 1;
            n_predict += ids.size();

            for (size_t i = 0; i < ids.size(); ++i) {
                prompt_tgt.push_back(id_last);
                id_last = ids[i];
                if (llama_vocab_is_eog(vocab, id_last)) {
                    has_eos = true;
                    break;
                }
                if (!warmup_pass) {
                    const std::string token_str = common_token_to_piece(ctx_tgt, id_last);
                    if (params.use_color && i + 1 < ids.size()) {
                        LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
                    } else {
                        LOG("%s", token_str.c_str());
                    }
                }
            }

            draft.clear();
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);
            if (ctx_dft) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
            }

            if ((target_gen >= 0 && n_predict >= target_gen) || has_eos) {
                break;
            }
        }

        llama_synchronize(ctx_tgt);
        if (ctx_dft) {
            llama_synchronize(ctx_dft);
        }
        const auto t_dec_end = ggml_time_us();

        if (!warmup_pass) {
            const int    n_input = (int) prompt_len;
            const double dec_sec = (t_dec_end - t_dec_start) / 1e6;
            const double dec_tps = dec_sec > 0.0 ? n_predict / dec_sec : 0.0;
            const double acc_pct = n_drafted > 0 ? 100.0 * n_accept / n_drafted : 0.0;
            run_dec_tps.push_back(dec_tps);
            run_acc_pct.push_back(acc_pct);

            LOG("\n\n");
            LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, prompt_len / ((t_enc_end - t_enc_start) / 1e6f));
            LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, dec_sec, dec_tps);

            LOG_INF("\n");
            LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
            LOG_INF("n_predict = %d\n", n_predict);
            LOG_INF("n_drafted = %d\n", n_drafted);
            LOG_INF("n_accept  = %d\n", n_accept);
            LOG_INF("accept    = %.3f%%\n", acc_pct);

            LOG_INF("\n");
            LOG_INF("draft:\n\n");
            common_speculative_print_stats(spec);

            LOG_INF("\n");
            LOG_INF("target:\n\n");
            common_perf_print(ctx_tgt, smpl.get());
        }

        common_speculative_free(spec);
        return 0;
    };

    if (do_warmup) {
        if (run_once(true, -1) != 0) {
            llama_backend_free();
            return 1;
        }
    }

    for (int r = 0; r < n_runs; ++r) {
        if (run_once(false, r) != 0) {
            llama_backend_free();
            return 1;
        }
    }

    if (run_dec_tps.size() > 1) {
        double sum_all = 0.0, sum_warm = 0.0, sum_acc = 0.0;
        for (size_t i = 0; i < run_dec_tps.size(); ++i) {
            sum_all += run_dec_tps[i];
            sum_acc += run_acc_pct[i];
            if (i >= 1) {
                sum_warm += run_dec_tps[i];
            }
        }
        LOG_INF("\nsummary (%zu runs): mean_all = %.3f t/s, mean_runs_2..%zu = %.3f t/s, mean_accept = %.2f%%\n",
                run_dec_tps.size(),
                sum_all  / run_dec_tps.size(),
                run_dec_tps.size(),
                sum_warm / (run_dec_tps.size() - 1),
                sum_acc  / run_acc_pct.size());
    }

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
