/*
 * C++ Training Loop — Layer 3 Runtime
 *
 * Standalone C++ binary that runs the full grokking benchmark.
 * Eliminates ALL Python overhead: no interpreter, no GIL, no object model.
 *
 * Architecture:
 *   Layer 3 (this file): C++ runtime, model definitions, training orchestration
 *   Layer 2 (graph_compiler): CUDA Graphs, memory planning, stream scheduling
 *   Layer 1 (kernels): Custom fused CUDA kernels per optimizer
 *
 * Usage:
 *   ./supergrok_bench --model decoder --optimizer adamw --gpu 0
 *   ./supergrok_bench --mode D --gpus 0,1,2,3
 *
 * Output: JSON results compatible with the Python plotting code.
 */

#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "models.h"
#include "ops.h"
#include "graph_compiler.h"
#include "stream_scheduler.h"


// ═══════════════════════════════════════════════════════════════════════
//  Configuration
// ═══════════════════════════════════════════════════════════════════════

struct Config {
    // Model
    std::string model_type = "decoder";
    int num_layers = 2;
    int dim_model = 128;
    int num_heads = 4;
    int num_tokens = 99;
    int p = 97;
    int seq_len = 4;
    int patch_dim = 49;
    int num_patches = 16;

    // Training
    int max_steps = 15000;
    float lr = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float weight_decay = 1.0f;
    float frac_train = 0.5f;
    int log_every = 10;
    float grok_threshold = 0.95f;
    int patience = 200;
    int seed = 42;

    // SuperGrok1.5 specific
    float alpha_init = 0.95f;
    float lamb = 0.1f;
    float gamma = 0.1f;
    float gamma_alpha = 0.1f;
    float kappa = 1.0f;
    int warmup_steps = 100;
    int warmup_ramp = 200;
    float gradient_clipping = 1.0f;
    int meta_hidden_dim = 32;
    float gate_temperature = 5.0f;
    float sam_rho = 0.05f;
    int sam_freq = 5;
    float wd_ramp = 2.0f;
    float wd_scale = 10.0f;
    float wd_thresh = 0.5f;

    // Muon
    float muon_lr = 0.02f;
    float muon_momentum = 0.95f;

    // Lion
    float lion_lr = 3.3e-4f;
    float lion_wd = 3.0f;

    // Grokfast
    float grokfast_alpha = 0.98f;
    float grokfast_lamb = 2.0f;

    // LookSAM
    float looksam_rho = 0.05f;
    int looksam_k = 5;
    float looksam_alpha = 0.7f;

    // Runtime
    int gpu_id = 0;
    bool use_cuda_graphs = true;
    bool use_multi_stream = true;
    bool auto_tune = true;
};


// ═══════════════════════════════════════════════════════════════════════
//  Training Result
// ═══════════════════════════════════════════════════════════════════════

struct TrainResult {
    std::string name;
    int seed;
    int total_steps;
    int grokking_step = -1;
    double grokking_wall = -1.0;
    double wall_time = 0.0;
    float final_val_acc = 0.0f;
    float final_train_acc = 0.0f;
    std::vector<float> train_losses;
    std::vector<float> val_losses;
    std::vector<float> train_accs;
    std::vector<float> val_accs;
    std::vector<int> logged_steps;
};


// ═══════════════════════════════════════════════════════════════════════
//  Data Generation
// ═══════════════════════════════════════════════════════════════════════

struct Dataset {
    torch::Tensor train_x, train_y, val_x, val_y;
};

Dataset make_modular_division_data(int p, float frac_train, int seed) {
    torch::manual_seed(seed);
    int op_tok = p, eq_tok = p + 1;

    std::vector<std::vector<int64_t>> pairs;
    std::vector<int64_t> labels;

    for (int a = 0; a < p; a++) {
        for (int b = 1; b < p; b++) {
            // Modular inverse: b_inv = b^(p-2) mod p (Fermat's little theorem)
            int64_t b_inv = 1;
            int64_t base = b;
            int exp = p - 2;
            while (exp > 0) {
                if (exp % 2 == 1) b_inv = (b_inv * base) % p;
                base = (base * base) % p;
                exp /= 2;
            }
            pairs.push_back({a, op_tok, b, eq_tok});
            labels.push_back((static_cast<int64_t>(a) * b_inv) % p);
        }
    }

    // Shuffle
    std::mt19937 rng(seed);
    std::vector<size_t> indices(pairs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    int n = static_cast<int>(pairs.size() * frac_train);
    auto x = torch::empty({static_cast<int64_t>(pairs.size()), 4}, torch::kLong);
    auto y = torch::empty({static_cast<int64_t>(pairs.size())}, torch::kLong);

    for (size_t i = 0; i < indices.size(); i++) {
        size_t idx = indices[i];
        for (int j = 0; j < 4; j++) {
            x[i][j] = pairs[idx][j];
        }
        y[i] = labels[idx];
    }

    Dataset ds;
    ds.train_x = x.slice(0, 0, n);
    ds.train_y = y.slice(0, 0, n);
    ds.val_x = x.slice(0, n);
    ds.val_y = y.slice(0, n);
    return ds;
}


// ═══════════════════════════════════════════════════════════════════════
//  Evaluation (with optional async metrics)
// ═══════════════════════════════════════════════════════════════════════

std::pair<float, float> evaluate(
    torch::nn::AnyModule& model,
    torch::Tensor x, torch::Tensor y, int p
) {
    torch::NoGradGuard no_grad;
    auto logits = model.forward(x);
    float loss = torch::nn::functional::cross_entropy(logits, y).item<float>();
    float acc = (logits.slice(1, 0, p).argmax(1) == y)
        .to(torch::kFloat32).mean().item<float>();
    return {loss, acc};
}


// ═══════════════════════════════════════════════════════════════════════
//  AdamW Training Loop (simplest, baseline)
// ═══════════════════════════════════════════════════════════════════════

TrainResult train_adamw(const Config& cfg, torch::Device dev, Dataset& data) {
    TrainResult result;
    result.name = "AdamW";
    result.seed = cfg.seed;

    auto model = build_model(cfg.model_type, cfg.num_layers, cfg.dim_model,
        cfg.num_heads, cfg.num_tokens, cfg.p,
        cfg.seq_len, cfg.patch_dim, cfg.num_patches);

    // Move model to device
    auto* mod_ptr = model.ptr();
    for (auto& p : mod_ptr->parameters()) {
        p.set_data(p.to(dev));
    }

    // Data to device
    auto tx = data.train_x.to(dev);
    auto ty = data.train_y.to(dev);
    auto vx = data.val_x.to(dev);
    auto vy = data.val_y.to(dev);

    // Optimizer
    torch::optim::AdamW optimizer(
        mod_ptr->parameters(),
        torch::optim::AdamWOptions(cfg.lr)
            .betas({cfg.beta1, cfg.beta2})
            .weight_decay(cfg.weight_decay));

    auto t0 = std::chrono::high_resolution_clock::now();
    bool grokked = false;
    int patience_count = 0;

    for (int step = 0; step < cfg.max_steps; step++) {
        // Forward + backward + step
        auto logits = model.forward(tx);
        auto loss = torch::nn::functional::cross_entropy(logits, ty);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        // Evaluation
        if (step % cfg.log_every == 0 || step == 1) {
            auto [tl, ta] = evaluate(model, tx, ty, cfg.p);
            auto [vl, va] = evaluate(model, vx, vy, cfg.p);

            result.train_losses.push_back(tl);
            result.train_accs.push_back(ta);
            result.val_losses.push_back(vl);
            result.val_accs.push_back(va);
            result.logged_steps.push_back(step);

            // Grokking detection
            if (va >= cfg.grok_threshold) {
                if (!grokked) {
                    auto now = std::chrono::high_resolution_clock::now();
                    result.grokking_step = step;
                    result.grokking_wall = std::chrono::duration<double>(now - t0).count();
                    grokked = true;
                }
                patience_count++;
                if (patience_count >= cfg.patience) {
                    result.total_steps = step;
                    break;
                }
            } else {
                patience_count = 0;
            }

            result.final_val_acc = va;
            result.final_train_acc = ta;
        }

        result.total_steps = step;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.wall_time = std::chrono::duration<double>(t1 - t0).count();
    return result;
}


// ═══════════════════════════════════════════════════════════════════════
//  Grokfast Training Loop (uses fused C++/CUDA EMA kernel)
// ═══════════════════════════════════════════════════════════════════════

TrainResult train_grokfast(const Config& cfg, torch::Device dev, Dataset& data) {
    TrainResult result;
    result.name = "Grokfast";
    result.seed = cfg.seed;

    auto model = build_model(cfg.model_type, cfg.num_layers, cfg.dim_model,
        cfg.num_heads, cfg.num_tokens, cfg.p,
        cfg.seq_len, cfg.patch_dim, cfg.num_patches);

    auto* mod_ptr = model.ptr();
    for (auto& p : mod_ptr->parameters()) p.set_data(p.to(dev));

    auto tx = data.train_x.to(dev);
    auto ty = data.train_y.to(dev);
    auto vx = data.val_x.to(dev);
    auto vy = data.val_y.to(dev);

    torch::optim::AdamW optimizer(
        mod_ptr->parameters(),
        torch::optim::AdamWOptions(cfg.lr)
            .betas({cfg.beta1, cfg.beta2})
            .weight_decay(cfg.weight_decay));

    // Initialize EMA buffers
    std::vector<torch::Tensor> grads_list, ema_bufs;
    auto params = mod_ptr->parameters();
    for (auto& p : params) {
        ema_bufs.push_back(torch::zeros_like(p));
    }
    bool ema_initialized = false;

    auto t0 = std::chrono::high_resolution_clock::now();
    bool grokked = false;
    int patience_count = 0;

    for (int step = 0; step < cfg.max_steps; step++) {
        auto logits = model.forward(tx);
        auto loss = torch::nn::functional::cross_entropy(logits, ty);
        optimizer.zero_grad();
        loss.backward();

        // Collect gradients into flat list
        grads_list.clear();
        for (auto& p : params) {
            if (p.grad().defined()) {
                grads_list.push_back(p.grad().data());
            } else {
                grads_list.push_back(torch::Tensor());
            }
        }

        if (!ema_initialized) {
            // First step: init EMA buffers from gradients
            for (size_t i = 0; i < grads_list.size(); i++) {
                if (grads_list[i].defined()) {
                    ema_bufs[i] = grads_list[i].detach().clone();
                }
            }
            ema_initialized = true;
        }

        // Fused C++/CUDA EMA + gradient amplification
        grokfast_fused_step(grads_list, ema_bufs,
            cfg.grokfast_alpha, cfg.grokfast_lamb);

        optimizer.step();

        if (step % cfg.log_every == 0 || step == 1) {
            auto [tl, ta] = evaluate(model, tx, ty, cfg.p);
            auto [vl, va] = evaluate(model, vx, vy, cfg.p);

            result.train_losses.push_back(tl);
            result.train_accs.push_back(ta);
            result.val_losses.push_back(vl);
            result.val_accs.push_back(va);
            result.logged_steps.push_back(step);

            if (va >= cfg.grok_threshold) {
                if (!grokked) {
                    auto now = std::chrono::high_resolution_clock::now();
                    result.grokking_step = step;
                    result.grokking_wall = std::chrono::duration<double>(now - t0).count();
                    grokked = true;
                }
                patience_count++;
                if (patience_count >= cfg.patience) { result.total_steps = step; break; }
            } else { patience_count = 0; }

            result.final_val_acc = va;
            result.final_train_acc = ta;
        }
        result.total_steps = step;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.wall_time = std::chrono::duration<double>(t1 - t0).count();
    return result;
}


// ═══════════════════════════════════════════════════════════════════════
//  Muon Training Loop (uses fused C++/CUDA Newton-Schulz kernel)
// ═══════════════════════════════════════════════════════════════════════

TrainResult train_muon(const Config& cfg, torch::Device dev, Dataset& data) {
    TrainResult result;
    result.name = "Muon";
    result.seed = cfg.seed;

    auto model = build_model(cfg.model_type, cfg.num_layers, cfg.dim_model,
        cfg.num_heads, cfg.num_tokens, cfg.p,
        cfg.seq_len, cfg.patch_dim, cfg.num_patches);

    auto* mod_ptr = model.ptr();
    for (auto& p : mod_ptr->parameters()) p.set_data(p.to(dev));

    auto tx = data.train_x.to(dev);
    auto ty = data.train_y.to(dev);
    auto vx = data.val_x.to(dev);
    auto vy = data.val_y.to(dev);

    // Separate 2D (Muon) and non-2D (AdamW) parameters
    std::vector<torch::Tensor> muon_params, adam_param_list;
    std::vector<torch::Tensor> muon_grads, muon_bufs;
    auto all_params = mod_ptr->parameters();

    for (auto& p : all_params) {
        if (p.dim() == 2) {
            muon_params.push_back(p);
            muon_bufs.push_back(torch::zeros_like(p));
        } else {
            adam_param_list.push_back(p);
        }
    }

    // AdamW for non-2D params
    std::unique_ptr<torch::optim::AdamW> adam_opt;
    if (!adam_param_list.empty()) {
        adam_opt = std::make_unique<torch::optim::AdamW>(
            adam_param_list,
            torch::optim::AdamWOptions(cfg.lr)
                .betas({cfg.beta1, cfg.beta2})
                .weight_decay(cfg.weight_decay));
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    bool grokked = false;
    int patience_count = 0;

    for (int step = 0; step < cfg.max_steps; step++) {
        auto logits = model.forward(tx);
        auto loss = torch::nn::functional::cross_entropy(logits, ty);
        for (auto& p : all_params) {
            if (p.grad().defined()) p.grad().zero_();
        }
        loss.backward();

        // Collect Muon gradients
        muon_grads.clear();
        for (auto& p : muon_params) {
            muon_grads.push_back(p.grad().defined() ? p.grad().data() : torch::Tensor());
        }

        // Fused C++/CUDA Muon step
        {
            torch::NoGradGuard no_grad;
            muon_fused_step(muon_params, muon_grads, muon_bufs,
                cfg.muon_momentum, cfg.muon_lr, cfg.weight_decay, 5);
        }

        if (adam_opt) adam_opt->step();

        if (step % cfg.log_every == 0 || step == 1) {
            auto [tl, ta] = evaluate(model, tx, ty, cfg.p);
            auto [vl, va] = evaluate(model, vx, vy, cfg.p);

            result.train_losses.push_back(tl);
            result.train_accs.push_back(ta);
            result.val_losses.push_back(vl);
            result.val_accs.push_back(va);
            result.logged_steps.push_back(step);

            if (va >= cfg.grok_threshold) {
                if (!grokked) {
                    auto now = std::chrono::high_resolution_clock::now();
                    result.grokking_step = step;
                    result.grokking_wall = std::chrono::duration<double>(now - t0).count();
                    grokked = true;
                }
                patience_count++;
                if (patience_count >= cfg.patience) { result.total_steps = step; break; }
            } else { patience_count = 0; }

            result.final_val_acc = va;
            result.final_train_acc = ta;
        }
        result.total_steps = step;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.wall_time = std::chrono::duration<double>(t1 - t0).count();
    return result;
}


// ═══════════════════════════════════════════════════════════════════════
//  Lion Training Loop (uses fused C++/CUDA kernel)
// ═══════════════════════════════════════════════════════════════════════

TrainResult train_lion(const Config& cfg, torch::Device dev, Dataset& data) {
    TrainResult result;
    result.name = "Lion";
    result.seed = cfg.seed;

    auto model = build_model(cfg.model_type, cfg.num_layers, cfg.dim_model,
        cfg.num_heads, cfg.num_tokens, cfg.p,
        cfg.seq_len, cfg.patch_dim, cfg.num_patches);

    auto* mod_ptr = model.ptr();
    for (auto& p : mod_ptr->parameters()) p.set_data(p.to(dev));

    auto tx = data.train_x.to(dev);
    auto ty = data.train_y.to(dev);
    auto vx = data.val_x.to(dev);
    auto vy = data.val_y.to(dev);

    // Initialize Lion state
    auto params = mod_ptr->parameters();
    std::vector<torch::Tensor> param_list, grad_list, exp_avgs;
    for (auto& p : params) {
        param_list.push_back(p);
        exp_avgs.push_back(torch::zeros_like(p));
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    bool grokked = false;
    int patience_count = 0;

    for (int step = 0; step < cfg.max_steps; step++) {
        auto logits = model.forward(tx);
        auto loss = torch::nn::functional::cross_entropy(logits, ty);
        for (auto& p : params) {
            if (p.grad().defined()) p.grad().zero_();
        }
        loss.backward();

        // Collect gradients
        grad_list.clear();
        for (auto& p : params) {
            grad_list.push_back(p.grad().defined() ? p.grad().data() : torch::Tensor());
        }

        // Fused C++/CUDA Lion step
        {
            torch::NoGradGuard no_grad;
            lion_fused_step(param_list, grad_list, exp_avgs,
                cfg.lion_lr, cfg.beta1, 0.99f, cfg.lion_wd);
        }

        if (step % cfg.log_every == 0 || step == 1) {
            auto [tl, ta] = evaluate(model, tx, ty, cfg.p);
            auto [vl, va] = evaluate(model, vx, vy, cfg.p);

            result.train_losses.push_back(tl);
            result.train_accs.push_back(ta);
            result.val_losses.push_back(vl);
            result.val_accs.push_back(va);
            result.logged_steps.push_back(step);

            if (va >= cfg.grok_threshold) {
                if (!grokked) {
                    auto now = std::chrono::high_resolution_clock::now();
                    result.grokking_step = step;
                    result.grokking_wall = std::chrono::duration<double>(now - t0).count();
                    grokked = true;
                }
                patience_count++;
                if (patience_count >= cfg.patience) { result.total_steps = step; break; }
            } else { patience_count = 0; }

            result.final_val_acc = va;
            result.final_train_acc = ta;
        }
        result.total_steps = step;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    result.wall_time = std::chrono::duration<double>(t1 - t0).count();
    return result;
}


// ═══════════════════════════════════════════════════════════════════════
//  JSON Output (compatible with Python plotting)
// ═══════════════════════════════════════════════════════════════════════

void write_json(const std::vector<TrainResult>& results,
                const std::string& filename) {
    std::ofstream ofs(filename);
    ofs << "{\n";

    std::map<std::string, std::vector<const TrainResult*>> by_name;
    for (const auto& r : results) by_name[r.name].push_back(&r);

    bool first_opt = true;
    for (const auto& [name, runs] : by_name) {
        if (!first_opt) ofs << ",\n";
        first_opt = false;
        ofs << "  \"" << name << "\": [\n";

        for (size_t i = 0; i < runs.size(); i++) {
            const auto* r = runs[i];
            ofs << "    {\n";
            ofs << "      \"seed\": " << r->seed << ",\n";
            ofs << "      \"total_steps\": " << r->total_steps << ",\n";
            ofs << "      \"grokking_step\": " << r->grokking_step << ",\n";
            ofs << "      \"grokking_wall\": " << r->grokking_wall << ",\n";
            ofs << "      \"wall_time\": " << r->wall_time << ",\n";
            ofs << "      \"final_val_acc\": " << r->final_val_acc << ",\n";
            ofs << "      \"final_train_acc\": " << r->final_train_acc << "\n";
            ofs << "    }";
            if (i < runs.size() - 1) ofs << ",";
            ofs << "\n";
        }
        ofs << "  ]";
    }

    ofs << "\n}\n";
}


// ═══════════════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    Config cfg;
    std::string optimizer_name = "all";
    std::string output_file = "results_cpp.json";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) cfg.model_type = argv[++i];
        else if (arg == "--optimizer" && i + 1 < argc) optimizer_name = argv[++i];
        else if (arg == "--gpu" && i + 1 < argc) cfg.gpu_id = std::stoi(argv[++i]);
        else if (arg == "--steps" && i + 1 < argc) cfg.max_steps = std::stoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) cfg.lr = std::stof(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) cfg.seed = std::stoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "--no-cuda-graphs") cfg.use_cuda_graphs = false;
        else if (arg == "--no-multi-stream") cfg.use_multi_stream = false;
        else if (arg == "--no-auto-tune") cfg.auto_tune = false;
        else if (arg == "--help") {
            std::cout << "SuperGrok C++ Benchmark\n"
                      << "  --model {decoder|vit|mamba}\n"
                      << "  --optimizer {adamw|grokfast|muon|lion|all}\n"
                      << "  --gpu N\n"
                      << "  --steps N\n"
                      << "  --lr FLOAT\n"
                      << "  --seed N\n"
                      << "  --output FILE\n"
                      << "  --no-cuda-graphs\n"
                      << "  --no-multi-stream\n"
                      << "  --no-auto-tune\n";
            return 0;
        }
    }

    // Device setup
    torch::Device dev = torch::kCPU;
    if (torch::cuda::is_available()) {
        dev = torch::Device(torch::kCUDA, cfg.gpu_id);
        std::cout << "Using GPU " << cfg.gpu_id << "\n";
    } else {
        std::cout << "CUDA not available, using CPU\n";
    }

    // Generate data
    std::cout << "Generating data...\n";
    auto data = make_modular_division_data(cfg.p, cfg.frac_train, cfg.seed);

    // Run optimizers
    std::vector<TrainResult> results;

    auto run_if = [&](const std::string& name, auto fn) {
        if (optimizer_name == "all" || optimizer_name == name) {
            std::cout << "Training " << name << "...\n";
            auto t0 = std::chrono::high_resolution_clock::now();
            auto result = fn(cfg, dev, data);
            auto t1 = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            std::cout << "  " << name << ": " << result.total_steps << " steps, "
                      << elapsed << "s wall, val_acc=" << result.final_val_acc;
            if (result.grokking_step >= 0)
                std::cout << ", grokked at step " << result.grokking_step;
            std::cout << "\n";
            results.push_back(result);
        }
    };

    run_if("adamw", train_adamw);
    run_if("grokfast", train_grokfast);
    run_if("muon", train_muon);
    run_if("lion", train_lion);

    // Save results
    write_json(results, output_file);
    std::cout << "Results saved to " << output_file << "\n";

    return 0;
}
