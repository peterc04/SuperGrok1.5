/*
 * Stream Scheduler — Multi-stream execution + kernel auto-tuning
 *
 * Manages multiple CUDA streams to overlap independent operations:
 *   - compute_stream:  Forward → backward → optimizer step
 *   - metric_stream:   Loss/accuracy computation (overlaps with optimizer)
 *   - transfer_stream: GPU→CPU metric transfers (async, deferred read)
 *
 * Also provides kernel auto-tuning: profiles different block sizes at
 * first launch and caches the optimal configuration per (kernel, N, GPU).
 *
 * Key optimization for SuperGrok: Instead of calling .item() every step
 * (which forces GPU→CPU sync), we compute metrics on metric_stream and
 * only synchronize when the values are actually needed (every alpha_freq).
 */
#pragma once

#include <torch/extension.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif


// ═══════════════════════════════════════════════════════════════════════
//  StreamScheduler — Multi-stream orchestration
// ═══════════════════════════════════════════════════════════════════════

class StreamScheduler {
public:
    StreamScheduler();
    ~StreamScheduler();

    // Initialize streams on a specific device
    void init(int device_id = 0);

    // Get streams for different purposes
#ifdef WITH_CUDA
    cudaStream_t compute_stream() const { return compute_stream_; }
    cudaStream_t metric_stream() const { return metric_stream_; }
    cudaStream_t transfer_stream() const { return transfer_stream_; }
#endif

    // Schedule metric computation on separate stream
    // Returns immediately — call read_metrics() to get values
    void schedule_metrics(
        torch::Tensor logits,
        torch::Tensor targets,
        int num_classes);

    // Read metric values (synchronizes metric_stream if needed)
    // Returns (loss, accuracy) pair
    std::pair<float, float> read_metrics();

    // Check if metrics are ready without blocking
    bool metrics_ready() const;

    // Insert event on compute stream, wait on metric stream
    // This creates a dependency: metrics can start after compute event
    void signal_compute_done();

    // Synchronize all streams (call at end of step if needed)
    void sync_all();

private:
#ifdef WITH_CUDA
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t metric_stream_ = nullptr;
    cudaStream_t transfer_stream_ = nullptr;
    cudaEvent_t compute_done_event_ = nullptr;
    cudaEvent_t metrics_done_event_ = nullptr;
#endif

    // Pinned host memory for async GPU→CPU transfers
    float* pinned_loss_ = nullptr;
    float* pinned_acc_ = nullptr;
    bool metrics_pending_ = false;
    int device_id_ = 0;
    bool initialized_ = false;
};


// ═══════════════════════════════════════════════════════════════════════
//  DeferredMetrics — Async metric collection without .item() sync
// ═══════════════════════════════════════════════════════════════════════

class DeferredMetrics {
public:
    DeferredMetrics() = default;

    // Initialize with buffer sizes
    void init(torch::Device device, int buffer_size = 64);

    // Record metrics asynchronously (no CPU sync)
    // Uses a ring buffer — old values overwritten
    void record(torch::Tensor loss, torch::Tensor accuracy, int step);

    // Read the most recent metrics (syncs only if needed)
    // Returns (loss, accuracy, step) or (-1, -1, -1) if nothing recorded
    std::tuple<float, float, int> read_latest();

    // Read metrics for a specific step (may need to sync)
    std::tuple<float, float, bool> read_step(int step);

private:
    // GPU-side ring buffers
    torch::Tensor loss_buffer_;     // [buffer_size] on GPU
    torch::Tensor acc_buffer_;      // [buffer_size] on GPU
    std::vector<int> step_index_;   // CPU-side step tracking

    // Pinned host-side copies
    float* pinned_losses_ = nullptr;
    float* pinned_accs_ = nullptr;

    int buffer_size_ = 0;
    int write_idx_ = 0;
    int read_idx_ = 0;
    bool initialized_ = false;
};


// ═══════════════════════════════════════════════════════════════════════
//  AutoTuner — Kernel block size auto-tuning
// ═══════════════════════════════════════════════════════════════════════

struct KernelConfig {
    int block_size;         // Threads per block (128, 256, 512)
    int shared_mem_bytes;   // Dynamic shared memory
    float measured_ms;      // Profiled execution time
};

class AutoTuner {
public:
    AutoTuner() = default;

    // Profile a kernel with different configurations and return the best one
    // The kernel_fn takes (block_size, shared_mem_bytes) and launches the kernel
    KernelConfig tune(
        const std::string& kernel_name,
        int tensor_numel,
        int gpu_sm_count,
        std::function<void(int block_size, int shared_mem)> kernel_fn,
        int num_trials = 5);

    // Get cached config for a kernel (returns default if not tuned)
    KernelConfig get_config(const std::string& kernel_name, int tensor_numel) const;

    // Check if a kernel has been tuned
    bool is_tuned(const std::string& kernel_name, int tensor_numel) const;

    // Save/load tuning results to/from file
    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    // Cache key: kernel_name + "_" + tensor_numel
    std::unordered_map<std::string, KernelConfig> cache_;

    std::string make_key(const std::string& name, int numel) const {
        return name + "_" + std::to_string(numel);
    }
};
