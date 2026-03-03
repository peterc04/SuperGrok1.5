/*
 * Stream Scheduler + Auto-Tuner — Implementation
 */

#include "stream_scheduler.h"
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

#ifdef WITH_CUDA
#include <c10/cuda/CUDAStream.h>
#endif


// ═══════════════════════════════════════════════════════════════════════
//  StreamScheduler Implementation
// ═══════════════════════════════════════════════════════════════════════

StreamScheduler::StreamScheduler() = default;

StreamScheduler::~StreamScheduler() {
#ifdef WITH_CUDA
    if (initialized_) {
        if (compute_stream_) cudaStreamDestroy(compute_stream_);
        if (metric_stream_) cudaStreamDestroy(metric_stream_);
        if (transfer_stream_) cudaStreamDestroy(transfer_stream_);
        if (compute_done_event_) cudaEventDestroy(compute_done_event_);
        if (metrics_done_event_) cudaEventDestroy(metrics_done_event_);
        if (pinned_loss_) cudaFreeHost(pinned_loss_);
        if (pinned_acc_) cudaFreeHost(pinned_acc_);
    }
#endif
}

void StreamScheduler::init(int device_id) {
#ifdef WITH_CUDA
    device_id_ = device_id;
    cudaSetDevice(device_id);

    // Create streams with high priority for compute, low for metrics
    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);

    cudaStreamCreateWithPriority(&compute_stream_, cudaStreamNonBlocking, greatest_priority);
    cudaStreamCreateWithPriority(&metric_stream_, cudaStreamNonBlocking, least_priority);
    cudaStreamCreateWithPriority(&transfer_stream_, cudaStreamNonBlocking, least_priority);

    // Create events for stream synchronization
    cudaEventCreateWithFlags(&compute_done_event_, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&metrics_done_event_, cudaEventDisableTiming);

    // Allocate pinned host memory for async transfers
    cudaMallocHost(&pinned_loss_, sizeof(float));
    cudaMallocHost(&pinned_acc_, sizeof(float));

    initialized_ = true;
#else
    (void)device_id;
#endif
}

void StreamScheduler::schedule_metrics(
    torch::Tensor logits,
    torch::Tensor targets,
    int num_classes
) {
#ifdef WITH_CUDA
    if (!initialized_) return;

    // Wait for compute to finish before computing metrics
    cudaStreamWaitEvent(metric_stream_, compute_done_event_, 0);

    // Compute loss and accuracy on metric stream
    // Note: We set the current stream to metric_stream_ temporarily
    auto old_stream = c10::cuda::getCurrentCUDAStream(device_id_);
    c10::cuda::setCurrentCUDAStream(
        c10::cuda::getStreamFromExternal(metric_stream_, device_id_));

    auto loss = torch::nn::functional::cross_entropy(
        logits.slice(1, 0, num_classes), targets);
    auto preds = logits.slice(1, 0, num_classes).argmax(1);
    auto acc = (preds == targets).to(torch::kFloat32).mean();

    // Async copy to pinned memory
    cudaMemcpyAsync(pinned_loss_, loss.data_ptr<float>(), sizeof(float),
                    cudaMemcpyDeviceToHost, metric_stream_);
    cudaMemcpyAsync(pinned_acc_, acc.data_ptr<float>(), sizeof(float),
                    cudaMemcpyDeviceToHost, metric_stream_);

    // Record event so we know when metrics are done
    cudaEventRecord(metrics_done_event_, metric_stream_);
    metrics_pending_ = true;

    // Restore original stream
    c10::cuda::setCurrentCUDAStream(old_stream);
#else
    (void)logits; (void)targets; (void)num_classes;
#endif
}

std::pair<float, float> StreamScheduler::read_metrics() {
#ifdef WITH_CUDA
    if (!metrics_pending_) return {-1.0f, -1.0f};

    // Wait for metrics to complete
    cudaEventSynchronize(metrics_done_event_);
    metrics_pending_ = false;

    return {*pinned_loss_, *pinned_acc_};
#else
    return {-1.0f, -1.0f};
#endif
}

bool StreamScheduler::metrics_ready() const {
#ifdef WITH_CUDA
    if (!metrics_pending_) return false;
    return cudaEventQuery(metrics_done_event_) == cudaSuccess;
#else
    return false;
#endif
}

void StreamScheduler::signal_compute_done() {
#ifdef WITH_CUDA
    if (!initialized_) return;
    cudaEventRecord(compute_done_event_, compute_stream_);
#endif
}

void StreamScheduler::sync_all() {
#ifdef WITH_CUDA
    if (!initialized_) return;
    cudaStreamSynchronize(compute_stream_);
    cudaStreamSynchronize(metric_stream_);
    cudaStreamSynchronize(transfer_stream_);
#endif
}


// ═══════════════════════════════════════════════════════════════════════
//  DeferredMetrics Implementation
// ═══════════════════════════════════════════════════════════════════════

void DeferredMetrics::init(torch::Device device, int buffer_size) {
    buffer_size_ = buffer_size;
    loss_buffer_ = torch::zeros({buffer_size}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    acc_buffer_ = torch::zeros({buffer_size}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    step_index_.resize(buffer_size, -1);

#ifdef WITH_CUDA
    if (device.is_cuda()) {
        cudaMallocHost(&pinned_losses_, buffer_size * sizeof(float));
        cudaMallocHost(&pinned_accs_, buffer_size * sizeof(float));
    } else
#endif
    {
        pinned_losses_ = new float[buffer_size];
        pinned_accs_ = new float[buffer_size];
    }
    initialized_ = true;
}

void DeferredMetrics::record(torch::Tensor loss, torch::Tensor accuracy, int step) {
    if (!initialized_) return;

    int idx = write_idx_ % buffer_size_;
    loss_buffer_[idx] = loss.detach();
    acc_buffer_[idx] = accuracy.detach();
    step_index_[idx] = step;
    write_idx_++;
}

std::tuple<float, float, int> DeferredMetrics::read_latest() {
    if (!initialized_ || write_idx_ == 0) return {-1.0f, -1.0f, -1};

    int idx = (write_idx_ - 1) % buffer_size_;
    float loss = loss_buffer_[idx].item<float>();
    float acc = acc_buffer_[idx].item<float>();
    return {loss, acc, step_index_[idx]};
}

std::tuple<float, float, bool> DeferredMetrics::read_step(int step) {
    if (!initialized_) return {-1.0f, -1.0f, false};

    for (int i = 0; i < buffer_size_; i++) {
        if (step_index_[i] == step) {
            float loss = loss_buffer_[i].item<float>();
            float acc = acc_buffer_[i].item<float>();
            return {loss, acc, true};
        }
    }
    return {-1.0f, -1.0f, false};
}


// ═══════════════════════════════════════════════════════════════════════
//  AutoTuner Implementation
// ═══════════════════════════════════════════════════════════════════════

KernelConfig AutoTuner::tune(
    const std::string& kernel_name,
    int tensor_numel,
    int gpu_sm_count,
    std::function<void(int, int)> kernel_fn,
    int num_trials
) {
    std::string key = make_key(kernel_name, tensor_numel);

    // Check cache first
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;

    // Candidate block sizes
    std::vector<int> block_sizes = {128, 256, 512};
    KernelConfig best = {256, 0, 1e9f};  // Default

#ifdef WITH_CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int bs : block_sizes) {
        // Skip if block count would be too low for good occupancy
        int num_blocks = (tensor_numel + bs - 1) / bs;
        if (num_blocks < gpu_sm_count / 2) continue;

        // Warm up
        kernel_fn(bs, 0);
        cudaDeviceSynchronize();

        // Profile
        float total_ms = 0.0f;
        for (int t = 0; t < num_trials; t++) {
            cudaEventRecord(start);
            kernel_fn(bs, 0);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            total_ms += ms;
        }

        float avg_ms = total_ms / num_trials;
        if (avg_ms < best.measured_ms) {
            best.block_size = bs;
            best.shared_mem_bytes = 0;
            best.measured_ms = avg_ms;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#else
    (void)kernel_fn; (void)num_trials; (void)gpu_sm_count;
#endif

    cache_[key] = best;
    return best;
}

KernelConfig AutoTuner::get_config(const std::string& kernel_name, int tensor_numel) const {
    std::string key = make_key(kernel_name, tensor_numel);
    auto it = cache_.find(key);
    if (it != cache_.end()) return it->second;
    return {256, 0, -1.0f};  // Default
}

bool AutoTuner::is_tuned(const std::string& kernel_name, int tensor_numel) const {
    return cache_.count(make_key(kernel_name, tensor_numel)) > 0;
}

void AutoTuner::save(const std::string& path) const {
    std::ofstream ofs(path);
    if (!ofs) return;
    for (const auto& [key, cfg] : cache_) {
        ofs << key << " " << cfg.block_size << " "
            << cfg.shared_mem_bytes << " " << cfg.measured_ms << "\n";
    }
}

void AutoTuner::load(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) return;
    std::string key;
    KernelConfig cfg;
    while (ifs >> key >> cfg.block_size >> cfg.shared_mem_bytes >> cfg.measured_ms) {
        cache_[key] = cfg;
    }
}
