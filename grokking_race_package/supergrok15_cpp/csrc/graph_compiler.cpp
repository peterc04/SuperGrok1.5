/*
 * Graph Compiler — Implementation
 *
 * CUDA Graph capture, static memory pool, and graph library.
 */

#include "graph_compiler.h"
#include <stdexcept>
#include <algorithm>

#ifdef WITH_CUDA
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#endif


// ═══════════════════════════════════════════════════════════════════════
//  MemoryPool Implementation
// ═══════════════════════════════════════════════════════════════════════

MemoryPool::~MemoryPool() {
    if (pool_) {
#ifdef WITH_CUDA
        if (device_.is_cuda()) {
            cudaFree(pool_);
        } else
#endif
        {
            free(pool_);
        }
        pool_ = nullptr;
    }
}

void MemoryPool::init(torch::Device device, size_t capacity_bytes) {
    device_ = device;
    capacity_ = capacity_bytes;
    used_ = 0;
    peak_used_ = 0;
    slots_.clear();

#ifdef WITH_CUDA
    if (device.is_cuda()) {
        cudaError_t err = cudaMalloc(&pool_, capacity_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("MemoryPool: cudaMalloc failed: ") +
                cudaGetErrorString(err));
        }
    } else
#endif
    {
        pool_ = malloc(capacity_bytes);
        if (!pool_) {
            throw std::runtime_error("MemoryPool: malloc failed");
        }
    }
}

torch::Tensor MemoryPool::allocate(
    size_t size_bytes, torch::ScalarType dtype,
    std::vector<int64_t> shape, int op_id
) {
    // Align to 256 bytes for optimal GPU access
    size_bytes = (size_bytes + 255) & ~255;

    // Try to find a reusable slot
    int reuse_idx = find_reusable_slot(size_bytes, op_id);
    if (reuse_idx >= 0) {
        auto& slot = slots_[reuse_idx];
        slot.in_use = true;
        slot.first_use = op_id;
        slot.last_use = op_id;  // Will be updated by caller
        void* ptr = static_cast<char*>(pool_) + slot.offset;

        auto options = torch::TensorOptions()
            .dtype(dtype)
            .device(device_);
        return torch::from_blob(ptr, shape, options);
    }

    // Allocate new slot
    if (used_ + size_bytes > capacity_) {
        throw std::runtime_error(
            "MemoryPool: out of memory (used=" + std::to_string(used_) +
            ", requested=" + std::to_string(size_bytes) +
            ", capacity=" + std::to_string(capacity_) + ")");
    }

    BufferSlot slot;
    slot.offset = used_;
    slot.size = size_bytes;
    slot.first_use = op_id;
    slot.last_use = op_id;
    slot.in_use = true;
    slots_.push_back(slot);

    void* ptr = static_cast<char*>(pool_) + used_;
    used_ += size_bytes;
    peak_used_ = std::max(peak_used_, used_);

    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(device_);
    return torch::from_blob(ptr, shape, options);
}

void MemoryPool::mark_op_complete(int op_id) {
    for (auto& slot : slots_) {
        if (slot.in_use && slot.last_use <= op_id) {
            slot.in_use = false;
        }
    }
}

void MemoryPool::reset() {
    for (auto& slot : slots_) {
        slot.in_use = false;
    }
    // Don't reset used_ — keep the watermark for the pool layout
    // This allows reuse of the same layout across steps
}

int MemoryPool::find_reusable_slot(size_t size_bytes, int current_op_id) {
    int best_idx = -1;
    size_t best_size = SIZE_MAX;

    for (int i = 0; i < static_cast<int>(slots_.size()); i++) {
        auto& slot = slots_[i];
        if (!slot.in_use && slot.size >= size_bytes && slot.size < best_size) {
            best_idx = i;
            best_size = slot.size;
        }
    }
    return best_idx;
}


// ═══════════════════════════════════════════════════════════════════════
//  GraphCapture Implementation
// ═══════════════════════════════════════════════════════════════════════

GraphCapture::~GraphCapture() {
#ifdef WITH_CUDA
    if (instance_) {
        cudaGraphExecDestroy(instance_);
        instance_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
#endif
}

void GraphCapture::begin_capture() {
#ifdef WITH_CUDA
    // Get the current CUDA stream from PyTorch
    stream_ = c10::cuda::getCurrentCUDAStream().stream();

    // Begin capture on this stream
    cudaError_t err = cudaStreamBeginCapture(
        stream_, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("GraphCapture: begin_capture failed: ") +
            cudaGetErrorString(err));
    }
#else
    throw std::runtime_error("GraphCapture requires CUDA");
#endif
}

void GraphCapture::end_capture() {
#ifdef WITH_CUDA
    cudaError_t err = cudaStreamEndCapture(stream_, &graph_);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("GraphCapture: end_capture failed: ") +
            cudaGetErrorString(err));
    }

    // Instantiate the graph for replay
    err = cudaGraphInstantiate(&instance_, graph_, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
        throw std::runtime_error(
            std::string("GraphCapture: instantiate failed: ") +
            cudaGetErrorString(err));
    }
    captured_ = true;
#else
    throw std::runtime_error("GraphCapture requires CUDA");
#endif
}

void GraphCapture::replay() {
#ifdef WITH_CUDA
    if (!captured_) {
        throw std::runtime_error("GraphCapture: no graph captured");
    }
    cudaError_t err = cudaGraphLaunch(instance_, stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("GraphCapture: replay failed: ") +
            cudaGetErrorString(err));
    }
#else
    throw std::runtime_error("GraphCapture requires CUDA");
#endif
}

void GraphCapture::synchronize() {
#ifdef WITH_CUDA
    if (stream_) {
        cudaStreamSynchronize(stream_);
    }
#endif
}


// ═══════════════════════════════════════════════════════════════════════
//  GraphLibrary Implementation
// ═══════════════════════════════════════════════════════════════════════

void GraphLibrary::capture(StepType type, std::function<void()> step_fn) {
    int key = static_cast<int>(type);
    auto& gc = graphs_[key];

    gc.begin_capture();
    step_fn();
    gc.end_capture();
}

void GraphLibrary::replay(StepType type) {
    int key = static_cast<int>(type);
    auto it = graphs_.find(key);
    if (it == graphs_.end()) {
        throw std::runtime_error("GraphLibrary: no graph for step type " +
                                 std::to_string(key));
    }
    it->second.replay();
}

bool GraphLibrary::has(StepType type) const {
    return graphs_.count(static_cast<int>(type)) > 0;
}

GraphCapture& GraphLibrary::get(StepType type) {
    return graphs_.at(static_cast<int>(type));
}
