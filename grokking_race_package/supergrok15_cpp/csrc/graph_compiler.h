/*
 * Graph Compiler — CUDA Graph Capture + Static Memory Planner
 *
 * Layer 2 of the optimization stack. Sits between the C++ runtime (Layer 3)
 * and the custom CUDA kernels (Layer 1).
 *
 * Components:
 *   1. MemoryPool     — Pre-allocated GPU buffer pool with lifetime-aware reuse
 *   2. GraphCapture   — CUDA Graph recording/replay for entire training steps
 *   3. GraphLibrary   — Multiple captured graphs for different step types
 *                        (normal step, SAM closure step, evaluation step)
 *
 * Key insight: Since tensor shapes are fixed throughout training (fixed batch
 * size, fixed model architecture), we can pre-plan ALL memory at init time
 * and capture the entire step as a CUDA Graph for ~3μs replay overhead.
 */
#pragma once

#include <torch/extension.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif


// ═══════════════════════════════════════════════════════════════════════
//  MemoryPool — Static pre-allocated GPU buffer pool
// ═══════════════════════════════════════════════════════════════════════

class MemoryPool {
public:
    struct BufferSlot {
        size_t offset;       // Byte offset into the pool
        size_t size;         // Buffer size in bytes
        int first_use;       // First operation that uses this buffer
        int last_use;        // Last operation that uses this buffer
        bool in_use;         // Currently allocated
    };

    MemoryPool() = default;
    ~MemoryPool();

    // Initialize pool on a specific device with a given capacity
    void init(torch::Device device, size_t capacity_bytes);

    // Request a buffer of given size. Returns a tensor view into the pool.
    // If lifetime info is provided, enables buffer reuse after last_use.
    torch::Tensor allocate(size_t size_bytes, torch::ScalarType dtype,
                           std::vector<int64_t> shape,
                           int op_id = -1);

    // Mark an operation as complete (enables buffer reuse for expired slots)
    void mark_op_complete(int op_id);

    // Reset all allocations (call between training steps if not using CUDA Graphs)
    void reset();

    // Get pool statistics
    size_t capacity() const { return capacity_; }
    size_t used() const { return used_; }
    size_t peak_used() const { return peak_used_; }
    int num_slots() const { return static_cast<int>(slots_.size()); }

private:
    void* pool_ = nullptr;
    size_t capacity_ = 0;
    size_t used_ = 0;
    size_t peak_used_ = 0;
    torch::Device device_ = torch::kCPU;
    std::vector<BufferSlot> slots_;

    // Find a free slot that can be reused (size >= requested, lifetime expired)
    int find_reusable_slot(size_t size_bytes, int current_op_id);
};


// ═══════════════════════════════════════════════════════════════════════
//  GraphCapture — CUDA Graph recording and replay
// ═══════════════════════════════════════════════════════════════════════

class GraphCapture {
public:
    GraphCapture() = default;
    ~GraphCapture();

    // Begin recording operations on the given stream
    // All subsequent CUDA operations will be recorded, not executed
    void begin_capture();

    // End recording and instantiate the graph for replay
    void end_capture();

    // Replay the entire captured graph (~3μs overhead)
    void replay();

    // Check if a graph has been captured
    bool is_captured() const { return captured_; }

    // Synchronize — wait for replay to complete
    void synchronize();

private:
#ifdef WITH_CUDA
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t instance_ = nullptr;
    cudaStream_t stream_ = nullptr;
#endif
    bool captured_ = false;
};


// ═══════════════════════════════════════════════════════════════════════
//  GraphLibrary — Multiple graphs for different step types
// ═══════════════════════════════════════════════════════════════════════

enum class StepType {
    NORMAL,           // Standard training step (forward + backward + optimizer)
    SAM_CLOSURE,      // Step with SAM perturbation (extra fwd/bwd)
    ALPHA_UPDATE,     // Step with validation forward for alpha/WD update
    SAM_AND_ALPHA,    // Both SAM and alpha update
    EVAL_ONLY,        // Evaluation step (forward only, no optimizer)
};

class GraphLibrary {
public:
    GraphLibrary() = default;

    // Capture a graph for a specific step type by running the provided function
    // The function should contain all CUDA operations for that step type
    void capture(StepType type, std::function<void()> step_fn);

    // Replay the graph for a specific step type
    void replay(StepType type);

    // Check if a graph exists for a step type
    bool has(StepType type) const;

    // Get capture for direct access
    GraphCapture& get(StepType type);

private:
    std::unordered_map<int, GraphCapture> graphs_;
};


// ═══════════════════════════════════════════════════════════════════════
//  Convenience: determine step type from training loop state
// ═══════════════════════════════════════════════════════════════════════

inline StepType classify_step(int step, int sam_freq, int alpha_freq) {
    bool is_sam = (sam_freq > 0) && (step % sam_freq == 0);
    bool is_alpha = (alpha_freq > 0) && (step % alpha_freq == 0);
    if (is_sam && is_alpha) return StepType::SAM_AND_ALPHA;
    if (is_sam) return StepType::SAM_CLOSURE;
    if (is_alpha) return StepType::ALPHA_UPDATE;
    return StepType::NORMAL;
}
