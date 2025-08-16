#include "prism/core/execution_mode.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

// CPU feature detection
#ifdef __x86_64__
#include <cpuid.h>
#include <immintrin.h>
#endif

// TBB includes (conditionally compiled)
#ifdef PRISM_ENABLE_TBB
#include <tbb/task_arena.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

// Opaque pointer implementations for TBB
namespace prism { namespace core { namespace tbb_impl {

#ifdef PRISM_ENABLE_TBB
struct TBBArena {
    tbb::task_arena arena;
    explicit TBBArena(int threads) : arena(threads) {}
};

struct TBBControl {
    tbb::global_control control;
    TBBControl(tbb::global_control::parameter param, size_t value) 
        : control(param, value) {}
};
#else
// Dummy implementations when TBB is not available
struct TBBArena {
    explicit TBBArena(int) {}
};

struct TBBControl {
    enum parameter { max_allowed_parallelism };
    TBBControl(parameter, size_t) {}
};
#endif

}}}

namespace prism {
namespace core {

// =============================================================================
// SingleThreadPipeline Implementation
// =============================================================================

SingleThreadPipeline::SingleThreadPipeline(const ExecutionConfig& config) {
    configure(config);
}

PooledPtr<PointCloudSoA> SingleThreadPipeline::execute(
    const PointCloudSoA& input,
    const PipelineStage& interpolation_stage,
    const PipelineStage& projection_stage,
    const PipelineStage& fusion_stage) {
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    if (config_.enable_metrics) {
        metrics_.reset();
        metrics_.points_processed = input.size();
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Stage 1: Temporal interpolation
    auto interp_start = std::chrono::high_resolution_clock::now();
    auto interpolated = interpolation_stage(input);
    auto interp_end = std::chrono::high_resolution_clock::now();
    
    if (config_.enable_metrics) {
        metrics_.interpolation_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            interp_end - interp_start);
    }
    
    // Stage 2: Coordinate projection
    auto proj_start = std::chrono::high_resolution_clock::now();
    auto projected = projection_stage(*interpolated);
    auto proj_end = std::chrono::high_resolution_clock::now();
    
    if (config_.enable_metrics) {
        metrics_.projection_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            proj_end - proj_start);
    }
    
    // Stage 3: Multi-sensor fusion
    auto fusion_start = std::chrono::high_resolution_clock::now();
    auto result = fusion_stage(*projected);
    auto fusion_end = std::chrono::high_resolution_clock::now();
    
    if (config_.enable_metrics) {
        metrics_.fusion_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            fusion_end - fusion_start);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    if (config_.enable_metrics) {
        metrics_.total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - start_time);
        metrics_.calculateThroughput();
    }
    
    return result;
}

const PipelineMetrics& SingleThreadPipeline::getMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void SingleThreadPipeline::resetMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_.reset();
}

void SingleThreadPipeline::configure(const ExecutionConfig& config) {
    config_ = config;
}

// =============================================================================
// CPUParallelPipeline Implementation
// =============================================================================

CPUParallelPipeline::CPUParallelPipeline(const ExecutionConfig& config) {
    configure(config);
}

CPUParallelPipeline::~CPUParallelPipeline() = default;

PooledPtr<PointCloudSoA> CPUParallelPipeline::execute(
    const PointCloudSoA& input,
    const PipelineStage& interpolation_stage,
    const PipelineStage& projection_stage,
    const PipelineStage& fusion_stage) {
    
#ifdef PRISM_ENABLE_TBB
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    if (config_.enable_metrics) {
        metrics_.reset();
        metrics_.points_processed = input.size();
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Stage 1: Temporal interpolation (parallel)
    auto interp_start = std::chrono::high_resolution_clock::now();
    auto interpolated = executeStageParallel(input, interpolation_stage);
    auto interp_end = std::chrono::high_resolution_clock::now();
    
    if (config_.enable_metrics) {
        metrics_.interpolation_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            interp_end - interp_start);
    }
    
    // Stage 2: Coordinate projection (parallel)
    auto proj_start = std::chrono::high_resolution_clock::now();
    auto projected = executeStageParallel(*interpolated, projection_stage);
    auto proj_end = std::chrono::high_resolution_clock::now();
    
    if (config_.enable_metrics) {
        metrics_.projection_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            proj_end - proj_start);
    }
    
    // Stage 3: Multi-sensor fusion (parallel)
    auto fusion_start = std::chrono::high_resolution_clock::now();
    auto result = executeStageParallel(*projected, fusion_stage);
    auto fusion_end = std::chrono::high_resolution_clock::now();
    
    if (config_.enable_metrics) {
        metrics_.fusion_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            fusion_end - fusion_start);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    if (config_.enable_metrics) {
        metrics_.total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - start_time);
        metrics_.calculateThroughput();
    }
    
    return result;
#else
    // Fallback to single-threaded if TBB not available
    SingleThreadPipeline fallback(config_);
    return fallback.execute(input, interpolation_stage, projection_stage, fusion_stage);
#endif
}

PooledPtr<PointCloudSoA> CPUParallelPipeline::executeStageParallel(
    const PointCloudSoA& input,
    const PipelineStage& stage) const {
    
#ifdef PRISM_ENABLE_TBB
    // For now, execute stage directly (parallel batching would be implemented here)
    // In a full implementation, this would split the point cloud into batches
    // and process them in parallel using TBB
    return stage(input);
#else
    return stage(input);
#endif
}

const PipelineMetrics& CPUParallelPipeline::getMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void CPUParallelPipeline::resetMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_.reset();
}

void CPUParallelPipeline::configure(const ExecutionConfig& config) {
    config_ = config;
    
    // Configure TBB thread arena using opaque pointers
    if (task_arena_) {
        task_arena_.reset();
    }
    
    if (thread_control_) {
        thread_control_.reset();
    }
    
    // Create new task arena with specified number of threads
    task_arena_ = std::make_unique<tbb_impl::TBBArena>(static_cast<int>(config_.num_threads));
    
    // Set global thread limit
#ifdef PRISM_ENABLE_TBB
    thread_control_ = std::make_unique<tbb_impl::TBBControl>(
        tbb::global_control::max_allowed_parallelism, 
        config_.num_threads);
#else
    thread_control_ = std::make_unique<tbb_impl::TBBControl>(
        tbb_impl::TBBControl::max_allowed_parallelism, 
        config_.num_threads);
#endif
}

bool CPUParallelPipeline::isAvailable() const {
#ifdef PRISM_ENABLE_TBB
    return true;
#else
    return false;
#endif
}

// =============================================================================
// SIMDPipeline Implementation
// =============================================================================

SIMDPipeline::SIMDPipeline(const ExecutionConfig& config) {
    detectSIMDFeatures();
    configure(config);
}

PooledPtr<PointCloudSoA> SIMDPipeline::execute(
    const PointCloudSoA& input,
    const PipelineStage& interpolation_stage,
    const PipelineStage& projection_stage,
    const PipelineStage& fusion_stage) {
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    if (config_.enable_metrics) {
        metrics_.reset();
        metrics_.points_processed = input.size();
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Stage 1: Temporal interpolation (vectorized)
    auto interp_start = std::chrono::high_resolution_clock::now();
    auto interpolated = executeStageVectorized(input, interpolation_stage);
    auto interp_end = std::chrono::high_resolution_clock::now();
    
    if (config_.enable_metrics) {
        metrics_.interpolation_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            interp_end - interp_start);
    }
    
    // Stage 2: Coordinate projection (vectorized)
    auto proj_start = std::chrono::high_resolution_clock::now();
    auto projected = executeStageVectorized(*interpolated, projection_stage);
    auto proj_end = std::chrono::high_resolution_clock::now();
    
    if (config_.enable_metrics) {
        metrics_.projection_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            proj_end - proj_start);
    }
    
    // Stage 3: Multi-sensor fusion (vectorized)
    auto fusion_start = std::chrono::high_resolution_clock::now();
    auto result = executeStageVectorized(*projected, fusion_stage);
    auto fusion_end = std::chrono::high_resolution_clock::now();
    
    if (config_.enable_metrics) {
        metrics_.fusion_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            fusion_end - fusion_start);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    if (config_.enable_metrics) {
        metrics_.total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            end_time - start_time);
        metrics_.calculateThroughput();
    }
    
    return result;
}

PooledPtr<PointCloudSoA> SIMDPipeline::executeStageVectorized(
    const PointCloudSoA& input,
    const PipelineStage& stage) const {
    
    // For now, execute stage directly (SIMD optimizations would be implemented here)
    // In a full implementation, this would apply vectorized operations to the 
    // PointCloudSoA arrays using AVX2/SSE intrinsics
    return stage(input);
}

const PipelineMetrics& SIMDPipeline::getMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void SIMDPipeline::resetMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_.reset();
}

void SIMDPipeline::configure(const ExecutionConfig& config) {
    config_ = config;
}

bool SIMDPipeline::isAvailable() const {
    if (config_.enable_avx2 && avx2_available_) return true;
    if (config_.enable_sse && sse_available_) return true;
    return false;
}

void SIMDPipeline::detectSIMDFeatures() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    
    // Check for SSE support
    __cpuid(1, eax, ebx, ecx, edx);
    sse_available_ = (edx & (1 << 25)) != 0; // SSE
    
    // Check for AVX2 support
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    avx2_available_ = (ebx & (1 << 5)) != 0; // AVX2
#else
    // Non-x86 platforms
    sse_available_ = false;
    avx2_available_ = false;
#endif
}

// =============================================================================
// ExecutionMode Implementation
// =============================================================================

ExecutionMode::ExecutionMode(Mode mode, const ExecutionConfig& config) 
    : current_mode_(mode), config_(config) {
    
    if (mode == Mode::AUTO) {
        current_mode_ = selectAutoMode(0); // Default selection
    }
    
    current_strategy_ = createStrategy(current_mode_);
}

void ExecutionMode::setMode(Mode mode) {
    if (mode == Mode::AUTO) {
        mode = selectAutoMode(0); // Default selection
    }
    
    if (mode != current_mode_) {
        current_mode_ = mode;
        current_strategy_ = createStrategy(current_mode_);
    }
}

void ExecutionMode::configure(const ExecutionConfig& config) {
    config_ = config;
    if (current_strategy_) {
        current_strategy_->configure(config);
    }
}

PooledPtr<PointCloudSoA> ExecutionMode::execute(
    const PointCloudSoA& input,
    const PipelineStage& interpolation_stage,
    const PipelineStage& projection_stage,
    const PipelineStage& fusion_stage) {
    
    // Auto-select mode based on input size if in AUTO mode
    if (current_mode_ == Mode::AUTO) {
        Mode optimal_mode = selectAutoMode(input.size());
        if (optimal_mode != current_mode_) {
            setMode(optimal_mode);
        }
    }
    
    if (!current_strategy_) {
        throw std::runtime_error("No execution strategy available");
    }
    
    return current_strategy_->execute(input, interpolation_stage, projection_stage, fusion_stage);
}

const PipelineMetrics& ExecutionMode::getMetrics() const {
    if (!current_strategy_) {
        static PipelineMetrics empty_metrics;
        return empty_metrics;
    }
    return current_strategy_->getMetrics();
}

void ExecutionMode::resetMetrics() {
    if (current_strategy_) {
        current_strategy_->resetMetrics();
    }
}

const char* ExecutionMode::getCurrentStrategyName() const {
    if (!current_strategy_) {
        return "None";
    }
    return current_strategy_->getName();
}

bool ExecutionMode::isModeAvailable(Mode mode) {
    switch (mode) {
        case Mode::SINGLE_THREAD:
            return true;
        case Mode::CPU_PARALLEL: {
            CPUParallelPipeline test_pipeline;
            return test_pipeline.isAvailable();
        }
        case Mode::SIMD: {
            SIMDPipeline test_pipeline;
            return test_pipeline.isAvailable();
        }
        case Mode::AUTO:
            return true; // Auto mode is always available (falls back to single thread)
        default:
            return false;
    }
}

ExecutionMode::Mode ExecutionMode::getOptimalMode(size_t input_size) {
    // Simple heuristics for mode selection
    constexpr size_t SIMD_THRESHOLD = 1000;      // Points where SIMD becomes beneficial
    constexpr size_t PARALLEL_THRESHOLD = 10000; // Points where parallelization helps
    
    if (input_size > PARALLEL_THRESHOLD && isModeAvailable(Mode::CPU_PARALLEL)) {
        return Mode::CPU_PARALLEL;
    }
    
    if (input_size > SIMD_THRESHOLD && isModeAvailable(Mode::SIMD)) {
        return Mode::SIMD;
    }
    
    return Mode::SINGLE_THREAD;
}

std::unique_ptr<ExecutionStrategy> ExecutionMode::createStrategy(Mode mode) {
    switch (mode) {
        case Mode::SINGLE_THREAD:
            return std::make_unique<SingleThreadPipeline>(config_);
        case Mode::CPU_PARALLEL:
            if (CPUParallelPipeline temp; temp.isAvailable()) {
                return std::make_unique<CPUParallelPipeline>(config_);
            }
            // Fallback to single thread
            return std::make_unique<SingleThreadPipeline>(config_);
        case Mode::SIMD:
            if (SIMDPipeline temp; temp.isAvailable()) {
                return std::make_unique<SIMDPipeline>(config_);
            }
            // Fallback to single thread
            return std::make_unique<SingleThreadPipeline>(config_);
        case Mode::AUTO:
            // AUTO mode should have been resolved before reaching here
            return std::make_unique<SingleThreadPipeline>(config_);
        default:
            throw std::invalid_argument("Unknown execution mode");
    }
}

ExecutionMode::Mode ExecutionMode::selectAutoMode(size_t input_size) const {
    return getOptimalMode(input_size);
}

// =============================================================================
// Utility Functions Implementation
// =============================================================================

namespace execution_utils {

const char* modeToString(ExecutionMode::Mode mode) {
    switch (mode) {
        case ExecutionMode::Mode::SINGLE_THREAD: return "SINGLE_THREAD";
        case ExecutionMode::Mode::CPU_PARALLEL: return "CPU_PARALLEL";
        case ExecutionMode::Mode::SIMD: return "SIMD";
        case ExecutionMode::Mode::AUTO: return "AUTO";
        default: return "UNKNOWN";
    }
}

ExecutionMode::Mode stringToMode(const std::string& mode_str) {
    if (mode_str == "SINGLE_THREAD") return ExecutionMode::Mode::SINGLE_THREAD;
    if (mode_str == "CPU_PARALLEL") return ExecutionMode::Mode::CPU_PARALLEL;
    if (mode_str == "SIMD") return ExecutionMode::Mode::SIMD;
    if (mode_str == "AUTO") return ExecutionMode::Mode::AUTO;
    
    throw std::invalid_argument("Unknown execution mode string: " + mode_str);
}

SystemInfo getSystemInfo() {
    SystemInfo info;
    
    // Get number of CPU cores
    info.num_cpu_cores = std::thread::hardware_concurrency();
    
    // Detect SIMD features
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    
    // Check for SSE support
    __cpuid(1, eax, ebx, ecx, edx);
    info.sse_supported = (edx & (1 << 25)) != 0;
    
    // Check for AVX2 support
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    info.avx2_supported = (ebx & (1 << 5)) != 0;
#else
    info.sse_supported = false;
    info.avx2_supported = false;
#endif
    
    // Estimate L3 cache size (simplified)
    info.l3_cache_size = 8 * 1024 * 1024; // 8MB default estimate
    
    return info;
}

} // namespace execution_utils

} // namespace core
} // namespace prism