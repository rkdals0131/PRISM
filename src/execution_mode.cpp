#include "prism/core/execution_mode.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>

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
// ExecutionMode Implementation
// =============================================================================

ExecutionMode::ExecutionMode(Mode mode, const ExecutionConfig& config) 
    : current_mode_(mode), config_(config) {
    
    current_strategy_ = createStrategy(current_mode_);
}

void ExecutionMode::setMode(Mode mode) {
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
        default:
            return false;
    }
}


std::unique_ptr<ExecutionStrategy> ExecutionMode::createStrategy(Mode mode) {
    switch (mode) {
        case Mode::SINGLE_THREAD:
            return std::make_unique<SingleThreadPipeline>(config_);
        default:
            throw std::invalid_argument("Unknown execution mode");
    }
}


// =============================================================================
// Utility Functions Implementation
// =============================================================================

namespace execution_utils {

const char* modeToString(ExecutionMode::Mode mode) {
    switch (mode) {
        case ExecutionMode::Mode::SINGLE_THREAD: return "SINGLE_THREAD";
        default: return "UNKNOWN";
    }
}

ExecutionMode::Mode stringToMode(const std::string& mode_str) {
    if (mode_str == "SINGLE_THREAD") return ExecutionMode::Mode::SINGLE_THREAD;
    
    throw std::invalid_argument("Unknown execution mode string: " + mode_str);
}

SystemInfo getSystemInfo() {
    SystemInfo info;
    
    // Get number of CPU cores
    info.num_cpu_cores = std::thread::hardware_concurrency();
    
    // Estimate L3 cache size (simplified)
    info.l3_cache_size = 8 * 1024 * 1024; // 8MB default estimate
    
    return info;
}

} // namespace execution_utils

} // namespace core
} // namespace prism