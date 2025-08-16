#pragma once

#include <memory>
#include <chrono>
#include <vector>
#include <functional>
#include <atomic>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>

#include "point_cloud_soa.hpp"
#include "memory_pool.hpp"

// Forward declarations for opaque pointers
namespace prism { namespace core { namespace tbb_impl {
    struct TBBArena;
    struct TBBControl;
}}}

namespace prism {
namespace core {

/**
 * @brief Performance metrics for pipeline execution
 */
struct PipelineMetrics {
    std::chrono::nanoseconds interpolation_time{0};
    std::chrono::nanoseconds projection_time{0};
    std::chrono::nanoseconds fusion_time{0};
    std::chrono::nanoseconds total_time{0};
    
    size_t points_processed{0};
    double throughput_points_per_second{0.0};
    
    // Reset all metrics
    void reset() {
        interpolation_time = std::chrono::nanoseconds{0};
        projection_time = std::chrono::nanoseconds{0};
        fusion_time = std::chrono::nanoseconds{0};
        total_time = std::chrono::nanoseconds{0};
        points_processed = 0;
        throughput_points_per_second = 0.0;
    }
    
    // Calculate throughput based on total time and points processed
    void calculateThroughput() {
        if (total_time.count() > 0 && points_processed > 0) {
            double seconds = std::chrono::duration<double>(total_time).count();
            throughput_points_per_second = points_processed / seconds;
        }
    }
};

/**
 * @brief Pipeline stage function type
 * Input: source point cloud, output: processed point cloud
 */
using PipelineStage = std::function<PooledPtr<PointCloudSoA>(const PointCloudSoA&)>;

/**
 * @brief Configuration for execution modes
 */
struct ExecutionConfig {
    // Thread pool configuration
    size_t num_threads{std::thread::hardware_concurrency()};
    
    // SIMD configuration
    bool enable_avx2{true};
    bool enable_sse{true};
    
    // Memory pool integration
    MemoryPool<PointCloudSoA>* memory_pool{nullptr};
    
    // Performance monitoring
    bool enable_metrics{true};
    
    // Batch processing configuration
    size_t batch_size{1000}; // Points per batch for parallel processing
    
    // Timeout configuration
    std::chrono::milliseconds stage_timeout{5000};
};

/**
 * @brief Abstract base class for execution strategies
 * 
 * Implements the Strategy pattern for different processing pipelines.
 * Each strategy provides different execution characteristics:
 * - Sequential vs Parallel processing
 * - SIMD optimization levels
 * - Memory management approaches
 */
class ExecutionStrategy {
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~ExecutionStrategy() = default;
    
    /**
     * @brief Execute the complete pipeline
     * 
     * @param input Input point cloud
     * @param interpolation_stage Temporal interpolation stage
     * @param projection_stage Coordinate projection stage
     * @param fusion_stage Multi-sensor fusion stage
     * @return Processed point cloud through all stages
     */
    virtual PooledPtr<PointCloudSoA> execute(
        const PointCloudSoA& input,
        const PipelineStage& interpolation_stage,
        const PipelineStage& projection_stage,
        const PipelineStage& fusion_stage) = 0;
    
    /**
     * @brief Get execution metrics from last run
     */
    virtual const PipelineMetrics& getMetrics() const = 0;
    
    /**
     * @brief Reset performance metrics
     */
    virtual void resetMetrics() = 0;
    
    /**
     * @brief Get strategy name for identification
     */
    virtual const char* getName() const = 0;
    
    /**
     * @brief Configure the execution strategy
     */
    virtual void configure(const ExecutionConfig& config) = 0;
    
    /**
     * @brief Check if strategy is available on current hardware
     */
    virtual bool isAvailable() const = 0;

protected:
    ExecutionConfig config_;
    mutable std::mutex metrics_mutex_;
    PipelineMetrics metrics_;
};

/**
 * @brief Sequential single-threaded execution strategy
 * 
 * Provides baseline performance and debugging capabilities.
 * Executes pipeline stages in order on a single thread.
 */
class SingleThreadPipeline : public ExecutionStrategy {
public:
    explicit SingleThreadPipeline(const ExecutionConfig& config = ExecutionConfig());
    
    PooledPtr<PointCloudSoA> execute(
        const PointCloudSoA& input,
        const PipelineStage& interpolation_stage,
        const PipelineStage& projection_stage,
        const PipelineStage& fusion_stage) override;
    
    const PipelineMetrics& getMetrics() const override;
    void resetMetrics() override;
    const char* getName() const override { return "SingleThreadPipeline"; }
    void configure(const ExecutionConfig& config) override;
    bool isAvailable() const override { return true; }
};

/**
 * @brief Multi-threaded CPU parallel execution strategy
 * 
 * Uses Intel TBB (Threading Building Blocks) for parallel processing.
 * Suitable for large point clouds that can be processed in parallel batches.
 */
class CPUParallelPipeline : public ExecutionStrategy {
public:
    explicit CPUParallelPipeline(const ExecutionConfig& config = ExecutionConfig());
    ~CPUParallelPipeline();
    
    PooledPtr<PointCloudSoA> execute(
        const PointCloudSoA& input,
        const PipelineStage& interpolation_stage,
        const PipelineStage& projection_stage,
        const PipelineStage& fusion_stage) override;
    
    const PipelineMetrics& getMetrics() const override;
    void resetMetrics() override;
    const char* getName() const override { return "CPUParallelPipeline"; }
    void configure(const ExecutionConfig& config) override;
    bool isAvailable() const override;

private:
    // Use opaque pointers to avoid TBB header dependencies
    std::unique_ptr<tbb_impl::TBBArena> task_arena_;
    std::unique_ptr<tbb_impl::TBBControl> thread_control_;
    
    /**
     * @brief Execute stage in parallel batches
     */
    PooledPtr<PointCloudSoA> executeStageParallel(
        const PointCloudSoA& input,
        const PipelineStage& stage) const;
};

/**
 * @brief SIMD-optimized execution strategy
 * 
 * Uses vectorized operations (AVX2/SSE) for mathematical computations.
 * Optimized for point cloud transformations and filtering operations.
 */
class SIMDPipeline : public ExecutionStrategy {
public:
    explicit SIMDPipeline(const ExecutionConfig& config = ExecutionConfig());
    
    PooledPtr<PointCloudSoA> execute(
        const PointCloudSoA& input,
        const PipelineStage& interpolation_stage,
        const PipelineStage& projection_stage,
        const PipelineStage& fusion_stage) override;
    
    const PipelineMetrics& getMetrics() const override;
    void resetMetrics() override;
    const char* getName() const override { return "SIMDPipeline"; }
    void configure(const ExecutionConfig& config) override;
    bool isAvailable() const override;

private:
    bool avx2_available_{false};
    bool sse_available_{false};
    
    /**
     * @brief Execute stage with SIMD optimizations
     */
    PooledPtr<PointCloudSoA> executeStageVectorized(
        const PointCloudSoA& input,
        const PipelineStage& stage) const;
    
    /**
     * @brief Check CPU features at runtime
     */
    void detectSIMDFeatures();
};

/**
 * @brief Execution mode manager and factory
 * 
 * Manages different execution strategies and provides a unified interface
 * for pipeline execution. Handles strategy selection and runtime switching.
 */
class ExecutionMode {
public:
    /**
     * @brief Available execution modes
     */
    enum class Mode {
        SINGLE_THREAD,   ///< Sequential single-threaded processing
        CPU_PARALLEL,    ///< Multi-threaded CPU processing with TBB
        SIMD,           ///< SIMD-optimized processing
        AUTO            ///< Automatic selection based on hardware and input size
    };
    
    /**
     * @brief Constructor with default mode
     */
    explicit ExecutionMode(Mode mode = Mode::AUTO, 
                          const ExecutionConfig& config = ExecutionConfig());
    
    /**
     * @brief Change execution mode
     */
    void setMode(Mode mode);
    
    /**
     * @brief Get current execution mode
     */
    Mode getMode() const { return current_mode_; }
    
    /**
     * @brief Update configuration
     */
    void configure(const ExecutionConfig& config);
    
    /**
     * @brief Execute pipeline with current strategy
     */
    PooledPtr<PointCloudSoA> execute(
        const PointCloudSoA& input,
        const PipelineStage& interpolation_stage,
        const PipelineStage& projection_stage,
        const PipelineStage& fusion_stage);
    
    /**
     * @brief Get performance metrics from last execution
     */
    const PipelineMetrics& getMetrics() const;
    
    /**
     * @brief Reset all performance metrics
     */
    void resetMetrics();
    
    /**
     * @brief Get name of current execution strategy
     */
    const char* getCurrentStrategyName() const;
    
    /**
     * @brief Check if a specific mode is available
     */
    static bool isModeAvailable(Mode mode);
    
    /**
     * @brief Get optimal mode for given input size and hardware
     */
    static Mode getOptimalMode(size_t input_size);

private:
    Mode current_mode_;
    ExecutionConfig config_;
    
    std::unique_ptr<ExecutionStrategy> current_strategy_;
    
    /**
     * @brief Create strategy for specified mode
     */
    std::unique_ptr<ExecutionStrategy> createStrategy(Mode mode);
    
    /**
     * @brief Automatically select best mode
     */
    Mode selectAutoMode(size_t input_size) const;
};

/**
 * @brief Utility functions for execution mode
 */
namespace execution_utils {
    /**
     * @brief Convert mode enum to string
     */
    const char* modeToString(ExecutionMode::Mode mode);
    
    /**
     * @brief Parse mode from string
     */
    ExecutionMode::Mode stringToMode(const std::string& mode_str);
    
    /**
     * @brief Get system information relevant to execution modes
     */
    struct SystemInfo {
        size_t num_cpu_cores;
        bool avx2_supported;
        bool sse_supported;
        size_t l3_cache_size;
    };
    
    SystemInfo getSystemInfo();
}

} // namespace core
} // namespace prism