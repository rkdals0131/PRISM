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
    // Memory pool integration
    MemoryPool<PointCloudSoA>* memory_pool{nullptr};
    
    // Performance monitoring
    bool enable_metrics{true};
    
    // Timeout configuration
    std::chrono::milliseconds stage_timeout{5000};
};

/**
 * @brief Abstract base class for execution strategies
 * 
 * Implements the Strategy pattern for different processing pipelines.
 * Currently supports only sequential single-threaded processing for simplicity.
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
        SINGLE_THREAD   ///< Sequential single-threaded processing
    };
    
    /**
     * @brief Constructor with default mode
     */
    explicit ExecutionMode(Mode mode = Mode::SINGLE_THREAD, 
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
    

private:
    Mode current_mode_;
    ExecutionConfig config_;
    
    std::unique_ptr<ExecutionStrategy> current_strategy_;
    
    /**
     * @brief Create strategy for specified mode
     */
    std::unique_ptr<ExecutionStrategy> createStrategy(Mode mode);
    
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
        size_t l3_cache_size;
    };
    
    SystemInfo getSystemInfo();
}

} // namespace core
} // namespace prism