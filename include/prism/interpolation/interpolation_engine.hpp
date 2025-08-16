#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include <functional>
#include <chrono>

#ifdef PRISM_ENABLE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_arena.h>
#include <tbb/global_control.h>
#endif

#include "prism/core/point_cloud_soa.hpp"
#include "prism/core/memory_pool.hpp"
#include "prism/core/execution_mode.hpp"
#include "catmull_rom_interpolator.hpp"
#include "beam_altitude_manager.hpp"

namespace prism {
namespace interpolation {

/**
 * @brief Configuration for interpolation engine
 */
struct InterpolationConfig {
    // OS1-32 specific parameters
    size_t input_channels = 32;        // OS1-32 has 32 beams
    size_t output_channels = 96;       // Triple density interpolation
    
    // Interpolation parameters
    float spline_tension = 0.5f;       // Catmull-Rom tension parameter (0-1)
    bool enable_discontinuity_detection = true;
    float discontinuity_threshold = 0.1f;  // Threshold for detecting discontinuities
    
    // Performance parameters
    bool enable_simd = true;           // Enable SIMD optimizations when available
    size_t batch_size = 1000;         // Points per batch for parallel processing
    
    // TBB parallelization parameters
    bool enable_tbb = true;            // Enable TBB parallel processing
    size_t grain_size = 64;            // TBB grain size for column-wise parallelization
    size_t max_threads = 0;            // Maximum threads (0 = auto-detect)
    size_t min_columns_for_parallel = 8;  // Minimum columns to justify parallelization
    
    // Memory management
    core::MemoryPool<core::PointCloudSoA>* memory_pool = nullptr;
    
    // Execution mode
    core::ExecutionMode::Mode execution_mode = core::ExecutionMode::Mode::AUTO;
};

/**
 * @brief Performance metrics for interpolation operations
 */
struct InterpolationMetrics {
    std::chrono::nanoseconds interpolation_time{0};
    std::chrono::nanoseconds beam_processing_time{0};
    std::chrono::nanoseconds discontinuity_detection_time{0};
    std::chrono::nanoseconds simd_optimization_time{0};
    std::chrono::nanoseconds parallel_processing_time{0};
    
    size_t input_points{0};
    size_t output_points{0};
    size_t discontinuities_detected{0};
    double interpolation_ratio{0.0};
    double throughput_points_per_second{0.0};
    
    // TBB specific metrics
    size_t columns_processed_parallel{0};
    size_t threads_used{0};
    double parallel_efficiency{0.0};
    
    void reset() {
        interpolation_time = std::chrono::nanoseconds{0};
        beam_processing_time = std::chrono::nanoseconds{0};
        discontinuity_detection_time = std::chrono::nanoseconds{0};
        simd_optimization_time = std::chrono::nanoseconds{0};
        parallel_processing_time = std::chrono::nanoseconds{0};
        input_points = 0;
        output_points = 0;
        discontinuities_detected = 0;
        interpolation_ratio = 0.0;
        throughput_points_per_second = 0.0;
        columns_processed_parallel = 0;
        threads_used = 0;
        parallel_efficiency = 0.0;
    }
    
    void calculateThroughput() {
        if (interpolation_time.count() > 0 && output_points > 0) {
            double seconds = std::chrono::duration<double>(interpolation_time).count();
            throughput_points_per_second = output_points / seconds;
        }
        if (input_points > 0) {
            interpolation_ratio = static_cast<double>(output_points) / input_points;
        }
    }
};

/**
 * @brief Result of interpolation operation
 */
struct InterpolationResult {
    core::PooledPtr<core::PointCloudSoA> interpolated_cloud;
    InterpolationMetrics metrics;
    bool success = false;
    std::string error_message;
    
    // Additional metadata
    size_t beams_processed = 0;
    std::vector<size_t> points_per_beam;
    std::vector<float> beam_altitudes;
};

/**
 * @brief Main interpolation engine for PRISM Phase 2
 * 
 * Implements temporal interpolation using Catmull-Rom cubic splines adapted
 * from the FILC algorithm. Designed specifically for OS1-32 LiDAR data
 * with 32 to 96 channel interpolation.
 * 
 * Features:
 * - Catmull-Rom cubic spline interpolation
 * - OS1-32 beam altitude management
 * - Discontinuity detection
 * - SIMD optimization ready
 * - Thread-safe operation
 * - Memory pool integration
 */
class InterpolationEngine {
public:
    /**
     * @brief Constructor with configuration
     * @param config Interpolation configuration
     */
    explicit InterpolationEngine(const InterpolationConfig& config = InterpolationConfig());
    
    /**
     * @brief Destructor
     */
    ~InterpolationEngine() = default;
    
    // Delete copy operations for thread safety
    InterpolationEngine(const InterpolationEngine&) = delete;
    InterpolationEngine& operator=(const InterpolationEngine&) = delete;
    
    // Allow move operations
    InterpolationEngine(InterpolationEngine&&) = default;
    InterpolationEngine& operator=(InterpolationEngine&&) = default;
    
    /**
     * @brief Configure the interpolation engine
     * @param config New configuration
     */
    void configure(const InterpolationConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const InterpolationConfig& getConfig() const noexcept { return config_; }
    
    /**
     * @brief Perform temporal interpolation on point cloud
     * @param input Input point cloud with OS1-32 data
     * @return Interpolation result with interpolated cloud and metrics
     */
    InterpolationResult interpolate(const core::PointCloudSoA& input);
    
    /**
     * @brief Perform interpolation with custom output channels
     * @param input Input point cloud
     * @param output_channels Number of output channels to generate
     * @return Interpolation result
     */
    InterpolationResult interpolate(const core::PointCloudSoA& input, 
                                  size_t output_channels);
    
    /**
     * @brief Get performance metrics from last interpolation
     * @return Performance metrics
     */
    const InterpolationMetrics& getMetrics() const noexcept;
    
    /**
     * @brief Reset performance metrics
     */
    void resetMetrics();
    
    /**
     * @brief Check if SIMD optimizations are available
     * @return True if SIMD is available and enabled
     */
    bool isSIMDAvailable() const noexcept;
    
    /**
     * @brief Check if TBB parallelization is available
     * @return True if TBB is available and enabled
     */
    bool isTBBAvailable() const noexcept;
    
    /**
     * @brief Get beam altitude manager
     * @return Reference to beam altitude manager
     */
    const BeamAltitudeManager& getBeamManager() const noexcept { return beam_manager_; }
    
    /**
     * @brief Validate input point cloud for interpolation
     * @param input Input point cloud
     * @return True if valid, false otherwise
     */
    bool validateInput(const core::PointCloudSoA& input) const;
    
    /**
     * @brief Get estimated output size for given input
     * @param input_size Input point cloud size
     * @param output_channels Number of output channels
     * @return Estimated output size
     */
    size_t estimateOutputSize(size_t input_size, size_t output_channels) const;

private:
    /**
     * @brief Process individual beam interpolation
     * @param beam_indices Indices of points belonging to a single beam
     * @param input Input point cloud containing source data
     * @param beam_altitude Target altitude for interpolation
     * @param output Output cloud to append results
     * @return Number of interpolated points generated
     */
    size_t processBeam(const std::vector<size_t>& beam_indices,
                      const core::PointCloudSoA& input,
                      float beam_altitude,
                      core::PointCloudSoA& output);
    
    /**
     * @brief Process multiple beams in parallel using TBB
     * @param beam_groups Vector of beam point indices grouped by beam
     * @param input Input point cloud
     * @param output_channels Number of output channels
     * @param output Output cloud to store results
     * @return Total number of interpolated points generated
     */
    size_t processBeamsParallel(const std::vector<std::vector<size_t>>& beam_groups,
                               const core::PointCloudSoA& input,
                               size_t output_channels,
                               core::PointCloudSoA& output);
    
    /**
     * @brief Thread-safe version of processBeam for parallel execution
     * @param beam_indices Indices of points belonging to a single beam
     * @param input Input point cloud containing source data
     * @param beam_altitude Target altitude for interpolation
     * @param local_output Thread-local output cloud
     * @param target_beam_id Target beam identifier
     * @return Number of interpolated points generated
     */
    size_t processBeamThreadSafe(const std::vector<size_t>& beam_indices,
                                const core::PointCloudSoA& input,
                                float beam_altitude,
                                core::PointCloudSoA& local_output,
                                uint16_t target_beam_id);
    
    /**
     * @brief Initialize TBB capabilities and thread management
     */
    void initializeTBB();
    
    /**
     * @brief Detect discontinuities in beam data
     * @param beam_indices Indices of points in the beam
     * @param input Input point cloud
     * @return Vector of discontinuity positions
     */
    std::vector<size_t> detectDiscontinuities(const std::vector<size_t>& beam_indices,
                                             const core::PointCloudSoA& input) const;
    
    /**
     * @brief Separate beam points by ring/channel
     * @param input Input point cloud
     * @return Vector of point indices grouped by beam
     */
    std::vector<std::vector<size_t>> separateBeams(const core::PointCloudSoA& input) const;
    
    /**
     * @brief Initialize SIMD capabilities
     */
    void initializeSIMD();
    
    /**
     * @brief Validate configuration parameters
     * @param config Configuration to validate
     * @return True if valid, false otherwise
     */
    bool validateConfig(const InterpolationConfig& config) const;
    
    /**
     * @brief Update metrics from const methods
     * @param duration Duration to add
     * @param discontinuities Number of discontinuities detected
     */
    void updateMetrics(std::chrono::nanoseconds duration, size_t discontinuities);

private:
    // Configuration
    InterpolationConfig config_;
    
    // Core components
    std::unique_ptr<CatmullRomInterpolator> interpolator_;
    BeamAltitudeManager beam_manager_;
    
    // Performance tracking
    mutable std::mutex metrics_mutex_;
    InterpolationMetrics metrics_;
    
    // SIMD capabilities
    bool simd_available_;
    bool avx2_supported_;
    bool sse_supported_;
    
    // Internal state
    std::atomic<bool> initialized_{false};
    
    // TBB parallelization state
    bool tbb_available_;
    size_t num_threads_;
    
#ifdef PRISM_ENABLE_TBB
    std::unique_ptr<tbb::global_control> thread_limiter_;
    std::unique_ptr<tbb::task_arena> task_arena_;
#endif
};

/**
 * @brief Factory function for creating interpolation engine
 * @param config Configuration for the engine
 * @return Unique pointer to interpolation engine
 */
std::unique_ptr<InterpolationEngine> createInterpolationEngine(
    const InterpolationConfig& config = InterpolationConfig());

/**
 * @brief Utility functions for interpolation
 */
namespace utils {
    /**
     * @brief Calculate optimal batch size for given input size
     * @param input_size Size of input point cloud
     * @param num_threads Number of available threads
     * @return Optimal batch size
     */
    size_t calculateOptimalBatchSize(size_t input_size, size_t num_threads);
    
    /**
     * @brief Convert interpolation metrics to string
     * @param metrics Metrics to convert
     * @return String representation
     */
    std::string metricsToString(const InterpolationMetrics& metrics);
    
    /**
     * @brief Validate OS1-32 specific parameters
     * @param config Configuration to validate
     * @return True if OS1-32 compatible
     */
    bool validateOS132Compatibility(const InterpolationConfig& config);
    
} // namespace utils

} // namespace interpolation
} // namespace prism