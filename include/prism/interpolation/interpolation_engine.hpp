#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include <functional>
#include <chrono>

#include "prism/core/point_cloud_soa.hpp"
#include "prism/core/memory_pool.hpp"
#include "prism/core/execution_mode.hpp"
#include "prism/utils/common_types.hpp"
#include "catmull_rom_interpolator.hpp"
#include "beam_altitude_manager.hpp"

namespace prism {
namespace interpolation {

/**
 * @brief Configuration for interpolation engine
 */
struct InterpolationConfig {
    // OS1-32 specific parameters
    size_t input_channels = 32;         // OS1-32 has 32 beams
    double scale_factor = 2.0;          // Vertical (beam count) scale (e.g., 2.0 => 64)
    
    // Interpolation parameters
    float spline_tension = 0.5f;       // Catmull-Rom tension parameter (0-1)
    bool enable_discontinuity_detection = true;
    float discontinuity_threshold = 0.1f;  // Threshold for detecting discontinuities
    
    // Memory management
    core::MemoryPool<core::PointCloudSoA>* memory_pool = nullptr;
    
    // Execution mode
    core::ExecutionMode::Mode execution_mode = core::ExecutionMode::Mode::SINGLE_THREAD;
    
    /**
     * @brief Load configuration from YAML node
     */
    void loadFromYaml(const YAML::Node& node) {
        using prism::utils::ConfigLoader;
        
        // Interpolation parameters
        scale_factor = ConfigLoader::readNestedParam(node, 
            "interpolation.scale_factor", scale_factor);
        spline_tension = ConfigLoader::readNestedParam(node, 
            "interpolation.catmull_rom.tension", spline_tension);
        enable_discontinuity_detection = ConfigLoader::readNestedParam(node,
            "interpolation.discontinuity.enabled", enable_discontinuity_detection);
        discontinuity_threshold = ConfigLoader::readNestedParam(node,
            "interpolation.discontinuity.threshold", discontinuity_threshold);
            
        // Execution mode from performance settings
        std::string mode = ConfigLoader::readNestedParam(node, 
            "system.execution_mode", std::string("single_thread"));
        if (mode == "single_thread" || mode == "multi_thread" || mode == "gpu") {
            // For now, only single thread is supported
            // Multi-thread and GPU modes can be added in future
            execution_mode = core::ExecutionMode::Mode::SINGLE_THREAD;
        }
    }
    
    /**
     * @brief Validate configuration
     */
    bool validate() const {
        return input_channels > 0 && 
               scale_factor >= 1.0 &&
               spline_tension >= 0.0f && spline_tension <= 1.0f &&
               discontinuity_threshold > 0.0f;
    }
    
    size_t getOutputBeams() const {
        return static_cast<size_t>(std::max(1.0, std::round(input_channels * scale_factor)));
    }
};

/**
 * @brief Performance metrics for interpolation operations
 * Now using BaseMetrics pattern for consistent copying
 */
class InterpolationMetrics : public prism::utils::BaseMetrics<InterpolationMetrics> {
public:
    // Using atomic helpers for thread-safe counters
    prism::utils::AtomicCounter input_points;
    prism::utils::AtomicCounter output_points;
    prism::utils::AtomicCounter discontinuities_detected;
    prism::utils::ThroughputMetric throughput;
    prism::utils::AtomicGauge interpolation_ratio;
    
    // Timing components (in milliseconds)
    prism::utils::AtomicGauge interpolation_time_ms;
    prism::utils::AtomicGauge beam_processing_time_ms;
    prism::utils::AtomicGauge discontinuity_detection_time_ms;
    
    /**
     * @brief Required implementation for BaseMetrics
     */
    void resetImpl() {
        input_points.reset();
        output_points.reset();
        discontinuities_detected.reset();
        throughput.reset();
        interpolation_ratio.reset();
        interpolation_time_ms.reset();
        beam_processing_time_ms.reset();
        discontinuity_detection_time_ms.reset();
    }
    
    /**
     * @brief Record a complete interpolation operation
     */
    void recordOperation(size_t in_points, size_t out_points, double processing_ms) {
        input_points.increment(in_points);
        output_points.increment(out_points);
        
        if (in_points > 0) {
            interpolation_ratio.set(static_cast<double>(out_points) / in_points);
        }
        
        throughput.recordOperation(processing_ms, out_points);
        interpolation_time_ms.set(processing_ms);
    }
    
    /**
     * @brief Get performance summary
     */
    struct Summary {
        double throughput_points_sec;
        double avg_latency_ms;
        double interpolation_ratio;
        size_t total_discontinuities;
    };
    
    Summary getSummary() const {
        return {
            throughput.getThroughput(),
            throughput.getAverageLatency(),
            interpolation_ratio.get(),
            static_cast<size_t>(discontinuities_detected.get())
        };
    }
};

/**
 * @brief Result of interpolation operation
 * Now extends BaseResult for common metadata
 */
struct InterpolationResult : public prism::utils::BaseResult {
    core::PooledPtr<core::PointCloudSoA> interpolated_cloud;
    InterpolationMetrics metrics;
    bool success = false;
    std::string error_message;
    
    // Additional metadata
    size_t beams_processed = 0;
    std::vector<size_t> points_per_beam;
    std::vector<float> beam_altitudes;
    
    // Default constructor
    InterpolationResult() = default;
    
    // Move constructor
    InterpolationResult(InterpolationResult&& other) noexcept = default;
    
    // Move assignment
    InterpolationResult& operator=(InterpolationResult&& other) noexcept = default;
    
    // Delete copy operations since we have unique_ptr
    InterpolationResult(const InterpolationResult&) = delete;
    InterpolationResult& operator=(const InterpolationResult&) = delete;
    
    /**
     * @brief Override clear to handle custom members
     */
    void clear() override {
        prism::utils::BaseResult::clear();  // Call base clear
        interpolated_cloud.reset();
        metrics.reset();
        success = false;
        error_message.clear();
        beams_processed = 0;
        points_per_beam.clear();
        beam_altitudes.clear();
    }
    
    /**
     * @brief Check if result is valid
     */
    bool isValid() const {
        return success && interpolated_cloud && !interpolated_cloud->empty();
    }
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
    // Vertical (inter-beam) interpolation: blend matched azimuth samples
    size_t processBeam(const std::vector<size_t>& beam_indices_low,
                      const std::vector<size_t>& beam_indices_high,
                      float interpolation_weight,
                      const core::PointCloudSoA& input,
                      uint16_t target_ring,
                      core::PointCloudSoA& output);
    
    
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
    
    // Internal state
    std::atomic<bool> initialized_{false};
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