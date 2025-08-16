#include "prism/interpolation/interpolation_engine.hpp"
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <cstring>
#include <thread>


namespace prism {
namespace interpolation {

InterpolationEngine::InterpolationEngine(const InterpolationConfig& config)
    : config_(config)
    , beam_manager_(BeamAltitudeConfig{})
{
    if (!validateConfig(config_)) {
        throw std::invalid_argument("Invalid interpolation configuration");
    }
    
    // Create Catmull-Rom interpolator with compatible configuration
    CatmullRomConfig catmull_config;
    catmull_config.tension = config_.spline_tension;
    catmull_config.discontinuity_threshold = config_.discontinuity_threshold;
    interpolator_ = std::make_unique<CatmullRomInterpolator>(catmull_config);
    
    // Configure beam altitude manager
    BeamAltitudeConfig beam_config;
    beam_config.input_beams = config_.input_channels;
    beam_config.output_beams = config_.output_channels;
    beam_manager_.configure(beam_config);
    
    // Initialize beam altitude manager with OS1-32 specifications
    if (!beam_manager_.initializeOS132Beams()) {
        throw std::runtime_error("Failed to initialize OS1-32 beam specifications");
    }
    
    if (!beam_manager_.generateInterpolatedBeams()) {
        throw std::runtime_error("Failed to generate interpolated beam configuration");
    }
    
    initialized_ = true;
    resetMetrics();
}

void InterpolationEngine::configure(const InterpolationConfig& config) {
    if (!validateConfig(config)) {
        throw std::invalid_argument("Invalid interpolation configuration");
    }
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    config_ = config;
    
    // Reconfigure Catmull-Rom interpolator
    if (interpolator_) {
        CatmullRomConfig catmull_config;
        catmull_config.tension = config_.spline_tension;
        catmull_config.discontinuity_threshold = config_.discontinuity_threshold;
        interpolator_->configure(catmull_config);
    }
    
    // Reinitialize beam altitude manager with new configuration
    // This is necessary because the beam manager needs to be properly initialized
    // with OS1-32 beam data before it can generate interpolated beams
    BeamAltitudeConfig beam_config;
    beam_config.input_beams = config_.input_channels;
    beam_config.output_beams = config_.output_channels;
    
    // Create a new beam manager to ensure proper initialization
    beam_manager_ = BeamAltitudeManager(beam_config);
    
    // The new beam manager constructor should have initialized the OS1-32 beams
    // and generated interpolated beams, so we don't need to call generateInterpolatedBeams() again
}

InterpolationResult InterpolationEngine::interpolate(const core::PointCloudSoA& input) {
    return interpolate(input, config_.output_channels);
}

InterpolationResult InterpolationEngine::interpolate(const core::PointCloudSoA& input, 
                                                   size_t output_channels) {
    InterpolationResult result;
    
    if (!initialized_) {
        result.error_message = "InterpolationEngine not properly initialized";
        return result;
    }
    
    // Validate input
    if (!validateInput(input)) {
        result.error_message = "Invalid input point cloud";
        return result;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Acquire output point cloud from memory pool
        if (config_.memory_pool) {
            size_t estimated_size = estimateOutputSize(input.size(), output_channels);
            result.interpolated_cloud = config_.memory_pool->acquire(estimated_size);
        } else {
            // Create regular unique_ptr and wrap it in PooledPtr
            auto raw_cloud = std::make_unique<core::PointCloudSoA>(
                estimateOutputSize(input.size(), output_channels));
            result.interpolated_cloud = core::PooledPtr<core::PointCloudSoA>(
                raw_cloud.release(), core::PoolDeleter<core::PointCloudSoA>{nullptr});
        }
        
        if (!result.interpolated_cloud) {
            result.error_message = "Failed to acquire memory for output point cloud";
            return result;
        }
        
        // Clear output cloud
        result.interpolated_cloud->clear();
        
        // Separate input points by beam/ring
        auto beam_start = std::chrono::high_resolution_clock::now();
        auto beam_groups = separateBeams(input);
        auto beam_end = std::chrono::high_resolution_clock::now();
        
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            metrics_.beam_processing_time += std::chrono::duration_cast<std::chrono::nanoseconds>(beam_end - beam_start);
        }
        
        // Process beams for interpolation (parallel if enabled and beneficial)
        result.beams_processed = beam_groups.size();
        result.points_per_beam.reserve(output_channels);
        result.beam_altitudes.reserve(output_channels);
        
        // Use serial processing for simplicity
        for (size_t target_beam = 0; target_beam < output_channels; ++target_beam) {
            const auto& interpolated_beam = beam_manager_.getInterpolatedBeam(target_beam);
            
            if (interpolated_beam.is_original) {
                // Direct copy for original beams
                if (interpolated_beam.source_beam_low < beam_groups.size()) {
                    const auto& beam_indices = beam_groups[interpolated_beam.source_beam_low];
                    for (size_t idx : beam_indices) {
                        result.interpolated_cloud->addPoint(
                            input.x[idx], input.y[idx], input.z[idx], input.intensity[idx],
                            input.hasColor() ? input.r[idx] : 0,
                            input.hasColor() ? input.g[idx] : 0,
                            input.hasColor() ? input.b[idx] : 0,
                            static_cast<uint16_t>(target_beam)
                        );
                    }
                    result.points_per_beam.push_back(beam_indices.size());
                } else {
                    result.points_per_beam.push_back(0);
                }
            } else {
                // Interpolate between source beams
                size_t points_added = processBeam(beam_groups[interpolated_beam.source_beam_low],
                                                input,
                                                interpolated_beam.altitude_angle,
                                                *result.interpolated_cloud);
                result.points_per_beam.push_back(points_added);
            }
            
            result.beam_altitudes.push_back(interpolated_beam.altitude_angle);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
        
        // Update metrics
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            metrics_.interpolation_time = total_time;
            metrics_.input_points = input.size();
            metrics_.output_points = result.interpolated_cloud->size();
            metrics_.calculateThroughput();
        }
        
        result.metrics = metrics_;
        result.success = true;
        
    } catch (const std::exception& e) {
        result.error_message = e.what();
        result.success = false;
    }
    
    return result;
}

size_t InterpolationEngine::processBeam(const std::vector<size_t>& beam_indices,
                                      const core::PointCloudSoA& input,
                                      float beam_altitude,
                                      core::PointCloudSoA& output) {
    if (beam_indices.empty() || !interpolator_) {
        return 0;
    }
    
    // Suppress unused parameter warning for now
    (void)beam_altitude;
    
    // Create control points from beam data
    std::vector<ControlPoint> control_points;
    control_points.reserve(beam_indices.size());
    
    // Sort beam indices by azimuth angle for proper interpolation
    std::vector<size_t> sorted_indices = beam_indices;
    // For now, assume they're already sorted. In production, we'd sort by atan2(y, x)
    
    for (size_t idx : sorted_indices) {
        control_points.emplace_back(
            input.x[idx], input.y[idx], input.z[idx], 
            input.intensity[idx], 0.0f  // timestamp placeholder
        );
    }
    
    // Interpolate using Catmull-Rom splines
    std::vector<ControlPoint> interpolated_points;
    size_t num_interpolated = std::max(size_t(1), beam_indices.size() * 2); // Double density
    
    if (!interpolator_->interpolate(control_points, num_interpolated, interpolated_points)) {
        return 0;
    }
    
    // Add interpolated points to output cloud
    size_t points_added = 0;
    for (const auto& point : interpolated_points) {
        // Adjust Z coordinate based on target beam altitude
        float adjusted_z = point.z; // In production, adjust based on beam_altitude
        
        output.addPoint(point.x, point.y, adjusted_z, point.intensity);
        ++points_added;
    }
    
    return points_added;
}

std::vector<size_t> InterpolationEngine::detectDiscontinuities(
    const std::vector<size_t>& beam_indices,
    const core::PointCloudSoA& input) const {
    
    std::vector<size_t> discontinuities;
    
    if (!config_.enable_discontinuity_detection || beam_indices.size() < 2) {
        return discontinuities;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 1; i < beam_indices.size(); ++i) {
        size_t idx1 = beam_indices[i - 1];
        size_t idx2 = beam_indices[i];
        
        // Calculate Euclidean distance
        float dx = input.x[idx2] - input.x[idx1];
        float dy = input.y[idx2] - input.y[idx1];
        float dz = input.z[idx2] - input.z[idx1];
        float distance = std::sqrt(dx*dx + dy*dy + dz*dz);
        
        if (distance > config_.discontinuity_threshold) {
            discontinuities.push_back(i);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    
    // Update metrics in a non-const context
    const_cast<InterpolationEngine*>(this)->updateMetrics(duration, discontinuities.size());
    
    return discontinuities;
}

std::vector<std::vector<size_t>> InterpolationEngine::separateBeams(const core::PointCloudSoA& input) const {
    std::vector<std::vector<size_t>> beam_groups(config_.input_channels);
    
    for (size_t i = 0; i < input.size(); ++i) {
        // If ring data is available, use it. Otherwise, calculate from geometry
        uint16_t ring = 0;
        if (input.hasRing() && i < input.ring.size()) {
            ring = input.ring[i];
        } else {
            // Calculate ring from elevation angle
            float elevation = std::atan2(input.z[i], std::sqrt(input.x[i] * input.x[i] + input.y[i] * input.y[i]));
            float normalized_elevation = (elevation + M_PI/6) / (M_PI/3);  // Assuming ±30 degree FOV
            ring = static_cast<uint16_t>(normalized_elevation * (config_.input_channels - 1));
            ring = std::min(ring, static_cast<uint16_t>(config_.input_channels - 1));
        }
        
        if (ring < config_.input_channels) {
            beam_groups[ring].push_back(i);
        }
    }
    
    // Sort each beam by azimuth angle for proper interpolation order
    for (auto& beam : beam_groups) {
        if (beam.size() > 1) {
            std::sort(beam.begin(), beam.end(), [&input](size_t a, size_t b) {
                float azimuth_a = std::atan2(input.y[a], input.x[a]);
                float azimuth_b = std::atan2(input.y[b], input.x[b]);
                return azimuth_a < azimuth_b;
            });
        }
    }
    
    return beam_groups;
}


bool InterpolationEngine::validateConfig(const InterpolationConfig& config) const {
    if (config.input_channels == 0 || config.output_channels == 0) {
        return false;
    }
    
    if (config.spline_tension < 0.0f || config.spline_tension > 1.0f) {
        return false;
    }
    
    if (config.discontinuity_threshold <= 0.0f) {
        return false;
    }
    
    // Memory pool is optional - nullptr is valid
    // The engine will allocate memory directly if no pool is provided
    
    return true;
}

bool InterpolationEngine::validateInput(const core::PointCloudSoA& input) const {
    if (input.empty()) {
        return false;
    }
    
    if (!input.validate()) {
        return false;
    }
    
    // Check if ring data is available for beam separation
    // NOTE: Ring data is generated in convertToSoA if not present in original data
    // So we'll proceed even without ring data in the input
    // if (!input.hasRing()) {
    //     // In production, we might generate ring data from geometry
    //     return false;
    // }
    
    return true;
}

size_t InterpolationEngine::estimateOutputSize(size_t input_size, size_t output_channels) const {
    if (input_size == 0 || config_.input_channels == 0) {
        return 0;
    }
    
    // Simple estimation: scale by channel ratio
    double channel_ratio = static_cast<double>(output_channels) / config_.input_channels;
    return static_cast<size_t>(input_size * channel_ratio * 1.2); // 20% buffer
}

void InterpolationEngine::resetMetrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_.reset();
}


void InterpolationEngine::updateMetrics(std::chrono::nanoseconds duration, size_t discontinuities) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_.discontinuity_detection_time += duration;
    metrics_.discontinuities_detected += discontinuities;
}

const InterpolationMetrics& InterpolationEngine::getMetrics() const noexcept {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

// Factory function
std::unique_ptr<InterpolationEngine> createInterpolationEngine(const InterpolationConfig& config) {
    return std::make_unique<InterpolationEngine>(config);
}

// Utility functions
namespace utils {

size_t calculateOptimalBatchSize(size_t input_size, size_t num_threads) {
    if (input_size == 0 || num_threads == 0) {
        return 1000; // Default batch size
    }
    
    // Aim for 4-8 batches per thread
    size_t target_batches = num_threads * 6;
    size_t batch_size = input_size / target_batches;
    
    // Clamp to reasonable range
    return std::max(size_t(100), std::min(size_t(10000), batch_size));
}

std::string metricsToString(const InterpolationMetrics& metrics) {
    std::ostringstream oss;
    oss << "InterpolationMetrics {\n";
    oss << "  Interpolation time: " << metrics.interpolation_time.count() << " ns\n";
    oss << "  Beam processing time: " << metrics.beam_processing_time.count() << " ns\n";
    oss << "  Discontinuity detection time: " << metrics.discontinuity_detection_time.count() << " ns\n";
    oss << "  Input points: " << metrics.input_points << "\n";
    oss << "  Output points: " << metrics.output_points << "\n";
    oss << "  Discontinuities detected: " << metrics.discontinuities_detected << "\n";
    oss << "  Interpolation ratio: " << metrics.interpolation_ratio << "\n";
    oss << "  Throughput: " << metrics.throughput_points_per_second << " points/sec\n";
    oss << "}";
    return oss.str();
}

bool validateOS132Compatibility(const InterpolationConfig& config) {
    return config.input_channels == 32 && 
           config.output_channels >= 32 && 
           config.output_channels <= 128;
}

} // namespace utils

} // namespace interpolation
} // namespace prism