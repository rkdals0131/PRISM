#include "prism/interpolation/catmull_rom_interpolator.hpp"
#include "prism/interpolation/simd_kernels.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <mutex>

#ifdef PRISM_ENABLE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#ifdef __x86_64__
#include <cpuid.h>
#include <immintrin.h>
#endif

namespace prism {
namespace interpolation {

// SplineSegment implementation
void SplineSegment::calculateCoefficients() {
    if (!valid || coefficients_valid_) {
        return;
    }
    
    const auto& p0 = control_points[0];
    const auto& p1 = control_points[1];
    const auto& p2 = control_points[2];
    const auto& p3 = control_points[3];
    
    // Catmull-Rom coefficient calculation
    // P(t) = 0.5 * [2*P1 + (-P0 + P2)*t + (2*P0 - 5*P1 + 4*P2 - P3)*t^2 + (-P0 + 3*P1 - 3*P2 + P3)*t^3]
    
    // X coefficients: a*t^3 + b*t^2 + c*t + d
    coeff_x_[0] = 0.5f * (-p0.x + 3.0f*p1.x - 3.0f*p2.x + p3.x);  // t^3 coefficient
    coeff_x_[1] = 0.5f * (2.0f*p0.x - 5.0f*p1.x + 4.0f*p2.x - p3.x); // t^2 coefficient
    coeff_x_[2] = 0.5f * (-p0.x + p2.x);                             // t coefficient
    coeff_x_[3] = p1.x;                                              // constant
    
    // Y coefficients
    coeff_y_[0] = 0.5f * (-p0.y + 3.0f*p1.y - 3.0f*p2.y + p3.y);
    coeff_y_[1] = 0.5f * (2.0f*p0.y - 5.0f*p1.y + 4.0f*p2.y - p3.y);
    coeff_y_[2] = 0.5f * (-p0.y + p2.y);
    coeff_y_[3] = p1.y;
    
    // Z coefficients
    coeff_z_[0] = 0.5f * (-p0.z + 3.0f*p1.z - 3.0f*p2.z + p3.z);
    coeff_z_[1] = 0.5f * (2.0f*p0.z - 5.0f*p1.z + 4.0f*p2.z - p3.z);
    coeff_z_[2] = 0.5f * (-p0.z + p2.z);
    coeff_z_[3] = p1.z;
    
    // Intensity coefficients
    coeff_i_[0] = 0.5f * (-p0.intensity + 3.0f*p1.intensity - 3.0f*p2.intensity + p3.intensity);
    coeff_i_[1] = 0.5f * (2.0f*p0.intensity - 5.0f*p1.intensity + 4.0f*p2.intensity - p3.intensity);
    coeff_i_[2] = 0.5f * (-p0.intensity + p2.intensity);
    coeff_i_[3] = p1.intensity;
    
    coefficients_valid_ = true;
}

ControlPoint SplineSegment::interpolate(float t) const {
    if (!valid || !coefficients_valid_) {
        return ControlPoint();
    }
    
    // Clamp t to valid range
    t = std::max(0.0f, std::min(1.0f, t));
    
    // Calculate powers of t
    float t2 = t * t;
    float t3 = t2 * t;
    
    // Evaluate polynomial: at^3 + bt^2 + ct + d
    ControlPoint result;
    result.x = coeff_x_[0]*t3 + coeff_x_[1]*t2 + coeff_x_[2]*t + coeff_x_[3];
    result.y = coeff_y_[0]*t3 + coeff_y_[1]*t2 + coeff_y_[2]*t + coeff_y_[3];
    result.z = coeff_z_[0]*t3 + coeff_z_[1]*t2 + coeff_z_[2]*t + coeff_z_[3];
    result.intensity = coeff_i_[0]*t3 + coeff_i_[1]*t2 + coeff_i_[2]*t + coeff_i_[3];
    
    // Interpolate timestamp linearly
    result.timestamp = control_points[1].timestamp + t * (control_points[2].timestamp - control_points[1].timestamp);
    
    return result;
}

void SplineSegment::interpolateBatch(const float* t_values, ControlPoint* results, size_t count) const {
    if (!valid || !coefficients_valid_) {
        return;
    }
    
    for (size_t i = 0; i < count; ++i) {
        results[i] = interpolate(t_values[i]);
    }
}

// CatmullRomInterpolator implementation
CatmullRomInterpolator::CatmullRomInterpolator(const CatmullRomConfig& config)
    : config_(config)
    , simd_available_(false)
    , avx2_supported_(false)
    , sse_supported_(false)
    , tbb_available_(false)
{
    initializeSIMD();
    initializeTBB();
    resetStats();
    
    // Initialize the SIMD kernel registry
    SIMDKernelRegistry::initialize();
}

void CatmullRomInterpolator::configure(const CatmullRomConfig& config) {
    config_ = config;
    if (config_.enable_simd && !simd_available_) {
        config_.enable_simd = false; // Disable if not supported
    }
}

bool CatmullRomInterpolator::interpolate(const std::vector<ControlPoint>& control_points,
                                       size_t num_interpolated,
                                       std::vector<ControlPoint>& output) {
    if (!validateControlPoints(control_points)) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Clear output and reserve space
    output.clear();
    size_t estimated_size = control_points.size() + num_interpolated * (control_points.size() - 3);
    output.reserve(estimated_size);
    
    // Build spline segments - use local variable instead of member
    std::vector<SplineSegment> local_segments;
    if (!buildSegments(control_points, local_segments)) {
        return false;
    }
    
    // Process segments (parallel if enabled and beneficial)
    if (config_.enable_tbb && tbb_available_ && 
        local_segments.size() >= config_.min_segments_for_parallel) {
        
        auto parallel_start = std::chrono::high_resolution_clock::now();
        
        // Use parallel processing
        if (!processSegmentsParallel(local_segments, num_interpolated, output)) {
            return false;
        }
        
        auto parallel_end = std::chrono::high_resolution_clock::now();
        stats_.parallel_computation_time += std::chrono::duration_cast<std::chrono::nanoseconds>(parallel_end - parallel_start);
        stats_.parallel_segments_processed = local_segments.size();
        
    } else {
        // Use serial processing with local variables
        for (const auto& segment : local_segments) {
            // Calculate parameter values for interpolation
            std::vector<float> local_t_values;
            if (config_.normalize_parameters) {
                calculateArcLengthParameterization(segment, num_interpolated, local_t_values);
            } else {
                // Uniform parameterization
                local_t_values.reserve(num_interpolated);
                for (size_t i = 0; i < num_interpolated; ++i) {
                    local_t_values.push_back(static_cast<float>(i) / (num_interpolated - 1));
                }
            }
            
            // Interpolate points
            std::vector<ControlPoint> local_points;
            local_points.resize(local_t_values.size());
            
            if (config_.enable_simd && simd_available_) {
                interpolateBatchSIMD(segment, local_t_values.data(), local_points.data(), local_t_values.size());
                stats_.simd_operations++;
            } else {
                interpolateBatchStandard(segment, local_t_values.data(), local_points.data(), local_t_values.size());
            }
            
            // Add interpolated points to output
            output.insert(output.end(), local_points.begin(), local_points.end());
            stats_.points_interpolated += local_points.size();
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.computation_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    stats_.segments_processed += local_segments.size();
    
    return true;
}

bool CatmullRomInterpolator::interpolateCustom(const std::vector<ControlPoint>& control_points,
                                             const std::vector<float>& t_values,
                                             std::vector<ControlPoint>& output) {
    if (!validateControlPoints(control_points) || t_values.empty()) {
        return false;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    output.clear();
    output.reserve(t_values.size());
    
    // Build spline segments - use local variable
    std::vector<SplineSegment> local_segments;
    if (!buildSegments(control_points, local_segments)) {
        return false;
    }
    
    // For custom interpolation, we need to map t_values to segments
    for (float t : t_values) {
        // Find which segment this t belongs to
        size_t segment_idx = static_cast<size_t>(t * (local_segments.size() - 1));
        segment_idx = std::min(segment_idx, local_segments.size() - 1);
        
        // Calculate local t within the segment
        float local_t = t * local_segments.size() - segment_idx;
        local_t = std::max(0.0f, std::min(1.0f, local_t));
        
        // Interpolate point
        ControlPoint interpolated = local_segments[segment_idx].interpolate(local_t);
        output.push_back(interpolated);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.computation_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    stats_.segments_processed += local_segments.size();
    stats_.points_interpolated += output.size();
    
    return true;
}

bool CatmullRomInterpolator::interpolateSegment(const ControlPoint& p0, const ControlPoint& p1,
                                              const ControlPoint& p2, const ControlPoint& p3,
                                              size_t num_points,
                                              std::vector<ControlPoint>& output) {
    if (num_points == 0) {
        return false;
    }
    
    // Create a single segment
    SplineSegment segment;
    segment.control_points = {p0, p1, p2, p3};
    segment.valid = true;
    segment.calculateCoefficients();
    
    // Generate uniform parameter values
    std::vector<float> t_values;
    t_values.reserve(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        t_values.push_back(static_cast<float>(i) / (num_points - 1));
    }
    
    // Interpolate
    output.clear();
    output.resize(num_points);
    
    if (config_.enable_simd && simd_available_) {
        interpolateBatchSIMD(segment, t_values.data(), output.data(), num_points);
        stats_.simd_operations++;
    } else {
        interpolateBatchStandard(segment, t_values.data(), output.data(), num_points);
    }
    
    stats_.segments_processed++;
    stats_.points_interpolated += num_points;
    
    return true;
}

std::vector<size_t> CatmullRomInterpolator::detectDiscontinuities(const std::vector<ControlPoint>& control_points) const {
    std::vector<size_t> discontinuities;
    
    if (control_points.size() < 2) {
        return discontinuities;
    }
    
    for (size_t i = 1; i < control_points.size(); ++i) {
        if (isDiscontinuity(control_points[i-1], control_points[i])) {
            discontinuities.push_back(i);
        }
    }
    
    return discontinuities;
}

bool CatmullRomInterpolator::buildSegments(const std::vector<ControlPoint>& control_points,
                                         std::vector<SplineSegment>& segments) const {
    if (control_points.size() < 4) {
        return false;
    }
    
    segments.clear();
    segments.reserve(control_points.size() - 3);
    
    // Detect discontinuities
    auto discontinuities = detectDiscontinuities(control_points);
    stats_.discontinuities_detected += discontinuities.size();
    
    // Create segments, breaking at discontinuities
    size_t start_idx = 0;
    for (size_t disc_idx : discontinuities) {
        // Create segments up to the discontinuity
        // Ensure we don't exceed bounds: need 4 points for a segment
        size_t end_idx = std::min(disc_idx, control_points.size()) - 3;
        for (size_t i = start_idx; i < end_idx && i + 3 < control_points.size(); ++i) {
            SplineSegment segment;
            segment.control_points[0] = control_points[i];
            segment.control_points[1] = control_points[i + 1];
            segment.control_points[2] = control_points[i + 2];
            segment.control_points[3] = control_points[i + 3];
            segment.valid = true;
            segment.calculateCoefficients();
            segments.push_back(segment);
        }
        start_idx = disc_idx;
    }
    
    // Create remaining segments after last discontinuity
    // Ensure we have enough points (need 4 points for a segment)
    if (start_idx + 3 < control_points.size()) {
        for (size_t i = start_idx; i + 3 < control_points.size(); ++i) {
            SplineSegment segment;
            segment.control_points[0] = control_points[i];
            segment.control_points[1] = control_points[i + 1];
            segment.control_points[2] = control_points[i + 2];
            segment.control_points[3] = control_points[i + 3];
            segment.valid = true;
            segment.calculateCoefficients();
            segments.push_back(segment);
        }
    }
    
    return !segments.empty();
}

void CatmullRomInterpolator::calculateArcLengthParameterization(const SplineSegment& segment,
                                                              size_t num_points,
                                                              std::vector<float>& t_values) const {
    t_values.clear();
    t_values.reserve(num_points);
    
    if (num_points == 0) {
        return;
    }
    
    if (num_points == 1) {
        t_values.push_back(0.5f);
        return;
    }
    
    // Estimate arc length by sampling the derivative
    const size_t arc_samples = 100;
    std::vector<float> arc_lengths;
    arc_lengths.reserve(arc_samples + 1);
    arc_lengths.push_back(0.0f);
    
    float total_length = 0.0f;
    for (size_t i = 1; i <= arc_samples; ++i) {
        float t = static_cast<float>(i) / arc_samples;
        float derivative_mag = calculateDerivativeMagnitude(segment, t);
        float dt = 1.0f / arc_samples;
        total_length += derivative_mag * dt;
        arc_lengths.push_back(total_length);
    }
    
    // Generate parameter values based on arc length
    for (size_t i = 0; i < num_points; ++i) {
        float target_length = (static_cast<float>(i) / (num_points - 1)) * total_length;
        
        // Find corresponding t value
        auto it = std::lower_bound(arc_lengths.begin(), arc_lengths.end(), target_length);
        size_t idx = std::distance(arc_lengths.begin(), it);
        
        float t;
        if (idx == 0) {
            t = 0.0f;
        } else if (idx >= arc_lengths.size()) {
            t = 1.0f;
        } else {
            // Linear interpolation between samples
            float alpha = (target_length - arc_lengths[idx-1]) / (arc_lengths[idx] - arc_lengths[idx-1]);
            t = (static_cast<float>(idx-1) + alpha) / arc_samples;
        }
        
        t_values.push_back(std::max(0.0f, std::min(1.0f, t)));
    }
}

void CatmullRomInterpolator::interpolateBatchSIMD(const SplineSegment& segment,
                                                const float* t_values,
                                                ControlPoint* output,
                                                size_t count) const {
    // Use the optimized SIMD kernel registry
    SIMDKernelRegistry::interpolateBatch(segment, t_values, output, count);
}

void CatmullRomInterpolator::interpolateBatchStandard(const SplineSegment& segment,
                                                    const float* t_values,
                                                    ControlPoint* output,
                                                    size_t count) const {
    for (size_t i = 0; i < count; ++i) {
        output[i] = segment.interpolate(t_values[i]);
    }
}

bool CatmullRomInterpolator::isDiscontinuity(const ControlPoint& p1, const ControlPoint& p2) const {
    float distance = p1.distanceTo(p2);
    return distance > config_.discontinuity_threshold;
}

void CatmullRomInterpolator::initializeSIMD() {
    // Initialize SIMD kernel registry which handles CPU feature detection
    SIMDKernelRegistry::initialize();
    
    // Update local flags for compatibility
    const auto& caps = SIMDKernelRegistry::getCapabilities();
    sse_supported_ = caps.sse_available;
    avx2_supported_ = caps.avx2_available;
    simd_available_ = SIMDKernelRegistry::isAvailable();
}

float CatmullRomInterpolator::calculateDerivativeMagnitude(const SplineSegment& segment, float t) {
    if (!segment.valid || !segment.coefficients_valid_) {
        return 0.0f;
    }
    
    // Derivative: 3at^2 + 2bt + c
    float t2 = t * t;
    
    float dx_dt = 3.0f * segment.coeff_x_[0] * t2 + 2.0f * segment.coeff_x_[1] * t + segment.coeff_x_[2];
    float dy_dt = 3.0f * segment.coeff_y_[0] * t2 + 2.0f * segment.coeff_y_[1] * t + segment.coeff_y_[2];
    float dz_dt = 3.0f * segment.coeff_z_[0] * t2 + 2.0f * segment.coeff_z_[1] * t + segment.coeff_z_[2];
    
    return std::sqrt(dx_dt*dx_dt + dy_dt*dy_dt + dz_dt*dz_dt);
}

bool CatmullRomInterpolator::validateControlPoints(const std::vector<ControlPoint>& control_points) const {
    return control_points.size() >= 4;
}

void CatmullRomInterpolator::resetStats() {
    stats_.reset();
}

bool CatmullRomInterpolator::isSIMDAvailable() const noexcept {
    return config_.enable_simd && simd_available_;
}

bool CatmullRomInterpolator::isTBBAvailable() const noexcept {
    return config_.enable_tbb && tbb_available_;
}

// Utility functions
namespace catmull_rom_utils {

std::vector<ControlPoint> createControlPoints(const float* x, const float* y, 
                                             const float* z, const float* intensity,
                                             size_t count) {
    std::vector<ControlPoint> points;
    points.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        points.emplace_back(x[i], y[i], z[i], intensity[i]);
    }
    
    return points;
}

void extractCoordinates(const std::vector<ControlPoint>& points,
                       std::vector<float>& x, std::vector<float>& y,
                       std::vector<float>& z, std::vector<float>& intensity) {
    size_t count = points.size();
    x.reserve(count);
    y.reserve(count);
    z.reserve(count);
    intensity.reserve(count);
    
    x.clear();
    y.clear();
    z.clear();
    intensity.clear();
    
    for (const auto& point : points) {
        x.push_back(point.x);
        y.push_back(point.y);
        z.push_back(point.z);
        intensity.push_back(point.intensity);
    }
}

size_t calculateInterpolationDensity(float arc_length, float target_point_density) {
    if (arc_length <= 0.0f || target_point_density <= 0.0f) {
        return 1;
    }
    
    size_t density = static_cast<size_t>(arc_length * target_point_density);
    return std::max(size_t(1), density);
}

void smoothControlPoints(std::vector<ControlPoint>& points, size_t window_size) {
    if (points.size() < window_size || window_size < 3) {
        return;
    }
    
    std::vector<ControlPoint> smoothed = points;
    size_t half_window = window_size / 2;
    
    for (size_t i = half_window; i < points.size() - half_window; ++i) {
        float sum_x = 0, sum_y = 0, sum_z = 0, sum_intensity = 0;
        
        for (size_t j = i - half_window; j <= i + half_window; ++j) {
            sum_x += points[j].x;
            sum_y += points[j].y;
            sum_z += points[j].z;
            sum_intensity += points[j].intensity;
        }
        
        float inv_window = 1.0f / window_size;
        smoothed[i].x = sum_x * inv_window;
        smoothed[i].y = sum_y * inv_window;
        smoothed[i].z = sum_z * inv_window;
        smoothed[i].intensity = sum_intensity * inv_window;
    }
    
    points = std::move(smoothed);
}

} // namespace catmull_rom_utils

} // namespace interpolation
} // namespace prism

// Implementation of TBB-specific methods for CatmullRomInterpolator
namespace prism {
namespace interpolation {

void CatmullRomInterpolator::initializeTBB() {
#ifdef PRISM_ENABLE_TBB
    try {
        // TBB should be initialized by the InterpolationEngine
        // Here we just check if it's available
        tbb_available_ = true;
    } catch (const std::exception&) {
        tbb_available_ = false;
    }
#else
    tbb_available_ = false;
#endif
}

bool CatmullRomInterpolator::processSegmentsParallel(
    const std::vector<SplineSegment>& segments,
    size_t num_interpolated,
    std::vector<ControlPoint>& output) {
    
#ifdef PRISM_ENABLE_TBB
    if (!tbb_available_) {
        return false;
    }
    
    // Clear output and estimate size
    output.clear();
    size_t estimated_size = segments.size() * num_interpolated;
    output.reserve(estimated_size);
    
    // Create a vector to store results from each segment
    std::vector<std::vector<ControlPoint>> segment_results(segments.size());
    
    // Mutex for thread-safe statistics updates
    std::mutex stats_mutex;
    
    try {
        tbb::parallel_for(
            tbb::blocked_range<size_t>(0, segments.size(), config_.grain_size),
            [&](const tbb::blocked_range<size_t>& range) {
                size_t local_simd_ops = 0;
                size_t local_points = 0;
                
                for (size_t seg_idx = range.begin(); seg_idx != range.end(); ++seg_idx) {
                    const auto& segment = segments[seg_idx];
                    
                    // Calculate parameter values for interpolation
                    std::vector<float> local_t_values;
                    if (config_.normalize_parameters) {
                        calculateArcLengthParameterization(segment, num_interpolated, local_t_values);
                    } else {
                        // Uniform parameterization
                        local_t_values.reserve(num_interpolated);
                        for (size_t i = 0; i < num_interpolated; ++i) {
                            local_t_values.push_back(static_cast<float>(i) / (num_interpolated - 1));
                        }
                    }
                    
                    // Interpolate points for this segment
                    std::vector<ControlPoint> local_points_vec;
                    local_points_vec.resize(local_t_values.size());
                    
                    if (config_.enable_simd && simd_available_) {
                        interpolateBatchSIMD(segment, local_t_values.data(), local_points_vec.data(), local_t_values.size());
                        local_simd_ops++;
                    } else {
                        interpolateBatchStandard(segment, local_t_values.data(), local_points_vec.data(), local_t_values.size());
                    }
                    
                    // Store results for this segment
                    segment_results[seg_idx] = std::move(local_points_vec);
                    local_points += segment_results[seg_idx].size();
                }
                
                // Update statistics in a thread-safe manner
                {
                    std::lock_guard<std::mutex> lock(stats_mutex);
                    stats_.simd_operations += local_simd_ops;
                    stats_.points_interpolated += local_points;
                }
            }
        );
        
        // Merge all segment results into the final output
        for (const auto& segment_points : segment_results) {
            output.insert(output.end(), segment_points.begin(), segment_points.end());
        }
        
        return true;
        
    } catch (const std::exception&) {
        return false;
    }
    
#else
    // Fallback when TBB is not available
    (void)segments;
    (void)num_interpolated;
    (void)output;
    return false;
#endif
}

} // namespace interpolation
} // namespace prism