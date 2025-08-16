#pragma once

#include <vector>
#include <array>
#include <cstddef>
#include <memory>
#include <chrono>
#include <cmath>


namespace prism {
namespace interpolation {

/**
 * @brief Control point for Catmull-Rom spline interpolation
 */
struct ControlPoint {
    float x, y, z;           // 3D coordinates
    float intensity;         // Intensity value
    float timestamp;         // Temporal information
    
    ControlPoint() : x(0), y(0), z(0), intensity(0), timestamp(0) {}
    ControlPoint(float px, float py, float pz, float pi, float pt = 0)
        : x(px), y(py), z(pz), intensity(pi), timestamp(pt) {}
    
    // Linear interpolation between two control points
    ControlPoint lerp(const ControlPoint& other, float t) const {
        return ControlPoint(
            x + t * (other.x - x),
            y + t * (other.y - y),
            z + t * (other.z - z),
            intensity + t * (other.intensity - intensity),
            timestamp + t * (other.timestamp - timestamp)
        );
    }
    
    // Distance to another point
    float distanceTo(const ControlPoint& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        float dz = z - other.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }
};

/**
 * @brief Spline segment for efficient batch processing
 */
struct SplineSegment {
    std::array<ControlPoint, 4> control_points;  // P0, P1, P2, P3
    float t_start = 0.0f;                        // Start parameter
    float t_end = 1.0f;                          // End parameter
    bool valid = false;                          // Whether segment is valid
    
    // Calculate spline coefficient matrices
    void calculateCoefficients();
    
    // Interpolate at parameter t (0 <= t <= 1)
    ControlPoint interpolate(float t) const;
    
    // Batch interpolate multiple t values
    void interpolateBatch(const float* t_values, ControlPoint* results, size_t count) const;
    
    // Cached coefficient matrices for efficient computation
    float coeff_x_[4];  // x coefficients
    float coeff_y_[4];  // y coefficients
    float coeff_z_[4];  // z coefficients
    float coeff_i_[4];  // intensity coefficients
    bool coefficients_valid_ = false;
};

/**
 * @brief Configuration for Catmull-Rom interpolation
 */
struct CatmullRomConfig {
    float tension = 0.5f;                    // Spline tension (0-1)
    float discontinuity_threshold = 0.1f;    // Distance threshold for discontinuities
    bool normalize_parameters = true;        // Normalize t parameters by arc length
    size_t max_points_per_segment = 100;     // Maximum points to interpolate per segment
};

/**
 * @brief Statistics for Catmull-Rom interpolation operations
 */
struct CatmullRomStats {
    size_t segments_processed = 0;
    size_t points_interpolated = 0;
    size_t discontinuities_detected = 0;
    std::chrono::nanoseconds computation_time{0};
    
    void reset() {
        segments_processed = 0;
        points_interpolated = 0;
        discontinuities_detected = 0;
        computation_time = std::chrono::nanoseconds{0};
    }
};

/**
 * @brief Catmull-Rom cubic spline interpolator adapted from FILC algorithm
 * 
 * Implements the Fast Interpolated Lidar Clustering (FILC) algorithm's
 * Catmull-Rom spline interpolation with optimizations for real-time
 * LiDAR point cloud processing.
 * 
 * Features:
 * - Cubic Catmull-Rom spline interpolation
 * - Discontinuity detection and handling
 * - Arc-length parameterization for smooth interpolation
 * - Memory-efficient segment-based processing
 */
class CatmullRomInterpolator {
public:
    /**
     * @brief Constructor with configuration
     * @param config Interpolation configuration
     */
    explicit CatmullRomInterpolator(const CatmullRomConfig& config = CatmullRomConfig());
    
    /**
     * @brief Destructor
     */
    ~CatmullRomInterpolator() = default;
    
    // Delete copy operations
    CatmullRomInterpolator(const CatmullRomInterpolator&) = delete;
    CatmullRomInterpolator& operator=(const CatmullRomInterpolator&) = delete;
    
    // Allow move operations
    CatmullRomInterpolator(CatmullRomInterpolator&&) = default;
    CatmullRomInterpolator& operator=(CatmullRomInterpolator&&) = default;
    
    /**
     * @brief Configure the interpolator
     * @param config New configuration
     */
    void configure(const CatmullRomConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const CatmullRomConfig& getConfig() const noexcept { return config_; }
    
    /**
     * @brief Interpolate points along a Catmull-Rom spline
     * @param control_points Input control points (minimum 4 required)
     * @param num_interpolated Number of points to interpolate between segments
     * @param output Output vector for interpolated points
     * @return True if interpolation succeeded
     */
    bool interpolate(const std::vector<ControlPoint>& control_points,
                    size_t num_interpolated,
                    std::vector<ControlPoint>& output);
    
    /**
     * @brief Interpolate with custom parameter values
     * @param control_points Input control points
     * @param t_values Parameter values for interpolation (0-1 range)
     * @param output Output vector for interpolated points
     * @return True if interpolation succeeded
     */
    bool interpolateCustom(const std::vector<ControlPoint>& control_points,
                          const std::vector<float>& t_values,
                          std::vector<ControlPoint>& output);
    
    /**
     * @brief Interpolate a single segment between four control points
     * @param p0 First control point
     * @param p1 Second control point (start of segment)
     * @param p2 Third control point (end of segment)
     * @param p3 Fourth control point
     * @param num_points Number of points to interpolate
     * @param output Output vector for interpolated points
     * @return True if interpolation succeeded
     */
    bool interpolateSegment(const ControlPoint& p0, const ControlPoint& p1,
                          const ControlPoint& p2, const ControlPoint& p3,
                          size_t num_points,
                          std::vector<ControlPoint>& output);
    
    /**
     * @brief Detect discontinuities in control points
     * @param control_points Input control points
     * @return Vector of indices where discontinuities occur
     */
    std::vector<size_t> detectDiscontinuities(const std::vector<ControlPoint>& control_points) const;
    
    /**
     * @brief Get interpolation statistics
     * @return Current statistics
     */
    const CatmullRomStats& getStats() const noexcept { return stats_; }
    
    /**
     * @brief Reset statistics
     */
    void resetStats();
    
    
    /**
     * @brief Validate control points for interpolation
     * @param control_points Control points to validate
     * @return True if valid for interpolation
     */
    bool validateControlPoints(const std::vector<ControlPoint>& control_points) const;

private:
    /**
     * @brief Build spline segments from control points
     * @param control_points Input control points
     * @param segments Output segments
     * @return True if segments were built successfully
     */
    bool buildSegments(const std::vector<ControlPoint>& control_points,
                      std::vector<SplineSegment>& segments) const;
    
    
    /**
     * @brief Calculate arc-length parameterization
     * @param segment Spline segment
     * @param num_points Number of points to parameterize
     * @param t_values Output parameter values
     */
    void calculateArcLengthParameterization(const SplineSegment& segment,
                                          size_t num_points,
                                          std::vector<float>& t_values) const;
    
    
    /**
     * @brief Standard batch interpolation
     * @param segment Spline segment
     * @param t_values Parameter values
     * @param output Output points
     * @param count Number of points to interpolate
     */
    void interpolateBatchStandard(const SplineSegment& segment,
                                const float* t_values,
                                ControlPoint* output,
                                size_t count) const;
    
    /**
     * @brief Check if discontinuity exists between two points
     * @param p1 First point
     * @param p2 Second point
     * @return True if discontinuity detected
     */
    bool isDiscontinuity(const ControlPoint& p1, const ControlPoint& p2) const;
    
    
    /**
     * @brief Calculate Catmull-Rom basis functions
     * @param t Parameter value (0-1)
     * @param tension Spline tension
     * @param basis Output basis function values
     */
    static void calculateBasisFunctions(float t, float tension, float basis[4]);
    
    /**
     * @brief Calculate Catmull-Rom derivatives for arc-length parameterization
     * @param segment Spline segment
     * @param t Parameter value
     * @return Derivative magnitude
     */
    static float calculateDerivativeMagnitude(const SplineSegment& segment, float t);

private:
    // Configuration
    CatmullRomConfig config_;
    
    // Statistics
    mutable CatmullRomStats stats_;
    
    
    // Note: Removed temp_t_values_, temp_points_, temp_segments_ member variables
    // to ensure thread safety. These are now local variables in each method.
};

/**
 * @brief Utility functions for Catmull-Rom interpolation
 */
namespace catmull_rom_utils {
    /**
     * @brief Create control points from coordinate arrays
     * @param x X coordinates
     * @param y Y coordinates
     * @param z Z coordinates
     * @param intensity Intensity values
     * @param count Number of points
     * @return Vector of control points
     */
    std::vector<ControlPoint> createControlPoints(const float* x, const float* y, 
                                                 const float* z, const float* intensity,
                                                 size_t count);
    
    /**
     * @brief Extract coordinates from control points
     * @param points Control points
     * @param x Output X coordinates
     * @param y Output Y coordinates
     * @param z Output Z coordinates
     * @param intensity Output intensity values
     */
    void extractCoordinates(const std::vector<ControlPoint>& points,
                          std::vector<float>& x, std::vector<float>& y,
                          std::vector<float>& z, std::vector<float>& intensity);
    
    /**
     * @brief Calculate optimal interpolation density for given arc length
     * @param arc_length Total arc length of spline
     * @param target_point_density Desired points per unit length
     * @return Number of interpolation points
     */
    size_t calculateInterpolationDensity(float arc_length, float target_point_density);
    
    /**
     * @brief Smooth control points to reduce noise
     * @param points Input/output control points
     * @param window_size Smoothing window size
     */
    void smoothControlPoints(std::vector<ControlPoint>& points, size_t window_size = 3);
    
} // namespace catmull_rom_utils

} // namespace interpolation
} // namespace prism