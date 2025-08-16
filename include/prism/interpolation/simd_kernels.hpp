#pragma once

#include <cstddef>
#include <cstdint>

namespace prism {
namespace interpolation {

// Forward declaration
struct ControlPoint;
struct SplineSegment;

/**
 * @brief Simple kernel registry for Catmull-Rom interpolation
 * 
 * Provides scalar implementation only for simplicity and readability.
 */
class InterpolationKernelRegistry {
public:
    /**
     * @brief Initialize the kernel registry
     */
    static void initialize();
    
    /**
     * @brief Batch interpolate using scalar kernel
     * 
     * @param segment Spline segment with precomputed coefficients
     * @param t_values Array of parameter values [0,1]
     * @param output Array of output control points
     * @param count Number of points to interpolate
     */
    static void interpolateBatch(const SplineSegment& segment,
                                const float* t_values,
                                ControlPoint* output,
                                size_t count);

private:
    static bool initialized_;
};

/**
 * @brief Scalar Catmull-Rom interpolation
 * 
 * Standard implementation for all systems.
 * 
 * @param coeff_x X-coordinate coefficients [a, b, c, d] for polynomial
 * @param coeff_y Y-coordinate coefficients [a, b, c, d] for polynomial
 * @param coeff_z Z-coordinate coefficients [a, b, c, d] for polynomial
 * @param coeff_i Intensity coefficients [a, b, c, d] for polynomial
 * @param t_values Array of parameter values
 * @param output Array of output control points
 * @param count Number of points to process
 */
void catmull_rom_scalar_kernel(const float* coeff_x, const float* coeff_y,
                              const float* coeff_z, const float* coeff_i,
                              const float* t_values, ControlPoint* output,
                              size_t count);

} // namespace interpolation
} // namespace prism