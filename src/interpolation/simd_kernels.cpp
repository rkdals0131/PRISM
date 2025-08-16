#include "prism/interpolation/simd_kernels.hpp"
#include "prism/interpolation/catmull_rom_interpolator.hpp"
#include <algorithm>
#include <cstring>

namespace prism {
namespace interpolation {

// Static member definitions
bool InterpolationKernelRegistry::initialized_ = false;

// InterpolationKernelRegistry implementation
void InterpolationKernelRegistry::initialize() {
    if (initialized_) {
        return;
    }
    
    initialized_ = true;
}

void InterpolationKernelRegistry::interpolateBatch(const SplineSegment& segment,
                                         const float* t_values,
                                         ControlPoint* output,
                                         size_t count) {
    if (!initialized_) {
        initialize();
    }
    
    if (!segment.coefficients_valid_ || count == 0) {
        return;
    }
    
    // Use scalar implementation for all cases
    catmull_rom_scalar_kernel(segment.coeff_x_, segment.coeff_y_, 
                            segment.coeff_z_, segment.coeff_i_,
                            t_values, output, count);
}

// Scalar Implementation (fallback)
void catmull_rom_scalar_kernel(const float* coeff_x, const float* coeff_y,
                              const float* coeff_z, const float* coeff_i,
                              const float* t_values, ControlPoint* output,
                              size_t count) {
    for (size_t i = 0; i < count; ++i) {
        float t = t_values[i];
        
        // Clamp t to valid range
        t = std::max(0.0f, std::min(1.0f, t));
        
        // Calculate powers of t
        float t2 = t * t;
        float t3 = t2 * t;
        
        // Evaluate polynomial: at^3 + bt^2 + ct + d
        output[i].x = coeff_x[0] * t3 + coeff_x[1] * t2 + coeff_x[2] * t + coeff_x[3];
        output[i].y = coeff_y[0] * t3 + coeff_y[1] * t2 + coeff_y[2] * t + coeff_y[3];
        output[i].z = coeff_z[0] * t3 + coeff_z[1] * t2 + coeff_z[2] * t + coeff_z[3];
        output[i].intensity = coeff_i[0] * t3 + coeff_i[1] * t2 + coeff_i[2] * t + coeff_i[3];
        output[i].timestamp = 0.0f; // Will be set appropriately by calling code
    }
}


} // namespace interpolation
} // namespace prism