#pragma once

#include <cstddef>
#include <cstdint>

namespace prism {
namespace interpolation {

// Forward declaration
struct ControlPoint;
struct SplineSegment;

/**
 * @brief SIMD optimization capabilities detected at runtime
 */
struct SIMDCapabilities {
    bool sse_available = false;
    bool avx_available = false;
    bool avx2_available = false;
    bool fma_available = false;
    
    /**
     * @brief Detect CPU SIMD capabilities at runtime
     */
    static SIMDCapabilities detect();
    
    /**
     * @brief Get the best available SIMD instruction set
     * @return String description of best available SIMD
     */
    const char* getBestInstructionSet() const;
};

/**
 * @brief SIMD kernel function pointer types
 */
using CatmullRomKernel4f = void(*)(const float* coeff_x, const float* coeff_y, 
                                   const float* coeff_z, const float* coeff_i,
                                   const float* t_values, ControlPoint* output, 
                                   size_t count);

using CatmullRomKernel8f = void(*)(const float* coeff_x, const float* coeff_y, 
                                   const float* coeff_z, const float* coeff_i,
                                   const float* t_values, ControlPoint* output, 
                                   size_t count);

/**
 * @brief SIMD kernel registry for Catmull-Rom interpolation
 * 
 * This class manages runtime selection of optimal SIMD kernels based on
 * CPU capabilities. It provides a unified interface for accessing the
 * fastest available implementation.
 */
class SIMDKernelRegistry {
public:
    /**
     * @brief Initialize the kernel registry
     * 
     * Detects CPU capabilities and sets up function pointers
     * to the optimal implementations.
     */
    static void initialize();
    
    /**
     * @brief Check if SIMD kernels are available
     * @return True if any SIMD optimization is available
     */
    static bool isAvailable();
    
    /**
     * @brief Get SIMD capabilities
     * @return Detected SIMD capabilities
     */
    static const SIMDCapabilities& getCapabilities();
    
    /**
     * @brief Batch interpolate using optimal SIMD kernel
     * 
     * Automatically selects the best available implementation (AVX2, SSE, or scalar)
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
    
    /**
     * @brief Get performance statistics for last operation
     * @return Estimated speedup compared to scalar implementation
     */
    static float getLastSpeedup();

private:
    static SIMDCapabilities capabilities_;
    static CatmullRomKernel4f sse_kernel_;
    static CatmullRomKernel8f avx2_kernel_;
    static bool initialized_;
    static float last_speedup_;
};

/**
 * @brief AVX2 optimized Catmull-Rom interpolation (8 floats at once)
 * 
 * Processes 8 parameter values simultaneously using 256-bit SIMD registers.
 * Requires AVX2 instruction set support.
 * 
 * @param coeff_x X-coordinate coefficients [a, b, c, d] for polynomial
 * @param coeff_y Y-coordinate coefficients [a, b, c, d] for polynomial  
 * @param coeff_z Z-coordinate coefficients [a, b, c, d] for polynomial
 * @param coeff_i Intensity coefficients [a, b, c, d] for polynomial
 * @param t_values Array of parameter values (must be aligned to 32 bytes)
 * @param output Array of output control points
 * @param count Number of points to process (will be rounded down to multiple of 8)
 */
void catmull_rom_avx2_kernel(const float* coeff_x, const float* coeff_y,
                            const float* coeff_z, const float* coeff_i,
                            const float* t_values, ControlPoint* output,
                            size_t count);

/**
 * @brief SSE optimized Catmull-Rom interpolation (4 floats at once)
 * 
 * Processes 4 parameter values simultaneously using 128-bit SIMD registers.
 * Requires SSE instruction set support.
 * 
 * @param coeff_x X-coordinate coefficients [a, b, c, d] for polynomial
 * @param coeff_y Y-coordinate coefficients [a, b, c, d] for polynomial
 * @param coeff_z Z-coordinate coefficients [a, b, c, d] for polynomial  
 * @param coeff_i Intensity coefficients [a, b, c, d] for polynomial
 * @param t_values Array of parameter values (must be aligned to 16 bytes)
 * @param output Array of output control points
 * @param count Number of points to process (will be rounded down to multiple of 4)
 */
void catmull_rom_sse_kernel(const float* coeff_x, const float* coeff_y,
                           const float* coeff_z, const float* coeff_i,
                           const float* t_values, ControlPoint* output,
                           size_t count);

/**
 * @brief Scalar fallback Catmull-Rom interpolation
 * 
 * Standard implementation for systems without SIMD support or when
 * SIMD alignment requirements cannot be met.
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

/**
 * @brief Utility functions for SIMD memory management
 */
namespace simd_utils {
    /**
     * @brief Check if pointer is aligned for SIMD operations
     * @param ptr Pointer to check
     * @param alignment Required alignment (16 for SSE, 32 for AVX2)
     * @return True if properly aligned
     */
    bool isAligned(const void* ptr, size_t alignment);
    
    /**
     * @brief Align memory size for SIMD operations
     * @param size Input size
     * @param simd_width SIMD width (4 for SSE, 8 for AVX2)
     * @return Size aligned to SIMD boundary
     */
    size_t alignSize(size_t size, size_t simd_width);
    
    /**
     * @brief Get optimal SIMD width for current CPU
     * @return 8 for AVX2, 4 for SSE, 1 for scalar
     */
    size_t getOptimalSIMDWidth();
    
    /**
     * @brief Prefetch data for SIMD operations
     * @param ptr Pointer to prefetch
     * @param size Size of data to prefetch
     */
    void prefetchData(const void* ptr, size_t size);
    
} // namespace simd_utils

} // namespace interpolation
} // namespace prism