#include "prism/interpolation/simd_kernels.hpp"
#include "prism/interpolation/catmull_rom_interpolator.hpp"
#include <algorithm>
#include <cstring>

#ifdef __x86_64__
#include <cpuid.h>
#include <immintrin.h>
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#endif

namespace prism {
namespace interpolation {

// Static member definitions
SIMDCapabilities SIMDKernelRegistry::capabilities_;
CatmullRomKernel4f SIMDKernelRegistry::sse_kernel_ = nullptr;
CatmullRomKernel8f SIMDKernelRegistry::avx2_kernel_ = nullptr;
bool SIMDKernelRegistry::initialized_ = false;
float SIMDKernelRegistry::last_speedup_ = 1.0f;

// SIMDCapabilities implementation
SIMDCapabilities SIMDCapabilities::detect() {
    SIMDCapabilities caps;
    
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    
    // Check basic CPUID support
    __cpuid(0, eax, ebx, ecx, edx);
    if (eax < 1) {
        return caps;
    }
    
    // Check for SSE support (CPUID.01H:EDX.SSE[bit 25])
    __cpuid(1, eax, ebx, ecx, edx);
    caps.sse_available = (edx & (1 << 25)) != 0;
    
    // Check for AVX support (CPUID.01H:ECX.AVX[bit 28])
    bool avx_supported = (ecx & (1 << 28)) != 0;
    bool osxsave = (ecx & (1 << 27)) != 0;
    
    if (avx_supported && osxsave) {
        // Check if OS supports AVX (XCR0[2:1] = '11b')
        unsigned long long xcr0 = _xgetbv(0);
        if ((xcr0 & 0x6) == 0x6) {
            caps.avx_available = true;
            
            // Check for AVX2 support (CPUID.07H:EBX.AVX2[bit 5])
            if (eax >= 7) {
                __cpuid_count(7, 0, eax, ebx, ecx, edx);
                caps.avx2_available = (ebx & (1 << 5)) != 0;
                caps.fma_available = (ebx & (1 << 12)) != 0;
            }
        }
    }
#endif
    
    return caps;
}

const char* SIMDCapabilities::getBestInstructionSet() const {
    if (avx2_available) return "AVX2";
    if (avx_available) return "AVX";
    if (sse_available) return "SSE";
    return "Scalar";
}

// SIMDKernelRegistry implementation
void SIMDKernelRegistry::initialize() {
    if (initialized_) {
        return;
    }
    
    capabilities_ = SIMDCapabilities::detect();
    
    // Set up function pointers based on capabilities
    if (capabilities_.sse_available) {
        sse_kernel_ = catmull_rom_sse_kernel;
    }
    
    if (capabilities_.avx2_available) {
        avx2_kernel_ = catmull_rom_avx2_kernel;
    }
    
    initialized_ = true;
}

bool SIMDKernelRegistry::isAvailable() {
    if (!initialized_) {
        initialize();
    }
    return capabilities_.sse_available || capabilities_.avx2_available;
}

const SIMDCapabilities& SIMDKernelRegistry::getCapabilities() {
    if (!initialized_) {
        initialize();
    }
    return capabilities_;
}

void SIMDKernelRegistry::interpolateBatch(const SplineSegment& segment,
                                         const float* t_values,
                                         ControlPoint* output,
                                         size_t count) {
    if (!initialized_) {
        initialize();
    }
    
    if (!segment.coefficients_valid_ || count == 0) {
        return;
    }
    
    // Choose optimal kernel based on capabilities and data size
    if (capabilities_.avx2_available && avx2_kernel_ && count >= 8) {
        // Use AVX2 for larger datasets
        size_t simd_count = (count / 8) * 8;
        avx2_kernel_(segment.coeff_x_, segment.coeff_y_, segment.coeff_z_, 
                    segment.coeff_i_, t_values, output, simd_count);
        
        // Handle remaining elements with scalar
        if (simd_count < count) {
            catmull_rom_scalar_kernel(segment.coeff_x_, segment.coeff_y_, 
                                    segment.coeff_z_, segment.coeff_i_,
                                    t_values + simd_count, output + simd_count, 
                                    count - simd_count);
        }
        last_speedup_ = 2.5f; // Estimated speedup for AVX2
    }
    else if (capabilities_.sse_available && sse_kernel_ && count >= 4) {
        // Use SSE for medium datasets
        size_t simd_count = (count / 4) * 4;
        sse_kernel_(segment.coeff_x_, segment.coeff_y_, segment.coeff_z_, 
                   segment.coeff_i_, t_values, output, simd_count);
        
        // Handle remaining elements with scalar
        if (simd_count < count) {
            catmull_rom_scalar_kernel(segment.coeff_x_, segment.coeff_y_, 
                                    segment.coeff_z_, segment.coeff_i_,
                                    t_values + simd_count, output + simd_count, 
                                    count - simd_count);
        }
        last_speedup_ = 1.8f; // Estimated speedup for SSE
    }
    else {
        // Fallback to scalar
        catmull_rom_scalar_kernel(segment.coeff_x_, segment.coeff_y_, 
                                segment.coeff_z_, segment.coeff_i_,
                                t_values, output, count);
        last_speedup_ = 1.0f;
    }
}

float SIMDKernelRegistry::getLastSpeedup() {
    return last_speedup_;
}

// AVX2 Implementation (8 floats at once)
void catmull_rom_avx2_kernel(const float* coeff_x, const float* coeff_y,
                             const float* coeff_z, const float* coeff_i,
                             const float* t_values, ControlPoint* output,
                             size_t count) {
#ifdef __AVX2__
    if (count < 8) {
        catmull_rom_scalar_kernel(coeff_x, coeff_y, coeff_z, coeff_i, t_values, output, count);
        return;
    }
    
    // Load coefficients into AVX2 registers
    __m256 coeff_x_a = _mm256_broadcast_ss(&coeff_x[0]); // t^3 coefficient
    __m256 coeff_x_b = _mm256_broadcast_ss(&coeff_x[1]); // t^2 coefficient
    __m256 coeff_x_c = _mm256_broadcast_ss(&coeff_x[2]); // t coefficient
    __m256 coeff_x_d = _mm256_broadcast_ss(&coeff_x[3]); // constant
    
    __m256 coeff_y_a = _mm256_broadcast_ss(&coeff_y[0]);
    __m256 coeff_y_b = _mm256_broadcast_ss(&coeff_y[1]);
    __m256 coeff_y_c = _mm256_broadcast_ss(&coeff_y[2]);
    __m256 coeff_y_d = _mm256_broadcast_ss(&coeff_y[3]);
    
    __m256 coeff_z_a = _mm256_broadcast_ss(&coeff_z[0]);
    __m256 coeff_z_b = _mm256_broadcast_ss(&coeff_z[1]);
    __m256 coeff_z_c = _mm256_broadcast_ss(&coeff_z[2]);
    __m256 coeff_z_d = _mm256_broadcast_ss(&coeff_z[3]);
    
    __m256 coeff_i_a = _mm256_broadcast_ss(&coeff_i[0]);
    __m256 coeff_i_b = _mm256_broadcast_ss(&coeff_i[1]);
    __m256 coeff_i_c = _mm256_broadcast_ss(&coeff_i[2]);
    __m256 coeff_i_d = _mm256_broadcast_ss(&coeff_i[3]);
    
    // Process 8 values at a time
    size_t simd_count = (count / 8) * 8;
    
    for (size_t i = 0; i < simd_count; i += 8) {
        // Load 8 t values
        __m256 t = _mm256_loadu_ps(&t_values[i]);
        
        // Calculate t^2 and t^3
        __m256 t2 = _mm256_mul_ps(t, t);
        __m256 t3 = _mm256_mul_ps(t2, t);
        
        // Calculate polynomial: at^3 + bt^2 + ct + d for each coordinate
        // X coordinate
        __m256 result_x = _mm256_fmadd_ps(coeff_x_a, t3, _mm256_fmadd_ps(coeff_x_b, t2, _mm256_fmadd_ps(coeff_x_c, t, coeff_x_d)));
        
        // Y coordinate  
        __m256 result_y = _mm256_fmadd_ps(coeff_y_a, t3, _mm256_fmadd_ps(coeff_y_b, t2, _mm256_fmadd_ps(coeff_y_c, t, coeff_y_d)));
        
        // Z coordinate
        __m256 result_z = _mm256_fmadd_ps(coeff_z_a, t3, _mm256_fmadd_ps(coeff_z_b, t2, _mm256_fmadd_ps(coeff_z_c, t, coeff_z_d)));
        
        // Intensity
        __m256 result_i = _mm256_fmadd_ps(coeff_i_a, t3, _mm256_fmadd_ps(coeff_i_b, t2, _mm256_fmadd_ps(coeff_i_c, t, coeff_i_d)));
        
        // Store results - need to interleave into ControlPoint structure
        alignas(32) float temp_x[8], temp_y[8], temp_z[8], temp_i[8];
        _mm256_store_ps(temp_x, result_x);
        _mm256_store_ps(temp_y, result_y);
        _mm256_store_ps(temp_z, result_z);
        _mm256_store_ps(temp_i, result_i);
        
        // Copy to output structure
        for (int j = 0; j < 8; ++j) {
            output[i + j].x = temp_x[j];
            output[i + j].y = temp_y[j];
            output[i + j].z = temp_z[j];
            output[i + j].intensity = temp_i[j];
            output[i + j].timestamp = 0.0f; // Will be set later if needed
        }
    }
    
    // Handle remaining elements with scalar fallback
    if (simd_count < count) {
        catmull_rom_scalar_kernel(coeff_x, coeff_y, coeff_z, coeff_i,
                                 t_values + simd_count, output + simd_count,
                                 count - simd_count);
    }
#else
    // Fallback to scalar if AVX2 not available at compile time
    catmull_rom_scalar_kernel(coeff_x, coeff_y, coeff_z, coeff_i, t_values, output, count);
#endif
}

// SSE Implementation (4 floats at once)
void catmull_rom_sse_kernel(const float* coeff_x, const float* coeff_y,
                           const float* coeff_z, const float* coeff_i,
                           const float* t_values, ControlPoint* output,
                           size_t count) {
#ifdef __SSE__
    if (count < 4) {
        catmull_rom_scalar_kernel(coeff_x, coeff_y, coeff_z, coeff_i, t_values, output, count);
        return;
    }
    
    // Load coefficients into SSE registers
    __m128 coeff_x_a = _mm_set1_ps(coeff_x[0]); // t^3 coefficient
    __m128 coeff_x_b = _mm_set1_ps(coeff_x[1]); // t^2 coefficient
    __m128 coeff_x_c = _mm_set1_ps(coeff_x[2]); // t coefficient
    __m128 coeff_x_d = _mm_set1_ps(coeff_x[3]); // constant
    
    __m128 coeff_y_a = _mm_set1_ps(coeff_y[0]);
    __m128 coeff_y_b = _mm_set1_ps(coeff_y[1]);
    __m128 coeff_y_c = _mm_set1_ps(coeff_y[2]);
    __m128 coeff_y_d = _mm_set1_ps(coeff_y[3]);
    
    __m128 coeff_z_a = _mm_set1_ps(coeff_z[0]);
    __m128 coeff_z_b = _mm_set1_ps(coeff_z[1]);
    __m128 coeff_z_c = _mm_set1_ps(coeff_z[2]);
    __m128 coeff_z_d = _mm_set1_ps(coeff_z[3]);
    
    __m128 coeff_i_a = _mm_set1_ps(coeff_i[0]);
    __m128 coeff_i_b = _mm_set1_ps(coeff_i[1]);
    __m128 coeff_i_c = _mm_set1_ps(coeff_i[2]);
    __m128 coeff_i_d = _mm_set1_ps(coeff_i[3]);
    
    // Process 4 values at a time
    size_t simd_count = (count / 4) * 4;
    
    for (size_t i = 0; i < simd_count; i += 4) {
        // Load 4 t values
        __m128 t = _mm_loadu_ps(&t_values[i]);
        
        // Calculate t^2 and t^3
        __m128 t2 = _mm_mul_ps(t, t);
        __m128 t3 = _mm_mul_ps(t2, t);
        
        // Calculate polynomial: at^3 + bt^2 + ct + d for each coordinate
        // X coordinate
        __m128 temp_x = _mm_mul_ps(coeff_x_a, t3);
        temp_x = _mm_add_ps(temp_x, _mm_mul_ps(coeff_x_b, t2));
        temp_x = _mm_add_ps(temp_x, _mm_mul_ps(coeff_x_c, t));
        __m128 result_x = _mm_add_ps(temp_x, coeff_x_d);
        
        // Y coordinate
        __m128 temp_y = _mm_mul_ps(coeff_y_a, t3);
        temp_y = _mm_add_ps(temp_y, _mm_mul_ps(coeff_y_b, t2));
        temp_y = _mm_add_ps(temp_y, _mm_mul_ps(coeff_y_c, t));
        __m128 result_y = _mm_add_ps(temp_y, coeff_y_d);
        
        // Z coordinate
        __m128 temp_z = _mm_mul_ps(coeff_z_a, t3);
        temp_z = _mm_add_ps(temp_z, _mm_mul_ps(coeff_z_b, t2));
        temp_z = _mm_add_ps(temp_z, _mm_mul_ps(coeff_z_c, t));
        __m128 result_z = _mm_add_ps(temp_z, coeff_z_d);
        
        // Intensity
        __m128 temp_i = _mm_mul_ps(coeff_i_a, t3);
        temp_i = _mm_add_ps(temp_i, _mm_mul_ps(coeff_i_b, t2));
        temp_i = _mm_add_ps(temp_i, _mm_mul_ps(coeff_i_c, t));
        __m128 result_i = _mm_add_ps(temp_i, coeff_i_d);
        
        // Store results
        alignas(16) float temp_x_arr[4], temp_y_arr[4], temp_z_arr[4], temp_i_arr[4];
        _mm_store_ps(temp_x_arr, result_x);
        _mm_store_ps(temp_y_arr, result_y);
        _mm_store_ps(temp_z_arr, result_z);
        _mm_store_ps(temp_i_arr, result_i);
        
        // Copy to output structure
        for (int j = 0; j < 4; ++j) {
            output[i + j].x = temp_x_arr[j];
            output[i + j].y = temp_y_arr[j];
            output[i + j].z = temp_z_arr[j];
            output[i + j].intensity = temp_i_arr[j];
            output[i + j].timestamp = 0.0f; // Will be set later if needed
        }
    }
    
    // Handle remaining elements with scalar fallback
    if (simd_count < count) {
        catmull_rom_scalar_kernel(coeff_x, coeff_y, coeff_z, coeff_i,
                                 t_values + simd_count, output + simd_count,
                                 count - simd_count);
    }
#else
    // Fallback to scalar if SSE not available at compile time
    catmull_rom_scalar_kernel(coeff_x, coeff_y, coeff_z, coeff_i, t_values, output, count);
#endif
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

// Utility functions
namespace simd_utils {

bool isAligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

size_t alignSize(size_t size, size_t simd_width) {
    return (size / simd_width) * simd_width;
}

size_t getOptimalSIMDWidth() {
    if (!SIMDKernelRegistry::isAvailable()) {
        return 1;
    }
    
    const auto& caps = SIMDKernelRegistry::getCapabilities();
    if (caps.avx2_available) {
        return 8;
    } else if (caps.sse_available) {
        return 4;
    }
    return 1;
}

void prefetchData(const void* ptr, size_t size) {
#ifdef __x86_64__
    const char* data = static_cast<const char*>(ptr);
    const size_t cache_line_size = 64;
    
    for (size_t i = 0; i < size; i += cache_line_size) {
        _mm_prefetch(data + i, _MM_HINT_T0);
    }
#else
    (void)ptr;
    (void)size;
#endif
}

} // namespace simd_utils

} // namespace interpolation
} // namespace prism