# PRISM Interpolation Integration Guide

## Overview

This document provides comprehensive guidance for integrating advanced interpolation algorithms into the PRISM package, based on the proven FILC implementation. The integration focuses on cubic spline interpolation, OS1-32 beam altitude handling, and performance optimizations including SIMD vectorization and memory-efficient processing.

## Analysis of Existing FILC Interpolation Logic

### Current Implementation Structure

The FILC `ImprovedInterpolationNode` provides a solid foundation with several key components:

```cpp
// Key components from improved_interpolation_node.cpp
class ImprovedInterpolationNode {
private:
    // OS1-32 specific beam altitudes (32 beams)
    std::vector<float> beam_altitudes_rad_;
    std::vector<float> interpolated_altitudes_;
    
    // Interpolation parameters
    double scale_factor_;                    // Default: 3.0 (32 → 96 beams)
    std::string interpolation_method_;       // "cubic" or "linear"
    
    // Image feature integration (optional)
    cv::Mat interpolated_range_image_;
    cv::Mat interpolated_signal_image_;
};
```

### Core Algorithm Analysis

**Catmull-Rom Cubic Spline Implementation:**
```cpp
// FILC's cubic interpolation algorithm
std::vector<float> interpolateAltitudesCubic(int target_size) {
    for (int i = 0; i < target_size; ++i) {
        float pos = i * (beam_altitudes_rad_.size() - 1) / static_cast<float>(target_size - 1);
        int idx = static_cast<int>(pos);
        float t = pos - idx;
        
        if (interpolation_method_ == "cubic" && idx > 0 && idx < beam_altitudes_rad_.size() - 2) {
            // Catmull-Rom coefficients
            float p0 = beam_altitudes_rad_[idx - 1];
            float p1 = beam_altitudes_rad_[idx];
            float p2 = beam_altitudes_rad_[idx + 1];
            float p3 = beam_altitudes_rad_[idx + 2];
            
            float t2 = t * t;
            float t3 = t2 * t;
            
            result[i] = 0.5f * ((2.0f * p1) +
                               (-p0 + p2) * t +
                               (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 +
                               (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
        }
    }
}
```

**Strengths of Current Implementation:**
- Catmull-Rom splines provide smooth C¹ continuity
- Proper handling of boundary conditions
- Discontinuity detection and handling
- Integration with range/signal images
- Performance monitoring and statistics

**Areas for Enhancement:**
- SIMD vectorization opportunities
- Memory layout optimization (Structure-of-Arrays)
- Multi-threading for column processing
- Advanced discontinuity detection algorithms

## Integration Strategy for Cubic Interpolation Algorithm

### 1. PRISM Interpolation Architecture

```cpp
// include/prism/interpolation/CubicInterpolator.hpp
#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <immintrin.h>  // SIMD intrinsics
#include <tbb/parallel_for.h>

namespace prism {
namespace interpolation {

class CubicInterpolator {
public:
    struct InterpolationConfig {
        float scale_factor = 3.0f;
        bool enable_simd = true;
        bool enable_discontinuity_detection = true;
        float discontinuity_threshold = 0.5f;  // meters
        bool preserve_original_points = true;
        int thread_count = -1;  // -1 for auto-detection
    };
    
    struct InterpolationResult {
        pcl::PointCloud<pcl::PointXYZI>::Ptr interpolated_cloud;
        std::vector<float> confidence_scores;
        std::vector<bool> discontinuity_flags;
        InterpolationMetrics metrics;
    };
    
    InterpolationResult interpolate(
        const pcl::PointCloud<pcl::PointXYZI>& input_cloud,
        const std::vector<float>& beam_altitudes,
        const InterpolationConfig& config = InterpolationConfig{});

private:
    // SIMD-optimized Catmull-Rom computation
    void catmullRomSIMD(const float* p0, const float* p1, const float* p2, const float* p3,
                        float t, float* result, size_t count);
    
    // Discontinuity detection with multiple criteria
    bool detectDiscontinuity(const pcl::PointXYZI& p1, const pcl::PointXYZI& p2,
                            const InterpolationConfig& config);
    
    // Memory-efficient column processing
    void processColumnBatch(const pcl::PointCloud<pcl::PointXYZI>& input,
                           size_t col_start, size_t col_count,
                           const InterpolationConfig& config,
                           InterpolationResult& result);
};

} // namespace interpolation
} // namespace prism
```

### 2. Enhanced Interpolation Pipeline

```cpp
// src/interpolation/CubicInterpolator.cpp
namespace prism {
namespace interpolation {

CubicInterpolator::InterpolationResult 
CubicInterpolator::interpolate(const pcl::PointCloud<pcl::PointXYZI>& input_cloud,
                              const std::vector<float>& beam_altitudes,
                              const InterpolationConfig& config) {
    
    InterpolationResult result;
    auto& metrics = result.metrics;
    metrics.start_time = std::chrono::high_resolution_clock::now();
    
    // 1. Input validation and preparation
    validateInputs(input_cloud, beam_altitudes, config);
    
    const size_t input_height = input_cloud.height;
    const size_t input_width = input_cloud.width;
    const size_t output_height = static_cast<size_t>(input_height * config.scale_factor);
    
    // 2. Memory allocation with optimal layout
    result.interpolated_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    result.interpolated_cloud->width = input_width;
    result.interpolated_cloud->height = output_height;
    result.interpolated_cloud->points.resize(output_height * input_width);
    result.interpolated_cloud->is_dense = false;
    
    result.confidence_scores.resize(output_height * input_width, 1.0f);
    result.discontinuity_flags.resize(output_height * input_width, false);
    
    // 3. Pre-compute interpolated altitudes
    std::vector<float> interpolated_altitudes = 
        computeInterpolatedAltitudes(beam_altitudes, output_height, config);
    
    // 4. Parallel column processing
    const int thread_count = (config.thread_count > 0) ? 
        config.thread_count : std::thread::hardware_concurrency();
    
    tbb::parallel_for(tbb::blocked_range<size_t>(0, input_width, 64),
        [&](const tbb::blocked_range<size_t>& range) {
            processColumnBatch(input_cloud, range.begin(), 
                             range.size(), config, result);
        });
    
    // 5. Post-processing and quality assessment
    assessInterpolationQuality(result);
    
    metrics.end_time = std::chrono::high_resolution_clock::now();
    metrics.processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
        metrics.end_time - metrics.start_time);
    
    return result;
}

} // namespace interpolation
} // namespace prism
```

## OS1-32 Beam Altitude Handling

### Beam Configuration Management

```cpp
// include/prism/interpolation/BeamAltitudeManager.hpp
namespace prism {
namespace interpolation {

class BeamAltitudeManager {
public:
    struct BeamConfiguration {
        std::vector<float> altitudes_deg;
        std::vector<float> altitudes_rad;
        std::string sensor_model;
        float vertical_resolution_deg;
        float vertical_fov_deg;
    };
    
    // OS1-32 specific configuration
    static BeamConfiguration getOS132Configuration() {
        BeamConfiguration config;
        config.sensor_model = "OS1-32";
        config.vertical_fov_deg = 45.0f;
        config.vertical_resolution_deg = 1.5f;
        
        // Exact OS1-32 beam altitudes from FILC
        config.altitudes_deg = {
            -16.611f, -16.084f, -15.557f, -15.029f, -14.502f, -13.975f,
            -13.447f, -12.920f, -12.393f, -11.865f, -11.338f, -10.811f,
            -10.283f, -9.756f, -9.229f, -8.701f, -8.174f, -7.646f,
            -7.119f, -6.592f, -6.064f, -5.537f, -5.010f, -4.482f,
            -3.955f, -3.428f, -2.900f, -2.373f, -1.846f, -1.318f,
            -0.791f, -0.264f
        };
        
        // Convert to radians
        config.altitudes_rad.resize(config.altitudes_deg.size());
        std::transform(config.altitudes_deg.begin(), config.altitudes_deg.end(),
                      config.altitudes_rad.begin(),
                      [](float deg) { return deg * M_PI / 180.0f; });
        
        return config;
    }
    
    // Generate interpolated beam altitudes
    std::vector<float> generateInterpolatedAltitudes(
        const BeamConfiguration& config,
        size_t target_beam_count,
        InterpolationMethod method = InterpolationMethod::CUBIC_CATMULL_ROM) {
        
        std::vector<float> result(target_beam_count);
        
        for (size_t i = 0; i < target_beam_count; ++i) {
            float pos = static_cast<float>(i) * (config.altitudes_rad.size() - 1) / 
                       (target_beam_count - 1);
            
            result[i] = interpolateAltitude(config.altitudes_rad, pos, method);
        }
        
        return result;
    }
    
private:
    float interpolateAltitude(const std::vector<float>& altitudes, 
                            float position,
                            InterpolationMethod method);
};

} // namespace interpolation
} // namespace prism
```

### Adaptive Beam Interpolation

```cpp
// Enhanced altitude interpolation with adaptive sampling
std::vector<float> BeamAltitudeManager::generateAdaptiveAltitudes(
    const BeamConfiguration& config,
    size_t target_beam_count,
    float curvature_threshold = 0.01f) {
    
    std::vector<float> result;
    result.reserve(target_beam_count);
    
    // Analyze curvature along the altitude curve
    std::vector<float> curvatures = computeCurvature(config.altitudes_rad);
    
    // Adaptive sampling: more points where curvature is high
    size_t current_beam = 0;
    for (size_t i = 0; i < target_beam_count; ++i) {
        float normalized_pos = static_cast<float>(i) / (target_beam_count - 1);
        
        // Find appropriate source beam based on curvature
        size_t source_idx = findAdaptiveIndex(curvatures, normalized_pos, curvature_threshold);
        
        float interpolated = interpolateAltitude(config.altitudes_rad, source_idx, 
                                               InterpolationMethod::CUBIC_CATMULL_ROM);
        result.push_back(interpolated);
    }
    
    return result;
}
```

## Performance Optimizations

### 1. SIMD Vectorization

```cpp
// SIMD-optimized Catmull-Rom computation using AVX2
void CubicInterpolator::catmullRomSIMD(const float* p0, const float* p1, 
                                      const float* p2, const float* p3,
                                      float t, float* result, size_t count) {
    
    const __m256 coeff_2 = _mm256_set1_ps(2.0f);
    const __m256 coeff_half = _mm256_set1_ps(0.5f);
    const __m256 coeff_5 = _mm256_set1_ps(5.0f);
    const __m256 coeff_4 = _mm256_set1_ps(4.0f);
    const __m256 coeff_3 = _mm256_set1_ps(3.0f);
    
    const __m256 t_vec = _mm256_set1_ps(t);
    const __m256 t2_vec = _mm256_mul_ps(t_vec, t_vec);
    const __m256 t3_vec = _mm256_mul_ps(t2_vec, t_vec);
    
    // Process 8 floats at once
    for (size_t i = 0; i < count; i += 8) {
        __m256 p0_vec = _mm256_load_ps(&p0[i]);
        __m256 p1_vec = _mm256_load_ps(&p1[i]);
        __m256 p2_vec = _mm256_load_ps(&p2[i]);
        __m256 p3_vec = _mm256_load_ps(&p3[i]);
        
        // Catmull-Rom formula vectorized
        __m256 term1 = _mm256_mul_ps(coeff_2, p1_vec);
        
        __m256 term2 = _mm256_sub_ps(p2_vec, p0_vec);
        term2 = _mm256_mul_ps(term2, t_vec);
        
        __m256 term3_a = _mm256_mul_ps(coeff_2, p0_vec);
        __m256 term3_b = _mm256_mul_ps(coeff_5, p1_vec);
        __m256 term3_c = _mm256_mul_ps(coeff_4, p2_vec);
        __m256 term3 = _mm256_add_ps(term3_a, _mm256_sub_ps(term3_c, _mm256_add_ps(term3_b, p3_vec)));
        term3 = _mm256_mul_ps(term3, t2_vec);
        
        __m256 term4_a = _mm256_sub_ps(p0_vec, _mm256_mul_ps(coeff_3, p1_vec));
        __m256 term4_b = _mm256_sub_ps(_mm256_mul_ps(coeff_3, p2_vec), p3_vec);
        __m256 term4 = _mm256_add_ps(term4_a, term4_b);
        term4 = _mm256_mul_ps(term4, t3_vec);
        
        __m256 result_vec = _mm256_add_ps(term1, _mm256_add_ps(term2, _mm256_add_ps(term3, term4)));
        result_vec = _mm256_mul_ps(result_vec, coeff_half);
        
        _mm256_store_ps(&result[i], result_vec);
    }
}
```

### 2. Memory-Efficient Processing Pipeline

```cpp
// Structure-of-Arrays layout for better cache performance
struct PointCloudSoA {
    std::vector<float> x, y, z, intensity;
    size_t size() const { return x.size(); }
    
    void resize(size_t new_size) {
        x.resize(new_size);
        y.resize(new_size);
        z.resize(new_size);
        intensity.resize(new_size);
    }
    
    // Convert from PCL format
    static PointCloudSoA fromPCL(const pcl::PointCloud<pcl::PointXYZI>& cloud) {
        PointCloudSoA soa;
        soa.resize(cloud.size());
        
        #pragma omp parallel for
        for (size_t i = 0; i < cloud.size(); ++i) {
            const auto& pt = cloud.points[i];
            soa.x[i] = pt.x;
            soa.y[i] = pt.y;
            soa.z[i] = pt.z;
            soa.intensity[i] = pt.intensity;
        }
        
        return soa;
    }
    
    // Convert to PCL format
    pcl::PointCloud<pcl::PointXYZI>::Ptr toPCL() const {
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        cloud->points.resize(size());
        
        #pragma omp parallel for
        for (size_t i = 0; i < size(); ++i) {
            auto& pt = cloud->points[i];
            pt.x = x[i];
            pt.y = y[i];
            pt.z = z[i];
            pt.intensity = intensity[i];
        }
        
        return cloud;
    }
};
```

### 3. Advanced Discontinuity Detection

```cpp
// Multi-criteria discontinuity detection
bool CubicInterpolator::detectDiscontinuity(const pcl::PointXYZI& p1, 
                                           const pcl::PointXYZI& p2,
                                           const InterpolationConfig& config) {
    
    // 1. Range-based discontinuity (primary criterion from FILC)
    const float r1 = std::sqrt(p1.x*p1.x + p1.y*p1.y + p1.z*p1.z);
    const float r2 = std::sqrt(p2.x*p2.x + p2.y*p2.y + p2.z*p2.z);
    const bool range_discontinuous = std::abs(r2 - r1) > config.discontinuity_threshold;
    
    // 2. Intensity-based discontinuity
    const bool intensity_discontinuous = std::abs(p2.intensity - p1.intensity) > 100.0f;
    
    // 3. Angular discontinuity
    const float dot_product = (p1.x*p2.x + p1.y*p2.y + p1.z*p2.z) / (r1 * r2);
    const float angle_diff = std::acos(std::clamp(dot_product, -1.0f, 1.0f));
    const bool angular_discontinuous = angle_diff > (5.0f * M_PI / 180.0f); // 5 degrees
    
    // 4. Invalid point detection
    const bool has_invalid = !std::isfinite(p1.x) || !std::isfinite(p1.y) || !std::isfinite(p1.z) ||
                           !std::isfinite(p2.x) || !std::isfinite(p2.y) || !std::isfinite(p2.z);
    
    return range_discontinuous || intensity_discontinuous || angular_discontinuous || has_invalid;
}
```

## Memory-Efficient Interpolation Pipeline

### 1. Memory Pool Management

```cpp
// include/prism/interpolation/MemoryPool.hpp
namespace prism {
namespace interpolation {

class InterpolationMemoryPool {
public:
    InterpolationMemoryPool(size_t pool_size_bytes = 16 * 1024 * 1024) // 16MB default
        : pool_size_(pool_size_bytes) {
        pool_memory_ = std::aligned_alloc(32, pool_size_); // 32-byte aligned for SIMD
        reset();
    }
    
    ~InterpolationMemoryPool() {
        std::free(pool_memory_);
    }
    
    template<typename T>
    T* allocate(size_t count) {
        const size_t bytes_needed = count * sizeof(T);
        const size_t aligned_bytes = (bytes_needed + 31) & ~31; // 32-byte alignment
        
        if (current_offset_ + aligned_bytes > pool_size_) {
            throw std::bad_alloc();
        }
        
        T* result = reinterpret_cast<T*>(
            static_cast<char*>(pool_memory_) + current_offset_);
        current_offset_ += aligned_bytes;
        
        return result;
    }
    
    void reset() {
        current_offset_ = 0;
    }
    
private:
    void* pool_memory_;
    size_t pool_size_;
    size_t current_offset_;
};

} // namespace interpolation
} // namespace prism
```

### 2. Streaming Processing Architecture

```cpp
// Process large point clouds in chunks to manage memory usage
class StreamingInterpolator {
public:
    struct ChunkResult {
        std::unique_ptr<PointCloudSoA> interpolated_chunk;
        std::vector<float> confidence_scores;
        size_t original_start_idx;
        size_t original_count;
    };
    
    std::vector<ChunkResult> processInChunks(
        const pcl::PointCloud<pcl::PointXYZI>& input,
        const std::vector<float>& beam_altitudes,
        const InterpolationConfig& config,
        size_t chunk_size = 10000) {
        
        std::vector<ChunkResult> results;
        
        for (size_t start = 0; start < input.size(); start += chunk_size) {
            size_t end = std::min(start + chunk_size, input.size());
            
            // Extract chunk
            pcl::PointCloud<pcl::PointXYZI> chunk;
            chunk.points.assign(input.points.begin() + start,
                               input.points.begin() + end);
            chunk.width = end - start;
            chunk.height = 1;
            
            // Process chunk
            auto interpolation_result = interpolator_.interpolate(chunk, beam_altitudes, config);
            
            ChunkResult chunk_result;
            chunk_result.interpolated_chunk = std::make_unique<PointCloudSoA>(
                PointCloudSoA::fromPCL(*interpolation_result.interpolated_cloud));
            chunk_result.confidence_scores = std::move(interpolation_result.confidence_scores);
            chunk_result.original_start_idx = start;
            chunk_result.original_count = end - start;
            
            results.push_back(std::move(chunk_result));
            
            // Reset memory pool for next chunk
            memory_pool_.reset();
        }
        
        return results;
    }
    
private:
    CubicInterpolator interpolator_;
    InterpolationMemoryPool memory_pool_;
};
```

## Benchmarking Methodology

### 1. Performance Benchmarking Framework

```cpp
// test/benchmark/interpolation_benchmark.cpp
#include <benchmark/benchmark.h>
#include "prism/interpolation/CubicInterpolator.hpp"

class InterpolationBenchmark {
public:
    static pcl::PointCloud<pcl::PointXYZI>::Ptr generateOS132Cloud() {
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        cloud->width = 1024;
        cloud->height = 32;
        cloud->points.resize(cloud->width * cloud->height);
        
        // Generate realistic OS1-32 data with some noise
        for (size_t row = 0; row < cloud->height; ++row) {
            float altitude = -16.611f + row * (16.347f / 31.0f); // degrees
            float altitude_rad = altitude * M_PI / 180.0f;
            
            for (size_t col = 0; col < cloud->width; ++col) {
                float azimuth = (col * 360.0f / cloud->width) * M_PI / 180.0f;
                float range = 10.0f + (rand() % 50); // 10-60m range
                
                size_t idx = row * cloud->width + col;
                auto& pt = cloud->points[idx];
                
                pt.x = range * std::cos(altitude_rad) * std::cos(azimuth);
                pt.y = range * std::cos(altitude_rad) * std::sin(azimuth);
                pt.z = range * std::sin(altitude_rad);
                pt.intensity = 50.0f + (rand() % 100);
            }
        }
        
        return cloud;
    }
};

// Benchmark different interpolation configurations
static void BM_CubicInterpolation_2x(benchmark::State& state) {
    auto cloud = InterpolationBenchmark::generateOS132Cloud();
    auto altitudes = prism::interpolation::BeamAltitudeManager::getOS132Configuration().altitudes_rad;
    
    prism::interpolation::CubicInterpolator interpolator;
    prism::interpolation::CubicInterpolator::InterpolationConfig config;
    config.scale_factor = 2.0f;
    config.enable_simd = true;
    
    for (auto _ : state) {
        auto result = interpolator.interpolate(*cloud, altitudes, config);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations() * cloud->size());
    state.SetBytesProcessed(state.iterations() * cloud->size() * sizeof(pcl::PointXYZI));
}
BENCHMARK(BM_CubicInterpolation_2x);

static void BM_CubicInterpolation_3x(benchmark::State& state) {
    auto cloud = InterpolationBenchmark::generateOS132Cloud();
    auto altitudes = prism::interpolation::BeamAltitudeManager::getOS132Configuration().altitudes_rad;
    
    prism::interpolation::CubicInterpolator interpolator;
    prism::interpolation::CubicInterpolator::InterpolationConfig config;
    config.scale_factor = 3.0f;
    config.enable_simd = true;
    
    for (auto _ : state) {
        auto result = interpolator.interpolate(*cloud, altitudes, config);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations() * cloud->size());
}
BENCHMARK(BM_CubicInterpolation_3x);

// Compare SIMD vs non-SIMD performance
static void BM_CubicInterpolation_SIMD_vs_Scalar(benchmark::State& state) {
    auto cloud = InterpolationBenchmark::generateOS132Cloud();
    auto altitudes = prism::interpolation::BeamAltitudeManager::getOS132Configuration().altitudes_rad;
    
    prism::interpolation::CubicInterpolator interpolator;
    prism::interpolation::CubicInterpolator::InterpolationConfig config;
    config.scale_factor = 3.0f;
    config.enable_simd = state.range(0) == 1; // 1 = SIMD, 0 = Scalar
    
    for (auto _ : state) {
        auto result = interpolator.interpolate(*cloud, altitudes, config);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetItemsProcessed(state.iterations() * cloud->size());
}
BENCHMARK(BM_CubicInterpolation_SIMD_vs_Scalar)->Arg(0)->Arg(1);

BENCHMARK_MAIN();
```

### 2. Quality Assessment Metrics

```cpp
// Quality assessment for interpolation results
struct InterpolationQualityMetrics {
    float smoothness_score;           // Measure of output smoothness
    float edge_preservation_score;    // How well discontinuities are preserved
    float intensity_consistency;      // Consistency of interpolated intensity values
    float geometric_accuracy;         // Accuracy vs ground truth (if available)
    
    float overall_quality() const {
        return 0.3f * smoothness_score + 
               0.3f * edge_preservation_score +
               0.2f * intensity_consistency +
               0.2f * geometric_accuracy;
    }
};

InterpolationQualityMetrics assessQuality(
    const pcl::PointCloud<pcl::PointXYZI>& original,
    const pcl::PointCloud<pcl::PointXYZI>& interpolated,
    const std::vector<bool>& discontinuity_flags) {
    
    InterpolationQualityMetrics metrics;
    
    // 1. Smoothness assessment
    metrics.smoothness_score = computeSmoothness(interpolated);
    
    // 2. Edge preservation assessment
    metrics.edge_preservation_score = assessEdgePreservation(
        original, interpolated, discontinuity_flags);
    
    // 3. Intensity consistency
    metrics.intensity_consistency = assessIntensityConsistency(interpolated);
    
    // 4. Geometric accuracy (if ground truth available)
    metrics.geometric_accuracy = 1.0f; // Placeholder
    
    return metrics;
}
```

## Integration with ROS2 Node

### Node Implementation

```cpp
// include/prism/nodes/InterpolationNode.hpp
namespace prism {
namespace nodes {

class InterpolationNode : public rclcpp::Node {
public:
    InterpolationNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    
private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void publishInterpolatedCloud(const interpolation::CubicInterpolator::InterpolationResult& result,
                                 const std_msgs::msg::Header& header);
    
    // ROS2 interfaces
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;
    rclcpp::TimerBase::SharedPtr stats_timer_;
    
    // Interpolation components
    std::unique_ptr<interpolation::CubicInterpolator> interpolator_;
    interpolation::CubicInterpolator::InterpolationConfig config_;
    std::vector<float> beam_altitudes_;
    
    // Performance monitoring
    rclcpp::Clock::SharedPtr clock_;
    size_t frame_count_;
    std::chrono::microseconds total_processing_time_;
};

} // namespace nodes
} // namespace prism
```

This integration strategy provides a comprehensive framework for implementing advanced cubic interpolation in PRISM, building upon the proven FILC approach while adding significant performance optimizations and enhanced capabilities.