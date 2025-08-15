# PRISM Performance Optimization Guide

## Overview

This document provides comprehensive guidance for optimizing the performance of the PRISM system across all processing stages: interpolation, projection, color mapping, and sensor fusion. The optimization strategies focus on multi-threading, SIMD vectorization, memory management, and real-time constraints to achieve high-throughput LiDAR-camera fusion.

## Multi-Threading Architecture with TBB

### 1. Thread Pool Architecture

```cpp
// include/prism/performance/ThreadManager.hpp
#pragma once

#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_pipeline.h>
#include <tbb/concurrent_queue.h>
#include <tbb/task_arena.h>
#include <thread>
#include <atomic>

namespace prism {
namespace performance {

class ThreadManager {
public:
    struct ThreadConfig {
        int interpolation_threads = -1;    // -1 for auto-detect
        int projection_threads = -1;
        int fusion_threads = -1;
        bool enable_numa_awareness = true;
        bool enable_thread_affinity = false;
        std::vector<int> cpu_affinity_list;
    };
    
    static ThreadManager& getInstance() {
        static ThreadManager instance;
        return instance;
    }
    
    void initialize(const ThreadConfig& config) {
        config_ = config;
        
        // Set global thread limit
        const int max_threads = std::thread::hardware_concurrency();
        const int total_requested = calculateTotalThreads(config);
        const int actual_threads = (total_requested > 0) ? 
            std::min(total_requested, max_threads) : max_threads;
        
        global_control_ = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, actual_threads);
        
        // Initialize specialized task arenas
        interpolation_arena_ = std::make_unique<tbb::task_arena>(
            getInterpolationThreadCount());
        projection_arena_ = std::make_unique<tbb::task_arena>(
            getProjectionThreadCount());
        fusion_arena_ = std::make_unique<tbb::task_arena>(
            getFusionThreadCount());
        
        // Set thread affinity if requested
        if (config_.enable_thread_affinity && !config_.cpu_affinity_list.empty()) {
            setupThreadAffinity();
        }
    }
    
    // Execute interpolation workload in dedicated arena
    template<typename Func>
    auto executeInterpolation(Func&& func) {
        return interpolation_arena_->execute([&func]() {
            return func();
        });
    }
    
    // Execute projection workload in dedicated arena
    template<typename Func>  
    auto executeProjection(Func&& func) {
        return projection_arena_->execute([&func]() {
            return func();
        });
    }
    
    // Execute fusion workload in dedicated arena
    template<typename Func>
    auto executeFusion(Func&& func) {
        return fusion_arena_->execute([&func]() {
            return func();
        });
    }
    
    int getInterpolationThreadCount() const {
        return (config_.interpolation_threads > 0) ? 
            config_.interpolation_threads : std::thread::hardware_concurrency() / 3;
    }
    
    int getProjectionThreadCount() const {
        return (config_.projection_threads > 0) ?
            config_.projection_threads : std::thread::hardware_concurrency() / 3;
    }
    
    int getFusionThreadCount() const {
        return (config_.fusion_threads > 0) ?
            config_.fusion_threads : std::thread::hardware_concurrency() / 3;
    }
    
private:
    ThreadConfig config_;
    std::unique_ptr<tbb::global_control> global_control_;
    std::unique_ptr<tbb::task_arena> interpolation_arena_;
    std::unique_ptr<tbb::task_arena> projection_arena_;
    std::unique_ptr<tbb::task_arena> fusion_arena_;
    
    int calculateTotalThreads(const ThreadConfig& config) const;
    void setupThreadAffinity();
};

} // namespace performance
} // namespace prism
```

### 2. Parallel Processing Patterns

```cpp
// include/prism/performance/ParallelPatterns.hpp
namespace prism {
namespace performance {

// Parallel column processing for interpolation
template<typename InputCloud, typename OutputCloud, typename ProcessFunc>
void processColumnsParallel(const InputCloud& input_cloud,
                           OutputCloud& output_cloud,
                           ProcessFunc process_column,
                           size_t grain_size = 64) {
    
    const size_t num_columns = input_cloud.width;
    
    // Use TBB parallel_for with blocked_range for load balancing
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, num_columns, grain_size),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t col = range.begin(); col < range.end(); ++col) {
                process_column(input_cloud, output_cloud, col);
            }
        }
    );
}

// Parallel point batch processing with dynamic load balancing
template<typename PointContainer, typename ProcessFunc>
void processPointBatchesParallel(const PointContainer& points,
                                ProcessFunc process_batch,
                                size_t min_batch_size = 1000) {
    
    const size_t total_points = points.size();
    const size_t num_threads = ThreadManager::getInstance().getProjectionThreadCount();
    const size_t batch_size = std::max(min_batch_size, total_points / (num_threads * 4));
    
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, total_points, batch_size),
        [&](const tbb::blocked_range<size_t>& range) {
            typename PointContainer::const_iterator begin = points.begin() + range.begin();
            typename PointContainer::const_iterator end = points.begin() + range.end();
            process_batch(begin, end, range.begin());
        }
    );
}

// Pipeline processing for streaming data
template<typename InputStage, typename IntermediateStage, typename OutputStage>
class ParallelPipeline {
public:
    struct PipelineConfig {
        size_t input_buffer_size = 4;
        size_t intermediate_buffer_size = 4;  
        size_t output_buffer_size = 4;
        bool enable_out_of_order = false;
    };
    
    ParallelPipeline(InputStage input_stage,
                    IntermediateStage intermediate_stage, 
                    OutputStage output_stage,
                    const PipelineConfig& config = PipelineConfig{})
        : input_stage_(std::move(input_stage))
        , intermediate_stage_(std::move(intermediate_stage))
        , output_stage_(std::move(output_stage))
        , config_(config) {}
    
    template<typename InputIterator, typename OutputIterator>
    void process(InputIterator input_begin, InputIterator input_end,
                OutputIterator output_begin) {
        
        using InputType = typename std::iterator_traits<InputIterator>::value_type;
        using IntermediateType = decltype(intermediate_stage_(*input_begin));
        using OutputType = decltype(output_stage_(std::declval<IntermediateType>()));
        
        tbb::parallel_pipeline(
            config_.input_buffer_size + config_.intermediate_buffer_size + config_.output_buffer_size,
            
            // Stage 1: Input processing
            tbb::make_filter<void, InputType>(
                tbb::filter::serial_in_order,
                [&](tbb::flow_control& fc) -> InputType {
                    if (input_begin == input_end) {
                        fc.stop();
                        return InputType{};
                    }
                    return *input_begin++;
                }
            ) &
            
            // Stage 2: Intermediate processing
            tbb::make_filter<InputType, IntermediateType>(
                config_.enable_out_of_order ? tbb::filter::parallel : tbb::filter::serial_in_order,
                [this](InputType input) -> IntermediateType {
                    return intermediate_stage_(input);
                }
            ) &
            
            // Stage 3: Output processing
            tbb::make_filter<IntermediateType, void>(
                tbb::filter::serial_in_order,
                [&](IntermediateType intermediate) {
                    *output_begin++ = output_stage_(intermediate);
                }
            )
        );
    }
    
private:
    InputStage input_stage_;
    IntermediateStage intermediate_stage_;
    OutputStage output_stage_;
    PipelineConfig config_;
};

} // namespace performance
} // namespace prism
```

## Memory Pool Management and Zero-Copy Strategies

### 1. Advanced Memory Pool System

```cpp
// include/prism/performance/MemoryPool.hpp
#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstdlib>
#include <algorithm>

namespace prism {
namespace performance {

template<typename T>
class MemoryPool {
public:
    struct PoolConfig {
        size_t initial_pool_size = 1024;      // Number of objects
        size_t max_pool_size = 16384;
        size_t growth_factor = 2;
        bool enable_thread_local = true;
        size_t alignment = alignof(T);
    };
    
    explicit MemoryPool(const PoolConfig& config = PoolConfig{})
        : config_(config)
        , pool_size_(config.initial_pool_size)
        , allocated_count_(0) {
        
        // Allocate aligned memory
        allocatePool(config_.initial_pool_size);
    }
    
    ~MemoryPool() {
        if (memory_pool_) {
            std::free(memory_pool_);
        }
    }
    
    // Thread-safe allocation (mutex-based for v1.0)
    T* allocate() {
        std::lock_guard<std::mutex> lock(mutex_);  // Mutex-protected, not lock-free
        
        if (!free_objects_.empty()) {
            T* obj = free_objects_.back();
            free_objects_.pop_back();
            allocated_count_++;
            return obj;
        }
        
        // Expand pool if needed
        if (allocated_count_ >= pool_size_ && pool_size_ < config_.max_pool_size) {
            expandPool();
        }
        
        // Allocate from pool
        if (next_free_index_ < pool_size_) {
            T* obj = reinterpret_cast<T*>(
                static_cast<char*>(memory_pool_) + next_free_index_ * sizeof(T));
            next_free_index_++;
            allocated_count_++;
            return obj;
        }
        
        // Pool exhausted, fall back to system allocation
        return static_cast<T*>(std::aligned_alloc(config_.alignment, sizeof(T)));
    }
    
    // Thread-safe deallocation
    void deallocate(T* obj) {
        if (!obj) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check if object belongs to our pool
        const char* pool_start = static_cast<char*>(memory_pool_);
        const char* pool_end = pool_start + pool_size_ * sizeof(T);
        const char* obj_ptr = reinterpret_cast<char*>(obj);
        
        if (obj_ptr >= pool_start && obj_ptr < pool_end) {
            // Object from pool
            free_objects_.push_back(obj);
            allocated_count_--;
        } else {
            // System-allocated object
            std::free(obj);
        }
    }
    
    // Bulk allocation for better performance
    std::vector<T*> allocateBatch(size_t count) {
        std::vector<T*> result;
        result.reserve(count);
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Try to satisfy from free objects first
        const size_t available_free = free_objects_.size();
        const size_t from_free = std::min(count, available_free);
        
        for (size_t i = 0; i < from_free; ++i) {
            result.push_back(free_objects_.back());
            free_objects_.pop_back();
        }
        
        allocated_count_ += from_free;
        count -= from_free;
        
        // Allocate remaining from pool
        while (count > 0 && next_free_index_ < pool_size_) {
            T* obj = reinterpret_cast<T*>(
                static_cast<char*>(memory_pool_) + next_free_index_ * sizeof(T));
            result.push_back(obj);
            next_free_index_++;
            allocated_count_++;
            count--;
        }
        
        // Expand pool if needed for remaining objects
        if (count > 0 && pool_size_ < config_.max_pool_size) {
            expandPool();
            
            // Try again from expanded pool
            while (count > 0 && next_free_index_ < pool_size_) {
                T* obj = reinterpret_cast<T*>(
                    static_cast<char*>(memory_pool_) + next_free_index_ * sizeof(T));
                result.push_back(obj);
                next_free_index_++;
                allocated_count_++;
                count--;
            }
        }
        
        // Fall back to system allocation for any remaining
        for (size_t i = 0; i < count; ++i) {
            result.push_back(static_cast<T*>(
                std::aligned_alloc(config_.alignment, sizeof(T))));
        }
        
        return result;
    }
    
    // Statistics
    size_t getAllocatedCount() const { 
        std::lock_guard<std::mutex> lock(mutex_);
        return allocated_count_; 
    }
    
    size_t getPoolSize() const { 
        std::lock_guard<std::mutex> lock(mutex_);
        return pool_size_; 
    }
    
    size_t getFreeCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_objects_.size();
    }
    
private:
    PoolConfig config_;
    void* memory_pool_ = nullptr;
    size_t pool_size_;
    size_t next_free_index_ = 0;
    std::atomic<size_t> allocated_count_;
    std::vector<T*> free_objects_;
    mutable std::mutex mutex_;
    
    void allocatePool(size_t size) {
        const size_t total_size = size * sizeof(T);
        memory_pool_ = std::aligned_alloc(config_.alignment, total_size);
        if (!memory_pool_) {
            throw std::bad_alloc();
        }
        pool_size_ = size;
        next_free_index_ = 0;
    }
    
    void expandPool() {
        const size_t new_size = std::min(
            pool_size_ * config_.growth_factor,
            config_.max_pool_size);
        
        if (new_size <= pool_size_) return;
        
        // Allocate new larger pool
        void* new_pool = std::aligned_alloc(config_.alignment, new_size * sizeof(T));
        if (!new_pool) return; // Keep using current pool
        
        // Copy existing data
        std::memcpy(new_pool, memory_pool_, pool_size_ * sizeof(T));
        
        // Update pointers in free_objects_
        const ptrdiff_t offset = static_cast<char*>(new_pool) - static_cast<char*>(memory_pool_);
        for (T*& ptr : free_objects_) {
            ptr = reinterpret_cast<T*>(reinterpret_cast<char*>(ptr) + offset);
        }
        
        // Switch to new pool
        std::free(memory_pool_);
        memory_pool_ = new_pool;
        pool_size_ = new_size;
    }
};

// Specialized memory pools for different data types
class PRISMMemoryManager {
public:
    static PRISMMemoryManager& getInstance() {
        static PRISMMemoryManager instance;
        return instance;
    }
    
    // Point cloud memory pools
    MemoryPool<pcl::PointXYZI>& getPointPool() { return point_pool_; }
    MemoryPool<Eigen::Vector3f>& getVector3Pool() { return vector3_pool_; }
    MemoryPool<cv::Point2f>& getPoint2Pool() { return point2_pool_; }
    MemoryPool<cv::Vec3b>& getColorPool() { return color_pool_; }
    
    // Bulk allocators
    std::vector<pcl::PointXYZI*> allocatePoints(size_t count) {
        return point_pool_.allocateBatch(count);
    }
    
    void deallocatePoints(const std::vector<pcl::PointXYZI*>& points) {
        for (auto* pt : points) {
            point_pool_.deallocate(pt);
        }
    }
    
private:
    MemoryPool<pcl::PointXYZI> point_pool_;
    MemoryPool<Eigen::Vector3f> vector3_pool_;
    MemoryPool<cv::Point2f> point2_pool_;
    MemoryPool<cv::Vec3b> color_pool_;
};

} // namespace performance
} // namespace prism
```

### 2. Zero-Copy Data Structures

```cpp
// include/prism/performance/ZeroCopyStructures.hpp
namespace prism {
namespace performance {

// Zero-copy point cloud wrapper
template<typename PointType>
class ZeroCopyPointCloud {
public:
    ZeroCopyPointCloud(PointType* data, size_t size, size_t capacity)
        : data_(data), size_(size), capacity_(capacity), owns_data_(false) {}
    
    // Move constructor for taking ownership
    ZeroCopyPointCloud(std::vector<PointType>&& points)
        : owned_data_(std::move(points)), size_(owned_data_.size())
        , capacity_(owned_data_.capacity()), owns_data_(true) {
        data_ = owned_data_.data();
    }
    
    // Accessors
    PointType* data() { return data_; }
    const PointType* data() const { return data_; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    
    PointType& operator[](size_t index) { return data_[index]; }
    const PointType& operator[](size_t index) const { return data_[index]; }
    
    // Iterator support
    PointType* begin() { return data_; }
    PointType* end() { return data_ + size_; }
    const PointType* begin() const { return data_; }
    const PointType* end() const { return data_ + size_; }
    
    // Resize (only if we own the data)
    void resize(size_t new_size) {
        if (owns_data_) {
            owned_data_.resize(new_size);
            data_ = owned_data_.data();
            size_ = new_size;
            capacity_ = owned_data_.capacity();
        } else if (new_size <= capacity_) {
            size_ = new_size;
        } else {
            throw std::runtime_error("Cannot resize non-owned data beyond capacity");
        }
    }
    
    // Create view of subset without copying
    ZeroCopyPointCloud<PointType> view(size_t offset, size_t count) {
        if (offset + count > size_) {
            throw std::out_of_range("View range exceeds data size");
        }
        return ZeroCopyPointCloud<PointType>(data_ + offset, count, capacity_ - offset);
    }
    
private:
    PointType* data_;
    std::vector<PointType> owned_data_;  // Only used if owns_data_ = true
    size_t size_;
    size_t capacity_;
    bool owns_data_;
};

// Memory-mapped file for large datasets
class MemoryMappedPointCloud {
public:
    MemoryMappedPointCloud(const std::string& filename, bool read_only = true)
        : filename_(filename), read_only_(read_only) {
        openFile();
        mapMemory();
    }
    
    ~MemoryMappedPointCloud() {
        if (mapped_data_) {
            munmap(mapped_data_, file_size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }
    
    template<typename PointType>
    ZeroCopyPointCloud<PointType> getPointCloud() {
        const size_t point_count = file_size_ / sizeof(PointType);
        return ZeroCopyPointCloud<PointType>(
            static_cast<PointType*>(mapped_data_), point_count, point_count);
    }
    
    void* data() { return mapped_data_; }
    size_t size() const { return file_size_; }
    
private:
    std::string filename_;
    bool read_only_;
    int fd_ = -1;
    void* mapped_data_ = nullptr;
    size_t file_size_ = 0;
    
    void openFile();
    void mapMemory();
};

} // namespace performance
} // namespace prism
```

## SIMD Vectorization Opportunities

### 1. SIMD Interpolation Kernels

```cpp
// include/prism/performance/SIMDKernels.hpp
#pragma once

#include <immintrin.h>
#include <cstring>

namespace prism {
namespace performance {
namespace simd {

// AVX2-optimized Catmull-Rom spline evaluation for 8 points simultaneously
inline void catmullRomAVX2(const float* p0, const float* p1, const float* p2, const float* p3,
                           float t, float* result) {
    const __m256 t_vec = _mm256_set1_ps(t);
    const __m256 t2_vec = _mm256_mul_ps(t_vec, t_vec);
    const __m256 t3_vec = _mm256_mul_ps(t2_vec, t_vec);
    
    // Load control points
    const __m256 p0_vec = _mm256_load_ps(p0);
    const __m256 p1_vec = _mm256_load_ps(p1);
    const __m256 p2_vec = _mm256_load_ps(p2);
    const __m256 p3_vec = _mm256_load_ps(p3);
    
    // Constants
    const __m256 c0_5 = _mm256_set1_ps(0.5f);
    const __m256 c2 = _mm256_set1_ps(2.0f);
    const __m256 c3 = _mm256_set1_ps(3.0f);
    const __m256 c4 = _mm256_set1_ps(4.0f);
    const __m256 c5 = _mm256_set1_ps(5.0f);
    
    // Catmull-Rom formula: 0.5 * (2*P1 + (-P0+P2)*t + (2*P0-5*P1+4*P2-P3)*t^2 + (-P0+3*P1-3*P2+P3)*t^3)
    
    // Term 1: 2 * P1
    const __m256 term1 = _mm256_mul_ps(c2, p1_vec);
    
    // Term 2: (-P0 + P2) * t
    const __m256 term2 = _mm256_mul_ps(_mm256_sub_ps(p2_vec, p0_vec), t_vec);
    
    // Term 3: (2*P0 - 5*P1 + 4*P2 - P3) * t^2
    const __m256 term3_coeff = _mm256_add_ps(
        _mm256_sub_ps(_mm256_mul_ps(c2, p0_vec), _mm256_mul_ps(c5, p1_vec)),
        _mm256_sub_ps(_mm256_mul_ps(c4, p2_vec), p3_vec)
    );
    const __m256 term3 = _mm256_mul_ps(term3_coeff, t2_vec);
    
    // Term 4: (-P0 + 3*P1 - 3*P2 + P3) * t^3
    const __m256 term4_coeff = _mm256_add_ps(
        _mm256_sub_ps(_mm256_mul_ps(c3, p1_vec), p0_vec),
        _mm256_sub_ps(p3_vec, _mm256_mul_ps(c3, p2_vec))
    );
    const __m256 term4 = _mm256_mul_ps(term4_coeff, t3_vec);
    
    // Combine terms
    const __m256 sum = _mm256_add_ps(
        _mm256_add_ps(term1, term2),
        _mm256_add_ps(term3, term4)
    );
    
    // Multiply by 0.5
    const __m256 result_vec = _mm256_mul_ps(sum, c0_5);
    
    // Store result
    _mm256_store_ps(result, result_vec);
}

// SIMD-optimized 3D point transformation (4 points at once)
inline void transformPoints4xAVX(const float* points_x, const float* points_y, const float* points_z,
                                const float* transform_matrix, // 4x4 matrix in row-major order
                                float* result_x, float* result_y, float* result_z) {
    
    // Load input points
    const __m128 px = _mm_load_ps(points_x);
    const __m128 py = _mm_load_ps(points_y);
    const __m128 pz = _mm_load_ps(points_z);
    const __m128 pw = _mm_set1_ps(1.0f); // Homogeneous coordinate
    
    // Load transformation matrix rows
    const __m128 m0 = _mm_load_ps(&transform_matrix[0]);  // Row 0: m00 m01 m02 m03
    const __m128 m1 = _mm_load_ps(&transform_matrix[4]);  // Row 1: m10 m11 m12 m13
    const __m128 m2 = _mm_load_ps(&transform_matrix[8]);  // Row 2: m20 m21 m22 m23
    
    // Extract matrix elements
    const __m128 m00 = _mm_shuffle_ps(m0, m0, _MM_SHUFFLE(0,0,0,0));
    const __m128 m01 = _mm_shuffle_ps(m0, m0, _MM_SHUFFLE(1,1,1,1));
    const __m128 m02 = _mm_shuffle_ps(m0, m0, _MM_SHUFFLE(2,2,2,2));
    const __m128 m03 = _mm_shuffle_ps(m0, m0, _MM_SHUFFLE(3,3,3,3));
    
    const __m128 m10 = _mm_shuffle_ps(m1, m1, _MM_SHUFFLE(0,0,0,0));
    const __m128 m11 = _mm_shuffle_ps(m1, m1, _MM_SHUFFLE(1,1,1,1));
    const __m128 m12 = _mm_shuffle_ps(m1, m1, _MM_SHUFFLE(2,2,2,2));
    const __m128 m13 = _mm_shuffle_ps(m1, m1, _MM_SHUFFLE(3,3,3,3));
    
    const __m128 m20 = _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(0,0,0,0));
    const __m128 m21 = _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(1,1,1,1));
    const __m128 m22 = _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(2,2,2,2));
    const __m128 m23 = _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(3,3,3,3));
    
    // Transform X coordinates
    const __m128 rx = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(m00, px), _mm_mul_ps(m01, py)),
        _mm_add_ps(_mm_mul_ps(m02, pz), _mm_mul_ps(m03, pw))
    );
    
    // Transform Y coordinates
    const __m128 ry = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(m10, px), _mm_mul_ps(m11, py)),
        _mm_add_ps(_mm_mul_ps(m12, pz), _mm_mul_ps(m13, pw))
    );
    
    // Transform Z coordinates
    const __m128 rz = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(m20, px), _mm_mul_ps(m21, py)),
        _mm_add_ps(_mm_mul_ps(m22, pz), _mm_mul_ps(m23, pw))
    );
    
    // Store results
    _mm_store_ps(result_x, rx);
    _mm_store_ps(result_y, ry);
    _mm_store_ps(result_z, rz);
}

// SIMD color interpolation (bilinear for 4 pixels)
inline void bilinearInterpolateAVX(const uint8_t* image_data, int width, int height,
                                  const float* pixel_x, const float* pixel_y,
                                  uint8_t* result_r, uint8_t* result_g, uint8_t* result_b) {
    
    for (int i = 0; i < 4; ++i) {
        const float px = pixel_x[i];
        const float py = pixel_y[i];
        
        const int x0 = static_cast<int>(px);
        const int y0 = static_cast<int>(py);
        const int x1 = std::min(x0 + 1, width - 1);
        const int y1 = std::min(y0 + 1, height - 1);
        
        const float fx = px - x0;
        const float fy = py - y0;
        
        // Get four neighboring pixels (assuming RGB format)
        const uint8_t* p00 = &image_data[(y0 * width + x0) * 3];
        const uint8_t* p10 = &image_data[(y0 * width + x1) * 3];
        const uint8_t* p01 = &image_data[(y1 * width + x0) * 3];
        const uint8_t* p11 = &image_data[(y1 * width + x1) * 3];
        
        // Interpolate each channel
        for (int c = 0; c < 3; ++c) {
            const float top = p00[c] * (1.0f - fx) + p10[c] * fx;
            const float bottom = p01[c] * (1.0f - fx) + p11[c] * fx;
            const float interpolated = top * (1.0f - fy) + bottom * fy;
            
            if (c == 0) result_r[i] = static_cast<uint8_t>(std::round(interpolated));
            else if (c == 1) result_g[i] = static_cast<uint8_t>(std::round(interpolated));
            else result_b[i] = static_cast<uint8_t>(std::round(interpolated));
        }
    }
}

// Detect SIMD capabilities at runtime
struct SIMDCapabilities {
    bool has_sse42 = false;
    bool has_avx = false;
    bool has_avx2 = false;
    bool has_avx512 = false;
    
    SIMDCapabilities() {
        detectCapabilities();
    }
    
private:
    void detectCapabilities() {
        int info[4];
        
        // Check for SSE4.2
        __cpuid(info, 1);
        has_sse42 = (info[2] & (1 << 20)) != 0;
        
        // Check for AVX
        has_avx = (info[2] & (1 << 28)) != 0;
        
        // Check for AVX2
        if (has_avx) {
            __cpuid(info, 7);
            has_avx2 = (info[1] & (1 << 5)) != 0;
            has_avx512 = (info[1] & (1 << 16)) != 0;
        }
    }
};

} // namespace simd
} // namespace performance  
} // namespace prism
```

### 2. Adaptive SIMD Dispatch

```cpp
// include/prism/performance/SIMDDispatcher.hpp
namespace prism {
namespace performance {

class SIMDDispatcher {
public:
    static SIMDDispatcher& getInstance() {
        static SIMDDispatcher instance;
        return instance;
    }
    
    // Function pointer types for different SIMD implementations
    using CatmullRomFunc = void(*)(const float*, const float*, const float*, const float*, 
                                  float, float*);
    using TransformFunc = void(*)(const float*, const float*, const float*,
                                 const float*, float*, float*, float*);
    using ColorInterpolateFunc = void(*)(const uint8_t*, int, int, const float*, const float*,
                                        uint8_t*, uint8_t*, uint8_t*);
    
    CatmullRomFunc getCatmullRomFunction() const {
        if (capabilities_.has_avx2) {
            return simd::catmullRomAVX2;
        } else {
            return catmullRomScalar;
        }
    }
    
    TransformFunc getTransformFunction() const {
        if (capabilities_.has_avx) {
            return simd::transformPoints4xAVX;
        } else {
            return transformScalar;
        }
    }
    
    ColorInterpolateFunc getColorInterpolateFunction() const {
        if (capabilities_.has_avx) {
            return simd::bilinearInterpolateAVX;
        } else {
            return colorInterpolateScalar;  
        }
    }
    
    const simd::SIMDCapabilities& getCapabilities() const {
        return capabilities_;
    }
    
private:
    simd::SIMDCapabilities capabilities_;
    
    // Scalar fallback implementations
    static void catmullRomScalar(const float* p0, const float* p1, const float* p2, const float* p3,
                                float t, float* result);
    static void transformScalar(const float* px, const float* py, const float* pz,
                               const float* matrix, float* rx, float* ry, float* rz);
    static void colorInterpolateScalar(const uint8_t* image, int width, int height,
                                      const float* px, const float* py,
                                      uint8_t* r, uint8_t* g, uint8_t* b);
};

} // namespace performance
} // namespace prism
```

## Cache-Efficient Data Structures

### 1. Structure-of-Arrays Layout

```cpp
// include/prism/performance/CacheOptimizedStructures.hpp
namespace prism {
namespace performance {

// Cache-optimized point cloud using SoA layout
template<typename Scalar = float>
class CacheOptimizedPointCloud {
public:
    struct PointData {
        std::vector<Scalar> x, y, z, intensity;
        std::vector<uint8_t> r, g, b;
        std::vector<uint32_t> ring;  // Beam/ring index
        
        size_t size() const { return x.size(); }
        
        void resize(size_t new_size) {
            x.resize(new_size);
            y.resize(new_size);
            z.resize(new_size);
            intensity.resize(new_size);
            r.resize(new_size);
            g.resize(new_size);
            b.resize(new_size);
            ring.resize(new_size);
        }
        
        void reserve(size_t capacity) {
            x.reserve(capacity);
            y.reserve(capacity);
            z.reserve(capacity);
            intensity.reserve(capacity);
            r.reserve(capacity);
            g.reserve(capacity);
            b.reserve(capacity);
            ring.reserve(capacity);
        }
        
        // Add point
        void push_back(Scalar x_val, Scalar y_val, Scalar z_val, Scalar intensity_val,
                      uint8_t r_val = 0, uint8_t g_val = 0, uint8_t b_val = 0,
                      uint32_t ring_val = 0) {
            x.push_back(x_val);
            y.push_back(y_val);
            z.push_back(z_val);
            intensity.push_back(intensity_val);
            r.push_back(r_val);
            g.push_back(g_val);
            b.push_back(b_val);
            ring.push_back(ring_val);
        }
        
        // Get pointers for SIMD processing
        const Scalar* getXData() const { return x.data(); }
        const Scalar* getYData() const { return y.data(); }
        const Scalar* getZData() const { return z.data(); }
        const Scalar* getIntensityData() const { return intensity.data(); }
        
        Scalar* getXData() { return x.data(); }
        Scalar* getYData() { return y.data(); }
        Scalar* getZData() { return z.data(); }
        Scalar* getIntensityData() { return intensity.data(); }
        
        uint8_t* getRData() { return r.data(); }
        uint8_t* getGData() { return g.data(); }
        uint8_t* getBData() { return b.data(); }
    };
    
    CacheOptimizedPointCloud() = default;
    
    explicit CacheOptimizedPointCloud(size_t initial_capacity) {
        data_.reserve(initial_capacity);
    }
    
    // Convert from PCL point cloud
    template<typename PCLPointType>
    static CacheOptimizedPointCloud fromPCL(const pcl::PointCloud<PCLPointType>& pcl_cloud) {
        CacheOptimizedPointCloud result(pcl_cloud.size());
        
        for (const auto& point : pcl_cloud.points) {
            if constexpr (std::is_same_v<PCLPointType, pcl::PointXYZI>) {
                result.data_.push_back(point.x, point.y, point.z, point.intensity);
            } else if constexpr (std::is_same_v<PCLPointType, pcl::PointXYZRGB>) {
                result.data_.push_back(point.x, point.y, point.z, 0.0f,
                                     point.r, point.g, point.b);
            }
            // Add more point types as needed
        }
        
        return result;
    }
    
    // Convert to PCL point cloud
    template<typename PCLPointType>
    typename pcl::PointCloud<PCLPointType>::Ptr toPCL() const {
        auto pcl_cloud = std::make_shared<pcl::PointCloud<PCLPointType>>();
        pcl_cloud->points.resize(data_.size());
        
        for (size_t i = 0; i < data_.size(); ++i) {
            auto& pcl_point = pcl_cloud->points[i];
            pcl_point.x = data_.x[i];
            pcl_point.y = data_.y[i];  
            pcl_point.z = data_.z[i];
            
            if constexpr (std::is_same_v<PCLPointType, pcl::PointXYZI>) {
                pcl_point.intensity = data_.intensity[i];
            } else if constexpr (std::is_same_v<PCLPointType, pcl::PointXYZRGB>) {
                pcl_point.r = data_.r[i];
                pcl_point.g = data_.g[i];
                pcl_point.b = data_.b[i];
            }
        }
        
        pcl_cloud->width = data_.size();
        pcl_cloud->height = 1;
        pcl_cloud->is_dense = false;
        
        return pcl_cloud;
    }
    
    // Accessors
    PointData& data() { return data_; }
    const PointData& data() const { return data_; }
    
    size_t size() const { return data_.size(); }
    void resize(size_t new_size) { data_.resize(new_size); }
    void reserve(size_t capacity) { data_.reserve(capacity); }
    
    // Column processing optimized for cache
    template<typename ProcessFunc>
    void processColumns(size_t width, ProcessFunc func) const {
        const size_t height = size() / width;
        
        // Process column by column for better cache locality
        for (size_t col = 0; col < width; ++col) {
            for (size_t row = 0; row < height; ++row) {
                const size_t index = row * width + col;
                func(index, data_.x[index], data_.y[index], data_.z[index], data_.intensity[index]);
            }
        }
    }
    
private:
    PointData data_;
};

// Memory-aligned containers for SIMD operations
template<typename T>
class AlignedVector {
public:
    static constexpr size_t ALIGNMENT = 32; // 256-bit alignment for AVX2
    
    AlignedVector() = default;
    
    explicit AlignedVector(size_t size) {
        resize(size);
    }
    
    ~AlignedVector() {
        if (data_) {
            std::free(data_);
        }
    }
    
    // Move constructor
    AlignedVector(AlignedVector&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }
    
    // Move assignment
    AlignedVector& operator=(AlignedVector&& other) noexcept {
        if (this != &other) {
            if (data_) {
                std::free(data_);
            }
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }
    
    void resize(size_t new_size) {
        if (new_size > capacity_) {
            reserve(new_size * 2); // Growth factor of 2
        }
        size_ = new_size;
    }
    
    void reserve(size_t new_capacity) {
        if (new_capacity <= capacity_) return;
        
        // Align capacity to SIMD boundary
        const size_t aligned_capacity = ((new_capacity + ALIGNMENT/sizeof(T) - 1) / 
                                        (ALIGNMENT/sizeof(T))) * (ALIGNMENT/sizeof(T));
        
        T* new_data = static_cast<T*>(std::aligned_alloc(ALIGNMENT, 
                                                        aligned_capacity * sizeof(T)));
        if (!new_data) {
            throw std::bad_alloc();
        }
        
        // Copy existing data
        if (data_ && size_ > 0) {
            std::memcpy(new_data, data_, size_ * sizeof(T));
        }
        
        if (data_) {
            std::free(data_);
        }
        
        data_ = new_data;
        capacity_ = aligned_capacity;
    }
    
    // Accessors
    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    
    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }
    
    // Iterator support
    T* begin() { return data_; }
    T* end() { return data_ + size_; }
    const T* begin() const { return data_; }
    const T* end() const { return data_ + size_; }
    
private:
    T* data_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;
};

} // namespace performance
} // namespace prism
```

## Lock-Free Data Structures (v1.1+, optional)

Note: v1.0은 mutex 기반의 thread-safe 구조를 기본으로 사용합니다. Lock-free 자료구조는 v1.1+에서 선택적으로 도입합니다.

### 1. Lock-Free Ring Buffer

```cpp
// include/prism/performance/LockFreeStructures.hpp
#pragma once

#include <atomic>
#include <memory>
#include <type_traits>

namespace prism {
namespace performance {

// Lock-free single-producer single-consumer ring buffer
template<typename T>
class LockFreeRingBuffer {
public:
    explicit LockFreeRingBuffer(size_t capacity)
        : capacity_(roundUpToPowerOfTwo(capacity))
        , mask_(capacity_ - 1)
        , buffer_(new T[capacity_])
        , head_(0)
        , tail_(0) {}
    
    ~LockFreeRingBuffer() {
        delete[] buffer_;
    }
    
    // Producer: push element (returns false if full)
    bool push(const T& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) & mask_;
        
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Buffer is full
        }
        
        buffer_[current_tail] = item;
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }
    
    // Producer: push element (move version)
    bool push(T&& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) & mask_;
        
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false; // Buffer is full
        }
        
        buffer_[current_tail] = std::move(item);
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }
    
    // Consumer: pop element (returns false if empty)
    bool pop(T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        
        if (current_head == tail_.load(std::memory_order_acquire)) {
            return false; // Buffer is empty
        }
        
        item = std::move(buffer_[current_head]);
        head_.store((current_head + 1) & mask_, std::memory_order_release);
        return true;
    }
    
    // Check if buffer is empty
    bool empty() const {
        return head_.load(std::memory_order_relaxed) == 
               tail_.load(std::memory_order_relaxed);
    }
    
    // Check if buffer is full
    bool full() const {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t next_tail = (current_tail + 1) & mask_;
        return next_tail == head_.load(std::memory_order_relaxed);
    }
    
    // Get approximate size (may not be exact due to concurrent access)
    size_t size() const {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        return (current_tail - current_head) & mask_;
    }
    
    size_t capacity() const {
        return capacity_;
    }
    
private:
    const size_t capacity_;
    const size_t mask_;
    T* buffer_;
    
    // Use different cache lines to avoid false sharing
    alignas(64) std::atomic<size_t> head_;
    alignas(64) std::atomic<size_t> tail_;
    
    static size_t roundUpToPowerOfTwo(size_t n) {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        n++;
        return n;
    }
};

// Lock-free stack for memory pool allocation
template<typename T>
class LockFreeStack {
public:
    struct Node {
        T data;
        std::atomic<Node*> next;
        
        Node(const T& item) : data(item), next(nullptr) {}
        Node(T&& item) : data(std::move(item)), next(nullptr) {}
    };
    
    LockFreeStack() : head_(nullptr) {}
    
    ~LockFreeStack() {
        while (Node* old_head = head_.load()) {
            head_.store(old_head->next);
            delete old_head;
        }
    }
    
    void push(const T& item) {
        Node* new_node = new Node(item);
        Node* old_head = head_.load();
        
        do {
            new_node->next.store(old_head);
        } while (!head_.compare_exchange_weak(old_head, new_node));
    }
    
    void push(T&& item) {
        Node* new_node = new Node(std::move(item));
        Node* old_head = head_.load();
        
        do {
            new_node->next.store(old_head);
        } while (!head_.compare_exchange_weak(old_head, new_node));
    }
    
    bool pop(T& result) {
        Node* old_head = head_.load();
        
        while (old_head && 
               !head_.compare_exchange_weak(old_head, old_head->next.load())) {
            // Retry with updated old_head value
        }
        
        if (!old_head) {
            return false; // Stack is empty
        }
        
        result = std::move(old_head->data);
        delete old_head;
        return true;
    }
    
    bool empty() const {
        return head_.load() == nullptr;
    }
    
private:
    std::atomic<Node*> head_;
};

} // namespace performance
} // namespace prism
```

## Real-Time Constraints and Latency Management

### 1. Real-Time Performance Monitor

```cpp
// include/prism/performance/RealTimeMonitor.hpp
namespace prism {
namespace performance {

class RealTimeMonitor {
public:
    struct LatencyMetrics {
        std::chrono::microseconds min_latency{std::chrono::microseconds::max()};
        std::chrono::microseconds max_latency{0};
        std::chrono::microseconds avg_latency{0};
        std::chrono::microseconds p95_latency{0};
        std::chrono::microseconds p99_latency{0};
        double throughput_hz = 0.0;
        size_t frame_count = 0;
        size_t missed_deadlines = 0;
    };
    
    struct RealTimeConfig {
        std::chrono::microseconds target_frame_time{33333}; // 30 FPS
        std::chrono::microseconds deadline_warning_threshold{30000}; // 30ms warning
        std::chrono::microseconds deadline_critical_threshold{40000}; // 40ms critical
        bool enable_adaptive_quality = true;
        bool enable_frame_dropping = false;
        size_t metrics_window_size = 1000;
    };
    
    RealTimeMonitor(const RealTimeConfig& config = RealTimeConfig{})
        : config_(config)
        , latency_samples_(config_.metrics_window_size) {}
    
    // Start timing a frame
    void startFrame() {
        frame_start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    // End timing and check deadline
    bool endFrame() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto frame_latency = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - frame_start_time_);
        
        // Record latency sample
        recordLatency(frame_latency);
        
        // Check deadline
        bool deadline_met = frame_latency <= config_.target_frame_time;
        if (!deadline_met) {
            missed_deadlines_++;
            
            if (frame_latency >= config_.deadline_critical_threshold) {
                handleCriticalDeadlineMiss(frame_latency);
            } else if (frame_latency >= config_.deadline_warning_threshold) {
                handleWarningDeadlineMiss(frame_latency);
            }
        }
        
        frame_count_++;
        return deadline_met;
    }
    
    // Get current performance metrics
    LatencyMetrics getMetrics() const {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        LatencyMetrics metrics;
        metrics.frame_count = frame_count_;
        metrics.missed_deadlines = missed_deadlines_;
        
        if (latency_samples_.empty()) {
            return metrics;
        }
        
        // Calculate statistics
        auto samples = latency_samples_;
        std::sort(samples.begin(), samples.end());
        
        metrics.min_latency = samples.front();
        metrics.max_latency = samples.back();
        
        // Average
        auto total = std::accumulate(samples.begin(), samples.end(), 
                                   std::chrono::microseconds{0});
        metrics.avg_latency = total / samples.size();
        
        // Percentiles
        metrics.p95_latency = samples[static_cast<size_t>(samples.size() * 0.95)];
        metrics.p99_latency = samples[static_cast<size_t>(samples.size() * 0.99)];
        
        // Throughput
        if (frame_count_ > 0 && total.count() > 0) {
            metrics.throughput_hz = 1000000.0 * frame_count_ / total.count();
        }
        
        return metrics;
    }
    
    // Adaptive quality control
    struct QualityRecommendation {
        float interpolation_scale_factor = 3.0f;
        bool enable_simd = true;
        int projection_threads = -1;
        bool enable_distortion_correction = true;
        bool enable_multi_camera_fusion = true;
        ColorExtractor::InterpolationMethod color_method = 
            ColorExtractor::InterpolationMethod::BILINEAR;
    };
    
    QualityRecommendation getAdaptiveQualityRecommendation() const {
        auto metrics = getMetrics();
        QualityRecommendation recommendation;
        
        if (config_.enable_adaptive_quality) {
            // Adjust quality based on performance
            const double deadline_miss_rate = static_cast<double>(metrics.missed_deadlines) / 
                                            std::max(1UL, metrics.frame_count);
            
            if (deadline_miss_rate > 0.1) { // >10% miss rate
                // Reduce quality for better performance
                recommendation.interpolation_scale_factor = 2.0f;
                recommendation.enable_distortion_correction = false;
                recommendation.color_method = ColorExtractor::InterpolationMethod::NEAREST_NEIGHBOR;
                
                if (deadline_miss_rate > 0.2) { // >20% miss rate
                    recommendation.enable_multi_camera_fusion = false;
                }
            } else if (deadline_miss_rate < 0.01 && 
                      metrics.avg_latency < config_.target_frame_time * 0.8) {
                // Increase quality if we have performance headroom
                recommendation.interpolation_scale_factor = 4.0f;
                recommendation.color_method = ColorExtractor::InterpolationMethod::BICUBIC;
            }
        }
        
        return recommendation;
    }
    
private:
    RealTimeConfig config_;
    std::chrono::high_resolution_clock::time_point frame_start_time_;
    
    mutable std::mutex metrics_mutex_;
    std::vector<std::chrono::microseconds> latency_samples_;
    size_t sample_index_ = 0;
    std::atomic<size_t> frame_count_{0};
    std::atomic<size_t> missed_deadlines_{0};
    
    void recordLatency(std::chrono::microseconds latency) {
        std::lock_guard<std::mutex> lock(metrics_mutex_);
        
        if (latency_samples_.size() < config_.metrics_window_size) {
            latency_samples_.push_back(latency);
        } else {
            latency_samples_[sample_index_] = latency;
            sample_index_ = (sample_index_ + 1) % config_.metrics_window_size;
        }
    }
    
    void handleWarningDeadlineMiss(std::chrono::microseconds latency) {
        // Log warning or trigger quality reduction
        RCLCPP_WARN_THROTTLE(
            rclcpp::get_logger("prism_performance"),
            *rclcpp::Clock::make_shared(RCL_STEADY_TIME), 1000,
            "Frame processing took %ld μs (target: %ld μs)",
            latency.count(), config_.target_frame_time.count());
    }
    
    void handleCriticalDeadlineMiss(std::chrono::microseconds latency) {
        // Log critical error or emergency quality reduction
        RCLCPP_ERROR_THROTTLE(
            rclcpp::get_logger("prism_performance"),
            *rclcpp::Clock::make_shared(RCL_STEADY_TIME), 1000,
            "Critical deadline miss: %ld μs (target: %ld μs)",
            latency.count(), config_.target_frame_time.count());
    }
};

} // namespace performance
} // namespace prism
```

## Performance Profiling and Benchmarking Tools

### 1. Comprehensive Benchmarking Framework

```cpp
// include/prism/performance/Benchmarking.hpp
namespace prism {
namespace performance {

class PRISMBenchmark {
public:
    struct BenchmarkConfig {
        size_t warmup_iterations = 10;
        size_t measurement_iterations = 100;
        bool enable_cpu_profiling = false;
        bool enable_memory_profiling = false;
        std::string output_file = "prism_benchmark_results.json";
    };
    
    struct BenchmarkResult {
        std::string test_name;
        std::chrono::microseconds min_time{std::chrono::microseconds::max()};
        std::chrono::microseconds max_time{0};
        std::chrono::microseconds mean_time{0};
        std::chrono::microseconds median_time{0};
        std::chrono::microseconds stddev_time{0};
        double throughput_hz = 0.0;
        size_t memory_peak_mb = 0;
        double cpu_usage_percent = 0.0;
    };
    
    // Benchmark interpolation performance
    static BenchmarkResult benchmarkInterpolation(
        const BenchmarkConfig& config = BenchmarkConfig{}) {
        
        // Generate test data
        auto test_cloud = generateTestPointCloud(32, 1024); // OS1-32 format
        auto beam_altitudes = generateBeamAltitudes();
        
        interpolation::CubicInterpolator interpolator;
        interpolation::CubicInterpolator::InterpolationConfig interp_config;
        interp_config.scale_factor = 3.0f;
        interp_config.enable_simd = true;
        
        return runBenchmark("interpolation", config, [&]() {
            auto result = interpolator.interpolate(*test_cloud, beam_altitudes, interp_config);
            benchmark::DoNotOptimize(result);
        });
    }
    
    // Benchmark projection performance
    static BenchmarkResult benchmarkProjection(
        const BenchmarkConfig& config = BenchmarkConfig{}) {
        
        // Generate test data
        auto lidar_points = generateTestLidarPoints(100000);
        auto camera_setup = generateTestCameraSetup();
        
        projection::CoordinateTransformer transformer;
        projection::ProjectionEngine projection_engine;
        
        return runBenchmark("projection", config, [&]() {
            auto transform_result = transformer.transformPointsBatch(
                *lidar_points, camera_setup.extrinsics[0]);
                
            auto projection_result = projection_engine.projectPointsBatch(
                transform_result.camera_points, 
                transform_result.valid_indices,
                camera_setup.intrinsics[0]);
                
            benchmark::DoNotOptimize(projection_result);
        });
    }
    
    // Benchmark color extraction performance
    static BenchmarkResult benchmarkColorExtraction(
        const BenchmarkConfig& config = BenchmarkConfig{}) {
        
        // Generate test data
        auto test_image = generateTestImage(640, 360);
        auto pixel_coords = generateTestPixelCoordinates(50000);
        auto point_indices = generateSequentialIndices(50000);
        
        projection::ColorExtractor color_extractor;
        projection::ColorExtractor::ExtractionConfig extract_config;
        extract_config.interpolation = projection::ColorExtractor::InterpolationMethod::BILINEAR;
        
        return runBenchmark("color_extraction", config, [&]() {
            auto result = color_extractor.extractColors(
                test_image, pixel_coords, point_indices, extract_config);
            benchmark::DoNotOptimize(result);
        });
    }
    
    // End-to-end pipeline benchmark
    static BenchmarkResult benchmarkFullPipeline(
        const BenchmarkConfig& config = BenchmarkConfig{}) {
        
        // Generate comprehensive test data
        auto lidar_cloud = generateTestPointCloud(32, 1024);
        auto camera_images = generateTestCameraImages(2, 640, 360);
        auto camera_setup = generateTestCameraSetup();
        
        projection::PRISMProjectionPipeline pipeline;
        std_msgs::msg::Header header;
        
        return runBenchmark("full_pipeline", config, [&]() {
            auto result = pipeline.processLiDARFrame(*lidar_cloud, camera_images, header);
            benchmark::DoNotOptimize(result);
        });
    }
    
    // Run all benchmarks and generate report
    static void runFullBenchmarkSuite(const BenchmarkConfig& config = BenchmarkConfig{}) {
        std::vector<BenchmarkResult> results;
        
        std::cout << "Running PRISM Performance Benchmark Suite...\n" << std::endl;
        
        std::cout << "1. Benchmarking Interpolation..." << std::endl;
        results.push_back(benchmarkInterpolation(config));
        
        std::cout << "2. Benchmarking Projection..." << std::endl;
        results.push_back(benchmarkProjection(config));
        
        std::cout << "3. Benchmarking Color Extraction..." << std::endl;
        results.push_back(benchmarkColorExtraction(config));
        
        std::cout << "4. Benchmarking Full Pipeline..." << std::endl;
        results.push_back(benchmarkFullPipeline(config));
        
        // Generate detailed report
        generateBenchmarkReport(results, config.output_file);
        printBenchmarkSummary(results);
    }
    
private:
    template<typename TestFunc>
    static BenchmarkResult runBenchmark(const std::string& name,
                                       const BenchmarkConfig& config,
                                       TestFunc test_function) {
        BenchmarkResult result;
        result.test_name = name;
        
        std::vector<std::chrono::microseconds> measurements;
        measurements.reserve(config.measurement_iterations);
        
        // Warmup
        for (size_t i = 0; i < config.warmup_iterations; ++i) {
            test_function();
        }
        
        // Actual measurements
        for (size_t i = 0; i < config.measurement_iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            test_function();
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            measurements.push_back(duration);
        }
        
        // Calculate statistics
        std::sort(measurements.begin(), measurements.end());
        
        result.min_time = measurements.front();
        result.max_time = measurements.back();
        result.median_time = measurements[measurements.size() / 2];
        
        auto total_time = std::accumulate(measurements.begin(), measurements.end(),
                                        std::chrono::microseconds{0});
        result.mean_time = total_time / measurements.size();
        
        // Calculate standard deviation
        double variance = 0.0;
        for (const auto& measurement : measurements) {
            double diff = measurement.count() - result.mean_time.count();
            variance += diff * diff;
        }
        result.stddev_time = std::chrono::microseconds{
            static_cast<long>(std::sqrt(variance / measurements.size()))};
        
        // Calculate throughput
        if (result.mean_time.count() > 0) {
            result.throughput_hz = 1000000.0 / result.mean_time.count();
        }
        
        return result;
    }
    
    static void generateBenchmarkReport(const std::vector<BenchmarkResult>& results,
                                       const std::string& filename);
    static void printBenchmarkSummary(const std::vector<BenchmarkResult>& results);
    
    // Test data generation functions
    static pcl::PointCloud<pcl::PointXYZI>::Ptr generateTestPointCloud(size_t height, size_t width);
    static std::vector<Eigen::Vector3f> generateTestLidarPoints(size_t count);
    static std::vector<cv::Mat> generateTestCameraImages(size_t count, int width, int height);
    static CalibrationManager::MultiCameraSetup generateTestCameraSetup();
    static cv::Mat generateTestImage(int width, int height);
    static std::vector<cv::Point2f> generateTestPixelCoordinates(size_t count);
    static std::vector<size_t> generateSequentialIndices(size_t count);
    static std::vector<float> generateBeamAltitudes();
};

} // namespace performance
} // namespace prism
```

## Single-Thread Fallback and Progressive Optimization

### 1. Single-Thread Implementation Strategy

The PRISM system implements a progressive optimization approach, starting from a robust single-threaded baseline that ensures correctness and debuggability:

```cpp
// include/prism/performance/ExecutionMode.hpp
#pragma once

namespace prism {
namespace performance {

enum class ExecutionMode {
    SINGLE_THREAD,      // Debug mode - sequential execution
    CPU_PARALLEL,       // Multi-threaded CPU execution
    CPU_SIMD,           // CPU with SIMD optimization
    GPU_CUDA,           // Full CUDA implementation (v1.1+)
    AUTO                // Runtime selection based on workload
};

class ExecutionManager {
public:
    struct RuntimeConfig {
        ExecutionMode mode = ExecutionMode::AUTO;
        bool force_single_thread_debug = false;
        bool enable_profiling = false;
        bool enable_validation = false;
        size_t workload_threshold_for_gpu = 100000; // points
    };
    
    static ExecutionMode selectOptimalMode(size_t point_count, 
                                          size_t image_count,
                                          const RuntimeConfig& config) {
        // Debug mode override
        if (config.force_single_thread_debug || config.enable_validation) {
            return ExecutionMode::SINGLE_THREAD;
        }
        
        // Auto selection based on workload size
        if (config.mode == ExecutionMode::AUTO) {
            if (point_count < 10000) {
                // Small workload - single thread is fastest (no overhead)
                return ExecutionMode::SINGLE_THREAD;
            } else if (point_count < 50000) {
                // Medium workload - CPU parallel is optimal
                return ExecutionMode::CPU_PARALLEL;
            } else if (point_count < config.workload_threshold_for_gpu) {
                // Large workload - SIMD provides best CPU performance
                return ExecutionMode::CPU_SIMD;
            } else if (checkCUDAAvailable()) {
                // Very large workload - GPU if available
                return ExecutionMode::GPU_CUDA;
            } else {
                // Fallback to best CPU mode
                return ExecutionMode::CPU_SIMD;
            }
        }
        
        return config.mode;
    }
    
    // Single-thread reference implementation for validation
    template<typename T>
    static void validateAgainstReference(const T& parallel_result,
                                        const std::function<T()>& reference_impl,
                                        double tolerance = 1e-6) {
        const T reference_result = reference_impl();
        if (!compareResults(parallel_result, reference_result, tolerance)) {
            throw std::runtime_error("Validation failed: Results differ from reference");
        }
    }
};

} // namespace performance
} // namespace prism
```

### 2. Debug-Friendly Single-Thread Pipeline

```cpp
// src/performance/SingleThreadPipeline.cpp
#include "prism/performance/SingleThreadPipeline.hpp"

namespace prism {
namespace performance {

class SingleThreadPipeline {
public:
    struct DebugConfig {
        bool step_through = false;          // Pause after each stage
        bool visualize_intermediate = false; // Show intermediate results
        bool log_timings = true;            // Log stage timings
        bool validate_outputs = false;      // Validate each stage output
    };
    
    // Single-threaded reference implementation
    sensor_msgs::msg::PointCloud2 process(
        const sensor_msgs::msg::PointCloud2& lidar_msg,
        const sensor_msgs::msg::Image& camera1_msg,
        const sensor_msgs::msg::Image& camera2_msg,
        const DebugConfig& debug = {}) {
        
        StageTimer timer("SingleThreadPipeline");
        
        // Stage 1: Convert messages
        timer.startStage("Message Conversion");
        auto point_cloud = convertToPointCloud(lidar_msg);
        auto image1 = convertToOpenCV(camera1_msg);
        auto image2 = convertToOpenCV(camera2_msg);
        timer.endStage();
        
        if (debug.step_through) waitForDebugger();
        
        // Stage 2: Interpolation
        timer.startStage("Interpolation");
        auto interpolated = interpolateChannels(point_cloud);
        timer.endStage();
        
        if (debug.visualize_intermediate) {
            visualizePointCloud(interpolated, "After Interpolation");
        }
        
        // Stage 3: Projection
        timer.startStage("Projection");
        auto proj1 = projectToCamera(interpolated, calibration_.camera1);
        auto proj2 = projectToCamera(interpolated, calibration_.camera2);
        timer.endStage();
        
        // Stage 4: Color extraction
        timer.startStage("Color Extraction");
        auto colored = extractColors(interpolated, image1, proj1, image2, proj2);
        timer.endStage();
        
        // Stage 5: Convert back to ROS message
        timer.startStage("Message Creation");
        auto output_msg = convertToROSMessage(colored);
        timer.endStage();
        
        if (debug.log_timings) {
            timer.printReport();
        }
        
        return output_msg;
    }
    
private:
    // Simple, readable implementations for debugging
    PointCloudSoA interpolateChannels(const PointCloudSoA& input) {
        PointCloudSoA output;
        const int scale_factor = 3;
        
        // Reserve space
        output.x.reserve(input.size() * scale_factor);
        output.y.reserve(input.size() * scale_factor);
        output.z.reserve(input.size() * scale_factor);
        
        // Simple interpolation - easy to debug
        for (size_t ring = 0; ring < 32 - 1; ++ring) {
            for (size_t col = 0; col < 1024; ++col) {
                size_t idx1 = ring * 1024 + col;
                size_t idx2 = (ring + 1) * 1024 + col;
                
                // Original point
                output.x.push_back(input.x[idx1]);
                output.y.push_back(input.y[idx1]);
                output.z.push_back(input.z[idx1]);
                
                // Interpolated points
                for (int i = 1; i < scale_factor; ++i) {
                    float t = static_cast<float>(i) / scale_factor;
                    output.x.push_back(lerp(input.x[idx1], input.x[idx2], t));
                    output.y.push_back(lerp(input.y[idx1], input.y[idx2], t));
                    output.z.push_back(lerp(input.z[idx1], input.z[idx2], t));
                }
            }
        }
        
        return output;
    }
    
    float lerp(float a, float b, float t) const {
        return a * (1.0f - t) + b * t;
    }
};

} // namespace performance
} // namespace prism
```

## GPU Acceleration Strategies (v1.1+ Future Enhancement)

### 1. OpenCV GPU Methods for Simple Operations

OpenCV's GPU module provides easy-to-use acceleration for common image processing operations without the complexity of custom CUDA kernels:

```cpp
// include/prism/gpu/OpenCVGPUAcceleration.hpp
#pragma once

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

namespace prism {
namespace gpu {

class OpenCVGPUProcessor {
public:
    OpenCVGPUProcessor() {
        if (!cv::cuda::getCudaEnabledDeviceCount()) {
            throw std::runtime_error("No CUDA devices available for OpenCV");
        }
        
        // Set optimal GPU device
        cv::cuda::setDevice(0);
        
        // Pre-allocate GPU memory for common operations
        preallocateGPUBuffers();
    }
    
    // GPU-accelerated image undistortion
    cv::cuda::GpuMat undistortImage(const cv::cuda::GpuMat& distorted_image,
                                    const CameraCalibration& calib) {
        if (!undistort_maps_initialized_) {
            initializeUndistortMaps(calib);
        }
        
        cv::cuda::GpuMat undistorted;
        cv::cuda::remap(distorted_image, undistorted,
                        gpu_map1_, gpu_map2_,
                        cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT);
        return undistorted;
    }
    
    // GPU-accelerated bilinear interpolation for color extraction
    cv::Vec3b extractColorGPU(const cv::cuda::GpuMat& image,
                              const std::vector<cv::Point2f>& points) {
        // Upload points to GPU
        cv::cuda::GpuMat gpu_points(points);
        
        // Use texture memory for fast interpolation
        cv::cuda::GpuMat colors;
        cv::cuda::remap(image, colors, gpu_points, cv::cuda::GpuMat(),
                       cv::INTER_LINEAR);
        
        // Download results
        cv::Mat cpu_colors;
        colors.download(cpu_colors);
        
        return cpu_colors.at<cv::Vec3b>(0);
    }
    
    // Batch resize for multi-scale processing
    std::vector<cv::cuda::GpuMat> createImagePyramid(const cv::cuda::GpuMat& image,
                                                     int levels = 3) {
        std::vector<cv::cuda::GpuMat> pyramid(levels);
        pyramid[0] = image;
        
        for (int i = 1; i < levels; ++i) {
            cv::cuda::pyrDown(pyramid[i-1], pyramid[i]);
        }
        
        return pyramid;
    }
    
    // GPU-accelerated color space conversion
    cv::cuda::GpuMat convertColorSpace(const cv::cuda::GpuMat& input,
                                       int conversion_code) {
        cv::cuda::GpuMat output;
        cv::cuda::cvtColor(input, output, conversion_code);
        return output;
    }
    
private:
    cv::cuda::GpuMat gpu_map1_, gpu_map2_;
    bool undistort_maps_initialized_ = false;
    
    void initializeUndistortMaps(const CameraCalibration& calib) {
        cv::Mat map1, map2;
        cv::initUndistortRectifyMap(calib.K, calib.dist_coeffs,
                                   cv::Mat(), calib.K,
                                   calib.image_size,
                                   CV_32FC1, map1, map2);
        
        gpu_map1_.upload(map1);
        gpu_map2_.upload(map2);
        undistort_maps_initialized_ = true;
    }
    
    void preallocateGPUBuffers() {
        // Pre-allocate to avoid runtime allocation overhead
        temp_buffer_ = cv::cuda::createContinuous(1080, 1920, CV_8UC3);
    }
    
    cv::cuda::GpuMat temp_buffer_;
};

} // namespace gpu
} // namespace prism
```

### 2. CUDA Kernels for Complex Operations

For performance-critical operations where OpenCV GPU methods are insufficient, custom CUDA kernels provide maximum performance:

```cpp
// include/prism/gpu/CUDAKernels.cuh
#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace prism {
namespace gpu {

// CUDA kernel for batch point projection
__global__ void projectPointsKernel(
    const float3* lidar_points,      // Input: 3D points
    const float* extrinsic_matrix,   // 4x4 transformation matrix
    const float* intrinsic_matrix,   // 3x3 camera matrix
    const float* dist_coeffs,        // Distortion coefficients
    float2* projected_points,        // Output: 2D projections
    bool* valid_projections,         // Output: validity flags
    int num_points) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    // Load point
    float3 point = lidar_points[idx];
    
    // Transform to camera coordinates (shared memory optimization)
    __shared__ float shared_extrinsic[16];
    if (threadIdx.x < 16) {
        shared_extrinsic[threadIdx.x] = extrinsic_matrix[threadIdx.x];
    }
    __syncthreads();
    
    // Apply transformation
    float3 cam_point;
    cam_point.x = shared_extrinsic[0] * point.x + shared_extrinsic[1] * point.y + 
                  shared_extrinsic[2] * point.z + shared_extrinsic[3];
    cam_point.y = shared_extrinsic[4] * point.x + shared_extrinsic[5] * point.y + 
                  shared_extrinsic[6] * point.z + shared_extrinsic[7];
    cam_point.z = shared_extrinsic[8] * point.x + shared_extrinsic[9] * point.y + 
                  shared_extrinsic[10] * point.z + shared_extrinsic[11];
    
    // Check if point is in front of camera
    if (cam_point.z <= 0.001f) {
        valid_projections[idx] = false;
        return;
    }
    
    // Project to image plane
    float inv_z = 1.0f / cam_point.z;
    float2 normalized;
    normalized.x = cam_point.x * inv_z;
    normalized.y = cam_point.y * inv_z;
    
    // Apply distortion (simplified radial model)
    float r2 = normalized.x * normalized.x + normalized.y * normalized.y;
    float radial = 1.0f + dist_coeffs[0] * r2 + dist_coeffs[1] * r2 * r2;
    
    normalized.x *= radial;
    normalized.y *= radial;
    
    // Apply intrinsic matrix
    projected_points[idx].x = intrinsic_matrix[0] * normalized.x + intrinsic_matrix[2];
    projected_points[idx].y = intrinsic_matrix[4] * normalized.y + intrinsic_matrix[5];
    
    // Check image bounds (assuming 640x360)
    valid_projections[idx] = (projected_points[idx].x >= 0 && 
                              projected_points[idx].x < 640 &&
                              projected_points[idx].y >= 0 && 
                              projected_points[idx].y < 360);
}

// CUDA kernel for parallel color extraction with bilinear interpolation
__global__ void extractColorsKernel(
    const uchar3* image1,           // Camera 1 image
    const uchar3* image2,           // Camera 2 image
    const float2* projections1,     // Projections to camera 1
    const float2* projections2,     // Projections to camera 2
    const bool* valid1,             // Valid flags for camera 1
    const bool* valid2,             // Valid flags for camera 2
    uchar3* output_colors,          // Output RGB values
    int num_points,
    int image_width,
    int image_height) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    uchar3 color = {0, 0, 0};
    float weight_sum = 0.0f;
    
    // Camera 1 contribution
    if (valid1[idx]) {
        float2 p = projections1[idx];
        
        // Bilinear interpolation
        int x0 = __float2int_rd(p.x);
        int y0 = __float2int_rd(p.y);
        int x1 = min(x0 + 1, image_width - 1);
        int y1 = min(y0 + 1, image_height - 1);
        
        float fx = p.x - x0;
        float fy = p.y - y0;
        
        // Texture memory would be more efficient here
        uchar3 c00 = image1[y0 * image_width + x0];
        uchar3 c10 = image1[y0 * image_width + x1];
        uchar3 c01 = image1[y1 * image_width + x0];
        uchar3 c11 = image1[y1 * image_width + x1];
        
        float3 interpolated;
        interpolated.x = (1-fx)*(1-fy)*c00.x + fx*(1-fy)*c10.x + 
                        (1-fx)*fy*c01.x + fx*fy*c11.x;
        interpolated.y = (1-fx)*(1-fy)*c00.y + fx*(1-fy)*c10.y + 
                        (1-fx)*fy*c01.y + fx*fy*c11.y;
        interpolated.z = (1-fx)*(1-fy)*c00.z + fx*(1-fy)*c10.z + 
                        (1-fx)*fy*c01.z + fx*fy*c11.z;
        
        // Distance-based weight (simplified)
        float weight = 1.0f;
        
        color.x += interpolated.x * weight;
        color.y += interpolated.y * weight;
        color.z += interpolated.z * weight;
        weight_sum += weight;
    }
    
    // Camera 2 contribution (similar logic)
    if (valid2[idx]) {
        // ... (similar bilinear interpolation for camera 2)
    }
    
    // Normalize and store
    if (weight_sum > 0) {
        output_colors[idx].x = color.x / weight_sum;
        output_colors[idx].y = color.y / weight_sum;
        output_colors[idx].z = color.z / weight_sum;
    } else {
        output_colors[idx] = {128, 128, 128}; // Default gray
    }
}

// Host wrapper class
class CUDAAccelerator {
public:
    CUDAAccelerator(size_t max_points = 100000) {
        // Allocate device memory
        cudaMalloc(&d_lidar_points_, max_points * sizeof(float3));
        cudaMalloc(&d_projected_points_, max_points * sizeof(float2) * 2);
        cudaMalloc(&d_valid_flags_, max_points * sizeof(bool) * 2);
        cudaMalloc(&d_colors_, max_points * sizeof(uchar3));
        
        // Allocate pinned host memory for fast transfers
        cudaMallocHost(&h_lidar_points_, max_points * sizeof(float3));
        cudaMallocHost(&h_colors_, max_points * sizeof(uchar3));
        
        // Create CUDA streams for async operations
        cudaStreamCreate(&projection_stream_);
        cudaStreamCreate(&color_stream_);
    }
    
    void processAsync(const PointCloudSoA& points,
                     const cv::Mat& image1,
                     const cv::Mat& image2,
                     const CalibrationData& calib) {
        
        const size_t num_points = points.size();
        
        // Copy point cloud to pinned memory
        // Use TBB for parallel copy instead of OpenMP
        tbb::parallel_for(tbb::blocked_range<size_t>(0, num_points),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    h_lidar_points_[i] = make_float3(points.x[i], points.y[i], points.z[i]);
                }
            });
        
        // Async copy to device
        cudaMemcpyAsync(d_lidar_points_, h_lidar_points_,
                       num_points * sizeof(float3),
                       cudaMemcpyHostToDevice, projection_stream_);
        
        // Upload images to device (can use texture memory for better performance)
        uploadImageToDevice(image1, d_image1_);
        uploadImageToDevice(image2, d_image2_);
        
        // Launch projection kernel
        const int threads_per_block = 256;
        const int num_blocks = (num_points + threads_per_block - 1) / threads_per_block;
        
        projectPointsKernel<<<num_blocks, threads_per_block, 0, projection_stream_>>>(
            d_lidar_points_, d_extrinsic1_, d_intrinsic1_, d_distortion1_,
            d_projected_points_, d_valid_flags_, num_points);
        
        // Launch color extraction kernel
        extractColorsKernel<<<num_blocks, threads_per_block, 0, color_stream_>>>(
            d_image1_, d_image2_, d_projected_points_, d_projected_points_ + num_points,
            d_valid_flags_, d_valid_flags_ + num_points,
            d_colors_, num_points, 640, 360);
        
        // Async copy results back
        cudaMemcpyAsync(h_colors_, d_colors_,
                       num_points * sizeof(uchar3),
                       cudaMemcpyDeviceToHost, color_stream_);
    }
    
    void synchronize() {
        cudaStreamSynchronize(projection_stream_);
        cudaStreamSynchronize(color_stream_);
    }
    
private:
    // Device memory
    float3* d_lidar_points_;
    float2* d_projected_points_;
    bool* d_valid_flags_;
    uchar3* d_colors_;
    uchar3* d_image1_;
    uchar3* d_image2_;
    
    // Calibration data on device
    float* d_extrinsic1_;
    float* d_intrinsic1_;
    float* d_distortion1_;
    
    // Pinned host memory
    float3* h_lidar_points_;
    uchar3* h_colors_;
    
    // CUDA streams
    cudaStream_t projection_stream_;
    cudaStream_t color_stream_;
};

} // namespace gpu
} // namespace prism
```

### 3. Adaptive GPU Usage Strategy

```cpp
// include/prism/gpu/AdaptiveGPUStrategy.hpp
#pragma once

namespace prism {
namespace gpu {

class AdaptiveGPUStrategy {
public:
    struct GPUMetrics {
        size_t available_memory;
        float gpu_utilization;
        float memory_bandwidth_utilization;
        float pcie_bandwidth_available;
        std::chrono::microseconds transfer_overhead;
    };
    
    static bool shouldUseGPU(size_t data_size_bytes,
                            std::chrono::microseconds cpu_baseline,
                            const GPUMetrics& metrics) {
        
        // Estimate transfer time
        const double pcie_bandwidth_gbps = 16.0; // PCIe 3.0 x16
        const double effective_bandwidth = pcie_bandwidth_gbps * 0.7; // ~70% efficiency
        const auto transfer_time = std::chrono::microseconds(
            static_cast<long>(data_size_bytes * 2 / (effective_bandwidth * 125000)));
        
        // Estimate GPU processing speedup
        const double expected_speedup = estimateSpeedup(data_size_bytes);
        const auto gpu_compute_time = std::chrono::microseconds(
            static_cast<long>(cpu_baseline.count() / expected_speedup));
        
        // Total GPU time including transfers
        const auto total_gpu_time = transfer_time + gpu_compute_time;
        
        // Use GPU only if total time is better than CPU
        return total_gpu_time < cpu_baseline * 0.8; // 20% improvement threshold
    }
    
private:
    static double estimateSpeedup(size_t data_size) {
        // Empirical model based on benchmarks
        if (data_size < 1024 * 1024) {       // < 1MB
            return 0.5;  // CPU is faster
        } else if (data_size < 10 * 1024 * 1024) { // < 10MB
            return 2.0;  // Modest speedup
        } else if (data_size < 100 * 1024 * 1024) { // < 100MB
            return 5.0;  // Good speedup
        } else {
            return 10.0; // Excellent speedup
        }
    }
};

} // namespace gpu
} // namespace prism
```

This comprehensive performance optimization guide provides the foundation for achieving real-time LiDAR-camera fusion in PRISM, with progressive optimization from single-threaded debugging to full GPU acceleration, allowing developers to choose the appropriate level of optimization based on their hardware capabilities and performance requirements.