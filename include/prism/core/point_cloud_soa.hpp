#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <stdexcept>

namespace prism {
namespace core {

/**
 * @brief Structure-of-Arrays point cloud representation for cache-efficient processing
 * 
 * This data structure separates point cloud attributes into individual arrays
 * to improve cache locality and enable SIMD vectorization. All coordinate arrays
 * are aligned to 32-byte boundaries for AVX2 optimization.
 */
class PointCloudSoA {
public:
    // Coordinate arrays (32-byte aligned for AVX2)
    alignas(32) std::vector<float> x;
    alignas(32) std::vector<float> y;
    alignas(32) std::vector<float> z;
    alignas(32) std::vector<float> intensity;
    
    // Color arrays
    std::vector<uint8_t> r;
    std::vector<uint8_t> g;
    std::vector<uint8_t> b;
    
    // Additional metadata
    std::vector<uint16_t> ring;  // LiDAR ring/channel information
    std::vector<float> timestamp; // Per-point timestamps (optional)
    
    // Constructors
    PointCloudSoA() = default;
    explicit PointCloudSoA(size_t initial_capacity);
    
    // Move constructor and assignment
    PointCloudSoA(PointCloudSoA&& other) noexcept = default;
    PointCloudSoA& operator=(PointCloudSoA&& other) noexcept = default;
    
    // Copy constructor and assignment
    PointCloudSoA(const PointCloudSoA& other) = default;
    PointCloudSoA& operator=(const PointCloudSoA& other) = default;
    
    /**
     * @brief Get the number of points in the cloud
     * @return Number of points
     */
    size_t size() const noexcept {
        return x.size();
    }
    
    /**
     * @brief Check if the cloud is empty
     * @return True if empty, false otherwise
     */
    bool empty() const noexcept {
        return x.empty();
    }
    
    /**
     * @brief Reserve capacity for all arrays
     * @param capacity Number of points to reserve space for
     */
    void reserve(size_t capacity);
    
    /**
     * @brief Resize all arrays to the specified size
     * @param new_size New size for all arrays
     */
    void resize(size_t new_size);
    
    /**
     * @brief Clear all arrays
     */
    void clear() noexcept;
    
    /**
     * @brief Add a point to the cloud
     * @param px X coordinate
     * @param py Y coordinate
     * @param pz Z coordinate
     * @param pi Intensity value
     * @param pr Red color value (default 0)
     * @param pg Green color value (default 0)
     * @param pb Blue color value (default 0)
     * @param pring Ring/channel ID (default 0)
     */
    void addPoint(float px, float py, float pz, float pi,
                  uint8_t pr = 0, uint8_t pg = 0, uint8_t pb = 0,
                  uint16_t pring = 0);
    
    /**
     * @brief Add multiple points in batch
     * @param points_x X coordinates
     * @param points_y Y coordinates
     * @param points_z Z coordinates
     * @param points_intensity Intensity values
     * @param count Number of points to add
     */
    void addPointsBatch(const float* points_x, const float* points_y,
                       const float* points_z, const float* points_intensity,
                       size_t count);
    
    /**
     * @brief Get pointers to coordinate data for SIMD processing
     * @return Pointer to X coordinate array
     */
    float* getXData() noexcept { return x.data(); }
    const float* getXData() const noexcept { return x.data(); }
    
    float* getYData() noexcept { return y.data(); }
    const float* getYData() const noexcept { return y.data(); }
    
    float* getZData() noexcept { return z.data(); }
    const float* getZData() const noexcept { return z.data(); }
    
    float* getIntensityData() noexcept { return intensity.data(); }
    const float* getIntensityData() const noexcept { return intensity.data(); }
    
    /**
     * @brief Get pointers to color data
     * @return Pointer to red color array
     */
    uint8_t* getRData() noexcept { return r.data(); }
    const uint8_t* getRData() const noexcept { return r.data(); }
    
    uint8_t* getGData() noexcept { return g.data(); }
    const uint8_t* getGData() const noexcept { return g.data(); }
    
    uint8_t* getBData() noexcept { return b.data(); }
    const uint8_t* getBData() const noexcept { return b.data(); }
    
    /**
     * @brief Create a view of a subset of points
     * @param start_idx Starting index
     * @param count Number of points in the view
     * @return New PointCloudSoA containing the subset (deep copy)
     */
    PointCloudSoA createView(size_t start_idx, size_t count) const;
    
    /**
     * @brief Validate that all arrays have the same size
     * @return True if valid, false otherwise
     */
    bool validate() const noexcept;
    
    /**
     * @brief Get memory usage in bytes
     * @return Total memory usage of all arrays
     */
    size_t getMemoryUsage() const noexcept;
    
    /**
     * @brief Swap contents with another PointCloudSoA
     * @param other The other point cloud
     */
    void swap(PointCloudSoA& other) noexcept;
    
    /**
     * @brief Apply a transformation matrix to all points
     * @param transform 4x4 transformation matrix (row-major)
     * @note This is a placeholder for future SIMD optimization
     */
    void transform(const float transform[16]);
    
    /**
     * @brief Check if color data is available
     * @return True if RGB arrays are populated
     */
    bool hasColor() const noexcept {
        return !r.empty() && r.size() == x.size();
    }
    
    /**
     * @brief Check if ring data is available
     * @return True if ring array is populated
     */
    bool hasRing() const noexcept {
        return !ring.empty() && ring.size() == x.size();
    }
    
    /**
     * @brief Check if timestamp data is available
     * @return True if timestamp array is populated
     */
    bool hasTimestamp() const noexcept {
        return !timestamp.empty() && timestamp.size() == x.size();
    }
    
    /**
     * @brief Enable color arrays and initialize with default values
     */
    void enableColor(uint8_t default_r = 0, uint8_t default_g = 0, uint8_t default_b = 0);
    
    /**
     * @brief Enable ring array and initialize with default value
     */
    void enableRing(uint16_t default_ring = 0);
    
    /**
     * @brief Enable timestamp array and initialize with default value
     */
    void enableTimestamp(float default_timestamp = 0.0f);
};

/**
 * @brief Utility functions for PointCloudSoA
 */
namespace utils {
    
    /**
     * @brief Merge multiple point clouds into one
     * @param clouds Vector of point clouds to merge
     * @return Merged point cloud
     */
    PointCloudSoA merge(const std::vector<PointCloudSoA>& clouds);
    
    /**
     * @brief Split a point cloud into multiple parts
     * @param cloud The cloud to split
     * @param num_parts Number of parts to split into
     * @return Vector of split point clouds
     */
    std::vector<PointCloudSoA> split(const PointCloudSoA& cloud, size_t num_parts);
    
    /**
     * @brief Check if memory is properly aligned for SIMD
     * @param ptr Pointer to check
     * @param alignment Required alignment in bytes
     * @return True if aligned, false otherwise
     */
    inline bool isAligned(const void* ptr, size_t alignment) noexcept {
        return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
    }
    
} // namespace utils

} // namespace core
} // namespace prism