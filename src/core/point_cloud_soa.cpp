#include "prism/core/point_cloud_soa.hpp"
#include <cstring>
#include <numeric>

namespace prism {
namespace core {

PointCloudSoA::PointCloudSoA(size_t initial_capacity) {
    reserve(initial_capacity);
}

void PointCloudSoA::reserve(size_t capacity) {
    x.reserve(capacity);
    y.reserve(capacity);
    z.reserve(capacity);
    intensity.reserve(capacity);
    
    // Only reserve color arrays if they're already in use
    if (!r.empty() || capacity > 0) {
        r.reserve(capacity);
        g.reserve(capacity);
        b.reserve(capacity);
    }
    
    // Only reserve ring array if it's already in use
    if (!ring.empty()) {
        ring.reserve(capacity);
    }
    
    // Only reserve timestamp array if it's already in use
    if (!timestamp.empty()) {
        timestamp.reserve(capacity);
    }
}

void PointCloudSoA::resize(size_t new_size) {
    x.resize(new_size);
    y.resize(new_size);
    z.resize(new_size);
    intensity.resize(new_size);
    
    // Resize optional arrays only if they're enabled
    if (!r.empty() || !g.empty() || !b.empty()) {
        r.resize(new_size);
        g.resize(new_size);
        b.resize(new_size);
    }
    
    if (!ring.empty()) {
        ring.resize(new_size);
    }
    
    if (!timestamp.empty()) {
        timestamp.resize(new_size);
    }
}

void PointCloudSoA::clear() noexcept {
    x.clear();
    y.clear();
    z.clear();
    intensity.clear();
    r.clear();
    g.clear();
    b.clear();
    ring.clear();
    timestamp.clear();
}

void PointCloudSoA::addPoint(float px, float py, float pz, float pi,
                             uint8_t pr, uint8_t pg, uint8_t pb,
                             uint16_t pring) {
    x.push_back(px);
    y.push_back(py);
    z.push_back(pz);
    intensity.push_back(pi);
    
    // Add color if color arrays are enabled
    if (hasColor() || (pr != 0 || pg != 0 || pb != 0)) {
        // Ensure color arrays are the same size as coordinate arrays
        if (r.size() < x.size() - 1) {
            r.resize(x.size() - 1, 0);
            g.resize(x.size() - 1, 0);
            b.resize(x.size() - 1, 0);
        }
        r.push_back(pr);
        g.push_back(pg);
        b.push_back(pb);
    }
    
    // Add ring if ring array is enabled
    if (hasRing() || pring != 0) {
        if (ring.size() < x.size() - 1) {
            ring.resize(x.size() - 1, 0);
        }
        ring.push_back(pring);
    }
}

void PointCloudSoA::addPointsBatch(const float* points_x, const float* points_y,
                                   const float* points_z, const float* points_intensity,
                                   size_t count) {
    if (count == 0) return;
    
    size_t old_size = x.size();
    size_t new_size = old_size + count;
    
    // Resize all arrays
    x.resize(new_size);
    y.resize(new_size);
    z.resize(new_size);
    intensity.resize(new_size);
    
    // Copy data using memcpy for efficiency
    std::memcpy(x.data() + old_size, points_x, count * sizeof(float));
    std::memcpy(y.data() + old_size, points_y, count * sizeof(float));
    std::memcpy(z.data() + old_size, points_z, count * sizeof(float));
    std::memcpy(intensity.data() + old_size, points_intensity, count * sizeof(float));
    
    // Resize optional arrays if needed
    if (hasColor()) {
        r.resize(new_size, 0);
        g.resize(new_size, 0);
        b.resize(new_size, 0);
    }
    
    if (hasRing()) {
        ring.resize(new_size, 0);
    }
    
    if (hasTimestamp()) {
        timestamp.resize(new_size, 0.0f);
    }
}

PointCloudSoA PointCloudSoA::createView(size_t start_idx, size_t count) const {
    if (start_idx >= size()) {
        throw std::out_of_range("Start index out of range");
    }
    
    size_t actual_count = std::min(count, size() - start_idx);
    PointCloudSoA view(actual_count);
    
    // Copy coordinate data
    view.x.assign(x.begin() + start_idx, x.begin() + start_idx + actual_count);
    view.y.assign(y.begin() + start_idx, y.begin() + start_idx + actual_count);
    view.z.assign(z.begin() + start_idx, z.begin() + start_idx + actual_count);
    view.intensity.assign(intensity.begin() + start_idx, 
                         intensity.begin() + start_idx + actual_count);
    
    // Copy optional data if available
    if (hasColor()) {
        view.r.assign(r.begin() + start_idx, r.begin() + start_idx + actual_count);
        view.g.assign(g.begin() + start_idx, g.begin() + start_idx + actual_count);
        view.b.assign(b.begin() + start_idx, b.begin() + start_idx + actual_count);
    }
    
    if (hasRing()) {
        view.ring.assign(ring.begin() + start_idx, ring.begin() + start_idx + actual_count);
    }
    
    if (hasTimestamp()) {
        view.timestamp.assign(timestamp.begin() + start_idx, 
                            timestamp.begin() + start_idx + actual_count);
    }
    
    return view;
}

bool PointCloudSoA::validate() const noexcept {
    size_t expected_size = x.size();
    
    // Check core arrays
    if (y.size() != expected_size || z.size() != expected_size || 
        intensity.size() != expected_size) {
        return false;
    }
    
    // Check optional arrays if they exist
    if (!r.empty() && (r.size() != expected_size || 
                       g.size() != expected_size || 
                       b.size() != expected_size)) {
        return false;
    }
    
    if (!ring.empty() && ring.size() != expected_size) {
        return false;
    }
    
    if (!timestamp.empty() && timestamp.size() != expected_size) {
        return false;
    }
    
    return true;
}

size_t PointCloudSoA::getMemoryUsage() const noexcept {
    size_t total = 0;
    
    // Core arrays
    total += x.capacity() * sizeof(float);
    total += y.capacity() * sizeof(float);
    total += z.capacity() * sizeof(float);
    total += intensity.capacity() * sizeof(float);
    
    // Optional arrays
    total += r.capacity() * sizeof(uint8_t);
    total += g.capacity() * sizeof(uint8_t);
    total += b.capacity() * sizeof(uint8_t);
    total += ring.capacity() * sizeof(uint16_t);
    total += timestamp.capacity() * sizeof(float);
    
    return total;
}

void PointCloudSoA::swap(PointCloudSoA& other) noexcept {
    x.swap(other.x);
    y.swap(other.y);
    z.swap(other.z);
    intensity.swap(other.intensity);
    r.swap(other.r);
    g.swap(other.g);
    b.swap(other.b);
    ring.swap(other.ring);
    timestamp.swap(other.timestamp);
}

void PointCloudSoA::transform(const float transform[16]) {
    // Simple transformation implementation
    // This can be optimized with SIMD in the future
    for (size_t i = 0; i < x.size(); ++i) {
        float px = x[i];
        float py = y[i];
        float pz = z[i];
        
        // Apply 4x4 transformation matrix
        x[i] = transform[0] * px + transform[1] * py + transform[2] * pz + transform[3];
        y[i] = transform[4] * px + transform[5] * py + transform[6] * pz + transform[7];
        z[i] = transform[8] * px + transform[9] * py + transform[10] * pz + transform[11];
        // Note: We ignore the homogeneous row [12, 13, 14, 15] for point clouds
    }
}

void PointCloudSoA::enableColor(uint8_t default_r, uint8_t default_g, uint8_t default_b) {
    if (!hasColor()) {
        r.resize(x.size(), default_r);
        g.resize(x.size(), default_g);
        b.resize(x.size(), default_b);
    }
}

void PointCloudSoA::enableRing(uint16_t default_ring) {
    if (!hasRing()) {
        ring.resize(x.size(), default_ring);
    }
}

void PointCloudSoA::enableTimestamp(float default_timestamp) {
    if (!hasTimestamp()) {
        timestamp.resize(x.size(), default_timestamp);
    }
}

// Utility functions implementation
namespace utils {

PointCloudSoA merge(const std::vector<PointCloudSoA>& clouds) {
    if (clouds.empty()) {
        return PointCloudSoA();
    }
    
    // Calculate total size
    size_t total_size = 0;
    for (const auto& cloud : clouds) {
        total_size += cloud.size();
    }
    
    PointCloudSoA result;
    result.reserve(total_size);
    
    // Check which optional arrays are present in any cloud
    bool has_color = false;
    bool has_ring = false;
    bool has_timestamp = false;
    
    for (const auto& cloud : clouds) {
        if (cloud.hasColor()) has_color = true;
        if (cloud.hasRing()) has_ring = true;
        if (cloud.hasTimestamp()) has_timestamp = true;
    }
    
    // Enable optional arrays if needed
    if (has_color) {
        result.r.reserve(total_size);
        result.g.reserve(total_size);
        result.b.reserve(total_size);
    }
    if (has_ring) {
        result.ring.reserve(total_size);
    }
    if (has_timestamp) {
        result.timestamp.reserve(total_size);
    }
    
    // Merge all clouds
    for (const auto& cloud : clouds) {
        // Append coordinate data
        result.x.insert(result.x.end(), cloud.x.begin(), cloud.x.end());
        result.y.insert(result.y.end(), cloud.y.begin(), cloud.y.end());
        result.z.insert(result.z.end(), cloud.z.begin(), cloud.z.end());
        result.intensity.insert(result.intensity.end(), 
                              cloud.intensity.begin(), cloud.intensity.end());
        
        // Append optional data
        if (has_color) {
            if (cloud.hasColor()) {
                result.r.insert(result.r.end(), cloud.r.begin(), cloud.r.end());
                result.g.insert(result.g.end(), cloud.g.begin(), cloud.g.end());
                result.b.insert(result.b.end(), cloud.b.begin(), cloud.b.end());
            } else {
                // Fill with default values
                result.r.resize(result.x.size(), 0);
                result.g.resize(result.x.size(), 0);
                result.b.resize(result.x.size(), 0);
            }
        }
        
        if (has_ring) {
            if (cloud.hasRing()) {
                result.ring.insert(result.ring.end(), cloud.ring.begin(), cloud.ring.end());
            } else {
                result.ring.resize(result.x.size(), 0);
            }
        }
        
        if (has_timestamp) {
            if (cloud.hasTimestamp()) {
                result.timestamp.insert(result.timestamp.end(), 
                                      cloud.timestamp.begin(), cloud.timestamp.end());
            } else {
                result.timestamp.resize(result.x.size(), 0.0f);
            }
        }
    }
    
    return result;
}

std::vector<PointCloudSoA> split(const PointCloudSoA& cloud, size_t num_parts) {
    if (num_parts == 0) {
        throw std::invalid_argument("Number of parts must be greater than 0");
    }
    
    std::vector<PointCloudSoA> result;
    result.reserve(num_parts);
    
    size_t points_per_part = cloud.size() / num_parts;
    size_t remainder = cloud.size() % num_parts;
    
    size_t start_idx = 0;
    for (size_t i = 0; i < num_parts; ++i) {
        size_t part_size = points_per_part;
        if (i < remainder) {
            part_size++; // Distribute remainder points among first parts
        }
        
        if (part_size > 0) {
            result.push_back(cloud.createView(start_idx, part_size));
            start_idx += part_size;
        } else {
            result.push_back(PointCloudSoA()); // Empty cloud
        }
    }
    
    return result;
}

} // namespace utils

} // namespace core
} // namespace prism