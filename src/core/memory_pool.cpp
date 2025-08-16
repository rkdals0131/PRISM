#include "prism/core/memory_pool.hpp"
#include "prism/core/point_cloud_soa.hpp"
#include <iostream>
#include <sstream>

namespace prism {
namespace core {

// Explicit instantiation for PointCloudSoA to improve compile times
template class MemoryPool<PointCloudSoA>;

// Factory function for creating optimized PointCloudSoA pools
std::unique_ptr<MemoryPool<PointCloudSoA>> createPointCloudPool(size_t initial_size, 
                                                                 size_t max_size) {
    MemoryPool<PointCloudSoA>::Config config;
    config.initial_size = initial_size > 0 ? initial_size : 20;
    config.max_size = max_size > 0 ? max_size : 100;
    config.allow_growth = true;
    config.block_when_empty = true;
    
    return std::make_unique<MemoryPool<PointCloudSoA>>(config);
}

// Utility function to format pool statistics
std::string formatPoolStats(const MemoryPool<PointCloudSoA>::Stats& stats) {
    std::stringstream ss;
    ss << "MemoryPool Statistics:\n"
       << "  Total Acquisitions: " << stats.total_acquisitions << "\n"
       << "  Total Releases: " << stats.total_releases << "\n"
       << "  Current Usage: " << stats.current_usage << "\n"
       << "  Peak Usage: " << stats.peak_usage << "\n"
       << "  Wait Count: " << stats.wait_count << "\n"
       << "  Growth Count: " << stats.growth_count << "\n";
    return ss.str();
}

} // namespace core
} // namespace prism