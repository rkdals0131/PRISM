#pragma once

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <chrono>
#include <limits>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_traits.h>
#include "prism/utils/common_types.hpp"

namespace prism {
namespace projection {

/**
 * @brief 3D point with intensity for LiDAR data
 */
struct LiDARPoint {
    float x, y, z;
    float intensity;
    size_t original_index; // Index in original input cloud order
    
    LiDARPoint() : x(0), y(0), z(0), intensity(0), original_index(0) {}
    LiDARPoint(float x_, float y_, float z_, float intensity_ = 0.0f, size_t index_ = 0) 
        : x(x_), y(y_), z(z_), intensity(intensity_), original_index(index_) {}
};

/**
 * @brief 2D pixel coordinates in image space
 */
struct PixelPoint {
    float u, v;  // Pixel coordinates
    float depth; // Distance from camera
    float intensity; // Original LiDAR intensity
    size_t original_index; // Back-reference to original LiDAR point index
    
    PixelPoint() : u(0), v(0), depth(0), intensity(0), original_index(0) {}
    PixelPoint(float u_, float v_, float depth_, float intensity_ = 0.0f, size_t index_ = 0)
        : u(u_), v(v_), depth(depth_), intensity(intensity_), original_index(index_) {}
    
    // Check if pixel is within image bounds
    bool isWithinBounds(int width, int height) const {
        return u >= 0 && u < width && v >= 0 && v < height;
    }
};

/**
 * @brief Projection result for a single camera
 */
struct CameraProjection {
    std::string camera_id;
    std::vector<PixelPoint> projected_points;
    cv::Mat debug_image; // For visualization
    size_t original_point_count = 0;
    size_t projected_point_count = 0;
    
    // Statistics
    float min_depth = std::numeric_limits<float>::max();
    float max_depth = std::numeric_limits<float>::lowest();
    float avg_depth = 0.0f;
    
    void clear() {
        projected_points.clear();
        debug_image = cv::Mat();
        original_point_count = 0;
        projected_point_count = 0;
        min_depth = std::numeric_limits<float>::max();
        max_depth = std::numeric_limits<float>::lowest();
        avg_depth = 0.0f;
    }
    
    void computeStatistics() {
        if (projected_points.empty()) {
            avg_depth = 0.0f;
            return;
        }
        
        float sum_depth = 0.0f;
        min_depth = std::numeric_limits<float>::max();
        max_depth = std::numeric_limits<float>::lowest();
        
        for (const auto& point : projected_points) {
            sum_depth += point.depth;
            min_depth = std::min(min_depth, point.depth);
            max_depth = std::max(max_depth, point.depth);
        }
        
        avg_depth = sum_depth / projected_points.size();
        projected_point_count = projected_points.size();
    }
};

/**
 * @brief Complete projection result for all cameras
 * Now extends BaseResult for common metadata
 */
struct ProjectionResult : public prism::utils::BaseResult {
    std::vector<CameraProjection> camera_projections;
    
    /**
     * @brief Override clear to handle custom members
     */
    void clear() override {
        prism::utils::BaseResult::clear();  // Call base clear
        camera_projections.clear();
    }
    
    // Get projection for specific camera
    CameraProjection* getCameraProjection(const std::string& camera_id) {
        for (auto& proj : camera_projections) {
            if (proj.camera_id == camera_id) {
                return &proj;
            }
        }
        return nullptr;
    }
    
    const CameraProjection* getCameraProjection(const std::string& camera_id) const {
        for (const auto& proj : camera_projections) {
            if (proj.camera_id == camera_id) {
                return &proj;
            }
        }
        return nullptr;
    }
};

/**
 * @brief Configuration for projection engine
 */
struct ProjectionConfig {
    // Depth filtering
    float min_depth = 0.1f;    // Minimum depth in meters
    float max_depth = 100.0f;  // Maximum depth in meters
    
    // Image bounds margin (pixels)
    int margin_pixels = 5;
    
    // Enable frustum culling (remove points behind camera)
    bool enable_frustum_culling = true;
    
    // Enable distortion correction
    bool enable_distortion_correction = true;
    
    // Debug visualization
    bool enable_debug_visualization = false;
    int debug_point_radius = 2;
    cv::Scalar debug_point_color = cv::Scalar(0, 255, 0); // Green
    
    // Performance settings
    bool enable_parallel_processing = true;
    size_t max_threads = 0; // 0 = auto-detect
    // Point sampling to reduce workload
    int sample_stride = 1;         // use every Nth point (>=1)
    size_t max_points_per_frame = 0; // 0 = no cap
    
    ProjectionConfig() = default;
    
    /**
     * @brief Load configuration from YAML node
     */
    void loadFromYaml(const YAML::Node& node) {
        using prism::utils::ConfigLoader;
        
        // Depth range
        min_depth = ConfigLoader::readNestedParam(node, 
            "projection.depth_range.min", min_depth);
        max_depth = ConfigLoader::readNestedParam(node, 
            "projection.depth_range.max", max_depth);
            
        // Image bounds
        margin_pixels = ConfigLoader::readNestedParam(node,
            "projection.image_bounds.margin_pixels", margin_pixels);
            
        // Processing options
        enable_frustum_culling = ConfigLoader::readNestedParam(node,
            "projection.frustum_culling", enable_frustum_culling);
        enable_distortion_correction = ConfigLoader::readNestedParam(node,
            "projection.distortion_correction", enable_distortion_correction);
            
        // Performance
        enable_parallel_processing = ConfigLoader::readNestedParam(node,
            "projection.parallel_cameras", enable_parallel_processing);
        sample_stride = ConfigLoader::readNestedParam(node,
            "projection.sample_stride", sample_stride);
        max_points_per_frame = ConfigLoader::readNestedParam(node,
            "projection.max_points", max_points_per_frame);
            
        // Debug
        enable_debug_visualization = ConfigLoader::readNestedParam(node,
            "projection.output_debug_images", enable_debug_visualization);
    }
    
    /**
     * @brief Validate configuration
     */
    bool validate() const {
        return min_depth > 0 && max_depth > min_depth && margin_pixels >= 0;
    }
};

/**
 * @brief Camera parameters for projection
 */
struct CameraParams {
    std::string camera_id;
    
    // Intrinsic parameters
    cv::Mat camera_matrix;     // 3x3 camera matrix (K)
    cv::Mat distortion_coeffs; // Distortion coefficients
    cv::Size image_size;       // Image dimensions
    
    // Extrinsic parameters (LiDAR to Camera transform)
    Eigen::Matrix4d T_cam_lidar; // 4x4 transformation matrix
    
    // Validation
    bool is_valid = false;
    
    CameraParams() {
        T_cam_lidar.setIdentity();
    }
    
    // Validate parameters
    bool validate() const {
        if (camera_id.empty()) return false;
        if (camera_matrix.empty() || camera_matrix.type() != CV_64F) return false;
        if (camera_matrix.rows != 3 || camera_matrix.cols != 3) return false;
        if (distortion_coeffs.empty() || distortion_coeffs.type() != CV_64F) return false;
        if (image_size.width <= 0 || image_size.height <= 0) return false;
        
        // Check if camera matrix has reasonable values
        double fx = camera_matrix.at<double>(0, 0);
        double fy = camera_matrix.at<double>(1, 1);
        double cx = camera_matrix.at<double>(0, 2);
        double cy = camera_matrix.at<double>(1, 2);
        
        if (fx <= 0 || fy <= 0 || cx < 0 || cy < 0) return false;
        if (cx >= image_size.width || cy >= image_size.height) return false;
        
        return true;
    }
};

/**
 * @brief Projection statistics for monitoring
 */
struct ProjectionStats {
    size_t total_projections = 0;
    size_t total_points_processed = 0;
    size_t total_points_projected = 0;
    double avg_processing_time_ms = 0.0;
    double min_processing_time_ms = std::numeric_limits<double>::max();
    double max_processing_time_ms = 0.0;
    
    // Per-camera statistics
    std::map<std::string, size_t> camera_projection_counts;
    std::map<std::string, double> camera_avg_processing_times;
    
    void reset() {
        total_projections = 0;
        total_points_processed = 0;
        total_points_projected = 0;
        avg_processing_time_ms = 0.0;
        min_processing_time_ms = std::numeric_limits<double>::max();
        max_processing_time_ms = 0.0;
        camera_projection_counts.clear();
        camera_avg_processing_times.clear();
    }
    
    void updateStats(const ProjectionResult& result) {
        total_projections++;
        total_points_processed += result.input_count;
        total_points_projected += result.output_count;
        
        // Update timing statistics  
        double processing_ms = result.getProcessingTimeMs();
        min_processing_time_ms = std::min(min_processing_time_ms, processing_ms);
        max_processing_time_ms = std::max(max_processing_time_ms, processing_ms);
        
        // Update average (running average)
        avg_processing_time_ms = (avg_processing_time_ms * (total_projections - 1) + 
                                 processing_ms) / total_projections;
        
        // Update per-camera statistics
        for (const auto& cam_proj : result.camera_projections) {
            camera_projection_counts[cam_proj.camera_id]++;
        }
    }
    
    double getProjectionSuccessRate() const {
        if (total_points_processed == 0) return 0.0;
        return static_cast<double>(total_points_projected) / total_points_processed * 100.0;
    }
};

/**
 * @brief Utility functions for type conversions
 */
namespace utils {

/**
 * @brief Convert PCL point cloud to LiDAR points
 */
template<typename PointT>
std::vector<LiDARPoint> convertFromPCL(const typename pcl::PointCloud<PointT>::ConstPtr& cloud) {
    std::vector<LiDARPoint> points;
    points.reserve(cloud->size());
    
    for (const auto& point : *cloud) {
        LiDARPoint lidar_point(point.x, point.y, point.z);
        
        // Try to extract intensity if available
        if constexpr (pcl::traits::has_field<PointT, pcl::fields::intensity>::value) {
            lidar_point.intensity = point.intensity;
        }
        
        points.emplace_back(lidar_point);
    }
    
    return points;
}

/**
 * @brief Convert Eigen matrix to OpenCV matrix
 */
inline cv::Mat eigenToCv(const Eigen::Matrix3d& eigen_mat) {
    cv::Mat cv_mat(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            cv_mat.at<double>(i, j) = eigen_mat(i, j);
        }
    }
    return cv_mat;
}

/**
 * @brief Convert Eigen vector to OpenCV matrix
 */
inline cv::Mat eigenToCv(const Eigen::VectorXd& eigen_vec) {
    cv::Mat cv_mat(eigen_vec.size(), 1, CV_64F);
    for (int i = 0; i < eigen_vec.size(); ++i) {
        cv_mat.at<double>(i, 0) = eigen_vec(i);
    }
    return cv_mat;
}

/**
 * @brief Create visualization image with projected points
 */
cv::Mat createVisualizationImage(const CameraProjection& projection, 
                                const cv::Size& image_size,
                                const ProjectionConfig& config = ProjectionConfig());

/**
 * @brief Apply color mapping based on depth
 */
cv::Scalar getDepthColor(float depth, float min_depth, float max_depth);

} // namespace utils

} // namespace projection
} // namespace prism