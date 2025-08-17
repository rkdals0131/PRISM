#pragma once

#include "projection_types.hpp"
#include "prism/core/calibration_manager.hpp"
#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <thread>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace prism {
namespace projection {

/**
 * @brief High-performance projection engine for LiDAR-to-camera projection
 * 
 * This engine implements CALICO's proven projection logic in a simplified form.
 * It supports dual camera projection with proper coordinate transformations,
 * frustum culling, and OpenCV-based distortion correction.
 * 
 * Key features:
 * - Multi-camera projection support
 * - Efficient coordinate transformations using Eigen
 * - Frustum culling to remove points behind cameras
 * - Image boundary checking with configurable margins
 * - Optional distortion correction using OpenCV
 * - Thread-safe operation with parallel processing support
 * - Debug visualization capabilities
 */
class ProjectionEngine {
public:
    /**
     * @brief Constructor with calibration manager
     * @param calibration_manager Shared calibration manager for camera parameters
     * @param config Projection configuration
     */
    explicit ProjectionEngine(
        std::shared_ptr<prism::core::CalibrationManager> calibration_manager,
        const ProjectionConfig& config = ProjectionConfig()
    );
    
    /**
     * @brief Destructor
     */
    ~ProjectionEngine();
    
    // Delete copy operations for thread safety
    ProjectionEngine(const ProjectionEngine&) = delete;
    ProjectionEngine& operator=(const ProjectionEngine&) = delete;
    
    // Allow move operations
    ProjectionEngine(ProjectionEngine&&) noexcept = default;
    ProjectionEngine& operator=(ProjectionEngine&&) noexcept = default;
    
    /**
     * @brief Initialize the projection engine
     * @param camera_ids List of camera IDs to initialize
     * @return True if initialization successful
     */
    bool initialize(const std::vector<std::string>& camera_ids);
    
    /**
     * @brief Project LiDAR points to all configured cameras
     * @param lidar_points Input LiDAR point cloud
     * @param result Output projection result
     * @return True if projection successful
     */
    bool projectToAllCameras(const std::vector<LiDARPoint>& lidar_points, 
                            ProjectionResult& result);
    
    /**
     * @brief Project LiDAR points to a specific camera
     * @param lidar_points Input LiDAR point cloud
     * @param camera_id Target camera ID
     * @param projection Output camera projection
     * @return True if projection successful
     */
    bool projectToCamera(const std::vector<LiDARPoint>& lidar_points,
                        const std::string& camera_id,
                        CameraProjection& projection);
    
    /**
     * @brief Project PCL point cloud to all cameras
     * @param cloud Input PCL point cloud
     * @param result Output projection result
     * @return True if projection successful
     */
    template<typename PointT>
    bool projectPCLToAllCameras(const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
                               ProjectionResult& result) {
        auto lidar_points = utils::convertFromPCL<PointT>(cloud);
        return projectToAllCameras(lidar_points, result);
    }
    
    /**
     * @brief Update configuration
     * @param config New configuration
     */
    void updateConfig(const ProjectionConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current projection configuration
     */
    ProjectionConfig getConfig() const;
    
    /**
     * @brief Add or update camera parameters
     * @param camera_id Camera identifier
     * @param params Camera parameters
     * @return True if added successfully
     */
    bool addCamera(const std::string& camera_id, const CameraParams& params);
    
    /**
     * @brief Remove camera from projection
     * @param camera_id Camera identifier
     * @return True if removed successfully
     */
    bool removeCamera(const std::string& camera_id);
    
    /**
     * @brief Get list of configured cameras
     * @return Vector of camera IDs
     */
    std::vector<std::string> getCameraIds() const;
    
    /**
     * @brief Check if camera is configured
     * @param camera_id Camera identifier
     * @return True if camera is available
     */
    bool hasCamera(const std::string& camera_id) const;
    
    /**
     * @brief Get projection statistics
     * @return Current projection statistics
     */
    ProjectionStats getStatistics() const;
    
    /**
     * @brief Reset projection statistics
     */
    void resetStatistics();
    
    /**
     * @brief Get camera parameters for debugging
     * @param camera_id Camera identifier
     * @return Camera parameters, nullptr if not found
     */
    std::shared_ptr<const CameraParams> getCameraParams(const std::string& camera_id) const;
    
    /**
     * @brief Force reload of calibration data
     * @return Number of cameras successfully reloaded
     */
    size_t reloadCalibration();
    
    /**
     * @brief Create debug visualization for projection result
     * @param result Projection result
     * @param images Input camera images (optional)
     * @return Map of camera_id to visualization images
     */
    std::unordered_map<std::string, cv::Mat> createDebugVisualization(
        const ProjectionResult& result,
        const std::unordered_map<std::string, cv::Mat>& images = {}
    );

private:
    /**
     * @brief Camera data for efficient projection
     */
    struct CameraData {
        CameraParams params;
        cv::Mat camera_matrix_inv;    // Inverse camera matrix for efficiency
        cv::Mat rvec, tvec;          // For OpenCV projectPoints
        bool is_valid = false;
        std::mutex mutex;             // Per-camera mutex for thread safety
        
        CameraData() {
            rvec = cv::Mat::zeros(3, 1, CV_64F);
            tvec = cv::Mat::zeros(3, 1, CV_64F);
        }
    };
    
    /**
     * @brief Load camera parameters from calibration manager
     * @param camera_id Camera identifier
     * @return True if loaded successfully
     */
    bool loadCameraFromCalibration(const std::string& camera_id);
    
    /**
     * @brief Transform LiDAR points to camera coordinate system
     * @param lidar_points Input LiDAR points
     * @param camera_data Camera transformation data
     * @param camera_points Output camera coordinate points
     */
    void transformPointsToCamera(const std::vector<LiDARPoint>& lidar_points,
                                const CameraData& camera_data,
                                std::vector<cv::Point3f>& camera_points,
                                std::vector<float>& intensities,
                                std::vector<size_t>& original_indices);
    
    /**
     * @brief Apply frustum culling to remove points behind camera
     * @param camera_points Camera coordinate points
     * @param intensities Point intensities
     * @param config Projection configuration
     */
    void applyFrustumCulling(std::vector<cv::Point3f>& camera_points,
                            std::vector<float>& intensities,
                            std::vector<size_t>& original_indices,
                            const ProjectionConfig& config);
    
    /**
     * @brief Project 3D camera points to 2D image coordinates
     * @param camera_points 3D points in camera coordinate system
     * @param intensities Point intensities
     * @param camera_data Camera parameters
     * @param config Projection configuration
     * @param pixel_points Output 2D pixel coordinates
     */
    void projectPointsToImage(const std::vector<cv::Point3f>& camera_points,
                             const std::vector<float>& intensities,
                             const std::vector<size_t>& original_indices,
                             const CameraData& camera_data,
                             const ProjectionConfig& config,
                             std::vector<PixelPoint>& pixel_points);
    
    /**
     * @brief Filter points within image boundaries
     * @param pixel_points Input/output pixel points
     * @param image_size Image dimensions
     * @param margin_pixels Boundary margin
     */
    void filterImageBounds(std::vector<PixelPoint>& pixel_points,
                          const cv::Size& image_size,
                          int margin_pixels);
    
    /**
     * @brief Create debug visualization image
     * @param projection Camera projection data
     * @param base_image Base camera image (optional)
     * @param config Projection configuration
     * @return Visualization image
     */
    cv::Mat createCameraVisualization(const CameraProjection& projection,
                                     const cv::Mat& base_image,
                                     const ProjectionConfig& config);
    
    /**
     * @brief Update camera data from calibration
     * @param camera_id Camera identifier
     * @param camera_data Camera data to update
     * @return True if updated successfully
     */
    bool updateCameraDataFromCalibration(const std::string& camera_id, 
                                        CameraData& camera_data);
    
    /**
     * @brief Validate camera transformation matrix
     * @param T_cam_lidar Transformation matrix
     * @return True if valid
     */
    bool validateTransformationMatrix(const Eigen::Matrix4d& T_cam_lidar) const;
    
    /**
     * @brief Convert Eigen transformation to OpenCV rotation/translation
     * @param T_cam_lidar Transformation matrix
     * @param rvec Output rotation vector
     * @param tvec Output translation vector
     */
    void transformationToRvecTvec(const Eigen::Matrix4d& T_cam_lidar,
                                 cv::Mat& rvec, cv::Mat& tvec);
    
    // Core components
    std::shared_ptr<prism::core::CalibrationManager> calibration_manager_;
    ProjectionConfig config_;
    
    // Camera data storage
    mutable std::shared_mutex cameras_mutex_;
    std::unordered_map<std::string, std::unique_ptr<CameraData>> cameras_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    ProjectionStats stats_;
    
    // State
    std::atomic<bool> is_initialized_{false};
    
    // Performance monitoring
    mutable std::mutex timing_mutex_;
    std::chrono::high_resolution_clock::time_point last_projection_time_;
};

/**
 * @brief Factory function to create projection engine with default calibration
 * @param config_directory Directory containing calibration files
 * @param config Projection configuration
 * @return Shared pointer to projection engine, nullptr if failed
 */
std::shared_ptr<ProjectionEngine> createProjectionEngine(
    const std::string& config_directory,
    const ProjectionConfig& config = ProjectionConfig()
);

/**
 * @brief Utility function to load camera parameters from YAML
 * @param intrinsic_yaml Path to intrinsic calibration YAML
 * @param extrinsic_yaml Path to extrinsic calibration YAML
 * @param camera_id Camera identifier
 * @return Camera parameters, invalid if loading failed
 */
CameraParams loadCameraParamsFromYAML(const std::string& intrinsic_yaml,
                                     const std::string& extrinsic_yaml,
                                     const std::string& camera_id);

/**
 * @brief Benchmark projection performance
 * @param engine Projection engine to benchmark
 * @param point_counts Vector of point cloud sizes to test
 * @param iterations Number of iterations per test
 * @return Map of point_count to average processing time (ms)
 */
std::map<size_t, double> benchmarkProjectionPerformance(
    ProjectionEngine& engine,
    const std::vector<size_t>& point_counts,
    size_t iterations = 10
);

} // namespace projection
} // namespace prism