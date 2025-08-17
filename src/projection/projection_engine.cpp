#include "prism/projection/projection_engine.hpp"
#include <algorithm>
#include <execution>
#include <chrono>
#include <cmath>
#include <iostream>

namespace prism {
namespace projection {

ProjectionEngine::ProjectionEngine(
    std::shared_ptr<prism::core::CalibrationManager> calibration_manager,
    const ProjectionConfig& config)
    : calibration_manager_(std::move(calibration_manager))
    , config_(config)
    , is_initialized_(false) {
    
    if (!calibration_manager_) {
        throw std::invalid_argument("Calibration manager cannot be null");
    }
    
    stats_.reset();
    last_projection_time_ = std::chrono::high_resolution_clock::now();
}

ProjectionEngine::~ProjectionEngine() = default;

bool ProjectionEngine::initialize(const std::vector<std::string>& camera_ids) {
    std::unique_lock<std::shared_mutex> lock(cameras_mutex_);
    
    cameras_.clear();
    bool all_success = true;
    
    for (const auto& camera_id : camera_ids) {
        if (!loadCameraFromCalibration(camera_id)) {
            std::cerr << "Failed to load camera: " << camera_id << std::endl;
            all_success = false;
            continue;
        }
    }
    
    is_initialized_ = all_success && !cameras_.empty();
    
    if (is_initialized_) {
        std::cout << "ProjectionEngine initialized with " << cameras_.size() << " cameras" << std::endl;
    }
    
    return is_initialized_;
}

bool ProjectionEngine::projectToAllCameras(const std::vector<LiDARPoint>& lidar_points, 
                                          ProjectionResult& result) {
    if (!is_initialized_) {
        return false;
    }
    
    // Handle empty point cloud as valid case
    if (lidar_points.empty()) {
        result.clear();
        result.input_count = 0;
        result.output_count = 0;
        result.timestamp = std::chrono::high_resolution_clock::now();
        return true;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    result.clear();
    // Apply sampling/capping for performance
    std::vector<LiDARPoint> sampled;
    sampled.reserve(lidar_points.size());
    const int stride = std::max(1, config_.sample_stride);
    for (size_t i = 0; i < lidar_points.size(); i += stride) {
        sampled.push_back(lidar_points[i]);
        if (config_.max_points_per_frame > 0 && sampled.size() >= config_.max_points_per_frame) break;
    }
    result.input_count = sampled.size();
    result.timestamp = start_time;
    
    // Get camera IDs for processing
    std::vector<std::string> camera_ids = getCameraIds();
    result.camera_projections.resize(camera_ids.size());
    
    // Process cameras in parallel if enabled
    if (config_.enable_parallel_processing && camera_ids.size() > 1) {
        std::for_each(std::execution::par_unseq, 
                     camera_ids.begin(), camera_ids.end(),
                     [this, &sampled, &result, &camera_ids](const std::string& camera_id) {
            auto it = std::find(camera_ids.begin(), camera_ids.end(), camera_id);
            size_t index = std::distance(camera_ids.begin(), it);
            
            CameraProjection& projection = result.camera_projections[index];
            this->projectToCamera(sampled, camera_id, projection);
        });
    } else {
        // Sequential processing
        for (size_t i = 0; i < camera_ids.size(); ++i) {
            CameraProjection& projection = result.camera_projections[i];
            if (!projectToCamera(sampled, camera_ids[i], projection)) {
                std::cerr << "Failed to project to camera: " << camera_ids[i] << std::endl;
            }
        }
    }
    
    // Compute final statistics
    result.output_count = 0;
    for (auto& projection : result.camera_projections) {
        projection.computeStatistics();
        result.output_count += projection.projected_point_count;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    
    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.updateStats(result);
    }
    
    return true;
}

bool ProjectionEngine::projectToCamera(const std::vector<LiDARPoint>& lidar_points,
                                      const std::string& camera_id,
                                      CameraProjection& projection) {
    projection.clear();
    projection.camera_id = camera_id;
    projection.original_point_count = lidar_points.size();
    
    // Get camera data
    std::shared_lock<std::shared_mutex> lock(cameras_mutex_);
    auto camera_it = cameras_.find(camera_id);
    if (camera_it == cameras_.end() || !camera_it->second->is_valid) {
        return false;
    }
    
    const auto& camera_data = *camera_it->second;
    lock.unlock();
    
    // Transform points to camera coordinate system
    std::vector<cv::Point3f> camera_points;
    std::vector<float> intensities;
    std::vector<size_t> original_indices;
    transformPointsToCamera(lidar_points, camera_data, camera_points, intensities, original_indices);
    
    if (camera_points.empty()) {
        return true; // No points to project, but not an error
    }
    
    // Apply frustum culling
    if (config_.enable_frustum_culling) {
        applyFrustumCulling(camera_points, intensities, original_indices, config_);
    }
    
    if (camera_points.empty()) {
        return true; // All points culled
    }
    
    // Project to image coordinates
    std::vector<PixelPoint> pixel_points;
    projectPointsToImage(camera_points, intensities, original_indices, camera_data, config_, pixel_points);
    
    // Filter points within image bounds
    filterImageBounds(pixel_points, camera_data.params.image_size, config_.margin_pixels);

    // original_index already propagated through transform/culling/projection
    
    projection.projected_points = std::move(pixel_points);
    return true;
}

void ProjectionEngine::updateConfig(const ProjectionConfig& config) {
    config_ = config;
}

ProjectionConfig ProjectionEngine::getConfig() const {
    return config_;
}

bool ProjectionEngine::addCamera(const std::string& camera_id, const CameraParams& params) {
    if (!params.validate()) {
        return false;
    }
    
    std::unique_lock<std::shared_mutex> lock(cameras_mutex_);
    
    auto camera_data = std::make_unique<CameraData>();
    camera_data->params = params;
    camera_data->is_valid = true;
    
    // Convert transformation to OpenCV format
    transformationToRvecTvec(params.T_cam_lidar, camera_data->rvec, camera_data->tvec);
    
    cameras_[camera_id] = std::move(camera_data);
    return true;
}

bool ProjectionEngine::removeCamera(const std::string& camera_id) {
    std::unique_lock<std::shared_mutex> lock(cameras_mutex_);
    return cameras_.erase(camera_id) > 0;
}

std::vector<std::string> ProjectionEngine::getCameraIds() const {
    std::shared_lock<std::shared_mutex> lock(cameras_mutex_);
    std::vector<std::string> ids;
    ids.reserve(cameras_.size());
    
    for (const auto& pair : cameras_) {
        if (pair.second->is_valid) {
            ids.push_back(pair.first);
        }
    }
    
    // Sort for consistent ordering
    std::sort(ids.begin(), ids.end());
    
    return ids;
}

bool ProjectionEngine::hasCamera(const std::string& camera_id) const {
    std::shared_lock<std::shared_mutex> lock(cameras_mutex_);
    auto it = cameras_.find(camera_id);
    return it != cameras_.end() && it->second->is_valid;
}

ProjectionStats ProjectionEngine::getStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void ProjectionEngine::resetStatistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.reset();
}

std::shared_ptr<const CameraParams> ProjectionEngine::getCameraParams(const std::string& camera_id) const {
    std::shared_lock<std::shared_mutex> lock(cameras_mutex_);
    auto it = cameras_.find(camera_id);
    if (it != cameras_.end() && it->second->is_valid) {
        return std::make_shared<const CameraParams>(it->second->params);
    }
    return nullptr;
}

size_t ProjectionEngine::reloadCalibration() {
    std::unique_lock<std::shared_mutex> lock(cameras_mutex_);
    
    size_t reloaded_count = 0;
    for (auto& pair : cameras_) {
        if (loadCameraFromCalibration(pair.first)) {
            reloaded_count++;
        }
    }
    
    return reloaded_count;
}

bool ProjectionEngine::loadCameraFromCalibration(const std::string& camera_id) {
    auto calibration = calibration_manager_->getCalibration(camera_id);
    if (!calibration) {
        std::cerr << "Calibration not found for " << camera_id << std::endl;
        return false;
    }
    if (!calibration->is_valid) {
        std::cerr << "Calibration not valid for " << camera_id << std::endl;
        return false;
    }
    
    auto camera_data = std::make_unique<CameraData>();
    
    // Set basic parameters
    camera_data->params.camera_id = camera_id;
    camera_data->params.camera_matrix = utils::eigenToCv(calibration->K);
    camera_data->params.distortion_coeffs = utils::eigenToCv(calibration->distortion);
    camera_data->params.image_size = cv::Size(calibration->width, calibration->height);
    
    // Set transformation (assuming T_ref_cam is LiDAR to camera)
    camera_data->params.T_cam_lidar = calibration->T_ref_cam;
    
    // Validate parameters
    if (!camera_data->params.validate()) {
        return false;
    }
    
    // Convert transformation to OpenCV format
    transformationToRvecTvec(camera_data->params.T_cam_lidar, 
                            camera_data->rvec, camera_data->tvec);
    
    camera_data->is_valid = true;
    cameras_[camera_id] = std::move(camera_data);
    
    return true;
}

void ProjectionEngine::transformPointsToCamera(const std::vector<LiDARPoint>& lidar_points,
                                              const CameraData& camera_data,
                                              std::vector<cv::Point3f>& camera_points,
                                              std::vector<float>& intensities,
                                              std::vector<size_t>& original_indices) {
    const auto& T_lidar_to_cam = camera_data.params.T_cam_lidar;
    
    // Transpose the matrix for row-vector multiplication (Hungarian style)
    // This is because calibration files are from hungarian_association package
    Eigen::Matrix4d T_transposed = T_lidar_to_cam.transpose();
    
    camera_points.reserve(lidar_points.size());
    intensities.reserve(lidar_points.size());
    original_indices.reserve(lidar_points.size());
    
    for (const auto& point : lidar_points) {
        // Apply transformation: P_cam = P_lidar * T.T (row vector multiplication)
        Eigen::RowVector4d lidar_point(point.x, point.y, point.z, 1.0);
        Eigen::RowVector4d camera_point = lidar_point * T_transposed;
        
        // Only add points in front of camera (z > 0)
        if (camera_point.z() > 1e-3) {  // 1mm minimum distance
            camera_points.emplace_back(
                static_cast<float>(camera_point.x()),
                static_cast<float>(camera_point.y()),
                static_cast<float>(camera_point.z())
            );
            intensities.push_back(point.intensity);
            original_indices.push_back(point.original_index);
        }
    }
}

void ProjectionEngine::applyFrustumCulling(std::vector<cv::Point3f>& camera_points,
                                          std::vector<float>& intensities,
                                          std::vector<size_t>& original_indices,
                                          const ProjectionConfig& config) {
    auto point_it = camera_points.begin();
    auto intensity_it = intensities.begin();
    auto index_it = original_indices.begin();
    
    while (point_it != camera_points.end()) {
        // Remove points behind camera (negative Z) or outside depth range
        if (point_it->z <= 0 || point_it->z < config.min_depth || point_it->z > config.max_depth) {
            point_it = camera_points.erase(point_it);
            intensity_it = intensities.erase(intensity_it);
            index_it = original_indices.erase(index_it);
        } else {
            ++point_it;
            ++intensity_it;
            ++index_it;
        }
    }
}

void ProjectionEngine::projectPointsToImage(const std::vector<cv::Point3f>& camera_points,
                                           const std::vector<float>& intensities,
                                           const std::vector<size_t>& original_indices,
                                           const CameraData& camera_data,
                                           const ProjectionConfig& config,
                                           std::vector<PixelPoint>& pixel_points) {
    if (camera_points.empty()) {
        return;
    }
    
    std::vector<cv::Point2f> image_points;
    
    if (config.enable_distortion_correction) {
        // Points are already in the camera frame. To avoid double-applying
        // extrinsics, project with zero rvec/tvec (CALICO behavior).
        cv::projectPoints(
            camera_points,
            cv::Vec3d(0.0, 0.0, 0.0),
            cv::Vec3d(0.0, 0.0, 0.0),
            camera_data.params.camera_matrix,
            camera_data.params.distortion_coeffs,
            image_points);
    } else {
        // Simple pinhole projection without distortion
        const cv::Mat& K = camera_data.params.camera_matrix;
        double fx = K.at<double>(0, 0);
        double fy = K.at<double>(1, 1);
        double cx = K.at<double>(0, 2);
        double cy = K.at<double>(1, 2);
        
        image_points.reserve(camera_points.size());
        for (const auto& point : camera_points) {
            if (point.z > 0) {
                float u = static_cast<float>(fx * point.x / point.z + cx);
                float v = static_cast<float>(fy * point.y / point.z + cy);
                image_points.emplace_back(u, v);
            }
        }
    }
    
    // Create pixel points with depth and intensity information
    pixel_points.reserve(image_points.size());
    for (size_t i = 0; i < image_points.size() && i < camera_points.size() && i < original_indices.size(); ++i) {
        pixel_points.emplace_back(
            image_points[i].x,
            image_points[i].y,
            camera_points[i].z,
            intensities[i],
            original_indices[i]
        );
    }
}

void ProjectionEngine::filterImageBounds(std::vector<PixelPoint>& pixel_points,
                                        const cv::Size& image_size,
                                        int margin_pixels) {
    pixel_points.erase(
        std::remove_if(pixel_points.begin(), pixel_points.end(),
            [&](const PixelPoint& point) {
                return point.u < margin_pixels || 
                       point.v < margin_pixels ||
                       point.u >= (image_size.width - margin_pixels) ||
                       point.v >= (image_size.height - margin_pixels);
            }),
        pixel_points.end()
    );
}

void ProjectionEngine::transformationToRvecTvec(const Eigen::Matrix4d& T_cam_lidar,
                                               cv::Mat& rvec, cv::Mat& tvec) {
    // Extract rotation matrix and translation vector
    Eigen::Matrix3d R = T_cam_lidar.block<3, 3>(0, 0);
    Eigen::Vector3d t = T_cam_lidar.block<3, 1>(0, 3);
    
    // Convert to OpenCV matrices
    cv::Mat R_cv(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R_cv.at<double>(i, j) = R(i, j);
        }
    }
    
    tvec = cv::Mat(3, 1, CV_64F);
    for (int i = 0; i < 3; ++i) {
        tvec.at<double>(i, 0) = t(i);
    }
    
    // Convert rotation matrix to rotation vector
    cv::Rodrigues(R_cv, rvec);
}

std::unordered_map<std::string, cv::Mat> ProjectionEngine::createDebugVisualization(
    const ProjectionResult& result,
    const std::unordered_map<std::string, cv::Mat>& images) {
    
    std::unordered_map<std::string, cv::Mat> visualizations;
    
    for (const auto& projection : result.camera_projections) {
        cv::Mat base_image;
        auto img_it = images.find(projection.camera_id);
        if (img_it != images.end()) {
            base_image = img_it->second.clone();
        } else {
            // Create default gray image
            auto camera_params = getCameraParams(projection.camera_id);
            if (camera_params) {
                base_image = cv::Mat::zeros(camera_params->image_size, CV_8UC3);
            } else {
                base_image = cv::Mat::zeros(480, 640, CV_8UC3);
            }
        }
        
        visualizations[projection.camera_id] = createCameraVisualization(
            projection, base_image, config_);
    }
    
    return visualizations;
}

cv::Mat ProjectionEngine::createCameraVisualization(const CameraProjection& projection,
                                                   const cv::Mat& base_image,
                                                   const ProjectionConfig& config) {
    cv::Mat vis_image = base_image.clone();
    
    if (vis_image.channels() == 1) {
        cv::cvtColor(vis_image, vis_image, cv::COLOR_GRAY2BGR);
    }
    
    // Draw projected points
    for (const auto& point : projection.projected_points) {
        cv::Scalar color = utils::getDepthColor(point.depth, 
                                               projection.min_depth, 
                                               projection.max_depth);
        
        cv::circle(vis_image, 
                  cv::Point(static_cast<int>(point.u), static_cast<int>(point.v)),
                  config.debug_point_radius, color, -1);
    }
    
    // Text overlay removed - handled by addStatusOverlay in projection_debug_node
    
    return vis_image;
}

// Utility function implementations
namespace utils {

cv::Mat createVisualizationImage(const CameraProjection& projection, 
                                const cv::Size& image_size,
                                const ProjectionConfig& config) {
    cv::Mat vis_image = cv::Mat::zeros(image_size, CV_8UC3);
    
    for (const auto& point : projection.projected_points) {
        cv::Scalar color = getDepthColor(point.depth, 
                                        projection.min_depth, 
                                        projection.max_depth);
        
        cv::circle(vis_image, 
                  cv::Point(static_cast<int>(point.u), static_cast<int>(point.v)),
                  config.debug_point_radius, color, -1);
    }
    
    return vis_image;
}

cv::Scalar getDepthColor(float depth, float min_depth, float max_depth) {
    if (max_depth <= min_depth) {
        return cv::Scalar(0, 255, 0); // Green for single depth
    }
    
    // Normalize depth to [0, 1]
    float normalized = (depth - min_depth) / (max_depth - min_depth);
    normalized = std::clamp(normalized, 0.0f, 1.0f);
    
    // Create color map (blue = close, red = far)
    int b = static_cast<int>(255 * (1.0f - normalized));
    int r = static_cast<int>(255 * normalized);
    int g = static_cast<int>(255 * std::sin(normalized * M_PI));
    
    return cv::Scalar(b, g, r);
}

} // namespace utils

// Factory functions
std::shared_ptr<ProjectionEngine> createProjectionEngine(
    const std::string& config_directory,
    const ProjectionConfig& config) {
    
    // Create calibration manager
    prism::core::CalibrationManager::Config cal_config;
    cal_config.calibration_directory = config_directory;
    
    auto calibration_manager = std::make_shared<prism::core::CalibrationManager>(cal_config);
    
    try {
        return std::make_shared<ProjectionEngine>(calibration_manager, config);
    } catch (const std::exception& e) {
        std::cerr << "Failed to create projection engine: " << e.what() << std::endl;
        return nullptr;
    }
}

CameraParams loadCameraParamsFromYAML(const std::string& intrinsic_yaml,
                                     const std::string& extrinsic_yaml,
                                     const std::string& camera_id) {
    CameraParams params;
    params.camera_id = camera_id;
    
    try {
        // Load intrinsic parameters
        YAML::Node intrinsic_node = YAML::LoadFile(intrinsic_yaml);
        if (!intrinsic_node[camera_id]) {
            return params; // Invalid
        }
        
        auto cam_node = intrinsic_node[camera_id];
        
        // Parse camera matrix
        if (cam_node["camera_matrix"]) {
            auto K_data = cam_node["camera_matrix"]["data"].as<std::vector<std::vector<double>>>();
            params.camera_matrix = cv::Mat(3, 3, CV_64F);
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    params.camera_matrix.at<double>(i, j) = K_data[i][j];
                }
            }
        }
        
        // Parse distortion coefficients
        if (cam_node["distortion_coefficients"]) {
            auto dist_data = cam_node["distortion_coefficients"]["data"].as<std::vector<double>>();
            params.distortion_coeffs = cv::Mat(dist_data.size(), 1, CV_64F);
            for (size_t i = 0; i < dist_data.size(); ++i) {
                params.distortion_coeffs.at<double>(i, 0) = dist_data[i];
            }
        }
        
        // Parse image size
        if (cam_node["image_size"]) {
            params.image_size.width = cam_node["image_size"]["width"].as<int>();
            params.image_size.height = cam_node["image_size"]["height"].as<int>();
        }
        
        // Load extrinsic parameters
        YAML::Node extrinsic_node = YAML::LoadFile(extrinsic_yaml);
        if (extrinsic_node[camera_id] && extrinsic_node[camera_id]["extrinsic_matrix"]) {
            auto T_data = extrinsic_node[camera_id]["extrinsic_matrix"].as<std::vector<std::vector<double>>>();
            
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    params.T_cam_lidar(i, j) = T_data[i][j];
                }
            }
        }
        
        params.is_valid = params.validate();
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading camera parameters: " << e.what() << std::endl;
        params.is_valid = false;
    }
    
    return params;
}

std::map<size_t, double> benchmarkProjectionPerformance(
    ProjectionEngine& engine,
    const std::vector<size_t>& point_counts,
    size_t iterations) {
    
    std::map<size_t, double> results;
    
    for (size_t point_count : point_counts) {
        // Generate test points
        std::vector<LiDARPoint> test_points;
        test_points.reserve(point_count);
        
        for (size_t i = 0; i < point_count; ++i) {
            float x = static_cast<float>(rand()) / RAND_MAX * 20.0f - 10.0f;
            float y = static_cast<float>(rand()) / RAND_MAX * 20.0f - 10.0f;
            float z = static_cast<float>(rand()) / RAND_MAX * 10.0f + 1.0f;
            float intensity = static_cast<float>(rand()) / RAND_MAX * 255.0f;
            
            test_points.emplace_back(x, y, z, intensity);
        }
        
        // Run benchmark
        double total_time = 0.0;
        for (size_t iter = 0; iter < iterations; ++iter) {
            ProjectionResult result;
            auto start = std::chrono::high_resolution_clock::now();
            
            engine.projectToAllCameras(test_points, result);
            
            auto end = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double, std::milli>(end - start).count();
        }
        
        results[point_count] = total_time / iterations;
    }
    
    return results;
}

} // namespace projection
} // namespace prism