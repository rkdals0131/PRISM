# PRISM Projection and Color Mapping Guide

## Overview

This document provides comprehensive technical documentation for camera-LiDAR coordinate transformation and color mapping in the PRISM system. The implementation builds upon the proven CALICO projection utilities, extending them for multi-camera fusion, advanced distortion correction, and high-performance color extraction with sub-pixel accuracy.

## Analysis of CALICO Projection System

### Current CALICO Implementation

The CALICO `ProjectionUtils` class provides a solid foundation for 3D-to-2D projection:

```cpp
// Key components from projection_utils.cpp
namespace calico {
namespace utils {

class ProjectionUtils {
public:
    // Core projection function
    static std::pair<std::vector<cv::Point2f>, std::vector<int>> 
    projectLidarToCamera(const std::vector<Point3D>& lidar_points,
                        const cv::Mat& camera_matrix,
                        const cv::Mat& dist_coeffs,
                        const Eigen::Matrix4d& extrinsic_matrix);
    
    // Coordinate transformation
    static std::vector<Point3D> transformPoints(
        const std::vector<Point3D>& points,
        const Eigen::Matrix4d& transform_matrix);
    
    // Image boundary validation
    static bool isPointInImage(const cv::Point2f& point, 
                              int image_width, int image_height);
};

} // namespace utils
} // namespace calico
```

**CALICO Calibration Format Analysis:**
```yaml
# Multi-camera extrinsic calibration
camera_1:
  extrinsic_matrix:
  - [-0.4976, 0.8674, -0.0046, -0.0013]  # R11, R12, R13, tx
  - [0.0716, 0.0358, -0.9968, -0.0831]   # R21, R22, R23, ty  
  - [-0.8644, -0.4964, -0.0799, -0.0813] # R31, R32, R33, tz
  - [0.0, 0.0, 0.0, 1.0]                 # Homogeneous row

# Multi-camera intrinsic calibration  
camera_1:
  camera_matrix:
    data:
    - [490.33, 0.0, 323.01]     # fx, 0, cx
    - [0.0, 489.74, 177.01]     # 0, fy, cy
    - [0.0, 0.0, 1.0]           # 0, 0, 1
  distortion_coefficients:
    data: [0.0595, -0.1772, -0.0033, 0.0024, 0.1209] # k1,k2,p1,p2,k3
  image_size:
    height: 360
    width: 640
```

**Strengths of CALICO Implementation:**
- Robust 3D-to-2D projection pipeline
- Proper handling of points behind camera
- Efficient coordinate transformations using Eigen
- Integration with OpenCV's distortion models
- Batch processing of point arrays

**Areas for PRISM Enhancement:**
- Multi-camera fusion strategies
- Sub-pixel color interpolation
- Occlusion handling and depth buffering  
- Real-time optimization with SIMD
- Advanced distortion correction models

## Camera-LiDAR Coordinate Transformation

### 1. Enhanced Transformation Pipeline

```cpp
// include/prism/projection/CoordinateTransformer.hpp
#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <immintrin.h>

namespace prism {
namespace projection {

class CoordinateTransformer {
public:
    struct TransformationResult {
        std::vector<Eigen::Vector3f> camera_points;
        std::vector<size_t> valid_indices;
        std::vector<float> depths;
        TransformationMetrics metrics;
    };
    
    struct CameraExtrinsics {
        Eigen::Matrix4f transform_matrix;  // LiDAR to camera transformation
        Eigen::Vector3f translation;
        Eigen::Matrix3f rotation;
        bool is_valid;
        
        // Factory method from CALICO YAML format
        static CameraExtrinsics fromYAMLMatrix(const std::vector<std::vector<float>>& matrix_data) {
            CameraExtrinsics extrinsics;
            
            // Convert from double precision YAML to single precision
            extrinsics.transform_matrix = Eigen::Matrix4f::Zero();
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    extrinsics.transform_matrix(i, j) = static_cast<float>(matrix_data[i][j]);
                }
            }
            
            // Extract rotation and translation components
            extrinsics.rotation = extrinsics.transform_matrix.block<3,3>(0,0);
            extrinsics.translation = extrinsics.transform_matrix.block<3,1>(0,3);
            
            // Validate transformation matrix
            extrinsics.is_valid = validateTransformation(extrinsics.transform_matrix);
            
            return extrinsics;
        }
    };
    
    // SIMD-optimized batch transformation
    TransformationResult transformPointsBatch(
        const pcl::PointCloud<pcl::PointXYZI>& lidar_points,
        const CameraExtrinsics& extrinsics,
        float min_depth = 0.1f,
        float max_depth = 100.0f);
        
private:
    // SIMD transformation for 4 points simultaneously  
    void transformFourPointsSIMD(const float* lidar_x, const float* lidar_y, const float* lidar_z,
                                const Eigen::Matrix4f& transform,
                                float* cam_x, float* cam_y, float* cam_z);
    
    static bool validateTransformation(const Eigen::Matrix4f& transform);
};

// SIMD implementation for batch transformation
CoordinateTransformer::TransformationResult 
CoordinateTransformer::transformPointsBatch(
    const pcl::PointCloud<pcl::PointXYZI>& lidar_points,
    const CameraExtrinsics& extrinsics,
    float min_depth, float max_depth) {
    
    TransformationResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const size_t point_count = lidar_points.size();
    result.camera_points.reserve(point_count);
    result.valid_indices.reserve(point_count);
    result.depths.reserve(point_count);
    
    // Process points in batches of 4 for SIMD efficiency
    const size_t batch_size = 4;
    const size_t full_batches = point_count / batch_size;
    const size_t remainder = point_count % batch_size;
    
    // Aligned temporary arrays for SIMD processing
    alignas(16) float lidar_x[4], lidar_y[4], lidar_z[4];
    alignas(16) float cam_x[4], cam_y[4], cam_z[4];
    
    // Process full batches with SIMD
    for (size_t batch = 0; batch < full_batches; ++batch) {
        const size_t base_idx = batch * batch_size;
        
        // Load 4 points
        for (int i = 0; i < 4; ++i) {
            const auto& pt = lidar_points.points[base_idx + i];
            lidar_x[i] = pt.x;
            lidar_y[i] = pt.y;
            lidar_z[i] = pt.z;
        }
        
        // Transform 4 points simultaneously
        transformFourPointsSIMD(lidar_x, lidar_y, lidar_z, 
                              extrinsics.transform_matrix,
                              cam_x, cam_y, cam_z);
        
        // Store valid results
        for (int i = 0; i < 4; ++i) {
            const float depth = cam_z[i];
            if (depth >= min_depth && depth <= max_depth && std::isfinite(depth)) {
                result.camera_points.emplace_back(cam_x[i], cam_y[i], cam_z[i]);
                result.valid_indices.push_back(base_idx + i);
                result.depths.push_back(depth);
            }
        }
    }
    
    // Process remaining points with scalar operations
    for (size_t i = full_batches * batch_size; i < point_count; ++i) {
        const auto& pt = lidar_points.points[i];
        Eigen::Vector4f lidar_point(pt.x, pt.y, pt.z, 1.0f);
        Eigen::Vector4f camera_point = extrinsics.transform_matrix * lidar_point;
        
        const float depth = camera_point.z();
        if (depth >= min_depth && depth <= max_depth && std::isfinite(depth)) {
            result.camera_points.emplace_back(camera_point.x(), camera_point.y(), camera_point.z());
            result.valid_indices.push_back(i);
            result.depths.push_back(depth);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.metrics.transformation_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.metrics.input_points = point_count;
    result.metrics.valid_points = result.camera_points.size();
    
    return result;
}

} // namespace projection
} // namespace prism
```

### 2. Advanced Calibration Management

```cpp
// include/prism/projection/CalibrationManager.hpp
namespace prism {
namespace projection {

class CalibrationManager {
public:
    struct CameraIntrinsics {
        cv::Mat camera_matrix;        // 3x3 intrinsic matrix
        cv::Mat distortion_coeffs;    // Distortion coefficients [k1,k2,p1,p2,k3]
        cv::Size image_size;          // Image dimensions
        float fx, fy, cx, cy;         // Individual parameters for fast access
        std::string distortion_model; // "radtan", "fisheye", "equidistant"
        
        // Factory method from CALICO YAML format
        static CameraIntrinsics fromYAML(const YAML::Node& calib_node) {
            CameraIntrinsics intrinsics;
            
            // Parse camera matrix
            const auto& cam_matrix_data = calib_node["camera_matrix"]["data"];
            intrinsics.camera_matrix = cv::Mat::eye(3, 3, CV_64F);
            
            intrinsics.fx = cam_matrix_data[0][0].as<float>();
            intrinsics.fy = cam_matrix_data[1][1].as<float>();
            intrinsics.cx = cam_matrix_data[0][2].as<float>();
            intrinsics.cy = cam_matrix_data[1][2].as<float>();
            
            intrinsics.camera_matrix.at<double>(0,0) = intrinsics.fx;
            intrinsics.camera_matrix.at<double>(1,1) = intrinsics.fy;
            intrinsics.camera_matrix.at<double>(0,2) = intrinsics.cx;
            intrinsics.camera_matrix.at<double>(1,2) = intrinsics.cy;
            
            // Parse distortion coefficients
            const auto& dist_data = calib_node["distortion_coefficients"]["data"];
            intrinsics.distortion_coeffs = cv::Mat::zeros(1, 5, CV_64F);
            for (int i = 0; i < 5 && i < dist_data.size(); ++i) {
                intrinsics.distortion_coeffs.at<double>(0, i) = dist_data[i].as<double>();
            }
            
            // Parse image size
            intrinsics.image_size.width = calib_node["image_size"]["width"].as<int>();
            intrinsics.image_size.height = calib_node["image_size"]["height"].as<int>();
            
            intrinsics.distortion_model = "radtan"; // Default OpenCV model
            
            return intrinsics;
        }
        
        // Validate calibration parameters
        bool isValid() const {
            return fx > 0 && fy > 0 && 
                   cx >= 0 && cx < image_size.width &&
                   cy >= 0 && cy < image_size.height &&
                   !camera_matrix.empty() && !distortion_coeffs.empty();
        }
    };
    
    struct MultiCameraSetup {
        std::vector<CameraIntrinsics> intrinsics;
        std::vector<CoordinateTransformer::CameraExtrinsics> extrinsics;
        std::vector<std::string> camera_names;
        size_t camera_count;
        
        void loadFromYAML(const std::string& intrinsic_file, 
                         const std::string& extrinsic_file) {
            YAML::Node intrinsic_config = YAML::LoadFile(intrinsic_file);
            YAML::Node extrinsic_config = YAML::LoadFile(extrinsic_file);
            
            // Count cameras
            camera_count = 0;
            for (const auto& node : intrinsic_config) {
                if (node.first.as<std::string>().find("camera_") == 0) {
                    camera_count++;
                }
            }
            
            intrinsics.reserve(camera_count);
            extrinsics.reserve(camera_count);
            camera_names.reserve(camera_count);
            
            // Load each camera's calibration
            for (size_t i = 1; i <= camera_count; ++i) {
                std::string camera_name = "camera_" + std::to_string(i);
                
                // Load intrinsics
                if (intrinsic_config[camera_name]) {
                    intrinsics.push_back(
                        CameraIntrinsics::fromYAML(intrinsic_config[camera_name]));
                }
                
                // Load extrinsics
                if (extrinsic_config[camera_name]) {
                    const auto& matrix_data = extrinsic_config[camera_name]["extrinsic_matrix"];
                    std::vector<std::vector<float>> matrix_vec;
                    
                    for (const auto& row : matrix_data) {
                        std::vector<float> row_vec;
                        for (const auto& val : row) {
                            row_vec.push_back(val.as<float>());
                        }
                        matrix_vec.push_back(row_vec);
                    }
                    
                    extrinsics.push_back(
                        CoordinateTransformer::CameraExtrinsics::fromYAMLMatrix(matrix_vec));
                }
                
                camera_names.push_back(camera_name);
            }
        }
        
        bool isValid() const {
            if (intrinsics.size() != camera_count || 
                extrinsics.size() != camera_count) return false;
                
            for (size_t i = 0; i < camera_count; ++i) {
                if (!intrinsics[i].isValid() || !extrinsics[i].is_valid) {
                    return false;
                }
            }
            return true;
        }
    };
    
private:
    MultiCameraSetup camera_setup_;
    std::mutex calibration_mutex_;
};

} // namespace projection  
} // namespace prism
```

## Projection Algorithm with Distortion Correction

### 1. Enhanced Projection Engine

```cpp
// include/prism/projection/ProjectionEngine.hpp
namespace prism {
namespace projection {

class ProjectionEngine {
public:
    struct ProjectionResult {
        std::vector<cv::Point2f> pixel_coordinates;
        std::vector<size_t> original_indices;
        std::vector<float> depths;
        std::vector<bool> in_image_bounds;
        ProjectionMetrics metrics;
    };
    
    struct ProjectionConfig {
        bool enable_distortion_correction = true;
        bool enable_subpixel_accuracy = true;
        bool filter_out_of_bounds = true;
        float depth_near_clip = 0.1f;
        float depth_far_clip = 100.0f;
        int num_threads = -1; // -1 for auto-detect
    };
    
    // High-performance batch projection with distortion correction
    ProjectionResult projectPointsBatch(
        const std::vector<Eigen::Vector3f>& camera_points,
        const std::vector<size_t>& point_indices,
        const CalibrationManager::CameraIntrinsics& intrinsics,
        const ProjectionConfig& config = ProjectionConfig{});
        
private:
    // SIMD-optimized projection without distortion
    void projectPointsSIMDNoDist(const float* cam_x, const float* cam_y, const float* cam_z,
                                const CalibrationManager::CameraIntrinsics& intrinsics,
                                float* pixel_x, float* pixel_y, int count);
    
    // High-precision distortion correction
    cv::Point2f applyDistortionCorrection(const cv::Point2f& normalized_point,
                                        const CalibrationManager::CameraIntrinsics& intrinsics);
    
    // Advanced distortion models
    cv::Point2f correctRadialTangential(const cv::Point2f& point, 
                                       const cv::Mat& dist_coeffs);
    cv::Point2f correctFisheye(const cv::Point2f& point, 
                              const cv::Mat& dist_coeffs);
};

// Implementation with SIMD optimization
ProjectionEngine::ProjectionResult 
ProjectionEngine::projectPointsBatch(
    const std::vector<Eigen::Vector3f>& camera_points,
    const std::vector<size_t>& point_indices,
    const CalibrationManager::CameraIntrinsics& intrinsics,
    const ProjectionConfig& config) {
    
    ProjectionResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    const size_t point_count = camera_points.size();
    result.pixel_coordinates.reserve(point_count);
    result.original_indices.reserve(point_count);
    result.depths.reserve(point_count);
    result.in_image_bounds.reserve(point_count);
    
    if (!config.enable_distortion_correction) {
        // Fast path: no distortion correction, use SIMD
        processWithoutDistortionSIMD(camera_points, point_indices, intrinsics, result);
    } else {
        // Precise path: with distortion correction
        processWithDistortionCorrection(camera_points, point_indices, intrinsics, result);
    }
    
    // Filter out-of-bounds points if requested
    if (config.filter_out_of_bounds) {
        filterInBounds(result, intrinsics.image_size);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.metrics.projection_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.metrics.input_points = point_count;
    result.metrics.projected_points = result.pixel_coordinates.size();
    
    return result;
}

void ProjectionEngine::processWithDistortionCorrection(
    const std::vector<Eigen::Vector3f>& camera_points,
    const std::vector<size_t>& point_indices,
    const CalibrationManager::CameraIntrinsics& intrinsics,
    ProjectionResult& result) {
    
    const float fx = intrinsics.fx;
    const float fy = intrinsics.fy; 
    const float cx = intrinsics.cx;
    const float cy = intrinsics.cy;
    
    #pragma omp parallel for
    for (size_t i = 0; i < camera_points.size(); ++i) {
        const auto& cp = camera_points[i];
        
        // Check depth bounds
        if (cp.z() <= 0.0f) continue;
        
        // Normalize to camera plane
        cv::Point2f normalized(cp.x() / cp.z(), cp.y() / cp.z());
        
        // Apply distortion correction
        cv::Point2f corrected = applyDistortionCorrection(normalized, intrinsics);
        
        // Project to pixel coordinates
        cv::Point2f pixel(
            corrected.x * fx + cx,
            corrected.y * fy + cy
        );
        
        // Thread-safe result storage
        #pragma omp critical
        {
            result.pixel_coordinates.push_back(pixel);
            result.original_indices.push_back(point_indices[i]);
            result.depths.push_back(cp.z());
            result.in_image_bounds.push_back(
                pixel.x >= 0 && pixel.x < intrinsics.image_size.width &&
                pixel.y >= 0 && pixel.y < intrinsics.image_size.height);
        }
    }
}

} // namespace projection
} // namespace prism
```

### 2. Advanced Distortion Models

```cpp
// Enhanced distortion correction supporting multiple models
cv::Point2f ProjectionEngine::applyDistortionCorrection(
    const cv::Point2f& normalized_point,
    const CalibrationManager::CameraIntrinsics& intrinsics) {
    
    if (intrinsics.distortion_model == "radtan") {
        return correctRadialTangential(normalized_point, intrinsics.distortion_coeffs);
    } else if (intrinsics.distortion_model == "fisheye") {
        return correctFisheye(normalized_point, intrinsics.distortion_coeffs);
    } else {
        // Default: no distortion
        return normalized_point;
    }
}

cv::Point2f ProjectionEngine::correctRadialTangential(
    const cv::Point2f& point, const cv::Mat& dist_coeffs) {
    
    const double k1 = dist_coeffs.at<double>(0, 0);
    const double k2 = dist_coeffs.at<double>(0, 1);
    const double p1 = dist_coeffs.at<double>(0, 2);
    const double p2 = dist_coeffs.at<double>(0, 3);
    const double k3 = (dist_coeffs.cols > 4) ? dist_coeffs.at<double>(0, 4) : 0.0;
    
    const double x = point.x;
    const double y = point.y;
    const double r2 = x*x + y*y;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    
    // Radial distortion factor
    const double radial_factor = 1.0 + k1*r2 + k2*r4 + k3*r6;
    
    // Tangential distortion
    const double dx_tangential = 2.0*p1*x*y + p2*(r2 + 2.0*x*x);
    const double dy_tangential = p1*(r2 + 2.0*y*y) + 2.0*p2*x*y;
    
    // Apply corrections
    const double x_corrected = x * radial_factor + dx_tangential;
    const double y_corrected = y * radial_factor + dy_tangential;
    
    return cv::Point2f(static_cast<float>(x_corrected), static_cast<float>(y_corrected));
}

// Fisheye distortion correction for wide-angle cameras  
cv::Point2f ProjectionEngine::correctFisheye(
    const cv::Point2f& point, const cv::Mat& dist_coeffs) {
    
    const double k1 = dist_coeffs.at<double>(0, 0);
    const double k2 = dist_coeffs.at<double>(0, 1);
    const double k3 = dist_coeffs.at<double>(0, 2);  
    const double k4 = dist_coeffs.at<double>(0, 3);
    
    const double x = point.x;
    const double y = point.y;
    const double r = std::sqrt(x*x + y*y);
    
    if (r < 1e-8) return point; // Avoid division by zero
    
    const double theta = std::atan(r);
    const double theta2 = theta * theta;
    const double theta4 = theta2 * theta2;
    const double theta6 = theta4 * theta2;
    const double theta8 = theta4 * theta4;
    
    const double theta_d = theta * (1.0 + k1*theta2 + k2*theta4 + k3*theta6 + k4*theta8);
    const double scale = theta_d / r;
    
    return cv::Point2f(static_cast<float>(x * scale), static_cast<float>(y * scale));
}
```

## RGB Extraction with Sub-pixel Accuracy

### 1. Color Extraction Engine

```cpp
// include/prism/projection/ColorExtractor.hpp
namespace prism {
namespace projection {

class ColorExtractor {
public:
    enum class InterpolationMethod {
        NEAREST_NEIGHBOR,
        BILINEAR,
        BICUBIC,
        LANCZOS
    };
    
    struct ColorExtractionResult {
        std::vector<cv::Vec3b> colors;           // RGB colors  
        std::vector<float> confidence_scores;    // Color reliability [0,1]
        std::vector<bool> valid_extractions;     // Successful extractions
        ColorExtractionMetrics metrics;
    };
    
    struct ExtractionConfig {
        InterpolationMethod interpolation = InterpolationMethod::BILINEAR;
        bool enable_subpixel = true;
        bool validate_pixel_bounds = true;
        float confidence_threshold = 0.7f;
        int blur_kernel_size = 0;  // 0 = no blur, >0 = Gaussian blur
    };
    
    // Extract colors from single image
    ColorExtractionResult extractColors(
        const cv::Mat& image,
        const std::vector<cv::Point2f>& pixel_coordinates,
        const std::vector<size_t>& point_indices,
        const ExtractionConfig& config = ExtractionConfig{});
    
private:
    // Sub-pixel interpolation methods
    cv::Vec3b interpolateNearest(const cv::Mat& image, const cv::Point2f& pixel);
    cv::Vec3b interpolateBilinear(const cv::Mat& image, const cv::Point2f& pixel);
    cv::Vec3b interpolateBicubic(const cv::Mat& image, const cv::Point2f& pixel);
    cv::Vec3b interpolateLanczos(const cv::Mat& image, const cv::Point2f& pixel);
    
    // Quality assessment for extracted colors
    float assessColorConfidence(const cv::Mat& image, const cv::Point2f& pixel,
                               const cv::Vec3b& extracted_color);
};

// High-precision bilinear interpolation implementation
cv::Vec3b ColorExtractor::interpolateBilinear(const cv::Mat& image, const cv::Point2f& pixel) {
    // Get integer coordinates and fractional parts
    const int x0 = static_cast<int>(std::floor(pixel.x));
    const int y0 = static_cast<int>(std::floor(pixel.y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    
    const float fx = pixel.x - x0;  // Fractional x
    const float fy = pixel.y - y0;  // Fractional y
    
    // Boundary checks
    if (x0 < 0 || y0 < 0 || x1 >= image.cols || y1 >= image.rows) {
        // Return nearest valid pixel for out-of-bounds
        const int safe_x = std::max(0, std::min(image.cols - 1, 
                                               static_cast<int>(std::round(pixel.x))));
        const int safe_y = std::max(0, std::min(image.rows - 1, 
                                               static_cast<int>(std::round(pixel.y))));
        return image.at<cv::Vec3b>(safe_y, safe_x);
    }
    
    // Get the four surrounding pixels
    const cv::Vec3b& p00 = image.at<cv::Vec3b>(y0, x0);  // Top-left
    const cv::Vec3b& p10 = image.at<cv::Vec3b>(y0, x1);  // Top-right  
    const cv::Vec3b& p01 = image.at<cv::Vec3b>(y1, x0);  // Bottom-left
    const cv::Vec3b& p11 = image.at<cv::Vec3b>(y1, x1);  // Bottom-right
    
    // Bilinear interpolation for each channel
    cv::Vec3b result;
    for (int c = 0; c < 3; ++c) {
        const float top = p00[c] * (1.0f - fx) + p10[c] * fx;
        const float bottom = p01[c] * (1.0f - fx) + p11[c] * fx;
        const float interpolated = top * (1.0f - fy) + bottom * fy;
        
        result[c] = static_cast<uchar>(std::round(std::clamp(interpolated, 0.0f, 255.0f)));
    }
    
    return result;
}

// Advanced bicubic interpolation for highest quality
cv::Vec3b ColorExtractor::interpolateBicubic(const cv::Mat& image, const cv::Point2f& pixel) {
    const int x = static_cast<int>(std::floor(pixel.x));
    const int y = static_cast<int>(std::floor(pixel.y));
    const float fx = pixel.x - x;
    const float fy = pixel.y - y;
    
    // Bicubic kernel function
    auto cubic = [](float t) -> float {
        const float abs_t = std::abs(t);
        if (abs_t <= 1.0f) {
            return 1.5f * abs_t*abs_t*abs_t - 2.5f * abs_t*abs_t + 1.0f;
        } else if (abs_t <= 2.0f) {
            return -0.5f * abs_t*abs_t*abs_t + 2.5f * abs_t*abs_t - 4.0f * abs_t + 2.0f;
        } else {
            return 0.0f;
        }
    };
    
    cv::Vec3f result(0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;
    
    // 4x4 neighborhood for bicubic interpolation
    for (int dy = -1; dy <= 2; ++dy) {
        for (int dx = -1; dx <= 2; ++dx) {
            const int sample_x = x + dx;
            const int sample_y = y + dy;
            
            // Boundary handling - clamp to image edges
            const int safe_x = std::max(0, std::min(image.cols - 1, sample_x));
            const int safe_y = std::max(0, std::min(image.rows - 1, sample_y));
            
            const float weight = cubic(fx - dx) * cubic(fy - dy);
            const cv::Vec3b& pixel_color = image.at<cv::Vec3b>(safe_y, safe_x);
            
            result[0] += weight * pixel_color[0];
            result[1] += weight * pixel_color[1]; 
            result[2] += weight * pixel_color[2];
            weight_sum += weight;
        }
    }
    
    // Normalize and clamp
    if (weight_sum > 1e-6f) {
        result /= weight_sum;
    }
    
    return cv::Vec3b(
        static_cast<uchar>(std::clamp(result[0], 0.0f, 255.0f)),
        static_cast<uchar>(std::clamp(result[1], 0.0f, 255.0f)),
        static_cast<uchar>(std::clamp(result[2], 0.0f, 255.0f))
    );
}

} // namespace projection
} // namespace prism
```

## Multi-Camera Fusion Strategy

### 1. Camera Fusion Engine

```cpp
// include/prism/projection/MultiCameraFusion.hpp
namespace prism {
namespace projection {

class MultiCameraFusion {
public:
    enum class FusionStrategy {
        NEAREST_CAMERA,           // Use closest camera
        WEIGHTED_DISTANCE,        // Weight by distance to camera
        CONFIDENCE_BASED,         // Weight by color confidence  
        ANGLE_WEIGHTED,           // Weight by viewing angle
        ADAPTIVE_FUSION           // Combine multiple strategies
    };
    
    struct FusionResult {
        std::vector<cv::Vec3b> fused_colors;
        std::vector<float> fusion_confidence;
        std::vector<int> primary_camera_id;
        std::vector<std::vector<int>> contributing_cameras;
        FusionMetrics metrics;
    };
    
    struct FusionConfig {
        FusionStrategy strategy = FusionStrategy::ADAPTIVE_FUSION;
        float distance_weight_alpha = 2.0f;     // Distance weighting exponent
        float angle_weight_threshold = 60.0f;   // Max viewing angle (degrees)
        float confidence_threshold = 0.5f;      // Min confidence for inclusion
        bool enable_outlier_rejection = true;
        int max_cameras_per_point = 3;          // Limit cameras for efficiency
    };
    
    // Fuse colors from multiple camera views
    FusionResult fuseMultiCameraColors(
        const std::vector<Eigen::Vector3f>& lidar_points,
        const std::vector<size_t>& point_indices,
        const std::vector<cv::Mat>& camera_images,
        const CalibrationManager::MultiCameraSetup& camera_setup,
        const FusionConfig& config = FusionConfig{});
        
private:
    // Fusion strategy implementations
    cv::Vec3b fuseNearest(const std::vector<std::pair<cv::Vec3b, float>>& camera_colors,
                         const std::vector<float>& distances);
    
    cv::Vec3b fuseWeightedDistance(const std::vector<std::pair<cv::Vec3b, float>>& camera_colors,
                                  const std::vector<float>& distances,
                                  float alpha);
    
    cv::Vec3b fuseConfidenceBased(const std::vector<std::pair<cv::Vec3b, float>>& camera_colors,
                                 const std::vector<float>& confidences);
    
    cv::Vec3b fuseAdaptive(const std::vector<std::pair<cv::Vec3b, float>>& camera_colors,
                          const std::vector<float>& distances,
                          const std::vector<float>& confidences,
                          const std::vector<float>& viewing_angles);
    
    // Quality assessment
    float computeViewingAngle(const Eigen::Vector3f& point,
                             const CoordinateTransformer::CameraExtrinsics& extrinsics);
};

// Adaptive fusion implementation
cv::Vec3b MultiCameraFusion::fuseAdaptive(
    const std::vector<std::pair<cv::Vec3b, float>>& camera_colors,
    const std::vector<float>& distances,
    const std::vector<float>& confidences, 
    const std::vector<float>& viewing_angles) {
    
    if (camera_colors.empty()) return cv::Vec3b(0, 0, 0);
    
    std::vector<float> weights(camera_colors.size());
    float total_weight = 0.0f;
    
    for (size_t i = 0; i < camera_colors.size(); ++i) {
        // Distance factor (closer is better)
        const float distance_factor = 1.0f / (1.0f + distances[i] * distances[i]);
        
        // Confidence factor
        const float confidence_factor = confidences[i];
        
        // Viewing angle factor (perpendicular is better)
        const float angle_rad = viewing_angles[i] * M_PI / 180.0f;
        const float angle_factor = std::cos(angle_rad);
        
        // Combined weight
        weights[i] = distance_factor * confidence_factor * angle_factor;
        total_weight += weights[i];
    }
    
    if (total_weight < 1e-6f) {
        // Fallback to simple average
        cv::Vec3f average(0.0f, 0.0f, 0.0f);
        for (const auto& [color, conf] : camera_colors) {
            average[0] += color[0];
            average[1] += color[1];
            average[2] += color[2];
        }
        average /= static_cast<float>(camera_colors.size());
        return cv::Vec3b(static_cast<uchar>(average[0]),
                        static_cast<uchar>(average[1]),
                        static_cast<uchar>(average[2]));
    }
    
    // Weighted fusion
    cv::Vec3f fused_color(0.0f, 0.0f, 0.0f);
    for (size_t i = 0; i < camera_colors.size(); ++i) {
        const float normalized_weight = weights[i] / total_weight;
        const cv::Vec3b& color = camera_colors[i].first;
        
        fused_color[0] += normalized_weight * color[0];
        fused_color[1] += normalized_weight * color[1];
        fused_color[2] += normalized_weight * color[2];
    }
    
    return cv::Vec3b(
        static_cast<uchar>(std::clamp(fused_color[0], 0.0f, 255.0f)),
        static_cast<uchar>(std::clamp(fused_color[1], 0.0f, 255.0f)),
        static_cast<uchar>(std::clamp(fused_color[2], 0.0f, 255.0f))
    );
}

} // namespace projection
} // namespace prism
```

## Occlusion Handling and Depth Buffering

### 1. Z-Buffer Implementation

```cpp
// include/prism/projection/DepthBuffer.hpp
namespace prism {
namespace projection {

class DepthBuffer {
public:
    struct OcclusionResult {
        std::vector<bool> is_visible;          // Point visibility flags
        std::vector<size_t> occluded_indices;  // Indices of occluded points
        std::vector<size_t> visible_indices;   // Indices of visible points  
        OcclusionMetrics metrics;
    };
    
    struct OcclusionConfig {
        float depth_tolerance = 0.05f;         // 5cm depth tolerance
        bool enable_depth_smoothing = true;
        int smoothing_kernel_size = 3;
        float occlusion_threshold = 0.1f;      // 10cm occlusion threshold
    };
    
    // Perform occlusion culling using depth buffer
    OcclusionResult performOcclusionCulling(
        const std::vector<cv::Point2f>& pixel_coordinates,
        const std::vector<float>& depths,
        const std::vector<size_t>& point_indices,
        const cv::Size& image_size,
        const OcclusionConfig& config = OcclusionConfig{});
        
private:
    // Depth buffer data structure
    struct DepthPixel {
        float min_depth;
        float max_depth;
        size_t point_count;
        std::vector<size_t> point_indices;
    };
    
    using DepthBufferGrid = std::vector<std::vector<DepthPixel>>;
    
    void buildDepthBuffer(const std::vector<cv::Point2f>& pixels,
                         const std::vector<float>& depths,
                         const std::vector<size_t>& indices,
                         const cv::Size& image_size,
                         DepthBufferGrid& depth_grid);
    
    void smoothDepthBuffer(DepthBufferGrid& depth_grid, int kernel_size);
};

// Depth buffer construction and occlusion testing
DepthBuffer::OcclusionResult DepthBuffer::performOcclusionCulling(
    const std::vector<cv::Point2f>& pixel_coordinates,
    const std::vector<float>& depths,
    const std::vector<size_t>& point_indices,
    const cv::Size& image_size,
    const OcclusionConfig& config) {
    
    OcclusionResult result;
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Initialize depth buffer grid
    DepthBufferGrid depth_grid(image_size.height, 
                              std::vector<DepthPixel>(image_size.width));
    
    // Build depth buffer
    buildDepthBuffer(pixel_coordinates, depths, point_indices, image_size, depth_grid);
    
    // Apply depth smoothing if enabled
    if (config.enable_depth_smoothing) {
        smoothDepthBuffer(depth_grid, config.smoothing_kernel_size);
    }
    
    // Perform occlusion testing
    result.is_visible.resize(pixel_coordinates.size());
    
    for (size_t i = 0; i < pixel_coordinates.size(); ++i) {
        const cv::Point2f& pixel = pixel_coordinates[i];
        const float point_depth = depths[i];
        
        const int x = static_cast<int>(std::round(pixel.x));
        const int y = static_cast<int>(std::round(pixel.y));
        
        // Check bounds
        if (x < 0 || y < 0 || x >= image_size.width || y >= image_size.height) {
            result.is_visible[i] = false;
            continue;
        }
        
        const DepthPixel& depth_pixel = depth_grid[y][x];
        
        // Test occlusion
        const bool is_front_most = (point_depth <= depth_pixel.min_depth + config.depth_tolerance);
        const bool within_threshold = (point_depth <= depth_pixel.min_depth + config.occlusion_threshold);
        
        result.is_visible[i] = is_front_most || within_threshold;
        
        if (result.is_visible[i]) {
            result.visible_indices.push_back(point_indices[i]);
        } else {
            result.occluded_indices.push_back(point_indices[i]);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.metrics.occlusion_test_time = 
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    result.metrics.total_points = pixel_coordinates.size();
    result.metrics.visible_points = result.visible_indices.size();
    result.metrics.occluded_points = result.occluded_indices.size();
    
    return result;
}

} // namespace projection
} // namespace prism
```

## Integration Example

### Complete Projection Pipeline

```cpp
// Example integration showing complete pipeline
namespace prism {
namespace projection {

class PRISMProjectionPipeline {
public:
    struct PipelineResult {
        pcl::PointCloud<prism::core::PointXYZIRGB>::Ptr colored_cloud;
        std::vector<ProjectionMetrics> camera_metrics;
        FusionMetrics fusion_metrics;
        std::chrono::microseconds total_processing_time;
    };
    
    PipelineResult processLiDARFrame(
        const pcl::PointCloud<pcl::PointXYZI>& lidar_cloud,
        const std::vector<cv::Mat>& camera_images,
        const std_msgs::msg::Header& header) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        PipelineResult result;
        
        // 1. Transform LiDAR points to all camera coordinate systems
        std::vector<CoordinateTransformer::TransformationResult> transforms;
        for (size_t cam_idx = 0; cam_idx < camera_setup_.camera_count; ++cam_idx) {
            auto transform_result = coord_transformer_.transformPointsBatch(
                lidar_cloud, camera_setup_.extrinsics[cam_idx]);
            transforms.push_back(std::move(transform_result));
        }
        
        // 2. Project to image coordinates for each camera
        std::vector<ProjectionEngine::ProjectionResult> projections;
        for (size_t cam_idx = 0; cam_idx < camera_setup_.camera_count; ++cam_idx) {
            auto projection_result = projection_engine_.projectPointsBatch(
                transforms[cam_idx].camera_points,
                transforms[cam_idx].valid_indices,
                camera_setup_.intrinsics[cam_idx]);
            projections.push_back(std::move(projection_result));
        }
        
        // 3. Extract colors from each camera image
        std::vector<ColorExtractor::ColorExtractionResult> extractions;
        for (size_t cam_idx = 0; cam_idx < camera_setup_.camera_count; ++cam_idx) {
            auto extraction_result = color_extractor_.extractColors(
                camera_images[cam_idx],
                projections[cam_idx].pixel_coordinates,
                projections[cam_idx].original_indices);
            extractions.push_back(std::move(extraction_result));
        }
        
        // 4. Fuse multi-camera colors
        auto fusion_result = fusion_engine_.fuseMultiCameraColors(
            /* combine all transform results */,
            /* combine all point indices */,
            camera_images,
            camera_setup_);
        
        // 5. Create colored point cloud
        result.colored_cloud = createColoredPointCloud(lidar_cloud, fusion_result);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.total_processing_time = 
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        return result;
    }
    
private:
    CalibrationManager::MultiCameraSetup camera_setup_;
    CoordinateTransformer coord_transformer_;
    ProjectionEngine projection_engine_;
    ColorExtractor color_extractor_;
    MultiCameraFusion fusion_engine_;
    DepthBuffer depth_buffer_;
};

} // namespace projection
} // namespace prism
```

This comprehensive projection and color mapping system provides the foundation for high-quality LiDAR-camera fusion in PRISM, with advanced features for handling complex real-world scenarios including multiple cameras, lens distortion, occlusion, and sub-pixel color accuracy.