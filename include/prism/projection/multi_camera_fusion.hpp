#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <memory>
#include <algorithm>
#include <chrono>
#include "prism/projection/color_extractor.hpp"

namespace prism {
namespace projection {

class MultiCameraFusion {
public:
    enum class FusionStrategy {
        AVERAGE,              // Simple averaging of colors
        WEIGHTED_AVERAGE,     // Confidence-weighted averaging
        MAX_CONFIDENCE,       // Select color with highest confidence
        MEDIAN,               // Median color for robustness
        ADAPTIVE              // Adaptive strategy based on context
    };
    
    struct CameraColorData {
        std::string camera_id;
        cv::Vec3b color;
        float confidence;
        float distance;       // Distance from optical center
        bool is_valid;
    };
    
    struct FusionConfig {
        FusionStrategy strategy;
        float confidence_threshold;
        float distance_weight_factor;  // Weight reduction based on distance
        float occlusion_threshold;     // For detecting occlusions
        bool enable_outlier_rejection;
        float outlier_threshold;       // Standard deviations for outlier
        
        FusionConfig()
            : strategy(FusionStrategy::WEIGHTED_AVERAGE),
              confidence_threshold(0.3f),
              distance_weight_factor(1.0f),
              occlusion_threshold(0.5f),
              enable_outlier_rejection(true),
              outlier_threshold(2.0f) {}
    };
    
    struct FusionResult {
        std::vector<cv::Vec3b> fused_colors;
        std::vector<float> fusion_confidence;
        std::vector<std::string> primary_camera_ids;  // Which camera contributed most
        std::vector<int> num_cameras_contributed;     // How many cameras had valid colors
        
        // Statistics
        size_t total_points = 0;
        size_t successful_fusions = 0;
        std::map<std::string, size_t> camera_contribution_count;
        std::chrono::microseconds fusion_time;
        
        void clear() {
            fused_colors.clear();
            fusion_confidence.clear();
            primary_camera_ids.clear();
            num_cameras_contributed.clear();
            total_points = 0;
            successful_fusions = 0;
            camera_contribution_count.clear();
        }
    };
    
    MultiCameraFusion() = default;
    ~MultiCameraFusion() = default;
    
    // Main fusion interface
    FusionResult fuseColors(
        const std::map<std::string, ColorExtractor::ColorExtractionResult>& camera_colors,
        const std::vector<size_t>& point_indices,
        const FusionConfig& config = FusionConfig());
    
    // Fusion with per-camera pixel distances (for distance weighting)
    FusionResult fuseColorsWithDistances(
        const std::map<std::string, ColorExtractor::ColorExtractionResult>& camera_colors,
        const std::map<std::string, std::vector<float>>& pixel_distances,
        const std::vector<size_t>& point_indices,
        const FusionConfig& config = FusionConfig());
    
private:
    // Fusion strategies
    cv::Vec3b fuseAverage(const std::vector<CameraColorData>& valid_colors);
    cv::Vec3b fuseWeightedAverage(const std::vector<CameraColorData>& valid_colors,
                                  const FusionConfig& config);
    cv::Vec3b fuseMaxConfidence(const std::vector<CameraColorData>& valid_colors);
    cv::Vec3b fuseMedian(const std::vector<CameraColorData>& valid_colors);
    cv::Vec3b fuseAdaptive(const std::vector<CameraColorData>& valid_colors,
                           const FusionConfig& config);
    
    // Helper methods
    std::vector<CameraColorData> collectValidColors(
        const std::map<std::string, ColorExtractor::ColorExtractionResult>& camera_colors,
        const std::map<std::string, std::vector<float>>& pixel_distances,
        size_t point_index,
        const FusionConfig& config);
    
    bool detectOutlier(const cv::Vec3b& color,
                       const std::vector<CameraColorData>& all_colors,
                       float threshold);
    
    float calculateFusionConfidence(const std::vector<CameraColorData>& valid_colors,
                                   const FusionConfig& config);
    
    std::string selectPrimaryCamera(const std::vector<CameraColorData>& valid_colors,
                                   const FusionConfig& config);
    
    // Color space conversion helpers
    cv::Vec3f rgbToLab(const cv::Vec3b& rgb);
    cv::Vec3b labToRgb(const cv::Vec3f& lab);
    
    // Statistics helpers
    cv::Vec3f calculateMean(const std::vector<CameraColorData>& colors);
    cv::Vec3f calculateStdDev(const std::vector<CameraColorData>& colors,
                             const cv::Vec3f& mean);
};

} // namespace projection
} // namespace prism