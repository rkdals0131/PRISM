#include "prism/projection/multi_camera_fusion.hpp"
#include <iostream>
#include <numeric>

namespace prism {
namespace projection {

MultiCameraFusion::FusionResult MultiCameraFusion::fuseColors(
    const std::map<std::string, ColorExtractor::ColorExtractionResult>& camera_colors,
    const std::vector<size_t>& point_indices,
    const FusionConfig& config) {
    
    // Create empty distance map for simplified interface
    std::map<std::string, std::vector<float>> empty_distances;
    return fuseColorsWithDistances(camera_colors, empty_distances, point_indices, config);
}

MultiCameraFusion::FusionResult MultiCameraFusion::fuseColorsWithDistances(
    const std::map<std::string, ColorExtractor::ColorExtractionResult>& camera_colors,
    const std::map<std::string, std::vector<float>>& pixel_distances,
    const std::vector<size_t>& point_indices,
    const FusionConfig& config) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    FusionResult result;
    
    if (camera_colors.empty() || point_indices.empty()) {
        std::cerr << "MultiCameraFusion: Empty input" << std::endl;
        return result;
    }
    
    result.total_points = point_indices.size();
    result.fused_colors.reserve(result.total_points);
    result.fusion_confidence.reserve(result.total_points);
    result.primary_camera_ids.reserve(result.total_points);
    result.num_cameras_contributed.reserve(result.total_points);
    
    // Process each point
    for (size_t point_idx : point_indices) {
        // Collect valid colors from all cameras for this point
        std::vector<CameraColorData> valid_colors = collectValidColors(
            camera_colors, pixel_distances, point_idx, config);
        
        if (valid_colors.empty()) {
            // No valid colors from any camera
            result.fused_colors.push_back(cv::Vec3b(0, 0, 0));
            result.fusion_confidence.push_back(0.0f);
            result.primary_camera_ids.push_back("");
            result.num_cameras_contributed.push_back(0);
            continue;
        }
        
        // Apply outlier rejection if enabled
        if (config.enable_outlier_rejection && valid_colors.size() >= 3) {
            std::vector<CameraColorData> filtered_colors;
            for (const auto& color_data : valid_colors) {
                if (!detectOutlier(color_data.color, valid_colors, config.outlier_threshold)) {
                    filtered_colors.push_back(color_data);
                }
            }
            if (!filtered_colors.empty()) {
                valid_colors = filtered_colors;
            }
        }
        
        // Perform fusion based on strategy
        cv::Vec3b fused_color;
        switch (config.strategy) {
            case FusionStrategy::AVERAGE:
                fused_color = fuseAverage(valid_colors);
                break;
            case FusionStrategy::WEIGHTED_AVERAGE:
                fused_color = fuseWeightedAverage(valid_colors, config);
                break;
            case FusionStrategy::MAX_CONFIDENCE:
                fused_color = fuseMaxConfidence(valid_colors);
                break;
            case FusionStrategy::MEDIAN:
                fused_color = fuseMedian(valid_colors);
                break;
            case FusionStrategy::ADAPTIVE:
                fused_color = fuseAdaptive(valid_colors, config);
                break;
            default:
                fused_color = fuseWeightedAverage(valid_colors, config);
        }
        
        // Calculate fusion confidence
        float fusion_conf = calculateFusionConfidence(valid_colors, config);
        
        // Determine primary camera
        std::string primary_cam = selectPrimaryCamera(valid_colors, config);
        
        // Update statistics
        for (const auto& color_data : valid_colors) {
            result.camera_contribution_count[color_data.camera_id]++;
        }
        
        result.fused_colors.push_back(fused_color);
        result.fusion_confidence.push_back(fusion_conf);
        result.primary_camera_ids.push_back(primary_cam);
        result.num_cameras_contributed.push_back(valid_colors.size());
        
        if (fusion_conf >= config.confidence_threshold) {
            result.successful_fusions++;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.fusion_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    
    return result;
}

cv::Vec3b MultiCameraFusion::fuseAverage(const std::vector<CameraColorData>& valid_colors) {
    if (valid_colors.empty()) {
        return cv::Vec3b(0, 0, 0);
    }
    
    cv::Vec3f sum(0, 0, 0);
    for (const auto& color_data : valid_colors) {
        sum[0] += color_data.color[0];
        sum[1] += color_data.color[1];
        sum[2] += color_data.color[2];
    }
    
    sum /= static_cast<float>(valid_colors.size());
    
    return cv::Vec3b(
        static_cast<uchar>(std::round(sum[0])),
        static_cast<uchar>(std::round(sum[1])),
        static_cast<uchar>(std::round(sum[2]))
    );
}

cv::Vec3b MultiCameraFusion::fuseWeightedAverage(
    const std::vector<CameraColorData>& valid_colors,
    const FusionConfig& config) {
    
    if (valid_colors.empty()) {
        return cv::Vec3b(0, 0, 0);
    }
    
    cv::Vec3f weighted_sum(0, 0, 0);
    float weight_sum = 0.0f;
    
    for (const auto& color_data : valid_colors) {
        // Calculate weight based on confidence and distance
        float weight = color_data.confidence;
        
        // Apply distance weighting if available
        if (color_data.distance > 0 && config.distance_weight_factor > 0) {
            // Reduce weight for points far from image center
            float distance_weight = 1.0f / (1.0f + config.distance_weight_factor * 
                                           color_data.distance / 1000.0f);
            weight *= distance_weight;
        }
        
        weighted_sum[0] += weight * color_data.color[0];
        weighted_sum[1] += weight * color_data.color[1];
        weighted_sum[2] += weight * color_data.color[2];
        weight_sum += weight;
    }
    
    if (weight_sum > 1e-6f) {
        weighted_sum /= weight_sum;
    }
    
    return cv::Vec3b(
        static_cast<uchar>(std::round(weighted_sum[0])),
        static_cast<uchar>(std::round(weighted_sum[1])),
        static_cast<uchar>(std::round(weighted_sum[2]))
    );
}

cv::Vec3b MultiCameraFusion::fuseMaxConfidence(const std::vector<CameraColorData>& valid_colors) {
    if (valid_colors.empty()) {
        return cv::Vec3b(0, 0, 0);
    }
    
    auto max_elem = std::max_element(valid_colors.begin(), valid_colors.end(),
        [](const CameraColorData& a, const CameraColorData& b) {
            return a.confidence < b.confidence;
        });
    
    return max_elem->color;
}

cv::Vec3b MultiCameraFusion::fuseMedian(const std::vector<CameraColorData>& valid_colors) {
    if (valid_colors.empty()) {
        return cv::Vec3b(0, 0, 0);
    }
    
    if (valid_colors.size() == 1) {
        return valid_colors[0].color;
    }
    
    // Collect color channels separately
    std::vector<uchar> channel_0, channel_1, channel_2;
    for (const auto& color_data : valid_colors) {
        channel_0.push_back(color_data.color[0]);
        channel_1.push_back(color_data.color[1]);
        channel_2.push_back(color_data.color[2]);
    }
    
    // Find median for each channel
    size_t n = valid_colors.size();
    size_t mid = n / 2;
    
    std::nth_element(channel_0.begin(), channel_0.begin() + mid, channel_0.end());
    std::nth_element(channel_1.begin(), channel_1.begin() + mid, channel_1.end());
    std::nth_element(channel_2.begin(), channel_2.begin() + mid, channel_2.end());
    
    uchar median_0 = channel_0[mid];
    uchar median_1 = channel_1[mid];
    uchar median_2 = channel_2[mid];
    
    // For even number of elements, average the two middle values
    if (n % 2 == 0 && n > 2) {
        std::nth_element(channel_0.begin(), channel_0.begin() + mid - 1, channel_0.end());
        std::nth_element(channel_1.begin(), channel_1.begin() + mid - 1, channel_1.end());
        std::nth_element(channel_2.begin(), channel_2.begin() + mid - 1, channel_2.end());
        
        median_0 = (median_0 + channel_0[mid - 1]) / 2;
        median_1 = (median_1 + channel_1[mid - 1]) / 2;
        median_2 = (median_2 + channel_2[mid - 1]) / 2;
    }
    
    return cv::Vec3b(median_0, median_1, median_2);
}

cv::Vec3b MultiCameraFusion::fuseAdaptive(
    const std::vector<CameraColorData>& valid_colors,
    const FusionConfig& config) {
    
    if (valid_colors.empty()) {
        return cv::Vec3b(0, 0, 0);
    }
    
    // Adaptive strategy based on number of cameras and confidence distribution
    size_t num_cameras = valid_colors.size();
    
    if (num_cameras == 1) {
        // Single camera: return as-is
        return valid_colors[0].color;
    } else if (num_cameras == 2) {
        // Two cameras: weighted average
        return fuseWeightedAverage(valid_colors, config);
    } else {
        // Multiple cameras: check variance
        cv::Vec3f mean = calculateMean(valid_colors);
        cv::Vec3f stddev = calculateStdDev(valid_colors, mean);
        
        float avg_stddev = (stddev[0] + stddev[1] + stddev[2]) / 3.0f;
        
        if (avg_stddev > 30.0f) {
            // High variance: use median for robustness
            return fuseMedian(valid_colors);
        } else {
            // Low variance: use weighted average
            return fuseWeightedAverage(valid_colors, config);
        }
    }
}

std::vector<MultiCameraFusion::CameraColorData> MultiCameraFusion::collectValidColors(
    const std::map<std::string, ColorExtractor::ColorExtractionResult>& camera_colors,
    const std::map<std::string, std::vector<float>>& pixel_distances,
    size_t point_index,
    const FusionConfig& config) {
    
    std::vector<CameraColorData> valid_colors;
    
    for (const auto& [camera_id, color_result] : camera_colors) {
        // Check if this point index exists in the results
        if (point_index >= color_result.colors.size()) {
            continue;
        }
        
        // Check if extraction was valid
        if (!color_result.valid_extractions[point_index]) {
            continue;
        }
        
        // Check confidence threshold
        if (color_result.confidence_scores[point_index] < config.confidence_threshold) {
            continue;
        }
        
        CameraColorData data;
        data.camera_id = camera_id;
        data.color = color_result.colors[point_index];
        data.confidence = color_result.confidence_scores[point_index];
        data.is_valid = true;
        
        // Get distance if available
        auto dist_it = pixel_distances.find(camera_id);
        if (dist_it != pixel_distances.end() && point_index < dist_it->second.size()) {
            data.distance = dist_it->second[point_index];
        } else {
            data.distance = 0.0f;
        }
        
        valid_colors.push_back(data);
    }
    
    return valid_colors;
}

bool MultiCameraFusion::detectOutlier(
    const cv::Vec3b& color,
    const std::vector<CameraColorData>& all_colors,
    float threshold) {
    
    if (all_colors.size() < 3) {
        return false;  // Not enough data for outlier detection
    }
    
    // Calculate mean and standard deviation
    cv::Vec3f mean = calculateMean(all_colors);
    cv::Vec3f stddev = calculateStdDev(all_colors, mean);
    
    // Check if color is outside threshold * stddev from mean
    for (int i = 0; i < 3; ++i) {
        float diff = std::abs(color[i] - mean[i]);
        if (stddev[i] > 1e-6f && diff > threshold * stddev[i]) {
            return true;  // Outlier detected
        }
    }
    
    return false;
}

float MultiCameraFusion::calculateFusionConfidence(
    const std::vector<CameraColorData>& valid_colors,
    const FusionConfig& config) {
    
    if (valid_colors.empty()) {
        return 0.0f;
    }
    
    // Base confidence is average of individual confidences
    float avg_confidence = 0.0f;
    for (const auto& color_data : valid_colors) {
        avg_confidence += color_data.confidence;
    }
    avg_confidence /= valid_colors.size();
    
    // Boost confidence if multiple cameras agree
    float agreement_boost = 1.0f;
    if (valid_colors.size() >= 2) {
        cv::Vec3f mean = calculateMean(valid_colors);
        cv::Vec3f stddev = calculateStdDev(valid_colors, mean);
        float avg_stddev = (stddev[0] + stddev[1] + stddev[2]) / 3.0f;
        
        // Lower standard deviation means better agreement
        if (avg_stddev < 10.0f) {
            agreement_boost = 1.2f;
        } else if (avg_stddev < 20.0f) {
            agreement_boost = 1.1f;
        }
    }
    
    // More cameras generally means higher confidence
    float camera_count_factor = 1.0f + 0.1f * (valid_colors.size() - 1);
    camera_count_factor = std::min(camera_count_factor, 1.5f);
    
    float final_confidence = avg_confidence * agreement_boost * camera_count_factor;
    return std::min(1.0f, final_confidence);
}

std::string MultiCameraFusion::selectPrimaryCamera(
    const std::vector<CameraColorData>& valid_colors,
    const FusionConfig& config) {
    
    if (valid_colors.empty()) {
        return "";
    }
    
    // Select camera with highest confidence (considering distance if available)
    float best_score = -1.0f;
    std::string best_camera;
    
    for (const auto& color_data : valid_colors) {
        float score = color_data.confidence;
        
        // Penalize based on distance from center if available
        if (color_data.distance > 0 && config.distance_weight_factor > 0) {
            float distance_penalty = 1.0f / (1.0f + config.distance_weight_factor * 
                                            color_data.distance / 1000.0f);
            score *= distance_penalty;
        }
        
        if (score > best_score) {
            best_score = score;
            best_camera = color_data.camera_id;
        }
    }
    
    return best_camera;
}

cv::Vec3f MultiCameraFusion::calculateMean(const std::vector<CameraColorData>& colors) {
    cv::Vec3f mean(0, 0, 0);
    
    for (const auto& color_data : colors) {
        mean[0] += color_data.color[0];
        mean[1] += color_data.color[1];
        mean[2] += color_data.color[2];
    }
    
    if (!colors.empty()) {
        mean /= static_cast<float>(colors.size());
    }
    
    return mean;
}

cv::Vec3f MultiCameraFusion::calculateStdDev(
    const std::vector<CameraColorData>& colors,
    const cv::Vec3f& mean) {
    
    cv::Vec3f variance(0, 0, 0);
    
    for (const auto& color_data : colors) {
        cv::Vec3f diff;
        diff[0] = color_data.color[0] - mean[0];
        diff[1] = color_data.color[1] - mean[1];
        diff[2] = color_data.color[2] - mean[2];
        
        variance[0] += diff[0] * diff[0];
        variance[1] += diff[1] * diff[1];
        variance[2] += diff[2] * diff[2];
    }
    
    if (!colors.empty()) {
        variance /= static_cast<float>(colors.size());
    }
    
    cv::Vec3f stddev;
    stddev[0] = std::sqrt(variance[0]);
    stddev[1] = std::sqrt(variance[1]);
    stddev[2] = std::sqrt(variance[2]);
    
    return stddev;
}

// Lab color space conversion for better color comparison (optional, not used in main logic yet)
cv::Vec3f MultiCameraFusion::rgbToLab(const cv::Vec3b& rgb) {
    cv::Mat rgb_mat(1, 1, CV_8UC3, cv::Scalar(rgb[0], rgb[1], rgb[2]));
    cv::Mat lab_mat;
    cv::cvtColor(rgb_mat, lab_mat, cv::COLOR_BGR2Lab);
    cv::Vec3b lab_b = lab_mat.at<cv::Vec3b>(0, 0);
    return cv::Vec3f(lab_b[0], lab_b[1], lab_b[2]);
}

cv::Vec3b MultiCameraFusion::labToRgb(const cv::Vec3f& lab) {
    cv::Mat lab_mat(1, 1, CV_8UC3, cv::Scalar(
        static_cast<uchar>(lab[0]),
        static_cast<uchar>(lab[1]),
        static_cast<uchar>(lab[2])
    ));
    cv::Mat rgb_mat;
    cv::cvtColor(lab_mat, rgb_mat, cv::COLOR_Lab2BGR);
    return rgb_mat.at<cv::Vec3b>(0, 0);
}

} // namespace projection
} // namespace prism