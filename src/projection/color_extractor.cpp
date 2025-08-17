#include "prism/projection/color_extractor.hpp"
#include <iostream>

namespace prism {
namespace projection {

ColorExtractor::ColorExtractionResult ColorExtractor::extractColors(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& pixel_coordinates,
    const ColorExtractionConfig& config) {
    
    std::vector<size_t> dummy_indices(pixel_coordinates.size());
    for (size_t i = 0; i < pixel_coordinates.size(); ++i) {
        dummy_indices[i] = i;
    }
    return extractColorsWithIndices(image, pixel_coordinates, dummy_indices, config);
}

ColorExtractor::ColorExtractionResult ColorExtractor::extractColorsWithIndices(
    const cv::Mat& image,
    const std::vector<cv::Point2f>& pixel_coordinates,
    const std::vector<size_t>& point_indices,
    const ColorExtractionConfig& config) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    ColorExtractionResult result;
    
    // Check input validity
    if (image.empty() || pixel_coordinates.empty()) {
        std::cerr << "ColorExtractor: Empty input image or coordinates" << std::endl;
        return result;
    }
    
    // Ensure image is in BGR format
    cv::Mat bgr_image;
    if (image.channels() == 1) {
        cv::cvtColor(image, bgr_image, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, bgr_image, cv::COLOR_BGRA2BGR);
    } else {
        bgr_image = image;
    }
    
    // Apply optional blur for noise reduction
    cv::Mat processed_image;
    if (config.blur_kernel_size > 0 && config.blur_kernel_size % 2 == 1) {
        cv::GaussianBlur(bgr_image, processed_image, 
                        cv::Size(config.blur_kernel_size, config.blur_kernel_size), 0);
    } else {
        processed_image = bgr_image;
    }
    
    // Reserve space for results
    const size_t num_points = pixel_coordinates.size();
    result.colors.reserve(num_points);
    result.confidence_scores.reserve(num_points);
    result.valid_extractions.reserve(num_points);
    result.output_count = num_points;
    
    // Process each pixel coordinate
    for (size_t i = 0; i < num_points; ++i) {
        const cv::Point2f& pixel = pixel_coordinates[i];
        
        // Check bounds if validation is enabled
        if (config.validate_pixel_bounds && !isInBounds(pixel, processed_image)) {
            result.colors.push_back(cv::Vec3b(0, 0, 0));
            result.confidence_scores.push_back(0.0f);
            result.valid_extractions.push_back(false);
            continue;
        }
        
        // Extract color based on interpolation method
        cv::Vec3b bgr_color;
        if (config.enable_subpixel) {
            switch (config.interpolation) {
                case InterpolationMethod::NEAREST_NEIGHBOR:
                    bgr_color = interpolateNearest(processed_image, pixel);
                    break;
                case InterpolationMethod::BICUBIC:
                    bgr_color = interpolateBicubic(processed_image, pixel);
                    break;
                case InterpolationMethod::BILINEAR:
                default:
                    bgr_color = interpolateBilinear(processed_image, pixel);
                    break;
            }
        } else {
            // Simple nearest neighbor without sub-pixel
            int x = static_cast<int>(std::round(pixel.x));
            int y = static_cast<int>(std::round(pixel.y));
            
            if (x >= 0 && x < processed_image.cols && y >= 0 && y < processed_image.rows) {
                bgr_color = processed_image.at<cv::Vec3b>(y, x);
            } else {
                bgr_color = cv::Vec3b(0, 0, 0);
            }
        }
        
        // Convert BGR to RGB for PCL compatibility
        cv::Vec3b rgb_color(bgr_color[2], bgr_color[1], bgr_color[0]);
        
        // Assess confidence
        float confidence = assessColorConfidence(processed_image, pixel, bgr_color);
        bool is_valid = confidence >= config.confidence_threshold;
        
        result.colors.push_back(rgb_color);
        result.confidence_scores.push_back(confidence);
        result.valid_extractions.push_back(is_valid);
        
        if (is_valid) {
            result.valid_colors++;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.processing_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    
    return result;
}

cv::Vec3b ColorExtractor::interpolateNearest(const cv::Mat& image, const cv::Point2f& pixel) {
    const int x = static_cast<int>(std::round(pixel.x));
    const int y = static_cast<int>(std::round(pixel.y));
    
    // Clamp to image bounds
    const int safe_x = std::max(0, std::min(image.cols - 1, x));
    const int safe_y = std::max(0, std::min(image.rows - 1, y));
    
    return image.at<cv::Vec3b>(safe_y, safe_x);
}

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
        
        result[c] = clamp255(interpolated);
    }
    
    return result;
}

cv::Vec3b ColorExtractor::interpolateBicubic(const cv::Mat& image, const cv::Point2f& pixel) {
    const int x = static_cast<int>(std::floor(pixel.x));
    const int y = static_cast<int>(std::floor(pixel.y));
    const float fx = pixel.x - x;
    const float fy = pixel.y - y;
    
    // Cubic interpolation kernel
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
        clamp255(result[0]),
        clamp255(result[1]),
        clamp255(result[2])
    );
}

float ColorExtractor::assessColorConfidence(const cv::Mat& image, const cv::Point2f& pixel,
                                           const cv::Vec3b& extracted_color) {
    // Simple confidence based on proximity to image bounds
    const float margin = 2.0f;
    
    float x_confidence = 1.0f;
    float y_confidence = 1.0f;
    
    // Reduce confidence near edges
    if (pixel.x < margin) {
        x_confidence = pixel.x / margin;
    } else if (pixel.x > image.cols - margin) {
        x_confidence = (image.cols - pixel.x) / margin;
    }
    
    if (pixel.y < margin) {
        y_confidence = pixel.y / margin;
    } else if (pixel.y > image.rows - margin) {
        y_confidence = (image.rows - pixel.y) / margin;
    }
    
    // Combined confidence
    float confidence = x_confidence * y_confidence;
    
    // Reduce confidence for very dark or very bright pixels (potential artifacts)
    float brightness = (extracted_color[0] + extracted_color[1] + extracted_color[2]) / 3.0f;
    if (brightness < 10.0f || brightness > 245.0f) {
        confidence *= 0.8f;
    }
    
    return std::max(0.0f, std::min(1.0f, confidence));
}

} // namespace projection
} // namespace prism