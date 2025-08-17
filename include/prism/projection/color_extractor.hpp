#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cmath>
#include "prism/utils/common_types.hpp"

namespace prism {
namespace projection {

class ColorExtractor {
public:
    enum class InterpolationMethod {
        NEAREST_NEIGHBOR,
        BILINEAR,
        BICUBIC
    };
    
    struct ColorExtractionConfig {
        InterpolationMethod interpolation;
        bool enable_subpixel;
        bool validate_pixel_bounds;
        float confidence_threshold;
        int blur_kernel_size;  // 0 = no blur
        
        ColorExtractionConfig()
            : interpolation(InterpolationMethod::BILINEAR),
              enable_subpixel(true),
              validate_pixel_bounds(true),
              confidence_threshold(0.7f),
              blur_kernel_size(0) {}
              
        /**
         * @brief Load configuration from YAML node
         */
        void loadFromYaml(const YAML::Node& node) {
            using prism::utils::ConfigLoader;
            
            // Interpolation method
            std::string method = ConfigLoader::readNestedParam(node,
                "color_extraction.interpolation_method", std::string("bilinear"));
            if (method == "nearest") {
                interpolation = InterpolationMethod::NEAREST_NEIGHBOR;
            } else if (method == "bicubic") {
                interpolation = InterpolationMethod::BICUBIC;
            } else {
                interpolation = InterpolationMethod::BILINEAR;
            }
            
            enable_subpixel = ConfigLoader::readNestedParam(node,
                "color_extraction.subpixel_accuracy", enable_subpixel);
            validate_pixel_bounds = ConfigLoader::readNestedParam(node,
                "color_extraction.validation.check_pixel_bounds", validate_pixel_bounds);
            confidence_threshold = ConfigLoader::readNestedParam(node,
                "color_extraction.quality.confidence_threshold", confidence_threshold);
            
            bool enable_blur = ConfigLoader::readNestedParam(node,
                "color_extraction.quality.enable_blur", false);
            blur_kernel_size = enable_blur ? 
                ConfigLoader::readNestedParam(node,
                    "color_extraction.quality.blur_kernel_size", 3) : 0;
        }
    };
    
    struct ColorExtractionResult : public prism::utils::BaseResult {
        std::vector<cv::Vec3b> colors;           // RGB colors  
        std::vector<float> confidence_scores;    // Color reliability [0,1]
        std::vector<bool> valid_extractions;     // Successful extractions
        
        // Metrics
        size_t valid_colors = 0;
        
        /**
         * @brief Override clear to handle custom members
         */
        void clear() override {
            prism::utils::BaseResult::clear();  // Call base clear
            colors.clear();
            confidence_scores.clear();
            valid_extractions.clear();
            valid_colors = 0;
        }
        
        /**
         * @brief Calculate extraction success rate
         */
        double getSuccessRate() const {
            return output_count > 0 ? 
                static_cast<double>(valid_colors) / output_count : 0.0;
        }
    };
    
    ColorExtractor() = default;
    ~ColorExtractor() = default;
    
    // Extract colors from single image
    ColorExtractionResult extractColors(
        const cv::Mat& image,
        const std::vector<cv::Point2f>& pixel_coordinates,
        const ColorExtractionConfig& config = ColorExtractionConfig());
    
    // Extract colors with point indices tracking
    ColorExtractionResult extractColorsWithIndices(
        const cv::Mat& image,
        const std::vector<cv::Point2f>& pixel_coordinates,
        const std::vector<size_t>& point_indices,
        const ColorExtractionConfig& config = ColorExtractionConfig());
    
private:
    // Sub-pixel interpolation methods
    cv::Vec3b interpolateNearest(const cv::Mat& image, const cv::Point2f& pixel);
    cv::Vec3b interpolateBilinear(const cv::Mat& image, const cv::Point2f& pixel);
    cv::Vec3b interpolateBicubic(const cv::Mat& image, const cv::Point2f& pixel);
    
    // Helper function to check if pixel is within image bounds
    bool isInBounds(const cv::Point2f& pixel, const cv::Mat& image) const {
        return pixel.x >= 0 && pixel.x < image.cols &&
               pixel.y >= 0 && pixel.y < image.rows;
    }
    
    // Quality assessment for extracted colors
    float assessColorConfidence(const cv::Mat& image, const cv::Point2f& pixel,
                               const cv::Vec3b& extracted_color);
    
    // Clamp value to [0, 255] range
    static inline uchar clamp255(float value) {
        return static_cast<uchar>(std::max(0.0f, std::min(255.0f, value)));
    }
};

} // namespace projection
} // namespace prism