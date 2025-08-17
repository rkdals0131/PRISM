#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "prism/projection/color_extractor.hpp"
#include "prism/projection/multi_camera_fusion.hpp"
#include "prism/core/calibration_manager.hpp"

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <chrono>

namespace prism {

class PrismColorNode : public rclcpp::Node {
public:
    PrismColorNode() : Node("prism_color_node") {
        RCLCPP_INFO(get_logger(), "Initializing PRISM Color Extraction Node");
        
        // Declare parameters
        declare_parameter("config_file", "");
        declare_parameter("enable_debug", false);
        declare_parameter("fusion_strategy", "weighted_average");
        declare_parameter("interpolation_method", "bilinear");
        declare_parameter("confidence_threshold", 0.5);
        declare_parameter("enable_outlier_rejection", true);
        
        // Load configuration
        std::string config_file = get_parameter("config_file").as_string();
        if (!config_file.empty() && std::filesystem::exists(config_file)) {
            loadConfiguration(config_file);
        } else {
            RCLCPP_WARN(get_logger(), "No valid config file specified, using defaults");
            setupDefaultConfiguration();
        }
        
        // Initialize calibration manager
        calibration_manager_ = std::make_shared<core::CalibrationManager>();
        
        // Initialize color extraction components
        color_extractor_ = std::make_unique<projection::ColorExtractor>();
        multi_camera_fusion_ = std::make_unique<projection::MultiCameraFusion>();
        
        // Setup subscribers and publishers
        setupROS2Interface();
        
        RCLCPP_INFO(get_logger(), "PRISM Color Node initialized successfully");
    }
    
private:
    void loadConfiguration(const std::string& config_file) {
        try {
            YAML::Node config = YAML::LoadFile(config_file);
            
            // Load camera configurations
            if (config["cameras"]) {
                for (const auto& cam_config : config["cameras"]) {
                    CameraConfig camera;
                    camera.id = cam_config["id"].as<std::string>();
                    camera.image_topic = cam_config["image_topic"].as<std::string>();
                    camera.calibration_file = cam_config["calibration_file"].as<std::string>();
                    camera_configs_.push_back(camera);
                }
            }
            
            // Load fusion configuration
            if (config["fusion"]) {
                auto fusion_config = config["fusion"];
                fusion_config_.confidence_threshold = 
                    fusion_config["confidence_threshold"].as<float>(0.5f);
                fusion_config_.distance_weight_factor = 
                    fusion_config["distance_weight_factor"].as<float>(1.0f);
                fusion_config_.enable_outlier_rejection = 
                    fusion_config["enable_outlier_rejection"].as<bool>(true);
                
                std::string strategy = fusion_config["strategy"].as<std::string>("weighted_average");
                if (strategy == "average") {
                    fusion_config_.strategy = projection::MultiCameraFusion::FusionStrategy::AVERAGE;
                } else if (strategy == "max_confidence") {
                    fusion_config_.strategy = projection::MultiCameraFusion::FusionStrategy::MAX_CONFIDENCE;
                } else if (strategy == "median") {
                    fusion_config_.strategy = projection::MultiCameraFusion::FusionStrategy::MEDIAN;
                } else if (strategy == "adaptive") {
                    fusion_config_.strategy = projection::MultiCameraFusion::FusionStrategy::ADAPTIVE;
                } else {
                    fusion_config_.strategy = projection::MultiCameraFusion::FusionStrategy::WEIGHTED_AVERAGE;
                }
            }
            
            // Load extraction configuration
            if (config["extraction"]) {
                auto extraction_config = config["extraction"];
                extraction_config_.enable_subpixel = 
                    extraction_config["enable_subpixel"].as<bool>(true);
                extraction_config_.confidence_threshold = 
                    extraction_config["confidence_threshold"].as<float>(0.7f);
                
                std::string interp = extraction_config["interpolation"].as<std::string>("bilinear");
                if (interp == "nearest") {
                    extraction_config_.interpolation = 
                        projection::ColorExtractor::InterpolationMethod::NEAREST_NEIGHBOR;
                } else if (interp == "bicubic") {
                    extraction_config_.interpolation = 
                        projection::ColorExtractor::InterpolationMethod::BICUBIC;
                } else {
                    extraction_config_.interpolation = 
                        projection::ColorExtractor::InterpolationMethod::BILINEAR;
                }
            }
            
            RCLCPP_INFO(get_logger(), "Loaded configuration from: %s", config_file.c_str());
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Failed to load configuration: %s", e.what());
            setupDefaultConfiguration();
        }
    }
    
    void setupDefaultConfiguration() {
        // Setup default two-camera configuration
        CameraConfig cam1;
        cam1.id = "camera_1";
        cam1.image_topic = "/camera_1/image_raw";
        cam1.calibration_file = "";
        camera_configs_.push_back(cam1);
        
        CameraConfig cam2;
        cam2.id = "camera_2";
        cam2.image_topic = "/camera_2/image_raw";
        cam2.calibration_file = "";
        camera_configs_.push_back(cam2);
        
        // Default fusion config is already initialized in struct constructor
        // Default extraction config is already initialized in struct constructor
    }
    
    void setupROS2Interface() {
        // Create image subscribers for each camera
        for (const auto& cam_config : camera_configs_) {
            auto sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
                shared_from_this(), cam_config.image_topic);
            image_subscribers_[cam_config.id] = sub;
            
            RCLCPP_INFO(get_logger(), "Subscribing to camera %s on topic: %s", 
                       cam_config.id.c_str(), cam_config.image_topic.c_str());
        }
        
        // Create projected points subscriber
        projected_points_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
            shared_from_this(), "/prism/projected_points");
        
        // Setup synchronizer (for simplicity, sync 2 cameras + projected points)
        if (camera_configs_.size() == 2) {
            typedef message_filters::sync_policies::ApproximateTime<
                sensor_msgs::msg::Image,
                sensor_msgs::msg::Image,
                sensor_msgs::msg::PointCloud2> SyncPolicy;
            
            sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
                SyncPolicy(10),
                *image_subscribers_[camera_configs_[0].id],
                *image_subscribers_[camera_configs_[1].id],
                *projected_points_sub_
            );
            
            sync_->registerCallback(
                std::bind(&PrismColorNode::syncCallback, this, 
                         std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
        }
        
        // Create publisher for colored point cloud
        colored_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            "/prism/colored_points", 10);
        
        // Debug publishers if enabled
        if (get_parameter("enable_debug").as_bool()) {
            debug_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
                "/prism/debug/color_extraction", 10);
        }
    }
    
    void syncCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr& img1,
        const sensor_msgs::msg::Image::ConstSharedPtr& img2,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& projected_points) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Convert ROS images to OpenCV
        cv_bridge::CvImagePtr cv_img1, cv_img2;
        try {
            cv_img1 = cv_bridge::toCvCopy(img1, sensor_msgs::image_encodings::BGR8);
            cv_img2 = cv_bridge::toCvCopy(img2, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }
        
        // Parse projected points (assuming they contain pixel coordinates for each camera)
        // This is a simplified version - in practice, you'd need proper parsing
        std::map<std::string, std::vector<cv::Point2f>> camera_pixels;
        std::vector<size_t> point_indices;
        
        // Convert PointCloud2 to PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*projected_points, *cloud);
        
        // For demonstration, assume we have pixel coordinates somehow
        // In practice, this would come from the projection engine
        size_t num_points = cloud->size();
        for (size_t i = 0; i < num_points; ++i) {
            point_indices.push_back(i);
            // Placeholder: would need actual projected pixel coordinates here
            camera_pixels[camera_configs_[0].id].push_back(cv::Point2f(100 + i*10, 100 + i*5));
            camera_pixels[camera_configs_[1].id].push_back(cv::Point2f(200 + i*8, 150 + i*6));
        }
        
        // Extract colors from each camera
        std::map<std::string, projection::ColorExtractor::ColorExtractionResult> extraction_results;
        
        extraction_results[camera_configs_[0].id] = color_extractor_->extractColors(
            cv_img1->image, camera_pixels[camera_configs_[0].id], extraction_config_);
        
        extraction_results[camera_configs_[1].id] = color_extractor_->extractColors(
            cv_img2->image, camera_pixels[camera_configs_[1].id], extraction_config_);
        
        // Fuse colors from multiple cameras
        auto fusion_result = multi_camera_fusion_->fuseColors(
            extraction_results, point_indices, fusion_config_);
        
        // Create colored point cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        colored_cloud->reserve(num_points);
        
        for (size_t i = 0; i < num_points; ++i) {
            pcl::PointXYZRGB point;
            point.x = cloud->points[i].x;
            point.y = cloud->points[i].y;
            point.z = cloud->points[i].z;
            
            if (i < fusion_result.fused_colors.size()) {
                const auto& color = fusion_result.fused_colors[i];
                point.r = color[2];  // OpenCV BGR to RGB
                point.g = color[1];
                point.b = color[0];
            } else {
                point.r = point.g = point.b = 0;
            }
            
            colored_cloud->push_back(point);
        }
        
        // Convert back to ROS message and publish
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(*colored_cloud, output_msg);
        output_msg.header = projected_points->header;
        colored_cloud_pub_->publish(output_msg);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
            "Color extraction and fusion completed in %ld ms. "
            "Processed %zu points, %zu successfully colored (%.1f%%)",
            duration.count(), num_points, fusion_result.successful_fusions,
            100.0f * fusion_result.successful_fusions / std::max(size_t(1), num_points));
    }
    
private:
    struct CameraConfig {
        std::string id;
        std::string image_topic;
        std::string calibration_file;
    };
    
    // Configuration
    std::vector<CameraConfig> camera_configs_;
    projection::ColorExtractor::ColorExtractionConfig extraction_config_;
    projection::MultiCameraFusion::FusionConfig fusion_config_;
    
    // Core components
    std::shared_ptr<core::CalibrationManager> calibration_manager_;
    std::unique_ptr<projection::ColorExtractor> color_extractor_;
    std::unique_ptr<projection::MultiCameraFusion> multi_camera_fusion_;
    
    // ROS2 interface
    std::map<std::string, std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>>> image_subscribers_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> projected_points_sub_;
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image, sensor_msgs::msg::Image, sensor_msgs::msg::PointCloud2>>> sync_;
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr colored_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
};

} // namespace prism

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<prism::PrismColorNode>();
        RCLCPP_INFO(node->get_logger(), "Starting PRISM Color Node");
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception in PRISM Color Node: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}