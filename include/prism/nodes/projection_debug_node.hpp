#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "prism/projection/projection_engine.hpp"
#include "prism/core/calibration_manager.hpp"

#include <memory>
#include <unordered_map>
#include <string>

namespace prism {
namespace nodes {

/**
 * @brief Debug node for visualizing LiDAR-to-camera projection
 * 
 * This node subscribes to LiDAR point clouds and camera images,
 * projects the LiDAR points onto the camera images using the
 * projection engine, and publishes visualization images showing
 * the projected points color-coded by depth.
 * 
 * Features:
 * - Subscribes to multiple camera topics
 * - Time-synchronized processing of LiDAR and camera data
 * - Real-time projection visualization
 * - Configurable projection parameters
 * - Performance monitoring and statistics
 */
class ProjectionDebugNode : public rclcpp::Node {
public:
    /**
     * @brief Constructor
     * @param options Node options
     */
    explicit ProjectionDebugNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    
    /**
     * @brief Destructor
     */
    ~ProjectionDebugNode() override;

private:
    /**
     * @brief Camera subscriber data
     */
    struct CameraSubscriber {
        std::string topic;
        std::shared_ptr<image_transport::Subscriber> subscriber;
        cv::Mat latest_image;
        rclcpp::Time latest_timestamp;
        mutable std::mutex image_mutex;
        
        CameraSubscriber() = default;
        CameraSubscriber(const std::string& topic_name) : topic(topic_name) {}
        
        // Delete copy constructor and assignment due to mutex
        CameraSubscriber(const CameraSubscriber&) = delete;
        CameraSubscriber& operator=(const CameraSubscriber&) = delete;
        
        // Custom move constructor and assignment
        CameraSubscriber(CameraSubscriber&& other) noexcept
            : topic(std::move(other.topic))
            , subscriber(std::move(other.subscriber))
            , latest_image(std::move(other.latest_image))
            , latest_timestamp(std::move(other.latest_timestamp)) {
        }
        
        CameraSubscriber& operator=(CameraSubscriber&& other) noexcept {
            if (this != &other) {
                topic = std::move(other.topic);
                subscriber = std::move(other.subscriber);
                latest_image = std::move(other.latest_image);
                latest_timestamp = std::move(other.latest_timestamp);
            }
            return *this;
        }
    };
    
    /**
     * @brief Initialize the node
     */
    void initialize();
    
    /**
     * @brief Load parameters from ROS parameter server
     */
    void loadParameters();
    
    /**
     * @brief Setup ROS subscribers and publishers
     */
    void setupROS();
    
    /**
     * @brief Initialize projection engine
     */
    void initializeProjectionEngine();
    
    /**
     * @brief Point cloud callback
     * @param msg Point cloud message
     */
    void rawCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void interpolatedCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    
    /**
     * @brief Camera image callback
     * @param camera_id Camera identifier
     * @param msg Image message
     */
    void cameraCallback(const std::string& camera_id, 
                       const sensor_msgs::msg::Image::ConstSharedPtr& msg);
    
    /**
     * @brief Process projection and create visualization
     * @param cloud_msg Point cloud message
     */
    void processProjection(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg);
    
    /**
     * @brief Create and publish projection visualization
     * @param result Projection result
     * @param timestamp Message timestamp
     */
    void publishVisualization(const projection::ProjectionResult& result,
                             const rclcpp::Time& timestamp);
    
    /**
     * @brief Get synchronized camera images
     * @param timestamp Target timestamp
     * @param tolerance Time tolerance for synchronization
     * @return Map of camera_id to images
     */
    std::unordered_map<std::string, cv::Mat> getSynchronizedImages(
        const rclcpp::Time& timestamp, 
        const rclcpp::Duration& tolerance);
    
    /**
     * @brief Timer callback for publishing statistics
     */
    void statisticsTimerCallback();
    
    /**
     * @brief Convert ROS point cloud to LiDAR points
     * @param cloud_msg ROS point cloud message
     * @return Vector of LiDAR points
     */
    std::vector<projection::LiDARPoint> convertPointCloud(
        const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg);
    
    /**
     * @brief Publish performance statistics
     */
    void publishStatistics();
    
    /**
     * @brief Create status overlay for visualization
     * @param image Input image
     * @param camera_id Camera identifier
     * @param projection Camera projection result
     * @return Image with status overlay
     */
    cv::Mat addStatusOverlay(const cv::Mat& image, 
                            const std::string& camera_id,
                            const projection::CameraProjection& projection);
    
    // Core components
    std::shared_ptr<prism::core::CalibrationManager> calibration_manager_;
    std::shared_ptr<projection::ProjectionEngine> projection_engine_;
    projection::ProjectionConfig projection_config_;
    
    // ROS components
    std::shared_ptr<image_transport::ImageTransport> image_transport_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscriber_raw_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_subscriber_interp_;
    
    // Camera subscribers and publishers
    std::unordered_map<std::string, CameraSubscriber> camera_subscribers_;
    std::unordered_map<std::string, image_transport::Publisher> visualization_publishers_;
    
    // Statistics and monitoring
    rclcpp::TimerBase::SharedPtr statistics_timer_;
    rclcpp::TimerBase::SharedPtr initialization_timer_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr statistics_publisher_;
    
    // Configuration parameters
    std::vector<std::string> camera_ids_;
    std::string calibration_directory_;
    std::string lidar_topic_;
    std::string interpolated_topic_;
    bool prefer_interpolated_ {true};
    rclcpp::Time last_interpolated_stamp_ {0,0,RCL_ROS_TIME};
    std::string output_topic_prefix_;
    
    // Timing and synchronization
    double time_tolerance_sec_;
    bool enable_time_sync_;
    
    // Visualization parameters
    bool enable_status_overlay_;
    int point_radius_;
    double overlay_alpha_;
    
    // Performance monitoring
    size_t processed_clouds_;
    size_t successful_projections_;
    rclcpp::Time last_statistics_time_;
    double avg_processing_time_ms_;
    
    // Thread safety
    std::mutex processing_mutex_;
    std::atomic<bool> is_initialized_;
    
    // Parameter validation
    bool validateParameters() const;
    
    /**
     * @brief Log projection statistics
     * @param result Projection result to log
     */
    void logProjectionResult(const projection::ProjectionResult& result);
    
    /**
     * @brief Check if all cameras have recent images
     * @param timestamp Reference timestamp
     * @param tolerance Time tolerance
     * @return True if all cameras have synchronized images
     */
    bool allCamerasReady(const rclcpp::Time& timestamp, 
                        const rclcpp::Duration& tolerance) const;
};

} // namespace nodes
} // namespace prism