#include "prism/nodes/projection_debug_node.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <std_msgs/msg/string.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>
#include <sstream>
#include <iomanip>

namespace prism {
namespace nodes {

ProjectionDebugNode::ProjectionDebugNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("projection_debug_node", options)
    , processed_clouds_(0)
    , successful_projections_(0)
    , avg_processing_time_ms_(0.0)
    , is_initialized_(false) {
    
    RCLCPP_INFO(this->get_logger(), "Initializing ProjectionDebugNode...");
    
    try {
        // Load parameters and initialize projection engine first
        loadParameters();
        
        if (!validateParameters()) {
            throw std::runtime_error("Invalid parameters");
        }
        
        initializeProjectionEngine();
        
        // Defer setupROS() call using a timer to ensure shared_from_this() is ready
        initialization_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            [this]() {
                try {
                    setupROS();
                    is_initialized_ = true;
                    last_statistics_time_ = this->get_clock()->now();
                    RCLCPP_INFO(this->get_logger(), "ProjectionDebugNode initialized successfully");
                    // Cancel the timer after successful initialization
                    initialization_timer_->cancel();
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "Failed to setup ROS interfaces: %s", e.what());
                    initialization_timer_->cancel();
                }
            });
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize ProjectionDebugNode: %s", e.what());
        throw;
    }
}

ProjectionDebugNode::~ProjectionDebugNode() {
    RCLCPP_INFO(this->get_logger(), "Shutting down ProjectionDebugNode");
}

void ProjectionDebugNode::initialize() {
    // This function is now deprecated - initialization is done in constructor
    // and setupROS() is deferred via timer
}

void ProjectionDebugNode::loadParameters() {
    // Camera configuration
    this->declare_parameter("camera_ids", std::vector<std::string>{"camera_1", "camera_2"});
    camera_ids_ = this->get_parameter("camera_ids").as_string_array();
    
    // Topic configuration
    this->declare_parameter("lidar_topic", "/ouster/points");
    lidar_topic_ = this->get_parameter("lidar_topic").as_string();
    
    this->declare_parameter("output_topic_prefix", "/prism/projection_debug");
    output_topic_prefix_ = this->get_parameter("output_topic_prefix").as_string();
    
    // Calibration
    // Default to package share config directory if not provided
    this->declare_parameter("calibration_directory", "");
    calibration_directory_ = this->get_parameter("calibration_directory").as_string();
    if (calibration_directory_.empty()) {
        try {
            // Resolve package share dir at runtime
            std::string pkg = "prism";
            std::string cmd = std::string("python3 - <<'PY'\n") +
                "import ament_index_python.packages as pk;\n" +
                "print(pk.get_package_share_directory('prism'))\n" +
                "PY";
            FILE* pipe = popen(cmd.c_str(), "r");
            if (pipe) {
                char buffer[512]; std::string result;
                while (fgets(buffer, sizeof(buffer), pipe) != nullptr) result += buffer;
                pclose(pipe);
                // Trim whitespace
                result.erase(result.find_last_not_of("\n\r \t") + 1);
                calibration_directory_ = result + "/config";
            }
        } catch (...) {
            calibration_directory_ = "/tmp/prism/config"; // last resort
        }
    }
    
    // Synchronization parameters
    this->declare_parameter("time_tolerance_sec", 0.1);
    time_tolerance_sec_ = this->get_parameter("time_tolerance_sec").as_double();
    
    this->declare_parameter("enable_time_sync", true);
    enable_time_sync_ = this->get_parameter("enable_time_sync").as_bool();
    
    // Visualization parameters
    this->declare_parameter("enable_status_overlay", true);
    enable_status_overlay_ = this->get_parameter("enable_status_overlay").as_bool();
    
    this->declare_parameter("point_radius", 2);
    point_radius_ = this->get_parameter("point_radius").as_int();
    
    this->declare_parameter("overlay_alpha", 0.7);
    overlay_alpha_ = this->get_parameter("overlay_alpha").as_double();
    
    // Projection configuration
    this->declare_parameter("projection.min_depth", 0.5);
    projection_config_.min_depth = this->get_parameter("projection.min_depth").as_double();
    
    this->declare_parameter("projection.max_depth", 50.0);
    projection_config_.max_depth = this->get_parameter("projection.max_depth").as_double();
    
    this->declare_parameter("projection.margin_pixels", 5);
    projection_config_.margin_pixels = this->get_parameter("projection.margin_pixels").as_int();
    
    this->declare_parameter("projection.enable_frustum_culling", true);
    projection_config_.enable_frustum_culling = this->get_parameter("projection.enable_frustum_culling").as_bool();
    
    this->declare_parameter("projection.enable_distortion_correction", true);
    projection_config_.enable_distortion_correction = this->get_parameter("projection.enable_distortion_correction").as_bool();
    
    this->declare_parameter("projection.enable_debug_visualization", true);
    projection_config_.enable_debug_visualization = this->get_parameter("projection.enable_debug_visualization").as_bool();
    
    projection_config_.debug_point_radius = point_radius_;
    
    RCLCPP_INFO(this->get_logger(), "Loaded parameters:");
    RCLCPP_INFO(this->get_logger(), "  - Cameras: %zu", camera_ids_.size());
    RCLCPP_INFO(this->get_logger(), "  - LiDAR topic: %s", lidar_topic_.c_str());
    // No interpolated preference logic; use lidar_topic_ as-is from YAML
    RCLCPP_INFO(this->get_logger(), "  - Calibration dir: %s", calibration_directory_.c_str());
    RCLCPP_INFO(this->get_logger(), "  - Depth range: %.2f - %.2f m", 
               projection_config_.min_depth, projection_config_.max_depth);
}

bool ProjectionDebugNode::validateParameters() const {
    if (camera_ids_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "No camera IDs specified");
        return false;
    }
    
    if (lidar_topic_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "LiDAR topic not specified");
        return false;
    }
    
    if (calibration_directory_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Calibration directory not specified");
        return false;
    }
    
    if (projection_config_.min_depth >= projection_config_.max_depth) {
        RCLCPP_ERROR(this->get_logger(), "Invalid depth range");
        return false;
    }
    
    if (time_tolerance_sec_ <= 0.0) {
        RCLCPP_ERROR(this->get_logger(), "Invalid time tolerance");
        return false;
    }
    
    return true;
}

void ProjectionDebugNode::initializeProjectionEngine() {
    // Create calibration manager
    prism::core::CalibrationManager::Config cal_config;
    cal_config.calibration_directory = calibration_directory_;
    cal_config.enable_hot_reload = true;
    
    calibration_manager_ = std::make_shared<prism::core::CalibrationManager>(cal_config);
    
    // Load calibrations for all cameras (CALICO style)
    std::string intrinsic_file = calibration_directory_ + "/multi_camera_intrinsic_calibration.yaml";
    std::string extrinsic_file = calibration_directory_ + "/multi_camera_extrinsic_calibration.yaml";
    
    try {
        // Load YAML files
        YAML::Node intrinsic_yaml = YAML::LoadFile(intrinsic_file);
        YAML::Node extrinsic_yaml = YAML::LoadFile(extrinsic_file);
        
        for (const auto& camera_id : camera_ids_) {
            // Check if camera exists in YAML files
            if (!intrinsic_yaml[camera_id]) {
                RCLCPP_WARN(this->get_logger(), "Camera %s not found in intrinsic calibration file", camera_id.c_str());
                continue;
            }
            
            // Create a new YAML node with correct structure for CalibrationManager
            YAML::Node camera_node;
            
            // Parse intrinsic parameters (CALICO format)
            YAML::Node intrinsic_node = intrinsic_yaml[camera_id];
            
            // Camera matrix - flatten nested array
            if (intrinsic_node["camera_matrix"] && intrinsic_node["camera_matrix"]["data"]) {
                std::vector<double> matrix_data;
                auto data = intrinsic_node["camera_matrix"]["data"];
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        matrix_data.push_back(data[i][j].as<double>());
                    }
                }
                camera_node["camera_matrix"]["data"] = matrix_data;
            }
            
            // Distortion coefficients - already flat
            if (intrinsic_node["distortion_coefficients"] && intrinsic_node["distortion_coefficients"]["data"]) {
                camera_node["distortion_coefficients"]["data"] = intrinsic_node["distortion_coefficients"]["data"];
            }
            
            // Image size
            if (intrinsic_node["image_size"]) {
                camera_node["image_width"] = intrinsic_node["image_size"]["width"];
                camera_node["image_height"] = intrinsic_node["image_size"]["height"];
            }
            
            // Parse extrinsic parameters if available
            if (extrinsic_yaml[camera_id]) {
                YAML::Node extrinsic_node = extrinsic_yaml[camera_id];
                
                // Extrinsic matrix - flatten nested array
                if (extrinsic_node["extrinsic_matrix"]) {
                    std::vector<double> extrinsic_data;
                    auto matrix = extrinsic_node["extrinsic_matrix"];
                    for (int i = 0; i < 4; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            extrinsic_data.push_back(matrix[i][j].as<double>());
                        }
                    }
                    camera_node["extrinsics"]["T"] = extrinsic_data;
                }
            }
            
            // Load calibration from formatted node
            if (!calibration_manager_->loadCalibrationFromNode(camera_id, camera_node)) {
                RCLCPP_WARN(this->get_logger(), "Failed to load calibration for camera: %s", camera_id.c_str());
            } else {
                RCLCPP_INFO(this->get_logger(), "Loaded calibration for camera: %s", camera_id.c_str());
            }
        }
    } catch (const YAML::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load calibration files: %s", e.what());
        throw std::runtime_error("Failed to load calibration files");
    }
    
    // Create projection engine
    projection_engine_ = std::make_shared<projection::ProjectionEngine>(
        calibration_manager_, projection_config_);
    
    if (!projection_engine_->initialize(camera_ids_)) {
        throw std::runtime_error("Failed to initialize projection engine");
    }
    
    RCLCPP_INFO(this->get_logger(), "Projection engine initialized with %zu cameras", 
               projection_engine_->getCameraIds().size());
}

void ProjectionDebugNode::setupROS() {
    // Initialize image transport
    image_transport_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    
    // Setup point cloud subscriber with Best Effort QoS for Ouster compatibility
    rclcpp::QoS qos(10);
    qos.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
    
    // Subscribe to a single LiDAR topic (raw or interpolated) specified in YAML
    cloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_topic_, qos,
        std::bind(&ProjectionDebugNode::pointCloudCallback, this, std::placeholders::_1));
    
    // Setup camera subscribers and visualization publishers
    for (const auto& camera_id : camera_ids_) {
        // Camera image subscriber
        std::string camera_topic = "/camera/" + camera_id + "/image_raw";
        
        CameraSubscriber camera_sub(camera_topic);
        camera_sub.subscriber = std::make_shared<image_transport::Subscriber>(
            image_transport_->subscribe(camera_topic, 10,
                [this, camera_id](const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
                    this->cameraCallback(camera_id, msg);
                }));
        
        camera_subscribers_.emplace(camera_id, std::move(camera_sub));
        
        // Visualization publisher (created only when debug visualization is enabled)
        std::string vis_topic;
        if (projection_config_.enable_debug_visualization) {
            vis_topic = output_topic_prefix_ + "/" + camera_id;
            visualization_publishers_[camera_id] = this->create_publisher<sensor_msgs::msg::Image>(vis_topic, rclcpp::QoS(10).reliable());
            RCLCPP_INFO(this->get_logger(), "Setup camera %s: %s -> %s", 
                       camera_id.c_str(), camera_topic.c_str(), vis_topic.c_str());
        } else {
            RCLCPP_INFO(this->get_logger(), "Setup camera %s: %s (debug visualization disabled)", 
                       camera_id.c_str(), camera_topic.c_str());
        }
    }
    
    // Statistics publisher
    statistics_publisher_ = this->create_publisher<std_msgs::msg::String>(
        output_topic_prefix_ + "/statistics", 10);
    
    // Statistics timer (publish every 5 seconds)
    statistics_timer_ = this->create_wall_timer(
        std::chrono::seconds(2),
        std::bind(&ProjectionDebugNode::statisticsTimerCallback, this));
    
    RCLCPP_INFO(this->get_logger(), "ROS setup complete");
}

void ProjectionDebugNode::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!is_initialized_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(processing_mutex_);
    
    try {
        processProjection(msg);
        processed_clouds_++;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error processing point cloud: %s", e.what());
    }
}

void ProjectionDebugNode::cameraCallback(const std::string& camera_id, 
                                        const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    auto it = camera_subscribers_.find(camera_id);
    if (it == camera_subscribers_.end()) {
        return;
    }
    
    try {
        auto cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
        
        std::lock_guard<std::mutex> lock(it->second.image_mutex);
        it->second.latest_image = cv_ptr->image.clone();
        it->second.latest_timestamp = msg->header.stamp;
        
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "CV bridge exception for camera %s: %s", 
                    camera_id.c_str(), e.what());
    }
}

void ProjectionDebugNode::processProjection(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Convert point cloud
    auto lidar_points = convertPointCloud(cloud_msg);
    if (lidar_points.empty()) {
        RCLCPP_WARN(this->get_logger(), "Received empty point cloud");
        return;
    }
    
    // Project to all cameras
    projection::ProjectionResult result;
    if (!projection_engine_->projectToAllCameras(lidar_points, result)) {
        RCLCPP_ERROR(this->get_logger(), "Projection failed");
        return;
    }
    
    // Create and publish visualization if enabled
    if (projection_config_.enable_debug_visualization) {
        publishVisualization(result, cloud_msg->header.stamp);
    }
    
    // Update statistics
    successful_projections_++;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double processing_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Update running average
    avg_processing_time_ms_ = (avg_processing_time_ms_ * (successful_projections_ - 1) + 
                              processing_time) / successful_projections_;
    
    // Log projection result
    if (successful_projections_ % 10 == 0) { // Log every 10th projection
        logProjectionResult(result);
    }

    // Publish updated statistics immediately for responsiveness
    publishStatistics();
}

void ProjectionDebugNode::publishVisualization(const projection::ProjectionResult& result,
                                              const rclcpp::Time& timestamp) {
    // Get synchronized camera images
    auto tolerance = rclcpp::Duration::from_nanoseconds(
        static_cast<int64_t>(time_tolerance_sec_ * 1e9));
    
    auto camera_images = getSynchronizedImages(timestamp, tolerance);
    
    // Create visualizations for each camera
    for (const auto& cam_projection : result.camera_projections) {
        const std::string& camera_id = cam_projection.camera_id;
        
        auto pub_it = visualization_publishers_.find(camera_id);
        if (pub_it == visualization_publishers_.end()) {
            continue;
        }
        
        cv::Mat base_image;
        auto img_it = camera_images.find(camera_id);
        if (img_it != camera_images.end()) {
            base_image = img_it->second;
        } else {
            // Create blank image if no camera image available
            auto camera_params = projection_engine_->getCameraParams(camera_id);
            if (camera_params) {
                base_image = cv::Mat::zeros(camera_params->image_size, CV_8UC3);
            } else {
                base_image = cv::Mat::zeros(480, 640, CV_8UC3);
            }
        }
        
        // Create visualization
        cv::Mat vis_image = projection_engine_->createDebugVisualization(
            result, {{camera_id, base_image}})[camera_id];
        
        // Add status overlay
        if (enable_status_overlay_) {
            vis_image = addStatusOverlay(vis_image, camera_id, cam_projection);
        }
        
        // Convert to ROS message and publish
        try {
            auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", vis_image).toImageMsg();
            msg->header.stamp = timestamp;
            msg->header.frame_id = camera_id;
            pub_it->second->publish(*msg);
            
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to publish visualization for %s: %s", 
                        camera_id.c_str(), e.what());
        }
    }
}

std::unordered_map<std::string, cv::Mat> ProjectionDebugNode::getSynchronizedImages(
    const rclcpp::Time& timestamp, 
    const rclcpp::Duration& tolerance) {
    
    std::unordered_map<std::string, cv::Mat> synchronized_images;
    
    for (const auto& pair : camera_subscribers_) {
        const std::string& camera_id = pair.first;
        const CameraSubscriber& camera_sub = pair.second;
        
        std::lock_guard<std::mutex> lock(camera_sub.image_mutex);
        
        if (!camera_sub.latest_image.empty()) {
            // Check time synchronization
            if (!enable_time_sync_ || 
                std::abs((timestamp - camera_sub.latest_timestamp).nanoseconds()) <= tolerance.nanoseconds()) {
                synchronized_images[camera_id] = camera_sub.latest_image.clone();
            } else {
                RCLCPP_DEBUG(this->get_logger(), 
                           "Camera %s image not synchronized (%.3f ms difference)", 
                           camera_id.c_str(),
                           std::abs((timestamp - camera_sub.latest_timestamp).nanoseconds()) / 1e6);
            }
        }
    }
    
    return synchronized_images;
}

std::vector<projection::LiDARPoint> ProjectionDebugNode::convertPointCloud(
    const sensor_msgs::msg::PointCloud2::SharedPtr& cloud_msg) {
    
    std::vector<projection::LiDARPoint> lidar_points;
    
    try {
        // Convert to PCL point cloud
        pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*cloud_msg, *pcl_cloud);
        
        // Convert to LiDAR points
        lidar_points.reserve(pcl_cloud->size());
        for (const auto& point : *pcl_cloud) {
            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) {
                lidar_points.emplace_back(point.x, point.y, point.z, point.intensity);
            }
        }
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error converting point cloud: %s", e.what());
    }
    
    return lidar_points;
}

cv::Mat ProjectionDebugNode::addStatusOverlay(const cv::Mat& image, 
                                             const std::string& camera_id,
                                             const projection::CameraProjection& projection) {
    cv::Mat overlay_image = image.clone();
    
    // Simple text overlay without background - only Camera ID and Avg time
    cv::Scalar text_color(0, 255, 0);  // Green text for visibility
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.6;
    int thickness = 2;
    
    // Camera ID
    std::string camera_text = "Camera: " + camera_id;
    cv::putText(overlay_image, camera_text, cv::Point(10, 30), 
                font, font_scale, text_color, thickness);
    
    // Performance info
    auto stats = projection_engine_->getStatistics();
    if (stats.total_projections > 0) {
        std::ostringstream perf_stream;
        perf_stream << "Avg time: " << std::fixed << std::setprecision(1) 
                   << stats.avg_processing_time_ms << " ms";
        cv::putText(overlay_image, perf_stream.str(), cv::Point(10, 60), 
                    font, font_scale, text_color, thickness);
    }
    
    return overlay_image;
}

void ProjectionDebugNode::statisticsTimerCallback() {
    publishStatistics();
}

void ProjectionDebugNode::publishStatistics() {
    // Expanded, vertical format with key metrics
    auto stats = projection_engine_->getStatistics();
    auto now = this->get_clock()->now();
    std_msgs::msg::String msg;
    std::ostringstream s;
    s << "\n";
    s << "=== PRISM Projection Debug Stats ===\n";
    s << "Processed clouds:         " << processed_clouds_ << "\n";
    s << "Successful projections:   " << successful_projections_ << "\n";
    if (processed_clouds_ > 0) {
        s << "Success rate:            " << std::fixed << std::setprecision(2)
          << (100.0 * successful_projections_ / processed_clouds_) << " %\n";
    }
    s << "Avg processing time:      " << std::fixed << std::setprecision(2)
      << avg_processing_time_ms_ << " ms\n";
    s << "Total points in:          " << stats.total_points_processed << "\n";
    s << "Total points projected:   " << stats.total_points_projected << "\n";
    s << "Projection success rate:  " << std::fixed << std::setprecision(2)
      << stats.getProjectionSuccessRate() << " %\n";
    msg.data = s.str();
    statistics_publisher_->publish(msg);
    // Periodic log
    if ((now - last_statistics_time_).seconds() >= 60) {
        RCLCPP_INFO(this->get_logger(), "%s", msg.data.c_str());
        last_statistics_time_ = now;
    }
}

void ProjectionDebugNode::logProjectionResult(const projection::ProjectionResult& result) {
    RCLCPP_DEBUG(this->get_logger(), 
                "Projection result: %zu -> %zu points in %.2f ms",
                result.input_count, 
                result.output_count,
                result.getProcessingTimeMs());
    
    for (const auto& cam_proj : result.camera_projections) {
        RCLCPP_DEBUG(this->get_logger(),
                    "  %s: %zu points, depth %.1f-%.1f m",
                    cam_proj.camera_id.c_str(),
                    cam_proj.projected_point_count,
                    cam_proj.min_depth,
                    cam_proj.max_depth);
    }
}

bool ProjectionDebugNode::allCamerasReady(const rclcpp::Time& timestamp, 
                                         const rclcpp::Duration& tolerance) const {
    for (const auto& pair : camera_subscribers_) {
        const CameraSubscriber& camera_sub = pair.second;
        
        std::lock_guard<std::mutex> lock(camera_sub.image_mutex);
        
        if (camera_sub.latest_image.empty()) {
            return false;
        }
        
        if (enable_time_sync_) {
            auto time_diff = std::abs((timestamp - camera_sub.latest_timestamp).nanoseconds());
            if (time_diff > tolerance.nanoseconds()) {
                return false;
            }
        }
    }
    
    return true;
}

} // namespace nodes
} // namespace prism

// ROS2 component registration
#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(prism::nodes::ProjectionDebugNode)

// Main function for standalone execution
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<prism::nodes::ProjectionDebugNode>();
    
    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node->get_logger(), "Exception in projection debug node: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}