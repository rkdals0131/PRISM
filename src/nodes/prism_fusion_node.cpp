#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// PRISM components
#include "prism/core/point_cloud_soa.hpp"
#include "prism/core/calibration_manager.hpp"
#include "prism/interpolation/interpolation_engine.hpp"
#include "prism/projection/projection_engine.hpp"
#include "prism/projection/color_extractor.hpp"
#include "prism/projection/multi_camera_fusion.hpp"

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include <atomic>
#include <std_msgs/msg/string.hpp>
#include <mutex>

namespace prism {

class PrismFusionNode : public rclcpp::Node {
public:
    PrismFusionNode() : Node("prism_fusion_node") {
        RCLCPP_INFO(get_logger(), "Initializing PRISM Fusion Node (Phase 5)");
        
        // Declare topic parameters
        declare_parameter("topics.lidar_input", "/ouster/points");
        declare_parameter("topics.camera_1_input", "/usb_cam_1/image_raw");
        declare_parameter("topics.camera_2_input", "/usb_cam_2/image_raw");
        declare_parameter("topics.colored_output", "/ouster/points/colored");
        declare_parameter("topics.debug_interpolated", "/prism/debug/interpolated");
        
        // Synchronization / rate control
        declare_parameter("synchronization.queue_size", 1);
        // Set to 0 to disable soft cap
        declare_parameter("synchronization.min_interval_ms", 0);
        // Use latest image cache instead of strict sync
        declare_parameter("synchronization.use_latest_image", true);
        declare_parameter("synchronization.image_freshness_ms", 150);

        // Declare camera array parameters - support for dynamic camera configuration
        declare_parameter("cameras", std::vector<std::string>());
        
        // Declare calibration parameters
        // Declare calibration path with package-relative default
        std::string default_calib_path;
        try {
            auto package_share_dir = ament_index_cpp::get_package_share_directory("prism");
            default_calib_path = package_share_dir + "/config";
        } catch (const std::exception& e) {
            // Fallback to relative path
            default_calib_path = "../share/prism/config";
        }
        declare_parameter("calibration.path", default_calib_path);
        declare_parameter("calibration.intrinsic_file", "multi_camera_intrinsic_calibration.yaml");
        declare_parameter("calibration.extrinsic_file", "multi_camera_extrinsic_calibration.yaml");
        
        // Declare interpolation parameters
        declare_parameter("interpolation.enabled", true);  // enable 2x interpolation by default
        declare_parameter("interpolation.scale_factor", 2.0); // vertical beams only
        declare_parameter("interpolation.input_channels", 32);
        declare_parameter("interpolation.spline_tension", 0.5);
        declare_parameter("interpolation.enable_discontinuity_detection", true);
        declare_parameter("interpolation.discontinuity_threshold", 0.5);
        declare_parameter("interpolation.grid_mode", true); // FILC-style row/column grid interpolation
        declare_parameter("interpolation.output_topic", "/ouster/points/interpolated");
        
        // Declare projection parameters
        declare_parameter("projection.enable_distortion_correction", true);
        declare_parameter("projection.enable_frustum_culling", true);
        declare_parameter("projection.min_depth", 0.5);
        declare_parameter("projection.max_depth", 100.0);
        declare_parameter("projection.parallel_cameras", false);
        declare_parameter("projection.margin_pixels", 5);
        
        // Declare extraction parameters
        declare_parameter("extraction.enable_subpixel", true);
        declare_parameter("extraction.confidence_threshold", 0.7);
        declare_parameter("extraction.blur_kernel_size", 0);
        declare_parameter("extraction.interpolation", "bilinear");
        
        // Declare fusion parameters
        declare_parameter("fusion.strategy", "weighted_average");
        declare_parameter("fusion.confidence_threshold", 0.5);
        declare_parameter("fusion.distance_weight_factor", 1.0);
        declare_parameter("fusion.enable_outlier_rejection", true);
        
        // Declare debug parameters
        declare_parameter("enable_debug", false);
        
        // Declare color mapping control parameter
        declare_parameter("enable_color_mapping", true);
        
        // Load configuration from parameters
        loadConfiguration();
        
        // Initialize core components
        initializeEngines();
        
        // NOTE: setupROS2Interface() must be called after shared_ptr is created
        // It will be called from initialize() method
        
        RCLCPP_INFO(get_logger(), "PRISM Fusion Node pre-initialized");
    }
    
    void initialize() {
        // This method must be called after shared_ptr is created
        setupROS2Interface();
        RCLCPP_INFO(get_logger(), "PRISM Fusion Node fully initialized");
    }
    
private:
    void loadConfiguration() {
        try {
            // Load sync and rate control configs
            sync_queue_size_ = static_cast<int>(get_parameter("synchronization.queue_size").as_int());
            min_interval_ms_ = static_cast<int>(get_parameter("synchronization.min_interval_ms").as_int());
            use_latest_image_ = get_parameter("synchronization.use_latest_image").as_bool();
            image_freshness_ms_ = static_cast<int>(get_parameter("synchronization.image_freshness_ms").as_int());

            // Load color mapping control
            enable_color_mapping_ = get_parameter("enable_color_mapping").as_bool();
            
            // Load interpolation config from parameters
            interpolation_enabled_ = get_parameter("interpolation.enabled").as_bool();
            const double scale_factor = get_parameter("interpolation.scale_factor").as_double();
            const int input_channels = static_cast<int>(get_parameter("interpolation.input_channels").as_int());
            interpolation_config_.spline_tension = 
                get_parameter("interpolation.spline_tension").as_double();
            interpolation_config_.enable_discontinuity_detection = 
                get_parameter("interpolation.enable_discontinuity_detection").as_bool();
            interpolation_config_.input_channels = static_cast<size_t>(std::max(1, input_channels));
            interpolation_config_.scale_factor = std::max(1.0, scale_factor);
            interpolation_scale_factor_ = interpolation_config_.scale_factor;
            grid_mode_ = get_parameter("interpolation.grid_mode").as_bool();
            discontinuity_threshold_ = get_parameter("interpolation.discontinuity_threshold").as_double();
            interpolated_output_topic_ = get_parameter("interpolation.output_topic").as_string();
            
            // Load projection config from parameters
            projection_config_.enable_distortion_correction = 
                get_parameter("projection.enable_distortion_correction").as_bool();
            projection_config_.enable_frustum_culling = 
                get_parameter("projection.enable_frustum_culling").as_bool();
            projection_config_.min_depth = 
                static_cast<float>(get_parameter("projection.min_depth").as_double());
            projection_config_.max_depth = 
                static_cast<float>(get_parameter("projection.max_depth").as_double());
            projection_config_.enable_parallel_processing = 
                get_parameter("projection.parallel_cameras").as_bool();
            projection_config_.margin_pixels = 
                static_cast<int>(get_parameter("projection.margin_pixels").as_int());
            
            // Load color extraction config from parameters
            extraction_config_.enable_subpixel = 
                get_parameter("extraction.enable_subpixel").as_bool();
            extraction_config_.confidence_threshold = 
                static_cast<float>(get_parameter("extraction.confidence_threshold").as_double());
            extraction_config_.blur_kernel_size = 
                static_cast<int>(get_parameter("extraction.blur_kernel_size").as_int());
            
            std::string interp = get_parameter("extraction.interpolation").as_string();
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
            
            // Load fusion config from parameters
            fusion_config_.confidence_threshold = 
                static_cast<float>(get_parameter("fusion.confidence_threshold").as_double());
            fusion_config_.distance_weight_factor = 
                static_cast<float>(get_parameter("fusion.distance_weight_factor").as_double());
            fusion_config_.enable_outlier_rejection = 
                get_parameter("fusion.enable_outlier_rejection").as_bool();
            
            std::string strategy = get_parameter("fusion.strategy").as_string();
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
            
            // Load camera configurations from parameters
            // Try to load from cameras array parameter first (from YAML structure)
            try {
                auto cameras_param = get_parameter("cameras");
                if (cameras_param.get_type() != rclcpp::ParameterType::PARAMETER_NOT_SET) {
                    // Cameras are loaded from YAML structure, handled by parameter server
                    // For now, fall back to hardcoded camera configuration
                    // TODO: Implement dynamic camera array parameter parsing
                }
            } catch (const std::exception& e) {
                RCLCPP_DEBUG(get_logger(), "No cameras array parameter found, using individual camera parameters");
            }
            
            // Load individual cameras (fallback or primary method)
            // Camera 1
            CameraConfig cam1;
            cam1.id = "camera_1";  // calibration 파일의 ID와 일치
            cam1.image_topic = get_parameter("topics.camera_1_input").as_string();
            camera_configs_.push_back(cam1);
            
            // Camera 2
            CameraConfig cam2;
            cam2.id = "camera_2";  // calibration 파일의 ID와 일치
            cam2.image_topic = get_parameter("topics.camera_2_input").as_string();
            camera_configs_.push_back(cam2);
            
            // Load calibration path from parameters
            calibration_path_ = get_parameter("calibration.path").as_string();
            
            // Load topic names from parameters
            lidar_input_topic_ = get_parameter("topics.lidar_input").as_string();
            colored_output_topic_ = get_parameter("topics.colored_output").as_string();
            debug_interpolated_topic_ = get_parameter("topics.debug_interpolated").as_string();
            
            RCLCPP_INFO(get_logger(), "Loaded configuration from ROS2 parameters");
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Failed to load configuration from parameters: %s", e.what());
            setupDefaultConfiguration();
        }
    }
    
    void setupDefaultConfiguration() {
        // Default two-camera configuration
        CameraConfig cam1;
        cam1.id = "camera_1";  // calibration 파일의 ID와 일치
        cam1.image_topic = "/usb_cam_1/image_raw";
        camera_configs_.push_back(cam1);
        
        CameraConfig cam2;
        cam2.id = "camera_2";  // calibration 파일의 ID와 일치
        cam2.image_topic = "/usb_cam_2/image_raw";
        camera_configs_.push_back(cam2);
        
        // Set default calibration path using package-relative path
        try {
            auto package_share_dir = ament_index_cpp::get_package_share_directory("prism");
            calibration_path_ = package_share_dir + "/config";
        } catch (const std::exception& e) {
            // Fallback to relative path
            calibration_path_ = "../share/prism/config";
        }
        
        // Set default topic names
        lidar_input_topic_ = "/ouster/points";
        colored_output_topic_ = "/ouster/points/colored";
        debug_interpolated_topic_ = "/prism/debug/interpolated";
        
        // Set default configuration values
        interpolation_config_.spline_tension = 0.5f;
        interpolation_config_.enable_discontinuity_detection = true;
        
        projection_config_.enable_distortion_correction = true;
        projection_config_.enable_frustum_culling = true;
        projection_config_.min_depth = 0.5f;
        projection_config_.max_depth = 100.0f;
        
        extraction_config_.enable_subpixel = true;
        extraction_config_.confidence_threshold = 0.7f;
        extraction_config_.blur_kernel_size = 0;
        extraction_config_.interpolation = projection::ColorExtractor::InterpolationMethod::BILINEAR;
        
        fusion_config_.strategy = projection::MultiCameraFusion::FusionStrategy::WEIGHTED_AVERAGE;
        fusion_config_.confidence_threshold = 0.5f;
        fusion_config_.distance_weight_factor = 1.0f;
        fusion_config_.enable_outlier_rejection = true;
    }
    
    void initializeEngines() {
        // Initialize calibration manager
        calibration_manager_ = std::make_shared<core::CalibrationManager>();
        
        // Load CALICO-style multi-camera calibration (intrinsics + extrinsics)
        try {
            const std::string intrinsic_file = calibration_path_ + "/multi_camera_intrinsic_calibration.yaml";
            const std::string extrinsic_file = calibration_path_ + "/multi_camera_extrinsic_calibration.yaml";
            
            if (std::filesystem::exists(intrinsic_file)) {
                YAML::Node intrinsic_yaml = YAML::LoadFile(intrinsic_file);
                YAML::Node extrinsic_yaml;
                if (std::filesystem::exists(extrinsic_file)) {
                    extrinsic_yaml = YAML::LoadFile(extrinsic_file);
                }
                
                for (const auto& cam_config : camera_configs_) {
                    const std::string& camera_id = cam_config.id;
                    
                    if (!intrinsic_yaml[camera_id]) {
                        RCLCPP_WARN(get_logger(), "Camera %s not found in intrinsic calibration file", camera_id.c_str());
                        continue;
                    }
                    
                    YAML::Node camera_node;
                    YAML::Node intrinsic_node = intrinsic_yaml[camera_id];
                    
                    // Camera matrix
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
                    
                    // Distortion coefficients
                    if (intrinsic_node["distortion_coefficients"] && intrinsic_node["distortion_coefficients"]["data"]) {
                        camera_node["distortion_coefficients"]["data"] = intrinsic_node["distortion_coefficients"]["data"];
                    }
                    
                    // Image size
                    if (intrinsic_node["image_size"]) {
                        camera_node["image_width"] = intrinsic_node["image_size"]["width"];
                        camera_node["image_height"] = intrinsic_node["image_size"]["height"];
                    }
                    
                    // Extrinsics (optional)
                    if (extrinsic_yaml && extrinsic_yaml[camera_id] && extrinsic_yaml[camera_id]["extrinsic_matrix"]) {
                        std::vector<double> extrinsic_data;
                        auto matrix = extrinsic_yaml[camera_id]["extrinsic_matrix"];
                        for (int i = 0; i < 4; ++i) {
                            for (int j = 0; j < 4; ++j) {
                                extrinsic_data.push_back(matrix[i][j].as<double>());
                            }
                        }
                        camera_node["extrinsics"]["T"] = extrinsic_data;
                    }
                    
                    if (!calibration_manager_->loadCalibrationFromNode(camera_id, camera_node)) {
                        RCLCPP_WARN(get_logger(), "Failed to load calibration for camera: %s", camera_id.c_str());
                    } else {
                        RCLCPP_INFO(get_logger(), "Loaded calibration for camera: %s", camera_id.c_str());
                    }
                }
            } else {
                RCLCPP_WARN(get_logger(), "Intrinsic calibration file not found: %s", intrinsic_file.c_str());
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Failed to load multi-camera calibration: %s", e.what());
        }
        
        // Initialize engines
        interpolation_engine_ = std::make_unique<interpolation::InterpolationEngine>();
        // Configure interpolation engine with loaded config (fatal if invalid)
        try {
            interpolation_engine_->configure(interpolation_config_);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Failed to configure interpolation engine: %s", e.what());
            interpolation_enabled_ = false; // disable interpolation to keep pipeline alive
        }
        projection_engine_ = std::make_unique<projection::ProjectionEngine>(calibration_manager_);
        // Apply projection configuration (e.g., disable parallel if requested)
        projection_engine_->updateConfig(projection_config_);
        color_extractor_ = std::make_unique<projection::ColorExtractor>();
        multi_camera_fusion_ = std::make_unique<projection::MultiCameraFusion>();
        
        // Initialize projection engine with configured camera IDs
        std::vector<std::string> camera_ids;
        camera_ids.reserve(camera_configs_.size());
        for (const auto& cam : camera_configs_) {
            camera_ids.push_back(cam.id);
        }
        if (!camera_ids.empty()) {
            if (!projection_engine_->initialize(camera_ids)) {
                RCLCPP_WARN(get_logger(), "Projection engine initialization failed (check calibrations)");
            }
        }
        
        RCLCPP_INFO(get_logger(), "All engines initialized");
    }
    
    void setupROS2Interface() {
        // Create QoS profile for sensor data: best-effort, small queue to avoid backlog
        auto sensor_qos = rclcpp::SensorDataQoS();
        sensor_qos.keep_last(1);
        
        if (use_latest_image_) {
            // Independent subscriptions: cache latest images per camera, LiDAR drives pipeline
            for (const auto& cam_config : camera_configs_) {
                RCLCPP_INFO(get_logger(), "Subscribing to camera %s on topic: %s", 
                           cam_config.id.c_str(), cam_config.image_topic.c_str());
                auto sub = create_subscription<sensor_msgs::msg::Image>(
                    cam_config.image_topic, sensor_qos,
                    [this, cam_id=cam_config.id](const sensor_msgs::msg::Image::SharedPtr msg){
                        std::lock_guard<std::mutex> lk(image_cache_mutex_);
                        image_cache_[cam_id] = msg;
                    }
                );
                image_plain_subs_.push_back(sub);
            }
            lidar_plain_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
                lidar_input_topic_, sensor_qos,
                std::bind(&PrismFusionNode::lidarCallback, this, std::placeholders::_1)
            );
        } else {
            // Original synchronizer path
            for (const auto& cam_config : camera_configs_) {
                auto sub = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
                    shared_from_this(), cam_config.image_topic, sensor_qos.get_rmw_qos_profile());
                image_subscribers_[cam_config.id] = sub;
                RCLCPP_INFO(get_logger(), "Subscribing to camera %s on topic: %s", 
                           cam_config.id.c_str(), cam_config.image_topic.c_str());
            }
            lidar_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
                shared_from_this(), lidar_input_topic_, sensor_qos.get_rmw_qos_profile());
            if (camera_configs_.size() == 2) {
                typedef message_filters::sync_policies::ApproximateTime<
                    sensor_msgs::msg::PointCloud2,
                    sensor_msgs::msg::Image,
                    sensor_msgs::msg::Image> SyncPolicy;
                sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
                    SyncPolicy(static_cast<uint32_t>(std::max(1, sync_queue_size_))),
                    *lidar_sub_,
                    *image_subscribers_[camera_configs_[0].id],
                    *image_subscribers_[camera_configs_[1].id]
                );
                sync_->registerCallback(
                    std::bind(&PrismFusionNode::syncCallback, this, 
                             std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
            }
        }
        
        // Create publishers with SensorData QoS to avoid backpressure
        auto pub_qos = rclcpp::SensorDataQoS();
        pub_qos.keep_last(1);
        pub_qos.best_effort();
        colored_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
            colored_output_topic_, pub_qos);
        
        // Create publisher for interpolated point cloud (if enabled)
        if (interpolation_enabled_) {
            interpolated_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
                interpolated_output_topic_, pub_qos);
        }

        // Debug statistics publisher (shared topic with projection debug)
        debug_stats_pub_ = create_publisher<std_msgs::msg::String>(
            "/prism/projection_debug/statistics", 10);
        
        // Debug publishers if enabled
        if (get_parameter("enable_debug").as_bool()) {
            debug_interpolated_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
                debug_interpolated_topic_, pub_qos);
        }
    }
    
    void syncCallback(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr& img1,
        const sensor_msgs::msg::Image::ConstSharedPtr& img2) {
        // Drop frames if previous processing is ongoing
        if (processing_.exchange(true)) {
            return;
        }
        struct ProcessingGuard {
            std::atomic<bool>& flag;
            explicit ProcessingGuard(std::atomic<bool>& f) : flag(f) {}
            ~ProcessingGuard() { flag.store(false); }
        } guard(processing_);

        // Enforce minimum processing interval (0 disables)
        const auto now_time = this->get_clock()->now();
        if (min_interval_ms_ > 0 && last_processed_time_.nanoseconds() != 0) {
            auto dt_ns = (now_time - last_processed_time_).nanoseconds();
            if (dt_ns < static_cast<int64_t>(min_interval_ms_) * 1000000LL) {
                return; // too soon; drop
            }
        }
        last_processed_time_ = now_time;

        auto t_start = std::chrono::high_resolution_clock::now();
        // Full cycle time since last publish (includes waiting)
        double cycle_ms = 0.0;
        if (last_publish_time_.nanoseconds() > 0) {
            cycle_ms = (now_time - last_publish_time_).nanoseconds() / 1e6;
        }
        
        // Convert ROS PointCloud2 to PCL (preserve intensity)
        pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *input_cloud);
        auto t_after_ros_to_pcl = std::chrono::high_resolution_clock::now();
        
        // Step 1: Interpolation (Phase 2)
        if (input_cloud->empty()) {
            RCLCPP_ERROR(get_logger(), "Interpolation failed: Invalid input point cloud (empty)");
            return;
        }
        
        core::PointCloudSoA soa_cloud;
        soa_cloud.reserve(input_cloud->size());
        for (const auto& p : *input_cloud) {
            soa_cloud.addPoint(p.x, p.y, p.z, p.intensity);
        }
        
        core::PointCloudSoA* interpolated = nullptr;
        // Keep interpolation result alive for the entire callback scope
        interpolation::InterpolationResult interp_result;
        if (interpolation_enabled_) {
            if (grid_mode_) {
                // FILC-style grid interpolation in-node for debug parity
                const int input_height = static_cast<int>(interpolation_config_.input_channels);
                const int input_width = 1024;
                const int output_height = static_cast<int>(std::round(input_height * interpolation_scale_factor_));
                // Build a temporary organized grid by nearest neighbor per column
                std::vector<float> azimuths(soa_cloud.size());
                for (size_t i = 0; i < soa_cloud.size(); ++i) {
                    azimuths[i] = std::atan2(soa_cloud.y[i], soa_cloud.x[i]);
                }
                std::vector<std::vector<size_t>> by_ring(input_height);
                // separate by ring (approx if ring missing)
                for (size_t i = 0; i < soa_cloud.size(); ++i) {
                    uint16_t ring = 0;
                    if (soa_cloud.hasRing() && i < soa_cloud.ring.size()) ring = soa_cloud.ring[i];
                    else {
                        float elev = std::atan2(soa_cloud.z[i], std::hypot(soa_cloud.x[i], soa_cloud.y[i]));
                        float norm = (elev + static_cast<float>(M_PI/6)) / static_cast<float>(M_PI/3);
                        ring = static_cast<uint16_t>(std::min<double>(input_height - 1, std::max<double>(0, std::round(norm * (input_height - 1)))));
                    }
                    if (ring < input_height) by_ring[ring].push_back(i);
                }
                std::vector<std::pair<float,size_t>> anchors; anchors.reserve(input_width);
                for (int c = 0; c < input_width; ++c) {
                    float a = -static_cast<float>(M_PI) + static_cast<float>(2*M_PI) * (static_cast<float>(c)/input_width);
                    anchors.emplace_back(a, static_cast<size_t>(c));
                }
                auto wrap = [](float a){ while(a>=static_cast<float>(M_PI)) a-=static_cast<float>(2*M_PI); while(a<-static_cast<float>(M_PI)) a+=static_cast<float>(2*M_PI); return a; };
                // grid low
                std::vector<std::vector<size_t>> grid_idx(input_height, std::vector<size_t>(input_width, static_cast<size_t>(-1)));
                for (int r = 0; r < input_height; ++r) {
                    auto& idxs = by_ring[r];
                    if (idxs.empty()) continue;
                    std::vector<std::pair<float,size_t>> by_az; by_az.reserve(idxs.size());
                    for (size_t idx: idxs) by_az.emplace_back(azimuths[idx], idx);
                    std::sort(by_az.begin(), by_az.end(), [](auto&a,auto&b){return a.first<b.first;});
                    size_t j=0;
                    for (int c = 0; c < input_width; ++c) {
                        float tgt = anchors[c].first;
                        while (j+1<by_az.size() && std::abs(wrap(by_az[j+1].first - tgt)) <= std::abs(wrap(by_az[j].first - tgt))) ++j;
                        grid_idx[r][c] = by_az[j].second;
                    }
                }
                // interpolate rows into a fixed-size grid (row-major)
                const size_t out_size = static_cast<size_t>(output_height) * input_width;
                std::vector<float> gx(out_size, 0.0f), gy(out_size, 0.0f), gz(out_size, 0.0f), gi(out_size, 0.0f);
                auto set_cell = [&](int rr, int cc, float x, float y, float z, float inten){
                    if (rr < 0 || rr >= output_height) return; if (cc < 0 || cc >= input_width) return;
                    size_t idx = static_cast<size_t>(rr) * input_width + cc;
                    gx[idx]=x; gy[idx]=y; gz[idx]=z; gi[idx]=inten;
                };
                double thresh = discontinuity_threshold_;
                for (int r = 0; r < input_height-1; ++r) {
                    for (int c = 0; c < input_width; ++c) {
                        size_t i1 = grid_idx[r][c], i2 = grid_idx[r+1][c];
                        if (i1==(size_t)-1 || i2==(size_t)-1) continue;
                        // copy base row into target row slot
                        int base_out_row = static_cast<int>(std::round(r * interpolation_scale_factor_));
                        set_cell(base_out_row, c, soa_cloud.x[i1], soa_cloud.y[i1], soa_cloud.z[i1], soa_cloud.intensity[i1]);
                        int inserts = static_cast<int>(std::max(1.0, interpolation_scale_factor_)) - 1;
                        float r1 = std::hypot(soa_cloud.x[i1], std::hypot(soa_cloud.y[i1], soa_cloud.z[i1]));
                        float r2 = std::hypot(soa_cloud.x[i2], std::hypot(soa_cloud.y[i2], soa_cloud.z[i2]));
                        bool discontinuous = std::abs(r2-r1) > thresh;
                        for (int k=1;k<=inserts;++k){
                            float t = static_cast<float>(k)/(inserts+1);
                            if (discontinuous) {
                                const size_t is = (t<0.5f)? i1:i2;
                                set_cell(base_out_row + k, c, soa_cloud.x[is], soa_cloud.y[is], soa_cloud.z[is], soa_cloud.intensity[is]);
                            } else {
                                float x = (1.0f-t)*soa_cloud.x[i1] + t*soa_cloud.x[i2];
                                float y = (1.0f-t)*soa_cloud.y[i1] + t*soa_cloud.y[i2];
                                float z = (1.0f-t)*soa_cloud.z[i1] + t*soa_cloud.z[i2];
                                float inten = (1.0f-t)*soa_cloud.intensity[i1] + t*soa_cloud.intensity[i2];
                                set_cell(base_out_row + k, c, x,y,z,inten);
                            }
                        }
                    }
                }
                // last input row copy
                int last_in = input_height - 1;
                int last_out = static_cast<int>(std::round(last_in * interpolation_scale_factor_));
                for (int c=0;c<input_width;++c){
                    size_t iL = grid_idx[last_in][c]; if (iL==(size_t)-1) continue;
                    set_cell(last_out, c, soa_cloud.x[iL], soa_cloud.y[iL], soa_cloud.z[iL], soa_cloud.intensity[iL]);
                }
                // flatten to SoA
                core::PointCloudSoA grid_out; grid_out.reserve(out_size);
                for (size_t idx=0; idx<out_size; ++idx) {
                    grid_out.addPoint(gx[idx], gy[idx], gz[idx], gi[idx]);
                }
                // use grid_out as interpolated
                interp_result.success = true;
                interp_result.interpolated_cloud = core::PooledPtr<core::PointCloudSoA>(new core::PointCloudSoA(std::move(grid_out)), core::PoolDeleter<core::PointCloudSoA>{nullptr});
                interpolated = interp_result.interpolated_cloud.get();
                // publish
                if (interpolated_cloud_pub_) {
                    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_interpolated(new pcl::PointCloud<pcl::PointXYZI>);
                    pcl_interpolated->reserve(interpolated->size());
                    for (size_t i = 0; i < interpolated->size(); ++i) {
                        pcl::PointXYZI p{interpolated->x[i], interpolated->y[i], interpolated->z[i], interpolated->intensity[i]};
                        pcl_interpolated->push_back(p);
                    }
                    sensor_msgs::msg::PointCloud2 interp_msg; pcl::toROSMsg(*pcl_interpolated, interp_msg);
                    interp_msg.header = lidar_msg->header; interpolated_cloud_pub_->publish(interp_msg);
                }
            } else {
                interp_result = interpolation_engine_->interpolate(soa_cloud);
                if (!interp_result.success) {
                    RCLCPP_ERROR(get_logger(), "Interpolation failed: %s", interp_result.error_message.c_str());
                    return;
                }
                interpolated = interp_result.interpolated_cloud.get();
                if (interpolated_cloud_pub_) {
                    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_interpolated(new pcl::PointCloud<pcl::PointXYZI>);
                    pcl_interpolated->reserve(interpolated->size());
                    for (size_t i = 0; i < interpolated->size(); ++i) {
                        pcl::PointXYZI p{interpolated->x[i], interpolated->y[i], interpolated->z[i], interpolated->intensity[i]};
                        pcl_interpolated->push_back(p);
                    }
                    sensor_msgs::msg::PointCloud2 interp_msg; pcl::toROSMsg(*pcl_interpolated, interp_msg);
                    interp_msg.header = lidar_msg->header; interpolated_cloud_pub_->publish(interp_msg);
                }
            }
        } else {
            // No interpolation: operate directly on input SoA
            interpolated = &soa_cloud;
        }
        auto t_after_interpolation = std::chrono::high_resolution_clock::now();
        
        // Variables for timing (declare outside to avoid scope issues)
        std::chrono::high_resolution_clock::time_point t_after_projection;
        std::chrono::high_resolution_clock::time_point t_after_fusion;
        size_t successful_fusions = 0;
        
        // Check if color mapping is enabled
        if (enable_color_mapping_) {
            // Step 2: Projection (Phase 3)
            std::map<std::string, cv::Mat> camera_images;
            cv_bridge::CvImagePtr cv_img1, cv_img2;
            try {
                cv_img1 = cv_bridge::toCvCopy(img1, sensor_msgs::image_encodings::BGR8);
                cv_img2 = cv_bridge::toCvCopy(img2, sensor_msgs::image_encodings::BGR8);
                camera_images[camera_configs_[0].id] = cv_img1->image;
                camera_images[camera_configs_[1].id] = cv_img2->image;
            } catch (cv_bridge::Exception& e) {
                RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }
            
            // Convert PointCloudSoA to LiDARPoint vector (preserve original indices)
            std::vector<projection::LiDARPoint> lidar_points;
            lidar_points.reserve(interpolated->size());
            for (size_t i = 0; i < interpolated->size(); ++i) {
                lidar_points.emplace_back(
                    interpolated->x[i], 
                    interpolated->y[i], 
                    interpolated->z[i],
                    1.0f,  // Default intensity
                    i      // Preserve original index
                );
            }
            
            projection::ProjectionResult projection_results;
            if (!projection_engine_->projectToAllCameras(lidar_points, projection_results)) {
                RCLCPP_ERROR(get_logger(), "Projection failed");
                return;
            }
            t_after_projection = std::chrono::high_resolution_clock::now();
            
            // Step 3: Color Extraction (Phase 4)
            std::map<std::string, projection::ColorExtractor::ColorExtractionResult> extraction_results;
            std::map<std::string, std::vector<float>> pixel_distances;
            
            const size_t total_points = interpolated->size();
            for (const auto& cam_proj : projection_results.camera_projections) {
                // Convert PixelPoint to cv::Point2f and preserve original point indices
                std::vector<cv::Point2f> pixel_coordinates;
                std::vector<size_t> point_indices;
                pixel_coordinates.reserve(cam_proj.projected_points.size());
                point_indices.reserve(cam_proj.projected_points.size());
                
                for (size_t i = 0; i < cam_proj.projected_points.size(); ++i) {
                    const auto& pixel_point = cam_proj.projected_points[i];
                    pixel_coordinates.emplace_back(pixel_point.u, pixel_point.v);
                    // Use original LiDAR point index
                    point_indices.push_back(pixel_point.original_index);
                }
                
                // Extract colors
                auto sparse_result = color_extractor_->extractColorsWithIndices(
                    camera_images[cam_proj.camera_id], 
                    pixel_coordinates,
                    point_indices,
                    extraction_config_);
                
                // Remap to dense per-original-index arrays for fusion
                projection::ColorExtractor::ColorExtractionResult dense_result;
                dense_result.colors.assign(total_points, cv::Vec3b(0, 0, 0));
                dense_result.confidence_scores.assign(total_points, 0.0f);
                dense_result.valid_extractions.assign(total_points, false);
                dense_result.output_count = total_points;
                dense_result.valid_colors = 0;
                
                // sparse_result: vectors are aligned to pixel_coordinates; point_indices[k] is original index
                for (size_t k = 0; k < point_indices.size() && k < sparse_result.colors.size(); ++k) {
                    const size_t orig_idx = point_indices[k];
                    if (orig_idx >= total_points) { continue; }
                    dense_result.colors[orig_idx] = sparse_result.colors[k];
                    dense_result.confidence_scores[orig_idx] = sparse_result.confidence_scores[k];
                    dense_result.valid_extractions[orig_idx] = sparse_result.valid_extractions[k];
                    if (sparse_result.valid_extractions[k]) {
                        dense_result.valid_colors++;
                    }
                }
                extraction_results[cam_proj.camera_id] = std::move(dense_result);
                
                // Calculate pixel distances from optical center
                auto calibration = calibration_manager_->getCalibration(cam_proj.camera_id);
                if (calibration) {
                    double cx = calibration->K(0, 2);
                    double cy = calibration->K(1, 2);
                    
                    std::vector<float> dense_distances(total_points, 0.0f);
                    for (size_t k = 0; k < point_indices.size() && k < pixel_coordinates.size(); ++k) {
                        const size_t orig_idx = point_indices[k];
                        if (orig_idx >= total_points) { continue; }
                        const auto& pixel = pixel_coordinates[k];
                        float dx = pixel.x - static_cast<float>(cx);
                        float dy = pixel.y - static_cast<float>(cy);
                        dense_distances[orig_idx] = std::sqrt(dx*dx + dy*dy);
                    }
                    pixel_distances[cam_proj.camera_id] = std::move(dense_distances);
                }
            }
            
            // Step 4: Multi-Camera Fusion (Phase 4)
            std::vector<size_t> all_indices;
            for (size_t i = 0; i < interpolated->size(); ++i) {
                all_indices.push_back(i);
            }
            
            auto fusion_result = multi_camera_fusion_->fuseColorsWithDistances(
                extraction_results, pixel_distances, all_indices, fusion_config_);
            t_after_fusion = std::chrono::high_resolution_clock::now();
            successful_fusions = fusion_result.successful_fusions;
            
            // Step 5: Create colored point cloud
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            colored_cloud->reserve(interpolated->size());
            
            for (size_t i = 0; i < interpolated->size(); ++i) {
                pcl::PointXYZRGB point;
                point.x = interpolated->x[i];
                point.y = interpolated->y[i];
                point.z = interpolated->z[i];
                
                if (i < fusion_result.fused_colors.size() && 
                    fusion_result.fusion_confidence[i] >= fusion_config_.confidence_threshold) {
                    const auto& rgb = fusion_result.fused_colors[i];
                    // Colors are stored as RGB in extraction/fusion results
                    point.r = rgb[0];
                    point.g = rgb[1];
                    point.b = rgb[2];
                } else {
                    // Default gray for uncolored points
                    point.r = point.g = point.b = 128;
                }
                
                colored_cloud->push_back(point);
            }
            
            // Publish colored point cloud
            sensor_msgs::msg::PointCloud2 output_msg;
            pcl::toROSMsg(*colored_cloud, output_msg);
            output_msg.header = lidar_msg->header;
            colored_cloud_pub_->publish(output_msg);
        } else {
            // Interpolation-only mode: publish interpolated cloud without colors
            pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            output_cloud->reserve(interpolated->size());
            for (size_t i = 0; i < interpolated->size(); ++i) {
                pcl::PointXYZI point;
                point.x = interpolated->x[i];
                point.y = interpolated->y[i];
                point.z = interpolated->z[i];
                point.intensity = interpolated->intensity[i];
                output_cloud->push_back(point);
            }
            
            sensor_msgs::msg::PointCloud2 output_msg; 
            pcl::toROSMsg(*output_cloud, output_msg);
            output_msg.header = lidar_msg->header; 
            colored_cloud_pub_->publish(output_msg);
            
            RCLCPP_INFO_ONCE(get_logger(), 
                "PRISM running in interpolation-only mode (color mapping disabled) [sync]");
        }
        auto t_after_publish = std::chrono::high_resolution_clock::now();
        
        // Calculate and log performance metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - t_start);
        
        // Log performance based on mode
        if (enable_color_mapping_) {
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                "PRISM Fusion Pipeline: %ld ms total | "
                "Points: %zu | Colored: %zu (%.1f%%)",
                duration.count(), 
                interpolated->size(),
                successful_fusions,
                100.0f * successful_fusions / std::max(size_t(1), interpolated->size()));
        } else {
            RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                "PRISM Interpolation-only: %ld ms | Points: %zu (FILC-like mode)",
                duration.count(),
                interpolated->size());
        }

        // Publish detailed per-stage timings
        if (debug_stats_pub_) {
            auto ms = [](auto dt){ return std::chrono::duration_cast<std::chrono::milliseconds>(dt).count(); };
            std_msgs::msg::String msg;
            std::ostringstream s;
            s << "\n=== PRISM Fusion Pipeline Timings ===\n";
            s << "ROS->PCL:             " << ms(t_after_ros_to_pcl - t_start) << " ms\n";
            // SoA build included in interpolation stage (reported below)
            s << "Interpolation:         " << ms(t_after_interpolation - t_after_ros_to_pcl) << " ms\n";
            if (enable_color_mapping_) {
                s << "Projection:            " << ms(t_after_projection - t_after_interpolation) << " ms\n";
                s << "Color extraction:      " << ms(t_after_fusion - t_after_projection) << " ms\n";
                s << "Fusion:                " << ms(t_after_fusion - t_after_projection) << " ms (incl. extraction)\n";
                s << "Publish colored cloud: " << ms(t_after_publish - t_after_fusion) << " ms\n";
            } else {
                s << "Mode:                  Interpolation-only (color mapping disabled)\n";
                s << "Publish cloud:         " << ms(t_after_publish - t_after_interpolation) << " ms\n";
            }
            s << "Total:                 " << duration.count() << " ms\n";
            if (cycle_ms > 0.0) {
                s << "Cycle (prev->this):   " << std::fixed << std::setprecision(2) << cycle_ms << " ms\n";
            }
            msg.data = s.str();
            debug_stats_pub_->publish(msg);
        }
        last_publish_time_ = this->get_clock()->now();
    }

    void lidarCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg) {
        // Drop frames if previous processing is ongoing
        if (processing_.exchange(true)) {
            return;
        }
        struct ProcessingGuard {
            std::atomic<bool>& flag;
            explicit ProcessingGuard(std::atomic<bool>& f) : flag(f) {}
            ~ProcessingGuard() { flag.store(false); }
        } guard(processing_);

        // Enforce minimum processing interval (0 disables)
        const auto now_time = this->get_clock()->now();
        if (min_interval_ms_ > 0 && last_processed_time_.nanoseconds() != 0) {
            auto dt_ns = (now_time - last_processed_time_).nanoseconds();
            if (dt_ns < static_cast<int64_t>(min_interval_ms_) * 1000000LL) {
                return; // too soon; drop
            }
        }
        last_processed_time_ = now_time;

        // Build latest camera images map from cache
        std::map<std::string, cv::Mat> camera_images;
        {
            std::lock_guard<std::mutex> lk(image_cache_mutex_);
            for (const auto& cam : camera_configs_) {
                auto it = image_cache_.find(cam.id);
                if (it == image_cache_.end()) return; // wait until all have an image once
                // freshness check (best-effort)
                if (image_freshness_ms_ > 0) {
                    auto age_ns = (now_time - it->second->header.stamp).nanoseconds();
                    (void)age_ns; // informational only
                }
                try {
                    auto cv_ptr = cv_bridge::toCvShare(it->second, sensor_msgs::image_encodings::BGR8);
                    camera_images[cam.id] = cv_ptr->image;
                } catch (...) { return; }
            }
        }
        // Process using prepared images
        processWithImages(lidar_msg, camera_images);
    }

    void processWithImages(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg,
                           const std::map<std::string, cv::Mat>& camera_images) {
        auto t_start = std::chrono::high_resolution_clock::now();
        double cycle_ms = 0.0;
        if (last_publish_time_.nanoseconds() > 0) {
            cycle_ms = (this->get_clock()->now() - last_publish_time_).nanoseconds() / 1e6;
        }

        // Convert ROS PointCloud2 to PCL (preserve intensity)
        pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*lidar_msg, *input_cloud);

        // Interpolation (optional)
        if (input_cloud->empty()) {
            RCLCPP_ERROR(get_logger(), "Interpolation failed: Invalid input point cloud (empty)");
            return;
        }
        core::PointCloudSoA soa_cloud; soa_cloud.reserve(input_cloud->size());
        for (const auto& p : *input_cloud) {
            soa_cloud.addPoint(p.x, p.y, p.z, p.intensity);
        }
        core::PointCloudSoA* interpolated = nullptr;
        interpolation::InterpolationResult interp_result;
        if (interpolation_enabled_) {
            if (grid_mode_) {
                // FILC-style row/column grid interpolation (1024 x input_channels)
                const int input_height = static_cast<int>(interpolation_config_.input_channels);
                const int input_width = 1024;
                const int output_height = static_cast<int>(std::round(input_height * interpolation_scale_factor_));

                // Compute azimuth per point
                std::vector<float> azimuths(soa_cloud.size());
                for (size_t i = 0; i < soa_cloud.size(); ++i) {
                    azimuths[i] = std::atan2(soa_cloud.y[i], soa_cloud.x[i]);
                }

                // Group indices by ring; approximate if ring channel not present
                std::vector<std::vector<size_t>> by_ring(input_height);
                for (size_t i = 0; i < soa_cloud.size(); ++i) {
                    uint16_t ring = 0;
                    if (soa_cloud.hasRing() && i < soa_cloud.ring.size()) {
                        ring = soa_cloud.ring[i];
                    } else {
                        float elev = std::atan2(soa_cloud.z[i], std::hypot(soa_cloud.x[i], soa_cloud.y[i]));
                        float norm = (elev + static_cast<float>(M_PI/6)) / static_cast<float>(M_PI/3);
                        ring = static_cast<uint16_t>(std::min<double>(input_height - 1, std::max<double>(0, std::round(norm * (input_height - 1)))));
                    }
                    if (ring < input_height) by_ring[ring].push_back(i);
                }

                // Build azimuth anchors per column
                std::vector<std::pair<float,size_t>> anchors; anchors.reserve(input_width);
                for (int c = 0; c < input_width; ++c) {
                    float a = -static_cast<float>(M_PI) + static_cast<float>(2*M_PI) * (static_cast<float>(c)/input_width);
                    anchors.emplace_back(a, static_cast<size_t>(c));
                }
                auto wrap = [](float a){ while(a>=static_cast<float>(M_PI)) a-=static_cast<float>(2*M_PI); while(a<-static_cast<float>(M_PI)) a+=static_cast<float>(2*M_PI); return a; };

                // For each ring, assign nearest neighbor per column
                std::vector<std::vector<size_t>> grid_idx(input_height, std::vector<size_t>(input_width, static_cast<size_t>(-1)));
                for (int r = 0; r < input_height; ++r) {
                    auto& idxs = by_ring[r];
                    if (idxs.empty()) continue;
                    std::vector<std::pair<float,size_t>> by_az; by_az.reserve(idxs.size());
                    for (size_t idx: idxs) by_az.emplace_back(azimuths[idx], idx);
                    std::sort(by_az.begin(), by_az.end(), [](auto&a,auto&b){return a.first<b.first;});
                    size_t j=0;
                    for (int c = 0; c < input_width; ++c) {
                        float tgt = anchors[c].first;
                        while (j+1<by_az.size() && std::abs(wrap(by_az[j+1].first - tgt)) <= std::abs(wrap(by_az[j].first - tgt))) ++j;
                        grid_idx[r][c] = by_az[j].second;
                    }
                }

                // Allocate output grid
                const size_t out_size = static_cast<size_t>(output_height) * input_width;
                std::vector<float> gx(out_size, 0.0f), gy(out_size, 0.0f), gz(out_size, 0.0f), gi(out_size, 0.0f);
                auto set_cell = [&](int rr, int cc, float x, float y, float z, float inten){
                    if (rr < 0 || rr >= output_height) return; 
                    if (cc < 0 || cc >= input_width) return;
                    size_t idx = static_cast<size_t>(rr) * input_width + cc;
                    gx[idx]=x; gy[idx]=y; gz[idx]=z; gi[idx]=inten;
                };

                double thresh = discontinuity_threshold_;
                for (int r = 0; r < input_height-1; ++r) {
                    for (int c = 0; c < input_width; ++c) {
                        size_t i1 = grid_idx[r][c], i2 = grid_idx[r+1][c];
                        if (i1==(size_t)-1 || i2==(size_t)-1) continue;
                        // copy base row into target row slot
                        int base_out_row = static_cast<int>(std::round(r * interpolation_scale_factor_));
                        set_cell(base_out_row, c, soa_cloud.x[i1], soa_cloud.y[i1], soa_cloud.z[i1], soa_cloud.intensity[i1]);
                        int inserts = static_cast<int>(std::max(1.0, interpolation_scale_factor_)) - 1;
                        float r1 = std::hypot(soa_cloud.x[i1], std::hypot(soa_cloud.y[i1], soa_cloud.z[i1]));
                        float r2 = std::hypot(soa_cloud.x[i2], std::hypot(soa_cloud.y[i2], soa_cloud.z[i2]));
                        bool discontinuous = std::abs(r2-r1) > thresh;
                        for (int k=1;k<=inserts;++k){
                            float t = static_cast<float>(k)/(inserts+1);
                            if (discontinuous) {
                                const size_t is = (t<0.5f)? i1:i2;
                                set_cell(base_out_row + k, c, soa_cloud.x[is], soa_cloud.y[is], soa_cloud.z[is], soa_cloud.intensity[is]);
                            } else {
                                float x = (1.0f-t)*soa_cloud.x[i1] + t*soa_cloud.x[i2];
                                float y = (1.0f-t)*soa_cloud.y[i1] + t*soa_cloud.y[i2];
                                float z = (1.0f-t)*soa_cloud.z[i1] + t*soa_cloud.z[i2];
                                float inten = (1.0f-t)*soa_cloud.intensity[i1] + t*soa_cloud.intensity[i2];
                                set_cell(base_out_row + k, c, x,y,z,inten);
                            }
                        }
                    }
                }
                // last input row copy
                int last_in = input_height - 1;
                int last_out = static_cast<int>(std::round(last_in * interpolation_scale_factor_));
                for (int c=0;c<input_width;++c){
                    size_t iL = grid_idx[last_in][c]; if (iL==(size_t)-1) continue;
                    set_cell(last_out, c, soa_cloud.x[iL], soa_cloud.y[iL], soa_cloud.z[iL], soa_cloud.intensity[iL]);
                }
                // flatten to SoA
                core::PointCloudSoA grid_out; grid_out.reserve(out_size);
                for (size_t idx=0; idx<out_size; ++idx) {
                    grid_out.addPoint(gx[idx], gy[idx], gz[idx], gi[idx]);
                }
                // use grid_out as interpolated
                interp_result.success = true;
                interp_result.interpolated_cloud = core::PooledPtr<core::PointCloudSoA>(new core::PointCloudSoA(std::move(grid_out)), core::PoolDeleter<core::PointCloudSoA>{nullptr});
                interpolated = interp_result.interpolated_cloud.get();

                // publish interpolated cloud if requested
                if (interpolated_cloud_pub_) {
                    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_interpolated(new pcl::PointCloud<pcl::PointXYZI>);
                    pcl_interpolated->reserve(interpolated->size());
                    for (size_t i = 0; i < interpolated->size(); ++i) {
                        pcl::PointXYZI p{interpolated->x[i], interpolated->y[i], interpolated->z[i], interpolated->intensity[i]};
                        pcl_interpolated->push_back(p);
                    }
                    sensor_msgs::msg::PointCloud2 interp_msg; pcl::toROSMsg(*pcl_interpolated, interp_msg);
                    interp_msg.header = lidar_msg->header; interpolated_cloud_pub_->publish(interp_msg);
                }
            } else {
                // Engine-based interpolation
                try {
                    interp_result = interpolation_engine_->interpolate(soa_cloud);
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(get_logger(), "Interpolation exception: %s", e.what());
                    return;
                }
                if (!interp_result.success) {
                    RCLCPP_ERROR(get_logger(), "Interpolation failed: %s", interp_result.error_message.c_str());
                    return;
                }
                interpolated = interp_result.interpolated_cloud.get();
                if (interpolated_cloud_pub_) {
                    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_interpolated(new pcl::PointCloud<pcl::PointXYZI>);
                    pcl_interpolated->reserve(interpolated->size());
                    for (size_t i = 0; i < interpolated->size(); ++i) {
                        pcl::PointXYZI p{interpolated->x[i], interpolated->y[i], interpolated->z[i], interpolated->intensity[i]};
                        pcl_interpolated->push_back(p);
                    }
                    sensor_msgs::msg::PointCloud2 interp_msg; pcl::toROSMsg(*pcl_interpolated, interp_msg);
                    interp_msg.header = lidar_msg->header; interpolated_cloud_pub_->publish(interp_msg);
                }
            }
        } else {
            interpolated = &soa_cloud;
        }

        // Check if color mapping is enabled
        if (enable_color_mapping_) {
            // Projection
            std::vector<projection::LiDARPoint> lidar_points; lidar_points.reserve(interpolated->size());
            for (size_t i = 0; i < interpolated->size(); ++i) {
                lidar_points.emplace_back(
                    interpolated->x[i], interpolated->y[i], interpolated->z[i], 1.0f, i);
            }
            projection::ProjectionResult projection_results;
            if (!projection_engine_->projectToAllCameras(lidar_points, projection_results)) {
                RCLCPP_ERROR(get_logger(), "Projection failed");
                return;
            }

            // Color extraction
            std::map<std::string, projection::ColorExtractor::ColorExtractionResult> extraction_results;
            std::map<std::string, std::vector<float>> pixel_distances;
            const size_t total_points = interpolated->size();
            for (const auto& cam_proj : projection_results.camera_projections) {
                // Prepare pixel coordinates and indices
                std::vector<cv::Point2f> pixel_coordinates; pixel_coordinates.reserve(cam_proj.projected_points.size());
                std::vector<size_t> point_indices; point_indices.reserve(cam_proj.projected_points.size());
                for (const auto& px : cam_proj.projected_points) {
                    pixel_coordinates.emplace_back(px.u, px.v);
                    point_indices.push_back(px.original_index);
                }
                auto it_img = camera_images.find(cam_proj.camera_id);
                if (it_img == camera_images.end()) { continue; }
                auto sparse_result = color_extractor_->extractColorsWithIndices(
                    it_img->second, pixel_coordinates, point_indices, extraction_config_);
                // Dense remap
                projection::ColorExtractor::ColorExtractionResult dense_result;
                dense_result.colors.assign(total_points, cv::Vec3b(0, 0, 0));
                dense_result.confidence_scores.assign(total_points, 0.0f);
                dense_result.valid_extractions.assign(total_points, false);
                dense_result.output_count = total_points;
                dense_result.valid_colors = 0;
                for (size_t k = 0; k < point_indices.size() && k < sparse_result.colors.size(); ++k) {
                    const size_t orig_idx = point_indices[k];
                    if (orig_idx >= total_points) continue;
                    dense_result.colors[orig_idx] = sparse_result.colors[k];
                    dense_result.confidence_scores[orig_idx] = sparse_result.confidence_scores[k];
                    dense_result.valid_extractions[orig_idx] = sparse_result.valid_extractions[k];
                    if (sparse_result.valid_extractions[k]) dense_result.valid_colors++;
                }
                extraction_results[cam_proj.camera_id] = std::move(dense_result);

                // Distances
                auto calibration = calibration_manager_->getCalibration(cam_proj.camera_id);
                if (calibration) {
                    double cx = calibration->K(0, 2);
                    double cy = calibration->K(1, 2);
                    std::vector<float> dense_distances(total_points, 0.0f);
                    for (size_t k = 0; k < point_indices.size() && k < pixel_coordinates.size(); ++k) {
                        const size_t orig_idx = point_indices[k];
                        if (orig_idx >= total_points) continue;
                        const auto& pixel = pixel_coordinates[k];
                        float dx = pixel.x - static_cast<float>(cx);
                        float dy = pixel.y - static_cast<float>(cy);
                        dense_distances[orig_idx] = std::sqrt(dx*dx + dy*dy);
                    }
                    pixel_distances[cam_proj.camera_id] = std::move(dense_distances);
                }
            }

            // Fusion
            std::vector<size_t> all_indices; all_indices.reserve(interpolated->size());
            for (size_t i = 0; i < interpolated->size(); ++i) all_indices.push_back(i);
            auto fusion_result = multi_camera_fusion_->fuseColorsWithDistances(
                extraction_results, pixel_distances, all_indices, fusion_config_);

            // Build colored cloud
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            colored_cloud->reserve(interpolated->size());
            for (size_t i = 0; i < interpolated->size(); ++i) {
                pcl::PointXYZRGB point;
                point.x = interpolated->x[i];
                point.y = interpolated->y[i];
                point.z = interpolated->z[i];
                if (i < fusion_result.fused_colors.size() &&
                    fusion_result.fusion_confidence[i] >= fusion_config_.confidence_threshold) {
                    const auto& rgb = fusion_result.fused_colors[i];
                    point.r = rgb[0]; point.g = rgb[1]; point.b = rgb[2];
                } else {
                    point.r = point.g = point.b = 128;
                }
                colored_cloud->push_back(point);
            }

            // Publish colored cloud
            sensor_msgs::msg::PointCloud2 output_msg; pcl::toROSMsg(*colored_cloud, output_msg);
            output_msg.header = lidar_msg->header; colored_cloud_pub_->publish(output_msg);

            // Log colored fusion stats
            RCLCPP_DEBUG(get_logger(),
                "PRISM Color Fusion: Points: %zu | Colored: %zu (%.1f%%)",
                interpolated->size(), fusion_result.successful_fusions,
                100.0f * fusion_result.successful_fusions / std::max<size_t>(1, interpolated->size()));
        } else {
            // Interpolation-only mode: publish interpolated cloud without colors
            pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZI>);
            output_cloud->reserve(interpolated->size());
            for (size_t i = 0; i < interpolated->size(); ++i) {
                pcl::PointXYZI point;
                point.x = interpolated->x[i];
                point.y = interpolated->y[i];
                point.z = interpolated->z[i];
                point.intensity = interpolated->intensity[i];
                output_cloud->push_back(point);
            }
            
            sensor_msgs::msg::PointCloud2 output_msg; 
            pcl::toROSMsg(*output_cloud, output_msg);
            output_msg.header = lidar_msg->header; 
            colored_cloud_pub_->publish(output_msg);
            
            RCLCPP_INFO_ONCE(get_logger(), 
                "PRISM running in interpolation-only mode (color mapping disabled)");
        }

        // Timings
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - t_start);
        
        // Log performance based on mode
        if (enable_color_mapping_) {
            RCLCPP_DEBUG(get_logger(),
                "PRISM Fusion Pipeline (cache): %ld ms | Points: %zu",
                duration.count(), interpolated->size());
        } else {
            RCLCPP_DEBUG(get_logger(),
                "PRISM Interpolation-only: %ld ms | Points: %zu (FILC-like mode)",
                duration.count(), interpolated->size());
        }

        if (debug_stats_pub_) {
            std_msgs::msg::String msg; std::ostringstream s;
            s << "\n=== PRISM Fusion Pipeline Timings (cache) ===\n";
            s << "Total:                 " << duration.count() << " ms\n";
            if (cycle_ms > 0.0) {
                s << "Cycle (prev->this):   " << std::fixed << std::setprecision(2) << cycle_ms << " ms\n";
            }
            msg.data = s.str();
            debug_stats_pub_->publish(msg);
        }
        last_publish_time_ = this->get_clock()->now();
    }
    
private:
    struct CameraConfig {
        std::string id;
        std::string image_topic;
    };
    
    // Configuration
    std::vector<CameraConfig> camera_configs_;
    std::string calibration_path_;
    std::string lidar_input_topic_;
    std::string colored_output_topic_;
    std::string debug_interpolated_topic_;
    std::string interpolated_output_topic_;
    bool interpolation_enabled_ {false};
    bool enable_color_mapping_ {true};
    bool grid_mode_ {true};
    double interpolation_scale_factor_ {2.0};
    double discontinuity_threshold_ {0.5};
    int sync_queue_size_ {3};
    int min_interval_ms_ {100};
    interpolation::InterpolationConfig interpolation_config_;
    projection::ProjectionConfig projection_config_;
    projection::ColorExtractor::ColorExtractionConfig extraction_config_;
    projection::MultiCameraFusion::FusionConfig fusion_config_;
    
    // Core components
    std::shared_ptr<core::CalibrationManager> calibration_manager_;
    std::unique_ptr<interpolation::InterpolationEngine> interpolation_engine_;
    std::unique_ptr<projection::ProjectionEngine> projection_engine_;
    std::unique_ptr<projection::ColorExtractor> color_extractor_;
    std::unique_ptr<projection::MultiCameraFusion> multi_camera_fusion_;
    
    // ROS2 interface
    std::map<std::string, std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>>> image_subscribers_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> lidar_sub_;
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image, sensor_msgs::msg::Image>>> sync_;
    // Plain subscriptions (use_latest_image_)
    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> image_plain_subs_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_plain_sub_;
    std::map<std::string, sensor_msgs::msg::Image::SharedPtr> image_cache_;
    std::mutex image_cache_mutex_;
    
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr colored_cloud_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr debug_interpolated_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr debug_stats_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr interpolated_cloud_pub_;

    // Processing control
    std::atomic<bool> processing_ {false};
    rclcpp::Time last_processed_time_ {0, 0, RCL_ROS_TIME};
    rclcpp::Time last_publish_time_ {0, 0, RCL_ROS_TIME};

    // Sync behavior
    bool use_latest_image_ {true};
    int image_freshness_ms_ {150};
};

} // namespace prism

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<prism::PrismFusionNode>();
        // Now that shared_ptr is created, we can safely call initialize()
        node->initialize();
        RCLCPP_INFO(node->get_logger(), "Starting PRISM Fusion Node");
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception in PRISM Fusion Node: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}