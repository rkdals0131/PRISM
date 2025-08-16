#pragma once

#include <unordered_map>
#include <string>
#include <memory>
#include <shared_mutex>
#include <mutex>
#include <atomic>
#include <chrono>
#include <functional>
#include <filesystem>
#include <thread>
#include <set>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

namespace prism {
namespace core {

/**
 * @brief Camera calibration data structure
 * 
 * Contains intrinsic and extrinsic calibration parameters for a single camera.
 * All matrices are stored in Eigen format for efficient computation.
 */
struct CameraCalibration {
    // Intrinsic parameters
    Eigen::Matrix3d K;                    // Camera matrix (3x3)
    Eigen::VectorXd distortion;           // Distortion coefficients [k1, k2, p1, p2, k3, ...]
    std::string distortion_model = "plumb_bob";  // Distortion model type
    
    // Image dimensions
    int width = 0;
    int height = 0;
    
    // Extrinsic parameters (camera to reference frame)
    Eigen::Matrix4d T_ref_cam;            // 4x4 transformation matrix
    
    // Metadata
    std::string camera_id;
    std::string reference_frame = "base_link";
    std::chrono::system_clock::time_point last_updated;
    
    // Validation status
    bool is_valid = false;
    std::string validation_error;
    
    /**
     * @brief Default constructor with identity initialization
     */
    CameraCalibration() {
        K.setIdentity();
        T_ref_cam.setIdentity();
        distortion.resize(5);
        distortion.setZero();
        last_updated = std::chrono::system_clock::now();
    }
    
    /**
     * @brief Check if calibration has valid intrinsic parameters
     * @return True if K matrix is positive definite and reasonable
     */
    bool hasValidIntrinsics() const;
    
    /**
     * @brief Check if calibration has valid extrinsic parameters
     * @return True if transformation matrix is valid SE(3)
     */
    bool hasValidExtrinsics() const;
    
    /**
     * @brief Validate all calibration parameters
     * @return True if calibration is complete and valid
     */
    bool validate();
    
    /**
     * @brief Get memory usage in bytes
     * @return Total memory footprint
     */
    size_t getMemoryUsage() const noexcept;
};

/**
 * @brief Transform chain element for multi-step transformations
 */
struct TransformChainElement {
    std::string from_frame;
    std::string to_frame;
    Eigen::Matrix4d transform;
    std::chrono::system_clock::time_point timestamp;
    bool is_static = true;
    
    TransformChainElement() {
        transform.setIdentity();
        timestamp = std::chrono::system_clock::now();
    }
};

/**
 * @brief Thread-safe calibration manager with hot-reload capability
 * 
 * Manages camera calibration data with fast access (<10μs) and runtime updates.
 * Uses shared_mutex for concurrent read access with exclusive write access.
 * Supports YAML file parsing and automatic validation.
 */
class CalibrationManager {
public:
    /**
     * @brief Configuration for calibration manager behavior
     */
    struct Config {
        std::string calibration_directory;
        bool enable_hot_reload;
        std::chrono::milliseconds file_check_interval;
        bool validate_on_load;
        bool cache_file_timestamps;
        size_t max_cache_size;
        
        Config() 
            : calibration_directory("./config/calibration")
            , enable_hot_reload(true)
            , file_check_interval(1000)
            , validate_on_load(true)
            , cache_file_timestamps(true)
            , max_cache_size(100) {}
    };
    
    /**
     * @brief Statistics for monitoring calibration manager
     */
    struct Stats {
        std::atomic<size_t> total_loads{0};
        std::atomic<size_t> cache_hits{0};
        std::atomic<size_t> cache_misses{0};
        std::atomic<size_t> hot_reloads{0};
        std::atomic<size_t> validation_failures{0};
        std::atomic<size_t> file_errors{0};
        
        // Performance metrics
        std::atomic<double> avg_load_time_us{0.0};
        std::atomic<double> avg_access_time_us{0.0};
        
        // Copy constructor for returning stats
        Stats() = default;
        Stats(const Stats& other) 
            : total_loads(other.total_loads.load())
            , cache_hits(other.cache_hits.load())
            , cache_misses(other.cache_misses.load())
            , hot_reloads(other.hot_reloads.load())
            , validation_failures(other.validation_failures.load())
            , file_errors(other.file_errors.load())
            , avg_load_time_us(other.avg_load_time_us.load())
            , avg_access_time_us(other.avg_access_time_us.load()) {}
    };
    
    /**
     * @brief Construct calibration manager with configuration
     * @param config Manager configuration
     */
    explicit CalibrationManager(const Config& config = Config());
    
    /**
     * @brief Destructor stops hot-reload thread
     */
    ~CalibrationManager();
    
    // Delete copy operations
    CalibrationManager(const CalibrationManager&) = delete;
    CalibrationManager& operator=(const CalibrationManager&) = delete;
    
    // Allow move operations
    CalibrationManager(CalibrationManager&&) noexcept = default;
    CalibrationManager& operator=(CalibrationManager&&) noexcept = default;
    
    /**
     * @brief Load calibration from YAML file
     * @param camera_id Unique identifier for the camera
     * @param yaml_file_path Path to YAML calibration file
     * @return True if loaded successfully
     */
    bool loadCalibration(const std::string& camera_id, const std::string& yaml_file_path);
    
    /**
     * @brief Load calibration from YAML node
     * @param camera_id Unique identifier for the camera
     * @param yaml_node Parsed YAML node containing calibration data
     * @return True if loaded successfully
     */
    bool loadCalibrationFromNode(const std::string& camera_id, const YAML::Node& yaml_node);
    
    /**
     * @brief Get calibration data for a camera (thread-safe, fast access)
     * @param camera_id Camera identifier
     * @return Shared pointer to calibration data, nullptr if not found
     */
    std::shared_ptr<const CameraCalibration> getCalibration(const std::string& camera_id) const;
    
    /**
     * @brief Check if calibration exists for camera
     * @param camera_id Camera identifier
     * @return True if calibration is loaded
     */
    bool hasCalibration(const std::string& camera_id) const;
    
    /**
     * @brief Get list of all loaded camera IDs
     * @return Vector of camera identifiers
     */
    std::vector<std::string> getCameraIds() const;
    
    /**
     * @brief Add or update a transform in the chain
     * @param from_frame Source frame
     * @param to_frame Target frame
     * @param transform 4x4 transformation matrix
     * @param is_static Whether transform is static or dynamic
     * @return True if added successfully
     */
    bool addTransform(const std::string& from_frame, const std::string& to_frame,
                     const Eigen::Matrix4d& transform, bool is_static = true);
    
    /**
     * @brief Get transform between two frames
     * @param from_frame Source frame
     * @param to_frame Target frame
     * @param transform Output transformation matrix
     * @return True if transform chain found
     */
    bool getTransform(const std::string& from_frame, const std::string& to_frame,
                     Eigen::Matrix4d& transform) const;
    
    /**
     * @brief Manually trigger reload of all calibrations
     * @return Number of calibrations successfully reloaded
     */
    size_t reloadAll();
    
    /**
     * @brief Remove calibration from cache
     * @param camera_id Camera identifier
     * @return True if removed
     */
    bool removeCalibration(const std::string& camera_id);
    
    /**
     * @brief Clear all calibrations
     */
    void clear();
    
    /**
     * @brief Get current statistics
     * @return Copy of current stats
     */
    Stats getStats() const { return stats_; }
    
    /**
     * @brief Get memory usage in bytes
     * @return Total memory usage of all calibrations
     */
    size_t getMemoryUsage() const;
    
    /**
     * @brief Set callback for calibration updates
     * @param callback Function called when calibration is updated
     */
    void setUpdateCallback(std::function<void(const std::string&)> callback);
    
    /**
     * @brief Enable or disable hot reload
     * @param enable Whether to enable hot reload
     */
    void setHotReloadEnabled(bool enable);

private:
    /**
     * @brief Calibration cache entry
     */
    struct CacheEntry {
        std::shared_ptr<CameraCalibration> calibration;
        std::string file_path;
        std::filesystem::file_time_type last_write_time;
        std::chrono::steady_clock::time_point last_access_time;
        
        CacheEntry() = default;
        CacheEntry(std::shared_ptr<CameraCalibration> cal, const std::string& path)
            : calibration(std::move(cal)), file_path(path)
            , last_access_time(std::chrono::steady_clock::now()) {
            if (std::filesystem::exists(file_path)) {
                last_write_time = std::filesystem::last_write_time(file_path);
            }
        }
    };
    
    /**
     * @brief Parse intrinsic parameters from YAML
     * @param node YAML node containing intrinsics
     * @param calibration Output calibration object
     * @return True if parsed successfully
     */
    bool parseIntrinsics(const YAML::Node& node, CameraCalibration& calibration);
    
    /**
     * @brief Parse extrinsic parameters from YAML
     * @param node YAML node containing extrinsics
     * @param calibration Output calibration object
     * @return True if parsed successfully
     */
    bool parseExtrinsics(const YAML::Node& node, CameraCalibration& calibration);
    
    /**
     * @brief Parse distortion parameters from YAML
     * @param node YAML node containing distortion
     * @param calibration Output calibration object
     * @return True if parsed successfully
     */
    bool parseDistortion(const YAML::Node& node, CameraCalibration& calibration);
    
    /**
     * @brief Validate parsed calibration data
     * @param calibration Calibration to validate
     * @return True if valid
     */
    bool validateCalibration(CameraCalibration& calibration);
    
    /**
     * @brief Hot reload thread function
     */
    void hotReloadThreadFunc();
    
    /**
     * @brief Check if file has been modified
     * @param camera_id Camera identifier
     * @return True if file was modified and reloaded
     */
    bool checkFileModification(const std::string& camera_id);
    
    /**
     * @brief Compute transform chain between frames using DFS
     * @param from_frame Source frame
     * @param to_frame Target frame
     * @param visited Set of visited frames (for cycle detection)
     * @param result Output transformation matrix
     * @return True if path found
     */
    bool computeTransformChain(const std::string& from_frame, const std::string& to_frame,
                              std::set<std::string>& visited, Eigen::Matrix4d& result) const;
    
    /**
     * @brief Update access time for cache entry
     * @param camera_id Camera identifier
     */
    void updateAccessTime(const std::string& camera_id) const;
    
    /**
     * @brief Remove least recently used entries if cache is full
     */
    void evictLRUEntries();
    
    // Configuration
    Config config_;
    
    // Calibration cache with thread-safe access
    mutable std::shared_mutex cache_mutex_;
    std::unordered_map<std::string, CacheEntry> calibration_cache_;
    
    // Transform chain for multi-camera setups
    mutable std::shared_mutex transform_mutex_;
    std::unordered_map<std::string, std::unordered_map<std::string, TransformChainElement>> transform_graph_;
    
    // Hot reload thread management
    std::atomic<bool> hot_reload_enabled_;
    std::atomic<bool> stop_hot_reload_;
    std::thread hot_reload_thread_;
    
    // Statistics
    mutable Stats stats_;
    
    // Update callback
    std::function<void(const std::string&)> update_callback_;
    mutable std::mutex callback_mutex_;
};

/**
 * @brief Utility functions for calibration management
 */
namespace utils {
    
    /**
     * @brief Load calibration from ROS camera_info format
     * @param yaml_path Path to ROS camera_info YAML file
     * @return Loaded calibration, or nullptr if failed
     */
    std::unique_ptr<CameraCalibration> loadROSCameraInfo(const std::string& yaml_path);
    
    /**
     * @brief Convert calibration to ROS camera_info format
     * @param calibration Input calibration
     * @return YAML node in ROS format
     */
    YAML::Node convertToROSFormat(const CameraCalibration& calibration);
    
    /**
     * @brief Validate camera matrix properties
     * @param K 3x3 camera matrix
     * @return True if valid (positive definite, reasonable values)
     */
    bool validateCameraMatrix(const Eigen::Matrix3d& K);
    
    /**
     * @brief Validate distortion coefficients
     * @param distortion Distortion vector
     * @param model Distortion model name
     * @return True if coefficients are reasonable
     */
    bool validateDistortionCoefficients(const Eigen::VectorXd& distortion, 
                                       const std::string& model);
    
    /**
     * @brief Validate transformation matrix
     * @param T 4x4 transformation matrix
     * @return True if valid SE(3) matrix
     */
    bool validateTransformationMatrix(const Eigen::Matrix4d& T);
    
    /**
     * @brief Create default calibration for testing
     * @param camera_id Camera identifier
     * @param width Image width
     * @param height Image height
     * @return Default calibration with reasonable parameters
     */
    std::unique_ptr<CameraCalibration> createDefaultCalibration(
        const std::string& camera_id, int width, int height);
    
} // namespace utils

} // namespace core
} // namespace prism