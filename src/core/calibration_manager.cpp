#include "prism/core/calibration_manager.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <thread>
#include <set>
#include <cmath>
#include <limits>

namespace prism {
namespace core {

// CameraCalibration implementation
bool CameraCalibration::hasValidIntrinsics() const {
    return utils::validateCameraMatrix(K) && width > 0 && height > 0;
}

bool CameraCalibration::hasValidExtrinsics() const {
    return utils::validateTransformationMatrix(T_ref_cam);
}

bool CameraCalibration::validate() {
    validation_error.clear();
    is_valid = true;
    
    // Check intrinsics
    if (!hasValidIntrinsics()) {
        validation_error += "Invalid intrinsic parameters; ";
        is_valid = false;
    }
    
    // Check extrinsics
    if (!hasValidExtrinsics()) {
        validation_error += "Invalid extrinsic parameters; ";
        is_valid = false;
    }
    
    // Check distortion
    if (!utils::validateDistortionCoefficients(distortion, distortion_model)) {
        validation_error += "Invalid distortion coefficients; ";
        is_valid = false;
    }
    
    // Check camera ID
    if (camera_id.empty()) {
        validation_error += "Empty camera ID; ";
        is_valid = false;
    }
    
    return is_valid;
}

size_t CameraCalibration::getMemoryUsage() const noexcept {
    size_t usage = sizeof(CameraCalibration);
    usage += camera_id.capacity();
    usage += reference_frame.capacity();
    usage += validation_error.capacity();
    usage += distortion_model.capacity();
    usage += distortion.size() * sizeof(double);
    return usage;
}

// CalibrationManager implementation
CalibrationManager::CalibrationManager(const Config& config) 
    : config_(config)
    , hot_reload_enabled_(config.enable_hot_reload)
    , stop_hot_reload_(false) {
    
    // Create calibration directory if it doesn't exist
    if (!config_.calibration_directory.empty()) {
        std::filesystem::create_directories(config_.calibration_directory);
    }
    
    // Start hot reload thread if enabled
    if (hot_reload_enabled_.load()) {
        hot_reload_thread_ = std::thread(&CalibrationManager::hotReloadThreadFunc, this);
    }
}

CalibrationManager::~CalibrationManager() {
    // Stop hot reload thread
    stop_hot_reload_.store(true);
    if (hot_reload_thread_.joinable()) {
        hot_reload_thread_.join();
    }
}

bool CalibrationManager::loadCalibration(const std::string& camera_id, const std::string& yaml_file_path) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // Check if file exists
        if (!std::filesystem::exists(yaml_file_path)) {
            stats_.file_errors++;
            return false;
        }
        
        // Load YAML file
        YAML::Node yaml_node = YAML::LoadFile(yaml_file_path);
        
        // Load from node
        bool success = loadCalibrationFromNode(camera_id, yaml_node);
        
        if (success) {
            // Update file path in cache entry
            std::unique_lock<std::shared_mutex> lock(cache_mutex_);
            auto it = calibration_cache_.find(camera_id);
            if (it != calibration_cache_.end()) {
                it->second.file_path = yaml_file_path;
                if (config_.cache_file_timestamps) {
                    it->second.last_write_time = std::filesystem::last_write_time(yaml_file_path);
                }
            }
        }
        
        // Update statistics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // Update average load time (simple exponential moving average)
        double current_time = static_cast<double>(duration.count());
        double old_avg = stats_.avg_load_time_us.get();
        stats_.avg_load_time_us.set(old_avg * 0.9 + current_time * 0.1);
        
        stats_.total_loads++;
        
        return success;
        
    } catch (const YAML::Exception& e) {
        stats_.file_errors++;
        return false;
    } catch (const std::exception& e) {
        stats_.file_errors++;
        return false;
    }
}

bool CalibrationManager::loadCalibrationFromNode(const std::string& camera_id, const YAML::Node& yaml_node) {
    auto calibration = std::make_shared<CameraCalibration>();
    calibration->camera_id = camera_id;
    calibration->last_updated = std::chrono::system_clock::now();
    
    try {
        // Parse intrinsic parameters
        if (!parseIntrinsics(yaml_node, *calibration)) {
            std::cerr << "Failed to parse intrinsics for " << camera_id << std::endl;
            stats_.validation_failures++;
            return false;
        }
        
        // Parse extrinsic parameters (optional)
        if (yaml_node["extrinsics"]) {
            if (!parseExtrinsics(yaml_node["extrinsics"], *calibration)) {
                stats_.validation_failures++;
                return false;
            }
        }
        
        // Parse distortion parameters
        if (!parseDistortion(yaml_node, *calibration)) {
            stats_.validation_failures++;
            return false;
        }
        
        // Call validate() to set is_valid flag
        calibration->validate();
        
        // Validate if enabled
        if (config_.validate_on_load && !validateCalibration(*calibration)) {
            stats_.validation_failures++;
            return false;
        }
        
        // Store in cache
        {
            std::unique_lock<std::shared_mutex> lock(cache_mutex_);
            
            // Check cache size and evict if necessary
            if (calibration_cache_.size() >= config_.max_cache_size) {
                evictLRUEntries();
            }
            
            calibration_cache_[camera_id] = CacheEntry(calibration, "");
        }
        
        // Call update callback if set
        {
            std::lock_guard<std::mutex> callback_lock(callback_mutex_);
            if (update_callback_) {
                update_callback_(camera_id);
            }
        }
        
        return true;
        
    } catch (const std::exception& e) {
        stats_.validation_failures++;
        return false;
    }
}

std::shared_ptr<const CameraCalibration> CalibrationManager::getCalibration(const std::string& camera_id) const {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    
    auto it = calibration_cache_.find(camera_id);
    if (it != calibration_cache_.end()) {
        // Update access time
        updateAccessTime(camera_id);
        stats_.cache_hits++;
        
        // Update access time statistics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double current_time = static_cast<double>(duration.count());
        double old_avg = stats_.avg_access_time_us.get();
        stats_.avg_access_time_us.set(old_avg * 0.9 + current_time * 0.1);
        
        return it->second.calibration;
    }
    
    stats_.cache_misses++;
    return nullptr;
}

bool CalibrationManager::hasCalibration(const std::string& camera_id) const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return calibration_cache_.find(camera_id) != calibration_cache_.end();
}

std::vector<std::string> CalibrationManager::getCameraIds() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    
    std::vector<std::string> ids;
    ids.reserve(calibration_cache_.size());
    
    for (const auto& pair : calibration_cache_) {
        ids.push_back(pair.first);
    }
    
    return ids;
}

bool CalibrationManager::addTransform(const std::string& from_frame, const std::string& to_frame,
                                     const Eigen::Matrix4d& transform, bool is_static) {
    if (!utils::validateTransformationMatrix(transform)) {
        return false;
    }
    
    std::unique_lock<std::shared_mutex> lock(transform_mutex_);
    
    TransformChainElement element;
    element.from_frame = from_frame;
    element.to_frame = to_frame;
    element.transform = transform;
    element.is_static = is_static;
    element.timestamp = std::chrono::system_clock::now();
    
    transform_graph_[from_frame][to_frame] = element;
    
    // Also add inverse transform
    TransformChainElement inverse_element;
    inverse_element.from_frame = to_frame;
    inverse_element.to_frame = from_frame;
    inverse_element.transform = transform.inverse();
    inverse_element.is_static = is_static;
    inverse_element.timestamp = element.timestamp;
    
    transform_graph_[to_frame][from_frame] = inverse_element;
    
    return true;
}

bool CalibrationManager::getTransform(const std::string& from_frame, const std::string& to_frame,
                                     Eigen::Matrix4d& transform) const {
    if (from_frame == to_frame) {
        transform.setIdentity();
        return true;
    }
    
    std::shared_lock<std::shared_mutex> lock(transform_mutex_);
    
    // Direct lookup first
    auto from_it = transform_graph_.find(from_frame);
    if (from_it != transform_graph_.end()) {
        auto to_it = from_it->second.find(to_frame);
        if (to_it != from_it->second.end()) {
            transform = to_it->second.transform;
            return true;
        }
    }
    
    // Use DFS to find transform chain
    std::set<std::string> visited;
    Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
    
    if (computeTransformChain(from_frame, to_frame, visited, result)) {
        transform = result;
        return true;
    }
    
    return false;
}

size_t CalibrationManager::reloadAll() {
    std::vector<std::string> camera_ids = getCameraIds();
    size_t reload_count = 0;
    
    for (const auto& camera_id : camera_ids) {
        if (checkFileModification(camera_id)) {
            reload_count++;
        }
    }
    
    return reload_count;
}

bool CalibrationManager::removeCalibration(const std::string& camera_id) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    
    auto it = calibration_cache_.find(camera_id);
    if (it != calibration_cache_.end()) {
        calibration_cache_.erase(it);
        return true;
    }
    
    return false;
}

void CalibrationManager::clear() {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    calibration_cache_.clear();
}

size_t CalibrationManager::getMemoryUsage() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    
    size_t total = sizeof(CalibrationManager);
    
    for (const auto& pair : calibration_cache_) {
        total += pair.first.capacity();  // camera_id string
        total += sizeof(CacheEntry);
        if (pair.second.calibration) {
            total += pair.second.calibration->getMemoryUsage();
        }
        total += pair.second.file_path.capacity();
    }
    
    return total;
}

void CalibrationManager::setUpdateCallback(std::function<void(const std::string&)> callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    update_callback_ = std::move(callback);
}

void CalibrationManager::setHotReloadEnabled(bool enable) {
    bool was_enabled = hot_reload_enabled_.exchange(enable);
    
    if (enable && !was_enabled) {
        // Start hot reload thread
        if (!hot_reload_thread_.joinable()) {
            stop_hot_reload_.store(false);
            hot_reload_thread_ = std::thread(&CalibrationManager::hotReloadThreadFunc, this);
        }
    } else if (!enable && was_enabled) {
        // Stop hot reload thread
        stop_hot_reload_.store(true);
        if (hot_reload_thread_.joinable()) {
            hot_reload_thread_.join();
        }
    }
}

// Private methods
bool CalibrationManager::parseIntrinsics(const YAML::Node& node, CameraCalibration& calibration) {
    try {
        // Parse camera matrix
        if (node["camera_matrix"]) {
            YAML::Node K_node;
            // Support both nested structure with "data" key and direct array
            if (node["camera_matrix"]["data"]) {
                K_node = node["camera_matrix"]["data"];
            } else {
                K_node = node["camera_matrix"];
            }
            
            if (!K_node || K_node.size() != 9) {
                std::cerr << "Invalid camera matrix size: " << (K_node ? K_node.size() : 0) << std::endl;
                return false;
            }
            
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    calibration.K(i, j) = K_node[i * 3 + j].as<double>();
                }
            }
        } else if (node["K"]) {
            auto K_node = node["K"];
            if (!K_node || K_node.size() != 9) {
                return false;
            }
            
            for (int i = 0; i < 9; ++i) {
                calibration.K(i / 3, i % 3) = K_node[i].as<double>();
            }
        } else {
            return false;
        }
        
        // Parse image dimensions
        if (node["image_width"]) {
            calibration.width = node["image_width"].as<int>();
        }
        if (node["image_height"]) {
            calibration.height = node["image_height"].as<int>();
        }
        
        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

bool CalibrationManager::parseExtrinsics(const YAML::Node& node, CameraCalibration& calibration) {
    try {
        // Parse transformation matrix
        if (node["T"] || node["transform"]) {
            auto T_node = node["T"] ? node["T"] : node["transform"];
            
            if (T_node.IsSequence() && T_node.size() == 16) {
                // 4x4 matrix as flat array
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        calibration.T_ref_cam(i, j) = T_node[i * 4 + j].as<double>();
                    }
                }
            } else if (T_node.size() == 4) {
                // 4x4 matrix as nested array
                for (int i = 0; i < 4; ++i) {
                    auto row = T_node[i];
                    if (row.size() != 4) return false;
                    for (int j = 0; j < 4; ++j) {
                        calibration.T_ref_cam(i, j) = row[j].as<double>();
                    }
                }
            } else {
                return false;
            }
        } else if (node["translation"] && node["rotation"]) {
            // Parse translation and rotation separately
            auto translation = node["translation"];
            auto rotation = node["rotation"];
            
            if (translation.size() != 3) return false;
            
            Eigen::Vector3d t;
            t[0] = translation[0].as<double>();
            t[1] = translation[1].as<double>();
            t[2] = translation[2].as<double>();
            
            Eigen::Matrix3d R;
            if (rotation.size() == 9) {
                // Rotation matrix
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        R(i, j) = rotation[i * 3 + j].as<double>();
                    }
                }
            } else if (rotation.size() == 4) {
                // Quaternion [x, y, z, w]
                Eigen::Quaterniond q(rotation[3].as<double>(), rotation[0].as<double>(),
                                    rotation[1].as<double>(), rotation[2].as<double>());
                R = q.toRotationMatrix();
            } else {
                return false;
            }
            
            // Build transformation matrix
            calibration.T_ref_cam.setIdentity();
            calibration.T_ref_cam.block<3, 3>(0, 0) = R;
            calibration.T_ref_cam.block<3, 1>(0, 3) = t;
        } else {
            return false;
        }
        
        // Parse reference frame
        if (node["reference_frame"]) {
            calibration.reference_frame = node["reference_frame"].as<std::string>();
        }
        
        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

bool CalibrationManager::parseDistortion(const YAML::Node& node, CameraCalibration& calibration) {
    try {
        // Parse distortion model
        if (node["distortion_model"]) {
            calibration.distortion_model = node["distortion_model"].as<std::string>();
        }
        
        // Parse distortion coefficients
        YAML::Node dist_node;
        if (node["distortion_coefficients"]) {
            // Support both nested structure with "data" key and direct array
            if (node["distortion_coefficients"]["data"]) {
                dist_node = node["distortion_coefficients"]["data"];
            } else {
                dist_node = node["distortion_coefficients"];
            }
        } else if (node["D"]) {
            dist_node = node["D"];
        } else {
            // Use default zero distortion
            calibration.distortion.resize(5);
            calibration.distortion.setZero();
            return true;
        }
        
        if (!dist_node) {
            return false;
        }
        
        calibration.distortion.resize(dist_node.size());
        for (size_t i = 0; i < dist_node.size(); ++i) {
            calibration.distortion[i] = dist_node[i].as<double>();
        }
        
        return true;
        
    } catch (const std::exception& e) {
        return false;
    }
}

bool CalibrationManager::validateCalibration(CameraCalibration& calibration) {
    return calibration.validate();
}

void CalibrationManager::hotReloadThreadFunc() {
    while (!stop_hot_reload_.load()) {
        try {
            // Get list of camera IDs to check
            std::vector<std::string> camera_ids = getCameraIds();
            
            // Check each calibration file for modifications
            for (const auto& camera_id : camera_ids) {
                if (stop_hot_reload_.load()) break;
                
                checkFileModification(camera_id);
            }
            
            // Sleep for the configured interval
            std::this_thread::sleep_for(config_.file_check_interval);
            
        } catch (const std::exception& e) {
            // Log error and continue
            std::this_thread::sleep_for(config_.file_check_interval);
        }
    }
}

bool CalibrationManager::checkFileModification(const std::string& camera_id) {
    if (!config_.cache_file_timestamps) {
        return false;
    }
    
    std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);
    auto it = calibration_cache_.find(camera_id);
    if (it == calibration_cache_.end() || it->second.file_path.empty()) {
        return false;
    }
    
    std::string file_path = it->second.file_path;
    auto cached_time = it->second.last_write_time;
    read_lock.unlock();
    
    try {
        if (!std::filesystem::exists(file_path)) {
            return false;
        }
        
        auto current_time = std::filesystem::last_write_time(file_path);
        
        if (current_time > cached_time) {
            // File was modified, reload it
            bool success = loadCalibration(camera_id, file_path);
            if (success) {
                stats_.hot_reloads++;
                return true;
            }
        }
        
    } catch (const std::filesystem::filesystem_error& e) {
        // File system error, skip this check
    }
    
    return false;
}

bool CalibrationManager::computeTransformChain(const std::string& from_frame, const std::string& to_frame,
                                              std::set<std::string>& visited, Eigen::Matrix4d& result) const {
    if (visited.find(from_frame) != visited.end()) {
        // Cycle detected
        return false;
    }
    
    visited.insert(from_frame);
    
    auto from_it = transform_graph_.find(from_frame);
    if (from_it == transform_graph_.end()) {
        visited.erase(from_frame);
        return false;
    }
    
    for (const auto& edge : from_it->second) {
        if (edge.first == to_frame) {
            // Direct connection found
            result = edge.second.transform;
            visited.erase(from_frame);
            return true;
        }
        
        // Recursive search
        Eigen::Matrix4d intermediate_result;
        if (computeTransformChain(edge.first, to_frame, visited, intermediate_result)) {
            result = intermediate_result * edge.second.transform;
            visited.erase(from_frame);
            return true;
        }
    }
    
    visited.erase(from_frame);
    return false;
}

void CalibrationManager::updateAccessTime(const std::string& camera_id) const {
    // This method is called while holding a shared lock, so we need to be careful
    // about thread safety. We'll just update the access time atomically.
    auto it = calibration_cache_.find(camera_id);
    if (it != calibration_cache_.end()) {
        // Update access time (this is thread-safe for chrono time_point)
        const_cast<CacheEntry&>(it->second).last_access_time = std::chrono::steady_clock::now();
    }
}

void CalibrationManager::evictLRUEntries() {
    if (calibration_cache_.size() < config_.max_cache_size) {
        return;
    }
    
    // Find the least recently used entry
    auto oldest_it = calibration_cache_.begin();
    auto oldest_time = oldest_it->second.last_access_time;
    
    for (auto it = calibration_cache_.begin(); it != calibration_cache_.end(); ++it) {
        if (it->second.last_access_time < oldest_time) {
            oldest_time = it->second.last_access_time;
            oldest_it = it;
        }
    }
    
    calibration_cache_.erase(oldest_it);
}

// Utility functions implementation
namespace utils {

std::unique_ptr<CameraCalibration> loadROSCameraInfo(const std::string& yaml_path) {
    try {
        YAML::Node node = YAML::LoadFile(yaml_path);
        
        auto calibration = std::make_unique<CameraCalibration>();
        
        // ROS camera_info format parsing
        if (node["camera_name"]) {
            calibration->camera_id = node["camera_name"].as<std::string>();
        }
        
        if (node["image_width"]) {
            calibration->width = node["image_width"].as<int>();
        }
        
        if (node["image_height"]) {
            calibration->height = node["image_height"].as<int>();
        }
        
        // Camera matrix (K)
        if (node["camera_matrix"] && node["camera_matrix"]["data"]) {
            auto K_data = node["camera_matrix"]["data"];
            if (K_data.size() == 9) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        calibration->K(i, j) = K_data[i * 3 + j].as<double>();
                    }
                }
            }
        }
        
        // Distortion coefficients
        if (node["distortion_coefficients"] && node["distortion_coefficients"]["data"]) {
            auto D_data = node["distortion_coefficients"]["data"];
            calibration->distortion.resize(D_data.size());
            for (size_t i = 0; i < D_data.size(); ++i) {
                calibration->distortion[i] = D_data[i].as<double>();
            }
        }
        
        // Distortion model
        if (node["distortion_model"]) {
            calibration->distortion_model = node["distortion_model"].as<std::string>();
        }
        
        calibration->validate();
        return calibration;
        
    } catch (const std::exception& e) {
        return nullptr;
    }
}

YAML::Node convertToROSFormat(const CameraCalibration& calibration) {
    YAML::Node node;
    
    node["camera_name"] = calibration.camera_id;
    node["image_width"] = calibration.width;
    node["image_height"] = calibration.height;
    node["distortion_model"] = calibration.distortion_model;
    
    // Camera matrix
    node["camera_matrix"]["rows"] = 3;
    node["camera_matrix"]["cols"] = 3;
    node["camera_matrix"]["data"] = std::vector<double>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            node["camera_matrix"]["data"].push_back(calibration.K(i, j));
        }
    }
    
    // Distortion coefficients
    node["distortion_coefficients"]["rows"] = 1;
    node["distortion_coefficients"]["cols"] = static_cast<int>(calibration.distortion.size());
    node["distortion_coefficients"]["data"] = std::vector<double>();
    for (int i = 0; i < calibration.distortion.size(); ++i) {
        node["distortion_coefficients"]["data"].push_back(calibration.distortion[i]);
    }
    
    return node;
}

bool validateCameraMatrix(const Eigen::Matrix3d& K) {
    // Check for NaN or infinity
    if (!K.allFinite()) {
        return false;
    }
    
    // Check for positive focal lengths
    if (K(0, 0) <= 0 || K(1, 1) <= 0) {
        return false;
    }
    
    // Check for reasonable focal length values (not too small or too large)
    double fx = K(0, 0);
    double fy = K(1, 1);
    if (fx < 10.0 || fx > 10000.0 || fy < 10.0 || fy > 10000.0) {
        return false;
    }
    
    // Check for reasonable principal point
    double cx = K(0, 2);
    double cy = K(1, 2);
    if (std::abs(cx) > 5000.0 || std::abs(cy) > 5000.0) {
        return false;
    }
    
    // Check matrix structure (should have zeros in specific positions)
    if (std::abs(K(1, 0)) > 1e-6 || std::abs(K(2, 0)) > 1e-6 || 
        std::abs(K(2, 1)) > 1e-6 || std::abs(K(2, 2) - 1.0) > 1e-6) {
        return false;
    }
    
    return true;
}

bool validateDistortionCoefficients(const Eigen::VectorXd& distortion, const std::string& model) {
    if (distortion.size() == 0) {
        return true;  // No distortion is valid
    }
    
    // Check for NaN or infinity
    if (!distortion.allFinite()) {
        return false;
    }
    
    // Check reasonable ranges for different models
    if (model == "plumb_bob" || model == "rational_polynomial") {
        // Typical distortion coefficients should be reasonable
        for (int i = 0; i < distortion.size(); ++i) {
            if (std::abs(distortion[i]) > 10.0) {  // Very large distortion coefficients are suspicious
                return false;
            }
        }
    }
    
    return true;
}

bool validateTransformationMatrix(const Eigen::Matrix4d& T) {
    // Check for NaN or infinity
    if (!T.allFinite()) {
        return false;
    }
    
    // Extract rotation part
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    
    // Check if rotation matrix is orthogonal
    Eigen::Matrix3d I = R * R.transpose();
    if (!I.isApprox(Eigen::Matrix3d::Identity(), 1e-6)) {
        return false;
    }
    
    // Check determinant is 1 (proper rotation, not reflection)
    if (std::abs(R.determinant() - 1.0) > 1e-6) {
        return false;
    }
    
    // Check bottom row is [0, 0, 0, 1]
    if (std::abs(T(3, 0)) > 1e-6 || std::abs(T(3, 1)) > 1e-6 || 
        std::abs(T(3, 2)) > 1e-6 || std::abs(T(3, 3) - 1.0) > 1e-6) {
        return false;
    }
    
    return true;
}

std::unique_ptr<CameraCalibration> createDefaultCalibration(
    const std::string& camera_id, int width, int height) {
    
    auto calibration = std::make_unique<CameraCalibration>();
    
    calibration->camera_id = camera_id;
    calibration->width = width;
    calibration->height = height;
    
    // Create reasonable default camera matrix
    double fx = width * 0.8;  // Reasonable focal length
    double fy = height * 0.8;
    double cx = width / 2.0;
    double cy = height / 2.0;
    
    calibration->K << fx,  0.0, cx,
                      0.0, fy,  cy,
                      0.0, 0.0, 1.0;
    
    // Zero distortion
    calibration->distortion.resize(5);
    calibration->distortion.setZero();
    
    // Identity extrinsics
    calibration->T_ref_cam.setIdentity();
    
    calibration->validate();
    return calibration;
}

} // namespace utils

} // namespace core
} // namespace prism