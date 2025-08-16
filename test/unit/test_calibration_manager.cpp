#include <gtest/gtest.h>
#include "prism/core/calibration_manager.hpp"
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>

using namespace prism::core;

class CalibrationManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = std::filesystem::temp_directory_path() / "test_calibration";
        std::filesystem::create_directories(test_dir_);
        
        // Create test YAML file
        test_yaml_path_ = test_dir_ / "test_camera.yaml";
        createTestYAML();
        
        // Configure manager
        config_.calibration_directory = test_dir_.string();
        config_.enable_hot_reload = false;  // Disable for unit tests
        config_.validate_on_load = true;
    }
    
    void TearDown() override {
        std::filesystem::remove_all(test_dir_);
    }
    
    void createTestYAML() {
        std::ofstream file(test_yaml_path_);
        file << "camera_name: test_camera\n";
        file << "image_width: 640\n";
        file << "image_height: 480\n";
        file << "camera_matrix:\n";
        file << "  rows: 3\n";
        file << "  cols: 3\n";
        file << "  data: [525.0, 0.0, 320.0, 0.0, 525.0, 240.0, 0.0, 0.0, 1.0]\n";
        file << "distortion_model: plumb_bob\n";
        file << "distortion_coefficients:\n";
        file << "  rows: 1\n";
        file << "  cols: 5\n";
        file << "  data: [0.1, -0.2, 0.001, 0.002, 0.0]\n";
        file << "extrinsics:\n";
        file << "  reference_frame: base_link\n";
        file << "  translation: [0.1, 0.0, 0.5]\n";
        file << "  rotation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]\n";
        file.close();
    }
    
    std::filesystem::path test_dir_;
    std::filesystem::path test_yaml_path_;
    CalibrationManager::Config config_;
};

TEST_F(CalibrationManagerTest, BasicConstruction) {
    CalibrationManager manager(config_);
    
    // Check initial state
    EXPECT_EQ(manager.getCameraIds().size(), 0);
    EXPECT_FALSE(manager.hasCalibration("test_camera"));
    
    auto stats = manager.getStats();
    EXPECT_EQ(stats.total_loads.load(), 0);
    EXPECT_EQ(stats.cache_hits.load(), 0);
    EXPECT_EQ(stats.cache_misses.load(), 0);
}

TEST_F(CalibrationManagerTest, LoadCalibrationFromFile) {
    CalibrationManager manager(config_);
    
    // Load calibration
    bool success = manager.loadCalibration("test_camera", test_yaml_path_.string());
    EXPECT_TRUE(success);
    
    // Check if loaded
    EXPECT_TRUE(manager.hasCalibration("test_camera"));
    
    auto camera_ids = manager.getCameraIds();
    EXPECT_EQ(camera_ids.size(), 1);
    EXPECT_EQ(camera_ids[0], "test_camera");
}

TEST_F(CalibrationManagerTest, GetCalibrationData) {
    CalibrationManager manager(config_);
    
    // Load calibration
    manager.loadCalibration("test_camera", test_yaml_path_.string());
    
    // Get calibration data
    auto calibration = manager.getCalibration("test_camera");
    ASSERT_NE(calibration, nullptr);
    
    // Check intrinsic parameters
    EXPECT_EQ(calibration->width, 640);
    EXPECT_EQ(calibration->height, 480);
    EXPECT_DOUBLE_EQ(calibration->K(0, 0), 525.0);  // fx
    EXPECT_DOUBLE_EQ(calibration->K(1, 1), 525.0);  // fy
    EXPECT_DOUBLE_EQ(calibration->K(0, 2), 320.0);  // cx
    EXPECT_DOUBLE_EQ(calibration->K(1, 2), 240.0);  // cy
    
    // Check distortion
    EXPECT_EQ(calibration->distortion_model, "plumb_bob");
    EXPECT_EQ(calibration->distortion.size(), 5);
    EXPECT_DOUBLE_EQ(calibration->distortion[0], 0.1);
    EXPECT_DOUBLE_EQ(calibration->distortion[1], -0.2);
    
    // Check extrinsics
    EXPECT_EQ(calibration->reference_frame, "base_link");
    EXPECT_DOUBLE_EQ(calibration->T_ref_cam(0, 3), 0.1);  // tx
    EXPECT_DOUBLE_EQ(calibration->T_ref_cam(1, 3), 0.0);  // ty
    EXPECT_DOUBLE_EQ(calibration->T_ref_cam(2, 3), 0.5);  // tz
}

TEST_F(CalibrationManagerTest, CachePerformance) {
    CalibrationManager manager(config_);
    manager.loadCalibration("test_camera", test_yaml_path_.string());
    
    // Measure access time
    auto start = std::chrono::high_resolution_clock::now();
    
    constexpr int num_accesses = 1000;
    for (int i = 0; i < num_accesses; ++i) {
        auto calibration = manager.getCalibration("test_camera");
        EXPECT_NE(calibration, nullptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Average access time should be < 10μs as specified
    double avg_time_us = static_cast<double>(duration.count()) / num_accesses;
    EXPECT_LT(avg_time_us, 10.0);
    
    // Check statistics
    auto stats = manager.getStats();
    EXPECT_EQ(stats.cache_hits.load(), num_accesses);
    EXPECT_EQ(stats.cache_misses.load(), 0);
}

TEST_F(CalibrationManagerTest, TransformChain) {
    CalibrationManager manager(config_);
    
    // Add some transforms
    Eigen::Matrix4d T1 = Eigen::Matrix4d::Identity();
    T1.block<3, 1>(0, 3) = Eigen::Vector3d(1.0, 0.0, 0.0);
    
    Eigen::Matrix4d T2 = Eigen::Matrix4d::Identity();
    T2.block<3, 1>(0, 3) = Eigen::Vector3d(0.0, 1.0, 0.0);
    
    EXPECT_TRUE(manager.addTransform("base_link", "camera1", T1));
    EXPECT_TRUE(manager.addTransform("camera1", "camera2", T2));
    
    // Test direct transform lookup
    Eigen::Matrix4d result;
    EXPECT_TRUE(manager.getTransform("base_link", "camera1", result));
    EXPECT_TRUE(result.isApprox(T1));
    
    // Test chained transform
    EXPECT_TRUE(manager.getTransform("base_link", "camera2", result));
    Eigen::Matrix4d expected = T2 * T1;
    EXPECT_TRUE(result.isApprox(expected));
    
    // Test identity transform
    EXPECT_TRUE(manager.getTransform("base_link", "base_link", result));
    EXPECT_TRUE(result.isApprox(Eigen::Matrix4d::Identity()));
}

TEST_F(CalibrationManagerTest, ValidationUtils) {
    // Test camera matrix validation
    Eigen::Matrix3d valid_K;
    valid_K << 500.0, 0.0, 320.0,
               0.0, 500.0, 240.0,
               0.0, 0.0, 1.0;
    EXPECT_TRUE(utils::validateCameraMatrix(valid_K));
    
    // Test invalid camera matrix (negative focal length)
    Eigen::Matrix3d invalid_K;
    invalid_K << -500.0, 0.0, 320.0,
                 0.0, 500.0, 240.0,
                 0.0, 0.0, 1.0;
    EXPECT_FALSE(utils::validateCameraMatrix(invalid_K));
    
    // Test transformation matrix validation
    Eigen::Matrix4d valid_T = Eigen::Matrix4d::Identity();
    EXPECT_TRUE(utils::validateTransformationMatrix(valid_T));
    
    // Test invalid transformation matrix (not SE(3))
    Eigen::Matrix4d invalid_T;
    invalid_T << 1.0, 0.0, 0.0, 1.0,
                 0.0, 2.0, 0.0, 0.0,  // Invalid scaling
                 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 1.0;
    EXPECT_FALSE(utils::validateTransformationMatrix(invalid_T));
}

TEST_F(CalibrationManagerTest, DefaultCalibrationCreation) {
    auto calibration = utils::createDefaultCalibration("default_cam", 640, 480);
    ASSERT_NE(calibration, nullptr);
    
    EXPECT_EQ(calibration->camera_id, "default_cam");
    EXPECT_EQ(calibration->width, 640);
    EXPECT_EQ(calibration->height, 480);
    EXPECT_TRUE(calibration->hasValidIntrinsics());
    EXPECT_TRUE(calibration->hasValidExtrinsics());
    EXPECT_TRUE(calibration->is_valid);
}

TEST_F(CalibrationManagerTest, MemoryUsage) {
    CalibrationManager manager(config_);
    
    size_t initial_usage = manager.getMemoryUsage();
    
    // Load calibration
    manager.loadCalibration("test_camera", test_yaml_path_.string());
    
    size_t after_load_usage = manager.getMemoryUsage();
    EXPECT_GT(after_load_usage, initial_usage);
    
    // Check individual calibration memory usage
    auto calibration = manager.getCalibration("test_camera");
    ASSERT_NE(calibration, nullptr);
    
    size_t cal_usage = calibration->getMemoryUsage();
    EXPECT_GT(cal_usage, 0);
}

TEST_F(CalibrationManagerTest, HotReloadDisabled) {
    // Test that hot reload can be disabled
    config_.enable_hot_reload = false;
    CalibrationManager manager(config_);
    
    manager.loadCalibration("test_camera", test_yaml_path_.string());
    
    // Modify the file
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    createTestYAML();  // Recreate with new timestamp
    
    // Wait and check that it wasn't reloaded
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto stats = manager.getStats();
    EXPECT_EQ(stats.hot_reloads.load(), 0);
}

TEST_F(CalibrationManagerTest, ErrorHandling) {
    CalibrationManager manager(config_);
    
    // Try to load non-existent file
    bool success = manager.loadCalibration("nonexistent", "/nonexistent/path.yaml");
    EXPECT_FALSE(success);
    
    // Try to get calibration that doesn't exist
    auto calibration = manager.getCalibration("nonexistent");
    EXPECT_EQ(calibration, nullptr);
    
    // Check statistics reflect the failures
    auto stats = manager.getStats();
    EXPECT_GT(stats.file_errors.load(), 0);
    EXPECT_GT(stats.cache_misses.load(), 0);
}

// Performance test to ensure <10μs access time requirement
TEST_F(CalibrationManagerTest, AccessTimeRequirement) {
    CalibrationManager manager(config_);
    manager.loadCalibration("test_camera", test_yaml_path_.string());
    
    // Warm up cache
    for (int i = 0; i < 100; ++i) {
        manager.getCalibration("test_camera");
    }
    
    // Measure access time for single access
    auto start = std::chrono::high_resolution_clock::now();
    auto calibration = manager.getCalibration("test_camera");
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double time_us = static_cast<double>(duration.count()) / 1000.0;
    
    EXPECT_NE(calibration, nullptr);
    EXPECT_LT(time_us, 10.0);  // <10μs requirement
    
    std::cout << "Access time: " << time_us << " μs" << std::endl;
}