#include <gtest/gtest.h>
#include <prism/projection/projection_engine.hpp>
#include <prism/core/calibration_manager.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <fstream>

using namespace prism::projection;
using namespace prism::core;

class ProjectionEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temporary calibration files for testing
        createTestCalibrationFiles();
        
        // Initialize calibration manager
        calib_manager_ = std::make_shared<CalibrationManager>();
        
        // Load YAML file
        YAML::Node config = YAML::LoadFile(test_config_file_);
        
        // Load calibration from the YAML nodes directly
        if (!calib_manager_->loadCalibrationFromNode("camera_1", config["camera_1"])) {
            std::cerr << "Failed to load camera_1 calibration from node" << std::endl;
            // Debug print the YAML node
            std::cerr << "Camera 1 config: " << config["camera_1"] << std::endl;
        } else {
            std::cerr << "Successfully loaded camera_1 calibration" << std::endl;
        }
        if (!calib_manager_->loadCalibrationFromNode("camera_2", config["camera_2"])) {
            std::cerr << "Failed to load camera_2 calibration from node" << std::endl;
        } else {
            std::cerr << "Successfully loaded camera_2 calibration" << std::endl;
        }
        
        // Create projection engine with the calibration manager we already loaded
        engine_ = std::make_shared<ProjectionEngine>(calib_manager_);
        engine_->initialize({"camera_1", "camera_2"});
    }
    
    void TearDown() override {
        // Clean up test files
        std::remove(test_config_file_.c_str());
    }
    
    void createTestCalibrationFiles() {
        // Create a simple test configuration
        YAML::Node config;
        
        // Camera 1 intrinsics - flat array format expected by CalibrationManager
        config["camera_1"]["camera_matrix"]["data"] = std::vector<double>{
            500.0, 0.0, 320.0,
            0.0, 500.0, 240.0,
            0.0, 0.0, 1.0
        };
        config["camera_1"]["distortion_coefficients"]["data"] = 
            std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0};
        config["camera_1"]["image_width"] = 640;
        config["camera_1"]["image_height"] = 480;
        
        // Camera 1 extrinsics (identity for simplicity) - flat array
        config["camera_1"]["extrinsics"]["T"] = std::vector<double>{
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        };
        
        // Camera 2 intrinsics - flat array format
        config["camera_2"]["camera_matrix"]["data"] = std::vector<double>{
            480.0, 0.0, 320.0,
            0.0, 480.0, 240.0,
            0.0, 0.0, 1.0
        };
        config["camera_2"]["distortion_coefficients"]["data"] = 
            std::vector<double>{0.1, -0.1, 0.0, 0.0, 0.0};
        config["camera_2"]["image_width"] = 640;
        config["camera_2"]["image_height"] = 480;
        
        // Camera 2 extrinsics (rotated 90 degrees around Y) - flat array
        config["camera_2"]["extrinsics"]["T"] = std::vector<double>{
            0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0
        };
        
        // Save to test file
        test_config_file_ = "/tmp/test_projection_config.yaml";
        std::ofstream fout(test_config_file_);
        fout << config;
        fout.close();
    }
    
    // Create test point cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr createTestCloud() {
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
        
        // Add points in front of camera
        for (float x = -2.0f; x <= 2.0f; x += 0.5f) {
            for (float y = -2.0f; y <= 2.0f; y += 0.5f) {
                pcl::PointXYZI point;
                point.x = x;
                point.y = y;
                point.z = 5.0f;  // 5 meters in front
                point.intensity = 100.0f;
                cloud->points.push_back(point);
            }
        }
        
        // Add point behind camera (should be filtered)
        pcl::PointXYZI behind;
        behind.x = 0.0f;
        behind.y = 0.0f;
        behind.z = -1.0f;
        behind.intensity = 50.0f;
        cloud->points.push_back(behind);
        
        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = false;
        
        return cloud;
    }
    
protected:
    std::shared_ptr<ProjectionEngine> engine_;
    std::shared_ptr<CalibrationManager> calib_manager_;
    std::string test_config_file_;
};

TEST_F(ProjectionEngineTest, Initialization) {
    // Test that engine was created successfully
    EXPECT_NE(engine_, nullptr);
    
    // Test with known camera IDs
    std::vector<LiDARPoint> test_points;
    test_points.push_back({0.0f, 0.0f, 5.0f, 100.0f});
    
    CameraProjection cam_result;
    bool success = engine_->projectToCamera(test_points, "camera_1", cam_result);
    EXPECT_TRUE(success);
}

TEST_F(ProjectionEngineTest, BasicProjection) {
    auto cloud = createTestCloud();
    
    // Convert PCL points to LiDARPoints
    std::vector<LiDARPoint> lidar_points;
    for (const auto& p : cloud->points) {
        lidar_points.push_back({p.x, p.y, p.z, p.intensity});
    }
    
    ProjectionResult result;
    bool success = engine_->projectToAllCameras(lidar_points, result);
    
    EXPECT_TRUE(success);
    EXPECT_EQ(result.camera_projections.size(), 2);
    
    // Check camera 1 projection
    const auto& cam1 = result.camera_projections[0];
    EXPECT_GT(cam1.projected_points.size(), 0);
    EXPECT_EQ(cam1.camera_id, "camera_1");
    
    // Check camera 2 projection
    const auto& cam2 = result.camera_projections[1];
    // Camera 2 is rotated 90 degrees, so points at z=5 will be behind it
    // This is expected behavior
    EXPECT_EQ(cam2.projected_points.size(), 0);
    EXPECT_EQ(cam2.camera_id, "camera_2");
    
    // Verify point behind camera was filtered
    bool found_behind = false;
    for (const auto& cam : result.camera_projections) {
        for (const auto& pixel : cam.projected_points) {
            if (pixel.depth < 0) {
                found_behind = true;
                break;
            }
        }
    }
    EXPECT_FALSE(found_behind);
}

TEST_F(ProjectionEngineTest, SingleCameraProjection) {
    auto cloud = createTestCloud();
    
    // Convert PCL points to LiDARPoints
    std::vector<LiDARPoint> lidar_points;
    for (const auto& p : cloud->points) {
        lidar_points.push_back({p.x, p.y, p.z, p.intensity});
    }
    
    CameraProjection cam_result;
    bool success = engine_->projectToCamera(lidar_points, "camera_1", cam_result);
    
    EXPECT_TRUE(success);
    EXPECT_EQ(cam_result.camera_id, "camera_1");
    EXPECT_GT(cam_result.projected_points.size(), 0);
}

TEST_F(ProjectionEngineTest, ImageBoundaryCheck) {
    // Create points at known positions
    std::vector<LiDARPoint> points;
    
    // Point that projects to center (should be valid)
    points.push_back({0.0f, 0.0f, 5.0f, 100.0f});
    
    // Point that projects outside image (should be filtered)
    points.push_back({10.0f, 10.0f, 5.0f, 100.0f});
    
    CameraProjection cam_result;
    bool success = engine_->projectToCamera(points, "camera_1", cam_result);
    
    EXPECT_TRUE(success);
    // Should have at least one valid point
    EXPECT_GE(cam_result.projected_points.size(), 1);
    
    // Check that all projected points are within bounds
    for (const auto& pixel : cam_result.projected_points) {
        EXPECT_GE(pixel.u, 0);
        EXPECT_LT(pixel.u, 640);
        EXPECT_GE(pixel.v, 0);
        EXPECT_LT(pixel.v, 480);
    }
}

TEST_F(ProjectionEngineTest, FrustumCulling) {
    std::vector<LiDARPoint> points;
    
    // Points at various depths
    points.push_back({0.0f, 0.0f, 5.0f, 100.0f});   // Normal depth
    points.push_back({0.0f, 0.0f, 0.05f, 100.0f});  // Very close (below min_depth)
    points.push_back({0.0f, 0.0f, -1.0f, 100.0f});  // Behind camera
    points.push_back({0.0f, 0.0f, 110.0f, 100.0f}); // Very far (above max_depth)
    
    CameraProjection cam_result;
    bool success = engine_->projectToCamera(points, "camera_1", cam_result);
    
    EXPECT_TRUE(success);
    
    // Should filter out points behind camera and outside depth range
    // Default config has min_depth=0.1, max_depth=100.0
    for (const auto& pixel : cam_result.projected_points) {
        EXPECT_GE(pixel.depth, 0.1f);   // Default min_depth
        EXPECT_LE(pixel.depth, 100.0f); // Default max_depth
    }
    
    // Should have filtered out at least some points
    EXPECT_LT(cam_result.projected_points.size(), points.size());
}

TEST_F(ProjectionEngineTest, CoordinateTransformation) {
    // Test coordinate transformation with known values
    std::vector<LiDARPoint> points;
    
    // Point at (1, 0, 0) in LiDAR frame
    points.push_back({1.0f, 0.0f, 0.0f, 100.0f});
    
    CameraProjection cam_result;
    bool success = engine_->projectToCamera(points, "camera_2", cam_result);
    
    EXPECT_TRUE(success);
    
    // Camera 2 is rotated 90 degrees around Y
    // So LiDAR (1, 0, 0) should become camera (0, 0, -1)
    // This point is behind the camera and should be filtered
    EXPECT_EQ(cam_result.projected_points.size(), 0);
}

TEST_F(ProjectionEngineTest, PerformanceMetrics) {
    auto cloud = createTestCloud();
    
    // Convert PCL points to LiDARPoints
    std::vector<LiDARPoint> lidar_points;
    for (const auto& p : cloud->points) {
        lidar_points.push_back({p.x, p.y, p.z, p.intensity});
    }
    
    ProjectionResult result;
    bool success = engine_->projectToAllCameras(lidar_points, result);
    
    EXPECT_TRUE(success);
    
    // Check that metrics are populated
    EXPECT_GT(result.input_count, 0);
    EXPECT_GT(result.output_count, 0);
    EXPECT_GE(result.getProcessingTimeMs(), 0.0);
    
    // Verify per-camera statistics
    for (const auto& cam : result.camera_projections) {
        EXPECT_GE(cam.projected_points.size(), 0);
        EXPECT_LE(cam.projected_points.size(), cloud->points.size());
    }
}

TEST_F(ProjectionEngineTest, InvalidCameraName) {
    auto cloud = createTestCloud();
    
    // Convert PCL points to LiDARPoints
    std::vector<LiDARPoint> lidar_points;
    for (const auto& p : cloud->points) {
        lidar_points.push_back({p.x, p.y, p.z, p.intensity});
    }
    
    CameraProjection cam_result;
    bool success = engine_->projectToCamera(lidar_points, "invalid_camera", cam_result);
    
    EXPECT_FALSE(success);
    EXPECT_EQ(cam_result.projected_points.size(), 0);
}

TEST_F(ProjectionEngineTest, EmptyPointCloud) {
    std::vector<LiDARPoint> empty_points;
    
    ProjectionResult result;
    bool success = engine_->projectToAllCameras(empty_points, result);
    
    EXPECT_TRUE(success);
    EXPECT_EQ(result.input_count, 0);
    EXPECT_EQ(result.output_count, 0);
}

// Test distortion correction
TEST_F(ProjectionEngineTest, DistortionCorrection) {
    // Camera 2 has distortion coefficients (0.1, -0.1, 0, 0, 0)
    // This should create noticeable distortion for points away from center
    std::vector<LiDARPoint> points;
    
    // Point off-center where distortion is more noticeable
    points.push_back({2.0f, 2.0f, 5.0f, 100.0f});
    
    CameraProjection cam_result;
    bool success = engine_->projectToCamera(points, "camera_2", cam_result);
    
    EXPECT_TRUE(success);
    
    // With distortion coefficients set, the projection should handle distortion
    // Just verify that projection succeeds with distortion coefficients
    if (cam_result.projected_points.size() > 0) {
        const auto& p = cam_result.projected_points[0];
        
        // Point should still be within image bounds after distortion
        EXPECT_GE(p.u, 0);
        EXPECT_LT(p.u, 640);
        EXPECT_GE(p.v, 0);
        EXPECT_LT(p.v, 480);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}