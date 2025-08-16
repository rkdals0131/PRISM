#include <gtest/gtest.h>
#include <prism/projection/projection_types.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace prism::projection;

class SimpleProjectionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Simple camera setup without CalibrationManager
    }
};

TEST_F(SimpleProjectionTest, BasicTypes) {
    // Test LiDARPoint
    LiDARPoint pt(1.0f, 2.0f, 3.0f, 100.0f);
    EXPECT_EQ(pt.x, 1.0f);
    EXPECT_EQ(pt.y, 2.0f);
    EXPECT_EQ(pt.z, 3.0f);
    EXPECT_EQ(pt.intensity, 100.0f);
    
    // Test PixelPoint
    PixelPoint pixel(320.0f, 240.0f, 5.0f, 100.0f);
    EXPECT_EQ(pixel.u, 320.0f);
    EXPECT_EQ(pixel.v, 240.0f);
    EXPECT_EQ(pixel.depth, 5.0f);
    EXPECT_TRUE(pixel.isWithinBounds(640, 480));
    EXPECT_FALSE(pixel.isWithinBounds(300, 200));
}

TEST_F(SimpleProjectionTest, CameraProjection) {
    CameraProjection proj;
    proj.camera_id = "test_camera";
    
    // Add some test points
    proj.projected_points.push_back(PixelPoint(100, 100, 5.0f, 50.0f));
    proj.projected_points.push_back(PixelPoint(200, 200, 10.0f, 100.0f));
    proj.projected_points.push_back(PixelPoint(300, 300, 15.0f, 150.0f));
    
    // Compute statistics
    proj.computeStatistics();
    
    EXPECT_EQ(proj.projected_point_count, 3);
    EXPECT_FLOAT_EQ(proj.min_depth, 5.0f);
    EXPECT_FLOAT_EQ(proj.max_depth, 15.0f);
    EXPECT_FLOAT_EQ(proj.avg_depth, 10.0f);
}

TEST_F(SimpleProjectionTest, ProjectionResult) {
    ProjectionResult result;
    
    // Add camera projections
    CameraProjection cam1;
    cam1.camera_id = "camera_1";
    cam1.projected_points.push_back(PixelPoint(100, 100, 5.0f, 50.0f));
    
    CameraProjection cam2;
    cam2.camera_id = "camera_2";
    cam2.projected_points.push_back(PixelPoint(200, 200, 10.0f, 100.0f));
    
    result.camera_projections.push_back(cam1);
    result.camera_projections.push_back(cam2);
    
    // Test getCameraProjection
    auto* proj1 = result.getCameraProjection("camera_1");
    ASSERT_NE(proj1, nullptr);
    EXPECT_EQ(proj1->camera_id, "camera_1");
    
    auto* proj3 = result.getCameraProjection("camera_3");
    EXPECT_EQ(proj3, nullptr);
}

TEST_F(SimpleProjectionTest, CameraParams) {
    CameraParams params;
    params.camera_id = "test_cam";
    
    // Set camera matrix
    params.camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    params.camera_matrix.at<double>(0, 0) = 500.0;  // fx
    params.camera_matrix.at<double>(1, 1) = 500.0;  // fy
    params.camera_matrix.at<double>(0, 2) = 320.0;  // cx
    params.camera_matrix.at<double>(1, 2) = 240.0;  // cy
    
    // Set distortion
    params.distortion_coeffs = cv::Mat::zeros(5, 1, CV_64F);
    
    // Set image size
    params.image_size = cv::Size(640, 480);
    
    // Validate
    EXPECT_TRUE(params.validate());
    
    // Test invalid params
    CameraParams invalid_params;
    EXPECT_FALSE(invalid_params.validate());
}

TEST_F(SimpleProjectionTest, ProjectionConfig) {
    ProjectionConfig config;
    
    EXPECT_FLOAT_EQ(config.min_depth, 0.1f);
    EXPECT_FLOAT_EQ(config.max_depth, 100.0f);
    EXPECT_TRUE(config.enable_frustum_culling);
    EXPECT_TRUE(config.enable_distortion_correction);
    EXPECT_FALSE(config.enable_debug_visualization);
}

TEST_F(SimpleProjectionTest, SimpleProjection) {
    // Simple pinhole projection test
    cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
    K.at<double>(0, 0) = 500.0;  // fx
    K.at<double>(1, 1) = 500.0;  // fy
    K.at<double>(0, 2) = 320.0;  // cx
    K.at<double>(1, 2) = 240.0;  // cy
    
    // 3D point in camera frame
    cv::Mat point3d = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 5.0);
    
    // Project to 2D
    cv::Mat projected = K * point3d;
    double u = projected.at<double>(0) / projected.at<double>(2);
    double v = projected.at<double>(1) / projected.at<double>(2);
    
    // Point at (0,0,5) should project to image center (320, 240)
    EXPECT_NEAR(u, 320.0, 0.01);
    EXPECT_NEAR(v, 240.0, 0.01);
}

TEST_F(SimpleProjectionTest, TransformationMatrix) {
    // Test transformation
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T(0, 3) = 1.0;  // Translation in x
    
    Eigen::Vector4d point(0, 0, 0, 1);
    Eigen::Vector4d transformed = T * point;
    
    EXPECT_DOUBLE_EQ(transformed(0), 1.0);
    EXPECT_DOUBLE_EQ(transformed(1), 0.0);
    EXPECT_DOUBLE_EQ(transformed(2), 0.0);
    EXPECT_DOUBLE_EQ(transformed(3), 1.0);
}

TEST_F(SimpleProjectionTest, ProjectionStats) {
    ProjectionStats stats;
    
    // Create test results
    ProjectionResult result1;
    result1.total_input_points = 1000;
    result1.total_projected_points = 800;
    result1.processing_time_ms = 10.0;
    
    ProjectionResult result2;
    result2.total_input_points = 1000;
    result2.total_projected_points = 850;
    result2.processing_time_ms = 12.0;
    
    // Update stats
    stats.updateStats(result1);
    stats.updateStats(result2);
    
    EXPECT_EQ(stats.total_projections, 2);
    EXPECT_EQ(stats.total_points_processed, 2000);
    EXPECT_EQ(stats.total_points_projected, 1650);
    EXPECT_DOUBLE_EQ(stats.min_processing_time_ms, 10.0);
    EXPECT_DOUBLE_EQ(stats.max_processing_time_ms, 12.0);
    EXPECT_NEAR(stats.avg_processing_time_ms, 11.0, 0.01);
    EXPECT_NEAR(stats.getProjectionSuccessRate(), 82.5, 0.01);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}