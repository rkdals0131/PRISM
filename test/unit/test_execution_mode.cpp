#include <gtest/gtest.h>
#include "prism/core/execution_mode.hpp"
#include "prism/core/point_cloud_soa.hpp"
#include "prism/core/memory_pool.hpp"

using namespace prism::core;

class ExecutionModeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a small test point cloud
        test_cloud.resize(100);
        for (size_t i = 0; i < test_cloud.size(); ++i) {
            test_cloud.x[i] = static_cast<float>(i) * 0.1f;
            test_cloud.y[i] = static_cast<float>(i) * 0.2f;
            test_cloud.z[i] = static_cast<float>(i) * 0.3f;
            test_cloud.intensity[i] = static_cast<float>(i) * 0.01f;
        }
        
        // Create memory pool
        typename MemoryPool<PointCloudSoA>::Config pool_config;
        pool_config.initial_size = 10;
        pool_config.max_size = 50;
        memory_pool = std::make_unique<MemoryPool<PointCloudSoA>>(pool_config);
        
        // Configure execution
        exec_config.memory_pool = memory_pool.get();
        exec_config.enable_metrics = true;
    }
    
    void TearDown() override {
        memory_pool.reset();
    }
    
    // Simple pipeline stages for testing
    PipelineStage createIdentityStage() {
        return [this](const PointCloudSoA& input) -> PooledPtr<PointCloudSoA> {
            auto result = memory_pool->acquire();
            *result = input; // Copy input to output
            return result;
        };
    }
    
    PipelineStage createScaleStage(float scale_factor) {
        return [this, scale_factor](const PointCloudSoA& input) -> PooledPtr<PointCloudSoA> {
            auto result = memory_pool->acquire();
            result->resize(input.size());
            
            for (size_t i = 0; i < input.size(); ++i) {
                result->x[i] = input.x[i] * scale_factor;
                result->y[i] = input.y[i] * scale_factor;
                result->z[i] = input.z[i] * scale_factor;
                result->intensity[i] = input.intensity[i];
            }
            
            return result;
        };
    }
    
    PointCloudSoA test_cloud;
    std::unique_ptr<MemoryPool<PointCloudSoA>> memory_pool;
    ExecutionConfig exec_config;
};

TEST_F(ExecutionModeTest, SingleThreadPipelineBasicExecution) {
    SingleThreadPipeline pipeline(exec_config);
    
    auto identity_stage = createIdentityStage();
    auto scale_stage = createScaleStage(2.0f);
    
    auto result = pipeline.execute(test_cloud, identity_stage, scale_stage, identity_stage);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->size(), test_cloud.size());
    
    // Check that scaling was applied in the projection stage
    for (size_t i = 0; i < result->size(); ++i) {
        EXPECT_FLOAT_EQ(result->x[i], test_cloud.x[i] * 2.0f);
        EXPECT_FLOAT_EQ(result->y[i], test_cloud.y[i] * 2.0f);
        EXPECT_FLOAT_EQ(result->z[i], test_cloud.z[i] * 2.0f);
    }
}

TEST_F(ExecutionModeTest, SingleThreadPipelineMetrics) {
    SingleThreadPipeline pipeline(exec_config);
    
    auto identity_stage = createIdentityStage();
    
    pipeline.execute(test_cloud, identity_stage, identity_stage, identity_stage);
    
    const auto& metrics = pipeline.getMetrics();
    
    EXPECT_GT(metrics.total_time.count(), 0);
    EXPECT_GT(metrics.interpolation_time.count(), 0);
    EXPECT_GT(metrics.projection_time.count(), 0);
    EXPECT_GT(metrics.fusion_time.count(), 0);
    EXPECT_EQ(metrics.points_processed, test_cloud.size());
    EXPECT_GT(metrics.throughput_points_per_second, 0.0);
}

TEST_F(ExecutionModeTest, ExecutionModeAutoSelection) {
    ExecutionMode exec_mode(ExecutionMode::Mode::SINGLE_THREAD, exec_config);
    
    auto identity_stage = createIdentityStage();
    
    auto result = exec_mode.execute(test_cloud, identity_stage, identity_stage, identity_stage);
    
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->size(), test_cloud.size());
    
    // Should have some valid strategy name
    EXPECT_NE(std::string(exec_mode.getCurrentStrategyName()), "None");
}

TEST_F(ExecutionModeTest, ExecutionModeModeAvailability) {
    // Single thread should always be available
    EXPECT_TRUE(ExecutionMode::isModeAvailable(ExecutionMode::Mode::SINGLE_THREAD));
}

TEST_F(ExecutionModeTest, ExecutionModeModeSettingAndSwitching) {
    ExecutionMode exec_mode(ExecutionMode::Mode::SINGLE_THREAD, exec_config);
    
    EXPECT_EQ(exec_mode.getMode(), ExecutionMode::Mode::SINGLE_THREAD);
    
    // Should remain single thread
    exec_mode.setMode(ExecutionMode::Mode::SINGLE_THREAD);
    EXPECT_EQ(exec_mode.getMode(), ExecutionMode::Mode::SINGLE_THREAD);
}


TEST_F(ExecutionModeTest, ExecutionUtilityFunctions) {
    // Test mode string conversion
    EXPECT_STREQ(execution_utils::modeToString(ExecutionMode::Mode::SINGLE_THREAD), "SINGLE_THREAD");
    
    // Test string to mode conversion
    EXPECT_EQ(execution_utils::stringToMode("SINGLE_THREAD"), ExecutionMode::Mode::SINGLE_THREAD);
    
    // Test invalid string
    EXPECT_THROW(execution_utils::stringToMode("INVALID"), std::invalid_argument);
    
    // Test system info function
    auto sys_info = execution_utils::getSystemInfo();
    EXPECT_GT(sys_info.num_cpu_cores, 0);
    EXPECT_GT(sys_info.l3_cache_size, 0);
}

TEST_F(ExecutionModeTest, OptimalModeSelection) {
    // Test optimal mode selection for different input sizes
    auto mode_small = ExecutionMode::getOptimalMode(100);
    auto mode_medium = ExecutionMode::getOptimalMode(5000);
    auto mode_large = ExecutionMode::getOptimalMode(50000);
    
    // Should return valid modes
    EXPECT_NE(mode_small, ExecutionMode::Mode::AUTO);
    EXPECT_NE(mode_medium, ExecutionMode::Mode::AUTO);
    EXPECT_NE(mode_large, ExecutionMode::Mode::AUTO);
    
    // Small inputs should prefer single thread
    EXPECT_EQ(mode_small, ExecutionMode::Mode::SINGLE_THREAD);
}

TEST_F(ExecutionModeTest, MetricsResetFunctionality) {
    ExecutionMode exec_mode(ExecutionMode::Mode::SINGLE_THREAD, exec_config);
    
    auto identity_stage = createIdentityStage();
    
    // Execute pipeline to generate metrics
    exec_mode.execute(test_cloud, identity_stage, identity_stage, identity_stage);
    
    const auto& metrics_before = exec_mode.getMetrics();
    EXPECT_GT(metrics_before.total_time.count(), 0);
    
    // Reset metrics
    exec_mode.resetMetrics();
    
    const auto& metrics_after = exec_mode.getMetrics();
    EXPECT_EQ(metrics_after.total_time.count(), 0);
    EXPECT_EQ(metrics_after.points_processed, 0);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}