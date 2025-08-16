#include <gtest/gtest.h>
#include "prism/core/memory_pool.hpp"
#include "prism/core/point_cloud_soa.hpp"
#include <thread>
#include <vector>
#include <chrono>
#include <random>

namespace prism {
namespace core {
namespace test {

// Simple test class for basic pool operations
struct TestObject {
    int value = 0;
    std::vector<int> data;
    
    TestObject() : value(42) {
        data.reserve(100);
    }
    
    void reserve(size_t capacity) {
        data.reserve(capacity);
    }
};

class MemoryPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create pools with different configurations
        basic_pool_ = std::make_unique<MemoryPool<TestObject>>();
        
        MemoryPool<TestObject>::Config no_growth_config;
        no_growth_config.initial_size = 5;
        no_growth_config.allow_growth = false;
        no_growth_config.block_when_empty = false;
        fixed_pool_ = std::make_unique<MemoryPool<TestObject>>(no_growth_config);
        
        cloud_pool_ = std::make_unique<MemoryPool<PointCloudSoA>>();
    }
    
    std::unique_ptr<MemoryPool<TestObject>> basic_pool_;
    std::unique_ptr<MemoryPool<TestObject>> fixed_pool_;
    std::unique_ptr<MemoryPool<PointCloudSoA>> cloud_pool_;
};

// Test basic acquire and release
TEST_F(MemoryPoolTest, BasicAcquireRelease) {
    auto obj1 = basic_pool_->acquire();
    ASSERT_TRUE(obj1);
    EXPECT_EQ(obj1->value, 42);
    
    auto obj2 = basic_pool_->acquire();
    ASSERT_TRUE(obj2);
    
    auto stats = basic_pool_->getStats();
    EXPECT_EQ(stats.total_acquisitions, 2);
    EXPECT_EQ(stats.current_usage, 2);
    
    // Objects should be automatically released when going out of scope
    obj1.reset();
    
    stats = basic_pool_->getStats();
    EXPECT_EQ(stats.total_releases, 1);
    EXPECT_EQ(stats.current_usage, 1);
}

// Test RAII automatic release
TEST_F(MemoryPoolTest, RAIIAutoRelease) {
    {
        auto obj = basic_pool_->acquire();
        ASSERT_TRUE(obj);
        // Object should be released when leaving scope
    }
    
    auto stats = basic_pool_->getStats();
    EXPECT_EQ(stats.total_acquisitions, 1);
    EXPECT_EQ(stats.total_releases, 1);
    EXPECT_EQ(stats.current_usage, 0);
}

// Test pool exhaustion with no growth
TEST_F(MemoryPoolTest, PoolExhaustion) {
    std::vector<PooledPtr<TestObject>> objects;
    
    // Acquire all objects from fixed pool
    for (size_t i = 0; i < 5; ++i) {
        auto obj = fixed_pool_->acquire();
        ASSERT_TRUE(obj);
        objects.push_back(std::move(obj));
    }
    
    // Pool should be exhausted
    auto obj = fixed_pool_->acquire();
    EXPECT_FALSE(obj);  // Should return nullptr since non-blocking
    
    // Release one and try again
    objects.pop_back();
    obj = fixed_pool_->acquire();
    EXPECT_TRUE(obj);
}

// Test pool growth
TEST_F(MemoryPoolTest, PoolGrowth) {
    std::vector<PooledPtr<TestObject>> objects;
    
    // Acquire more than initial size
    for (size_t i = 0; i < 15; ++i) {
        auto obj = basic_pool_->acquire();
        ASSERT_TRUE(obj);
        objects.push_back(std::move(obj));
    }
    
    auto stats = basic_pool_->getStats();
    EXPECT_GT(stats.growth_count, 0);
    EXPECT_EQ(stats.current_usage, 15);
}

// Test concurrent access
TEST_F(MemoryPoolTest, ConcurrentAccess) {
    const size_t num_threads = 4;
    const size_t ops_per_thread = 100;
    std::atomic<size_t> successful_acquisitions{0};
    
    auto worker = [this, &successful_acquisitions]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> hold_time(1, 10);
        
        for (size_t i = 0; i < ops_per_thread; ++i) {
            auto obj = basic_pool_->acquire();
            if (obj) {
                successful_acquisitions++;
                // Hold object for random time
                std::this_thread::sleep_for(
                    std::chrono::microseconds(hold_time(gen)));
            }
            // Object automatically released
        }
    };
    
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    auto stats = basic_pool_->getStats();
    EXPECT_EQ(stats.total_acquisitions, successful_acquisitions);
    EXPECT_EQ(stats.total_releases, successful_acquisitions);
    EXPECT_EQ(stats.current_usage, 0);  // All should be released
}

// Test PointCloudSoA pool
TEST_F(MemoryPoolTest, PointCloudSoAPool) {
    auto cloud1 = cloud_pool_->acquire(1000);  // With capacity hint
    ASSERT_TRUE(cloud1);
    
    // Add some points
    cloud1->addPoint(1.0f, 2.0f, 3.0f, 0.5f);
    cloud1->addPoint(4.0f, 5.0f, 6.0f, 0.7f);
    EXPECT_EQ(cloud1->size(), 2);
    
    auto cloud2 = cloud_pool_->acquire();
    ASSERT_TRUE(cloud2);
    
    // Enable optional arrays
    cloud2->enableColor();
    cloud2->enableRing();
    
    // Add a point to ensure arrays are properly sized
    cloud2->addPoint(7.0f, 8.0f, 9.0f, 0.3f, 255, 128, 64, 5);
    EXPECT_TRUE(cloud2->hasColor());
    EXPECT_TRUE(cloud2->hasRing());
    EXPECT_EQ(cloud2->size(), 1);
    
    // Test that clouds are independent
    EXPECT_NE(cloud1->size(), cloud2->size());
}

// Test memory pool with blocking wait
TEST_F(MemoryPoolTest, BlockingWait) {
    MemoryPool<TestObject>::Config blocking_config;
    blocking_config.initial_size = 1;
    blocking_config.allow_growth = false;
    blocking_config.block_when_empty = true;
    
    MemoryPool<TestObject> blocking_pool(blocking_config);
    
    auto obj1 = blocking_pool.acquire();
    ASSERT_TRUE(obj1);
    
    std::atomic<bool> acquired{false};
    std::thread waiter([&blocking_pool, &acquired]() {
        auto obj = blocking_pool.acquire();  // Should block until obj1 is released
        EXPECT_TRUE(obj);
        acquired = true;
    });
    
    // Give waiter time to block
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_FALSE(acquired);  // Should still be waiting
    
    // Release object, waiter should proceed
    obj1.reset();
    waiter.join();
    
    EXPECT_TRUE(acquired);
}

// Test peak usage tracking
TEST_F(MemoryPoolTest, PeakUsageTracking) {
    std::vector<PooledPtr<TestObject>> objects;
    
    // Acquire 5 objects
    for (size_t i = 0; i < 5; ++i) {
        objects.push_back(basic_pool_->acquire());
    }
    
    auto stats = basic_pool_->getStats();
    EXPECT_EQ(stats.peak_usage, 5);
    
    // Release 3
    objects.resize(2);
    
    // Acquire 2 more
    for (size_t i = 0; i < 2; ++i) {
        objects.push_back(basic_pool_->acquire());
    }
    
    stats = basic_pool_->getStats();
    EXPECT_EQ(stats.peak_usage, 5);  // Peak should remain at 5
    EXPECT_EQ(stats.current_usage, 4);
}

// Test that memory is reused
TEST_F(MemoryPoolTest, MemoryReuse) {
    // Get an object and note its address
    void* first_addr = nullptr;
    {
        auto obj = fixed_pool_->acquire();
        first_addr = obj.get();
    }
    
    // Get another object - should reuse the same memory
    {
        auto obj = fixed_pool_->acquire();
        void* second_addr = obj.get();
        EXPECT_EQ(first_addr, second_addr);
    }
}

// Benchmark test for pool vs dynamic allocation
TEST_F(MemoryPoolTest, PerformanceComparison) {
    const size_t iterations = 1000;
    
    // Test pool allocation
    auto pool_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        auto obj = cloud_pool_->acquire();
        obj->reserve(1000);
        obj->addPoint(1.0f, 2.0f, 3.0f, 0.5f);
    }
    auto pool_end = std::chrono::high_resolution_clock::now();
    auto pool_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        pool_end - pool_start);
    
    // Test dynamic allocation
    auto dynamic_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        auto obj = std::make_unique<PointCloudSoA>();
        obj->reserve(1000);
        obj->addPoint(1.0f, 2.0f, 3.0f, 0.5f);
    }
    auto dynamic_end = std::chrono::high_resolution_clock::now();
    auto dynamic_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        dynamic_end - dynamic_start);
    
    // Pool should be faster (or at least not significantly slower)
    std::cout << "Pool allocation: " << pool_duration.count() << " us\n";
    std::cout << "Dynamic allocation: " << dynamic_duration.count() << " us\n";
    
    // Pool is typically faster, but we'll just ensure it's not terribly slow
    EXPECT_LT(pool_duration.count(), dynamic_duration.count() * 2);
}

} // namespace test
} // namespace core
} // namespace prism

// Main function for running tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}