#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <new>
#include <cstddef>

namespace prism {
namespace core {

// Forward declaration
template <typename T>
class MemoryPool;

/**
 * @brief Custom deleter for pool-managed objects
 * 
 * Automatically returns objects to the pool when unique_ptr goes out of scope
 */
template <typename T>
struct PoolDeleter {
    void operator()(T* ptr) const {
        if (pool && ptr) {
            pool->release(ptr);
        }
    }
    MemoryPool<T>* pool = nullptr;
};

/**
 * @brief Type alias for smart pointer with pool deleter
 */
template <typename T>
using PooledPtr = std::unique_ptr<T, PoolDeleter<T>>;

/**
 * @brief Thread-safe memory pool for zero-copy operations
 * 
 * Provides pre-allocated objects with RAII lifetime management.
 * Uses mutex-based synchronization for v1.0 (lock-free in v1.1).
 * 
 * @tparam T Type of objects managed by the pool
 */
template <typename T>
class MemoryPool {
public:
    /**
     * @brief Configuration for memory pool behavior
     */
    struct Config {
        size_t initial_size = 10;        // Initial pool size
        size_t max_size = 100;            // Maximum pool size (0 = unlimited)
        bool allow_growth = true;         // Allow dynamic growth
        bool block_when_empty = true;     // Block vs return nullptr when empty
    };
    
    /**
     * @brief Statistics for monitoring pool usage
     */
    struct Stats {
        std::atomic<size_t> total_acquisitions{0};
        std::atomic<size_t> total_releases{0};
        std::atomic<size_t> current_usage{0};
        std::atomic<size_t> peak_usage{0};
        std::atomic<size_t> wait_count{0};        // Times threads had to wait
        std::atomic<size_t> growth_count{0};       // Times pool was grown
        
        // Copy constructor for returning stats
        Stats() = default;
        Stats(const Stats& other) 
            : total_acquisitions(other.total_acquisitions.load())
            , total_releases(other.total_releases.load())
            , current_usage(other.current_usage.load())
            , peak_usage(other.peak_usage.load())
            , wait_count(other.wait_count.load())
            , growth_count(other.growth_count.load()) {}
        
        // Assignment operator for stats
        Stats& operator=(const Stats& other) {
            total_acquisitions = other.total_acquisitions.load();
            total_releases = other.total_releases.load();
            current_usage = other.current_usage.load();
            peak_usage = other.peak_usage.load();
            wait_count = other.wait_count.load();
            growth_count = other.growth_count.load();
            return *this;
        }
    };
    
    /**
     * @brief Construct a memory pool with given configuration
     * @param config Pool configuration parameters
     */
    explicit MemoryPool(const Config& config = Config())
        : config_(config) {
        // Pre-allocate initial pool
        memory_blocks_.reserve(config.initial_size);
        grow(config.initial_size);
    }
    
    /**
     * @brief Destructor ensures all objects are properly cleaned up
     */
    ~MemoryPool() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check for outstanding allocations
        if (stats_.current_usage > 0) {
            // In production, log warning about leaked objects
            // For now, we'll just clean up
        }
        
        // Destroy all objects in the pool
        for (auto& block : memory_blocks_) {
            if (block.in_use) {
                // Call destructor for active objects
                block.asObject()->~T();
            }
        }
    }
    
    // Delete copy operations
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
    
    // Allow move operations
    MemoryPool(MemoryPool&&) = default;
    MemoryPool& operator=(MemoryPool&&) = default;
    
    /**
     * @brief Acquire an object from the pool
     * @return Smart pointer to pooled object, or nullptr if pool is empty and non-blocking
     */
    PooledPtr<T> acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Wait for available object if configured to block
        if (config_.block_when_empty) {
            condition_.wait(lock, [this] { 
                return head_of_free_list_ != nullptr || tryGrow(); 
            });
        }
        
        // Try to get from free list
        if (head_of_free_list_ == nullptr) {
            // Try to grow if allowed
            if (!tryGrow()) {
                return PooledPtr<T>(nullptr, PoolDeleter<T>{this});
            }
        }
        
        // Get the next free node
        Node* node = head_of_free_list_;
        head_of_free_list_ = *node->asNextPtr();
        
        // Construct object using placement new
        T* obj = new (node->asObject()) T();
        
        // Mark as in use
        node->in_use = true;
        
        // Update statistics
        stats_.total_acquisitions++;
        size_t current = ++stats_.current_usage;
        
        // Update peak usage
        size_t peak = stats_.peak_usage.load();
        while (current > peak && 
               !stats_.peak_usage.compare_exchange_weak(peak, current)) {
            // Retry if concurrent update
        }
        
        return PooledPtr<T>(obj, PoolDeleter<T>{this});
    }
    
    /**
     * @brief Acquire with initial capacity hint
     * @param capacity Hint for initial capacity to reserve
     * @return Smart pointer to pooled object
     */
    template<typename U = T>
    typename std::enable_if<std::is_same<U, T>::value, PooledPtr<T>>::type
    acquire(size_t capacity) {
        auto ptr = acquire();
        if (ptr) {
            // Try to call reserve if it exists
            tryReserve(ptr.get(), capacity);
        }
        return ptr;
    }
    
private:
    // Helper to call reserve if it exists
    template<typename U>
    auto tryReserve(U* obj, size_t capacity) 
        -> decltype(obj->reserve(capacity), void()) {
        obj->reserve(capacity);
    }
    
    // Fallback when reserve doesn't exist
    void tryReserve(...) {
        // Do nothing if reserve doesn't exist
    }
    
public:
    
    /**
     * @brief Release an object back to the pool
     * @param obj Pointer to object to release
     * 
     * Note: This is called automatically by PoolDeleter
     */
    void release(T* obj) {
        if (!obj) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Find the node containing this object
        Node* node = nullptr;
        for (auto& block : memory_blocks_) {
            if (block.asObject() == obj) {
                node = &block;
                break;
            }
        }
        
        if (!node || !node->in_use) {
            // Double-free or invalid pointer - this is a bug
            // In production, log error
            return;
        }
        
        // Call destructor
        obj->~T();
        
        // Clear memory for security/debugging (optional)
        // std::memset(&node->object, 0, sizeof(T));
        
        // Mark as not in use
        node->in_use = false;
        
        // Add back to free list
        *node->asNextPtr() = head_of_free_list_;
        head_of_free_list_ = node;
        
        // Update statistics
        stats_.total_releases++;
        stats_.current_usage--;
        
        // Wake up one waiting thread
        condition_.notify_one();
    }
    
    /**
     * @brief Get current pool statistics
     * @return Copy of current statistics
     */
    Stats getStats() const {
        return stats_;
    }
    
    /**
     * @brief Get current pool size
     * @return Total number of objects in pool
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return memory_blocks_.size();
    }
    
    /**
     * @brief Get number of available objects
     * @return Number of objects currently available
     */
    size_t available() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t count = 0;
        Node* current = head_of_free_list_;
        while (current) {
            count++;
            current = *current->asNextPtr();
        }
        return count;
    }
    
private:
    /**
     * @brief Internal node structure for intrusive linked list
     */
    struct Node {
        // Use aligned storage for the object
        alignas(alignof(T)) unsigned char storage[sizeof(T)];
        bool in_use = false;            // Track usage for debugging
        
        // Access as object when in use
        T* asObject() {
            return reinterpret_cast<T*>(&storage[0]);
        }
        
        // Access as free list node when not in use
        Node** asNextPtr() {
            return reinterpret_cast<Node**>(&storage[0]);
        }
        
        // Ensure proper alignment for T
        Node() : storage{}, in_use(false) {}
        ~Node() {}
    };
    
    /**
     * @brief Try to grow the pool
     * @return True if growth succeeded, false otherwise
     */
    bool tryGrow() {
        if (!config_.allow_growth) {
            return false;
        }
        
        if (config_.max_size > 0 && memory_blocks_.size() >= config_.max_size) {
            return false;
        }
        
        // Grow by 50% or to max_size
        size_t current_size = memory_blocks_.size();
        size_t growth = current_size / 2;
        if (growth < 1) growth = 1;
        
        if (config_.max_size > 0) {
            growth = std::min(growth, config_.max_size - current_size);
        }
        
        grow(growth);
        stats_.growth_count++;
        return true;
    }
    
    /**
     * @brief Grow the pool by specified amount
     * @param count Number of objects to add
     */
    void grow(size_t count) {
        size_t old_size = memory_blocks_.size();
        memory_blocks_.resize(old_size + count);
        
        // Link new nodes into free list
        for (size_t i = old_size; i < memory_blocks_.size(); ++i) {
            Node* node = &memory_blocks_[i];
            *node->asNextPtr() = head_of_free_list_;
            head_of_free_list_ = node;
        }
    }
    
    // Configuration
    Config config_;
    
    // Memory storage
    std::vector<Node> memory_blocks_;
    Node* head_of_free_list_ = nullptr;
    
    // Synchronization
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    
    // Statistics
    Stats stats_;
};

// Forward declaration
class PointCloudSoA;

/**
 * @brief Helper function to create optimal configuration for PointCloudSoA pools
 */
inline typename MemoryPool<PointCloudSoA>::Config getOptimalPointCloudPoolConfig() {
    typename MemoryPool<PointCloudSoA>::Config config;
    config.initial_size = 20;      // Good for pipeline stages
    config.max_size = 100;          // Prevent excessive memory
    config.allow_growth = true;
    config.block_when_empty = true;
    return config;
}

} // namespace core
} // namespace prism