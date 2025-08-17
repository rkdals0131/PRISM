#pragma once

#include <atomic>
#include <chrono>
#include <vector>
#include <string>
#include <functional>
#include <type_traits>
#include <sstream>
#include <cmath>
#include <limits>
#include <yaml-cpp/yaml.h>

namespace prism {
namespace utils {

/**
 * @brief Base class for all result types with common metadata
 * 
 * Provides timestamp, processing time tracking, and clear() interface
 * for all processing result types in PRISM
 */
class BaseResult {
public:
    using TimePoint = std::chrono::high_resolution_clock::time_point;
    using Duration = std::chrono::microseconds;
    
    TimePoint timestamp;
    Duration processing_time{0};
    size_t input_count{0};
    size_t output_count{0};
    
    BaseResult() : timestamp(std::chrono::high_resolution_clock::now()) {}
    virtual ~BaseResult() = default;
    
    /**
     * @brief Clear all data and reset to initial state
     */
    virtual void clear() {
        timestamp = std::chrono::high_resolution_clock::now();
        processing_time = Duration{0};
        input_count = 0;
        output_count = 0;
    }
    
    /**
     * @brief Calculate processing rate
     * @return Items processed per second
     */
    double getThroughput() const {
        if (processing_time.count() == 0 || output_count == 0) return 0.0;
        double seconds = std::chrono::duration<double>(processing_time).count();
        return output_count / seconds;
    }
    
    /**
     * @brief Get processing time in milliseconds
     */
    double getProcessingTimeMs() const {
        return std::chrono::duration<double, std::milli>(processing_time).count();
    }
};

/**
 * @brief Thread-safe atomic counter for metrics
 */
class AtomicCounter {
public:
    AtomicCounter(int64_t initial = 0) : value_(initial) {}
    
    // Copy constructor for snapshots
    AtomicCounter(const AtomicCounter& other) 
        : value_(other.value_.load(std::memory_order_acquire)) {}
    
    // Assignment operator
    AtomicCounter& operator=(const AtomicCounter& other) {
        if (this != &other) {
            value_.store(other.value_.load(std::memory_order_acquire), 
                        std::memory_order_release);
        }
        return *this;
    }
    
    void increment(int64_t delta = 1) { 
        value_.fetch_add(delta, std::memory_order_relaxed); 
    }
    
    void decrement(int64_t delta = 1) { 
        value_.fetch_sub(delta, std::memory_order_relaxed); 
    }
    
    int64_t get() const { 
        return value_.load(std::memory_order_relaxed); 
    }
    
    void set(int64_t val) { 
        value_.store(val, std::memory_order_relaxed); 
    }
    
    void reset() { 
        value_.store(0, std::memory_order_relaxed); 
    }
    
    // For copying stats snapshots
    int64_t snapshot() const { 
        return value_.load(std::memory_order_acquire); 
    }
    
    // Operators for convenience
    int64_t operator++() {  // prefix - return new value
        return value_.fetch_add(1, std::memory_order_relaxed) + 1;
    }
    
    int64_t operator++(int) {  // postfix - return old value
        return value_.fetch_add(1, std::memory_order_relaxed);
    }
    
    int64_t operator--() {  // prefix - return new value
        return value_.fetch_sub(1, std::memory_order_relaxed) - 1;
    }
    
    int64_t operator--(int) {  // postfix - return old value
        return value_.fetch_sub(1, std::memory_order_relaxed);
    }
    
    // Comparison operators
    bool operator>(int64_t val) const {
        return value_.load(std::memory_order_relaxed) > val;
    }
    
    bool operator<(int64_t val) const {
        return value_.load(std::memory_order_relaxed) < val;
    }
    
    bool operator>=(int64_t val) const {
        return value_.load(std::memory_order_relaxed) >= val;
    }
    
    bool operator<=(int64_t val) const {
        return value_.load(std::memory_order_relaxed) <= val;
    }
    
    bool operator==(int64_t val) const {
        return value_.load(std::memory_order_relaxed) == val;
    }
    
    bool operator!=(int64_t val) const {
        return value_.load(std::memory_order_relaxed) != val;
    }
    
    // Allow access to underlying atomic for advanced operations
    std::atomic<int64_t>& atomic() { return value_; }
    const std::atomic<int64_t>& atomic() const { return value_; }
    
    // Direct load/store with memory order control
    int64_t load(std::memory_order order = std::memory_order_seq_cst) const {
        return value_.load(order);
    }
    
    void store(int64_t val, std::memory_order order = std::memory_order_seq_cst) {
        value_.store(val, order);
    }
    
    bool compare_exchange_weak(int64_t& expected, int64_t desired,
                               std::memory_order order = std::memory_order_seq_cst) {
        return value_.compare_exchange_weak(expected, desired, order);
    }
    
private:
    std::atomic<int64_t> value_;
};

/**
 * @brief Thread-safe atomic gauge (can go up and down)
 */
class AtomicGauge {
public:
    AtomicGauge(double initial = 0.0) : value_(initial) {}
    
    // Copy constructor for snapshots
    AtomicGauge(const AtomicGauge& other)
        : value_(other.value_.load(std::memory_order_acquire)) {}
    
    // Assignment operator
    AtomicGauge& operator=(const AtomicGauge& other) {
        if (this != &other) {
            value_.store(other.value_.load(std::memory_order_acquire),
                        std::memory_order_release);
        }
        return *this;
    }
    
    void set(double val) { 
        value_.store(val, std::memory_order_relaxed); 
    }
    
    double get() const { 
        return value_.load(std::memory_order_relaxed); 
    }
    
    void reset() { 
        value_.store(0.0, std::memory_order_relaxed); 
    }
    
    double snapshot() const { 
        return value_.load(std::memory_order_acquire); 
    }
    
    void updateMax(double val) {
        double current = value_.load(std::memory_order_relaxed);
        while (val > current) {
            if (value_.compare_exchange_weak(current, val, 
                                            std::memory_order_relaxed)) {
                break;
            }
        }
    }
    
    void updateMin(double val) {
        double current = value_.load(std::memory_order_relaxed);
        while (val < current) {
            if (value_.compare_exchange_weak(current, val, 
                                            std::memory_order_relaxed)) {
                break;
            }
        }
    }
    
    // Comparison operators
    bool operator>(double val) const {
        return value_.load(std::memory_order_relaxed) > val;
    }
    
    bool operator<(double val) const {
        return value_.load(std::memory_order_relaxed) < val;
    }
    
    bool operator>=(double val) const {
        return value_.load(std::memory_order_relaxed) >= val;
    }
    
    bool operator<=(double val) const {
        return value_.load(std::memory_order_relaxed) <= val;
    }
    
    bool operator==(double val) const {
        return value_.load(std::memory_order_relaxed) == val;
    }
    
    bool operator!=(double val) const {
        return value_.load(std::memory_order_relaxed) != val;
    }
    
private:
    std::atomic<double> value_;
};

/**
 * @brief Base class for metrics/statistics with common operations
 * 
 * Provides CRTP (Curiously Recurring Template Pattern) for type-safe
 * copy constructor and assignment operator generation
 */
template<typename Derived>
class BaseMetrics {
public:
    /**
     * @brief Reset all metrics to initial state
     * Derived class must implement resetImpl()
     */
    void reset() {
        static_cast<Derived*>(this)->resetImpl();
    }
    
    /**
     * @brief Create a snapshot copy of current metrics
     * Automatically handles atomic variable copying
     */
    Derived snapshot() const {
        return Derived(*static_cast<const Derived*>(this));
    }
    
protected:
    BaseMetrics() = default;
    ~BaseMetrics() = default;
};

/**
 * @brief Throughput calculator for performance metrics
 */
class ThroughputMetric {
public:
    ThroughputMetric() = default;
    
    // Copy constructor for snapshots
    ThroughputMetric(const ThroughputMetric& other)
        : total_duration_ms_(other.total_duration_ms_.load(std::memory_order_acquire))
        , total_items_(other.total_items_.load(std::memory_order_acquire))
        , operation_count_(other.operation_count_.load(std::memory_order_acquire)) {}
    
    // Assignment operator
    ThroughputMetric& operator=(const ThroughputMetric& other) {
        if (this != &other) {
            total_duration_ms_.store(other.total_duration_ms_.load(std::memory_order_acquire),
                                    std::memory_order_release);
            total_items_.store(other.total_items_.load(std::memory_order_acquire),
                              std::memory_order_release);
            operation_count_.store(other.operation_count_.load(std::memory_order_acquire),
                                  std::memory_order_release);
        }
        return *this;
    }
    
    void recordOperation(double duration_ms, size_t items_processed = 1) {
        // Use compare_exchange for atomic double addition
        double current = total_duration_ms_.load(std::memory_order_relaxed);
        double new_val = current + duration_ms;
        while (!total_duration_ms_.compare_exchange_weak(current, new_val, 
                                                         std::memory_order_relaxed)) {
            new_val = current + duration_ms;
        }
        
        total_items_.fetch_add(items_processed, std::memory_order_relaxed);
        operation_count_.fetch_add(1, std::memory_order_relaxed);
    }
    
    double getThroughput() const {
        double duration = total_duration_ms_.load(std::memory_order_relaxed);
        if (duration <= 0) return 0.0;
        
        size_t items = total_items_.load(std::memory_order_relaxed);
        return (items * 1000.0) / duration; // items per second
    }
    
    double getAverageLatency() const {
        size_t ops = operation_count_.load(std::memory_order_relaxed);
        if (ops == 0) return 0.0;
        
        double duration = total_duration_ms_.load(std::memory_order_relaxed);
        return duration / ops;
    }
    
    void reset() {
        total_duration_ms_.store(0, std::memory_order_relaxed);
        total_items_.store(0, std::memory_order_relaxed);
        operation_count_.store(0, std::memory_order_relaxed);
    }
    
    struct Snapshot {
        double throughput;
        double avg_latency;
        size_t total_items;
        size_t operation_count;
    };
    
    Snapshot getSnapshot() const {
        return {
            getThroughput(),
            getAverageLatency(),
            total_items_.load(std::memory_order_acquire),
            operation_count_.load(std::memory_order_acquire)
        };
    }
    
private:
    std::atomic<double> total_duration_ms_{0};
    std::atomic<size_t> total_items_{0};
    std::atomic<size_t> operation_count_{0};
};

/**
 * @brief Configuration loader utility for consistent YAML parsing
 */
class ConfigLoader {
public:
    /**
     * @brief Load configuration with validation
     * @tparam T Configuration type with loadFromYaml() method
     */
    template<typename T>
    static bool load(T& config, const YAML::Node& node, 
                     std::function<bool(const T&)> validator = nullptr) {
        try {
            // Call type-specific loading
            config.loadFromYaml(node);
            
            // Optional validation
            if (validator && !validator(config)) {
                return false;
            }
            
            return true;
        } catch (const YAML::Exception& e) {
            // Log error
            return false;
        }
    }
    
    /**
     * @brief Helper to read value with default
     */
    template<typename T>
    static T readParam(const YAML::Node& node, const std::string& key, 
                      const T& default_value) {
        if (node[key]) {
            return node[key].as<T>();
        }
        return default_value;
    }
    
    /**
     * @brief Read nested parameter with path like "a.b.c"
     */
    template<typename T>
    static T readNestedParam(const YAML::Node& node, const std::string& path, 
                            const T& default_value) {
        std::vector<std::string> keys;
        std::stringstream ss(path);
        std::string key;
        
        while (std::getline(ss, key, '.')) {
            keys.push_back(key);
        }
        
        YAML::Node current = node;
        for (const auto& k : keys) {
            if (!current[k]) {
                return default_value;
            }
            current = current[k];
        }
        
        return current.as<T>();
    }
};

/**
 * @brief RAII timer for automatic duration measurement
 */
class ScopedTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::microseconds;
    
    explicit ScopedTimer(Duration& output)
        : output_(output), start_(Clock::now()) {}
    
    ~ScopedTimer() {
        output_ = std::chrono::duration_cast<Duration>(Clock::now() - start_);
    }
    
    // Delete copy/move to ensure single measurement
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    
private:
    Duration& output_;
    TimePoint start_;
};

/**
 * @brief Statistics aggregator for common statistical operations
 */
class StatsAggregator {
public:
    void addSample(double value) {
        sum_ += value;
        sum_squared_ += value * value;
        count_++;
        
        if (value < min_) min_ = value;
        if (value > max_) max_ = value;
    }
    
    double getMean() const {
        return count_ > 0 ? sum_ / count_ : 0.0;
    }
    
    double getStdDev() const {
        if (count_ <= 1) return 0.0;
        double mean = getMean();
        double variance = (sum_squared_ / count_) - (mean * mean);
        return std::sqrt(std::max(0.0, variance));
    }
    
    double getMin() const { return min_; }
    double getMax() const { return max_; }
    size_t getCount() const { return count_; }
    
    void reset() {
        sum_ = 0.0;
        sum_squared_ = 0.0;
        count_ = 0;
        min_ = std::numeric_limits<double>::max();
        max_ = std::numeric_limits<double>::lowest();
    }
    
private:
    double sum_{0.0};
    double sum_squared_{0.0};
    size_t count_{0};
    double min_{std::numeric_limits<double>::max()};
    double max_{std::numeric_limits<double>::lowest()};
};

} // namespace utils
} // namespace prism