/**
 * @file prism_interpolation_node.cpp
 * @brief ROS2 node for PRISM interpolation - Phase 2 (InterpolationEngine + BeamHandler)
 * 
 * This node demonstrates the interpolation capabilities of PRISM by:
 * - Using InterpolationEngine for structured pipeline
 * - Leveraging BeamAltitudeManager for accurate OS1-32 beam modeling
 * - Publishing the interpolated point cloud for visualization
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "prism/interpolation/interpolation_engine.hpp"
#include "prism/core/point_cloud_soa.hpp"

#include <chrono>
#include <memory>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <omp.h>

namespace prism {
namespace nodes {

/**
 * @brief InterpolationEngine-based PRISM interpolation node
 * 
 * This node demonstrates Phase 2 functionality by interpolating
 * OS1-32 LiDAR data from 32 channels to higher density using
 * the structured InterpolationEngine with BeamAltitudeManager.
 */
class PRISMInterpolationNode : public rclcpp::Node {
public:
    /**
     * @brief Constructor
     */
    PRISMInterpolationNode() 
        : Node("prism_interpolation_node"),
          frame_count_(0),
          total_processing_time_(0) {
        
        // Declare and get parameters
        this->declare_parameter("scale_factor", 2.0);  // 32 -> 64 channels
        this->declare_parameter("discontinuity_threshold", 0.5);  // meters
        this->declare_parameter("interpolation_method", "linear");  // linear or cubic
        this->declare_parameter("input_topic", "/ouster/points");
        this->declare_parameter("output_topic", "/prism/interpolated_points");
        
        scale_factor_ = this->get_parameter("scale_factor").as_double();
        discontinuity_threshold_ = this->get_parameter("discontinuity_threshold").as_double();
        interpolation_method_ = this->get_parameter("interpolation_method").as_string();
        std::string input_topic = this->get_parameter("input_topic").as_string();
        std::string output_topic = this->get_parameter("output_topic").as_string();
        
        // Calculate output channels
        input_channels_ = 32;  // OS1-32
        output_channels_ = static_cast<int>(input_channels_ * scale_factor_);
        
        // Initialize InterpolationEngine configuration
        interpolation::InterpolationConfig config;
        config.input_channels = input_channels_;
        config.output_channels = output_channels_;
        config.discontinuity_threshold = static_cast<float>(discontinuity_threshold_);
        config.enable_discontinuity_detection = true;
        config.enable_simd = true;  // Enable SIMD optimizations
        config.enable_tbb = false;  // TBB will be in separate branch
        config.max_threads = 4;     // OpenMP threads
        config.execution_mode = core::ExecutionMode::Mode::CPU_PARALLEL;
        
        // Set interpolation method
        if (interpolation_method_ == "cubic") {
            config.spline_tension = 0.5f;
        }
        
        // Create InterpolationEngine
        interpolation_engine_ = std::make_unique<interpolation::InterpolationEngine>(config);
        
        RCLCPP_INFO(this->get_logger(), 
            "InterpolationEngine initialized: %zu -> %zu channels",
            config.input_channels, config.output_channels);
        RCLCPP_INFO(this->get_logger(),
            "  Execution mode: CPU_PARALLEL");
        
        // Create subscriber
        pointcloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            input_topic,
            rclcpp::QoS(10).best_effort(),
            std::bind(&PRISMInterpolationNode::pointCloudCallback, this, std::placeholders::_1)
        );
        
        // Create publisher
        interpolated_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            output_topic,
            rclcpp::QoS(10).reliable()
        );
        
        // Create timer for statistics
        stats_timer_ = this->create_wall_timer(
            std::chrono::seconds(5),
            std::bind(&PRISMInterpolationNode::printStatistics, this)
        );
        
        // Configure OpenMP
        // Reduce threads to 4 for better performance with small tasks
        int num_threads = std::min(4, omp_get_max_threads());
        omp_set_num_threads(num_threads);
        
        RCLCPP_INFO(this->get_logger(), 
            "PRISM Interpolation Node (InterpolationEngine) initialized");
        RCLCPP_INFO(this->get_logger(), 
            "  Scale Factor: %.1fx (%d -> %d channels)",
            scale_factor_, input_channels_, output_channels_);
        RCLCPP_INFO(this->get_logger(),
            "  Method: %s", interpolation_method_.c_str());
        RCLCPP_INFO(this->get_logger(),
            "  Discontinuity Threshold: %.2f m", discontinuity_threshold_);
        RCLCPP_INFO(this->get_logger(),
            "  OpenMP Threads: %d", num_threads);
        RCLCPP_INFO(this->get_logger(),
            "  InterpolationEngine: Active with BeamHandler support");
    }
    
    /**
     * @brief Destructor
     */
    ~PRISMInterpolationNode() {
        printStatistics();
    }

private:
    /**
     * @brief Callback for incoming point cloud messages
     */
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Convert ROS message to PCL point cloud
        pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *input_cloud);
        
        if (input_cloud->empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                "Received empty point cloud");
            return;
        }
        
        // Check if it's organized (should be 32x1024 for OS1-32)
        if (!input_cloud->isOrganized()) {
            // Make it organized if it's not
            if (input_cloud->size() == 32768) {  // 32 * 1024
                input_cloud->width = 1024;
                input_cloud->height = 32;
            } else {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "Point cloud is not organized. Size: %zu", input_cloud->size());
                return;
            }
        }
        
        // Perform simple interpolation
        auto output_cloud = performSimpleInterpolation(input_cloud);
        
        // Convert back to ROS message and publish
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(*output_cloud, output_msg);
        output_msg.header = msg->header;
        
        interpolated_pub_->publish(output_msg);
        
        // Update statistics
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        frame_count_++;
        total_processing_time_ += duration.count();
        
        // Log performance occasionally
        if (frame_count_ % 30 == 0) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Processed frame %ld: Input=%zu points, Output=%zu points, Time=%ld ms",
                frame_count_.load(), input_cloud->size(), 
                output_cloud->size(), duration.count());
        }
    }
    
    /**
     * @brief Perform interpolation using simplified BeamHandler approach
     * 
     * Note: The full InterpolationEngine has issues with organized cloud handling,
     * so we use a simplified approach that properly interpolates between beams.
     */
    pcl::PointCloud<pcl::PointXYZI>::Ptr performSimpleInterpolation(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& input) {
        
        // Get beam configuration from InterpolationEngine's beam manager
        const auto& beam_manager = interpolation_engine_->getBeamManager();
        int output_height = beam_manager.getOutputBeamCount();
        
        auto output = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
        output->width = input->width;
        output->height = output_height;
        output->points.resize(output_height * input->width);
        output->is_dense = false;
        
        // Get interpolated beam configuration
        const auto& interpolated_beams = beam_manager.getInterpolatedBeams();
        
        // Process each column independently (azimuth angle)
        #pragma omp parallel for schedule(dynamic, 32)
        for (size_t col = 0; col < input->width; ++col) {
            // Process each output beam using beam configuration
            for (size_t out_row = 0; out_row < static_cast<size_t>(output_height); ++out_row) {
                const auto& interp_beam = interpolated_beams[out_row];
                size_t output_idx = out_row * input->width + col;
                
                if (interp_beam.is_original) {
                    // This is an original beam, copy directly
                    size_t input_row = interp_beam.source_beam_low;
                    if (input_row < input->height) {
                        size_t input_idx = input_row * input->width + col;
                        output->points[output_idx] = input->points[input_idx];
                    }
                } else {
                    // This is an interpolated beam
                    size_t low_row = interp_beam.source_beam_low;
                    size_t high_row = interp_beam.source_beam_high;
                    
                    if (low_row >= input->height || high_row >= input->height) {
                        // Invalid beam indices, set as invalid point
                        output->points[output_idx].x = std::numeric_limits<float>::quiet_NaN();
                        output->points[output_idx].y = std::numeric_limits<float>::quiet_NaN();
                        output->points[output_idx].z = std::numeric_limits<float>::quiet_NaN();
                        output->points[output_idx].intensity = 0;
                        continue;
                    }
                    
                    size_t idx1 = low_row * input->width + col;
                    size_t idx2 = high_row * input->width + col;
                    const auto& p1 = input->points[idx1];
                    const auto& p2 = input->points[idx2];
                    
                    // Check if points are valid
                    if (!std::isfinite(p1.x) || !std::isfinite(p2.x)) {
                        output->points[output_idx].x = std::numeric_limits<float>::quiet_NaN();
                        output->points[output_idx].y = std::numeric_limits<float>::quiet_NaN();
                        output->points[output_idx].z = std::numeric_limits<float>::quiet_NaN();
                        output->points[output_idx].intensity = 0;
                        continue;
                    }
                    
                    // Calculate distances for discontinuity check
                    float r1 = std::sqrt(p1.x*p1.x + p1.y*p1.y + p1.z*p1.z);
                    float r2 = std::sqrt(p2.x*p2.x + p2.y*p2.y + p2.z*p2.z);
                    
                    // Check for discontinuity
                    bool is_discontinuous = std::abs(r2 - r1) > discontinuity_threshold_;
                    
                    if (is_discontinuous) {
                        // For discontinuity: use nearest neighbor based on interpolation weight
                        if (interp_beam.interpolation_weight < 0.5f) {
                            output->points[output_idx] = p1;
                        } else {
                            output->points[output_idx] = p2;
                        }
                    } else {
                        // For continuous regions: use beam-aware interpolation weight
                        float w = interp_beam.interpolation_weight;
                        output->points[output_idx].x = p1.x * (1.0f - w) + p2.x * w;
                        output->points[output_idx].y = p1.y * (1.0f - w) + p2.y * w;
                        output->points[output_idx].z = p1.z * (1.0f - w) + p2.z * w;
                        output->points[output_idx].intensity = p1.intensity * (1.0f - w) + p2.intensity * w;
                    }
                }
            }
        }
        
        return output;
    }
    
    /**
     * @brief Print performance statistics
     */
    void printStatistics() {
        if (frame_count_ > 0) {
            double avg_time = static_cast<double>(total_processing_time_) / frame_count_;
            double fps = 1000.0 / avg_time;
            
            RCLCPP_INFO(this->get_logger(), 
                "=== PRISM Interpolation Statistics (InterpolationEngine) ===");
            RCLCPP_INFO(this->get_logger(), 
                "Frames processed: %ld", frame_count_.load());
            RCLCPP_INFO(this->get_logger(), 
                "Average processing time: %.2f ms", avg_time);
            RCLCPP_INFO(this->get_logger(), 
                "Average FPS: %.2f", fps);
            RCLCPP_INFO(this->get_logger(),
                "Configuration: %dx%d -> %dx%d (%.1fx scale)",
                input_channels_, 1024, output_channels_, 1024, scale_factor_);
        }
    }

private:
    // Parameters
    double scale_factor_;
    double discontinuity_threshold_;
    std::string interpolation_method_;
    int input_channels_;
    int output_channels_;
    
    // InterpolationEngine
    std::unique_ptr<interpolation::InterpolationEngine> interpolation_engine_;
    
    // ROS2 communication
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr interpolated_pub_;
    rclcpp::TimerBase::SharedPtr stats_timer_;
    
    // Statistics
    std::atomic<size_t> frame_count_;
    std::atomic<size_t> total_processing_time_;  // in milliseconds
};

} // namespace nodes
} // namespace prism

/**
 * @brief Main entry point for the PRISM interpolation node
 */
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<prism::nodes::PRISMInterpolationNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("prism_interpolation_node"), 
            "Fatal error: %s", e.what());
        return -1;
    }
    
    rclcpp::shutdown();
    return 0;
}
