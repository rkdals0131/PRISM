#pragma once

#include <vector>
#include <array>
#include <unordered_map>
#include <cstddef>
#include <cmath>
#include <memory>

namespace prism {
namespace interpolation {

/**
 * @brief OS1-32 LiDAR beam specifications
 */
struct BeamSpecification {
    size_t beam_id;              // Beam identifier (0-31 for OS1-32)
    float altitude_angle;        // Altitude angle in radians
    float altitude_angle_deg;    // Altitude angle in degrees
    float azimuth_resolution;    // Azimuth resolution for this beam
    bool is_valid;               // Whether this beam is valid/active
    
    BeamSpecification() : beam_id(0), altitude_angle(0), altitude_angle_deg(0), 
                         azimuth_resolution(0), is_valid(false) {}
    
    BeamSpecification(size_t id, float alt_rad, float az_res = 0.0f) 
        : beam_id(id), altitude_angle(alt_rad), 
          altitude_angle_deg(alt_rad * 180.0f / M_PI),
          azimuth_resolution(az_res), is_valid(true) {}
};

/**
 * @brief Interpolated beam configuration
 */
struct InterpolatedBeam {
    size_t interpolated_beam_id; // New beam ID for interpolated data
    float altitude_angle;        // Interpolated altitude angle
    size_t source_beam_low;      // Lower source beam ID
    size_t source_beam_high;     // Upper source beam ID
    float interpolation_weight;  // Weight for interpolation (0-1)
    bool is_original;            // True if this is an original OS1-32 beam
    
    InterpolatedBeam() : interpolated_beam_id(0), altitude_angle(0), 
                        source_beam_low(0), source_beam_high(0),
                        interpolation_weight(0), is_original(false) {}
};

/**
 * @brief Configuration for beam altitude management
 */
struct BeamAltitudeConfig {
    size_t input_beams = 32;         // OS1-32 has 32 beams
    size_t output_beams = 96;        // Target interpolated beam count
    
    // OS1-32 specific parameters
    float min_altitude_deg = -16.6f; // OS1-32 minimum altitude angle
    float max_altitude_deg = 16.6f;  // OS1-32 maximum altitude angle
    
    // Interpolation parameters
    bool preserve_original_beams = true;  // Keep original beams in output
    bool uniform_distribution = true;     // Distribute beams uniformly
    float interpolation_bias = 0.0f;      // Bias towards certain altitudes (-1 to 1)
    
    // Validation parameters
    bool validate_beam_order = true;      // Ensure beams are ordered by altitude
    float min_beam_separation_deg = 0.1f; // Minimum separation between beams
};

/**
 * @brief Statistics for beam altitude management
 */
struct BeamAltitudeStats {
    size_t original_beams_processed = 0;
    size_t interpolated_beams_generated = 0;
    size_t invalid_beams_detected = 0;
    float altitude_range_deg = 0.0f;
    float average_beam_separation_deg = 0.0f;
    float interpolation_density = 0.0f;  // Ratio of output to input beams
    
    void reset() {
        original_beams_processed = 0;
        interpolated_beams_generated = 0;
        invalid_beams_detected = 0;
        altitude_range_deg = 0.0f;
        average_beam_separation_deg = 0.0f;
        interpolation_density = 0.0f;
    }
};

/**
 * @brief Beam altitude manager for OS1-32 LiDAR interpolation
 * 
 * Manages the mapping between OS1-32's 32 physical beams and the
 * interpolated 96-beam output. Handles beam altitude calculations,
 * interpolation weight computation, and beam ordering.
 * 
 * The OS1-32 has 32 beams with altitude angles ranging from -16.6°
 * to +16.6°. This manager creates 96 virtual beams by interpolating
 * between the physical beams, tripling the vertical resolution.
 */
class BeamAltitudeManager {
public:
    /**
     * @brief Constructor with configuration
     * @param config Beam altitude configuration
     */
    explicit BeamAltitudeManager(const BeamAltitudeConfig& config = BeamAltitudeConfig());
    
    /**
     * @brief Destructor
     */
    ~BeamAltitudeManager() = default;
    
    // Delete copy operations
    BeamAltitudeManager(const BeamAltitudeManager&) = delete;
    BeamAltitudeManager& operator=(const BeamAltitudeManager&) = delete;
    
    // Allow move operations
    BeamAltitudeManager(BeamAltitudeManager&&) = default;
    BeamAltitudeManager& operator=(BeamAltitudeManager&&) = default;
    
    /**
     * @brief Configure the beam altitude manager
     * @param config New configuration
     */
    void configure(const BeamAltitudeConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    const BeamAltitudeConfig& getConfig() const noexcept { return config_; }
    
    /**
     * @brief Initialize with OS1-32 beam specifications
     * @return True if initialization succeeded
     */
    bool initializeOS132Beams();
    
    /**
     * @brief Get OS1-32 beam specification by beam ID
     * @param beam_id Beam ID (0-31)
     * @return Beam specification, invalid if beam_id is out of range
     */
    BeamSpecification getOS132Beam(size_t beam_id) const;
    
    /**
     * @brief Get all OS1-32 beam specifications
     * @return Vector of beam specifications
     */
    const std::vector<BeamSpecification>& getOS132Beams() const noexcept { return os132_beams_; }
    
    /**
     * @brief Generate interpolated beam configuration
     * @return True if generation succeeded
     */
    bool generateInterpolatedBeams();
    
    /**
     * @brief Get interpolated beam by ID
     * @param beam_id Interpolated beam ID (0-95 for 96 beams)
     * @return Interpolated beam specification
     */
    InterpolatedBeam getInterpolatedBeam(size_t beam_id) const;
    
    /**
     * @brief Get all interpolated beam specifications
     * @return Vector of interpolated beams
     */
    const std::vector<InterpolatedBeam>& getInterpolatedBeams() const noexcept { return interpolated_beams_; }
    
    /**
     * @brief Find nearest OS1-32 beams for given altitude angle
     * @param altitude_angle Target altitude angle in radians
     * @param lower_beam Output: lower beam ID
     * @param upper_beam Output: upper beam ID
     * @param weight Output: interpolation weight (0-1)
     * @return True if valid beams found
     */
    bool findNearestBeams(float altitude_angle, size_t& lower_beam, 
                         size_t& upper_beam, float& weight) const;
    
    /**
     * @brief Calculate altitude angle for interpolated beam
     * @param interpolated_beam_id Interpolated beam ID
     * @return Altitude angle in radians
     */
    float calculateInterpolatedAltitude(size_t interpolated_beam_id) const;
    
    /**
     * @brief Map original beam to interpolated beam indices
     * @param original_beam_id Original OS1-32 beam ID
     * @return Vector of interpolated beam IDs that correspond to this original beam
     */
    std::vector<size_t> mapOriginalToInterpolated(size_t original_beam_id) const;
    
    /**
     * @brief Get beam altitude statistics
     * @return Current statistics
     */
    const BeamAltitudeStats& getStats() const noexcept { return stats_; }
    
    /**
     * @brief Reset statistics
     */
    void resetStats();
    
    /**
     * @brief Validate beam configuration
     * @return True if configuration is valid
     */
    bool validateConfiguration() const;
    
    /**
     * @brief Get number of input beams (OS1-32)
     * @return Number of input beams
     */
    size_t getInputBeamCount() const noexcept { return config_.input_beams; }
    
    /**
     * @brief Get number of output beams (interpolated)
     * @return Number of output beams
     */
    size_t getOutputBeamCount() const noexcept { return config_.output_beams; }
    
    /**
     * @brief Check if manager is properly initialized
     * @return True if initialized
     */
    bool isInitialized() const noexcept { return initialized_; }

private:
    /**
     * @brief Initialize OS1-32 beam altitude angles
     */
    void initializeOS132BeamAltitudes();
    
    /**
     * @brief Sort beams by altitude angle
     */
    void sortBeamsByAltitude();
    
    /**
     * @brief Calculate interpolation weights for uniform distribution
     */
    void calculateUniformDistribution();
    
    /**
     * @brief Calculate interpolation weights with bias
     */
    void calculateBiasedDistribution();
    
    /**
     * @brief Validate beam ordering and separation
     * @return True if valid
     */
    bool validateBeamOrdering() const;
    
    /**
     * @brief Update statistics based on current configuration
     */
    void updateStats();
    
    /**
     * @brief Find beam index by altitude angle
     * @param altitude_angle Target altitude angle
     * @return Beam index, or SIZE_MAX if not found
     */
    size_t findBeamByAltitude(float altitude_angle) const;

private:
    // Configuration
    BeamAltitudeConfig config_;
    
    // OS1-32 beam specifications
    std::vector<BeamSpecification> os132_beams_;
    
    // Interpolated beam configuration
    std::vector<InterpolatedBeam> interpolated_beams_;
    
    // Lookup tables for performance
    std::unordered_map<size_t, std::vector<size_t>> original_to_interpolated_map_;
    
    // Statistics
    mutable BeamAltitudeStats stats_;
    
    // State
    bool initialized_;
    
    // Static OS1-32 altitude angles (in degrees)
    static constexpr std::array<float, 32> OS132_ALTITUDE_ANGLES_DEG = {
        16.6f, 16.0f, 15.4f, 14.8f, 14.2f, 13.6f, 13.0f, 12.4f,
        11.8f, 11.2f, 10.6f, 10.0f, 9.4f, 8.8f, 8.2f, 7.6f,
        7.0f, 6.4f, 5.8f, 5.2f, 4.6f, 4.0f, 3.4f, 2.8f,
        2.2f, 1.6f, 1.0f, 0.4f, -0.2f, -0.8f, -1.4f, -2.0f
    };
};

/**
 * @brief Utility functions for beam altitude management
 */
namespace beam_utils {
    /**
     * @brief Convert degrees to radians
     * @param degrees Angle in degrees
     * @return Angle in radians
     */
    inline float degreesToRadians(float degrees) {
        return degrees * static_cast<float>(M_PI) / 180.0f;
    }
    
    /**
     * @brief Convert radians to degrees
     * @param radians Angle in radians
     * @return Angle in degrees
     */
    inline float radiansToDegrees(float radians) {
        return radians * 180.0f / static_cast<float>(M_PI);
    }
    
    /**
     * @brief Calculate angular distance between two altitude angles
     * @param angle1 First angle in radians
     * @param angle2 Second angle in radians
     * @return Angular distance in radians
     */
    float calculateAngularDistance(float angle1, float angle2);
    
    /**
     * @brief Validate OS1-32 beam ID
     * @param beam_id Beam ID to validate
     * @return True if valid OS1-32 beam ID
     */
    inline bool isValidOS132BeamID(size_t beam_id) {
        return beam_id < 32;
    }
    
    /**
     * @brief Calculate expected beam separation for uniform distribution
     * @param min_altitude Minimum altitude angle in degrees
     * @param max_altitude Maximum altitude angle in degrees
     * @param num_beams Number of beams to distribute
     * @return Expected beam separation in degrees
     */
    float calculateUniformBeamSeparation(float min_altitude, float max_altitude, size_t num_beams);
    
    /**
     * @brief Generate test beam configuration for validation
     * @param config Configuration to use
     * @return Vector of test beam specifications
     */
    std::vector<BeamSpecification> generateTestBeamConfiguration(const BeamAltitudeConfig& config);
    
} // namespace beam_utils

} // namespace interpolation
} // namespace prism