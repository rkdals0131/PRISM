#include "prism/interpolation/beam_altitude_manager.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace prism {
namespace interpolation {

// Static altitude angles for OS1-32 (defined in header)
constexpr std::array<float, 32> BeamAltitudeManager::OS132_ALTITUDE_ANGLES_DEG;

BeamAltitudeManager::BeamAltitudeManager(const BeamAltitudeConfig& config)
    : config_(config)
    , initialized_(false)
{
    configure(config);
}

void BeamAltitudeManager::configure(const BeamAltitudeConfig& config) {
    config_ = config;
    
    if (!validateConfiguration()) {
        throw std::invalid_argument("Invalid beam altitude configuration");
    }
    initialized_ = false;
    
    // Clear existing data
    os132_beams_.clear();
    interpolated_beams_.clear();
    original_to_interpolated_map_.clear();
    resetStats();
}

bool BeamAltitudeManager::initializeOS132Beams() {
    os132_beams_.clear();
    os132_beams_.reserve(32);
    
    // Initialize with OS1-32 altitude angles
    for (size_t i = 0; i < 32; ++i) {
        float altitude_deg = OS132_ALTITUDE_ANGLES_DEG[i];
        float altitude_rad = beam_utils::degreesToRadians(altitude_deg);
        
        BeamSpecification spec(i, altitude_rad);
        spec.altitude_angle_deg = altitude_deg;
        spec.azimuth_resolution = 2.0f * static_cast<float>(M_PI) / 1024.0f; // OS1-32 has ~1024 azimuth samples
        
        os132_beams_.push_back(spec);
    }
    
    // Sort beams by altitude angle (ascending)
    sortBeamsByAltitude();
    
    stats_.original_beams_processed = os132_beams_.size();
    updateStats();
    
    return true;
}

BeamSpecification BeamAltitudeManager::getOS132Beam(size_t beam_id) const {
    if (beam_id >= os132_beams_.size()) {
        return BeamSpecification(); // Invalid beam
    }
    return os132_beams_[beam_id];
}

bool BeamAltitudeManager::generateInterpolatedBeams() {
    if (os132_beams_.empty()) {
        return false;
    }
    
    interpolated_beams_.clear();
    interpolated_beams_.reserve(config_.output_beams);
    original_to_interpolated_map_.clear();
    
    if (config_.uniform_distribution) {
        calculateUniformDistribution();
    } else {
        calculateBiasedDistribution();
    }
    
    // Build mapping from original to interpolated beams
    for (size_t i = 0; i < interpolated_beams_.size(); ++i) {
        const auto& interp_beam = interpolated_beams_[i];
        if (interp_beam.is_original) {
            original_to_interpolated_map_[interp_beam.source_beam_low].push_back(i);
        }
    }
    
    stats_.interpolated_beams_generated = interpolated_beams_.size();
    updateStats();
    
    if (!validateBeamOrdering()) {
        stats_.invalid_beams_detected++;
        return false;
    }
    
    initialized_ = true;
    return true;
}

InterpolatedBeam BeamAltitudeManager::getInterpolatedBeam(size_t beam_id) const {
    if (beam_id >= interpolated_beams_.size()) {
        return InterpolatedBeam(); // Invalid beam
    }
    return interpolated_beams_[beam_id];
}

bool BeamAltitudeManager::findNearestBeams(float altitude_angle, size_t& lower_beam, 
                                         size_t& upper_beam, float& weight) const {
    if (os132_beams_.empty()) {
        return false;
    }
    
    // Find the beam with altitude closest to but less than target
    lower_beam = 0;
    upper_beam = 0;
    
    for (size_t i = 0; i < os132_beams_.size(); ++i) {
        if (os132_beams_[i].altitude_angle <= altitude_angle) {
            lower_beam = i;
        } else {
            upper_beam = i;
            break;
        }
    }
    
    // Handle edge cases
    if (lower_beam == upper_beam) {
        if (altitude_angle < os132_beams_[0].altitude_angle) {
            // Below minimum - use first two beams
            lower_beam = 0;
            upper_beam = 1;
        } else {
            // Above maximum - use last two beams
            lower_beam = os132_beams_.size() - 2;
            upper_beam = os132_beams_.size() - 1;
        }
    }
    
    // Calculate interpolation weight
    float lower_alt = os132_beams_[lower_beam].altitude_angle;
    float upper_alt = os132_beams_[upper_beam].altitude_angle;
    
    if (std::abs(upper_alt - lower_alt) < 1e-6f) {
        weight = 0.5f; // Equal altitudes
    } else {
        weight = (altitude_angle - lower_alt) / (upper_alt - lower_alt);
        weight = std::max(0.0f, std::min(1.0f, weight));
    }
    
    return true;
}

float BeamAltitudeManager::calculateInterpolatedAltitude(size_t interpolated_beam_id) const {
    if (interpolated_beam_id >= interpolated_beams_.size()) {
        return 0.0f;
    }
    
    return interpolated_beams_[interpolated_beam_id].altitude_angle;
}

std::vector<size_t> BeamAltitudeManager::mapOriginalToInterpolated(size_t original_beam_id) const {
    auto it = original_to_interpolated_map_.find(original_beam_id);
    if (it != original_to_interpolated_map_.end()) {
        return it->second;
    }
    return std::vector<size_t>();
}

void BeamAltitudeManager::initializeOS132BeamAltitudes() {
    // Already handled in initializeOS132Beams()
}

void BeamAltitudeManager::sortBeamsByAltitude() {
    std::sort(os132_beams_.begin(), os132_beams_.end(),
              [](const BeamSpecification& a, const BeamSpecification& b) {
                  return a.altitude_angle < b.altitude_angle;
              });
    
    // Update beam IDs to match sorted order
    for (size_t i = 0; i < os132_beams_.size(); ++i) {
        os132_beams_[i].beam_id = i;
    }
}

void BeamAltitudeManager::calculateUniformDistribution() {
    if (os132_beams_.empty()) {
        return;
    }
    
    float min_altitude = os132_beams_.front().altitude_angle;
    float max_altitude = os132_beams_.back().altitude_angle;
    float altitude_range = max_altitude - min_altitude;
    
    if (altitude_range <= 0.0f) {
        return;
    }
    
    for (size_t i = 0; i < config_.output_beams; ++i) {
        InterpolatedBeam interp_beam;
        interp_beam.interpolated_beam_id = i;
        
        // Calculate target altitude for this interpolated beam
        float progress = static_cast<float>(i) / (config_.output_beams - 1);
        float target_altitude = min_altitude + progress * altitude_range;
        
        // Find nearest OS1-32 beams
        size_t lower_beam, upper_beam;
        float weight;
        if (findNearestBeams(target_altitude, lower_beam, upper_beam, weight)) {
            interp_beam.source_beam_low = lower_beam;
            interp_beam.source_beam_high = upper_beam;
            interp_beam.interpolation_weight = weight;
            interp_beam.altitude_angle = target_altitude;
            
            // Check if this matches an original beam closely
            const float angle_tolerance = beam_utils::degreesToRadians(0.1f);
            interp_beam.is_original = false;
            
            if (config_.preserve_original_beams) {
                for (const auto& original : os132_beams_) {
                    if (std::abs(original.altitude_angle - target_altitude) < angle_tolerance) {
                        interp_beam.is_original = true;
                        interp_beam.source_beam_low = original.beam_id;
                        interp_beam.interpolation_weight = 0.0f;
                        break;
                    }
                }
            }
        } else {
            // Fallback - linear interpolation
            interp_beam.altitude_angle = target_altitude;
            interp_beam.source_beam_low = 0;
            interp_beam.source_beam_high = 0;
            interp_beam.interpolation_weight = 0.0f;
            interp_beam.is_original = false;
        }
        
        interpolated_beams_.push_back(interp_beam);
    }
}

void BeamAltitudeManager::calculateBiasedDistribution() {
    // For now, use uniform distribution
    // In production, implement bias-based distribution
    calculateUniformDistribution();
    
    // Apply bias by adjusting altitude angles
    if (std::abs(config_.interpolation_bias) > 1e-6f) {
        for (auto& beam : interpolated_beams_) {
            // Apply bias as a non-linear transformation
            // Positive bias concentrates beams towards higher altitudes
            // Negative bias concentrates beams towards lower altitudes
            float normalized_alt = (beam.altitude_angle - config_.min_altitude_deg) / 
                                 (config_.max_altitude_deg - config_.min_altitude_deg);
            
            // Apply bias transformation
            float biased_alt = std::pow(normalized_alt, 1.0f + config_.interpolation_bias);
            
            beam.altitude_angle = config_.min_altitude_deg + 
                                biased_alt * (config_.max_altitude_deg - config_.min_altitude_deg);
        }
    }
}

bool BeamAltitudeManager::validateBeamOrdering() const {
    if (interpolated_beams_.size() < 2) {
        return true; // Single beam is always ordered
    }
    
    for (size_t i = 1; i < interpolated_beams_.size(); ++i) {
        float prev_alt = interpolated_beams_[i-1].altitude_angle;
        float curr_alt = interpolated_beams_[i].altitude_angle;
        
        if (curr_alt < prev_alt) {
            return false; // Not properly ordered
        }
        
        float separation = std::abs(curr_alt - prev_alt);
        float min_separation = beam_utils::degreesToRadians(config_.min_beam_separation_deg);
        
        if (separation < min_separation && separation > 1e-6f) {
            return false; // Beams too close together
        }
    }
    
    return true;
}

void BeamAltitudeManager::updateStats() {
    if (os132_beams_.empty()) {
        return;
    }
    
    float min_alt = os132_beams_.front().altitude_angle_deg;
    float max_alt = os132_beams_.back().altitude_angle_deg;
    stats_.altitude_range_deg = max_alt - min_alt;
    
    if (os132_beams_.size() > 1) {
        stats_.average_beam_separation_deg = stats_.altitude_range_deg / (os132_beams_.size() - 1);
    }
    
    if (config_.input_beams > 0) {
        stats_.interpolation_density = static_cast<float>(config_.output_beams) / config_.input_beams;
    }
}

size_t BeamAltitudeManager::findBeamByAltitude(float altitude_angle) const {
    for (size_t i = 0; i < os132_beams_.size(); ++i) {
        if (std::abs(os132_beams_[i].altitude_angle - altitude_angle) < 1e-6f) {
            return i;
        }
    }
    return SIZE_MAX; // Not found
}

bool BeamAltitudeManager::validateConfiguration() const {
    if (config_.input_beams == 0 || config_.output_beams == 0) {
        return false;
    }
    
    if (config_.output_beams < config_.input_beams) {
        return false; // Can't reduce beam count
    }
    
    if (config_.max_altitude_deg <= config_.min_altitude_deg) {
        return false;
    }
    
    if (config_.min_beam_separation_deg <= 0.0f) {
        return false;
    }
    
    if (std::abs(config_.interpolation_bias) > 2.0f) {
        return false; // Reasonable bias range
    }
    
    return true;
}

void BeamAltitudeManager::resetStats() {
    stats_.reset();
}

// Utility functions
namespace beam_utils {

float calculateAngularDistance(float angle1, float angle2) {
    float diff = std::abs(angle1 - angle2);
    return std::min(diff, 2.0f * static_cast<float>(M_PI) - diff); // Handle wrap-around
}

float calculateUniformBeamSeparation(float min_altitude, float max_altitude, size_t num_beams) {
    if (num_beams <= 1) {
        return 0.0f;
    }
    
    return (max_altitude - min_altitude) / (num_beams - 1);
}

std::vector<BeamSpecification> generateTestBeamConfiguration(const BeamAltitudeConfig& config) {
    std::vector<BeamSpecification> beams;
    beams.reserve(config.input_beams);
    
    float altitude_range = config.max_altitude_deg - config.min_altitude_deg;
    
    for (size_t i = 0; i < config.input_beams; ++i) {
        float progress = static_cast<float>(i) / (config.input_beams - 1);
        float altitude_deg = config.min_altitude_deg + progress * altitude_range;
        float altitude_rad = degreesToRadians(altitude_deg);
        
        BeamSpecification spec(i, altitude_rad);
        spec.azimuth_resolution = 2.0f * static_cast<float>(M_PI) / 1024.0f;
        beams.push_back(spec);
    }
    
    return beams;
}

} // namespace beam_utils

} // namespace interpolation
} // namespace prism