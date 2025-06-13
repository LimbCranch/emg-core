//! Predefined simulation profiles for different scenarios
//! Location: src/hal/simulation/profiles.rs

use super::config::{ArtifactConfig, MuscleConfig, NoiseConfig, SimulationConfig};
use crate::hal::types::GestureType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SimulationProfile {
    pub name: String,
    pub description: String,
    pub muscle_characteristics: MuscleCharacteristics,
    pub noise_characteristics: NoiseCharacteristics,
    pub gesture_library: Vec<GesturePattern>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MuscleCharacteristics {
    pub fiber_composition: FiberComposition,
    pub strength_level: StrengthLevel,
    pub fatigue_resistance: FatigueResistance,
    pub neural_efficiency: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NoiseCharacteristics {
    pub environment_type: EnvironmentType,
    pub electrode_quality: ElectrodeQuality,
    pub signal_conditioning: SignalConditioningQuality,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GesturePattern {
    pub gesture_type: GestureType,
    pub activation_profile: ActivationProfile,
    pub variability_factor: f32,
    pub success_rate: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum FiberComposition {
    EnduranceAthlete,    // High Type I
    PowerAthlete,        // High Type II
    Balanced,            // Mixed composition
    Elderly,             // Reduced Type II
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum StrengthLevel {
    Low,
    Normal,
    High,
    Athlete,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum FatigueResistance {
    Low,
    Normal,
    High,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum EnvironmentType {
    Clinical,           // Low interference
    Home,              // Moderate interference
    Industrial,        // High interference
    Mobile,            // Variable interference
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ElectrodeQuality {
    Research,          // Low impedance, stable
    Clinical,          // Good quality
    Consumer,          // Moderate quality
    Poor,              // High impedance, unstable
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum SignalConditioningQuality {
    Professional,      // High-end amplifiers
    Standard,          // Typical medical grade
    Consumer,          // Basic amplification
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ActivationProfile {
    pub onset_time_ms: f32,
    pub peak_time_ms: f32,
    pub offset_time_ms: f32,
    pub amplitude_consistency: f32,
}

impl SimulationProfile {
    /// Healthy adult user in clinical environment
    pub fn healthy_user() -> Self {
        Self {
            name: "healthy_user".to_string(),
            description: "Healthy adult with normal muscle function in clinical setting".to_string(),
            muscle_characteristics: MuscleCharacteristics {
                fiber_composition: FiberComposition::Balanced,
                strength_level: StrengthLevel::Normal,
                fatigue_resistance: FatigueResistance::Normal,
                neural_efficiency: 0.85,
            },
            noise_characteristics: NoiseCharacteristics {
                environment_type: EnvironmentType::Clinical,
                electrode_quality: ElectrodeQuality::Clinical,
                signal_conditioning: SignalConditioningQuality::Standard,
            },
            gesture_library: Self::create_standard_gesture_library(),
        }
    }

    /// Amputee baseline with residual limb characteristics
    pub fn amputee_baseline() -> Self {
        Self {
            name: "amputee_baseline".to_string(),
            description: "Below-elbow amputee with good residual muscle function".to_string(),
            muscle_characteristics: MuscleCharacteristics {
                fiber_composition: FiberComposition::Balanced,
                strength_level: StrengthLevel::Normal,
                fatigue_resistance: FatigueResistance::High, // Often better due to training
                neural_efficiency: 0.75, // Slightly reduced due to amputation
            },
            noise_characteristics: NoiseCharacteristics {
                environment_type: EnvironmentType::Home,
                electrode_quality: ElectrodeQuality::Consumer,
                signal_conditioning: SignalConditioningQuality::Consumer,
            },
            gesture_library: Self::create_amputee_gesture_library(),
        }
    }

    /// Stress test profile with challenging conditions
    pub fn stress_test() -> Self {
        Self {
            name: "stress_test".to_string(),
            description: "Challenging conditions for system stress testing".to_string(),
            muscle_characteristics: MuscleCharacteristics {
                fiber_composition: FiberComposition::Elderly,
                strength_level: StrengthLevel::Low,
                fatigue_resistance: FatigueResistance::Low,
                neural_efficiency: 0.60,
            },
            noise_characteristics: NoiseCharacteristics {
                environment_type: EnvironmentType::Industrial,
                electrode_quality: ElectrodeQuality::Poor,
                signal_conditioning: SignalConditioningQuality::Consumer,
            },
            gesture_library: Self::create_stress_test_gesture_library(),
        }
    }

    /// Athletic user with enhanced muscle characteristics
    pub fn athletic_user() -> Self {
        Self {
            name: "athletic_user".to_string(),
            description: "Athletic user with enhanced muscle control and strength".to_string(),
            muscle_characteristics: MuscleCharacteristics {
                fiber_composition: FiberComposition::PowerAthlete,
                strength_level: StrengthLevel::Athlete,
                fatigue_resistance: FatigueResistance::High,
                neural_efficiency: 0.95,
            },
            noise_characteristics: NoiseCharacteristics {
                environment_type: EnvironmentType::Clinical,
                electrode_quality: ElectrodeQuality::Research,
                signal_conditioning: SignalConditioningQuality::Professional,
            },
            gesture_library: Self::create_athletic_gesture_library(),
        }
    }

    pub fn to_simulation_config(&self) -> SimulationConfig {
        SimulationConfig {
            profile_name: self.name.clone(),
            sample_rate_hz: 2000,
            channel_count: 8,
            muscle_config: self.create_muscle_config(),
            noise_config: self.create_noise_config(),
            artifact_config: self.create_artifact_config(),
        }
    }

    fn create_muscle_config(&self) -> MuscleConfig {
        let (fatigue_rate, recovery_rate) = match self.muscle_characteristics.fatigue_resistance {
            FatigueResistance::Low => (0.02, 0.03),
            FatigueResistance::Normal => (0.01, 0.05),
            FatigueResistance::High => (0.005, 0.08),
        };

        let mvc_factor = match self.muscle_characteristics.strength_level {
            StrengthLevel::Low => 0.6,
            StrengthLevel::Normal => 1.0,
            StrengthLevel::High => 1.3,
            StrengthLevel::Athlete => 1.6,
        };

        MuscleConfig {
            fiber_recruitment_curve: self.muscle_characteristics.neural_efficiency * 0.4,
            maximum_voluntary_contraction: mvc_factor,
            fatigue_rate,
            recovery_rate,
        }
    }

    fn create_noise_config(&self) -> NoiseConfig {
        let (thermal_noise, powerline_freq, powerline_amp) = match self.noise_characteristics.environment_type {
            EnvironmentType::Clinical => (-65.0, 50.0, 0.01),
            EnvironmentType::Home => (-60.0, 50.0, 0.02),
            EnvironmentType::Industrial => (-55.0, 50.0, 0.05),
            EnvironmentType::Mobile => (-58.0, 50.0, 0.03),
        };

        let (base_impedance, impedance_var) = match self.noise_characteristics.electrode_quality {
            ElectrodeQuality::Research => (2000.0, 200.0),
            ElectrodeQuality::Clinical => (5000.0, 1000.0),
            ElectrodeQuality::Consumer => (8000.0, 2000.0),
            ElectrodeQuality::Poor => (15000.0, 5000.0),
        };

        NoiseConfig {
            thermal_noise_power_dbm: thermal_noise,
            powerline_frequency_hz: powerline_freq,
            powerline_amplitude_factor: powerline_amp,
            electrode_impedance_base: base_impedance,
            electrode_impedance_variance: impedance_var,
        }
    }

    fn create_artifact_config(&self) -> ArtifactConfig {
        let motion_prob = match self.noise_characteristics.environment_type {
            EnvironmentType::Clinical => 0.005,
            EnvironmentType::Home => 0.015,
            EnvironmentType::Industrial => 0.030,
            EnvironmentType::Mobile => 0.040,
        };

        ArtifactConfig {
            motion_artifact_probability: motion_prob,
            motion_artifact_amplitude: 0.15,
            electrode_pop_probability: 0.0005,
            cable_movement_probability: motion_prob * 0.5,
        }
    }

    fn create_standard_gesture_library() -> Vec<GesturePattern> {
        vec![
            GesturePattern {
                gesture_type: GestureType::HandClose,
                activation_profile: ActivationProfile {
                    onset_time_ms: 80.0,
                    peak_time_ms: 200.0,
                    offset_time_ms: 150.0,
                    amplitude_consistency: 0.9,
                },
                variability_factor: 0.15,
                success_rate: 0.95,
            },
            GesturePattern {
                gesture_type: GestureType::HandOpen,
                activation_profile: ActivationProfile {
                    onset_time_ms: 120.0,
                    peak_time_ms: 250.0,
                    offset_time_ms: 180.0,
                    amplitude_consistency: 0.85,
                },
                variability_factor: 0.20,
                success_rate: 0.92,
            },
            GesturePattern {
                gesture_type: GestureType::WristFlexion,
                activation_profile: ActivationProfile {
                    onset_time_ms: 60.0,
                    peak_time_ms: 180.0,
                    offset_time_ms: 120.0,
                    amplitude_consistency: 0.88,
                },
                variability_factor: 0.18,
                success_rate: 0.93,
            },
            GesturePattern {
                gesture_type: GestureType::WristExtension,
                activation_profile: ActivationProfile {
                    onset_time_ms: 70.0,
                    peak_time_ms: 190.0,
                    offset_time_ms: 130.0,
                    amplitude_consistency: 0.86,
                },
                variability_factor: 0.22,
                success_rate: 0.90,
            },
        ]
    }

    fn create_amputee_gesture_library() -> Vec<GesturePattern> {
        // Slightly modified patterns for amputee users
        Self::create_standard_gesture_library()
            .into_iter()
            .map(|mut pattern| {
                pattern.activation_profile.onset_time_ms *= 1.1; // Slightly slower
                pattern.variability_factor *= 1.2; // More variable
                pattern.success_rate *= 0.95; // Slightly lower success rate
                pattern
            })
            .collect()
    }

    fn create_stress_test_gesture_library() -> Vec<GesturePattern> {
        Self::create_standard_gesture_library()
            .into_iter()
            .map(|mut pattern| {
                pattern.activation_profile.amplitude_consistency *= 0.7; // Less consistent
                pattern.variability_factor *= 2.0; // Much more variable
                pattern.success_rate *= 0.8; // Lower success rate
                pattern
            })
            .collect()
    }

    fn create_athletic_gesture_library() -> Vec<GesturePattern> {
        Self::create_standard_gesture_library()
            .into_iter()
            .map(|mut pattern| {
                pattern.activation_profile.onset_time_ms *= 0.8; // Faster response
                pattern.activation_profile.amplitude_consistency *= 1.1; // More consistent
                pattern.variability_factor *= 0.8; // Less variable
                pattern.success_rate = (pattern.success_rate * 1.05).min(0.99); // Higher success
                pattern
            })
            .collect()
    }
}