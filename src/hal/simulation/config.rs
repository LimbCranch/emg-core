//! Simulation configuration structures
//! Location: src/hal/simulation/config.rs

use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SimulationConfig {
    pub profile_name: String,
    pub sample_rate_hz: u32,
    pub channel_count: usize,
    pub muscle_config: MuscleConfig,
    pub noise_config: NoiseConfig,
    pub artifact_config: ArtifactConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MuscleConfig {
    pub fiber_recruitment_curve: f32,
    pub maximum_voluntary_contraction: f32,
    pub fatigue_rate: f32,
    pub recovery_rate: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NoiseConfig {
    pub thermal_noise_power_dbm: f32,
    pub powerline_frequency_hz: f32,
    pub powerline_amplitude_factor: f32,
    pub electrode_impedance_base: f32,
    pub electrode_impedance_variance: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ArtifactConfig {
    pub motion_artifact_probability: f32,
    pub motion_artifact_amplitude: f32,
    pub electrode_pop_probability: f32,
    pub cable_movement_probability: f32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            profile_name: "healthy_user".to_string(),
            sample_rate_hz: 2000,
            channel_count: 8,
            muscle_config: MuscleConfig::default(),
            noise_config: NoiseConfig::default(),
            artifact_config: ArtifactConfig::default(),
        }
    }
}

impl Default for MuscleConfig {
    fn default() -> Self {
        Self {
            fiber_recruitment_curve: 0.3,
            maximum_voluntary_contraction: 1.0,
            fatigue_rate: 0.01,
            recovery_rate: 0.05,
        }
    }
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            thermal_noise_power_dbm: -60.0,
            powerline_frequency_hz: 50.0,
            powerline_amplitude_factor: 0.02,
            electrode_impedance_base: 5000.0,
            electrode_impedance_variance: 1000.0,
        }
    }
}

impl Default for ArtifactConfig {
    fn default() -> Self {
        Self {
            motion_artifact_probability: 0.02,
            motion_artifact_amplitude: 0.1,
            electrode_pop_probability: 0.001,
            cable_movement_probability: 0.005,
        }
    }
}