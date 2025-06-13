//! EMG signal generation for realistic simulation
//! Location: src/hal/simulation/signal_generator.rs

use crate::hal::types::{EmgSample, QualityMetrics, GestureType};
use super::config::SimulationConfig;
use super::muscle_model::MuscleModel;
use super::noise_models::NoiseModel;
use super::artifact_injection::ArtifactInjector;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct EmgSignalGenerator {
    muscle_model: MuscleModel,
    noise_model: NoiseModel,
    artifact_injector: ArtifactInjector,
    config: SimulationConfig,
    sequence_counter: AtomicU32,
    phase_accumulator: f32,
}

impl EmgSignalGenerator {
    pub fn new(config: SimulationConfig) -> Self {
        Self {
            muscle_model: MuscleModel::new(&config.muscle_config),
            noise_model: NoiseModel::new(&config.noise_config),
            artifact_injector: ArtifactInjector::new(&config.artifact_config),
            config,
            sequence_counter: AtomicU32::new(0),
            phase_accumulator: 0.0,
        }
    }

    pub fn generate_sample(&mut self, gesture: GestureType, activation_level: f32) -> EmgSample {
        let mut channels = Vec::with_capacity(self.config.channel_count);

        // Generate base EMG signal for each channel
        for channel_idx in 0..self.config.channel_count {
            let base_signal = self.muscle_model.generate_activation(
                gesture,
                activation_level,
                channel_idx
            );

            let noisy_signal = self.noise_model.add_noise(base_signal);
            let final_signal = self.artifact_injector.maybe_inject_artifact(noisy_signal);

            channels.push(final_signal);
        }

        // Update phase accumulator for time-based effects
        self.phase_accumulator += 2.0 * std::f32::consts::PI / self.config.sample_rate_hz as f32;
        if self.phase_accumulator > 2.0 * std::f32::consts::PI {
            self.phase_accumulator -= 2.0 * std::f32::consts::PI;
        }

        let quality = self.calculate_quality_metrics(&channels);


        EmgSample {
            timestamp: self.get_current_timestamp(),
            sequence: self.get_next_sequence(),
            channels,
            quality_indicators: quality,
        }
    }

    fn get_current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    }

    fn get_next_sequence(&self) -> u32 {
        self.sequence_counter.fetch_add(1, Ordering::SeqCst)
    }

    fn calculate_quality_metrics(&self, channels: &[f32]) -> QualityMetrics {
        // Calculate signal power
        let signal_power: f32 = channels.iter().map(|x| x * x).sum::<f32>() / channels.len() as f32;

        // Estimate noise floor based on configuration
        let noise_power = 10_f32.powf(self.config.noise_config.thermal_noise_power_dbm / 10.0) / 1000.0;

        // Calculate SNR in dB
        let snr_db = if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            60.0 // High SNR when no noise
        };

        // Simulate electrode impedances
        let mut impedances = Vec::with_capacity(channels.len());
        for _ in 0..channels.len() {
            let impedance = self.config.noise_config.electrode_impedance_base +
                (rand::random::<f32>() - 0.5) * self.config.noise_config.electrode_impedance_variance;
            impedances.push(impedance.max(100.0)); // Minimum realistic impedance
        }

        // Check for saturation
        let saturation_detected = channels.iter().any(|&x| x.abs() > 0.95);

        QualityMetrics {
            snr_db,
            artifact_detected: self.artifact_injector.last_injection_occurred(),
            contact_impedance_kohm: impedances,
            signal_saturation: saturation_detected,
        }
    }
}