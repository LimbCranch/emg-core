// src/hal/simulator.rs
//! EMG device simulator for testing and development

use crate::hal::{EmgDevice, EmgSample, DeviceInfo, DeviceCapabilities, QualityMetrics};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use std::error::Error;
use std::fmt;

/// Configuration for EMG simulator
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SimulatorConfig {
    pub profile: String,
    pub noise_level: f32,
    pub artifact_probability: f32,
    pub channel_count: usize,
    pub sample_rate_hz: u32,
    pub gesture_duration_ms: u32,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            profile: "healthy_user".to_string(),
            noise_level: 0.05,
            artifact_probability: 0.02,
            channel_count: 8,
            sample_rate_hz: 2000,
            gesture_duration_ms: 1000,
        }
    }
}

/// EMG device simulator
pub struct SimulatorDevice {
    config: SimulatorConfig,
    sample_generator: SampleGenerator,
    is_running: AtomicBool,
    sequence_counter: AtomicU32,
}

/// Internal sample generator for realistic EMG data
struct SampleGenerator {
    gesture_patterns: Vec<GesturePattern>,
    current_gesture: usize,
    gesture_progress: f32,
    noise_generator: NoiseGenerator,
}

/// EMG gesture pattern for simulation
#[derive(Debug, Clone)]
struct GesturePattern {
    name: String,
    activation_levels: Vec<f32>,
    duration_ms: u32,
}

/// Noise generator for realistic EMG simulation
struct NoiseGenerator {
    noise_level: f32,
    powerline_phase: f32,
}

#[derive(Debug)]
pub enum SimulatorError {
    ConfigurationError(String),
    GenerationError(String),
}

impl fmt::Display for SimulatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimulatorError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            SimulatorError::GenerationError(msg) => write!(f, "Generation error: {}", msg),
        }
    }
}

impl Error for SimulatorError {}

impl SimulatorDevice {
    /// Create new simulator with configuration
    pub fn new(config: SimulatorConfig) -> Result<Self, SimulatorError> {
        let sample_generator = SampleGenerator::new(&config)?;

        Ok(Self {
            config,
            sample_generator,
            is_running: AtomicBool::new(false),
            sequence_counter: AtomicU32::new(0),
        })
    }

    /// Create simulator with default healthy user profile
    pub fn healthy_user() -> Result<Self, SimulatorError> {
        Self::new(SimulatorConfig::default())
    }

    fn get_current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }
}

impl EmgDevice for SimulatorDevice {
    type Error = SimulatorError;

    async fn initialize(&mut self) -> Result<(), Self::Error> {
        // Simulator always initializes successfully
        Ok(())
    }

    async fn start_acquisition(&mut self) -> Result<(), Self::Error> {
        self.is_running.store(true, Ordering::Relaxed);
        Ok(())
    }

    async fn stop_acquisition(&mut self) -> Result<(), Self::Error> {
        self.is_running.store(false, Ordering::Relaxed);
        Ok(())
    }

    async fn read_sample(&mut self) -> Result<EmgSample, Self::Error> {
        if !self.is_running.load(Ordering::Relaxed) {
            return Err(SimulatorError::GenerationError("Device not running".to_string()));
        }

        let sequence = self.sequence_counter.fetch_add(1, Ordering::Relaxed);
        let timestamp = Self::get_current_timestamp();

        let channels = self.sample_generator.generate_channels()?;
        let quality_indicators = self.sample_generator.generate_quality_metrics(&channels);

        Ok(EmgSample {
            timestamp,
            sequence,
            channels,
            quality_indicators,
        })
    }

    fn get_device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: "EMG Simulator".to_string(),
            version: "1.0.0".to_string(),
            serial_number: "SIM-001".to_string(),
            capabilities: DeviceCapabilities {
                max_channels: self.config.channel_count,
                max_sample_rate_hz: self.config.sample_rate_hz,
                has_builtin_filters: false,
                supports_impedance_check: true,
                supports_calibration: true,
            },
        }
    }

    fn get_channel_count(&self) -> usize {
        self.config.channel_count
    }

    fn get_sampling_rate(&self) -> u32 {
        self.config.sample_rate_hz
    }
}

impl SampleGenerator {
    fn new(config: &SimulatorConfig) -> Result<Self, SimulatorError> {
        let gesture_patterns = Self::create_default_gestures(config.channel_count);
        let noise_generator = NoiseGenerator::new(config.noise_level);

        Ok(Self {
            gesture_patterns,
            current_gesture: 0,
            gesture_progress: 0.0,
            noise_generator,
        })
    }

    fn create_default_gestures(channel_count: usize) -> Vec<GesturePattern> {
        vec![
            GesturePattern {
                name: "rest".to_string(),
                activation_levels: vec![0.1; channel_count],
                duration_ms: 1000,
            },
            GesturePattern {
                name: "fist".to_string(),
                activation_levels: (0..channel_count)
                    .map(|i| 0.8 - (i as f32 * 0.1))
                    .collect(),
                duration_ms: 500,
            },
            GesturePattern {
                name: "open_hand".to_string(),
                activation_levels: (0..channel_count)
                    .map(|i| 0.6 - (i as f32 * 0.05))
                    .collect(),
                duration_ms: 500,
            },
        ]
    }

    fn generate_channels(&mut self) -> Result<Vec<f32>, SimulatorError> {
        let current_pattern = &self.gesture_patterns[self.current_gesture];
        let mut channels = Vec::with_capacity(current_pattern.activation_levels.len());

        for &base_level in &current_pattern.activation_levels {
            let muscle_activation = base_level * self.get_activation_envelope();
            let noisy_signal = self.noise_generator.add_noise(muscle_activation);
            channels.push(noisy_signal);
        }

        // Progress gesture
        self.gesture_progress += 1.0 / (current_pattern.duration_ms as f32);
        if self.gesture_progress >= 1.0 {
            self.gesture_progress = 0.0;
            self.current_gesture = (self.current_gesture + 1) % self.gesture_patterns.len();
        }

        Ok(channels)
    }

    fn get_activation_envelope(&self) -> f32 {
        // Simple envelope: ramp up, hold, ramp down
        if self.gesture_progress < 0.2 {
            self.gesture_progress / 0.2
        } else if self.gesture_progress < 0.8 {
            1.0
        } else {
            (1.0 - self.gesture_progress) / 0.2
        }
    }

    fn generate_quality_metrics(&self, channels: &[f32]) -> QualityMetrics {
        let snr_db = 25.0 + (channels.iter().map(|&x| x.abs()).sum::<f32>() / channels.len() as f32) * 10.0;
        let contact_impedance = vec![10.0; channels.len()]; // Good contact

        QualityMetrics {
            snr_db,
            contact_impedance_kohm: contact_impedance,
            artifact_detected: false,
            signal_saturation: channels.iter().any(|&x| x.abs() > 0.95),
        }
    }
}

impl NoiseGenerator {
    fn new(noise_level: f32) -> Self {
        Self {
            noise_level,
            powerline_phase: 0.0,
        }
    }

    fn add_noise(&mut self, signal: f32) -> f32 {
        let thermal_noise = self.generate_thermal_noise();
        let powerline_noise = self.generate_powerline_noise();

        signal + thermal_noise + powerline_noise
    }

    fn generate_thermal_noise(&self) -> f32 {
        // Generate noise in range [-noise_level/2, noise_level/2]
        simple_rng::next_f32_range(-self.noise_level * 0.5, self.noise_level * 0.5)
    }

    fn generate_powerline_noise(&mut self) -> f32 {
        // 50Hz powerline interference
        let noise = (self.powerline_phase * 2.0 * std::f32::consts::PI).sin() * self.noise_level * 0.1;
        self.powerline_phase += 50.0 / 2000.0; // 50Hz at 2kHz sample rate
        if self.powerline_phase >= 1.0 {
            self.powerline_phase -= 1.0;
        }
        noise
    }
}

// Simple pseudo-random number generator for simulation
mod simple_rng {
    use std::sync::atomic::{AtomicU32, Ordering};

    static SEED: AtomicU32 = AtomicU32::new(12345);

    pub fn next_u32() -> u32 {
        let current = SEED.load(Ordering::Relaxed);
        let mut x = current;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        SEED.store(x, Ordering::Relaxed);
        x
    }

    pub fn next_f32() -> f32 {
        (next_u32() as f32) / (u32::MAX as f32)
    }

    pub fn next_f32_range(min: f32, max: f32) -> f32 {
        min + (max - min) * next_f32()
    }
}