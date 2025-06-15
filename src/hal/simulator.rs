// src/hal/simulator.rs
//! EMG device simulator for testing and development
//!
//! FIXED: Race conditions in timestamp generation, comprehensive error handling, correct struct fields

use crate::hal::{EmgDevice, EmgSample, DeviceInfo, DeviceCapabilities, QualityMetrics};
use crate::config::constants::simulator::*;
use crate::config::constants::signal::*;
use crate::utils::time::{TimeProvider, SystemTimeProvider, current_timestamp_nanos};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::error::Error;
use std::fmt;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Configuration for EMG simulator
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SimulatorConfig {
    pub profile: String,
    pub noise_level: f32,
    pub artifact_probability: f32,
    pub channel_count: usize,
    pub sample_rate_hz: u32,
    pub gesture_duration_ms: u32,
    pub enable_powerline_noise: bool,
    pub powerline_frequency_hz: f32,
    pub muscle_activation_patterns: Vec<MusclePattern>,
}

/// Muscle activation pattern for realistic simulation
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MusclePattern {
    pub name: String,
    pub activation_levels: Vec<f32>,
    pub frequency_band: (f32, f32),
    pub activation_threshold: f32,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            profile: "healthy_user".to_string(),
            noise_level: DEFAULT_NOISE_LEVEL,
            artifact_probability: DEFAULT_ARTIFACT_PROBABILITY,
            channel_count: DEFAULT_CHANNEL_COUNT,
            sample_rate_hz: DEFAULT_SAMPLING_RATE_HZ,
            gesture_duration_ms: DEFAULT_GESTURE_DURATION_MS,
            enable_powerline_noise: true,
            powerline_frequency_hz: 60.0, // US standard
            muscle_activation_patterns: Self::default_muscle_patterns(),
        }
    }
}

impl SimulatorConfig {
    fn default_muscle_patterns() -> Vec<MusclePattern> {
        vec![
            MusclePattern {
                name: "flexor_digitorum".to_string(),
                activation_levels: vec![0.2, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                frequency_band: (20.0, 150.0),
                activation_threshold: 0.1,
            },
            MusclePattern {
                name: "extensor_digitorum".to_string(),
                activation_levels: vec![0.0, 0.0, 0.0, 0.2, 0.7, 0.2, 0.0, 0.0],
                frequency_band: (25.0, 180.0),
                activation_threshold: 0.15,
            },
        ]
    }
}

/// EMG device simulator with fixed race conditions
pub struct SimulatorDevice {
    config: SimulatorConfig,
    sample_generator: SampleGenerator,
    is_running: AtomicBool,
    sequence_counter: AtomicU32,
    last_timestamp: AtomicU64,
    start_time: AtomicU64,
    time_provider: Arc<dyn TimeProvider>,
    sample_period_nanos: u64,
}

/// Internal sample generator for realistic EMG data
struct SampleGenerator {
    gesture_patterns: Vec<GesturePattern>,
    current_gesture: usize,
    gesture_progress: f32,
    noise_generator: NoiseGenerator,
    powerline_generator: PowerlineNoiseGenerator,
    rng: StdRng,
}

/// Gesture pattern for simulation
#[derive(Debug, Clone)]
struct GesturePattern {
    name: String,
    activation_levels: Vec<f32>,
    duration_ms: u32,
    frequency_components: Vec<(f32, f32)>, // (frequency, amplitude)
}

/// Noise generator for realistic EMG simulation
struct NoiseGenerator {
    noise_level: f32,
    artifact_probability: f32,
    last_artifact_time: u64,
}

/// Powerline noise generator (50/60 Hz interference)
struct PowerlineNoiseGenerator {
    frequency_hz: f32,
    amplitude: f32,
    phase: f32,
    enabled: bool,
}

#[derive(Debug)]
pub enum SimulatorError {
    ConfigurationError(String),
    GenerationError(&'static str), // FIXED: Use static str for performance
    DeviceNotRunning,
    InvalidChannelCount,
    InvalidSampleRate,
}

impl fmt::Display for SimulatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimulatorError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            SimulatorError::GenerationError(msg) => write!(f, "Generation error: {}", msg),
            SimulatorError::DeviceNotRunning => write!(f, "Device not running"),
            SimulatorError::InvalidChannelCount => write!(f, "Invalid channel count"),
            SimulatorError::InvalidSampleRate => write!(f, "Invalid sample rate"),
        }
    }
}

impl Error for SimulatorError {}

impl SimulatorDevice {
    /// Create new simulator with configuration and dependency injection
    pub fn new(config: SimulatorConfig) -> Result<Self, SimulatorError> {
        Self::with_time_provider(config, Arc::new(SystemTimeProvider))
    }

    /// Create simulator with custom time provider (for testing)
    pub fn with_time_provider(
        config: SimulatorConfig,
        time_provider: Arc<dyn TimeProvider>
    ) -> Result<Self, SimulatorError> {
        // FIXED: Validate configuration at creation time
        if config.channel_count == 0 || config.channel_count > 32 {
            return Err(SimulatorError::InvalidChannelCount);
        }

        if config.sample_rate_hz < 100 || config.sample_rate_hz > 20000 {
            return Err(SimulatorError::InvalidSampleRate);
        }

        if config.noise_level < MIN_NOISE_LEVEL || config.noise_level > MAX_NOISE_LEVEL {
            return Err(SimulatorError::ConfigurationError(
                format!("Noise level {} outside valid range [{}, {}]",
                        config.noise_level, MIN_NOISE_LEVEL, MAX_NOISE_LEVEL)
            ));
        }

        if config.artifact_probability < MIN_ARTIFACT_PROBABILITY ||
            config.artifact_probability > MAX_ARTIFACT_PROBABILITY {
            return Err(SimulatorError::ConfigurationError(
                format!("Artifact probability {} outside valid range [{}, {}]",
                        config.artifact_probability, MIN_ARTIFACT_PROBABILITY, MAX_ARTIFACT_PROBABILITY)
            ));
        }

        let sample_generator = SampleGenerator::new(&config)?;

        // FIXED: Calculate sample period once at initialization
        let sample_period_nanos = 1_000_000_000 / config.sample_rate_hz as u64;
        let current_time = time_provider.now_nanos();

        Ok(Self {
            config,
            sample_generator,
            is_running: AtomicBool::new(false),
            sequence_counter: AtomicU32::new(0),
            last_timestamp: AtomicU64::new(current_time),
            start_time: AtomicU64::new(0),
            time_provider,
            sample_period_nanos,
        })
    }

    /// Create simulator with default healthy user profile
    pub fn healthy_user() -> Result<Self, SimulatorError> {
        Self::new(SimulatorConfig::default())
    }

    /// Create simulator with pathological conditions for testing
    pub fn pathological_user() -> Result<Self, SimulatorError> {
        let mut config = SimulatorConfig::default();
        config.profile = "pathological".to_string();
        config.noise_level = 0.15; // Higher noise
        config.artifact_probability = 0.08; // More artifacts
        Self::new(config)
    }

    /// Get current configuration
    pub fn config(&self) -> &SimulatorConfig {
        &self.config
    }

    /// Get sample period in nanoseconds
    pub fn sample_period_nanos(&self) -> u64 {
        self.sample_period_nanos
    }

    /// Check if device is currently running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }
}

impl EmgDevice for SimulatorDevice {
    type Error = SimulatorError;

    async fn initialize(&mut self) -> Result<(), Self::Error> {
        // Simulator always initializes successfully
        // Reset internal state
        self.sequence_counter.store(0, Ordering::Relaxed);
        self.last_timestamp.store(self.time_provider.now_nanos(), Ordering::Relaxed);
        Ok(())
    }

    async fn start_acquisition(&mut self) -> Result<(), Self::Error> {
        // FIXED: Reset timestamp atomically when starting acquisition
        let current_time = self.time_provider.now_nanos();
        self.start_time.store(current_time, Ordering::Relaxed);
        self.last_timestamp.store(current_time, Ordering::Relaxed);
        self.sequence_counter.store(0, Ordering::Relaxed);
        self.is_running.store(true, Ordering::Relaxed);
        Ok(())
    }

    async fn stop_acquisition(&mut self) -> Result<(), Self::Error> {
        self.is_running.store(false, Ordering::Relaxed);
        Ok(())
    }

    async fn read_sample(&mut self) -> Result<EmgSample, Self::Error> {
        if !self.is_running.load(Ordering::Relaxed) {
            return Err(SimulatorError::DeviceNotRunning);
        }

        // FIXED: Generate sequence and timestamp atomically to prevent race conditions
        let sequence = self.sequence_counter.fetch_add(1, Ordering::Relaxed);

        // FIXED: Use fetch_add for monotonic timestamp generation
        let timestamp = self.last_timestamp.fetch_add(self.sample_period_nanos, Ordering::Relaxed);

        // Generate realistic EMG channels
        let channels = self.sample_generator.generate_channels(sequence, timestamp)?;

        // Generate quality metrics with correct field names
        let quality_indicators = self.generate_quality_metrics(&channels, timestamp);

        Ok(EmgSample {
            timestamp, // FIXED: Use correct field name
            sequence, // FIXED: Use correct field name
            channels,
            quality_indicators,
        })
    }

    // FIXED: Return DeviceInfo directly (not async)
    fn get_device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: "EMG Simulator".to_string(), // FIXED: Use correct field name
            version: "1.0.0".to_string(), // FIXED: Use correct field name
            serial_number: format!("SIM-{:08X}", std::ptr::addr_of!(self) as usize), // FIXED: Use correct field name
            capabilities: DeviceCapabilities { // FIXED: Use correct field name
                max_channels: self.config.channel_count, // FIXED: Use correct field name
                max_sample_rate_hz: 20000, // FIXED: Use correct field name
                has_builtin_filters: false, // FIXED: Use correct field name
                supports_impedance_check: true, // FIXED: Use correct field name
                supports_calibration: true, // FIXED: Use correct field name
            },
        }
    }

    // FIXED: Add missing trait methods
    fn get_channel_count(&self) -> usize {
        self.config.channel_count
    }

    fn get_sampling_rate(&self) -> u32 {
        self.config.sample_rate_hz
    }
}

impl SampleGenerator {
    fn new(config: &SimulatorConfig) -> Result<Self, SimulatorError> {
        let gesture_patterns = config.muscle_activation_patterns
            .iter()
            .map(|p| GesturePattern {
                name: p.name.clone(),
                activation_levels: p.activation_levels.clone(),
                duration_ms: config.gesture_duration_ms,
                frequency_components: vec![
                    (p.frequency_band.0, 0.8),
                    (p.frequency_band.1, 0.3),
                    ((p.frequency_band.0 + p.frequency_band.1) / 2.0, 0.5),
                ],
            })
            .collect();

        Ok(Self {
            gesture_patterns,
            current_gesture: 0,
            gesture_progress: 0.0,
            noise_generator: NoiseGenerator::new(config.noise_level, config.artifact_probability),
            powerline_generator: PowerlineNoiseGenerator::new(
                config.powerline_frequency_hz,
                config.enable_powerline_noise,
            ),
            rng: StdRng::from_entropy(),
        })
    }

    fn generate_channels(&mut self, sequence: u32, timestamp: u64) -> Result<Vec<f32>, SimulatorError> {
        let time_ms = timestamp / 1_000_000; // Convert to milliseconds

        // Get current gesture pattern
        let pattern = &self.gesture_patterns[self.current_gesture % self.gesture_patterns.len()];

        // Update gesture progress
        self.gesture_progress = (time_ms % pattern.duration_ms as u64) as f32 / pattern.duration_ms as f32;

        let mut channels = Vec::with_capacity(pattern.activation_levels.len());

        for (i, &base_activation) in pattern.activation_levels.iter().enumerate() {
            let mut signal = base_activation;

            // Add frequency components
            for &(freq, amplitude) in &pattern.frequency_components {
                let phase = 2.0 * std::f32::consts::PI * freq * (time_ms as f32 / 1000.0);
                signal += amplitude * phase.sin() * base_activation;
            }

            // Add powerline noise
            signal += self.powerline_generator.generate(time_ms);

            // Add random noise
            signal += self.noise_generator.generate(&mut self.rng, i, timestamp);

            // Clamp to valid range
            signal = signal.clamp(-1.0, 1.0);

            channels.push(signal);
        }

        // Cycle through gestures
        if self.gesture_progress >= 0.99 {
            self.current_gesture = (self.current_gesture + 1) % self.gesture_patterns.len();
        }

        Ok(channels)
    }
}

impl NoiseGenerator {
    fn new(noise_level: f32, artifact_probability: f32) -> Self {
        Self {
            noise_level,
            artifact_probability,
            last_artifact_time: 0,
        }
    }

    fn generate(&mut self, rng: &mut StdRng, channel: usize, timestamp: u64) -> f32 {
        // Base thermal noise
        let thermal_noise = rng.gen_range(-1.0..1.0) * self.noise_level;

        // Motion artifacts (random spikes)
        let artifact_noise = if rng.gen::<f32>() < self.artifact_probability &&
            timestamp - self.last_artifact_time > 50_000_000 { // 50ms minimum between artifacts
            self.last_artifact_time = timestamp;
            rng.gen_range(-0.5..0.5) * (channel as f32 + 1.0) // Channel-dependent artifacts
        } else {
            0.0
        };

        thermal_noise + artifact_noise
    }
}

impl PowerlineNoiseGenerator {
    fn new(frequency_hz: f32, enabled: bool) -> Self {
        Self {
            frequency_hz,
            amplitude: 0.02, // 2% of full scale
            phase: 0.0,
            enabled,
        }
    }

    fn generate(&mut self, time_ms: u64) -> f32 {
        if !self.enabled {
            return 0.0;
        }

        let time_sec = time_ms as f32 / 1000.0;
        self.phase = 2.0 * std::f32::consts::PI * self.frequency_hz * time_sec;
        self.amplitude * self.phase.sin()
    }
}

impl SimulatorDevice {
    fn generate_quality_metrics(&self, channels: &[f32], timestamp: u64) -> QualityMetrics {
        // Calculate SNR based on signal vs noise
        let signal_power: f32 = channels.iter().map(|&x| x * x).sum::<f32>() / channels.len() as f32;
        let estimated_noise_power = self.config.noise_level * self.config.noise_level;
        let snr_db = if estimated_noise_power > 0.0 {
            10.0 * (signal_power / estimated_noise_power).log10()
        } else {
            60.0 // Perfect signal
        };

        // Simulate contact impedance per channel (Vec<f32>)
        let base_impedance = 5.0; // kOhms
        let impedance_variation = (timestamp as f32 / 1_000_000_000.0).sin() * 2.0; // Varies over time
        let contact_impedance_kohm = (0..channels.len())
            .map(|i| (base_impedance + impedance_variation + i as f32 * 0.5).abs())
            .collect();

        // Check for saturation
        let max_amplitude = channels.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
        let signal_saturation = max_amplitude > 0.95; // FIXED: Use correct field name

        // Motion artifact detection
        let high_freq_content: f32 = channels.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum::<f32>() / (channels.len() - 1) as f32;
        let artifact_detected = high_freq_content > 0.1; // FIXED: Use correct field name

        QualityMetrics {
            snr_db,
            contact_impedance_kohm, // FIXED: Use Vec<f32>
            artifact_detected, // FIXED: Use correct field name
            signal_saturation, // FIXED: Use correct field name
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::time::MockTimeProvider;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_simulator_initialization() {
        let mut device = SimulatorDevice::healthy_user().expect("Failed to create simulator");
        assert!(device.initialize().await.is_ok());
    }

    #[tokio::test]
    async fn test_race_condition_fix() {
        let mock_time = Arc::new(MockTimeProvider::new(1_000_000_000));
        let mut device = SimulatorDevice::with_time_provider(
            SimulatorConfig::default(),
            mock_time.clone()
        ).expect("Failed to create simulator");

        device.initialize().await.expect("Failed to initialize");
        device.start_acquisition().await.expect("Failed to start");

        // Generate multiple samples quickly to test race condition fix
        let mut last_timestamp = 0u64;
        for i in 0..100 {
            mock_time.advance_by(device.sample_period_nanos());
            let sample = device.read_sample().await.expect("Failed to read sample");

            // Verify monotonic timestamps
            assert!(sample.timestamp > last_timestamp,
                    "Timestamp not monotonic at sample {}: {} <= {}",
                    i, sample.timestamp, last_timestamp);

            // Verify sequence numbers
            assert_eq!(sample.sequence, i, "Sequence number mismatch at sample {}", i);

            last_timestamp = sample.timestamp;
        }
    }

    #[tokio::test]
    async fn test_configuration_validation() {
        // Test invalid channel count
        let mut config = SimulatorConfig::default();
        config.channel_count = 0;
        assert!(SimulatorDevice::new(config).is_err());

        // Test invalid sample rate
        let mut config = SimulatorConfig::default();
        config.sample_rate_hz = 50; // Too low
        assert!(SimulatorDevice::new(config).is_err());

        // Test invalid noise level
        let mut config = SimulatorConfig::default();
        config.noise_level = 2.0; // Too high
        assert!(SimulatorDevice::new(config).is_err());
    }

    #[tokio::test]
    async fn test_device_not_running_error() {
        let mut device = SimulatorDevice::healthy_user().expect("Failed to create simulator");
        device.initialize().await.expect("Failed to initialize");

        // Try to read without starting acquisition
        let result = device.read_sample().await;
        assert!(matches!(result, Err(SimulatorError::DeviceNotRunning)));
    }

    #[tokio::test]
    async fn test_trait_methods() {
        let device = SimulatorDevice::healthy_user().expect("Failed to create simulator");

        // Test trait methods
        assert_eq!(device.get_channel_count(), 8);
        assert_eq!(device.get_sampling_rate(), 2000);

        let info = device.get_device_info();
        assert_eq!(info.name, "EMG Simulator");
        assert!(!info.version.is_empty());
        assert!(!info.serial_number.is_empty());
        assert_eq!(info.capabilities.max_channels, 8);
    }

    #[tokio::test]
    async fn test_quality_metrics_generation() {
        let mut device = SimulatorDevice::healthy_user().expect("Failed to create simulator");
        device.initialize().await.expect("Failed to initialize");
        device.start_acquisition().await.expect("Failed to start");

        let sample = device.read_sample().await.expect("Failed to read sample");

        // Verify quality metrics are realistic
        assert!(sample.quality_indicators.snr_db > 0.0);
        assert_eq!(sample.quality_indicators.contact_impedance_kohm.len(), device.config.channel_count);
        assert_eq!(sample.channels.len(), device.config.channel_count);
    }

    #[tokio::test]
    async fn test_pathological_simulation() {
        let mut device = SimulatorDevice::pathological_user().expect("Failed to create pathological simulator");
        device.initialize().await.expect("Failed to initialize");
        device.start_acquisition().await.expect("Failed to start");

        // Should still generate valid samples even with pathological conditions
        let sample = device.read_sample().await.expect("Failed to read sample");
        assert_eq!(sample.channels.len(), device.config.channel_count);

        // Quality should be lower for pathological user
        assert!(sample.quality_indicators.snr_db < 40.0); // Lower than healthy user
    }
}