//! Hardware Abstraction Layer for EMG devices
//!
//! This module defines the core traits and types for EMG device communication.
//! All device implementations must implement the EmgDevice trait.

pub mod simulator;
 pub mod usb_driver;
 pub mod serial_driver;
pub mod types;
pub mod traits;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::Ordering;
use crate::hal::serial_driver::SerialConfig;
use crate::utils::PacketValidator;

/// Core trait that all EMG devices must implement
#[async_trait]
pub trait EmgDevice: Send + Sync {
    /// Error type specific to this device
    type Error: std::error::Error + Send + Sync + 'static;

    /// Initialize the device
    async fn initialize(&mut self) -> Result<(), Self::Error>;

    /// Connect to the device
    async fn connect(&mut self) -> Result<(), Self::Error>;

    /// Disconnect from the device
    async fn disconnect(&mut self) -> Result<(), Self::Error>;

    /// Start data acquisition
    async fn start_acquisition(&mut self) -> Result<(), Self::Error>;

    /// Stop data acquisition
    async fn stop_acquisition(&mut self) -> Result<(), Self::Error>;

    /// Read a single sample from all channels
    async fn read_sample(&mut self) -> Result<EmgSample, Self::Error>;

    /// Get the number of channels
    fn get_channel_count(&self) -> usize;

    /// Get the current sampling rate
    fn get_sampling_rate(&self) -> u32;

    /// Get device information
    async fn get_device_info(&self) -> Result<DeviceInfo, Self::Error>;

    /// Get device status
    async fn get_status(&self) -> Result<DeviceStatus, Self::Error>;
    async fn configure(&mut self, config: Self::Config) -> Result<(), Self::Error>;
    type Config;
}

/// EMG sample containing data from all channels at a specific timestamp
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmgSample {
    /// Timestamp in nanoseconds since Unix epoch
    pub timestamp: u64,

    /// Sequence number for ordering samples
    pub sequence: u64,

    /// Channel data in volts (one value per channel)
    pub channel_data: Vec<f32>,

    /// Optional quality metrics for this sample
    pub quality_metrics: Option<QualityMetrics>,
}

/// Quality metrics for signal assessment
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Overall signal quality (0.0 to 1.0)
    pub signal_quality: f32,

    /// Noise level estimate
    pub noise_level: f32,

    /// Whether artifacts were detected in this sample
    pub artifact_detected: bool,

    /// Signal-to-noise ratio in dB
    pub snr_db: f32,
}

/// Device information and capabilities
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Unique device identifier
    pub device_id: String,

    /// Human-readable device type
    pub device_type: String,

    /// Firmware version
    pub firmware_version: String,

    /// Device serial number
    pub serial_number: String,

    /// Device capabilities
    pub capabilities: DeviceCapabilities,
}

/// Device capabilities and specifications
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Maximum supported sampling rate in Hz
    pub max_sampling_rate_hz: u32,

    /// Number of channels
    pub channel_count: usize,

    /// ADC resolution in bits
    pub resolution_bits: u8,

    /// Input range in millivolts
    pub input_range_mv: f32,

    /// Whether the device supports differential inputs
    pub supports_differential: bool,

    /// Whether the device has hardware filters
    pub supports_hardware_filters: bool,
}

/// Current device status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceStatus {
    /// Whether the device is connected
    pub is_connected: bool,

    /// Whether the device is currently streaming data
    pub is_streaming: bool,

    /// Current sampling rate in Hz
    pub sample_rate_hz: u32,

    /// Number of samples processed
    pub samples_processed: u64,

    /// Number of errors encountered
    pub error_count: u64,

    /// Last error message (if any)
    pub last_error: Option<String>,
}

/// Common errors that can occur with EMG devices
#[derive(Debug, Clone, PartialEq)]
pub enum HalError {
    /// Device not found or not available
    DeviceNotFound(String),

    /// Connection failed
    ConnectionFailed(String),

    /// Device is not connected
    NotConnected,

    /// Device is not initialized
    NotInitialized,

    /// Configuration error
    ConfigurationError(String),

    /// Data acquisition error
    AcquisitionError(String),

    /// Timeout occurred
    Timeout(String),

    /// Hardware error
    HardwareError(String),

    /// Protocol error
    ProtocolError(String),

    /// Data corruption detected
    DataCorruption(String),

    /// Buffer overflow
    BufferOverflow,

    /// Invalid parameter
    InvalidParameter(String),

    /// Operation not supported
    NotSupported(String),
}

impl fmt::Display for HalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HalError::DeviceNotFound(device) => write!(f, "Device not found: {}", device),
            HalError::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            HalError::NotConnected => write!(f, "Device is not connected"),
            HalError::NotInitialized => write!(f, "Device is not initialized"),
            HalError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            HalError::AcquisitionError(msg) => write!(f, "Data acquisition error: {}", msg),
            HalError::Timeout(msg) => write!(f, "Timeout: {}", msg),
            HalError::HardwareError(msg) => write!(f, "Hardware error: {}", msg),
            HalError::ProtocolError(msg) => write!(f, "Protocol error: {}", msg),
            HalError::DataCorruption(msg) => write!(f, "Data corruption: {}", msg),
            HalError::BufferOverflow => write!(f, "Buffer overflow"),
            HalError::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            HalError::NotSupported(msg) => write!(f, "Operation not supported: {}", msg),
        }
    }
}

impl std::error::Error for HalError {}

/// Device factory for creating EMG devices
pub struct DeviceFactory;

impl DeviceFactory {
    /// Create a simulator device
    pub fn create_simulator(config: simulator::SimulatorConfig) -> Result<simulator::EmgSimulator, simulator::SimulatorError> {
        simulator::EmgSimulator::new(config)
    }

    /// List available devices (placeholder implementation)
    pub async fn list_devices() -> Result<Vec<DeviceInfo>, HalError> {
        // This would scan for actual USB/Serial devices in a real implementation
        Ok(vec![])
    }

    /// Auto-detect and connect to the first available device
    pub async fn auto_connect() -> Result<Box<dyn EmgDevice<Error = HalError, Config=()>>, HalError> {
        // This would try to connect to available devices in a real implementation
        Err(HalError::DeviceNotFound("No devices found".to_string()))
    }
}

/// Utility functions for EMG data processing
pub mod utils {
    use super::*;

    /// Convert ADC counts to voltage
    pub fn adc_to_voltage(adc_value: i32, resolution_bits: u8, reference_voltage: f32) -> f32 {
        let max_value = (1 << (resolution_bits - 1)) as f32;
        (adc_value as f32 / max_value) * reference_voltage
    }

    /// Convert voltage to ADC counts
    pub fn voltage_to_adc(voltage: f32, resolution_bits: u8, reference_voltage: f32) -> i32 {
        let max_value = (1 << (resolution_bits - 1)) as f32;
        (voltage / reference_voltage * max_value) as i32
    }

    /// Calculate RMS value of a signal
    pub fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }

        let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Detect signal saturation
    pub fn detect_saturation(samples: &[f32], threshold: f32) -> bool {
        samples.iter().any(|&x| x.abs() >= threshold)
    }

    /// Calculate signal-to-noise ratio in dB
    pub fn calculate_snr_db(signal_power: f32, noise_power: f32) -> f32 {
        if noise_power <= 0.0 {
            return f32::INFINITY;
        }
        10.0 * (signal_power / noise_power).log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::constants::signal;

    #[test]
    fn test_emg_sample_creation() {
        let sample = EmgSample {
            timestamp: 1000000000,
            sequence: 1,
            channel_data: vec![0.001, -0.002, 0.0015, -0.0008],
            quality_metrics: Some(QualityMetrics {
                signal_quality: 0.95,
                noise_level: 0.01,
                artifact_detected: false,
                snr_db: 40.0,
            }),
        };

        assert_eq!(sample.channel_data.len(), 4);
        assert!(sample.quality_metrics.is_some());
    }

    #[test]
    fn test_device_info_creation() {
        let info = DeviceInfo {
            device_id: "test_device".to_string(),
            device_type: "Test Device".to_string(),
            firmware_version: "1.0.0".to_string(),
            serial_number: "TEST123".to_string(),
            capabilities: DeviceCapabilities {
                max_sampling_rate_hz: signal::MAX_SAMPLING_RATE_HZ,
                channel_count: 8,
                resolution_bits: 24,
                input_range_mv: 10.0,
                supports_differential: true,
                supports_hardware_filters: false,
            },
        };

        assert_eq!(info.device_id, "test_device");
        assert_eq!(info.capabilities.channel_count, 8);
    }

    #[test]
    fn test_adc_conversion() {
        let voltage = utils::adc_to_voltage(1000, 16, 3.3);
        assert!((voltage - 0.1).abs() < 0.01); // Should be approximately 0.1V

        let adc = utils::voltage_to_adc(0.1, 16, 3.3);
        assert!((adc - 1000).abs() < 10); // Should be approximately 1000
    }

    #[test]
    fn test_rms_calculation() {
        let samples = vec![1.0, -1.0, 1.0, -1.0];
        let rms = utils::calculate_rms(&samples);
        assert!((rms - 1.0).abs() < 0.001);

        let empty_samples: Vec<f32> = vec![];
        let rms_empty = utils::calculate_rms(&empty_samples);
        assert_eq!(rms_empty, 0.0);
    }

    #[test]
    fn test_saturation_detection() {
        let normal_samples = vec![0.1, -0.2, 0.15, -0.05];
        assert!(!utils::detect_saturation(&normal_samples, 0.9));

        let saturated_samples = vec![0.1, -0.2, 0.95, -0.05];
        assert!(utils::detect_saturation(&saturated_samples, 0.9));
    }

    #[test]
    fn test_snr_calculation() {
        let snr = utils::calculate_snr_db(1.0, 0.1);
        assert!((snr - 10.0).abs() < 0.1); // Should be 10 dB

        let infinite_snr = utils::calculate_snr_db(1.0, 0.0);
        assert!(infinite_snr.is_infinite());
    }

    #[test]
    fn test_hal_error_display() {
        let error = HalError::DeviceNotFound("USB123".to_string());
        let error_string = format!("{}", error);
        assert!(error_string.contains("Device not found"));
        assert!(error_string.contains("USB123"));
    }
}