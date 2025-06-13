// src/config/mod.rs
//! Enhanced configuration management system

pub mod constants;
pub mod schema_validator;
pub mod loader;
pub mod device_config;
pub mod processing_config;
pub(crate) mod processing;

pub use constants::*;
pub use schema_validator::{SchemaValidator, ValidationError};
pub use loader::{ConfigLoader, ConfigError};
pub use device_config::*;
pub use processing_config::*;

use serde::{Deserialize, Serialize};
use crate::hal::{DeviceType, ThreadPriority};

/// Complete system configuration with enhanced validation
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SystemConfig {
    pub system: SystemSettings,
    pub hal: HalConfig,
    pub processing: ProcessingConfig,
    pub communication: CommunicationConfig,
}

/// Enhanced system settings with validation
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SystemSettings {
    #[serde(default = "defaults::sampling_rate_hz")]
    pub sampling_rate_hz: u32,

    #[serde(default = "defaults::channel_count")]
    pub channel_count: usize,

    #[serde(default = "defaults::buffer_size_samples")]
    pub buffer_size_samples: usize,

    #[serde(default = "defaults::latency_target_ms")]
    pub latency_target_ms: u32,

    #[serde(default = "defaults::thread_priority")]
    pub thread_priority: ThreadPriority,

    #[serde(default = "defaults::enable_safety_monitoring")]
    pub enable_safety_monitoring: bool,

    #[serde(default = "defaults::watchdog_timeout_ms")]
    pub watchdog_timeout_ms: u32,
}

/// Enhanced HAL configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct HalConfig {
    #[serde(default = "defaults::device_type")]
    pub device_type: DeviceType,

    #[serde(default = "defaults::connection_timeout_ms")]
    pub connection_timeout_ms: u32,

    #[serde(default = "defaults::retry_attempts")]
    pub retry_attempts: u32,

    #[serde(default = "defaults::auto_reconnect")]
    pub auto_reconnect: bool,

    #[serde(default = "defaults::device_scan_timeout_ms")]
    pub device_scan_timeout_ms: u32,

    pub simulator: Option<crate::hal::simulator::SimulatorConfig>,
    pub usb: Option<crate::hal::usb_driver::UsbDeviceConfig>,
    pub serial: Option<crate::hal::serial_driver::SerialConfig>,
}

/// Communication layer configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CommunicationConfig {
    #[serde(default = "defaults::shared_memory_size_mb")]
    pub shared_memory_size_mb: usize,

    #[serde(default = "defaults::message_queue_size")]
    pub message_queue_size: usize,

    #[serde(default = "defaults::max_message_size_bytes")]
    pub max_message_size_bytes: usize,

    #[serde(default = "defaults::heartbeat_interval_ms")]
    pub heartbeat_interval_ms: u32,

    #[serde(default = "defaults::enable_compression")]
    pub enable_compression: bool,
}

/// Default value providers using constants
mod defaults {
    use crate::hal::{DeviceType, ThreadPriority};
    use crate::config::constants::*;

    pub fn sampling_rate_hz() -> u32 { signal::DEFAULT_SAMPLING_RATE_HZ }
    pub fn channel_count() -> usize { signal::DEFAULT_CHANNEL_COUNT }
    pub fn buffer_size_samples() -> usize { signal::DEFAULT_BUFFER_SIZE_SAMPLES }
    pub fn latency_target_ms() -> u32 { performance::DEFAULT_LATENCY_TARGET_MS }
    pub fn thread_priority() -> ThreadPriority { ThreadPriority::RealTime }
    pub fn enable_safety_monitoring() -> bool { true }
    pub fn watchdog_timeout_ms() -> u32 { performance::WATCHDOG_TIMEOUT_MS }

    pub fn device_type() -> DeviceType { DeviceType::Simulator }
    pub fn connection_timeout_ms() -> u32 { hal::DEFAULT_CONNECTION_TIMEOUT_MS }
    pub fn retry_attempts() -> u32 { hal::DEFAULT_RETRY_ATTEMPTS }
    pub fn auto_reconnect() -> bool { true }
    pub fn device_scan_timeout_ms() -> u32 { hal::DEVICE_SCAN_TIMEOUT_MS }

    pub fn shared_memory_size_mb() -> usize { communication::DEFAULT_SHARED_MEMORY_SIZE_MB }
    pub fn message_queue_size() -> usize { communication::DEFAULT_MESSAGE_QUEUE_SIZE }
    pub fn max_message_size_bytes() -> usize { communication::MAX_MESSAGE_SIZE_BYTES }
    pub fn heartbeat_interval_ms() -> u32 { communication::HEARTBEAT_INTERVAL_MS }
    pub fn enable_compression() -> bool { false }
}

impl Default for SystemSettings {
    fn default() -> Self {
        Self {
            sampling_rate_hz: defaults::sampling_rate_hz(),
            channel_count: defaults::channel_count(),
            buffer_size_samples: defaults::buffer_size_samples(),
            latency_target_ms: defaults::latency_target_ms(),
            thread_priority: defaults::thread_priority(),
            enable_safety_monitoring: defaults::enable_safety_monitoring(),
            watchdog_timeout_ms: defaults::watchdog_timeout_ms(),
        }
    }
}

impl Default for HalConfig {
    fn default() -> Self {
        Self {
            device_type: defaults::device_type(),
            connection_timeout_ms: defaults::connection_timeout_ms(),
            retry_attempts: defaults::retry_attempts(),
            auto_reconnect: defaults::auto_reconnect(),
            device_scan_timeout_ms: defaults::device_scan_timeout_ms(),
            simulator: Some(crate::hal::simulator::SimulatorConfig::default()),
            usb: None,
            serial: None,
        }
    }
}

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            shared_memory_size_mb: defaults::shared_memory_size_mb(),
            message_queue_size: defaults::message_queue_size(),
            max_message_size_bytes: defaults::max_message_size_bytes(),
            heartbeat_interval_ms: defaults::heartbeat_interval_ms(),
            enable_compression: defaults::enable_compression(),
        }
    }
}

impl Default for SystemConfig {
    fn default() -> Self {
        Self {
            system: SystemSettings::default(),
            hal: HalConfig::default(),
            processing: ProcessingConfig::default(),
            communication: CommunicationConfig::default(),
        }
    }
}

/// Configuration utility functions
impl SystemConfig {
    /// Validate configuration consistency
    pub fn validate_consistency(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Check sampling rate vs latency target
        let max_samples_per_window = (self.system.sampling_rate_hz * self.system.latency_target_ms) / 1000;
        if max_samples_per_window > self.system.buffer_size_samples as u32 {
            errors.push(format!(
                "Buffer size too small for latency target: need {} samples, have {}",
                max_samples_per_window, self.system.buffer_size_samples
            ));
        }

        // Check filter frequencies vs sampling rate
        let nyquist = self.system.sampling_rate_hz as f32 / 2.0;
        if self.processing.filter_bank.lowpass_cutoff_hz >= nyquist {
            errors.push(format!(
                "Lowpass cutoff ({} Hz) must be less than Nyquist frequency ({} Hz)",
                self.processing.filter_bank.lowpass_cutoff_hz, nyquist
            ));
        }

        // Check notch filter frequencies
        for freq in &self.processing.filter_bank.notch_filters.frequencies_hz {
            if *freq >= nyquist {
                errors.push(format!(
                    "Notch filter frequency ({} Hz) must be less than Nyquist frequency ({} Hz)",
                    freq, nyquist
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get effective buffer size based on configuration
    pub fn get_effective_buffer_size(&self) -> usize {
        let min_size = (self.system.sampling_rate_hz * self.system.latency_target_ms / 1000) as usize * 2;
        self.system.buffer_size_samples.max(min_size).next_power_of_two()
    }

    /// Check if configuration is suitable for real-time operation
    pub fn is_realtime_capable(&self) -> bool {
        self.system.latency_target_ms <= performance::MAX_LATENCY_TARGET_MS &&
            self.system.thread_priority == ThreadPriority::RealTime &&
            self.system.enable_safety_monitoring
    }

    /// Get configuration summary
    pub fn get_summary(&self) -> ConfigSummary {
        ConfigSummary {
            sampling_rate_hz: self.system.sampling_rate_hz,
            channel_count: self.system.channel_count,
            latency_target_ms: self.system.latency_target_ms,
            device_type: self.hal.device_type.clone(),
            is_realtime: self.is_realtime_capable(),
            buffer_size: self.get_effective_buffer_size(),
        }
    }
}

/// Configuration summary for display/logging
#[derive(Debug, Clone, Serialize)]
pub struct ConfigSummary {
    pub sampling_rate_hz: u32,
    pub channel_count: usize,
    pub latency_target_ms: u32,
    pub device_type: DeviceType,
    pub is_realtime: bool,
    pub buffer_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_creation() {
        let config = SystemConfig::default();
        assert_eq!(config.system.sampling_rate_hz, signal::DEFAULT_SAMPLING_RATE_HZ);
        assert_eq!(config.system.channel_count, signal::DEFAULT_CHANNEL_COUNT);
        assert!(config.validate_consistency().is_ok());
    }

    #[test]
    fn test_config_serialization() {
        let config = SystemConfig::default();
        let toml_str = toml::to_string(&config).unwrap();
        let deserialized: SystemConfig = toml::from_str(&toml_str).unwrap();

        assert_eq!(config.system.sampling_rate_hz, deserialized.system.sampling_rate_hz);
    }

    #[test]
    fn test_config_validation() {
        let mut config = SystemConfig::default();

        // Make buffer too small
        config.system.buffer_size_samples = 100;
        config.system.latency_target_ms = 100;

        assert!(config.validate_consistency().is_err());
    }

    #[test]
    fn test_effective_buffer_size() {
        let config = SystemConfig::default();
        let buffer_size = config.get_effective_buffer_size();

        assert!(buffer_size.is_power_of_two());
        assert!(buffer_size >= config.system.buffer_size_samples);
    }

    #[test]
    fn test_realtime_capability() {
        let mut config = SystemConfig::default();
        assert!(config.is_realtime_capable());

        config.system.thread_priority = ThreadPriority::Normal;
        assert!(!config.is_realtime_capable());
    }
}