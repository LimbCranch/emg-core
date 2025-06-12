// src/config/mod.rs
//! Configuration management for EMG system

pub mod device_config;
pub mod processing_config;

pub use device_config::*;
pub use processing_config::*;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use crate::hal::{DeviceType, ThreadPriority};

/// Complete system configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SystemConfig {
    pub system: SystemSettings,
    pub hal: HalConfig,
    pub processing: ProcessingConfig,
    pub communication: CommunicationConfig,
}

/// Core system settings
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SystemSettings {
    pub sampling_rate_hz: u32,
    pub channel_count: usize,
    pub buffer_size_samples: usize,
    pub latency_target_ms: u32,
    pub thread_priority: ThreadPriority,
}

/// Hardware abstraction layer configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct HalConfig {
    pub device_type: DeviceType,
    pub connection_timeout_ms: u32,
    pub retry_attempts: u32,
    pub auto_reconnect: bool,
    pub simulator: Option<crate::hal::simulator::SimulatorConfig>,
    pub usb: Option<crate::hal::usb_driver::UsbDeviceConfig>,
    pub serial: Option<crate::hal::serial_driver::SerialConfig>,
}

/// Communication layer configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CommunicationConfig {
    pub shared_memory_size_mb: usize,
    pub message_queue_size: usize,
}

impl Default for SystemSettings {
    fn default() -> Self {
        Self {
            sampling_rate_hz: crate::constants::DEFAULT_SAMPLING_RATE_HZ,
            channel_count: crate::constants::DEFAULT_CHANNEL_COUNT,
            buffer_size_samples: crate::constants::DEFAULT_BUFFER_SIZE_SAMPLES,
            latency_target_ms: crate::constants::DEFAULT_LATENCY_TARGET_MS,
            thread_priority: ThreadPriority::RealTime,
        }
    }
}

impl Default for HalConfig {
    fn default() -> Self {
        Self {
            device_type: DeviceType::Simulator,
            connection_timeout_ms: 5000,
            retry_attempts: 3,
            auto_reconnect: true,
            simulator: Some(crate::hal::simulator::SimulatorConfig::default()),
            usb: None,
            serial: None,
        }
    }
}

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            shared_memory_size_mb: 64,
            message_queue_size: 1024,
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

/// Configuration loader with hierarchical loading
pub struct ConfigLoader {
    config_paths: Vec<PathBuf>,
}

impl ConfigLoader {
    /// Create new config loader with default paths
    pub fn new() -> Self {
        Self {
            config_paths: Self::discover_config_paths(),
        }
    }

    /// Create config loader with custom paths
    pub fn with_paths(paths: Vec<PathBuf>) -> Self {
        Self {
            config_paths: paths,
        }
    }

    /// Load complete system configuration
    pub fn load_system_config(&self) -> Result<SystemConfig, config::ConfigError> {
        let mut config_builder = config::Config::builder();

        // Load base configuration
        config_builder = config_builder.add_source(
            config::Config::try_from(&SystemConfig::default())?
        );

        // Load configuration files in order
        for path in &self.config_paths {
            if path.exists() {
                config_builder = config_builder.add_source(
                    config::File::from(path.clone()).required(false)
                );
            }
        }

        // Environment variables override everything
        config_builder = config_builder.add_source(
            config::Environment::with_prefix("EMG")
                .separator("_")
                .try_parsing(true)
        );

        let config = config_builder.build()?;
        config.try_deserialize()
    }

    fn discover_config_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // System-wide configuration
        paths.push(PathBuf::from("/etc/emg/config.toml"));

        // User configuration
        if let Some(config_dir) = dirs::config_dir() {
            paths.push(config_dir.join("emg").join("config.toml"));
        }

        // Local configuration
        paths.push(PathBuf::from("config.toml"));
        paths.push(PathBuf::from("config/default.toml"));
        paths.push(PathBuf::from("config/local.toml"));

        paths
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

// Add dirs crate functionality locally for cross-platform config directories
mod dirs {
    use std::path::PathBuf;

    pub fn config_dir() -> Option<PathBuf> {
        #[cfg(target_os = "linux")]
        {
            std::env::var_os("XDG_CONFIG_HOME")
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config"))
                })
        }
        #[cfg(target_os = "macos")]
        {
            std::env::var_os("HOME")
                .map(|home| PathBuf::from(home).join("Library/Application Support"))
        }
        #[cfg(target_os = "windows")]
        {
            std::env::var_os("APPDATA").map(PathBuf::from)
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = SystemConfig::default();
        assert!(config.system.sampling_rate_hz > 0);
        assert!(config.system.channel_count > 0);
        assert!(config.hal.connection_timeout_ms > 0);
    }

    #[test]
    fn test_config_serialization() {
        let config = SystemConfig::default();
        let toml_str = toml::to_string(&config).unwrap();
        let deserialized: SystemConfig = toml::from_str(&toml_str).unwrap();

        assert_eq!(config.system.sampling_rate_hz, deserialized.system.sampling_rate_hz);
        assert_eq!(config.system.channel_count, deserialized.system.channel_count);
    }

    #[test]
    fn test_config_loader_creation() {
        let loader = ConfigLoader::new();
        assert!(!loader.config_paths.is_empty());
    }
}