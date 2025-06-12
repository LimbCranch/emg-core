// src/config/device_config.rs
//! Device-specific configuration structures

use serde::{Deserialize, Serialize};

/// Device-specific configuration union
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DeviceSpecificConfig {
    Simulator(crate::hal::simulator::SimulatorConfig),
    Usb(crate::hal::usb_driver::UsbDeviceConfig),
    Serial(crate::hal::serial_driver::SerialConfig),
}

impl Default for DeviceSpecificConfig {
    fn default() -> Self {
        Self::Simulator(crate::hal::simulator::SimulatorConfig::default())
    }
}

/// Device capability validation
pub fn validate_device_config(config: &DeviceSpecificConfig) -> Result<(), String> {
    match config {
        DeviceSpecificConfig::Simulator(sim_config) => {
            if sim_config.channel_count == 0 {
                return Err("Simulator channel count must be greater than 0".to_string());
            }
            if sim_config.sample_rate_hz == 0 {
                return Err("Simulator sample rate must be greater than 0".to_string());
            }
            if sim_config.noise_level < 0.0 || sim_config.noise_level > 1.0 {
                return Err("Simulator noise level must be between 0.0 and 1.0".to_string());
            }
        }
        DeviceSpecificConfig::Usb(usb_config) => {
            if usb_config.timeout_ms == 0 {
                return Err("USB timeout must be greater than 0".to_string());
            }
            if usb_config.buffer_size == 0 {
                return Err("USB buffer size must be greater than 0".to_string());
            }
        }
        DeviceSpecificConfig::Serial(serial_config) => {
            if serial_config.baud_rate == 0 {
                return Err("Serial baud rate must be greater than 0".to_string());
            }
            if serial_config.timeout_ms == 0 {
                return Err("Serial timeout must be greater than 0".to_string());
            }
            if serial_config.port_name.is_empty() {
                return Err("Serial port name cannot be empty".to_string());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulator_config_validation() {
        let mut config = crate::hal::simulator::SimulatorConfig::default();
        assert!(validate_device_config(&DeviceSpecificConfig::Simulator(config.clone())).is_ok());

        config.noise_level = 1.5;
        assert!(validate_device_config(&DeviceSpecificConfig::Simulator(config)).is_err());
    }

    #[test]
    fn test_usb_config_validation() {
        let mut config = crate::hal::usb_driver::UsbDeviceConfig::default();
        assert!(validate_device_config(&DeviceSpecificConfig::Usb(config.clone())).is_ok());

        config.timeout_ms = 0;
        assert!(validate_device_config(&DeviceSpecificConfig::Usb(config)).is_err());
    }

    #[test]
    fn test_serial_config_validation() {
        let mut config = crate::hal::serial_driver::SerialConfig::default();
        assert!(validate_device_config(&DeviceSpecificConfig::Serial(config.clone())).is_ok());

        config.port_name = String::new();
        assert!(validate_device_config(&DeviceSpecificConfig::Serial(config)).is_err());
    }
}