// src/hal/types.rs
//! Core types for EMG device abstraction

use serde::{Deserialize, Serialize};

/// Single EMG sample with metadata
#[derive(Debug, Clone, PartialEq)]
pub struct EmgSample {
    pub timestamp: u64,
    pub sequence: u32,
    pub channels: Vec<f32>,
    pub quality_indicators: QualityMetrics,
}

/// Device information and capabilities
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub name: String,
    pub version: String,
    pub serial_number: String,
    pub capabilities: DeviceCapabilities,
}

/// Signal quality metrics for each sample
#[derive(Debug, Clone, PartialEq)]
pub struct QualityMetrics {
    pub snr_db: f32,
    pub contact_impedance_kohm: Vec<f32>,
    pub artifact_detected: bool,
    pub signal_saturation: bool,
}

/// Device capability flags
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    pub max_channels: usize,
    pub max_sample_rate_hz: u32,
    pub has_builtin_filters: bool,
    pub supports_impedance_check: bool,
    pub supports_calibration: bool,
}

/// Device connection types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeviceType {
    Simulator,
    Usb,
    Serial,
    Bluetooth,
}

/// Thread priority levels from config
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ThreadPriority {
    Normal,
    High,
    #[serde(rename = "realtime")]
    RealTime,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            snr_db: 0.0,
            contact_impedance_kohm: Vec::new(),
            artifact_detected: false,
            signal_saturation: false,
        }
    }
}

impl Default for DeviceCapabilities {
    fn default() -> Self {
        Self {
            max_channels: 8,
            max_sample_rate_hz: 4000,
            has_builtin_filters: false,
            supports_impedance_check: false,
            supports_calibration: false,
        }
    }
}