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



#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum GestureType {
    /// Hand closing/grasping motion
    HandClose,
    /// Hand opening/releasing motion  
    HandOpen,
    /// Wrist flexion (bending inward)
    WristFlexion,
    /// Wrist extension (bending outward)
    WristExtension,
    /// Wrist pronation (palm down rotation)
    WristPronation,
    /// Wrist supination (palm up rotation)
    WristSupination,
    /// Index finger pointing
    IndexPoint,
    /// Thumb activation
    ThumbUp,
    /// Rest/no intentional movement
    Rest,
    /// Co-contraction (multiple muscles active)
    CoContraction,
}

impl Default for GestureType {
    fn default() -> Self {
        Self::Rest
    }
}

impl std::fmt::Display for GestureType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::HandClose => "Hand Close",
            Self::HandOpen => "Hand Open",
            Self::WristFlexion => "Wrist Flexion",
            Self::WristExtension => "Wrist Extension",
            Self::WristPronation => "Wrist Pronation",
            Self::WristSupination => "Wrist Supination",
            Self::IndexPoint => "Index Point",
            Self::ThumbUp => "Thumb Up",
            Self::Rest => "Rest",
            Self::CoContraction => "Co-contraction",
        };
        write!(f, "{}", name)
    }
}

/// Extended EMG sample with simulation metadata
#[derive(Debug, Clone)]
pub struct SimulatedEmgSample {
    /// Base EMG sample data
    pub base_sample: EmgSample,
    /// Ground truth gesture being simulated
    pub ground_truth_gesture: GestureType,
    /// Activation level (0.0 to 1.0)
    pub activation_level: f32,
    /// Simulation metadata
    pub simulation_metadata: SimulationMetadata,
}

/// Metadata about the simulation state
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SimulationMetadata {
    /// Profile name used for simulation
    pub profile_name: String,
    /// Whether artifacts were injected in this sample
    pub artifacts_present: bool,
    /// Current fatigue levels per channel (0.0 to 1.0)
    pub fatigue_levels: Vec<f32>,
    /// Signal-to-noise ratio before artifact injection
    pub clean_snr_db: f32,
}