//! EMG-Core: High-performance EMG signal acquisition and processing library
//!
//! This library provides a comprehensive framework for EMG (electromyography) signal
//! acquisition, processing, and analysis. It features:
//!
//! - Hardware abstraction layer for multiple EMG devices
//! - Real-time signal processing pipeline
//! - Lock-free data structures for low-latency operations
//! - Comprehensive configuration management
//! - Utility functions for common EMG operations
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use emg_core::hal::{DeviceFactory, EmgDevice};
//! use emg_core::hal::simulator::{SimulatorConfig, EmgSimulator};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a simulator device
//!     let config = SimulatorConfig::default();
//!     let mut device = DeviceFactory::create_simulator(config)?;
//!     
//!     // Initialize and connect
//!     device.initialize().await?;
//!     device.connect().await?;
//!     device.start_acquisition().await?;
//!     
//!     // Read samples
//!     for _ in 0..10 {
//!         let sample = device.read_sample().await?;
//!         println!("Sample: {:?}", sample);
//!     }
//!     
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_inception)]

pub mod config;
pub mod hal;
pub mod utils;

// Re-export commonly used types for convenience
pub use hal::{
    EmgDevice, EmgSample, DeviceInfo, DeviceCapabilities, QualityMetrics, DeviceStatus,
    HalError, DeviceFactory,
};

pub use utils::{
    time::{current_timestamp_nanos, TimeProvider},
    validation::{ValidationResult, ValidationError},
    bounds::BoundsResult,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library name
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Get library information
pub fn version_info() -> VersionInfo {
    VersionInfo {
        name: NAME.to_string(),
        version: VERSION.to_string(),
        description: "High-performance EMG signal acquisition and processing library".to_string(),
        features: vec![
            "Hardware abstraction layer".to_string(),
            "Real-time signal processing".to_string(),
            "Lock-free data structures".to_string(),
            "Comprehensive configuration management".to_string(),
            "Utility functions for EMG operations".to_string(),
        ],
    }
}

/// Library version information
#[derive(Debug, Clone)]
pub struct VersionInfo {
    /// Library name
    pub name: String,
    /// Version string
    pub version: String,
    /// Description
    pub description: String,
    /// List of features
    pub features: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        let info = version_info();
        assert_eq!(info.name, NAME);
        assert_eq!(info.version, VERSION);
        assert!(!info.features.is_empty());
    }

    #[test]
    fn test_constants() {
        assert!(!VERSION.is_empty());
        assert!(!NAME.is_empty());
    }
}