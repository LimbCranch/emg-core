// src/lib.rs
//! EMG Real-time Core Library
//!
//! High-performance EMG signal processing for prosthetic control

#![doc = include_str!("../README.md")]
//#![cfg_attr(not(feature = "std"), no_std)]

pub mod processing;


pub mod hal;
pub mod config;
pub mod acquisition;
mod error;
mod validation;
pub mod utils;
// Re-export commonly used types

pub use processing::{FilterBank, QualityMonitor, WindowManager};


pub use hal::{EmgDevice, EmgSample, DeviceInfo, DeviceCapabilities, QualityMetrics, DeviceType, ThreadPriority};
pub use acquisition::{LockFreeRingBuffer, MpmcRingBuffer, SampleSynchronizer, BufferManager};

// Re-export device implementations
pub use hal::simulator::{SimulatorDevice, SimulatorConfig};
pub use hal::usb_driver::{UsbEmgDevice, UsbDeviceConfig};
pub use hal::serial_driver::{SerialEmgDevice, SerialConfig};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration constants
pub mod constants {
    /// Default sampling rate in Hz
    pub const DEFAULT_SAMPLING_RATE_HZ: u32 = 2000;

    /// Default channel count
    pub const DEFAULT_CHANNEL_COUNT: usize = 8;

    /// Default buffer size in samples
    pub const DEFAULT_BUFFER_SIZE_SAMPLES: usize = 4096;

    /// Default latency target in milliseconds
    pub const DEFAULT_LATENCY_TARGET_MS: u32 = 20;
}

/// Result type for EMG operations
pub type EmgResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_is_set() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_constants_are_reasonable() {
        assert!(constants::DEFAULT_SAMPLING_RATE_HZ > 0);
        assert!(constants::DEFAULT_CHANNEL_COUNT > 0);
        assert!(constants::DEFAULT_BUFFER_SIZE_SAMPLES > 0);
        assert!(constants::DEFAULT_LATENCY_TARGET_MS > 0);
    }
}