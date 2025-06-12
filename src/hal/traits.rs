// src/hal/traits.rs
//! Core HAL traits for EMG device abstraction

use crate::hal::types::{EmgSample, DeviceInfo};
use std::error::Error;

/// Main trait for EMG device implementations
pub trait EmgDevice: Send + Sync {
    type Error: Error + Send + Sync + 'static;

    /// Initialize the device
    async fn initialize(&mut self) -> Result<(), Self::Error>;

    /// Start signal acquisition
    async fn start_acquisition(&mut self) -> Result<(), Self::Error>;

    /// Stop signal acquisition
    async fn stop_acquisition(&mut self) -> Result<(), Self::Error>;

    /// Read next EMG sample
    async fn read_sample(&mut self) -> Result<EmgSample, Self::Error>;

    /// Get device information
    fn get_device_info(&self) -> DeviceInfo;

    /// Get number of channels
    fn get_channel_count(&self) -> usize;

    /// Get sampling rate in Hz
    fn get_sampling_rate(&self) -> u32;
}