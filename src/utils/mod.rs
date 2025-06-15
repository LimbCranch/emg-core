//! Common utility functions for the EMG-Core system
//!
//! This module provides reusable utilities across the EMG system:
//! - Time and timestamp management with monotonic clock support
//! - Data validation and bounds checking utilities  
//! - Configuration validation helpers
//! - Data integrity verification functions
//! - Common mathematical and conversion utilities
//!
//! All utilities follow the no-magic-numbers principle with constants
//! defined in the config module.

pub mod time;
pub mod validation;
pub mod bounds;
pub mod conversion;
pub mod integrity;

// Re-export commonly used functions for convenience
pub use time::{
    current_timestamp_nanos,
    current_timestamp_micros,
    monotonic_timestamp_nanos,
    validate_timestamp,
    TimestampValidator,
    TimeProvider,
    MonotonicTimeProvider,
    SystemTimeProvider,
};

pub use validation::{
    Validator,
    PacketValidator,
    ConfigValidator,
    ValidationError,
    ValidationResult,
    validate_range,
    validate_string_length,
    validate_array_bounds,
};

pub use bounds::{
    BoundsChecker,
    check_array_bounds,
    check_slice_bounds,
    check_numeric_range,
    check_buffer_capacity,
    ensure_packet_size,
    BoundsError,
};

pub use conversion::{
    adc_to_voltage,
    voltage_to_adc,
    samples_to_bytes,
    bytes_to_samples,
    frequency_to_period_nanos,
    period_nanos_to_frequency,
    ConversionError,
};

pub use integrity::{
    calculate_checksum,
    verify_checksum,
    calculate_crc8,
    calculate_crc16,
    verify_packet_integrity,
    ChecksumType,
    IntegrityError,
};

/// Common result type for utility functions
pub type UtilResult<T> = Result<T, UtilError>;

/// Unified error type for all utility functions
#[derive(Debug, Clone, PartialEq)]
pub enum UtilError {
    /// Time-related errors
    Time(time::TimeError),
    /// Validation errors  
    Validation(validation::ValidationError),
    /// Bounds checking errors
    Bounds(bounds::BoundsError),
    /// Conversion errors
    Conversion(conversion::ConversionError),
    /// Data integrity errors
    Integrity(integrity::IntegrityError),
}

impl std::fmt::Display for UtilError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UtilError::Time(e) => write!(f, "Time error: {}", e),
            UtilError::Validation(e) => write!(f, "Validation error: {}", e),
            UtilError::Bounds(e) => write!(f, "Bounds error: {}", e),
            UtilError::Conversion(e) => write!(f, "Conversion error: {}", e),
            UtilError::Integrity(e) => write!(f, "Integrity error: {}", e),
        }
    }
}

impl std::error::Error for UtilError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            UtilError::Time(e) => Some(e),
            UtilError::Validation(e) => Some(e),
            UtilError::Bounds(e) => Some(e),
            UtilError::Conversion(e) => Some(e),
            UtilError::Integrity(e) => Some(e),
        }
    }
}

// Implement From traits for error conversion
impl From<time::TimeError> for UtilError {
    fn from(err: time::TimeError) -> Self {
        UtilError::Time(err)
    }
}

impl From<validation::ValidationError> for UtilError {
    fn from(err: validation::ValidationError) -> Self {
        UtilError::Validation(err)
    }
}

impl From<bounds::BoundsError> for UtilError {
    fn from(err: bounds::BoundsError) -> Self {
        UtilError::Bounds(err)
    }
}

impl From<conversion::ConversionError> for UtilError {
    fn from(err: conversion::ConversionError) -> Self {
        UtilError::Conversion(err)
    }
}

impl From<integrity::IntegrityError> for UtilError {
    fn from(err: integrity::IntegrityError) -> Self {
        UtilError::Integrity(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_conversion() {
        let time_error = time::TimeError::InvalidTimestamp(0);
        let util_error: UtilError = time_error.into();

        match util_error {
            UtilError::Time(_) => {},
            _ => panic!("Expected Time error"),
        }
    }

    #[test]
    fn test_error_display() {
        let error = UtilError::Time(time::TimeError::MonotonicClockNotAvailable);
        let error_string = format!("{}", error);
        assert!(error_string.contains("Time error"));
    }

    /*#[test]
    fn test_error_source() {
        let time_error = time::TimeError::InvalidTimestamp(123);
        let util_error = UtilError::Time(time_error);
        assert!(util_error.source().is_some());
    }*/
}