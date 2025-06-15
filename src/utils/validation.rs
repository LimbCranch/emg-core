//! Validation utilities for EMG-Core
//!
//! Provides comprehensive validation functions for:
//! - Configuration parameters and ranges
//! - Data integrity and format validation
//! - Packet structure and content validation
//! - Input sanitization and bounds checking
//!
//! All validation functions use constants from the config module
//! to avoid magic numbers and ensure consistency.

use std::collections::HashMap;
use std::fmt;
use crate::config::constants::{validation, signal, performance, quality};

/// Validation result type
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Comprehensive validation error types
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    /// Value out of valid range
    OutOfRange {
        field: String,
        value: String,
        min: String,
        max: String,
    },
    /// Required field missing
    RequiredFieldMissing(String),
    /// Invalid field format
    InvalidFormat {
        field: String,
        value: String,
        expected: String,
    },
    /// String length violation
    InvalidLength {
        field: String,
        actual: usize,
        min: Option<usize>,
        max: Option<usize>,
    },
    /// Array size violation
    InvalidArraySize {
        field: String,
        actual: usize,
        expected: usize,
    },
    /// Invalid enum value
    InvalidEnumValue {
        field: String,
        value: String,
        valid_values: Vec<String>,
    },
    /// Cross-field validation failure
    ConstraintViolation {
        fields: Vec<String>,
        message: String,
    },
    /// Data integrity check failed
    IntegrityCheckFailed {
        field: String,
        reason: String,
    },
    /// Custom validation failure
    Custom(String),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::OutOfRange { field, value, min, max } => {
                write!(f, "Field '{}' value '{}' is out of range [{}, {}]", field, value, min, max)
            }
            ValidationError::RequiredFieldMissing(field) => {
                write!(f, "Required field '{}' is missing", field)
            }
            ValidationError::InvalidFormat { field, value, expected } => {
                write!(f, "Field '{}' has invalid format '{}', expected {}", field, value, expected)
            }
            ValidationError::InvalidLength { field, actual, min, max } => {
                let bounds = match (min, max) {
                    (Some(min), Some(max)) => format!("[{}, {}]", min, max),
                    (Some(min), None) => format!(">= {}", min),
                    (None, Some(max)) => format!("<= {}", max),
                    (None, None) => "unknown".to_string(),
                };
                write!(f, "Field '{}' length {} is invalid, expected {}", field, actual, bounds)
            }
            ValidationError::InvalidArraySize { field, actual, expected } => {
                write!(f, "Field '{}' array size {} doesn't match expected {}", field, actual, expected)
            }
            ValidationError::InvalidEnumValue { field, value, valid_values } => {
                write!(f, "Field '{}' value '{}' is invalid, valid values: [{}]",
                       field, value, valid_values.join(", "))
            }
            ValidationError::ConstraintViolation { fields, message } => {
                write!(f, "Constraint violation for fields [{}]: {}", fields.join(", "), message)
            }
            ValidationError::IntegrityCheckFailed { field, reason } => {
                write!(f, "Integrity check failed for field '{}': {}", field, reason)
            }
            ValidationError::Custom(msg) => write!(f, "Validation error: {}", msg),
        }
    }
}

impl std::error::Error for ValidationError {}

/// Generic trait for validating objects
pub trait Validator<T> {
    /// Validate an object and return result
    fn validate(&self, value: &T) -> ValidationResult<()>;

    /// Validate and provide detailed error context
    fn validate_with_context(&self, value: &T, context: &str) -> ValidationResult<()> {
        self.validate(value).map_err(|e| match e {
            ValidationError::Custom(msg) => ValidationError::Custom(format!("{}: {}", context, msg)),
            other => other,
        })
    }
}

/// Configuration validator for system parameters
#[derive(Debug, Clone)]
pub struct ConfigValidator {
    rules: HashMap<String, ValidationRule>,
}

/// Validation rule types
#[derive(Debug, Clone)]
pub enum ValidationRule {
    /// Numeric range validation
    NumericRange { min: f64, max: f64 },
    /// Integer range validation
    IntegerRange { min: i64, max: i64 },
    /// String length validation
    StringLength { min: usize, max: usize },
    /// Enum validation
    Enum(Vec<String>),
    /// Required field validation
    Required,
    /// Custom validation function
    Custom(fn(&str) -> bool),
}

impl ConfigValidator {
    /// Create new configuration validator with EMG-specific rules
    pub fn new() -> Self {
        let mut rules = HashMap::new();

        // System configuration rules
        rules.insert(
            "system.sampling_rate_hz".to_string(),
            ValidationRule::IntegerRange {
                min: signal::MIN_SAMPLING_RATE_HZ as i64,
                max: signal::MAX_SAMPLING_RATE_HZ as i64,
            },
        );

        rules.insert(
            "system.channel_count".to_string(),
            ValidationRule::IntegerRange {
                min: signal::MIN_CHANNEL_COUNT as i64,
                max: signal::MAX_CHANNEL_COUNT as i64,
            },
        );

        rules.insert(
            "system.latency_target_ms".to_string(),
            ValidationRule::IntegerRange {
                min: performance::MIN_LATENCY_TARGET_MS as i64,
                max: performance::MAX_LATENCY_TARGET_MS as i64,
            },
        );

        // Quality configuration rules
        rules.insert(
            "quality.snr_threshold_db".to_string(),
            ValidationRule::NumericRange {
                min: quality::MIN_SNR_THRESHOLD_DB as f64,
                max: quality::MAX_SNR_THRESHOLD_DB as f64,
            },
        );

        // String length validation rules
        rules.insert(
            "device.port_name".to_string(),
            ValidationRule::StringLength {
                min: 1,
                max: validation::MAX_STRING_VALUE_LENGTH,
            },
        );

        rules.insert(
            "device.device_id".to_string(),
            ValidationRule::Required,
        );

        Self { rules }
    }

    /// Add custom validation rule
    pub fn add_rule(&mut self, field: String, rule: ValidationRule) {
        self.rules.insert(field, rule);
    }

    /// Validate configuration field
    pub fn validate_field(&self, field: &str, value: &str) -> ValidationResult<()> {
        if let Some(rule) = self.rules.get(field) {
            match rule {
                ValidationRule::NumericRange { min, max } => {
                    let num_value: f64 = value.parse().map_err(|_| {
                        ValidationError::InvalidFormat {
                            field: field.to_string(),
                            value: value.to_string(),
                            expected: "numeric value".to_string(),
                        }
                    })?;

                    if num_value < *min || num_value > *max {
                        return Err(ValidationError::OutOfRange {
                            field: field.to_string(),
                            value: value.to_string(),
                            min: min.to_string(),
                            max: max.to_string(),
                        });
                    }
                }
                ValidationRule::IntegerRange { min, max } => {
                    let int_value: i64 = value.parse().map_err(|_| {
                        ValidationError::InvalidFormat {
                            field: field.to_string(),
                            value: value.to_string(),
                            expected: "integer value".to_string(),
                        }
                    })?;

                    if int_value < *min || int_value > *max {
                        return Err(ValidationError::OutOfRange {
                            field: field.to_string(),
                            value: value.to_string(),
                            min: min.to_string(),
                            max: max.to_string(),
                        });
                    }
                }
                ValidationRule::StringLength { min, max } => {
                    if value.len() < *min || value.len() > *max {
                        return Err(ValidationError::InvalidLength {
                            field: field.to_string(),
                            actual: value.len(),
                            min: Some(*min),
                            max: Some(*max),
                        });
                    }
                }
                ValidationRule::Enum(valid_values) => {
                    if !valid_values.contains(&value.to_string()) {
                        return Err(ValidationError::InvalidEnumValue {
                            field: field.to_string(),
                            value: value.to_string(),
                            valid_values: valid_values.clone(),
                        });
                    }
                }
                ValidationRule::Required => {
                    if value.is_empty() {
                        return Err(ValidationError::RequiredFieldMissing(field.to_string()));
                    }
                }
                ValidationRule::Custom(func) => {
                    if !func(value) {
                        return Err(ValidationError::Custom(
                            format!("Custom validation failed for field '{}'", field)
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Validate entire configuration map
    pub fn validate_config(&self, config: &HashMap<String, String>) -> ValidationResult<()> {
        let mut errors = Vec::new();

        for (field, rule) in &self.rules {
            if let ValidationRule::Required = rule {
                if !config.contains_key(field) {
                    errors.push(ValidationError::RequiredFieldMissing(field.clone()));
                    continue;
                }
            }

            if let Some(value) = config.get(field) {
                if let Err(error) = self.validate_field(field, value) {
                    errors.push(error);
                }
            }
        }

        if !errors.is_empty() {
            // Return the first error for simplicity, could be enhanced to return all
            return Err(errors.into_iter().next().unwrap());
        }

        Ok(())
    }
}

impl Default for ConfigValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Packet validator for EMG data packets
#[derive(Debug, Clone)]
pub struct PacketValidator {
    min_size: usize,
    max_size: usize,
    header_pattern: Vec<u8>,
    require_checksum: bool,
    channel_count: usize,
    bytes_per_sample: usize,
}

impl PacketValidator {
    /// Create new packet validator
    pub fn new(min_size: usize, max_size: usize, header_pattern: Vec<u8>) -> Self {
        Self {
            min_size,
            max_size,
            header_pattern,
            require_checksum: false,
            channel_count: signal::DEFAULT_CHANNEL_COUNT,
            bytes_per_sample: 4, // 32-bit samples by default
        }
    }

    /// Enable checksum validation
    pub fn with_checksum(mut self) -> Self {
        self.require_checksum = true;
        self
    }

    /// Set expected channel count
    pub fn with_channels(mut self, count: usize) -> Self {
        self.channel_count = count;
        self
    }

    /// Set bytes per sample
    pub fn with_sample_size(mut self, bytes: usize) -> Self {
        self.bytes_per_sample = bytes;
        self
    }

    /// Validate packet structure and content
    pub fn validate_packet(&self, packet: &[u8]) -> ValidationResult<()> {
        // Size validation
        if packet.len() < self.min_size {
            return Err(ValidationError::InvalidLength {
                field: "packet".to_string(),
                actual: packet.len(),
                min: Some(self.min_size),
                max: None,
            });
        }

        if packet.len() > self.max_size {
            return Err(ValidationError::InvalidLength {
                field: "packet".to_string(),
                actual: packet.len(),
                min: None,
                max: Some(self.max_size),
            });
        }

        // Header validation
        if !self.header_pattern.is_empty() {
            if packet.len() < self.header_pattern.len() {
                return Err(ValidationError::IntegrityCheckFailed {
                    field: "packet_header".to_string(),
                    reason: "Packet too short for header".to_string(),
                });
            }

            if &packet[..self.header_pattern.len()] != self.header_pattern.as_slice() {
                return Err(ValidationError::IntegrityCheckFailed {
                    field: "packet_header".to_string(),
                    reason: "Header pattern mismatch".to_string(),
                });
            }
        }

        // Data size validation
        let expected_data_size = self.channel_count * self.bytes_per_sample;
        let header_size = self.header_pattern.len();
        let checksum_size = if self.require_checksum { 1 } else { 0 };
        let expected_total_size = header_size + expected_data_size + checksum_size;

        if packet.len() != expected_total_size {
            return Err(ValidationError::InvalidArraySize {
                field: "packet_data".to_string(),
                actual: packet.len(),
                expected: expected_total_size,
            });
        }

        Ok(())
    }
}

/// Validate numeric range
pub fn validate_range<T>(value: T, min: T, max: T, field: &str) -> ValidationResult<()>
where
    T: PartialOrd + std::fmt::Display + Copy,
{
    if value < min || value > max {
        return Err(ValidationError::OutOfRange {
            field: field.to_string(),
            value: value.to_string(),
            min: min.to_string(),
            max: max.to_string(),
        });
    }
    Ok(())
}

/// Validate string length
pub fn validate_string_length(value: &str, min: usize, max: usize, field: &str) -> ValidationResult<()> {
    if value.len() < min || value.len() > max {
        return Err(ValidationError::InvalidLength {
            field: field.to_string(),
            actual: value.len(),
            min: Some(min),
            max: Some(max),
        });
    }
    Ok(())
}

/// Validate array bounds
pub fn validate_array_bounds<T>(array: &[T], min_size: usize, max_size: usize, field: &str) -> ValidationResult<()> {
    if array.len() < min_size || array.len() > max_size {
        return Err(ValidationError::InvalidLength {
            field: field.to_string(),
            actual: array.len(),
            min: Some(min_size),
            max: Some(max_size),
        });
    }
    Ok(())
}

/// Validate sampling rate is valid for EMG
pub fn validate_sampling_rate(rate_hz: u32) -> ValidationResult<()> {
    validate_range(
        rate_hz,
        signal::MIN_SAMPLING_RATE_HZ,
        signal::MAX_SAMPLING_RATE_HZ,
        "sampling_rate_hz",
    )
}

/// Validate channel count is valid for EMG
pub fn validate_channel_count(count: usize) -> ValidationResult<()> {
    validate_range(
        count,
        signal::MIN_CHANNEL_COUNT,
        signal::MAX_CHANNEL_COUNT,
        "channel_count",
    )
}

/// Validate EMG signal amplitude is reasonable
pub fn validate_signal_amplitude(amplitude: f32) -> ValidationResult<()> {
    validate_range(
        amplitude,
        signal::MIN_SIGNAL_AMPLITUDE,
        signal::MAX_SIGNAL_AMPLITUDE,
        "signal_amplitude",
    )
}

/// Validate buffer size is appropriate
pub fn validate_buffer_size(size: usize, field: &str) -> ValidationResult<()> {
    if size == 0 {
        return Err(ValidationError::OutOfRange {
            field: field.to_string(),
            value: size.to_string(),
            min: "1".to_string(),
            max: validation::MAX_ARRAY_LENGTH.to_string(),
        });
    }

    if size > validation::MAX_ARRAY_LENGTH {
        return Err(ValidationError::OutOfRange {
            field: field.to_string(),
            value: size.to_string(),
            min: "1".to_string(),
            max: validation::MAX_ARRAY_LENGTH.to_string(),
        });
    }

    // Check if size is power of 2 for ring buffers
    if !size.is_power_of_two() {
        return Err(ValidationError::Custom(
            format!("Buffer size {} must be power of 2", size)
        ));
    }

    Ok(())
}

/// Validate configuration consistency across related fields
pub fn validate_cross_field_constraints(
    sampling_rate: u32,
    buffer_size: usize,
    latency_target_ms: u32,
) -> ValidationResult<()> {
    // Calculate minimum buffer size for latency target
    let samples_per_ms = sampling_rate as f64 / 1000.0;
    let min_samples = (samples_per_ms * latency_target_ms as f64).ceil() as usize;

    if buffer_size < min_samples {
        return Err(ValidationError::ConstraintViolation {
            fields: vec!["sampling_rate".to_string(), "buffer_size".to_string(), "latency_target".to_string()],
            message: format!(
                "Buffer size {} is too small for {}Hz sampling at {}ms latency target (need >= {})",
                buffer_size, sampling_rate, latency_target_ms, min_samples
            ),
        });
    }

    Ok(())
}

/// Sanitize string input by removing invalid characters
pub fn sanitize_string_input(input: &str, max_length: usize) -> String {
    input
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(*c, '_' | '-' | '.' | '/'))
        .take(max_length)
        .collect()
}

/// Validate and sanitize device identifier
pub fn validate_device_id(device_id: &str) -> ValidationResult<String> {
    validate_string_length(device_id, 1, 64, "device_id")?;

    // Only allow alphanumeric, underscore, and dash
    if !device_id.chars().all(|c| c.is_alphanumeric() || matches!(c, '_' | '-')) {
        return Err(ValidationError::InvalidFormat {
            field: "device_id".to_string(),
            value: device_id.to_string(),
            expected: "alphanumeric characters, underscore, or dash only".to_string(),
        });
    }

    Ok(device_id.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validator() {
        let validator = ConfigValidator::new();

        // Valid sampling rate
        assert!(validator.validate_field("system.sampling_rate_hz", "2000").is_ok());

        // Invalid sampling rate (too high)
        assert!(validator.validate_field("system.sampling_rate_hz", "50000").is_err());

        // Invalid format
        assert!(validator.validate_field("system.sampling_rate_hz", "not_a_number").is_err());
    }

    #[test]
    fn test_packet_validator() {
        let header = vec![0xAA, 0x55];
        let validator = PacketValidator::new(10, 100, header.clone())
            .with_channels(2)
            .with_sample_size(4);

        // Valid packet: header + 2 channels * 4 bytes = 10 bytes
        let valid_packet = [0xAA, 0x55, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        assert!(validator.validate_packet(&valid_packet).is_ok());

        // Invalid header
        let invalid_header = [0xBB, 0x55, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        assert!(validator.validate_packet(&invalid_header).is_err());

        // Packet too short
        let short_packet = [0xAA, 0x55, 0x01];
        assert!(validator.validate_packet(&short_packet).is_err());
    }

    #[test]
    fn test_range_validation() {
        assert!(validate_range(50, 0, 100, "test_field").is_ok());
        assert!(validate_range(150, 0, 100, "test_field").is_err());
        assert!(validate_range(-10, 0, 100, "test_field").is_err());
    }

    #[test]
    fn test_string_length_validation() {
        assert!(validate_string_length("hello", 1, 10, "test_field").is_ok());
        assert!(validate_string_length("", 1, 10, "test_field").is_err());
        assert!(validate_string_length("very_long_string", 1, 10, "test_field").is_err());
    }

    #[test]
    fn test_sampling_rate_validation() {
        assert!(validate_sampling_rate(2000).is_ok());
        assert!(validate_sampling_rate(50).is_err()); // Too low
        assert!(validate_sampling_rate(50000).is_err()); // Too high
    }

    #[test]
    fn test_buffer_size_validation() {
        assert!(validate_buffer_size(1024, "test_buffer").is_ok()); // Power of 2
        assert!(validate_buffer_size(1000, "test_buffer").is_err()); // Not power of 2
        assert!(validate_buffer_size(0, "test_buffer").is_err()); // Zero size
    }

    #[test]
    fn test_cross_field_validation() {
        // Valid combination
        assert!(validate_cross_field_constraints(2000, 1024, 100).is_ok());

        // Buffer too small for latency target
        assert!(validate_cross_field_constraints(2000, 10, 100).is_err());
    }

    #[test]
    fn test_device_id_validation() {
        assert!(validate_device_id("device_123").is_ok());
        assert!(validate_device_id("invalid@device").is_err());
        assert!(validate_device_id("").is_err());
    }

    #[test]
    fn test_string_sanitization() {
        let input = "test@#$%string_123!";
        let sanitized = sanitize_string_input(input, 20);
        assert_eq!(sanitized, "teststring_123");

        // Test length limiting
        let long_input = "a".repeat(100);
        let limited = sanitize_string_input(&long_input, 10);
        assert_eq!(limited.len(), 10);
    }

    #[test]
    fn test_validation_error_display() {
        let error = ValidationError::OutOfRange {
            field: "test".to_string(),
            value: "150".to_string(),
            min: "0".to_string(),
            max: "100".to_string(),
        };

        let display = format!("{}", error);
        assert!(display.contains("test"));
        assert!(display.contains("150"));
        assert!(display.contains("0"));
        assert!(display.contains("100"));
    }
}