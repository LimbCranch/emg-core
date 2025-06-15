// src/validation/mod.rs
//! Validation utilities for input data and configurations

use std::error::Error;
use std::fmt;

/// Generic validator trait
pub trait Validator<T> {
    type Error: Error + Send + Sync + 'static;

    fn validate(&self, input: &T) -> Result<(), Self::Error>;
}

/// Validation error types
#[derive(Debug, Clone)]
pub enum ValidationError {
    TooSmall(usize, usize), // actual, expected_min
    TooLarge(usize, usize), // actual, max_allowed
    InvalidFormat(String),
    InvalidRange(String),
    CorruptedData(String),
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::TooSmall(actual, expected) => {
                write!(f, "Input too small: got {}, expected at least {}", actual, expected)
            }
            ValidationError::TooLarge(actual, max) => {
                write!(f, "Input too large: got {}, maximum allowed {}", actual, max)
            }
            ValidationError::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
            ValidationError::InvalidRange(msg) => write!(f, "Value out of range: {}", msg),
            ValidationError::CorruptedData(msg) => write!(f, "Corrupted data: {}", msg),
        }
    }
}

impl Error for ValidationError {}

/// Packet validator for serial/USB communications
#[derive(Clone, Debug)]
pub struct PacketValidator {
    min_size: usize,
    max_size: usize,
    expected_header: Vec<u8>,
    checksum_enabled: bool,
}

impl PacketValidator {
    pub fn new(min_size: usize, max_size: usize, header: Vec<u8>) -> Self {
        Self {
            min_size,
            max_size,
            expected_header: header,
            checksum_enabled: false,
        }
    }

    pub fn with_checksum(mut self) -> Self {
        self.checksum_enabled = true;
        self
    }
}

impl Validator<Vec<u8>> for PacketValidator {
    type Error = ValidationError;

    fn validate(&self, packet: &Vec<u8>) -> Result<(), Self::Error> {
        // Size validation
        if packet.len() < self.min_size {
            return Err(ValidationError::TooSmall(packet.len(), self.min_size));
        }
        if packet.len() > self.max_size {
            return Err(ValidationError::TooLarge(packet.len(), self.max_size));
        }

        // Header validation
        if packet.len() < self.expected_header.len() {
            return Err(ValidationError::InvalidFormat(
                "Packet smaller than header".to_string()
            ));
        }
        if &packet[..self.expected_header.len()] != &self.expected_header {
            return Err(ValidationError::InvalidFormat(
                "Invalid packet header".to_string()
            ));
        }

        // Checksum validation if enabled
        if self.checksum_enabled && packet.len() > 0 {
            let data_len = packet.len() - 1;
            let expected_checksum = packet[data_len];
            let calculated_checksum = packet[..data_len]
                .iter()
                .fold(0u8, |acc, &b| acc.wrapping_add(b));

            if expected_checksum != calculated_checksum {
                return Err(ValidationError::CorruptedData(
                    format!("Checksum mismatch: expected 0x{:02X}, got 0x{:02X}",
                            calculated_checksum, expected_checksum)
                ));
            }
        }

        Ok(())
    }
}

/// Range validator for numeric values
pub struct RangeValidator<T> {
    min: T,
    max: T,
}

impl<T> RangeValidator<T> {
    pub fn new(min: T, max: T) -> Self {
        Self { min, max }
    }
}

impl<T> Validator<T> for RangeValidator<T>
where
    T: PartialOrd + fmt::Display + Clone,
{
    type Error = ValidationError;

    fn validate(&self, value: &T) -> Result<(), Self::Error> {
        if value < &self.min || value > &self.max {
            return Err(ValidationError::InvalidRange(
                format!("Value {} outside range [{}, {}]", value, self.min, self.max)
            ));
        }
        Ok(())
    }
}

/// String length validator
pub struct LengthValidator {
    min_length: usize,
    max_length: usize,
}

impl LengthValidator {
    pub fn new(min_length: usize, max_length: usize) -> Self {
        Self { min_length, max_length }
    }
}

impl Validator<String> for LengthValidator {
    type Error = ValidationError;

    fn validate(&self, input: &String) -> Result<(), Self::Error> {
        if input.len() < self.min_length {
            return Err(ValidationError::TooSmall(input.len(), self.min_length));
        }
        if input.len() > self.max_length {
            return Err(ValidationError::TooLarge(input.len(), self.max_length));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_validator() {
        let header = vec![0xAA, 0x55];
        let validator = PacketValidator::new(10, 100, header.clone());

        // Test valid packet
        let mut valid_packet = header.clone();
        valid_packet.extend_from_slice(&[0u8; 8]); // Add data to meet min size
        assert!(validator.validate(&valid_packet).is_ok());

        // Test too small
        let small_packet = vec![0xAA];
        assert!(validator.validate(&small_packet).is_err());

        // Test too large
        let large_packet = vec![0u8; 200];
        assert!(validator.validate(&large_packet).is_err());

        // Test invalid header
        let mut invalid_header = vec![0xFF, 0x55];
        invalid_header.extend_from_slice(&[0u8; 8]);
        assert!(validator.validate(&invalid_header).is_err());
    }

    #[test]
    fn test_checksum_validator() {
        let header = vec![0xAA, 0x55];
        let validator = PacketValidator::new(5, 100, header.clone()).with_checksum();

        // Create packet with correct checksum
        let mut packet = header;
        packet.extend_from_slice(&[0x01, 0x02]);
        let checksum = packet.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
        packet.push(checksum);

        assert!(validator.validate(&packet).is_ok());

        // Corrupt checksum
        if let Some(last) = packet.last_mut() {
            *last = last.wrapping_add(1);
        }
        assert!(validator.validate(&packet).is_err());
    }

    #[test]
    fn test_range_validator() {
        let validator = RangeValidator::new(0, 100);

        assert!(validator.validate(&50).is_ok());
        assert!(validator.validate(&0).is_ok());
        assert!(validator.validate(&100).is_ok());
        assert!(validator.validate(&-1).is_err());
        assert!(validator.validate(&101).is_err());
    }

    #[test]
    fn test_length_validator() {
        let validator = LengthValidator::new(5, 20);

        assert!(validator.validate(&"hello".to_string()).is_ok());
        assert!(validator.validate(&"hi".to_string()).is_err()); // Too short
        assert!(validator.validate(&"this is way too long for the validator".to_string()).is_err()); // Too long
    }
}