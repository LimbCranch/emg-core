use std::error::Error;
use std::fmt;

pub trait Validator<T> {
    type Error;
    fn validate(&self, input: &T) -> Result<(), Self::Error>;
}

/// Validation error types
#[derive(Debug)]
pub enum ValidationError {
    TooSmall(usize, usize), // actual, expected
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
                    format!("Checksum mismatch: expected {}, got {}",
                            expected_checksum, calculated_checksum)
                ));
            }
        }

        Ok(())
    }
}

/// EMG sample validator
pub struct SampleValidator {
    expected_channels: usize,
    voltage_range: (f32, f32),
}

impl SampleValidator {
    pub fn new(channels: usize, min_voltage: f32, max_voltage: f32) -> Self {
        Self {
            expected_channels: channels,
            voltage_range: (min_voltage, max_voltage),
        }
    }
}

impl Validator<Vec<f32>> for SampleValidator {
    type Error = ValidationError;

    fn validate(&self, channels: &Vec<f32>) -> Result<(), Self::Error> {
        if channels.len() != self.expected_channels {
            return Err(ValidationError::InvalidFormat(
                format!("Expected {} channels, got {}", self.expected_channels, channels.len())
            ));
        }

        for (i, &value) in channels.iter().enumerate() {
            if !value.is_finite() {
                return Err(ValidationError::InvalidRange(
                    format!("Channel {} has non-finite value: {}", i, value)
                ));
            }
            if value < self.voltage_range.0 || value > self.voltage_range.1 {
                return Err(ValidationError::InvalidRange(
                    format!("Channel {} value {} outside range [{}, {}]",
                            i, value, self.voltage_range.0, self.voltage_range.1)
                ));
            }
        }

        Ok(())
    }
}

// ================================================================================
// CRITICAL FIX #6: Device State Management
// File: src/hal/device_state.rs (NEW FILE)
// ================================================================================

/// Better device state management instead of Option<T>
#[derive(Debug, Clone)]
pub enum DeviceState<T> {
    Disconnected,
    Connecting,
    Connected(T),
    Error(String),
}

impl<T> DeviceState<T> {
    pub fn is_connected(&self) -> bool {
        matches!(self, DeviceState::Connected(_))
    }

    pub fn is_disconnected(&self) -> bool {
        matches!(self, DeviceState::Disconnected)
    }

    pub fn get_handle(&self) -> Option<&T> {
        match self {
            DeviceState::Connected(handle) => Some(handle),
            _ => None,
        }
    }

    pub fn take_handle(self) -> Option<T> {
        match self {
            DeviceState::Connected(handle) => Some(handle),
            _ => None,
        }
    }
}

// ================================================================================
// USAGE EXAMPLES AND TESTS
// ================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_validator() {
        let validator = PacketValidator::new(5, 100, vec![0xAA, 0x55])
            .with_checksum();

        // Valid packet
        let valid_packet = vec![0xAA, 0x55, 0x01, 0x02, 0x03];
        assert!(validator.validate(&valid_packet).is_ok());

        // Too small
        let small_packet = vec![0xAA, 0x55];
        assert!(validator.validate(&small_packet).is_err());

        // Wrong header
        let wrong_header = vec![0xBB, 0x55, 0x01, 0x02, 0x03];
        assert!(validator.validate(&wrong_header).is_err());
    }

    /*#[test]
    fn test_mock_time_provider() {
        let provider = MockTimeProvider::new(1000);
        assert_eq!(provider.now_nanos(), 1000);

        provider.advance_by(500);
        assert_eq!(provider.now_nanos(), 1500);

        provider.set_time(2000);
        assert_eq!(provider.now_nanos(), 2000);
    }*/

    #[test]
    fn test_device_state() {
        let state: DeviceState<String> = DeviceState::Disconnected;
        assert!(!state.is_connected());

        let state = DeviceState::Connected("device_handle".to_string());
        assert!(state.is_connected());
        assert_eq!(state.get_handle(), Some(&"device_handle".to_string()));
    }
}