use std::error::Error;
use std::fmt;

/// Unified error type for EMG system
#[derive(Debug)]
pub enum EmgError {
    /// Device-specific errors
    Device(Box<dyn Error + Send + Sync>),
    /// Configuration-related errors
    Configuration(String),
    /// Signal processing errors
    Processing(String),
    /// Buffer overflow or underflow
    BufferOverflow { channel: usize, details: String },
    /// Invalid input data
    InvalidData(String),
    /// Communication/connection errors
    Communication(String),
    /// Timing/synchronization errors
    Timing(String),
}

impl fmt::Display for EmgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmgError::Device(e) => write!(f, "Device error: {}", e),
            EmgError::Configuration(msg) => write!(f, "Configuration error: {}", msg),
            EmgError::Processing(msg) => write!(f, "Processing error: {}", msg),
            EmgError::BufferOverflow { channel, details } => {
                write!(f, "Buffer overflow on channel {}: {}", channel, details)
            }
            EmgError::InvalidData(msg) => write!(f, "Invalid data: {}", msg),
            EmgError::Communication(msg) => write!(f, "Communication error: {}", msg),
            EmgError::Timing(msg) => write!(f, "Timing error: {}", msg),
        }
    }
}

impl Error for EmgError {}

// FIX: Conversion from specific error types
impl From<crate::hal::simulator::SimulatorError> for EmgError {
    fn from(err: crate::hal::simulator::SimulatorError) -> Self {
        EmgError::Device(Box::new(err))
    }
}

impl From<crate::hal::serial_driver::SerialError> for EmgError {
    fn from(err: crate::hal::serial_driver::SerialError) -> Self {
        EmgError::Device(Box::new(err))
    }
}