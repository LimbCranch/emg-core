// src/error.rs
//! Unified error handling system for EMG Core
//!
//! This module provides a comprehensive error system that standardizes error handling
//! across all EMG system components, ensuring consistent error propagation, context
//! preservation, and debugging capabilities.

use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::time::SystemTime;
use serde::{Deserialize, Serialize};

/// Unified error type for the entire EMG system
///
/// This enum provides a standardized way to handle errors across all system components.
/// All device-specific errors are converted to this type to ensure consistent error
/// handling and improve debugging capabilities.
#[derive(Debug, Clone)]
pub enum EmgError {
    /// Device-specific hardware errors
    Device {
        device_type: DeviceType,
        error: Arc<dyn Error + Send + Sync>,
        context: ErrorContext,
    },

    /// Configuration and setup errors
    Configuration {
        component: String,
        reason: String,
        context: ErrorContext,
    },

    /// Signal processing errors
    Processing {
        stage: ProcessingStage,
        reason: String,
        context: ErrorContext,
    },

    /// Buffer overflow or underflow errors
    BufferOverflow {
        buffer_type: BufferType,
        channel: Option<usize>,
        capacity: usize,
        attempted_size: usize,
        context: ErrorContext,
    },

    /// Invalid input data errors
    InvalidData {
        data_type: String,
        reason: String,
        expected: Option<String>,
        actual: Option<String>,
        context: ErrorContext,
    },

    /// Communication and connection errors
    Communication {
        protocol: CommunicationProtocol,
        operation: String,
        reason: String,
        retry_count: u32,
        context: ErrorContext,
    },

    /// Timing and synchronization errors
    Timing {
        expected_timing: Option<u64>,
        actual_timing: Option<u64>,
        drift_ns: Option<i64>,
        reason: String,
        context: ErrorContext,
    },

    /// Memory allocation and management errors
    Memory {
        operation: String,
        requested_bytes: Option<usize>,
        available_bytes: Option<usize>,
        context: ErrorContext,
    },

    /// Real-time constraint violations
    RealTime {
        constraint_type: RealTimeConstraint,
        deadline_ns: Option<u64>,
        actual_duration_ns: Option<u64>,
        severity: Severity,
        context: ErrorContext,
    },

    /// Authentication and authorization errors
    Security {
        operation: String,
        reason: String,
        context: ErrorContext,
    },

    /// Resource exhaustion errors
    ResourceExhausted {
        resource_type: ResourceType,
        limit: usize,
        requested: usize,
        context: ErrorContext,
    },

    /// System-level errors
    System {
        subsystem: String,
        error_code: Option<i32>,
        reason: String,
        context: ErrorContext,
    },
}

/// Device types for error categorization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DeviceType {
    Simulator,
    UsbEmg,
    SerialEmg,
    NetworkEmg,
    FileSource,
    Unknown(String),
}

/// Signal processing stages for error tracking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProcessingStage {
    Acquisition,
    Filtering,
    Amplification,
    DigitalSignalProcessing,
    FeatureExtraction,
    MachineLearning,
    Output,
}

/// Buffer types for overflow tracking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BufferType {
    RingBuffer,
    CircularBuffer,
    LinearBuffer,
    PacketBuffer,
    SampleBuffer,
}

/// Communication protocols for error tracking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CommunicationProtocol {
    Serial,
    Usb,
    Tcp,
    Udp,
    I2c,
    Spi,
    Custom(String),
}

/// Real-time constraint types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RealTimeConstraint {
    SampleDeadline,
    ProcessingLatency,
    BufferUnderrun,
    SynchronizationDrift,
    InterruptLatency,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Severity {
    Low,      // System can continue normally
    Medium,   // System degraded but functional
    High,     // System compromised, immediate attention needed
    Critical, // System failure imminent, emergency stop required
}

/// Resource types for exhaustion tracking
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResourceType {
    Memory,
    FileDescriptors,
    NetworkConnections,
    Threads,
    CpuTime,
    DiskSpace,
    BusCapacity,
}

/// Error context for debugging and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub timestamp: SystemTime,
    pub thread_id: Option<String>,
    pub component: String,
    pub operation: String,
    pub file: Option<&'static str>,
    pub line: Option<u32>,
    pub additional_info: std::collections::HashMap<String, String>,
    pub chain: Vec<String>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(component: &str, operation: &str) -> Self {
        Self {
            timestamp: SystemTime::now(),
            thread_id: Self::current_thread_id(),
            component: component.to_string(),
            operation: operation.to_string(),
            file: None,
            line: None,
            additional_info: std::collections::HashMap::new(),
            chain: Vec::new(),
        }
    }

    /// Create error context with file and line information
    pub fn with_location(
        component: &str,
        operation: &str,
        file: &'static str,
        line: u32,
    ) -> Self {
        let mut context = Self::new(component, operation);
        context.file = Some(file);
        context.line = Some(line);
        context
    }

    /// Add additional information to the context
    pub fn add_info<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.additional_info.insert(key.into(), value.into());
        self
    }

    /// Add to the error chain
    pub fn add_to_chain(mut self, error: &str) -> Self {
        self.chain.push(error.to_string());
        self
    }

    fn current_thread_id() -> Option<String> {
        std::thread::current().name().map(|s| s.to_string())
    }
}

/// Macro for creating error context with file and line info
#[macro_export]
macro_rules! error_context {
    ($component:expr, $operation:expr) => {
        $crate::error::ErrorContext::with_location($component, $operation, file!(), line!())
    };
}

impl fmt::Display for EmgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmgError::Device { device_type, error, context } => {
                write!(f, "[{}] Device error in {}: {} (at {}:{})",
                       device_type, context.component, error,
                       context.file.unwrap_or("unknown"), context.line.unwrap_or(0))
            }
            EmgError::Configuration { component, reason, context } => {
                write!(f, "[CONFIG] Configuration error in {}: {} ({})",
                       component, reason, context.operation)
            }
            EmgError::Processing { stage, reason, context } => {
                write!(f, "[PROCESSING] {:?} stage error: {} ({})",
                       stage, reason, context.operation)
            }
            EmgError::BufferOverflow { buffer_type, channel, capacity, attempted_size, context } => {
                match channel {
                    Some(ch) => write!(f, "[BUFFER] {:?} overflow on channel {}: tried to write {} bytes to {}-byte buffer ({})",
                                       buffer_type, ch, attempted_size, capacity, context.operation),
                    None => write!(f, "[BUFFER] {:?} overflow: tried to write {} bytes to {}-byte buffer ({})",
                                   buffer_type, attempted_size, capacity, context.operation),
                }
            }
            EmgError::InvalidData { data_type, reason, expected, actual, context } => {
                match (expected, actual) {
                    (Some(exp), Some(act)) => write!(f, "[DATA] Invalid {}: {} (expected: {}, got: {}) ({})",
                                                     data_type, reason, exp, act, context.operation),
                    _ => write!(f, "[DATA] Invalid {}: {} ({})", data_type, reason, context.operation),
                }
            }
            EmgError::Communication { protocol, operation, reason, retry_count, context } => {
                write!(f, "[COMM] {:?} communication error during {}: {} (retries: {}) ({})",
                       protocol, operation, reason, retry_count, context.operation)
            }
            EmgError::Timing { expected_timing, actual_timing, drift_ns, reason, context } => {
                match (expected_timing, actual_timing, drift_ns) {
                    (Some(exp), Some(act), Some(drift)) => {
                        write!(f, "[TIMING] Timing violation: {} (expected: {}ns, actual: {}ns, drift: {}ns) ({})",
                               reason, exp, act, drift, context.operation)
                    }
                    _ => write!(f, "[TIMING] Timing error: {} ({})", reason, context.operation),
                }
            }
            EmgError::Memory { operation, requested_bytes, available_bytes, context } => {
                match (requested_bytes, available_bytes) {
                    (Some(req), Some(avail)) => write!(f, "[MEMORY] Memory error during {}: requested {} bytes, {} available ({})",
                                                       operation, req, avail, context.operation),
                    _ => write!(f, "[MEMORY] Memory error during {}: {} ", operation, context.operation),
                }
            }
            EmgError::RealTime { constraint_type, deadline_ns, actual_duration_ns, severity, context } => {
                write!(f, "[RT-{:?}] Real-time constraint violation ({:?}): {:?} ({})",
                       severity, constraint_type, deadline_ns, context.operation)
            }
            EmgError::Security { operation, reason, context } => {
                write!(f, "[SECURITY] Security error during {}: {} ({})", operation, reason, context.operation)
            }
            EmgError::ResourceExhausted { resource_type, limit, requested, context } => {
                write!(f, "[RESOURCE] {:?} exhausted: requested {}, limit {} ({})",
                       resource_type, requested, limit, context.operation)
            }
            EmgError::System { subsystem, error_code, reason, context } => {
                match error_code {
                    Some(code) => write!(f, "[SYSTEM] {} error (code: {}): {} ({})",
                                         subsystem, code, reason, context.operation),
                    None => write!(f, "[SYSTEM] {} error: {} ({})", subsystem, reason, context.operation),
                }
            }
        }
    }
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Simulator => write!(f, "SIMULATOR"),
            DeviceType::UsbEmg => write!(f, "USB-EMG"),
            DeviceType::SerialEmg => write!(f, "SERIAL-EMG"),
            DeviceType::NetworkEmg => write!(f, "NETWORK-EMG"),
            DeviceType::FileSource => write!(f, "FILE-SOURCE"),
            DeviceType::Unknown(name) => write!(f, "UNKNOWN-{}", name),
        }
    }
}

impl Error for EmgError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            EmgError::Device { error, .. } => Some(error.as_ref()),
            _ => None,
        }
    }
}

// Ensure EmgError is Send + Sync
unsafe impl Send for EmgError {}
unsafe impl Sync for EmgError {}

/// Conversion from simulator errors
impl From<crate::hal::simulator::SimulatorError> for EmgError {
    fn from(err: crate::hal::simulator::SimulatorError) -> Self {
        let context = error_context!("simulator", "device_operation");
        EmgError::Device {
            device_type: DeviceType::Simulator,
            error: Arc::new(err),
            context,
        }
    }
}

/// Conversion from USB errors
impl From<crate::hal::usb_driver::UsbError> for EmgError {
    fn from(err: crate::hal::usb_driver::UsbError) -> Self {
        let context = error_context!("usb_driver", "device_operation");
        EmgError::Device {
            device_type: DeviceType::UsbEmg,
            error: Arc::new(err),
            context,
        }
    }
}

/// Conversion from serial errors
impl From<crate::hal::serial_driver::SerialError> for EmgError {
    fn from(err: crate::hal::serial_driver::SerialError) -> Self {
        let context = error_context!("serial_driver", "device_operation");
        EmgError::Device {
            device_type: DeviceType::SerialEmg,
            error: Arc::new(err),
            context,
        }
    }
}

/// Result type alias for EMG operations
pub type EmgResult<T> = Result<T, EmgError>;

/// Error builder for convenient error construction
pub struct EmgErrorBuilder {
    component: String,
    operation: String,
}

impl EmgErrorBuilder {
    pub fn new(component: &str, operation: &str) -> Self {
        Self {
            component: component.to_string(),
            operation: operation.to_string(),
        }
    }

    pub fn configuration(self, reason: &str) -> EmgError {
        
        let context = ErrorContext::new(&self.component, &self.operation);
        
        
        EmgError::Configuration {
            component: self.component,
            reason: reason.to_string(),
            context,
        }
    }

    pub fn processing(self, stage: ProcessingStage, reason: &str) -> EmgError {
        EmgError::Processing {
            stage,
            reason: reason.to_string(),
            context: ErrorContext::new(&self.component, &self.operation),
        }
    }

    pub fn buffer_overflow(
        self,
        buffer_type: BufferType,
        capacity: usize,
        attempted_size: usize,
    ) -> EmgError {
        EmgError::BufferOverflow {
            buffer_type,
            channel: None,
            capacity,
            attempted_size,
            context: ErrorContext::new(&self.component, &self.operation),
        }
    }

    pub fn invalid_data(self, data_type: &str, reason: &str) -> EmgError {
        EmgError::InvalidData {
            data_type: data_type.to_string(),
            reason: reason.to_string(),
            expected: None,
            actual: None,
            context: ErrorContext::new(&self.component, &self.operation),
        }
    }

    pub fn communication(
        self,
        protocol: CommunicationProtocol,
        reason: &str,
        retry_count: u32,
    ) -> EmgError {
        EmgError::Communication {
            protocol,
            operation: self.operation.clone(),
            reason: reason.to_string(),
            retry_count,
            context: ErrorContext::new(&self.component, &self.operation),
        }
    }

    pub fn timing(self, reason: &str) -> EmgError {
        EmgError::Timing {
            expected_timing: None,
            actual_timing: None,
            drift_ns: None,
            reason: reason.to_string(),
            context: ErrorContext::new(&self.component, &self.operation),
        }
    }

    pub fn real_time(
        self,
        constraint_type: RealTimeConstraint,
        severity: Severity,
    ) -> EmgError {
        EmgError::RealTime {
            constraint_type,
            deadline_ns: None,
            actual_duration_ns: None,
            severity,
            context: ErrorContext::new(&self.component, &self.operation),
        }
    }
}

/// Convenience trait for error building
pub trait IntoEmgError<T> {
    fn emg_err(self, component: &str, operation: &str) -> EmgResult<T>;
}

impl<T, E> IntoEmgError<T> for Result<T, E>
where
    E: Error + Send + Sync + 'static,
{
    fn emg_err(self, component: &str, operation: &str) -> EmgResult<T> {
        self.map_err(|err| {
            EmgError::System {
                subsystem: component.to_string(),
                error_code: None,
                reason: err.to_string(),
                context: ErrorContext::new(component, operation),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_creation() {
        let context = ErrorContext::new("test_component", "test_operation");
        assert_eq!(context.component, "test_component");
        assert_eq!(context.operation, "test_operation");
        assert!(context.timestamp <= SystemTime::now());
    }

    #[test]
    fn test_error_builder() {
        let builder = EmgErrorBuilder::new("test_component", "test_op");
        let err = builder.configuration("test configuration error");

        match err {
            EmgError::Configuration { component, reason, .. } => {
                assert_eq!(component, "test_component");
                assert_eq!(reason, "test configuration error");
            }
            _ => panic!("Expected configuration error"),
        }
    }

    #[test]
    fn test_error_display() {
        let err = EmgError::BufferOverflow {
            buffer_type: BufferType::RingBuffer,
            channel: Some(1),
            capacity: 1024,
            attempted_size: 2048,
            context: ErrorContext::new("test", "test_op"),
        };

        let display = format!("{}", err);
        assert!(display.contains("RingBuffer"));
        assert!(display.contains("channel 1"));
        assert!(display.contains("2048"));
        assert!(display.contains("1024"));
    }

    #[test]
    fn test_error_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EmgError>();
    }

    #[test]
    fn test_error_conversion() {
        use crate::hal::simulator::SimulatorError;

        let sim_err = SimulatorError::ConfigurationError("test".to_string());
        let emg_err: EmgError = sim_err.into();

        match emg_err {
            EmgError::Device { device_type, .. } => {
                assert_eq!(device_type, DeviceType::Simulator);
            }
            _ => panic!("Expected device error"),
        }
    }

    #[test]
    fn test_into_emg_error_trait() {
        let result: Result<i32, std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "test error"
        ));

        let emg_result = result.emg_err("test_component", "test_operation");
        assert!(emg_result.is_err());

        match emg_result.unwrap_err() {
            EmgError::System { subsystem, reason, .. } => {
                assert_eq!(subsystem, "test_component");
                assert!(reason.contains("test error"));
            }
            _ => panic!("Expected system error"),
        }
    }
}