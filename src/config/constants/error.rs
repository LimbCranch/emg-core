// src/config/constants/error.rs
//! Error handling constants and configuration
//!
//! This module centralizes all constants used in the error handling system
//! to avoid magic numbers and provide consistent configuration across
//! the entire EMG system.

/// Maximum length for error messages to prevent excessive memory usage
pub const MAX_ERROR_MESSAGE_LENGTH: usize = 1024;

/// Maximum length for error component names
pub const MAX_COMPONENT_NAME_LENGTH: usize = 64;

/// Maximum length for error operation names
pub const MAX_OPERATION_NAME_LENGTH: usize = 128;

/// Maximum number of entries in error chain
pub const MAX_ERROR_CHAIN_LENGTH: usize = 16;

/// Maximum number of additional info key-value pairs in error context
pub const MAX_ADDITIONAL_INFO_ENTRIES: usize = 32;

/// Maximum length for additional info keys
pub const MAX_INFO_KEY_LENGTH: usize = 64;

/// Maximum length for additional info values
pub const MAX_INFO_VALUE_LENGTH: usize = 256;

/// Default retry count for communication errors
pub const DEFAULT_COMMUNICATION_RETRY_COUNT: u32 = 3;

/// Maximum retry count for any operation
pub const MAX_RETRY_COUNT: u32 = 10;

/// Default timeout for error recovery operations (milliseconds)
pub const DEFAULT_ERROR_RECOVERY_TIMEOUT_MS: u32 = 5000;

/// Maximum timeout for error recovery operations (milliseconds)
pub const MAX_ERROR_RECOVERY_TIMEOUT_MS: u32 = 30000;

/// Minimum timeout for error recovery operations (milliseconds)
pub const MIN_ERROR_RECOVERY_TIMEOUT_MS: u32 = 100;

/// Default buffer size for error logging (bytes)
pub const DEFAULT_ERROR_LOG_BUFFER_SIZE: usize = 8192;

/// Maximum buffer size for error logging (bytes)
pub const MAX_ERROR_LOG_BUFFER_SIZE: usize = 1024 * 1024; // 1MB

/// Number of recent errors to keep in memory for debugging
pub const ERROR_HISTORY_SIZE: usize = 100;

/// Maximum size for serialized error data (bytes)
pub const MAX_SERIALIZED_ERROR_SIZE: usize = 4096;

/// Real-time constraint violation thresholds
pub mod real_time_thresholds {
    /// Sample deadline threshold in nanoseconds (1ms)
    pub const SAMPLE_DEADLINE_THRESHOLD_NS: u64 = 1_000_000;

    /// Processing latency threshold in nanoseconds (100μs)
    pub const PROCESSING_LATENCY_THRESHOLD_NS: u64 = 100_000;

    /// Buffer underrun threshold in nanoseconds (10μs)
    pub const BUFFER_UNDERRUN_THRESHOLD_NS: u64 = 10_000;

    /// Synchronization drift threshold in nanoseconds (50μs)
    pub const SYNC_DRIFT_THRESHOLD_NS: u64 = 50_000;

    /// Interrupt latency threshold in nanoseconds (10μs)
    pub const INTERRUPT_LATENCY_THRESHOLD_NS: u64 = 10_000;

    /// Critical timing threshold multiplier (3x normal threshold)
    pub const CRITICAL_TIMING_MULTIPLIER: u64 = 3;

    /// High priority timing threshold multiplier (2x normal threshold)
    pub const HIGH_PRIORITY_TIMING_MULTIPLIER: u64 = 2;
}

/// Buffer overflow thresholds and limits
pub mod buffer_thresholds {
    /// Default ring buffer size (samples)
    pub const DEFAULT_RING_BUFFER_SIZE: usize = 8192;

    /// Maximum ring buffer size (samples)
    pub const MAX_RING_BUFFER_SIZE: usize = 1024 * 1024; // 1M samples

    /// Minimum ring buffer size (samples)
    pub const MIN_RING_BUFFER_SIZE: usize = 64;

    /// Buffer utilization warning threshold (percentage)
    pub const BUFFER_UTILIZATION_WARNING_PERCENT: f32 = 75.0;

    /// Buffer utilization critical threshold (percentage)
    pub const BUFFER_UTILIZATION_CRITICAL_PERCENT: f32 = 90.0;

    /// Buffer overflow protection threshold (percentage)
    pub const BUFFER_OVERFLOW_PROTECTION_PERCENT: f32 = 95.0;

    /// Default packet buffer size (bytes)
    pub const DEFAULT_PACKET_BUFFER_SIZE: usize = 4096;

    /// Maximum packet buffer size (bytes)
    pub const MAX_PACKET_BUFFER_SIZE: usize = 64 * 1024; // 64KB
}

/// Memory allocation thresholds and limits
pub mod memory_thresholds {
    /// Default memory allocation warning threshold (bytes)
    pub const MEMORY_WARNING_THRESHOLD_BYTES: usize = 1024 * 1024; // 1MB

    /// Maximum single allocation size (bytes)
    pub const MAX_SINGLE_ALLOCATION_BYTES: usize = 16 * 1024 * 1024; // 16MB

    /// Memory pressure detection threshold (percentage of available memory)
    pub const MEMORY_PRESSURE_THRESHOLD_PERCENT: f32 = 80.0;

    /// Emergency memory threshold (percentage of available memory)
    pub const EMERGENCY_MEMORY_THRESHOLD_PERCENT: f32 = 95.0;

    /// Memory allocation retry count
    pub const MEMORY_ALLOCATION_RETRY_COUNT: u32 = 3;

    /// Memory allocation retry delay (milliseconds)
    pub const MEMORY_ALLOCATION_RETRY_DELAY_MS: u32 = 100;
}

/// Communication error thresholds and timeouts
pub mod communication_thresholds {
    /// Default communication timeout (milliseconds)
    pub const DEFAULT_COMMUNICATION_TIMEOUT_MS: u32 = 1000;

    /// Minimum communication timeout (milliseconds)
    pub const MIN_COMMUNICATION_TIMEOUT_MS: u32 = 10;

    /// Maximum communication timeout (milliseconds)
    pub const MAX_COMMUNICATION_TIMEOUT_MS: u32 = 30000;

    /// Serial communication baud rate limits
    pub const MIN_SERIAL_BAUD_RATE: u32 = 9600;
    pub const MAX_SERIAL_BAUD_RATE: u32 = 2_000_000;

    /// USB transfer timeout (milliseconds)
    pub const USB_TRANSFER_TIMEOUT_MS: u32 = 1000;

    /// Network socket timeout (milliseconds)
    pub const NETWORK_SOCKET_TIMEOUT_MS: u32 = 5000;

    /// Maximum packet size for communication (bytes)
    pub const MAX_COMMUNICATION_PACKET_SIZE: usize = 8192;

    /// Connection retry backoff base delay (milliseconds)
    pub const CONNECTION_RETRY_BASE_DELAY_MS: u32 = 100;

    /// Connection retry maximum delay (milliseconds)
    pub const CONNECTION_RETRY_MAX_DELAY_MS: u32 = 5000;

    /// Connection retry backoff multiplier
    pub const CONNECTION_RETRY_BACKOFF_MULTIPLIER: f32 = 2.0;
}

/// Signal processing error thresholds
pub mod signal_processing_thresholds {
    /// Maximum signal amplitude (normalized)
    pub const MAX_SIGNAL_AMPLITUDE: f32 = 10.0;

    /// Minimum signal amplitude (normalized)
    pub const MIN_SIGNAL_AMPLITUDE: f32 = -10.0;

    /// Signal saturation threshold (percentage of max amplitude)
    pub const SIGNAL_SATURATION_THRESHOLD_PERCENT: f32 = 95.0;

    /// Signal clipping detection threshold
    pub const SIGNAL_CLIPPING_THRESHOLD: f32 = 0.99;

    /// Maximum filter coefficient value
    pub const MAX_FILTER_COEFFICIENT: f64 = 10.0;

    /// Minimum filter coefficient value
    pub const MIN_FILTER_COEFFICIENT: f64 = -10.0;

    /// FFT size limits
    pub const MIN_FFT_SIZE: usize = 64;
    pub const MAX_FFT_SIZE: usize = 16384;

    /// Sample rate limits (Hz)
    pub const MIN_SAMPLE_RATE_HZ: u32 = 100;
    pub const MAX_SAMPLE_RATE_HZ: u32 = 50000;

    /// Channel count limits
    pub const MIN_CHANNEL_COUNT: usize = 1;
    pub const MAX_CHANNEL_COUNT: usize = 64;
}

/// Device validation thresholds
pub mod device_validation_thresholds {
    /// Maximum device initialization time (milliseconds)
    pub const MAX_DEVICE_INIT_TIME_MS: u32 = 10000;

    /// Device health check interval (milliseconds)
    pub const DEVICE_HEALTH_CHECK_INTERVAL_MS: u32 = 1000;

    /// Device response timeout (milliseconds)
    pub const DEVICE_RESPONSE_TIMEOUT_MS: u32 = 500;

    /// Maximum consecutive device errors before marking as failed
    pub const MAX_CONSECUTIVE_DEVICE_ERRORS: u32 = 5;

    /// Device recovery cooldown period (milliseconds)
    pub const DEVICE_RECOVERY_COOLDOWN_MS: u32 = 5000;

    /// Self-test timeout (milliseconds)
    pub const SELF_TEST_TIMEOUT_MS: u32 = 30000;

    /// Calibration timeout (milliseconds)
    pub const CALIBRATION_TIMEOUT_MS: u32 = 60000;

    /// Maximum temperature for normal operation (Celsius)
    pub const MAX_OPERATING_TEMPERATURE_C: f32 = 70.0;

    /// Minimum temperature for normal operation (Celsius)
    pub const MIN_OPERATING_TEMPERATURE_C: f32 = -10.0;

    /// Maximum power consumption warning threshold (milliwatts)
    pub const MAX_POWER_CONSUMPTION_MW: f32 = 5000.0;
}

/// Error severity escalation thresholds
pub mod severity_thresholds {
    /// Number of low severity errors before escalating to medium
    pub const LOW_TO_MEDIUM_ERROR_COUNT: u32 = 10;

    /// Number of medium severity errors before escalating to high
    pub const MEDIUM_TO_HIGH_ERROR_COUNT: u32 = 5;

    /// Number of high severity errors before escalating to critical
    pub const HIGH_TO_CRITICAL_ERROR_COUNT: u32 = 3;

    /// Time window for error count evaluation (milliseconds)
    pub const ERROR_COUNT_TIME_WINDOW_MS: u32 = 10000;

    /// Automatic severity escalation timeout (milliseconds)
    pub const SEVERITY_ESCALATION_TIMEOUT_MS: u32 = 30000;

    /// Severity de-escalation cooldown period (milliseconds)
    pub const SEVERITY_DEESCALATION_COOLDOWN_MS: u32 = 60000;
}

/// Resource exhaustion thresholds
pub mod resource_thresholds {
    /// Maximum number of open file descriptors
    pub const MAX_FILE_DESCRIPTORS: usize = 1024;

    /// Maximum number of network connections
    pub const MAX_NETWORK_CONNECTIONS: usize = 100;

    /// Maximum number of worker threads
    pub const MAX_WORKER_THREADS: usize = 16;

    /// Maximum CPU usage percentage for normal operation
    pub const MAX_CPU_USAGE_PERCENT: f32 = 80.0;

    /// Maximum disk usage percentage
    pub const MAX_DISK_USAGE_PERCENT: f32 = 90.0;

    /// Resource monitoring interval (milliseconds)
    pub const RESOURCE_MONITOR_INTERVAL_MS: u32 = 5000;

    /// Resource cleanup threshold (percentage)
    pub const RESOURCE_CLEANUP_THRESHOLD_PERCENT: f32 = 75.0;
}

/// Error logging and debugging constants
pub mod logging_constants {
    /// Maximum log file size (bytes)
    pub const MAX_LOG_FILE_SIZE_BYTES: usize = 100 * 1024 * 1024; // 100MB

    /// Number of log files to keep in rotation
    pub const LOG_FILE_ROTATION_COUNT: usize = 5;

    /// Log entry maximum length (characters)
    pub const MAX_LOG_ENTRY_LENGTH: usize = 4096;

    /// Structured log field maximum length
    pub const MAX_LOG_FIELD_LENGTH: usize = 512;

    /// Error correlation ID length
    pub const ERROR_CORRELATION_ID_LENGTH: usize = 16;

    /// Debug trace maximum depth
    pub const MAX_DEBUG_TRACE_DEPTH: usize = 32;

    /// Performance metric sampling interval (milliseconds)
    pub const PERFORMANCE_SAMPLING_INTERVAL_MS: u32 = 1000;
}

/// Validation and sanitization constants
pub mod validation_constants {
    /// Maximum string input length for validation
    pub const MAX_STRING_INPUT_LENGTH: usize = 1024;

    /// Minimum string input length for validation
    pub const MIN_STRING_INPUT_LENGTH: usize = 1;

    /// Maximum numeric value for validation
    pub const MAX_NUMERIC_VALUE: f64 = 1e12;

    /// Minimum numeric value for validation
    pub const MIN_NUMERIC_VALUE: f64 = -1e12;

    /// Regular expression pattern length limit
    pub const MAX_REGEX_PATTERN_LENGTH: usize = 256;

    /// Configuration key maximum length
    pub const MAX_CONFIG_KEY_LENGTH: usize = 128;

    /// Configuration value maximum length
    pub const MAX_CONFIG_VALUE_LENGTH: usize = 1024;

    /// Checksum validation timeout (milliseconds)
    pub const CHECKSUM_VALIDATION_TIMEOUT_MS: u32 = 1000;
}

/// Performance and optimization constants
pub mod performance_constants {
    /// Cache size for error message formatting
    pub const ERROR_MESSAGE_CACHE_SIZE: usize = 256;

    /// Error context pool size for reuse
    pub const ERROR_CONTEXT_POOL_SIZE: usize = 128;

    /// Maximum error processing time (microseconds)
    pub const MAX_ERROR_PROCESSING_TIME_US: u64 = 100;

    /// Error handling performance benchmark threshold (errors/second)
    pub const ERROR_HANDLING_PERFORMANCE_THRESHOLD: u64 = 100_000;

    /// Memory allocation alignment for error structures
    pub const ERROR_STRUCT_ALIGNMENT: usize = 64;

    /// Error serialization buffer pre-allocation size
    pub const ERROR_SERIALIZATION_BUFFER_SIZE: usize = 2048;
}

/// Default error messages for common scenarios
pub mod default_messages {
    pub const DEVICE_NOT_INITIALIZED: &str = "Device has not been initialized";
    pub const DEVICE_NOT_CONNECTED: &str = "Device is not connected";
    pub const INVALID_CONFIGURATION: &str = "Invalid device configuration";
    pub const COMMUNICATION_TIMEOUT: &str = "Communication timeout occurred";
    pub const BUFFER_OVERFLOW: &str = "Buffer overflow detected";
    pub const SAMPLE_RATE_INVALID: &str = "Invalid sample rate specified";
    pub const CHANNEL_COUNT_INVALID: &str = "Invalid channel count specified";
    pub const REAL_TIME_VIOLATION: &str = "Real-time constraint violation";
    pub const MEMORY_ALLOCATION_FAILED: &str = "Memory allocation failed";
    pub const RESOURCE_EXHAUSTED: &str = "System resource exhausted";
    pub const PERMISSION_DENIED: &str = "Permission denied";
    pub const FILE_NOT_FOUND: &str = "Required file not found";
    pub const NETWORK_UNREACHABLE: &str = "Network unreachable";
    pub const OPERATION_CANCELLED: &str = "Operation was cancelled";
    pub const OPERATION_TIMEOUT: &str = "Operation timed out";
}

/// Error code ranges for different subsystems
pub mod error_codes {
    // Device error codes: 1000-1999
    pub const DEVICE_ERROR_BASE: i32 = 1000;
    pub const DEVICE_NOT_FOUND: i32 = 1001;
    pub const DEVICE_INITIALIZATION_FAILED: i32 = 1002;
    pub const DEVICE_COMMUNICATION_FAILED: i32 = 1003;
    pub const DEVICE_CALIBRATION_FAILED: i32 = 1004;
    pub const DEVICE_SELF_TEST_FAILED: i32 = 1005;

    // Configuration error codes: 2000-2999
    pub const CONFIG_ERROR_BASE: i32 = 2000;
    pub const CONFIG_INVALID_FORMAT: i32 = 2001;
    pub const CONFIG_MISSING_REQUIRED: i32 = 2002;
    pub const CONFIG_VALUE_OUT_OF_RANGE: i32 = 2003;
    pub const CONFIG_VALIDATION_FAILED: i32 = 2004;

    // Communication error codes: 3000-3999
    pub const COMM_ERROR_BASE: i32 = 3000;
    pub const COMM_TIMEOUT: i32 = 3001;
    pub const COMM_PROTOCOL_ERROR: i32 = 3002;
    pub const COMM_CHECKSUM_MISMATCH: i32 = 3003;
    pub const COMM_CONNECTION_LOST: i32 = 3004;

    // Processing error codes: 4000-4999
    pub const PROCESSING_ERROR_BASE: i32 = 4000;
    pub const PROCESSING_FILTER_OVERFLOW: i32 = 4001;
    pub const PROCESSING_FFT_FAILED: i32 = 4002;
    pub const PROCESSING_INVALID_SAMPLE: i32 = 4003;

    // System error codes: 5000-5999
    pub const SYSTEM_ERROR_BASE: i32 = 5000;
    pub const SYSTEM_OUT_OF_MEMORY: i32 = 5001;
    pub const SYSTEM_RESOURCE_EXHAUSTED: i32 = 5002;
    pub const SYSTEM_PERMISSION_DENIED: i32 = 5003;
    pub const SYSTEM_IO_ERROR: i32 = 5004;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_are_reasonable() {
        // Test that constants have reasonable values
        assert!(MAX_ERROR_MESSAGE_LENGTH > 0);
        assert!(MAX_ERROR_MESSAGE_LENGTH < 10000);

        assert!(DEFAULT_COMMUNICATION_RETRY_COUNT > 0);
        assert!(DEFAULT_COMMUNICATION_RETRY_COUNT <= MAX_RETRY_COUNT);

        assert!(MIN_ERROR_RECOVERY_TIMEOUT_MS < DEFAULT_ERROR_RECOVERY_TIMEOUT_MS);
        assert!(DEFAULT_ERROR_RECOVERY_TIMEOUT_MS < MAX_ERROR_RECOVERY_TIMEOUT_MS);

        // Test buffer thresholds
        assert!(buffer_thresholds::MIN_RING_BUFFER_SIZE < buffer_thresholds::DEFAULT_RING_BUFFER_SIZE);
        assert!(buffer_thresholds::DEFAULT_RING_BUFFER_SIZE < buffer_thresholds::MAX_RING_BUFFER_SIZE);

        // Test real-time thresholds
        assert!(real_time_thresholds::BUFFER_UNDERRUN_THRESHOLD_NS <
            real_time_thresholds::PROCESSING_LATENCY_THRESHOLD_NS);
        assert!(real_time_thresholds::PROCESSING_LATENCY_THRESHOLD_NS <
            real_time_thresholds::SAMPLE_DEADLINE_THRESHOLD_NS);
    }

    #[test]
    fn test_error_code_ranges() {
        // Test that error code ranges don't overlap
        assert!(error_codes::DEVICE_ERROR_BASE < error_codes::CONFIG_ERROR_BASE);
        assert!(error_codes::CONFIG_ERROR_BASE < error_codes::COMM_ERROR_BASE);
        assert!(error_codes::COMM_ERROR_BASE < error_codes::PROCESSING_ERROR_BASE);
        assert!(error_codes::PROCESSING_ERROR_BASE < error_codes::SYSTEM_ERROR_BASE);

        // Test specific error codes are within their ranges
        assert!(error_codes::DEVICE_NOT_FOUND > error_codes::DEVICE_ERROR_BASE);
        assert!(error_codes::DEVICE_NOT_FOUND < error_codes::CONFIG_ERROR_BASE);
    }

    #[test]
    fn test_performance_constants() {
        // Test that performance constants are reasonable
        assert!(performance_constants::ERROR_MESSAGE_CACHE_SIZE > 0);
        assert!(performance_constants::ERROR_CONTEXT_POOL_SIZE > 0);
        assert!(performance_constants::MAX_ERROR_PROCESSING_TIME_US > 0);
        assert!(performance_constants::ERROR_HANDLING_PERFORMANCE_THRESHOLD > 1000);
    }

    #[test]
    fn test_default_messages() {
        // Test that default messages are not empty
        assert!(!default_messages::DEVICE_NOT_INITIALIZED.is_empty());
        assert!(!default_messages::INVALID_CONFIGURATION.is_empty());
        assert!(!default_messages::COMMUNICATION_TIMEOUT.is_empty());

        // Test message lengths are reasonable
        assert!(default_messages::DEVICE_NOT_INITIALIZED.len() < MAX_ERROR_MESSAGE_LENGTH);
        assert!(default_messages::REAL_TIME_VIOLATION.len() < MAX_ERROR_MESSAGE_LENGTH);
    }
}