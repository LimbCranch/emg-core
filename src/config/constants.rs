// src/config/constants.rs
//! System-wide configuration constants
//! UPDATED: Added all missing constants required by utility functions

mod error;

/// Signal processing constants
pub mod signal {
    pub const DEFAULT_SAMPLING_RATE_HZ: u32 = 2000;
    pub const DEFAULT_CHANNEL_COUNT: usize = 8;
    pub const DEFAULT_BUFFER_SIZE_SAMPLES: usize = 4096;
    pub const MIN_SAMPLING_RATE_HZ: u32 = 500;
    pub const MAX_SAMPLING_RATE_HZ: u32 = 10000;
    pub const MAX_CHANNEL_COUNT: usize = 32;
    pub const MIN_CHANNEL_COUNT: usize = 1;

    // ADC resolution constants
    pub const DEFAULT_ADC_RESOLUTION_BITS: u8 = 24;
    pub const MIN_ADC_RESOLUTION_BITS: u8 = 8;
    pub const MAX_ADC_RESOLUTION_BITS: u8 = 32;

    
    // Signal range constants
    pub const DEFAULT_INPUT_RANGE_MV: (f32, f32) = (-2500.0, 2500.0);
    pub const SIGNAL_SATURATION_THRESHOLD: f32 = 0.95;
    pub const SIGNAL_CLAMP_MIN: f32 = -1.0;
    pub const SIGNAL_CLAMP_MAX: f32 = 1.0;

    // ADDED: Signal amplitude ranges for validation
    pub const MIN_SIGNAL_AMPLITUDE: f32 = -10.0; // -10V (unreasonably large but safe bound)
    pub const MAX_SIGNAL_AMPLITUDE: f32 = 10.0; // +10V (unreasonably large but safe bound)
    pub const DEFAULT_SIGNAL_AMPLITUDE: f32 = 0.001; // 1mV typical EMG amplitude
    pub const TYPICAL_EMG_AMPLITUDE_MV: f32 = 1.0; // 1mV
    pub const MAX_REASONABLE_EMG_AMPLITUDE_MV: f32 = 10.0; // 10mV

    // ADDED: Sample format constants
    pub const DEFAULT_SAMPLE_FORMAT: &str = "signed_int";
    pub const SUPPORTED_SAMPLE_FORMATS: &[&str] =
        &["signed_int", "unsigned_int", "float32", "float64"];

    // ADDED: Frequency analysis constants
    pub const EMG_FREQUENCY_RANGE_HZ: (f32, f32) = (20.0, 500.0); // Typical EMG bandwidth
    pub const POWERLINE_INTERFERENCE_50HZ: f32 = 50.0;
    pub const POWERLINE_INTERFERENCE_60HZ: f32 = 60.0;

    /// Sample rate limits (Hz)
    pub const MIN_SAMPLE_RATE_HZ: u32 = 100;
    pub const MAX_SAMPLE_RATE_HZ: u32 = 50000;




}

/// Performance and timing constants
pub mod performance {
    
    pub const DEFAULT_LATENCY_TARGET_MS: u32 = 20;
    pub const MAX_LATENCY_TARGET_MS: u32 = 100;
    pub const MIN_LATENCY_TARGET_MS: u32 = 1;
    pub const WATCHDOG_TIMEOUT_MS: u32 = 100;
    pub const PROCESSING_TIMEOUT_MS: u32 = 50;

    // Timing precision constants
    pub const NANOSECONDS_PER_SECOND: u64 = 1_000_000_000;
    pub const MICROSECONDS_PER_SECOND: u64 = 1_000_000;
    pub const MILLISECONDS_PER_SECOND: u64 = 1_000;

    // Sample period calculation helpers
    pub const MIN_SAMPLE_PERIOD_NANOS: u64 = NANOSECONDS_PER_SECOND / MAX_SAMPLING_RATE_HZ as u64;
    pub const MAX_SAMPLE_PERIOD_NANOS: u64 = NANOSECONDS_PER_SECOND / MIN_SAMPLING_RATE_HZ as u64;

    // ADDED: Time provider constants
    pub const MONOTONIC_TIME_RESOLUTION_NANOS: u64 = 1; // 1 nanosecond resolution
    pub const SYSTEM_TIME_RESOLUTION_NANOS: u64 = 1000; // 1 microsecond typical resolution
    pub const TIME_DRIFT_WARNING_THRESHOLD_NANOS: u64 = 1_000_000; // 1ms drift warning
    pub const TIME_DRIFT_ERROR_THRESHOLD_NANOS: u64 = 10_000_000; // 10ms drift error

    // ADDED: Performance monitoring constants
    pub const PERFORMANCE_SAMPLE_INTERVAL_MS: u64 = 1000; // 1 second sampling
    pub const MAX_TIMESTAMP_AGE_SECONDS: u64 = 3600; // 1 hour
    pub const MIN_TIMESTAMP_AGE_SECONDS: u64 = 0;

    use super::signal::{MAX_SAMPLING_RATE_HZ, MIN_SAMPLING_RATE_HZ};
}

/// Hardware abstraction layer constants
pub mod hal {

    // Base HAL constants
    pub const DEFAULT_CONNECTION_TIMEOUT_MS: u32 = 5000;
    pub const DEFAULT_RETRY_ATTEMPTS: u32 = 3;
    pub const MAX_RETRY_ATTEMPTS: u32 = 10;
    pub const DEVICE_DISCOVERY_TIMEOUT_MS: u32 = 10000;
    pub const MAX_CONNECTION_TIMEOUT_MS: u32 = 30000;

    // Retry timing constants
    pub const RETRY_INITIAL_DELAY_MS: u32 = 100;
    pub const RETRY_MAX_DELAY_MS: u32 = 5000;
    pub const RETRY_MULTIPLIER: f32 = 2.0;

    // Device capability defaults
    pub const DEFAULT_MIN_CHANNELS: usize = 1;

    // Simulator constants
    pub const DEFAULT_NOISE_LEVEL: f32 = 0.1;
    pub const DEFAULT_ARTIFACT_PROBABILITY: f32 = 0.02;
    pub const DEFAULT_GESTURE_DURATION_MS: u32 = 1000;
    pub const DEFAULT_POWERLINE_FREQ_HZ: f32 = 50.0;
    pub const DEFAULT_AMPLITUDE_RANGE_MV: (f32, f32) = (-2500.0, 2500.0);
    pub const DEFAULT_BASE_IMPEDANCE_KOHM: f32 = 10.0;

    // USB device constants
    pub const DEFAULT_USB_VENDOR_ID: u16 = 0x1234;
    pub const DEFAULT_USB_PRODUCT_ID: u16 = 0x5678;
    pub const DEFAULT_USB_INTERFACE: u8 = 0;
    pub const DEFAULT_USB_TIMEOUT_MS: u32 = 1000;
    pub const DEFAULT_USB_BUFFER_SIZE: usize = 4096;
    pub const DEFAULT_USB_PACKET_SIZE: usize = 512;
    pub const DEFAULT_USB_BUFFER_COUNT: usize = 4;
    pub const MAX_USB_TIMEOUT_MS: u32 = 10000;
    pub const MAX_USB_BUFFER_SIZE: usize = 1048576; // 1MB
    pub const DEFAULT_USB_DATA_IN_ENDPOINT: u8 = 0x81;
    pub const DEFAULT_USB_CONTROL_ENDPOINT: u8 = 0x00;

    // USB protocol constants
    pub const DEFAULT_PACKET_HEADER_SIZE: usize = 8;
    pub const MAX_PACKET_LOSS_PERCENT: f32 = 5.0;
    pub const USB_IDLE_TIMEOUT_SEC: u32 = 300; // 5 minutes

    // Serial device constants
    pub const DEFAULT_SERIAL_PORT: &str = "/dev/ttyUSB0";
    pub const DEFAULT_BAUD_RATE: u32 = 115200;
    pub const MIN_BAUD_RATE: u32 = 300;
    pub const MAX_BAUD_RATE: u32 = 3000000;
    pub const DEFAULT_DATA_BITS: u8 = 8;
    pub const DEFAULT_SERIAL_TIMEOUT_MS: u32 = 1000;
    pub const DEFAULT_SERIAL_RX_BUFFER_SIZE: usize = 8192;
    pub const DEFAULT_SERIAL_TX_BUFFER_SIZE: usize = 1024;
    pub const DEFAULT_MAX_FRAME_SIZE: usize = 2048;
    pub const DEFAULT_FRAME_TIMEOUT_MS: u32 = 100;

    // Bluetooth constants
    pub const DEFAULT_BT_SERVICE_UUID: &str = "12345678-1234-1234-1234-123456789abc";
    pub const DEFAULT_BT_DATA_CHAR_UUID: &str = "87654321-4321-4321-4321-cba987654321";
    pub const DEFAULT_BT_CONNECTION_TIMEOUT_MS: u32 = 10000;
    pub const DEFAULT_BT_SCAN_TIMEOUT_MS: u32 = 30000;
    pub const DEFAULT_BT_CONNECTION_INTERVAL_MS: f32 = 15.0;
    pub const DEFAULT_BT_MTU: u16 = 247;
    pub const MIN_BT_MTU: u16 = 23;
    pub const MAX_BT_MTU: u16 = 512;
}

/// Communication layer constants
pub mod communication {
    pub const DEFAULT_SHARED_MEMORY_SIZE_MB: usize = 64;
    pub const DEFAULT_MESSAGE_QUEUE_SIZE: usize = 1024;
    pub const MAX_MESSAGE_SIZE_BYTES: usize = 65536;
    pub const HEARTBEAT_INTERVAL_MS: u32 = 1000;
}

/// Signal processing filter constants
pub mod filters {
    pub const DEFAULT_HIGHPASS_CUTOFF_HZ: f32 = 20.0;
    pub const DEFAULT_LOWPASS_CUTOFF_HZ: f32 = 500.0;
    pub const DEFAULT_FILTER_ORDER: usize = 4;
    pub const MIN_FILTER_ORDER: usize = 1;
    pub const MAX_FILTER_ORDER: usize = 8;
    pub const POWERLINE_FREQ_50HZ: f32 = 50.0;
    pub const POWERLINE_FREQ_60HZ: f32 = 60.0;
    pub const DEFAULT_NOTCH_BANDWIDTH_HZ: f32 = 2.0;

    // Filter response constants
    pub const FILTER_SETTLING_TIME_SAMPLES: usize = 100;
    pub const MIN_CUTOFF_FREQUENCY_HZ: f32 = 1.0;
    pub const MAX_CUTOFF_FREQUENCY_HZ: f32 = 1000.0;
}

/// Quality monitoring constants
pub mod quality {
    pub const DEFAULT_SNR_THRESHOLD_DB: f32 = 20.0;
    pub const MIN_SNR_THRESHOLD_DB: f32 = 0.0;
    pub const MAX_SNR_THRESHOLD_DB: f32 = 60.0;
    pub const DEFAULT_CONTACT_IMPEDANCE_MAX_KOHM: f32 = 50.0;
    pub const DEFAULT_SATURATION_THRESHOLD: f32 = 0.95;
    pub const ARTIFACT_DETECTION_THRESHOLD: f32 = 2.0;

    // Quality assessment constants
    pub const SNR_EXCELLENT_THRESHOLD_DB: f32 = 40.0;
    pub const SNR_GOOD_THRESHOLD_DB: f32 = 25.0;
    pub const SNR_POOR_THRESHOLD_DB: f32 = 15.0;
    pub const IMPEDANCE_EXCELLENT_THRESHOLD_KOHM: f32 = 10.0;
    pub const IMPEDANCE_GOOD_THRESHOLD_KOHM: f32 = 25.0;
    pub const IMPEDANCE_POOR_THRESHOLD_KOHM: f32 = 50.0;

    // Motion artifact detection constants
    pub const MOTION_ARTIFACT_VARIANCE_THRESHOLD: f32 = 0.25;
    pub const MOTION_ARTIFACT_PEAK_THRESHOLD: f32 = 0.8;
    pub const MIN_ARTIFACT_DURATION_MS: u64 = 10;
    pub const MAX_ARTIFACT_DURATION_MS: u64 = 500;

    // ADDED: Advanced quality metrics
    pub const SIGNAL_QUALITY_EXCELLENT_THRESHOLD: f32 = 0.95;
    pub const SIGNAL_QUALITY_GOOD_THRESHOLD: f32 = 0.80;
    pub const SIGNAL_QUALITY_POOR_THRESHOLD: f32 = 0.60;
    pub const SIGNAL_QUALITY_UNACCEPTABLE_THRESHOLD: f32 = 0.40;

    // ADDED: Quality calculation constants
    pub const QUALITY_ASSESSMENT_WINDOW_SIZE: usize = 1000; // samples
    pub const QUALITY_UPDATE_INTERVAL_MS: u32 = 100; // Update every 100ms
    pub const QUALITY_HISTORY_LENGTH: usize = 100; // Keep 100 quality measurements

    // ADDED: Advanced artifact detection
    pub const BASELINE_DRIFT_THRESHOLD_UV: f32 = 100.0; // 100µV baseline drift
    pub const POWERLINE_CONTAMINATION_THRESHOLD: f32 = 0.1; // 10% of signal
    pub const EMG_FREQUENCY_CONTENT_THRESHOLD: f32 = 0.8; // 80% energy in EMG band
}

/// Windowing constants
pub mod windowing {
    pub const DEFAULT_WINDOW_SIZE_SAMPLES: usize = 256;
    pub const DEFAULT_OVERLAP_PERCENT: f32 = 50.0;
    pub const MIN_OVERLAP_PERCENT: f32 = 0.0;
    pub const MAX_OVERLAP_PERCENT: f32 = 99.0;
    pub const MIN_WINDOW_SIZE: usize = 16;
    pub const MAX_WINDOW_SIZE: usize = 8192;

    // Window function constants
    pub const HANN_WINDOW_ALPHA: f32 = 0.5;
    pub const HAMMING_WINDOW_ALPHA: f32 = 0.54;
    pub const HAMMING_WINDOW_BETA: f32 = 0.46;
    pub const BLACKMAN_WINDOW_A0: f32 = 0.42;
    pub const BLACKMAN_WINDOW_A1: f32 = 0.5;
    pub const BLACKMAN_WINDOW_A2: f32 = 0.08;
}

/// Simulator constants
pub mod simulation {
    pub const DEFAULT_NOISE_LEVEL: f32 = 0.05;
    pub const DEFAULT_ARTIFACT_PROBABILITY: f32 = 0.02;
    pub const DEFAULT_GESTURE_DURATION_MS: u32 = 1000;
    pub const MIN_NOISE_LEVEL: f32 = 0.0;
    pub const MAX_NOISE_LEVEL: f32 = 1.0;
    pub const MIN_ARTIFACT_PROBABILITY: f32 = 0.0;
    pub const MAX_ARTIFACT_PROBABILITY: f32 = 1.0;

    // Simulator timing constants
    pub const MIN_GESTURE_DURATION_MS: u32 = 100;
    pub const MAX_GESTURE_DURATION_MS: u32 = 10000;
    pub const SIMULATOR_TIMESTAMP_PRECISION_NANOS: u64 = 1000; // 1 microsecond

    // Muscle simulation constants
    pub const DEFAULT_MUSCLE_ACTIVATION_THRESHOLD: f32 = 0.1;
    pub const MAX_MUSCLE_ACTIVATION_LEVEL: f32 = 1.0;
    pub const MIN_MUSCLE_ACTIVATION_LEVEL: f32 = 0.0;
    pub const DEFAULT_MUSCLE_FREQ_LOW_HZ: f32 = 20.0;
    pub const DEFAULT_MUSCLE_FREQ_HIGH_HZ: f32 = 150.0;

    // Powerline noise constants
    pub const POWERLINE_NOISE_AMPLITUDE: f32 = 0.02; // 2% of full scale
    pub const POWERLINE_PHASE_INCREMENT: f32 = 2.0 * std::f32::consts::PI;

    // Artifact simulation constants
    pub const MIN_ARTIFACT_INTERVAL_NANOS: u64 = 50_000_000; // 50ms
    pub const MAX_ARTIFACT_AMPLITUDE: f32 = 0.5;
    pub const THERMAL_NOISE_FACTOR: f32 = 1.0;
    pub const CHANNEL_ARTIFACT_SCALING: f32 = 1.0;
}

/// Serial communication constants
pub mod serial {
    // Serial protocol constants
    pub const DEFAULT_BAUD_RATE: u32 = 115200;
    pub const MIN_BAUD_RATE: u32 = 9600;
    pub const MAX_BAUD_RATE: u32 = 4_000_000;

    pub const DEFAULT_DATA_BITS: u8 = 8;
    pub const MIN_DATA_BITS: u8 = 5;
    pub const MAX_DATA_BITS: u8 = 8;

    pub const DEFAULT_STOP_BITS: u8 = 1;
    pub const MIN_STOP_BITS: u8 = 1;
    pub const MAX_STOP_BITS: u8 = 2;

    // Packet structure constants
    pub const DEFAULT_HEADER_BYTES: [u8; 4] = [0xAA, 0x55, 0xA5, 0x5A];
    pub const DEFAULT_FOOTER_BYTES: [u8; 2] = [0xDE, 0xAD];
    pub const DEFAULT_BYTES_PER_CHANNEL: usize = 4;

    // Timeout constants
    pub const DEFAULT_TIMEOUT_MS: u32 = 1000;
    pub const MIN_TIMEOUT_MS: u32 = 10;
    pub const MAX_TIMEOUT_MS: u32 = 60000;

    // Packet validation constants
    pub const MIN_PACKET_HEADER_SIZE: usize = 2;
    pub const MAX_PACKET_HEADER_SIZE: usize = 8;
    pub const CHECKSUM_SIZE_BYTES: usize = 1;
    pub const CRC16_SIZE_BYTES: usize = 2;

    // CRC constants
    pub const CRC8_POLYNOMIAL: u8 = 0x07; // CRC-8-CCITT
    pub const CRC8_INIT_VALUE: u8 = 0xFF;
    pub const CRC16_POLYNOMIAL: u16 = 0x1021; // CRC-16-CCITT
    pub const CRC16_INIT_VALUE: u16 = 0xFFFF;

    // Channel conversion constants
    pub const ADC_8BIT_SCALE: f32 = 128.0;
    pub const ADC_16BIT_SCALE: f32 = 32768.0;
    pub const ADC_24BIT_SCALE: f32 = 8388608.0;
    pub const ADC_24BIT_SIGN_MASK: u8 = 0x80;
    pub const ADC_24BIT_SIGN_EXTEND: u8 = 0xFF;

    // ADDED: Protocol timing constants
    pub const PACKET_SYNC_TIMEOUT_MS: u32 = 100;
    pub const INTER_PACKET_GAP_MS: u32 = 1;
    pub const MAX_PACKET_AGE_MS: u32 = 1000;

    // ADDED: Error detection constants
    pub const MAX_CONSECUTIVE_ERRORS: u32 = 10;
    pub const ERROR_RECOVERY_DELAY_MS: u32 = 100;
    pub const CHECKSUM_MISMATCH_THRESHOLD: u32 = 5;

    // ADDED: Channel mapping constants
    pub const MAX_CHANNELS_PER_PACKET: usize = 32;
    pub const MIN_CHANNELS_PER_PACKET: usize = 1;
    pub const DEFAULT_CHANNEL_MAPPING: &[usize] = &[0, 1, 2, 3, 4, 5, 6, 7];
}

/// USB communication constants
pub mod usb {
    // USB device constants
    pub const DEFAULT_VENDOR_ID: u16 = 0x1234;
    pub const DEFAULT_PRODUCT_ID: u16 = 0x5678;
    pub const DEFAULT_USB_TIMEOUT_MS: u32 = 1000;
    pub const USB_BULK_TRANSFER_SIZE: usize = 64;
    pub const USB_MAX_PACKET_SIZE: usize = 512;

    // USB endpoint constants
    pub const DEFAULT_BULK_IN_ENDPOINT: u8 = 0x81;
    pub const DEFAULT_BULK_OUT_ENDPOINT: u8 = 0x02;
    pub const DEFAULT_INTERRUPT_ENDPOINT: u8 = 0x83;

    // USB power constants
    pub const USB_SELF_POWERED: bool = true;
    pub const USB_MAX_POWER_MA: u16 = 500;
    pub const USB_REMOTE_WAKEUP: bool = false;
}

/// Buffer management constants
pub mod buffers {
    pub const MIN_BUFFER_SIZE: usize = 64;
    pub const MAX_BUFFER_SIZE: usize = 1048576; // 1MB
    pub const OVERFLOW_PROTECTION_THRESHOLD: f32 = 0.9;
    pub const UNDERRUN_THRESHOLD: f32 = 0.1;
    pub const DEFAULT_RING_BUFFER_CAPACITY: usize = 1024;

    // Buffer health monitoring constants
    pub const BUFFER_UTILIZATION_WARNING_THRESHOLD: f32 = 0.8;
    pub const BUFFER_UTILIZATION_CRITICAL_THRESHOLD: f32 = 0.95;
    pub const BUFFER_VARIANCE_THRESHOLD: f32 = 0.3;
    pub const MIN_BUFFER_SYNC_SAMPLES: usize = 10;

    // Memory alignment constants
    pub const CACHE_LINE_SIZE: usize = 64;
    pub const MEMORY_ALIGNMENT: usize = 16;
    pub const LOCK_FREE_PADDING: usize = 128;
}

/// File system paths
pub mod paths {
    pub const SYSTEM_CONFIG_PATH: &str = "/etc/emg/config.toml";
    pub const USER_CONFIG_DIR: &str = ".config/emg";
    pub const LOCAL_CONFIG_FILE: &str = "config.toml";
    pub const DEFAULT_CONFIG_FILE: &str = "config/default.toml";
    pub const SCHEMA_DIR: &str = "schemas";

    // Additional path constants
    pub const LOG_DIR: &str = "logs";
    pub const CACHE_DIR: &str = "cache";
    pub const TEMP_DIR: &str = "tmp";
    pub const BACKUP_DIR: &str = "backups";

    // File extension constants
    pub const CONFIG_EXTENSION: &str = ".toml";
    pub const LOG_EXTENSION: &str = ".log";
    pub const DATA_EXTENSION: &str = ".emg";
    pub const SCHEMA_EXTENSION: &str = ".json";
}

/// Validation constants
pub mod validation {
    // Validation thresholds
    pub const MAX_CONFIG_FILE_SIZE_BYTES: usize = 1_048_576; // 1MB
    pub const MAX_FIELD_NAME_LENGTH: usize = 64;
    pub const MAX_STRING_VALUE_LENGTH: usize = 256;
    pub const MAX_ARRAY_LENGTH: usize = 1000;

    // Validation error constants
    pub const VALIDATION_MAX_ERRORS: usize = 100;
    pub const VALIDATION_TIMEOUT_MS: u32 = 5000;

    // Numeric validation constants
    pub const FLOAT_EPSILON: f32 = 1e-6;
    pub const DOUBLE_EPSILON: f64 = 1e-12;
    pub const MAX_PERCENTAGE: f32 = 100.0;
    pub const MIN_PERCENTAGE: f32 = 0.0;

    // ADDED: Advanced validation constants
    pub const MAX_VALIDATION_RECURSION_DEPTH: usize = 10;
    pub const VALIDATION_CACHE_SIZE: usize = 1000;
    pub const MAX_VALIDATION_ITERATIONS: usize = 100;

    // ADDED: String validation constants
    pub const MIN_DEVICE_ID_LENGTH: usize = 1;
    pub const MAX_DEVICE_ID_LENGTH: usize = 64;
    pub const MIN_PORT_NAME_LENGTH: usize = 1;
    pub const MAX_PORT_NAME_LENGTH: usize = 256;

    // ADDED: Numeric precision constants
    pub const TIMESTAMP_PRECISION_NANOS: u64 = 1000; // 1 microsecond precision
    pub const FREQUENCY_PRECISION_HZ: f32 = 0.1; // 0.1 Hz precision
    pub const AMPLITUDE_PRECISION_UV: f32 = 0.1; // 0.1 µV precision

    // ADDED: Cross-validation constants
    pub const MAX_CROSS_VALIDATION_FIELDS: usize = 10;
    pub const CROSS_VALIDATION_TIMEOUT_MS: u32 = 100;
}

/// Error handling constants
pub mod errors {
    // Error message constants
    pub const MAX_ERROR_MESSAGE_LENGTH: usize = 512;
    pub const MAX_ERROR_CONTEXT_DEPTH: usize = 10;
    pub const ERROR_TRACE_BUFFER_SIZE: usize = 1000;

    // Error retry constants
    pub const DEFAULT_ERROR_RETRY_COUNT: u32 = 3;
    pub const MAX_ERROR_RETRY_COUNT: u32 = 10;
    pub const ERROR_RETRY_BACKOFF_MS: u64 = 100;
    pub const ERROR_RETRY_MAX_DELAY_MS: u64 = 5000;

    // ADDED: Error classification constants
    pub const ERROR_SEVERITY_FATAL: u8 = 4;
    pub const ERROR_SEVERITY_ERROR: u8 = 3;
    pub const ERROR_SEVERITY_WARNING: u8 = 2;
    pub const ERROR_SEVERITY_INFO: u8 = 1;
    pub const ERROR_SEVERITY_DEBUG: u8 = 0;

    // ADDED: Error recovery constants
    pub const MAX_ERROR_RECOVERY_ATTEMPTS: u32 = 5;
    pub const ERROR_RECOVERY_EXPONENTIAL_BASE: f32 = 2.0;
    pub const ERROR_RECOVERY_MAX_DELAY_SECONDS: u64 = 60;

    // ADDED: Error aggregation constants
    pub const ERROR_AGGREGATION_WINDOW_SECONDS: u64 = 60; // 1 minute windows
    pub const MAX_ERRORS_PER_WINDOW: u32 = 100;
    pub const ERROR_RATE_THRESHOLD_PER_SECOND: f32 = 1.0; // 1 error/sec threshold
}

/// Testing constants
pub mod testing {
    // Test configuration constants
    pub const TEST_SAMPLE_RATE_HZ: u32 = 1000;
    pub const TEST_CHANNEL_COUNT: usize = 4;
    pub const TEST_BUFFER_SIZE: usize = 256;
    pub const TEST_TIMEOUT_MS: u32 = 100;

    // Test data constants
    pub const TEST_SIGNAL_AMPLITUDE: f32 = 0.5;
    pub const TEST_SIGNAL_FREQUENCY_HZ: f32 = 50.0;
    pub const TEST_NOISE_LEVEL: f32 = 0.01;
    pub const TEST_SAMPLES_COUNT: usize = 1000;

    // Mock device constants
    pub const MOCK_DEVICE_LATENCY_MS: u32 = 5;
    pub const MOCK_PACKET_SIZE: usize = 64;
    pub const MOCK_ERROR_RATE: f32 = 0.001; // 0.1% error rate
}

/// ADDED: Data integrity constants
pub mod integrity {
    // Data integrity constants
    pub const CORRUPTION_DETECTION_WINDOW_SIZE: usize = 100;
    pub const MAX_CORRUPTION_PATTERNS: usize = 10;
    pub const HASH_CACHE_SIZE: usize = 1000;

    // Checksum algorithm constants
    pub const CHECKSUM_POLYNOMIAL_CRC32: u32 = 0xEDB88320; // IEEE 802.3 CRC-32
    pub const CHECKSUM_INIT_VALUE_CRC32: u32 = 0xFFFFFFFF;
    pub const CHECKSUM_XOR_OUTPUT_CRC32: u32 = 0xFFFFFFFF;

    // Integrity check intervals
    pub const INTEGRITY_CHECK_INTERVAL_PACKETS: u32 = 100;
    pub const INTEGRITY_VERIFY_TIMEOUT_MS: u32 = 10;
    pub const MAX_INTEGRITY_ERRORS_PER_SECOND: u32 = 10;
}

/// ADDED: Conversion constants
pub mod conversion {
    // ADC conversion constants
    pub const ADC_VOLTAGE_REFERENCE_DEFAULT: f32 = 3.3; // 3.3V reference
    pub const ADC_GAIN_DEFAULT: f32 = 1000.0; // 1000x amplifier gain
    pub const ADC_OFFSET_DEFAULT: f32 = 0.0; // No DC offset

    // Unit conversion constants
    pub const VOLTS_TO_MICROVOLTS: f32 = 1_000_000.0;
    pub const MICROVOLTS_TO_VOLTS: f32 = 1.0 / 1_000_000.0;
    pub const MILLIVOLTS_TO_VOLTS: f32 = 1.0 / 1_000.0;
    pub const VOLTS_TO_MILLIVOLTS: f32 = 1_000.0;

    // Sample format conversion constants
    pub const FLOAT32_EPSILON: f32 = f32::EPSILON;
    pub const FLOAT64_EPSILON: f64 = f64::EPSILON;
    pub const SIGNED_TO_UNSIGNED_OFFSET_8BIT: u8 = 128;
    pub const SIGNED_TO_UNSIGNED_OFFSET_16BIT: u16 = 32768;
    pub const SIGNED_TO_UNSIGNED_OFFSET_24BIT: u32 = 8388608;

    // Conversion precision constants
    pub const MAX_CONVERSION_ERROR_PERCENT: f32 = 0.01; // 1% max error
    pub const CONVERSION_PRECISION_BITS: u8 = 16; // 16-bit precision for intermediate calculations
}

/// ADDED: Bounds checking constants
pub mod bounds {
    // Bounds checking constants
    pub const MAX_ARRAY_INDEX_CHECK_DEPTH: usize = 100;
    pub const BOUNDS_CHECK_CACHE_SIZE: usize = 1000;
    pub const MAX_SLICE_OPERATIONS_PER_CHECK: usize = 10;

    // Memory safety constants
    pub const STACK_OVERFLOW_DETECTION_SIZE: usize = 1024;
    pub const HEAP_OVERFLOW_DETECTION_SIZE: usize = 4096;
    pub const POINTER_ALIGNMENT_BYTES: usize = 8;

    // Ring buffer bounds constants
    pub const RING_BUFFER_MIN_CAPACITY: usize = 2; // Must be power of 2
    pub const RING_BUFFER_MAX_CAPACITY: usize = 1048576; // 1M elements max
    pub const RING_BUFFER_GUARD_SIZE: usize = 64; // Guard band size
}

/// ADDED: Time-related constants
pub mod time {
    // Time-related constants
    pub const UNIX_EPOCH_YEAR: u32 = 1970;
    pub const SECONDS_PER_MINUTE: u64 = 60;
    pub const SECONDS_PER_HOUR: u64 = 3600;
    pub const SECONDS_PER_DAY: u64 = 86400;

    // Timestamp validation constants
    pub const MIN_VALID_TIMESTAMP: u64 = 1_000_000_000_000_000_000; // Year 2001
    pub const MAX_VALID_TIMESTAMP: u64 = 4_000_000_000_000_000_000; // Year 2096
    pub const TIMESTAMP_FUTURE_TOLERANCE_SECONDS: u64 = 3600; // 1 hour in future

    // Clock synchronization constants
    pub const CLOCK_SYNC_INTERVAL_SECONDS: u64 = 300; // 5 minutes
    pub const CLOCK_DRIFT_CORRECTION_THRESHOLD_NANOS: u64 = 1_000_000; // 1ms
    pub const MAX_CLOCK_ADJUSTMENT_NANOS: u64 = 100_000_000; // 100ms max adjustment

    // High-resolution timing constants
    pub const PRECISION_SLEEP_THRESHOLD_NANOS: u64 = 1_000_000; // 1ms - use spin wait below this
    pub const SPIN_WAIT_MAX_ITERATIONS: u64 = 1_000_000; // Prevent infinite spin
    pub const TIMER_RESOLUTION_NANOS: u64 = 100; // 100ns timer resolution
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_consistency() {
        // Test that min/max constants are consistent
        assert!(signal::MIN_SAMPLING_RATE_HZ < signal::MAX_SAMPLING_RATE_HZ);
        assert!(signal::MIN_CHANNEL_COUNT < signal::MAX_CHANNEL_COUNT);
        assert!(performance::MIN_LATENCY_TARGET_MS < performance::MAX_LATENCY_TARGET_MS);

        // Test that buffer sizes are reasonable
        assert!(buffers::MIN_BUFFER_SIZE < buffers::MAX_BUFFER_SIZE);
        assert!(buffers::DEFAULT_RING_BUFFER_CAPACITY >= buffers::MIN_BUFFER_SIZE);
        assert!(buffers::DEFAULT_RING_BUFFER_CAPACITY <= buffers::MAX_BUFFER_SIZE);

        // Test quality thresholds are in valid ranges
        assert!(quality::MIN_SNR_THRESHOLD_DB <= quality::MAX_SNR_THRESHOLD_DB);
        assert!(quality::DEFAULT_SNR_THRESHOLD_DB >= quality::MIN_SNR_THRESHOLD_DB);
        assert!(quality::DEFAULT_SNR_THRESHOLD_DB <= quality::MAX_SNR_THRESHOLD_DB);

        // Test new signal amplitude ranges
        assert!(signal::MIN_SIGNAL_AMPLITUDE < signal::MAX_SIGNAL_AMPLITUDE);
        assert!(signal::DEFAULT_SIGNAL_AMPLITUDE >= signal::MIN_SIGNAL_AMPLITUDE);
        assert!(signal::DEFAULT_SIGNAL_AMPLITUDE <= signal::MAX_SIGNAL_AMPLITUDE);
    }

    #[test]
    fn test_serial_constants_validity() {
        // Test baud rate ranges
        assert!(serial::MIN_BAUD_RATE < serial::MAX_BAUD_RATE);
        assert!(serial::DEFAULT_BAUD_RATE >= serial::MIN_BAUD_RATE);
        assert!(serial::DEFAULT_BAUD_RATE <= serial::MAX_BAUD_RATE);

        // Test data bits
        assert!(serial::MIN_DATA_BITS <= serial::DEFAULT_DATA_BITS);
        assert!(serial::DEFAULT_DATA_BITS <= serial::MAX_DATA_BITS);

        // Test timeout ranges
        assert!(serial::MIN_TIMEOUT_MS < serial::MAX_TIMEOUT_MS);
        assert!(serial::DEFAULT_TIMEOUT_MS >= serial::MIN_TIMEOUT_MS);
        assert!(serial::DEFAULT_TIMEOUT_MS <= serial::MAX_TIMEOUT_MS);

        // Test new channel mapping constants
        assert!(serial::MIN_CHANNELS_PER_PACKET <= serial::MAX_CHANNELS_PER_PACKET);
        assert!(!serial::DEFAULT_CHANNEL_MAPPING.is_empty());
    }

    #[test]
    fn test_simulation_constants_validity() {
        // Test noise level ranges
        assert!(simulation::MIN_NOISE_LEVEL <= simulation::MAX_NOISE_LEVEL);
        assert!(simulation::DEFAULT_NOISE_LEVEL >= simulation::MIN_NOISE_LEVEL);
        assert!(simulation::DEFAULT_NOISE_LEVEL <= simulation::MAX_NOISE_LEVEL);

        // Test artifact probability ranges
        assert!(simulation::MIN_ARTIFACT_PROBABILITY <= simulation::MAX_ARTIFACT_PROBABILITY);
        assert!(simulation::DEFAULT_ARTIFACT_PROBABILITY >= simulation::MIN_ARTIFACT_PROBABILITY);
        assert!(simulation::DEFAULT_ARTIFACT_PROBABILITY <= simulation::MAX_ARTIFACT_PROBABILITY);

        // Test gesture duration ranges
        assert!(simulation::MIN_GESTURE_DURATION_MS < simulation::MAX_GESTURE_DURATION_MS);
        assert!(simulation::DEFAULT_GESTURE_DURATION_MS >= simulation::MIN_GESTURE_DURATION_MS);
        assert!(simulation::DEFAULT_GESTURE_DURATION_MS <= simulation::MAX_GESTURE_DURATION_MS);
    }

    #[test]
    fn test_mathematical_constants() {
        // Test that floating point constants are finite
        assert!(quality::DEFAULT_SNR_THRESHOLD_DB.is_finite());
        assert!(filters::DEFAULT_HIGHPASS_CUTOFF_HZ.is_finite());
        assert!(filters::DEFAULT_LOWPASS_CUTOFF_HZ.is_finite());

        // Test that percentages are in valid range
        assert!(windowing::DEFAULT_OVERLAP_PERCENT >= windowing::MIN_OVERLAP_PERCENT);
        assert!(windowing::DEFAULT_OVERLAP_PERCENT <= windowing::MAX_OVERLAP_PERCENT);

        // Test signal normalization ranges
        assert!(signal::SIGNAL_CLAMP_MIN < signal::SIGNAL_CLAMP_MAX);
        assert!(signal::SIGNAL_SATURATION_THRESHOLD > 0.0);
        assert!(signal::SIGNAL_SATURATION_THRESHOLD <= 1.0);
    }

    #[test]
    fn test_new_validation_constants() {
        // Test validation constants
        assert!(validation::MIN_DEVICE_ID_LENGTH <= validation::MAX_DEVICE_ID_LENGTH);
        assert!(validation::MIN_PORT_NAME_LENGTH <= validation::MAX_PORT_NAME_LENGTH);
        assert!(validation::MAX_VALIDATION_RECURSION_DEPTH > 0);
        assert!(validation::VALIDATION_CACHE_SIZE > 0);
    }

    #[test]
    fn test_new_time_constants() {
        // Test time constants
        assert!(time::MIN_VALID_TIMESTAMP < time::MAX_VALID_TIMESTAMP);
        assert!(time::PRECISION_SLEEP_THRESHOLD_NANOS > 0);
        assert!(time::TIMER_RESOLUTION_NANOS > 0);
        assert!(time::SECONDS_PER_MINUTE == 60);
        assert!(time::SECONDS_PER_HOUR == 3600);
    }

    #[test]
    fn test_new_bounds_constants() {
        // Test bounds constants
        assert!(bounds::RING_BUFFER_MIN_CAPACITY.is_power_of_two());
        assert!(bounds::RING_BUFFER_MAX_CAPACITY.is_power_of_two());
        assert!(bounds::RING_BUFFER_MIN_CAPACITY < bounds::RING_BUFFER_MAX_CAPACITY);
        assert!(bounds::POINTER_ALIGNMENT_BYTES.is_power_of_two());
    }

    #[test]
    fn test_new_conversion_constants() {
        // Test conversion constants
        assert!(conversion::ADC_VOLTAGE_REFERENCE_DEFAULT > 0.0);
        assert!(conversion::ADC_GAIN_DEFAULT > 0.0);
        assert!(conversion::VOLTS_TO_MICROVOLTS == 1_000_000.0);
        assert!(conversion::MAX_CONVERSION_ERROR_PERCENT > 0.0);
        assert!(conversion::MAX_CONVERSION_ERROR_PERCENT < 1.0);
    }

    #[test]
    fn test_new_quality_constants() {
        // Test quality threshold ordering
        assert!(
            quality::SIGNAL_QUALITY_UNACCEPTABLE_THRESHOLD < quality::SIGNAL_QUALITY_POOR_THRESHOLD
        );
        assert!(quality::SIGNAL_QUALITY_POOR_THRESHOLD < quality::SIGNAL_QUALITY_GOOD_THRESHOLD);
        assert!(
            quality::SIGNAL_QUALITY_GOOD_THRESHOLD < quality::SIGNAL_QUALITY_EXCELLENT_THRESHOLD
        );

        // Test quality update intervals
        assert!(quality::QUALITY_UPDATE_INTERVAL_MS > 0);
        assert!(quality::QUALITY_ASSESSMENT_WINDOW_SIZE > 0);
        assert!(quality::QUALITY_HISTORY_LENGTH > 0);
    }
}
