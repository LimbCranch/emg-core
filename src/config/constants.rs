// src/config/constants.rs
//! System-wide configuration constants
//! FIXED: Added all missing constants to eliminate magic numbers

/// Signal processing constants
pub mod signal {
    pub const DEFAULT_SAMPLING_RATE_HZ: u32 = 2000;
    pub const DEFAULT_CHANNEL_COUNT: usize = 8;
    pub const DEFAULT_BUFFER_SIZE_SAMPLES: usize = 4096;
    pub const MIN_SAMPLING_RATE_HZ: u32 = 500;
    pub const MAX_SAMPLING_RATE_HZ: u32 = 10000;
    pub const MAX_CHANNEL_COUNT: usize = 32;
    pub const MIN_CHANNEL_COUNT: usize = 1;

    // FIXED: Added ADC resolution constants
    pub const DEFAULT_ADC_RESOLUTION_BITS: u8 = 24;
    pub const MIN_ADC_RESOLUTION_BITS: u8 = 8;
    pub const MAX_ADC_RESOLUTION_BITS: u8 = 32;

    // FIXED: Added signal range constants
    pub const DEFAULT_INPUT_RANGE_MV: (f32, f32) = (-2500.0, 2500.0);
    pub const SIGNAL_SATURATION_THRESHOLD: f32 = 0.95;
    pub const SIGNAL_CLAMP_MIN: f32 = -1.0;
    pub const SIGNAL_CLAMP_MAX: f32 = 1.0;
}

/// Performance and timing constants
pub mod performance {
    pub const DEFAULT_LATENCY_TARGET_MS: u32 = 20;
    pub const MAX_LATENCY_TARGET_MS: u32 = 100;
    pub const MIN_LATENCY_TARGET_MS: u32 = 1;
    pub const WATCHDOG_TIMEOUT_MS: u32 = 100;
    pub const PROCESSING_TIMEOUT_MS: u32 = 50;

    // FIXED: Added timing precision constants
    pub const NANOSECONDS_PER_SECOND: u64 = 1_000_000_000;
    pub const MICROSECONDS_PER_SECOND: u64 = 1_000_000;
    pub const MILLISECONDS_PER_SECOND: u64 = 1_000;

    // FIXED: Added sample period calculation helpers
    pub const MIN_SAMPLE_PERIOD_NANOS: u64 = NANOSECONDS_PER_SECOND / MAX_SAMPLING_RATE_HZ as u64;
    pub const MAX_SAMPLE_PERIOD_NANOS: u64 = NANOSECONDS_PER_SECOND / MIN_SAMPLING_RATE_HZ as u64;

    use super::signal::{MAX_SAMPLING_RATE_HZ, MIN_SAMPLING_RATE_HZ};
}

/// Hardware abstraction layer constants
pub mod hal {
    pub const DEFAULT_CONNECTION_TIMEOUT_MS: u32 = 5000;
    pub const DEFAULT_RETRY_ATTEMPTS: u32 = 3;
    pub const MAX_RETRY_ATTEMPTS: u32 = 10;
    pub const DEVICE_SCAN_TIMEOUT_MS: u32 = 10000;

    // FIXED: Added retry timing constants
    pub const RETRY_BASE_DELAY_MS: u64 = 100;
    pub const RETRY_MAX_DELAY_MS: u64 = 5000;
    pub const CONNECTION_CHECK_INTERVAL_MS: u64 = 1000;

    // FIXED: Added buffer size constants
    pub const DEFAULT_READ_BUFFER_SIZE: usize = 4096;
    pub const MIN_READ_BUFFER_SIZE: usize = 64;
    pub const MAX_READ_BUFFER_SIZE: usize = 1_048_576; // 1MB
    pub const DEFAULT_MAX_PACKET_SIZE: usize = 1024;
    pub const MIN_PACKET_SIZE: usize = 16;
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

    // FIXED: Added filter response constants
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

    // FIXED: Added quality assessment constants
    pub const SNR_EXCELLENT_THRESHOLD_DB: f32 = 40.0;
    pub const SNR_GOOD_THRESHOLD_DB: f32 = 25.0;
    pub const SNR_POOR_THRESHOLD_DB: f32 = 15.0;
    pub const IMPEDANCE_EXCELLENT_THRESHOLD_KOHM: f32 = 10.0;
    pub const IMPEDANCE_GOOD_THRESHOLD_KOHM: f32 = 25.0;
    pub const IMPEDANCE_POOR_THRESHOLD_KOHM: f32 = 50.0;

    // FIXED: Added motion artifact detection constants
    pub const MOTION_ARTIFACT_VARIANCE_THRESHOLD: f32 = 0.25;
    pub const MOTION_ARTIFACT_PEAK_THRESHOLD: f32 = 0.8;
    pub const MIN_ARTIFACT_DURATION_MS: u64 = 10;
    pub const MAX_ARTIFACT_DURATION_MS: u64 = 500;
}

/// Windowing constants
pub mod windowing {
    pub const DEFAULT_WINDOW_SIZE_SAMPLES: usize = 256;
    pub const DEFAULT_OVERLAP_PERCENT: f32 = 50.0;
    pub const MIN_OVERLAP_PERCENT: f32 = 0.0;
    pub const MAX_OVERLAP_PERCENT: f32 = 99.0;
    pub const MIN_WINDOW_SIZE: usize = 16;
    pub const MAX_WINDOW_SIZE: usize = 8192;

    // FIXED: Added window function constants
    pub const HANN_WINDOW_ALPHA: f32 = 0.5;
    pub const HAMMING_WINDOW_ALPHA: f32 = 0.54;
    pub const HAMMING_WINDOW_BETA: f32 = 0.46;
    pub const BLACKMAN_WINDOW_A0: f32 = 0.42;
    pub const BLACKMAN_WINDOW_A1: f32 = 0.5;
    pub const BLACKMAN_WINDOW_A2: f32 = 0.08;
}

/// Simulator constants
pub mod simulator {
    pub const DEFAULT_NOISE_LEVEL: f32 = 0.05;
    pub const DEFAULT_ARTIFACT_PROBABILITY: f32 = 0.02;
    pub const DEFAULT_GESTURE_DURATION_MS: u32 = 1000;
    pub const MIN_NOISE_LEVEL: f32 = 0.0;
    pub const MAX_NOISE_LEVEL: f32 = 1.0;
    pub const MIN_ARTIFACT_PROBABILITY: f32 = 0.0;
    pub const MAX_ARTIFACT_PROBABILITY: f32 = 1.0;

    // FIXED: Added simulator timing constants
    pub const MIN_GESTURE_DURATION_MS: u32 = 100;
    pub const MAX_GESTURE_DURATION_MS: u32 = 10000;
    pub const SIMULATOR_TIMESTAMP_PRECISION_NANOS: u64 = 1000; // 1 microsecond

    // FIXED: Added muscle simulation constants
    pub const DEFAULT_MUSCLE_ACTIVATION_THRESHOLD: f32 = 0.1;
    pub const MAX_MUSCLE_ACTIVATION_LEVEL: f32 = 1.0;
    pub const MIN_MUSCLE_ACTIVATION_LEVEL: f32 = 0.0;
    pub const DEFAULT_MUSCLE_FREQ_LOW_HZ: f32 = 20.0;
    pub const DEFAULT_MUSCLE_FREQ_HIGH_HZ: f32 = 150.0;

    // FIXED: Added powerline noise constants
    pub const POWERLINE_NOISE_AMPLITUDE: f32 = 0.02; // 2% of full scale
    pub const POWERLINE_PHASE_INCREMENT: f32 = 2.0 * std::f32::consts::PI;

    // FIXED: Added artifact simulation constants
    pub const MIN_ARTIFACT_INTERVAL_NANOS: u64 = 50_000_000; // 50ms
    pub const MAX_ARTIFACT_AMPLITUDE: f32 = 0.5;
    pub const THERMAL_NOISE_FACTOR: f32 = 1.0;
    pub const CHANNEL_ARTIFACT_SCALING: f32 = 1.0;
}

/// Serial communication constants
pub mod serial {
    // FIXED: Added serial protocol constants
    pub const DEFAULT_BAUD_RATE: u32 = 115200;
    pub const MIN_BAUD_RATE: u32 = 9600;
    pub const MAX_BAUD_RATE: u32 = 4_000_000;

    pub const DEFAULT_DATA_BITS: u8 = 8;
    pub const MIN_DATA_BITS: u8 = 5;
    pub const MAX_DATA_BITS: u8 = 8;

    pub const DEFAULT_STOP_BITS: u8 = 1;
    pub const MIN_STOP_BITS: u8 = 1;
    pub const MAX_STOP_BITS: u8 = 2;

    // FIXED: Added packet structure constants
    pub const DEFAULT_HEADER_BYTES: [u8; 4] = [0xAA, 0x55, 0xA5, 0x5A];
    pub const DEFAULT_FOOTER_BYTES: [u8; 2] = [0xDE, 0xAD];
    pub const DEFAULT_BYTES_PER_CHANNEL: usize = 4;

    // FIXED: Added timeout constants
    pub const DEFAULT_TIMEOUT_MS: u32 = 1000;
    pub const MIN_TIMEOUT_MS: u32 = 10;
    pub const MAX_TIMEOUT_MS: u32 = 60000;

    // FIXED: Added packet validation constants
    pub const MIN_PACKET_HEADER_SIZE: usize = 2;
    pub const MAX_PACKET_HEADER_SIZE: usize = 8;
    pub const CHECKSUM_SIZE_BYTES: usize = 1;
    pub const CRC16_SIZE_BYTES: usize = 2;

    // FIXED: Added CRC constants
    pub const CRC8_POLYNOMIAL: u8 = 0x07; // CRC-8-CCITT
    pub const CRC8_INIT_VALUE: u8 = 0xFF;
    pub const CRC16_POLYNOMIAL: u16 = 0x1021; // CRC-16-CCITT
    pub const CRC16_INIT_VALUE: u16 = 0xFFFF;

    // FIXED: Added channel conversion constants
    pub const ADC_8BIT_SCALE: f32 = 128.0;
    pub const ADC_16BIT_SCALE: f32 = 32768.0;
    pub const ADC_24BIT_SCALE: f32 = 8388608.0;
    pub const ADC_24BIT_SIGN_MASK: u8 = 0x80;
    pub const ADC_24BIT_SIGN_EXTEND: u8 = 0xFF;
}

/// USB communication constants
pub mod usb {
    // FIXED: Added USB device constants
    pub const DEFAULT_VENDOR_ID: u16 = 0x1234;
    pub const DEFAULT_PRODUCT_ID: u16 = 0x5678;
    pub const DEFAULT_USB_TIMEOUT_MS: u32 = 1000;
    pub const USB_BULK_TRANSFER_SIZE: usize = 64;
    pub const USB_MAX_PACKET_SIZE: usize = 512;

    // FIXED: Added USB endpoint constants
    pub const DEFAULT_BULK_IN_ENDPOINT: u8 = 0x81;
    pub const DEFAULT_BULK_OUT_ENDPOINT: u8 = 0x02;
    pub const DEFAULT_INTERRUPT_ENDPOINT: u8 = 0x83;

    // FIXED: Added USB power constants
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

    // FIXED: Added buffer health monitoring constants
    pub const BUFFER_UTILIZATION_WARNING_THRESHOLD: f32 = 0.8;
    pub const BUFFER_UTILIZATION_CRITICAL_THRESHOLD: f32 = 0.95;
    pub const BUFFER_VARIANCE_THRESHOLD: f32 = 0.3;
    pub const MIN_BUFFER_SYNC_SAMPLES: usize = 10;

    // FIXED: Added memory alignment constants
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

    // FIXED: Added additional path constants
    pub const LOG_DIR: &str = "logs";
    pub const CACHE_DIR: &str = "cache";
    pub const TEMP_DIR: &str = "tmp";
    pub const BACKUP_DIR: &str = "backups";

    // FIXED: Added file extension constants
    pub const CONFIG_EXTENSION: &str = ".toml";
    pub const LOG_EXTENSION: &str = ".log";
    pub const DATA_EXTENSION: &str = ".emg";
    pub const SCHEMA_EXTENSION: &str = ".json";
}

/// Validation constants
pub mod validation {
    // FIXED: Added validation thresholds
    pub const MAX_CONFIG_FILE_SIZE_BYTES: usize = 1_048_576; // 1MB
    pub const MAX_FIELD_NAME_LENGTH: usize = 64;
    pub const MAX_STRING_VALUE_LENGTH: usize = 256;
    pub const MAX_ARRAY_LENGTH: usize = 1000;

    // FIXED: Added validation error constants
    pub const VALIDATION_MAX_ERRORS: usize = 100;
    pub const VALIDATION_TIMEOUT_MS: u32 = 5000;

    // FIXED: Added numeric validation constants
    pub const FLOAT_EPSILON: f32 = 1e-6;
    pub const DOUBLE_EPSILON: f64 = 1e-12;
    pub const MAX_PERCENTAGE: f32 = 100.0;
    pub const MIN_PERCENTAGE: f32 = 0.0;
}

/// Error handling constants
pub mod errors {
    // FIXED: Added error message constants
    pub const MAX_ERROR_MESSAGE_LENGTH: usize = 512;
    pub const MAX_ERROR_CONTEXT_DEPTH: usize = 10;
    pub const ERROR_TRACE_BUFFER_SIZE: usize = 1000;

    // FIXED: Added error retry constants
    pub const DEFAULT_ERROR_RETRY_COUNT: u32 = 3;
    pub const MAX_ERROR_RETRY_COUNT: u32 = 10;
    pub const ERROR_RETRY_BACKOFF_MS: u64 = 100;
    pub const ERROR_RETRY_MAX_DELAY_MS: u64 = 5000;
}

/// Testing constants
pub mod testing {
    // FIXED: Added test configuration constants
    pub const TEST_SAMPLE_RATE_HZ: u32 = 1000;
    pub const TEST_CHANNEL_COUNT: usize = 4;
    pub const TEST_BUFFER_SIZE: usize = 256;
    pub const TEST_TIMEOUT_MS: u32 = 100;

    // FIXED: Added test data constants
    pub const TEST_SIGNAL_AMPLITUDE: f32 = 0.5;
    pub const TEST_SIGNAL_FREQUENCY_HZ: f32 = 50.0;
    pub const TEST_NOISE_LEVEL: f32 = 0.01;
    pub const TEST_SAMPLES_COUNT: usize = 1000;

    // FIXED: Added mock device constants
    pub const MOCK_DEVICE_LATENCY_MS: u32 = 5;
    pub const MOCK_PACKET_SIZE: usize = 64;
    pub const MOCK_ERROR_RATE: f32 = 0.001; // 0.1% error rate
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
    }

    #[test]
    fn test_simulator_constants_validity() {
        // Test noise level ranges
        assert!(simulator::MIN_NOISE_LEVEL <= simulator::MAX_NOISE_LEVEL);
        assert!(simulator::DEFAULT_NOISE_LEVEL >= simulator::MIN_NOISE_LEVEL);
        assert!(simulator::DEFAULT_NOISE_LEVEL <= simulator::MAX_NOISE_LEVEL);

        // Test artifact probability ranges
        assert!(simulator::MIN_ARTIFACT_PROBABILITY <= simulator::MAX_ARTIFACT_PROBABILITY);
        assert!(simulator::DEFAULT_ARTIFACT_PROBABILITY >= simulator::MIN_ARTIFACT_PROBABILITY);
        assert!(simulator::DEFAULT_ARTIFACT_PROBABILITY <= simulator::MAX_ARTIFACT_PROBABILITY);

        // Test gesture duration ranges
        assert!(simulator::MIN_GESTURE_DURATION_MS < simulator::MAX_GESTURE_DURATION_MS);
        assert!(simulator::DEFAULT_GESTURE_DURATION_MS >= simulator::MIN_GESTURE_DURATION_MS);
        assert!(simulator::DEFAULT_GESTURE_DURATION_MS <= simulator::MAX_GESTURE_DURATION_MS);
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
}