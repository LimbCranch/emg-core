// src/config/constants.rs
//! System-wide configuration constants

/// Signal processing constants
pub mod signal {
    pub const DEFAULT_SAMPLING_RATE_HZ: u32 = 2000;
    pub const DEFAULT_CHANNEL_COUNT: usize = 8;
    pub const DEFAULT_BUFFER_SIZE_SAMPLES: usize = 4096;
    pub const MIN_SAMPLING_RATE_HZ: u32 = 500;
    pub const MAX_SAMPLING_RATE_HZ: u32 = 10000;
    pub const MAX_CHANNEL_COUNT: usize = 32;
}

/// Performance and timing constants
pub mod performance {
    pub const DEFAULT_LATENCY_TARGET_MS: u32 = 20;
    pub const MAX_LATENCY_TARGET_MS: u32 = 100;
    pub const MIN_LATENCY_TARGET_MS: u32 = 1;
    pub const WATCHDOG_TIMEOUT_MS: u32 = 100;
    pub const PROCESSING_TIMEOUT_MS: u32 = 50;
}

/// Hardware abstraction layer constants
pub mod hal {
    pub const DEFAULT_CONNECTION_TIMEOUT_MS: u32 = 5000;
    pub const DEFAULT_RETRY_ATTEMPTS: u32 = 3;
    pub const MAX_RETRY_ATTEMPTS: u32 = 10;
    pub const DEVICE_SCAN_TIMEOUT_MS: u32 = 10000;
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
}

/// Quality monitoring constants
pub mod quality {
    pub const DEFAULT_SNR_THRESHOLD_DB: f32 = 20.0;
    pub const MIN_SNR_THRESHOLD_DB: f32 = 0.0;
    pub const MAX_SNR_THRESHOLD_DB: f32 = 60.0;
    pub const DEFAULT_CONTACT_IMPEDANCE_MAX_KOHM: f32 = 50.0;
    pub const DEFAULT_SATURATION_THRESHOLD: f32 = 0.95;
    pub const ARTIFACT_DETECTION_THRESHOLD: f32 = 2.0;
}

/// Windowing constants
pub mod windowing {
    pub const DEFAULT_WINDOW_SIZE_SAMPLES: usize = 256;
    pub const DEFAULT_OVERLAP_PERCENT: f32 = 50.0;
    pub const MIN_OVERLAP_PERCENT: f32 = 0.0;
    pub const MAX_OVERLAP_PERCENT: f32 = 99.0;
    pub const MIN_WINDOW_SIZE: usize = 16;
    pub const MAX_WINDOW_SIZE: usize = 8192;
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
}

/// Buffer management constants
pub mod buffers {
    pub const MIN_BUFFER_SIZE: usize = 64;
    pub const MAX_BUFFER_SIZE: usize = 1048576; // 1MB
    pub const OVERFLOW_PROTECTION_THRESHOLD: f32 = 0.9;
    pub const UNDERRUN_THRESHOLD: f32 = 0.1;
    pub const DEFAULT_RING_BUFFER_CAPACITY: usize = 1024;
}

/// File system paths
pub mod paths {
    pub const SYSTEM_CONFIG_PATH: &str = "/etc/emg/config.toml";
    pub const USER_CONFIG_DIR: &str = ".config/emg";
    pub const LOCAL_CONFIG_FILE: &str = "config.toml";
    pub const DEFAULT_CONFIG_FILE: &str = "config/default.toml";
    pub const SCHEMA_DIR: &str = "schemas";
}