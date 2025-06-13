
// ================================================================================
// Configuration constants
// File: src/config/processing.rs (NEW)
// ================================================================================

//! Processing module configuration constants

// Feature extraction constants
pub const TIME_DOMAIN_FEATURE_COUNT: usize = 8;
pub const FREQUENCY_DOMAIN_BASE_FEATURES: usize = 5;
pub const DEFAULT_FFT_SIZE: usize = 256;
pub const DEFAULT_FREQUENCY_BANDS: &[(f32, f32)] = &[
    (20.0, 50.0),    // Low frequency muscle activation
    (50.0, 150.0),   // Primary EMG frequency range
    (150.0, 300.0),  // High frequency components
    (300.0, 500.0),  // Very high frequency (artifacts/noise)
];

// Signal processing constants
pub const DEFAULT_FILTER_ORDER: usize = 4;
pub const BUTTERWORTH_Q_FACTOR: f32 = 0.707; // 1/sqrt(2)
pub const NOTCH_FILTER_Q: f32 = 10.0;
pub const POWERLINE_FREQUENCIES: &[f32] = &[50.0, 60.0]; // EU and US powerline

// Window function constants
pub const DEFAULT_WINDOW_OVERLAP: f32 = 0.5; // 50% overlap
pub const MIN_WINDOW_SIZE: usize = 64;
pub const MAX_WINDOW_SIZE: usize = 2048;

// Performance targets
pub const MAX_PROCESSING_LATENCY_MS: f32 = 2.0;
pub const TARGET_FEATURE_EXTRACTION_TIME_US: f32 = 500.0;
