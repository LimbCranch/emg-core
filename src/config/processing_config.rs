// src/config/processing_config.rs
//! Signal processing configuration structures

use serde::{Deserialize, Serialize};

/// Complete processing pipeline configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ProcessingConfig {
    pub filter_bank: FilterBankConfig,
    pub quality_monitoring: QualityConfig,
    pub windowing: WindowingConfig,
}

/// Filter bank configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FilterBankConfig {
    pub highpass_cutoff_hz: f32,
    pub lowpass_cutoff_hz: f32,
    pub filter_order: usize,
    pub filter_type: FilterType,
    pub notch_filters: NotchFilterConfig,
}

/// Notch filter configuration for powerline interference
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct NotchFilterConfig {
    pub frequencies_hz: Vec<f32>,
    pub bandwidth_hz: f32,
}

/// Signal quality monitoring configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct QualityConfig {
    pub snr_threshold_db: f32,
    pub artifact_detection_enabled: bool,
    pub contact_impedance_max_kohm: f32,
    pub saturation_threshold: f32,
}

/// Windowing configuration for feature extraction
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct WindowingConfig {
    pub window_size_samples: usize,
    pub overlap_percent: f32,
    pub window_type: WindowType,
}

/// Filter type enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FilterType {
    Butterworth,
    Chebyshev1,
    Chebyshev2,
    Elliptic,
}

/// Window function types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WindowType {
    Rectangular,
    Hamming,
    Hanning,
    Blackman,
    Kaiser,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            filter_bank: FilterBankConfig::default(),
            quality_monitoring: QualityConfig::default(),
            windowing: WindowingConfig::default(),
        }
    }
}

impl Default for FilterBankConfig {
    fn default() -> Self {
        Self {
            highpass_cutoff_hz: 20.0,
            lowpass_cutoff_hz: 500.0,
            filter_order: 4,
            filter_type: FilterType::Butterworth,
            notch_filters: NotchFilterConfig::default(),
        }
    }
}

impl Default for NotchFilterConfig {
    fn default() -> Self {
        Self {
            frequencies_hz: vec![50.0, 60.0], // Power line interference
            bandwidth_hz: 2.0,
        }
    }
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            snr_threshold_db: 20.0,
            artifact_detection_enabled: true,
            contact_impedance_max_kohm: 50.0,
            saturation_threshold: 0.95,
        }
    }
}

impl Default for WindowingConfig {
    fn default() -> Self {
        Self {
            window_size_samples: 256,
            overlap_percent: 50.0,
            window_type: WindowType::Hamming,
        }
    }
}

/// Validate processing configuration
pub fn validate_processing_config(config: &ProcessingConfig) -> Result<(), String> {
    // Validate filter bank
    if config.filter_bank.highpass_cutoff_hz <= 0.0 {
        return Err("Highpass cutoff frequency must be positive".to_string());
    }
    if config.filter_bank.lowpass_cutoff_hz <= config.filter_bank.highpass_cutoff_hz {
        return Err("Lowpass cutoff must be higher than highpass cutoff".to_string());
    }
    if config.filter_bank.filter_order == 0 {
        return Err("Filter order must be greater than 0".to_string());
    }

    // Validate notch filters_v1
    for &freq in &config.filter_bank.notch_filters.frequencies_hz {
        if freq <= 0.0 {
            return Err("Notch filter frequencies must be positive".to_string());
        }
    }
    if config.filter_bank.notch_filters.bandwidth_hz <= 0.0 {
        return Err("Notch filter bandwidth must be positive".to_string());
    }

    // Validate quality monitoring
    if config.quality_monitoring.snr_threshold_db < 0.0 {
        return Err("SNR threshold cannot be negative".to_string());
    }
    if config.quality_monitoring.contact_impedance_max_kohm <= 0.0 {
        return Err("Maximum contact impedance must be positive".to_string());
    }
    if config.quality_monitoring.saturation_threshold <= 0.0 ||
        config.quality_monitoring.saturation_threshold > 1.0 {
        return Err("Saturation threshold must be between 0 and 1".to_string());
    }

    // Validate windowing
    if config.windowing.window_size_samples == 0 {
        return Err("Window size must be greater than 0".to_string());
    }
    if config.windowing.overlap_percent < 0.0 || config.windowing.overlap_percent >= 100.0 {
        return Err("Window overlap must be between 0 and 100 percent".to_string());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = ProcessingConfig::default();
        assert!(validate_processing_config(&config).is_ok());
    }

    #[test]
    fn test_invalid_filter_config() {
        let mut config = ProcessingConfig::default();
        config.filter_bank.highpass_cutoff_hz = 0.0;
        assert!(validate_processing_config(&config).is_err());

        config.filter_bank.highpass_cutoff_hz = 100.0;
        config.filter_bank.lowpass_cutoff_hz = 50.0;
        assert!(validate_processing_config(&config).is_err());
    }

    #[test]
    fn test_invalid_quality_config() {
        let mut config = ProcessingConfig::default();
        config.quality_monitoring.saturation_threshold = 1.5;
        assert!(validate_processing_config(&config).is_err());
    }

    #[test]
    fn test_invalid_windowing_config() {
        let mut config = ProcessingConfig::default();
        config.windowing.overlap_percent = 150.0;
        assert!(validate_processing_config(&config).is_err());
    }

    #[test]
    fn test_config_serialization() {
        let config = ProcessingConfig::default();
        let toml_str = toml::to_string(&config).unwrap();
        let deserialized: ProcessingConfig = toml::from_str(&toml_str).unwrap();

        assert_eq!(config.filter_bank.highpass_cutoff_hz, deserialized.filter_bank.highpass_cutoff_hz);
        assert_eq!(config.quality_monitoring.snr_threshold_db, deserialized.quality_monitoring.snr_threshold_db);
    }
}