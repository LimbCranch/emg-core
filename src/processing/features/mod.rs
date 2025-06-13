//! EMG signal feature extraction module
//!
//! This module provides comprehensive feature extraction for EMG signals including:
//! - Time domain features (RMS, MAV, ZC, SSC, WL)
//! - Frequency domain features (MNF, MDF, spectral moments)
//! - Time-frequency features (wavelet coefficients)
//! - Advanced features for gesture recognition

pub mod time_domain;
pub mod frequency;
pub mod wavelets;

use crate::error::EmgError;
use crate::processing::pipeline::FeatureVector;
use std::collections::VecDeque;

pub use time_domain::TimeDomainExtractor;
pub use frequency::FrequencyDomainExtractor;
pub use wavelets::WaveletExtractor;

/// Configuration for feature extraction
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    pub window_size: usize,
    pub overlap_ratio: f32,
    pub sample_rate_hz: u32,
    pub enable_time_domain: bool,
    pub enable_frequency_domain: bool,
    pub enable_wavelets: bool,
    pub frequency_bands: Vec<(f32, f32)>, // (low, high) Hz pairs
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            window_size: 256,           // ~128ms at 2kHz
            overlap_ratio: 0.5,         // 50% overlap
            sample_rate_hz: 2000,
            enable_time_domain: true,
            enable_frequency_domain: true,
            enable_wavelets: false,      // Computationally expensive
            frequency_bands: vec![
                (20.0, 50.0),    // Low frequency
                (50.0, 150.0),   // Mid frequency
                (150.0, 300.0),  // High frequency
                (300.0, 500.0),  // Very high frequency
            ],
        }
    }
}

/// Main feature extractor that coordinates all feature extraction methods
pub struct FeatureExtractor {
    config: FeatureConfig,
    time_domain: TimeDomainExtractor,
    frequency_domain: FrequencyDomainExtractor,
    wavelet: Option<WaveletExtractor>,
    sample_buffer: VecDeque<Vec<f32>>,
    current_window: Vec<Vec<f32>>,
    samples_since_last_extraction: usize,
}

impl FeatureExtractor {
    pub fn new(config: &FeatureConfig) -> Result<Self, EmgError> {
        let time_domain = TimeDomainExtractor::new(config)?;
        let frequency_domain = FrequencyDomainExtractor::new(config)?;
        let wavelet = if config.enable_wavelets {
            Some(WaveletExtractor::new(config)?)
        } else {
            None
        };

        Ok(Self {
            config: config.clone(),
            time_domain,
            frequency_domain,
            wavelet,
            sample_buffer: VecDeque::with_capacity(config.window_size * 2),
            current_window: Vec::new(),
            samples_since_last_extraction: 0,
        })
    }

    /// Extract features from a single sample (streaming mode)
    pub fn extract_features(&mut self, channels: &[f32]) -> Result<FeatureVector, EmgError> {
        // Add sample to buffer
        self.sample_buffer.push_back(channels.to_vec());
        self.samples_since_last_extraction += 1;

        // Keep buffer size manageable
        while self.sample_buffer.len() > self.config.window_size * 2 {
            self.sample_buffer.pop_front();
        }

        // Check if we have enough samples for feature extraction
        let step_size = (self.config.window_size as f32 * (1.0 - self.config.overlap_ratio)) as usize;
        if self.samples_since_last_extraction < step_size.max(1) {
            // Return previous features or zeros if not enough samples
            return Ok(FeatureVector::default_for_channels(channels.len()));
        }

        // Extract windowed data
        if self.sample_buffer.len() >= self.config.window_size {
            let start_idx = self.sample_buffer.len() - self.config.window_size;
            self.current_window = self.sample_buffer
                .range(start_idx..)
                .cloned()
                .collect();

            self.samples_since_last_extraction = 0;
            self.extract_features_from_window()
        } else {
            Ok(FeatureVector::default_for_channels(channels.len()))
        }
    }

    /// Extract features from a complete window of samples
    pub fn extract_features_from_window(&mut self) -> Result<FeatureVector, EmgError> {
        if self.current_window.is_empty() {
            return Err(EmgError::Processing("No window data available".to_string()));
        }

        let channel_count = self.current_window[0].len();
        let mut feature_vector = FeatureVector::default_for_channels(channel_count);

        // Time domain features
        if self.config.enable_time_domain {
            let time_features = self.time_domain.extract(&self.current_window)?;
            feature_vector.rms = time_features.rms;
            feature_vector.mean_absolute_value = time_features.mean_absolute_value;
            feature_vector.zero_crossings = time_features.zero_crossings;
            feature_vector.slope_sign_changes = time_features.slope_sign_changes;
            feature_vector.waveform_length = time_features.waveform_length;
        }

        // Frequency domain features
        if self.config.enable_frequency_domain {
            let freq_features = self.frequency_domain.extract(&self.current_window)?;
            feature_vector.spectral_centroid = freq_features.spectral_centroid;
            feature_vector.mean_frequency = freq_features.mean_frequency;
            feature_vector.median_frequency = freq_features.median_frequency;
            feature_vector.power_spectral_density = freq_features.power_spectral_density;
            feature_vector.frequency_band_powers = freq_features.frequency_band_powers;
        }

        // Wavelet features (optional)
        if let Some(ref mut wavelet_extractor) = self.wavelet {
            let wavelet_features = wavelet_extractor.extract(&self.current_window)?;
            feature_vector.wavelet_coefficients = Some(wavelet_features.coefficients);
            feature_vector.wavelet_energy = Some(wavelet_features.energy_distribution);
        }

        Ok(feature_vector)
    }

    /// Reset the feature extractor state
    pub fn reset(&mut self) {
        self.sample_buffer.clear();
        self.current_window.clear();
        self.samples_since_last_extraction = 0;
        self.time_domain.reset();
        self.frequency_domain.reset();
        if let Some(ref mut wavelet) = self.wavelet {
            wavelet.reset();
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &FeatureConfig {
        &self.config
    }
}