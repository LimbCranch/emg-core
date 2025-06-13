use crate::hal::types::{EmgSample, QualityMetrics};
use crate::processing::filters::{Filter, FilterChain};
use crate::processing::filters_v1::*;

use crate::processing::features::FeatureExtractor;
use crate::processing::noise_reduction::NoiseReducer;
use crate::error::EmgError;
use crate::utils::time::TimeProvider;
use std::sync::Arc;
use crate::processing::FeatureConfig;

/// Real-time signal processing pipeline
pub struct SignalPipeline {
    filter_chain: FilterChain,
    noise_reducer: NoiseReducer,
    feature_extractor: FeatureExtractor,
    config: PipelineConfig,
    time_provider: Arc<dyn TimeProvider>,
    performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub enable_filtering: bool,
    pub enable_noise_reduction: bool,
    pub enable_feature_extraction: bool,
    pub latency_target_ms: f32,
    pub quality_threshold: f32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            enable_filtering: true,
            enable_noise_reduction: true,
            enable_feature_extraction: true,
            latency_target_ms: 2.0,
            quality_threshold: 20.0, // SNR in dB
        }
    }
}

#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    pub total_samples_processed: u64,
    pub average_processing_time_us: f32,
    pub max_processing_time_us: f32,
    pub quality_violations: u64,
    pub latency_violations: u64,
}

/// Processed EMG sample with extracted features
#[derive(Debug, Clone)]
pub struct ProcessedSample {
    pub original: EmgSample,
    pub filtered_channels: Vec<f32>,
    pub features: FeatureVector,
    pub processing_time_us: f32,
    pub quality_score: f32,
}

// ================================================================================
// Updated FeatureVector definition
// File: src/processing/pipeline.rs (UPDATE)
// ================================================================================

/// Extended feature vector with all EMG features
#[derive(Debug, Clone)]
pub struct FeatureVector {
    // Time domain features
    pub rms: Vec<f32>,
    pub mean_absolute_value: Vec<f32>,
    pub zero_crossings: Vec<u32>,
    pub slope_sign_changes: Vec<u32>,
    pub waveform_length: Vec<f32>,
    pub variance: Vec<f32>,
    pub skewness: Vec<f32>,
    pub kurtosis: Vec<f32>,

    // Frequency domain features
    pub spectral_centroid: Vec<f32>,
    pub mean_frequency: Vec<f32>,
    pub median_frequency: Vec<f32>,
    pub power_spectral_density: Vec<Vec<f32>>,
    pub frequency_band_powers: Vec<Vec<f32>>,
    pub spectral_rolloff: Vec<f32>,
    pub spectral_flux: Vec<f32>,

    // Wavelet features (optional)
    pub wavelet_coefficients: Option<Vec<Vec<f32>>>,
    pub wavelet_energy: Option<Vec<Vec<f32>>>,
}

impl SignalPipeline {
    pub fn new(config: PipelineConfig,feature_config: FeatureConfig , time_provider: Arc<dyn TimeProvider>) -> Result<Self, EmgError> {
        let filter_chain = FilterChain::new(&config)?;
        let noise_reducer = NoiseReducer::new(&config)?;
        let feature_extractor = FeatureExtractor::new(&feature_config)?;

        Ok(Self {
            filter_chain,
            noise_reducer,
            feature_extractor,
            config,
            time_provider,
            performance_metrics: PerformanceMetrics::default(),
        })
    }

    /// Process a single EMG sample through the complete pipeline
    pub fn process_sample(&mut self, sample: EmgSample) -> Result<ProcessedSample, EmgError> {
        let start_time = self.time_provider.now_nanos();

        // Validate input quality
        if sample.quality_indicators.snr_db < self.config.quality_threshold {
            self.performance_metrics.quality_violations += 1;
        }

        // Stage 1: Digital filtering
        let filtered_channels = if self.config.enable_filtering {
            self.filter_chain.process(&sample.channels)?
        } else {
            sample.channels.clone()
        };

        // Stage 2: Noise reduction
        let denoised_channels = if self.config.enable_noise_reduction {
            self.noise_reducer.reduce_noise(&filtered_channels)?
        } else {
            filtered_channels
        };

        // Stage 3: Feature extraction
        let features = if self.config.enable_feature_extraction {
            self.feature_extractor.extract_features(&denoised_channels)?
        } else {
            FeatureVector::default_for_channels(denoised_channels.len())
        };

        let end_time = self.time_provider.now_nanos();
        let processing_time_us = (end_time - start_time) as f32 / 1000.0;

        // Update performance metrics
        self.update_performance_metrics(processing_time_us);

        // Check latency constraint
        let latency_target_us = self.config.latency_target_ms * 1000.0;
        if processing_time_us > latency_target_us {
            self.performance_metrics.latency_violations += 1;
        }

        let quality_score = self.calculate_quality_score(&sample.quality_indicators, &features);

        Ok(ProcessedSample {
            original: sample,
            filtered_channels: denoised_channels,
            features,
            processing_time_us,
            quality_score,
        })
    }

    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> &PerformanceMetrics {
        &self.performance_metrics
    }

    /// Reset performance metrics
    pub fn reset_metrics(&mut self) {
        self.performance_metrics = PerformanceMetrics::default();
    }

    fn update_performance_metrics(&mut self, processing_time_us: f32) {
        self.performance_metrics.total_samples_processed += 1;

        let n = self.performance_metrics.total_samples_processed as f32;
        self.performance_metrics.average_processing_time_us =
            (self.performance_metrics.average_processing_time_us * (n - 1.0) + processing_time_us) / n;

        if processing_time_us > self.performance_metrics.max_processing_time_us {
            self.performance_metrics.max_processing_time_us = processing_time_us;
        }
    }

    fn calculate_quality_score(&self, quality: &QualityMetrics, features: &FeatureVector) -> f32 {
        let mut score = quality.snr_db / 40.0; // Normalize SNR to 0-1 range

        // Penalize for artifacts
        if quality.artifact_detected {
            score *= 0.5;
        }

        // Penalize for saturation
        if quality.signal_saturation {
            score *= 0.3;
        }

        // Consider signal stability (low variance in features indicates good quality)
        let rms_variance = calculate_variance(&features.rms);
        score *= (1.0 - rms_variance.min(1.0));

        score.max(0.0).min(1.0)
    }
}


impl FeatureVector {
    pub fn default_for_channels(channel_count: usize) -> Self {
        Self {
            rms: vec![0.0; channel_count],
            mean_absolute_value: vec![0.0; channel_count],
            zero_crossings: vec![0; channel_count],
            slope_sign_changes: vec![0; channel_count],
            waveform_length: vec![0.0; channel_count],
            variance: vec![0.0; channel_count],
            skewness: vec![0.0; channel_count],
            kurtosis: vec![0.0; channel_count],
            spectral_centroid: vec![0.0; channel_count],
            mean_frequency: vec![0.0; channel_count],
            median_frequency: vec![0.0; channel_count],
            power_spectral_density: vec![vec![0.0; 128]; channel_count], // 128 frequency bins
            frequency_band_powers: vec![vec![0.0; 4]; channel_count], // 4 frequency bands
            spectral_rolloff: vec![0.0; channel_count],
            spectral_flux: vec![0.0; channel_count],
            wavelet_coefficients: None,
            wavelet_energy: None,
        }
    }

    /// Get total number of features for machine learning
    pub fn feature_count(&self) -> usize {
        let mut count = 0;

        // Time domain: 8 features per channel
        count += self.rms.len() * 8;

        // Frequency domain: 4 scalar + PSD bins + band powers per channel
        count += self.spectral_centroid.len() * 4; // scalar features
        count += self.power_spectral_density.iter()
            .map(|psd| psd.len())
            .sum::<usize>();
        count += self.frequency_band_powers.iter()
            .map(|bands| bands.len())
            .sum::<usize>();

        // Wavelet features if present
        if let Some(ref coeffs) = self.wavelet_coefficients {
            count += coeffs.iter().map(|c| c.len()).sum::<usize>();
        }

        count
    }

    /// Flatten all features into a single vector for ML algorithms
    pub fn to_flat_vector(&self) -> Vec<f32> {
        let mut features = Vec::with_capacity(self.feature_count());

        // Time domain features
        features.extend(&self.rms);
        features.extend(&self.mean_absolute_value);
        features.extend(self.zero_crossings.iter().map(|&x| x as f32));
        features.extend(self.slope_sign_changes.iter().map(|&x| x as f32));
        features.extend(&self.waveform_length);
        features.extend(&self.variance);
        features.extend(&self.skewness);
        features.extend(&self.kurtosis);

        // Frequency domain features
        features.extend(&self.spectral_centroid);
        features.extend(&self.mean_frequency);
        features.extend(&self.median_frequency);
        features.extend(&self.spectral_rolloff);
        features.extend(&self.spectral_flux);

        // PSD features (flatten)
        for psd in &self.power_spectral_density {
            features.extend(psd);
        }

        // Band power features
        for bands in &self.frequency_band_powers {
            features.extend(bands);
        }

        // Wavelet features if present
        if let Some(ref coeffs) = self.wavelet_coefficients {
            for channel_coeffs in coeffs {
                features.extend(channel_coeffs);
            }
        }

        features
    }

    /// Get feature names for interpretability
    pub fn feature_names(&self) -> Vec<String> {
        let mut names = Vec::with_capacity(self.feature_count());
        let channel_count = self.rms.len();

        // Time domain feature names
        for ch in 0..channel_count {
            names.push(format!("ch{}_rms", ch));
            names.push(format!("ch{}_mav", ch));
            names.push(format!("ch{}_zc", ch));
            names.push(format!("ch{}_ssc", ch));
            names.push(format!("ch{}_wl", ch));
            names.push(format!("ch{}_var", ch));
            names.push(format!("ch{}_skew", ch));
            names.push(format!("ch{}_kurt", ch));
        }

        // Frequency domain feature names
        for ch in 0..channel_count {
            names.push(format!("ch{}_spec_centroid", ch));
            names.push(format!("ch{}_mean_freq", ch));
            names.push(format!("ch{}_median_freq", ch));
            names.push(format!("ch{}_spec_rolloff", ch));
            names.push(format!("ch{}_spec_flux", ch));
        }

        // PSD bin names
        for ch in 0..channel_count {
            for bin in 0..self.power_spectral_density[ch].len() {
                names.push(format!("ch{}_psd_bin{}", ch, bin));
            }
        }

        // Band power names
        for ch in 0..channel_count {
            for band in 0..self.frequency_band_powers[ch].len() {
                names.push(format!("ch{}_band{}_power", ch, band));
            }
        }

        // Wavelet feature names
        if let Some(ref coeffs) = self.wavelet_coefficients {
            for ch in 0..coeffs.len() {
                for coeff_idx in 0..coeffs[ch].len() {
                    names.push(format!("ch{}_wavelet_coeff{}", ch, coeff_idx));
                }
            }
        }

        names
    }
}


fn calculate_variance(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>() / values.len() as f32;
    variance
}
                    