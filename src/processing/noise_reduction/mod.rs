// ================================================================================
// File: src/processing/noise_reduction/mod.rs
// ================================================================================

//! Noise reduction algorithms for EMG signal processing
//!
//! This module implements various noise reduction techniques including:
//! - Adaptive filtering
//! - Spectral subtraction
//! - Wiener filtering
//! - Empirical mode decomposition (EMD) based denoising

pub mod adaptive_filter;
pub mod spectral_subtraction;
pub mod wiener_filter;

use crate::error::EmgError;
use crate::processing::pipeline::PipelineConfig;

pub use adaptive_filter::AdaptiveFilter;
pub use spectral_subtraction::SpectralSubtraction;
pub use wiener_filter::WienerFilter;

/// Configuration for noise reduction algorithms
#[derive(Debug, Clone)]
pub struct NoiseReductionConfig {
    pub algorithm: NoiseReductionAlgorithm,
    pub adaptation_rate: f32,
    pub noise_floor_db: f32,
    pub over_subtraction_factor: f32,
    pub spectral_floor: f32,
    pub frame_size: usize,
    pub overlap_ratio: f32,
}

#[derive(Debug, Clone)]
pub enum NoiseReductionAlgorithm {
    Adaptive,           // LMS/NLMS adaptive filtering
    SpectralSubtraction, // Frequency domain spectral subtraction
    Wiener,             // Wiener filtering
    Hybrid,             // Combination of methods
}

impl Default for NoiseReductionConfig {
    fn default() -> Self {
        Self {
            algorithm: NoiseReductionAlgorithm::Adaptive,
            adaptation_rate: 0.01,
            noise_floor_db: -40.0,
            over_subtraction_factor: 2.0,
            spectral_floor: 0.1,
            frame_size: 256,
            overlap_ratio: 0.5,
        }
    }
}

/// Main noise reducer that coordinates different denoising algorithms
pub struct NoiseReducer {
    config: NoiseReductionConfig,
    adaptive_filter: Option<AdaptiveFilter>,
    spectral_subtraction: Option<SpectralSubtraction>,
    wiener_filter: Option<WienerFilter>,
    channel_count: usize,
}

impl NoiseReducer {
    pub fn new(pipeline_config: &PipelineConfig) -> Result<Self, EmgError> {
        let config = NoiseReductionConfig::default();

        let mut noise_reducer = Self {
            config: config.clone(),
            adaptive_filter: None,
            spectral_subtraction: None,
            wiener_filter: None,
            channel_count: 8, // Default, will be updated on first use
        };

        // Initialize the selected algorithm
        noise_reducer.initialize_algorithm()?;

        Ok(noise_reducer)
    }

    pub fn with_config(config: NoiseReductionConfig) -> Result<Self, EmgError> {
        let mut noise_reducer = Self {
            config: config.clone(),
            adaptive_filter: None,
            spectral_subtraction: None,
            wiener_filter: None,
            channel_count: 8,
        };

        noise_reducer.initialize_algorithm()?;
        Ok(noise_reducer)
    }

    /// Reduce noise in the input signal
    pub fn reduce_noise(&mut self, input: &[f32]) -> Result<Vec<f32>, EmgError> {
        // Update channel count if needed
        if self.channel_count != input.len() {
            self.channel_count = input.len();
            self.initialize_algorithm()?;
        }

        match self.config.algorithm {
            NoiseReductionAlgorithm::Adaptive => {
                if let Some(ref mut filter) = self.adaptive_filter {
                    filter.filter(input)
                } else {
                    Err(EmgError::Processing("Adaptive filter not initialized".to_string()))
                }
            }

            NoiseReductionAlgorithm::SpectralSubtraction => {
                if let Some(ref mut subtractor) = self.spectral_subtraction {
                    subtractor.process(input)
                } else {
                    Err(EmgError::Processing("Spectral subtraction not initialized".to_string()))
                }
            }

            NoiseReductionAlgorithm::Wiener => {
                if let Some(ref mut filter) = self.wiener_filter {
                    filter.filter(input)
                } else {
                    Err(EmgError::Processing("Wiener filter not initialized".to_string()))
                }
            }

            NoiseReductionAlgorithm::Hybrid => {
                // Apply multiple algorithms in sequence
                let mut output = input.to_vec();

                if let Some(ref mut adaptive) = self.adaptive_filter {
                    output = adaptive.filter(&output)?;
                }

                if let Some(ref mut spectral) = self.spectral_subtraction {
                    output = spectral.process(&output)?;
                }

                Ok(output)
            }
        }
    }

    /// Reset all noise reduction algorithms
    pub fn reset(&mut self) {
        if let Some(ref mut filter) = self.adaptive_filter {
            filter.reset();
        }
        if let Some(ref mut subtractor) = self.spectral_subtraction {
            subtractor.reset();
        }
        if let Some(ref mut filter) = self.wiener_filter {
            filter.reset();
        }
    }

    /// Get the current configuration
    pub fn config(&self) -> &NoiseReductionConfig {
        &self.config
    }

    /// Update the noise reduction algorithm
    pub fn set_algorithm(&mut self, algorithm: NoiseReductionAlgorithm) -> Result<(), EmgError> {
        self.config.algorithm = algorithm;
        self.initialize_algorithm()
    }

    // Private methods
    fn initialize_algorithm(&mut self) -> Result<(), EmgError> {
        match self.config.algorithm {
            NoiseReductionAlgorithm::Adaptive => {
                self.adaptive_filter = Some(AdaptiveFilter::new(&self.config, self.channel_count)?);
            }

            NoiseReductionAlgorithm::SpectralSubtraction => {
                self.spectral_subtraction = Some(SpectralSubtraction::new(&self.config)?);
            }

            NoiseReductionAlgorithm::Wiener => {
                self.wiener_filter = Some(WienerFilter::new(&self.config, self.channel_count)?);
            }

            NoiseReductionAlgorithm::Hybrid => {
                // Initialize all algorithms for hybrid approach
                self.adaptive_filter = Some(AdaptiveFilter::new(&self.config, self.channel_count)?);
                self.spectral_subtraction = Some(SpectralSubtraction::new(&self.config)?);
                self.wiener_filter = Some(WienerFilter::new(&self.config, self.channel_count)?);
            }
        }

        Ok(())
    }
}
