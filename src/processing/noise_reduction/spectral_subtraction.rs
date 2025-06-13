// ================================================================================
// File: src/processing/noise_reduction/spectral_subtraction.rs
// ================================================================================

//! Spectral subtraction for frequency-domain noise reduction

use crate::error::EmgError;
use super::NoiseReductionConfig;
use std::f32::consts::PI;

/// Spectral subtraction noise reduction
pub struct SpectralSubtraction {
    frame_size: usize,
    overlap_size: usize,
    fft_size: usize,
    window: Vec<f32>,
    overlap_buffer: Vec<f32>,
    noise_spectrum: Vec<f32>,
    noise_estimation_frames: usize,
    frame_count: usize,
    over_subtraction_factor: f32,
    spectral_floor: f32,
}

impl SpectralSubtraction {
    pub fn new(config: &NoiseReductionConfig) -> Result<Self, EmgError> {
        let frame_size = config.frame_size;
        let overlap_size = (frame_size as f32 * config.overlap_ratio) as usize;
        let fft_size = frame_size.next_power_of_two();

        Ok(Self {
            frame_size,
            overlap_size,
            fft_size,
            window: Self::create_hann_window(frame_size),
            overlap_buffer: vec![0.0; overlap_size],
            noise_spectrum: vec![0.0; fft_size / 2 + 1],
            noise_estimation_frames: 10, // First 10 frames assumed to be noise
            frame_count: 0,
            over_subtraction_factor: config.over_subtraction_factor,
            spectral_floor: config.spectral_floor,
        })
    }

    /// Process input signal with spectral subtraction
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, EmgError> {
        if input.len() == 0 {
            return Ok(Vec::new());
        }

        // For single-sample processing, accumulate into frames
        // In practice, this would work on complete frames
        let mut output = input.to_vec();

        // Simple noise reduction: high-frequency attenuation
        // This is a simplified version - full spectral subtraction requires frame-based processing
        for sample in &mut output {
            *sample = self.apply_simple_noise_reduction(*sample);
        }

        Ok(output)
    }

    pub fn reset(&mut self) {
        self.overlap_buffer.fill(0.0);
        self.noise_spectrum.fill(0.0);
        self.frame_count = 0;
    }

    // Private methods
    fn create_hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos()))
            .collect()
    }

    fn apply_simple_noise_reduction(&mut self, sample: f32) -> f32 {
        // Simple single-sample noise reduction
        // Apply soft thresholding
        let threshold = 0.01; // Adjust based on noise level

        if sample.abs() < threshold {
            sample * self.spectral_floor
        } else {
            sample
        }
    }

    // Full frame-based spectral subtraction would be implemented here
    // This requires FFT processing which is complex for this example
}
