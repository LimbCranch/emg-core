
// ================================================================================
// File: src/processing/noise_reduction/wiener_filter.rs
// ================================================================================

//! Wiener filtering for optimal noise reduction

use crate::error::EmgError;
use super::NoiseReductionConfig;

/// Wiener filter for noise reduction
pub struct WienerFilter {
    filter_order: usize,
    channel_count: usize,
    signal_buffer: Vec<Vec<f32>>,
    noise_variance: f32,
    signal_variance: f32,
    adaptation_rate: f32,
}

impl WienerFilter {
    pub fn new(config: &NoiseReductionConfig, channel_count: usize) -> Result<Self, EmgError> {
        let filter_order = 16;

        Ok(Self {
            filter_order,
            channel_count,
            signal_buffer: vec![vec![0.0; filter_order]; channel_count],
            noise_variance: 0.01,    // Initial estimate
            signal_variance: 1.0,    // Initial estimate
            adaptation_rate: config.adaptation_rate,
        })
    }

    /// Apply Wiener filtering
    pub fn filter(&mut self, input: &[f32]) -> Result<Vec<f32>, EmgError> {
        if input.len() != self.channel_count {
            return Err(EmgError::Processing(
                format!("Input channel count mismatch: expected {}, got {}",
                        self.channel_count, input.len())
            ));
        }

        let mut output = Vec::with_capacity(self.channel_count);

        for ch in 0..self.channel_count {
            // Update signal buffer
            self.signal_buffer[ch].rotate_right(1);
            self.signal_buffer[ch][0] = input[ch];

            // Apply Wiener filter
            let filtered = self.apply_wiener_filter(ch, input[ch]);
            output.push(filtered);

            // Update noise and signal variance estimates
            self.update_variance_estimates(input[ch], filtered);
        }

        Ok(output)
    }

    pub fn reset(&mut self) {
        for ch in 0..self.channel_count {
            self.signal_buffer[ch].fill(0.0);
        }
        self.noise_variance = 0.01;
        self.signal_variance = 1.0;
    }

    // Private methods
    fn apply_wiener_filter(&self, channel: usize, input: f32) -> f32 {
        // Simplified Wiener filter: H = S² / (S² + N²)
        let wiener_gain = self.signal_variance / (self.signal_variance + self.noise_variance);
        input * wiener_gain
    }

    fn update_variance_estimates(&mut self, input: f32, output: f32) {
        // Simple variance estimation using exponential smoothing
        let alpha = self.adaptation_rate;

        // Estimate signal variance from output
        self.signal_variance = (1.0 - alpha) * self.signal_variance + alpha * output * output;

        // Estimate noise variance from difference
        let noise_estimate = input - output;
        self.noise_variance = (1.0 - alpha) * self.noise_variance + alpha * noise_estimate * noise_estimate;

        // Ensure positive variances
        self.signal_variance = self.signal_variance.max(0.001);
        self.noise_variance = self.noise_variance.max(0.001);
    }
}
