// ================================================================================
// File: src/processing/noise_reduction/adaptive_filter.rs
// ================================================================================

//! Adaptive filtering for EMG noise reduction using LMS and NLMS algorithms

use crate::error::EmgError;
use super::NoiseReductionConfig;

/// Adaptive filter using Least Mean Squares (LMS) algorithm
pub struct AdaptiveFilter {
    filter_order: usize,
    adaptation_rate: f32,
    filter_coefficients: Vec<Vec<f32>>, // Per-channel coefficients
    input_buffer: Vec<Vec<f32>>,        // Delay line for each channel
    reference_buffer: Vec<f32>,         // Reference noise signal
    channel_count: usize,
    algorithm: AdaptiveAlgorithm,
}

#[derive(Debug, Clone)]
pub enum AdaptiveAlgorithm {
    LMS,    // Least Mean Squares
    NLMS,   // Normalized LMS
    RLS,    // Recursive Least Squares (more complex but faster convergence)
}

impl AdaptiveFilter {
    pub fn new(config: &NoiseReductionConfig, channel_count: usize) -> Result<Self, EmgError> {
        let filter_order = 32; // Good balance between performance and adaptation speed

        Ok(Self {
            filter_order,
            adaptation_rate: config.adaptation_rate,
            filter_coefficients: vec![vec![0.0; filter_order]; channel_count],
            input_buffer: vec![vec![0.0; filter_order]; channel_count],
            reference_buffer: vec![0.0; filter_order],
            channel_count,
            algorithm: AdaptiveAlgorithm::NLMS, // NLMS is more stable
        })
    }

    /// Filter the input signal using adaptive filtering
    pub fn filter(&mut self, input: &[f32]) -> Result<Vec<f32>, EmgError> {
        if input.len() != self.channel_count {
            return Err(EmgError::Processing(
                format!("Input channel count mismatch: expected {}, got {}",
                        self.channel_count, input.len())
            ));
        }

        let mut output = Vec::with_capacity(self.channel_count);

        for ch in 0..self.channel_count {
            // Update input buffer (delay line)
            self.input_buffer[ch].rotate_right(1);
            self.input_buffer[ch][0] = input[ch];

            // Estimate reference noise (simple approach: use neighboring channels)
            let reference = self.estimate_reference_noise(ch, input);

            // Update reference buffer
            self.reference_buffer.rotate_right(1);
            self.reference_buffer[0] = reference;

            // Apply adaptive filter
            let filtered_output = self.apply_adaptive_filter(ch)?;
            output.push(filtered_output);

            // Update filter coefficients
            let error = input[ch] - filtered_output;
            self.update_coefficients(ch, error)?;
        }

        Ok(output)
    }

    pub fn reset(&mut self) {
        for ch in 0..self.channel_count {
            self.filter_coefficients[ch].fill(0.0);
            self.input_buffer[ch].fill(0.0);
        }
        self.reference_buffer.fill(0.0);
    }

    // Private methods
    fn estimate_reference_noise(&self, current_channel: usize, input: &[f32]) -> f32 {
        // Simple approach: use weighted average of other channels
        // In practice, this could be a dedicated reference microphone or more sophisticated estimation
        let mut reference = 0.0;
        let mut count = 0;

        for (ch, &value) in input.iter().enumerate() {
            if ch != current_channel {
                reference += value;
                count += 1;
            }
        }

        if count > 0 {
            reference / count as f32
        } else {
            0.0
        }
    }

    fn apply_adaptive_filter(&self, channel: usize) -> Result<f32, EmgError> {
        // Convolve filter coefficients with reference buffer
        let mut output = 0.0;

        for i in 0..self.filter_order {
            output += self.filter_coefficients[channel][i] * self.reference_buffer[i];
        }

        Ok(output)
    }

    fn update_coefficients(&mut self, channel: usize, error: f32) -> Result<(), EmgError> {
        match self.algorithm {
            AdaptiveAlgorithm::LMS => {
                // Standard LMS update: w(n+1) = w(n) + μ * e(n) * x(n)
                for i in 0..self.filter_order {
                    self.filter_coefficients[channel][i] +=
                        self.adaptation_rate * error * self.reference_buffer[i];
                }
            }

            AdaptiveAlgorithm::NLMS => {
                // Normalized LMS: w(n+1) = w(n) + (μ / (δ + ||x||²)) * e(n) * x(n)
                let input_power: f32 = self.reference_buffer.iter().map(|&x| x * x).sum();
                let normalization = 1.0 / (0.001 + input_power); // δ = 0.001 for numerical stability

                for i in 0..self.filter_order {
                    self.filter_coefficients[channel][i] +=
                        self.adaptation_rate * normalization * error * self.reference_buffer[i];
                }
            }

            AdaptiveAlgorithm::RLS => {
                // RLS is more complex and would require additional state variables
                // For now, fall back to NLMS
                return self.update_coefficients_nlms(channel, error);
            }
        }

        Ok(())
    }

    fn update_coefficients_nlms(&mut self, channel: usize, error: f32) -> Result<(), EmgError> {
        let input_power: f32 = self.reference_buffer.iter().map(|&x| x * x).sum();
        let normalization = 1.0 / (0.001 + input_power);

        for i in 0..self.filter_order {
            self.filter_coefficients[channel][i] +=
                self.adaptation_rate * normalization * error * self.reference_buffer[i];
        }

        Ok(())
    }
}