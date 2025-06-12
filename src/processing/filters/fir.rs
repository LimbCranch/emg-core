// src/processing/filters/fir.rs
//! FIR (Finite Impulse Response) digital filters

use super::FilterError;

/// FIR filter implementation
pub struct FirFilter {
    coefficients: Vec<f32>,
    delay_line: Vec<f32>,
    index: usize,
    length: usize,
}

impl FirFilter {
    /// Create FIR filter from coefficients
    pub fn new(coefficients: Vec<f32>) -> Result<Self, FilterError> {
        if coefficients.is_empty() {
            return Err(FilterError::InvalidCoefficients("Empty coefficients".to_string()));
        }

        let length = coefficients.len();
        Ok(Self {
            coefficients,
            delay_line: vec![0.0; length],
            index: 0,
            length,
        })
    }

    /// Create low-pass FIR using windowed sinc
    pub fn lowpass_windowed_sinc(cutoff: f32, sample_rate: f32, length: usize) -> Result<Self, FilterError> {
        if length % 2 == 0 {
            return Err(FilterError::InvalidParameters("Length must be odd".to_string()));
        }
        if cutoff <= 0.0 || cutoff >= sample_rate / 2.0 {
            return Err(FilterError::InvalidParameters("Invalid cutoff frequency".to_string()));
        }

        let mut coefficients = vec![0.0; length];
        let fc = cutoff / sample_rate;
        let m = (length - 1) / 2;

        for i in 0..length {
            let n = i as isize - m as isize;
            if n == 0 {
                coefficients[i] = 2.0 * fc;
            } else {
                let n_f = n as f32;
                coefficients[i] = (2.0 * std::f32::consts::PI * fc * n_f).sin() / (std::f32::consts::PI * n_f);
            }

            // Apply Hamming window
            let window = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (length - 1) as f32).cos();
            coefficients[i] *= window;
        }

        Self::new(coefficients)
    }

    /// Process single sample
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Store input in circular delay line
        self.delay_line[self.index] = input;

        // Calculate output using convolution
        let mut output = 0.0;
        for i in 0..self.length {
            let delay_index = (self.index + self.length - i) % self.length;
            output += self.coefficients[i] * self.delay_line[delay_index];
        }

        // Update circular buffer index
        self.index = (self.index + 1) % self.length;

        output
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.delay_line.fill(0.0);
        self.index = 0;
    }

    /// Get filter length
    pub fn length(&self) -> usize {
        self.length
    }

    /// Get coefficients
    pub fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fir_creation() {
        let coeffs = vec![0.1, 0.2, 0.4, 0.2, 0.1];
        let filter = FirFilter::new(coeffs);
        assert!(filter.is_ok());
        assert_eq!(filter.unwrap().length(), 5);
    }

    #[test]
    fn test_lowpass_windowed_sinc() {
        let filter = FirFilter::lowpass_windowed_sinc(100.0, 1000.0, 21);
        assert!(filter.is_ok());

        let filter = filter.unwrap();
        assert_eq!(filter.length(), 21);

        // Check symmetry
        let coeffs = filter.coefficients();
        for i in 0..10 {
            assert!((coeffs[i] - coeffs[20 - i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fir_processing() {
        let coeffs = vec![0.5, 0.5];
        let mut filter = FirFilter::new(coeffs).unwrap();

        let output1 = filter.process_sample(1.0);
        let output2 = filter.process_sample(1.0);

        assert_eq!(output1, 0.5);
        assert_eq!(output2, 1.0);
    }

    #[test]
    fn test_invalid_parameters() {
        assert!(FirFilter::new(vec![]).is_err());
        assert!(FirFilter::lowpass_windowed_sinc(100.0, 1000.0, 20).is_err()); // Even length
        assert!(FirFilter::lowpass_windowed_sinc(600.0, 1000.0, 21).is_err()); // Cutoff too high
    }
}