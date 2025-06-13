// src/processing/filters/iir.rs
//! IIR (Infinite Impulse Response) digital filters

use super::{FilterError, BandType, IirCoefficients};

/// IIR filter with configurable parameters
pub struct IirFilter {
    coefficients: IirCoefficients,
    x_history: Vec<f32>, // Input history
    y_history: Vec<f32>, // Output history
    order: usize,
}

impl IirFilter {
    /// Create Butterworth filter
    pub fn butterworth(order: usize, cutoff: f32, sample_rate: f32, band_type: BandType) -> Result<Self, FilterError> {
        if order == 0 || order > 8 {
            return Err(FilterError::InvalidParameters("Order must be 1-8".to_string()));
        }
        if cutoff <= 0.0 || cutoff >= sample_rate / 2.0 {
            return Err(FilterError::InvalidParameters("Invalid cutoff frequency".to_string()));
        }

        let coefficients = Self::calculate_butterworth_coefficients(order, cutoff, sample_rate, band_type)?;
        Ok(Self::new(coefficients))
    }

    /// Create filter from coefficients
    pub fn new(coefficients: IirCoefficients) -> Self {
        let order = coefficients.a.len() - 1;
        Self {
            x_history: vec![0.0; coefficients.b.len()],
            y_history: vec![0.0; coefficients.a.len()],
            coefficients,
            order,
        }
    }

    /// Process single sample using Direct Form II
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Shift input history
        for i in (1..self.x_history.len()).rev() {
            self.x_history[i] = self.x_history[i - 1];
        }
        self.x_history[0] = input;

        // Calculate output
        let mut output = 0.0;

        // Numerator (b coefficients)
        for i in 0..self.coefficients.b.len() {
            output += self.coefficients.b[i] * self.x_history[i];
        }

        // Denominator (a coefficients, skip a[0])
        for i in 1..self.coefficients.a.len() {
            output -= self.coefficients.a[i] * self.y_history[i];
        }

        // Normalize by a[0]
        output /= self.coefficients.a[0];

        // Shift output history
        for i in (1..self.y_history.len()).rev() {
            self.y_history[i] = self.y_history[i - 1];
        }
        self.y_history[0] = output;

        output
    }

    /// Reset filter state
    pub fn reset(&mut self) {
        self.x_history.fill(0.0);
        self.y_history.fill(0.0);
    }

    /// Get filter order
    pub fn order(&self) -> usize {
        self.order
    }

    fn calculate_butterworth_coefficients(
        order: usize,
        cutoff: f32,
        sample_rate: f32,
        band_type: BandType
    ) -> Result<IirCoefficients, FilterError> {
        let nyquist = sample_rate / 2.0;
        let normalized_cutoff = cutoff / nyquist;

        if normalized_cutoff >= 1.0 {
            return Err(FilterError::InvalidParameters("Cutoff too high".to_string()));
        }

        // Pre-warp frequency for bilinear transform
        let omega_c = (std::f32::consts::PI * normalized_cutoff).tan();

        match band_type {
            BandType::Lowpass => Self::butterworth_lowpass(order, omega_c),
            BandType::Highpass => Self::butterworth_highpass(order, omega_c),
            _ => Err(FilterError::InvalidParameters("Bandtype not yet implemented".to_string())),
        }
    }

    fn butterworth_lowpass(order: usize, omega_c: f32) -> Result<IirCoefficients, FilterError> {
        match order {
            1 => {
                let k = omega_c;
                let norm = 1.0 + k;
                Ok(IirCoefficients {
                    b: vec![k / norm, k / norm],
                    a: vec![1.0, (k - 1.0) / norm],
                })
            },
            2 => {
                let k = omega_c;
                let k2 = k * k;
                let sqrt2 = std::f32::consts::SQRT_2;
                let norm = 1.0 + k * sqrt2 + k2;
                Ok(IirCoefficients {
                    b: vec![k2 / norm, 2.0 * k2 / norm, k2 / norm],
                    a: vec![1.0, (2.0 * k2 - 2.0) / norm, (1.0 - k * sqrt2 + k2) / norm],
                })
            },
            _ => Err(FilterError::InvalidParameters("Only 1st and 2nd order implemented".to_string())),
        }
    }

    fn butterworth_highpass(order: usize, omega_c: f32) -> Result<IirCoefficients, FilterError> {
        match order {
            1 => {
                let k = omega_c;
                let norm = 1.0 + k;
                Ok(IirCoefficients {
                    b: vec![1.0 / norm, -1.0 / norm],
                    a: vec![1.0, (k - 1.0) / norm],
                })
            },
            2 => {
                let k = omega_c;
                let k2 = k * k;
                let sqrt2 = std::f32::consts::SQRT_2;
                let norm = 1.0 + k * sqrt2 + k2;
                Ok(IirCoefficients {
                    b: vec![1.0 / norm, -2.0 / norm, 1.0 / norm],
                    a: vec![1.0, (2.0 * k2 - 2.0) / norm, (1.0 - k * sqrt2 + k2) / norm],
                })
            },
            _ => Err(FilterError::InvalidParameters("Only 1st and 2nd order implemented".to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_butterworth_lowpass_creation() {
        let filter = IirFilter::butterworth(2, 100.0, 1000.0, BandType::Lowpass);
        assert!(filter.is_ok());
        assert_eq!(filter.unwrap().order(), 2);
    }

    #[test]
    fn test_filter_processing() {
        let mut filter = IirFilter::butterworth(1, 10.0, 100.0, BandType::Lowpass).unwrap();

        let output1 = filter.process_sample(1.0);
        let output2 = filter.process_sample(0.0);

        assert!(output1 > 0.0);
        assert!(output2 < output1);
    }

    #[test]
    fn test_invalid_parameters() {
        assert!(IirFilter::butterworth(0, 100.0, 1000.0, BandType::Lowpass).is_err());
        assert!(IirFilter::butterworth(2, 600.0, 1000.0, BandType::Lowpass).is_err());
        assert!(IirFilter::butterworth(2, 0.0, 1000.0, BandType::Lowpass).is_err());
    }
}