// src/processing/filters_v1/notch.rs
//! Notch filters_v1 for powerline interference removal

use super::FilterError;

/// Notch filter for removing specific frequency
pub struct NotchFilter {
    biquad_sections: Vec<BiquadSection>,
    center_freq: f32,
    bandwidth: f32,
    sample_rate: f32,
}

/// Single biquad section for notch filtering
struct BiquadSection {
    b0: f32, b1: f32, b2: f32,
    a1: f32, a2: f32,
    x1: f32, x2: f32,
    y1: f32, y2: f32,
}

impl NotchFilter {
    /// Create notch filter for specific frequency
    pub fn new(center_freq: f32, bandwidth: f32, sample_rate: f32) -> Result<Self, FilterError> {
        if center_freq <= 0.0 || center_freq >= sample_rate / 2.0 {
            return Err(FilterError::InvalidParameters("Invalid center frequency".to_string()));
        }
        if bandwidth <= 0.0 || bandwidth >= center_freq {
            return Err(FilterError::InvalidParameters("Invalid bandwidth".to_string()));
        }

        let section = BiquadSection::notch(center_freq, bandwidth, sample_rate)?;

        Ok(Self {
            biquad_sections: vec![section],
            center_freq,
            bandwidth,
            sample_rate,
        })
    }

    /// Create multiple notch filters_v1 for powerline harmonics
    pub fn powerline_harmonics(base_freq: f32, bandwidth: f32, sample_rate: f32, harmonics: usize) -> Result<Self, FilterError> {
        let mut sections = Vec::new();

        for harmonic in 1..=harmonics {
            let freq = base_freq * harmonic as f32;
            if freq < sample_rate / 2.0 {
                sections.push(BiquadSection::notch(freq, bandwidth, sample_rate)?);
            }
        }

        if sections.is_empty() {
            return Err(FilterError::InvalidParameters("No valid harmonics".to_string()));
        }

        Ok(Self {
            biquad_sections: sections,
            center_freq: base_freq,
            bandwidth,
            sample_rate,
        })
    }

    /// Process single sample through all sections
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let mut output = input;
        for section in &mut self.biquad_sections {
            output = section.process(output);
        }
        output
    }

    /// Reset all filter states
    pub fn reset(&mut self) {
        for section in &mut self.biquad_sections {
            section.reset();
        }
    }

    /// Get center frequency
    pub fn center_frequency(&self) -> f32 {
        self.center_freq
    }

    /// Get bandwidth
    pub fn bandwidth(&self) -> f32 {
        self.bandwidth
    }

    /// Get number of sections
    pub fn section_count(&self) -> usize {
        self.biquad_sections.len()
    }
}

impl BiquadSection {
    fn notch(center_freq: f32, bandwidth: f32, sample_rate: f32) -> Result<Self, FilterError> {
        let omega = 2.0 * std::f32::consts::PI * center_freq / sample_rate;
        let alpha = (2.0 * std::f32::consts::PI * bandwidth / sample_rate).sin() / 2.0;

        let cos_omega = omega.cos();
        let norm = 1.0 + alpha;

        Ok(Self {
            b0: 1.0 / norm,
            b1: -2.0 * cos_omega / norm,
            b2: 1.0 / norm,
            a1: -2.0 * cos_omega / norm,
            a2: (1.0 - alpha) / norm,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        })
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1 - self.a2 * self.y2;

        // Update delay line
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_notch_filter_creation() {
        let filter = NotchFilter::new(50.0, 2.0, 1000.0);
        assert!(filter.is_ok());

        let filter = filter.unwrap();
        assert_eq!(filter.center_frequency(), 50.0);
        assert_eq!(filter.bandwidth(), 2.0);
        assert_eq!(filter.section_count(), 1);
    }

    #[test]
    fn test_powerline_harmonics() {
        let filter = NotchFilter::powerline_harmonics(50.0, 2.0, 1000.0, 3);
        assert!(filter.is_ok());

        let filter = filter.unwrap();
        assert_eq!(filter.section_count(), 3); // 50, 100, 150 Hz
    }

    #[test]
    fn test_notch_processing() {
        let mut filter = NotchFilter::new(50.0, 2.0, 1000.0).unwrap();

        // Test with DC (should pass through)
        let dc_output = filter.process_sample(1.0);
        assert!((dc_output - 1.0).abs() < 0.1);

        filter.reset();

        // Test processing doesn't crash
        for _ in 0..100 {
            filter.process_sample(0.5 * (50.0 * 2.0 * std::f32::consts::PI * 0.001).sin());
        }
    }

    #[test]
    fn test_invalid_parameters() {
        assert!(NotchFilter::new(0.0, 2.0, 1000.0).is_err()); // Invalid frequency
        assert!(NotchFilter::new(600.0, 2.0, 1000.0).is_err()); // Frequency too high
        assert!(NotchFilter::new(50.0, 0.0, 1000.0).is_err()); // Invalid bandwidth
        assert!(NotchFilter::new(50.0, 100.0, 1000.0).is_err()); // Bandwidth too wide
    }
}