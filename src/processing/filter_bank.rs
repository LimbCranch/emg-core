// src/processing/filter_bank.rs
//! Filter bank combining multiple filters

use crate::config::processing_config::{FilterBankConfig};
use crate::processing::filters::{IirFilter, NotchFilter, FilterError, BandType};

/// Filter bank with multiple stages
pub struct FilterBank {
    highpass: Option<IirFilter>,
    lowpass: Option<IirFilter>,
    notch_filters: Vec<NotchFilter>,
    config: FilterBankConfig,
    is_initialized: bool,
}

impl FilterBank {
    /// Create filter bank from configuration
    pub fn from_config(config: FilterBankConfig, sample_rate: f32) -> Result<Self, FilterError> {
        let mut bank = Self {
            highpass: None,
            lowpass: None,
            notch_filters: Vec::new(),
            config: config.clone(),
            is_initialized: false,
        };

        // Create highpass filter
        if config.highpass_cutoff_hz > 0.0 {
            let highpass = IirFilter::butterworth(
                config.filter_order,
                config.highpass_cutoff_hz,
                sample_rate,
                BandType::Highpass
            )?;
            bank.highpass = Some(highpass);
        }

        // Create lowpass filter
        if config.lowpass_cutoff_hz > 0.0 && config.lowpass_cutoff_hz < sample_rate / 2.0 {
            let lowpass = IirFilter::butterworth(
                config.filter_order,
                config.lowpass_cutoff_hz,
                sample_rate,
                BandType::Lowpass
            )?;
            bank.lowpass = Some(lowpass);
        }

        // Create notch filters
        for &freq in &config.notch_filters.frequencies_hz {
            if freq > 0.0 && freq < sample_rate / 2.0 {
                let notch = NotchFilter::new(freq, config.notch_filters.bandwidth_hz, sample_rate)?;
                bank.notch_filters.push(notch);
            }
        }

        bank.is_initialized = true;
        Ok(bank)
    }

    /// Process single sample through filter bank
    pub fn process_sample(&mut self, sample: f32) -> f32 {
        if !self.is_initialized {
            return sample;
        }

        let mut output = sample;

        // Apply highpass filter first
        if let Some(ref mut highpass) = self.highpass {
            output = highpass.process_sample(output);
        }

        // Apply lowpass filter
        if let Some(ref mut lowpass) = self.lowpass {
            output = lowpass.process_sample(output);
        }

        // Apply all notch filters
        for notch in &mut self.notch_filters {
            output = notch.process_sample(output);
        }

        output
    }

    /// Process multiple channels
    pub fn process_channels(&mut self, samples: &[f32]) -> Vec<f32> {
        samples.iter()
            .map(|&sample| self.process_sample(sample))
            .collect()
    }

    /// Reset all filter states
    pub fn reset(&mut self) {
        if let Some(ref mut highpass) = self.highpass {
            highpass.reset();
        }
        if let Some(ref mut lowpass) = self.lowpass {
            lowpass.reset();
        }
        for notch in &mut self.notch_filters {
            notch.reset();
        }
    }

    /// Get configuration
    pub fn config(&self) -> &FilterBankConfig {
        &self.config
    }

    /// Check if initialized
    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    /// Get filter counts
    pub fn filter_counts(&self) -> (usize, usize, usize) {
        (
            if self.highpass.is_some() { 1 } else { 0 },
            if self.lowpass.is_some() { 1 } else { 0 },
            self.notch_filters.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::processing_config::{FilterBankConfig, NotchFilterConfig, FilterType};

    #[test]
    fn test_filter_bank_creation() {
        let config = FilterBankConfig {
            highpass_cutoff_hz: 20.0,
            lowpass_cutoff_hz: 500.0,
            filter_order: 4,
            filter_type: FilterType::Butterworth,
            notch_filters: NotchFilterConfig {
                frequencies_hz: vec![50.0, 60.0],
                bandwidth_hz: 2.0,
            },
        };

        let bank = FilterBank::from_config(config, 2000.0);
        assert!(bank.is_ok());

        let bank = bank.unwrap();
        assert!(bank.is_initialized());

        let (hp_count, lp_count, notch_count) = bank.filter_counts();
        assert_eq!(hp_count, 1);
        assert_eq!(lp_count, 1);
        assert_eq!(notch_count, 2);
    }

    #[test]
    fn test_filter_bank_processing() {
        let config = FilterBankConfig {
            highpass_cutoff_hz: 20.0,
            lowpass_cutoff_hz: 500.0,
            filter_order: 2,
            filter_type: FilterType::Butterworth,
            notch_filters: NotchFilterConfig {
                frequencies_hz: vec![50.0],
                bandwidth_hz: 2.0,
            },
        };

        let mut bank = FilterBank::from_config(config, 2000.0).unwrap();

        // Test single sample
        let output = bank.process_sample(1.0);
        assert!(output.is_finite());

        // Test multi-channel
        let inputs = vec![0.1, 0.2, 0.3, 0.4];
        let outputs = bank.process_channels(&inputs);
        assert_eq!(outputs.len(), 4);
        assert!(outputs.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_filter_bank_reset() {
        let config = FilterBankConfig::default();
        let mut bank = FilterBank::from_config(config, 2000.0).unwrap();

        // Process some samples
        for _ in 0..10 {
            bank.process_sample(1.0);
        }

        // Reset shouldn't crash
        bank.reset();

        // Should still work after reset
        let output = bank.process_sample(1.0);
        assert!(output.is_finite());
    }
}