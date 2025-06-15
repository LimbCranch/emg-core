//! Time domain feature extraction for EMG signals

use crate::error::{EmgError, ErrorContext};
use crate::config::processing::TIME_DOMAIN_FEATURE_COUNT;

/// Time domain features extracted from EMG signals
#[derive(Debug, Clone)]
pub struct TimeDomainFeatures {
    pub rms: Vec<f32>,                    // Root Mean Square
    pub mean_absolute_value: Vec<f32>,    // Mean Absolute Value (MAV)
    pub zero_crossings: Vec<u32>,         // Zero Crossing Rate
    pub slope_sign_changes: Vec<u32>,     // Slope Sign Changes
    pub waveform_length: Vec<f32>,        // Waveform Length
    pub variance: Vec<f32>,               // Signal Variance
    pub skewness: Vec<f32>,               // Third statistical moment
    pub kurtosis: Vec<f32>,               // Fourth statistical moment
}

impl TimeDomainFeatures {
    pub fn new(channel_count: usize) -> Self {
        Self {
            rms: vec![0.0; channel_count],
            mean_absolute_value: vec![0.0; channel_count],
            zero_crossings: vec![0; channel_count],
            slope_sign_changes: vec![0; channel_count],
            waveform_length: vec![0.0; channel_count],
            variance: vec![0.0; channel_count],
            skewness: vec![0.0; channel_count],
            kurtosis: vec![0.0; channel_count],
        }
    }
}

/// Time domain feature extractor
pub struct TimeDomainExtractor {
    threshold: f32,  // Threshold for zero crossing detection
}

impl TimeDomainExtractor {
    pub fn new(_config: &super::FeatureConfig) -> Result<Self, EmgError> {
        Ok(Self {
            threshold: 0.001, // Small threshold to avoid noise triggering zero crossings
        })
    }

    /// Extract time domain features from a window of samples
    pub fn extract(&mut self, window: &[Vec<f32>]) -> Result<TimeDomainFeatures, EmgError> {
        if window.is_empty() {
            return Err(EmgError::Timing {
                expected_timing: None,
                actual_timing: None,
                drift_ns: None,
                reason: "Empty window provided".to_string(),
                context: ErrorContext {
                    timestamp: (std::time::SystemTime::now()),
                    thread_id: None,
                    component: "".to_string(),
                    operation: "".to_string(),
                    file: None,
                    line: None,
                    additional_info: Default::default(),
                    chain: vec![],
                },
            });

            //return Err(EmgError:: Processing("Empty window provided".to_string()));
        }

        let channel_count = window[0].len();
        let window_size = window.len();
        let mut features = TimeDomainFeatures::new(channel_count);

        for ch in 0..channel_count {
            // Extract channel data
            let channel_data: Vec<f32> = window.iter()
                .map(|sample| sample[ch])
                .collect();

            // Calculate basic statistics
            let mean = self.calculate_mean(&channel_data);
            let variance = self.calculate_variance(&channel_data, mean);
            let std_dev = variance.sqrt();

            // Root Mean Square (RMS)
            features.rms[ch] = self.calculate_rms(&channel_data);

            // Mean Absolute Value (MAV)
            features.mean_absolute_value[ch] = self.calculate_mav(&channel_data);

            // Zero Crossings (ZC)
            features.zero_crossings[ch] = self.calculate_zero_crossings(&channel_data);

            // Slope Sign Changes (SSC)
            features.slope_sign_changes[ch] = self.calculate_slope_sign_changes(&channel_data);

            // Waveform Length (WL)
            features.waveform_length[ch] = self.calculate_waveform_length(&channel_data);

            // Statistical moments
            features.variance[ch] = variance;
            features.skewness[ch] = self.calculate_skewness(&channel_data, mean, std_dev);
            features.kurtosis[ch] = self.calculate_kurtosis(&channel_data, mean, std_dev);
        }

        Ok(features)
    }

    pub fn reset(&mut self) {
        // No state to reset for time domain features
    }

    // Private calculation methods
    fn calculate_mean(&self, data: &[f32]) -> f32 {
        data.iter().sum::<f32>() / data.len() as f32
    }

    fn calculate_variance(&self, data: &[f32], mean: f32) -> f32 {
        data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / data.len() as f32
    }

    fn calculate_rms(&self, data: &[f32]) -> f32 {
        let sum_squares: f32 = data.iter().map(|&x| x * x).sum();
        (sum_squares / data.len() as f32).sqrt()
    }

    fn calculate_mav(&self, data: &[f32]) -> f32 {
        data.iter().map(|&x| x.abs()).sum::<f32>() / data.len() as f32
    }

    fn calculate_zero_crossings(&self, data: &[f32]) -> u32 {
        let mut count = 0;
        for window in data.windows(2) {
            let [curr, next] = [window[0], window[1]];
            if (curr >= self.threshold && next <= -self.threshold) ||
                (curr <= -self.threshold && next >= self.threshold) {
                count += 1;
            }
        }
        count
    }

    fn calculate_slope_sign_changes(&self, data: &[f32]) -> u32 {
        if data.len() < 3 {
            return 0;
        }

        let mut count = 0;
        for i in 1..data.len() - 1 {
            let prev_slope = data[i] - data[i - 1];
            let next_slope = data[i + 1] - data[i];

            if (prev_slope > self.threshold && next_slope < -self.threshold) ||
                (prev_slope < -self.threshold && next_slope > self.threshold) {
                count += 1;
            }
        }
        count
    }

    fn calculate_waveform_length(&self, data: &[f32]) -> f32 {
        data.windows(2)
            .map(|window| (window[1] - window[0]).abs())
            .sum()
    }

    fn calculate_skewness(&self, data: &[f32], mean: f32, std_dev: f32) -> f32 {
        if std_dev == 0.0 {
            return 0.0;
        }

        let n = data.len() as f32;
        let sum_cubed = data.iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum::<f32>();

        (n / ((n - 1.0) * (n - 2.0))) * sum_cubed
    }

    fn calculate_kurtosis(&self, data: &[f32], mean: f32, std_dev: f32) -> f32 {
        if std_dev == 0.0 {
            return 0.0;
        }

        let n = data.len() as f32;
        let sum_fourth = data.iter()
            .map(|&x| ((x - mean) / std_dev).powi(4))
            .sum::<f32>();

        (n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0))) * sum_fourth -
            (3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0)))
    }
}
