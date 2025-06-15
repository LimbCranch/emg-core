//! Frequency domain feature extraction for EMG signals

use crate::error::EmgError;
use std::f32::consts::PI;

/// Frequency domain features extracted from EMG signals
#[derive(Debug, Clone)]
pub struct FrequencyDomainFeatures {
    pub spectral_centroid: Vec<f32>,        // Spectral centroid (center of mass)
    pub mean_frequency: Vec<f32>,           // Mean frequency
    pub median_frequency: Vec<f32>,         // Median frequency
    pub power_spectral_density: Vec<Vec<f32>>, // PSD for each channel
    pub frequency_band_powers: Vec<Vec<f32>>,  // Power in specific frequency bands
    pub spectral_rolloff: Vec<f32>,         // Frequency below which 85% of energy lies
    pub spectral_flux: Vec<f32>,            // Measure of spectral change
}

impl FrequencyDomainFeatures {
    pub fn new(channel_count: usize, fft_size: usize, band_count: usize) -> Self {
        Self {
            spectral_centroid: vec![0.0; channel_count],
            mean_frequency: vec![0.0; channel_count],
            median_frequency: vec![0.0; channel_count],
            power_spectral_density: vec![vec![0.0; fft_size / 2]; channel_count],
            frequency_band_powers: vec![vec![0.0; band_count]; channel_count],
            spectral_rolloff: vec![0.0; channel_count],
            spectral_flux: vec![0.0; channel_count],
        }
    }
}

/// Frequency domain feature extractor using FFT
pub struct FrequencyDomainExtractor {
    sample_rate: f32,
    fft_size: usize,
    frequency_bands: Vec<(f32, f32)>,
    window_function: Vec<f32>,
    previous_spectrum: Vec<Vec<f32>>, // For spectral flux calculation
}

impl FrequencyDomainExtractor {
    pub fn new(config: &super::FeatureConfig) -> Result<Self, EmgError> {
        let fft_size = config.window_size.next_power_of_two();
        let window_function = Self::create_hann_window(config.window_size);

        Ok(Self {
            sample_rate: config.sample_rate_hz as f32,
            fft_size,
            frequency_bands: config.frequency_bands.clone(),
            window_function,
            previous_spectrum: Vec::new(),
        })
    }

    /// Extract frequency domain features from a window of samples
    pub fn extract(&mut self, window: &[Vec<f32>]) -> Result<FrequencyDomainFeatures, EmgError> {
        if window.is_empty() {
            return Err(EmgError::Processing("Empty window provided".to_string()));
        }

        let channel_count = window[0].len();
        let mut features = FrequencyDomainFeatures::new(
            channel_count,
            self.fft_size,
            self.frequency_bands.len()
        );

        // Initialize previous spectrum if needed
        if self.previous_spectrum.len() != channel_count {
            self.previous_spectrum = vec![vec![0.0; self.fft_size / 2]; channel_count];
        }

        for ch in 0..channel_count {
            // Extract and window the channel data
            let mut channel_data: Vec<f32> = window.iter()
                .enumerate()
                .map(|(i, sample)| {
                    let window_val = if i < self.window_function.len() {
                        self.window_function[i]
                    } else {
                        0.0
                    };
                    sample[ch] * window_val
                })
                .collect();

            // Zero-pad to FFT size
            channel_data.resize(self.fft_size, 0.0);

            // Compute FFT
            let spectrum = self.compute_fft(&channel_data)?;
            let magnitude_spectrum = self.compute_magnitude_spectrum(&spectrum);
            let power_spectrum = self.compute_power_spectrum(&magnitude_spectrum);

            // Store PSD
            features.power_spectral_density[ch] = power_spectrum[..self.fft_size / 2].to_vec();

            // Calculate frequency features
            features.spectral_centroid[ch] = self.calculate_spectral_centroid(&power_spectrum);
            features.mean_frequency[ch] = self.calculate_mean_frequency(&power_spectrum);
            features.median_frequency[ch] = self.calculate_median_frequency(&power_spectrum);
            features.spectral_rolloff[ch] = self.calculate_spectral_rolloff(&power_spectrum, 0.85);
            features.spectral_flux[ch] = self.calculate_spectral_flux(
                &power_spectrum[..self.fft_size / 2],
                &self.previous_spectrum[ch]
            );

            // Calculate frequency band powers
            features.frequency_band_powers[ch] = self.calculate_frequency_band_powers(&power_spectrum);

            // Update previous spectrum for flux calculation
            self.previous_spectrum[ch] = power_spectrum[..self.fft_size / 2].to_vec();
        }

        Ok(features)
    }

    pub fn reset(&mut self) {
        self.previous_spectrum.clear();
    }

    // Private calculation methods
    fn create_hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size - 1) as f32).cos()))
            .collect()
    }

    fn compute_fft(&self, data: &[f32]) -> Result<Vec<f32>, EmgError> {
        // Simple DFT implementation for now
        // In production, use a proper FFT library like rustfft
        let n = data.len();
        let mut result = vec![0.0; n * 2]; // Real and imaginary parts interleaved

        for k in 0..n / 2 {
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for j in 0..n {
                let angle = -2.0 * PI * (k * j) as f32 / n as f32;
                real_sum += data[j] * angle.cos();
                imag_sum += data[j] * angle.sin();
            }

            result[k * 2] = real_sum;
            result[k * 2 + 1] = imag_sum;
        }

        Ok(result)
    }

    fn compute_magnitude_spectrum(&self, fft_result: &[f32]) -> Vec<f32> {
        fft_result
            .chunks(2)
            .map(|complex| (complex[0].powi(2) + complex[1].powi(2)).sqrt())
            .collect()
    }

    fn compute_power_spectrum(&self, magnitude_spectrum: &[f32]) -> Vec<f32> {
        magnitude_spectrum
            .iter()
            .map(|&mag| mag.powi(2))
            .collect()
    }

    fn calculate_spectral_centroid(&self, power_spectrum: &[f32]) -> f32 {
        let total_power: f32 = power_spectrum.iter().sum();
        if total_power == 0.0 {
            return 0.0;
        }

        let weighted_sum: f32 = power_spectrum
            .iter()
            .enumerate()
            .map(|(i, &power)| {
                let freq = i as f32 * self.sample_rate / (2.0 * power_spectrum.len() as f32);
                freq * power
            })
            .sum();

        weighted_sum / total_power
    }

    fn calculate_mean_frequency(&self, power_spectrum: &[f32]) -> f32 {
        self.calculate_spectral_centroid(power_spectrum)
    }

    fn calculate_median_frequency(&self, power_spectrum: &[f32]) -> f32 {
        let total_power: f32 = power_spectrum.iter().sum();
        if total_power == 0.0 {
            return 0.0;
        }

        let half_power = total_power / 2.0;
        let mut cumulative_power = 0.0;

        for (i, &power) in power_spectrum.iter().enumerate() {
            cumulative_power += power;
            if cumulative_power >= half_power {
                return i as f32 * self.sample_rate / (2.0 * power_spectrum.len() as f32);
            }
        }

        // Fallback
        self.sample_rate / 4.0
    }

    fn calculate_spectral_rolloff(&self, power_spectrum: &[f32], threshold: f32) -> f32 {
        let total_power: f32 = power_spectrum.iter().sum();
        if total_power == 0.0 {
            return 0.0;
        }

        let target_power = threshold * total_power;
        let mut cumulative_power = 0.0;

        for (i, &power) in power_spectrum.iter().enumerate() {
            cumulative_power += power;
            if cumulative_power >= target_power {
                return i as f32 * self.sample_rate / (2.0 * power_spectrum.len() as f32);
            }
        }

        self.sample_rate / 2.0
    }

    fn calculate_spectral_flux(&self, current_spectrum: &[f32], previous_spectrum: &[f32]) -> f32 {
        if previous_spectrum.len() != current_spectrum.len() {
            return 0.0;
        }

        current_spectrum
            .iter()
            .zip(previous_spectrum.iter())
            .map(|(&curr, &prev)| (curr - prev).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    fn calculate_frequency_band_powers(&self, power_spectrum: &[f32]) -> Vec<f32> {
        let mut band_powers = Vec::with_capacity(self.frequency_bands.len());

        for &(low_freq, high_freq) in &self.frequency_bands {
            let low_bin = (low_freq * power_spectrum.len() as f32 * 2.0 / self.sample_rate) as usize;
            let high_bin = (high_freq * power_spectrum.len() as f32 * 2.0 / self.sample_rate) as usize;

            let band_power = power_spectrum
                .get(low_bin..high_bin.min(power_spectrum.len()))
                .unwrap_or(&[])
                .iter()
                .sum();

            band_powers.push(band_power);
        }

        band_powers
    }
}
