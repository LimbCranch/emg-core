// src/processing/windowing.rs
//! Windowing functions for signal processing

use crate::config::processing_config::{WindowingConfig, WindowType};
use std::collections::VecDeque;

/// Window manager for signal segmentation
pub struct WindowManager {
    config: WindowingConfig,
    buffer: VecDeque<Vec<f32>>,
    window_function: Vec<f32>,
    hop_size: usize,
    ready_windows: VecDeque<Vec<f32>>,
}

/// Window analysis results
#[derive(Debug, Clone)]
pub struct WindowAnalysis {
    pub windowed_data: Vec<f32>,
    pub window_energy: f32,
    pub peak_amplitude: f32,
    pub rms_amplitude: f32,
}

impl WindowManager {
    /// Create window manager from configuration
    pub fn new(config: WindowingConfig) -> Self {
        let hop_size = ((1.0 - config.overlap_percent / 100.0) * config.window_size_samples as f32) as usize;
        let window_function = Self::generate_window_function(&config.window_type, config.window_size_samples);

        Self {
            config,
            buffer: VecDeque::new(),
            window_function,
            hop_size,
            ready_windows: VecDeque::new(),
        }
    }

    /// Add sample to windowing buffer
    pub fn add_sample(&mut self, channels: Vec<f32>) {
        self.buffer.push_back(channels);

        // Check if we can create a new window
        if self.buffer.len() >= self.config.window_size_samples {
            self.create_window();
        }
    }

    /// Get next ready window if available
    pub fn get_window(&mut self) -> Option<WindowAnalysis> {
        self.ready_windows.pop_front().map(|windowed_data| {
            let window_energy = windowed_data.iter().map(|&x| x * x).sum::<f32>();
            let peak_amplitude = windowed_data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
            let rms_amplitude = (window_energy / windowed_data.len() as f32).sqrt();

            WindowAnalysis {
                windowed_data,
                window_energy,
                peak_amplitude,
                rms_amplitude,
            }
        })
    }

    /// Check if windows are available
    pub fn has_windows(&self) -> bool {
        !self.ready_windows.is_empty()
    }

    /// Get number of pending windows
    pub fn pending_windows(&self) -> usize {
        self.ready_windows.len()
    }

    /// Reset windowing state
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.ready_windows.clear();
    }

    /// Get configuration
    pub fn config(&self) -> &WindowingConfig {
        &self.config
    }

    fn create_window(&mut self) {
        if self.buffer.len() < self.config.window_size_samples {
            return;
        }

        // Extract window data (assuming single channel for now)
        let mut window_data = Vec::with_capacity(self.config.window_size_samples);
        for i in 0..self.config.window_size_samples {
            if let Some(sample) = self.buffer.get(i) {
                // Use first channel or average all channels
                let value = if sample.is_empty() {
                    0.0
                } else if sample.len() == 1 {
                    sample[0]
                } else {
                    sample.iter().sum::<f32>() / sample.len() as f32
                };
                window_data.push(value);
            }
        }

        // Apply window function
        let windowed_data: Vec<f32> = window_data.iter()
            .zip(self.window_function.iter())
            .map(|(&sample, &window)| sample * window)
            .collect();

        self.ready_windows.push_back(windowed_data);

        // Remove samples based on hop size
        for _ in 0..self.hop_size {
            self.buffer.pop_front();
        }
    }

    fn generate_window_function(window_type: &WindowType, size: usize) -> Vec<f32> {
        match window_type {
            WindowType::Rectangular => vec![1.0; size],
            WindowType::Hamming => {
                (0..size).map(|i| {
                    0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos()
                }).collect()
            },
            WindowType::Hanning => {
                (0..size).map(|i| {
                    0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos())
                }).collect()
            },
            WindowType::Blackman => {
                (0..size).map(|i| {
                    let n = i as f32 / (size - 1) as f32;
                    0.42 - 0.5 * (2.0 * std::f32::consts::PI * n).cos()
                        + 0.08 * (4.0 * std::f32::consts::PI * n).cos()
                }).collect()
            },
            WindowType::Kaiser => {
                // Simplified Kaiser window (beta = 5.0)
                let beta = 5.0;
                let i0_beta = Self::modified_bessel_i0(beta);
                (0..size).map(|i| {
                    let n = 2.0 * i as f32 / (size - 1) as f32 - 1.0;
                    let arg = beta * (1.0 - n * n).sqrt();
                    Self::modified_bessel_i0(arg) / i0_beta
                }).collect()
            }
        }
    }

    // Approximation of modified Bessel function I0
    fn modified_bessel_i0(x: f32) -> f32 {
        let ax = x.abs();
        if ax < 3.75 {
            let y = (x / 3.75).powi(2);
            1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
                + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
        } else {
            let y = 3.75 / ax;
            (ax.exp() / ax.sqrt()) * (0.39894228 + y * (0.01328592
                + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281
                + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633
                + y * 0.00392377))))))))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_window_manager_creation() {
        let config = WindowingConfig::default();
        let manager = WindowManager::new(config);
        assert_eq!(manager.config.window_size_samples, 256);
        assert!(!manager.has_windows());
    }

    #[test]
    fn test_window_functions() {
        let hamming = WindowManager::generate_window_function(&WindowType::Hamming, 10);
        assert_eq!(hamming.len(), 10);
        assert!((hamming[0] - 0.08).abs() < 0.01); // Hamming starts near 0.08

        let hanning = WindowManager::generate_window_function(&WindowType::Hanning, 10);
        assert_eq!(hanning.len(), 10);
        assert!(hanning[0] < 0.01); // Hanning starts at 0

        let rect = WindowManager::generate_window_function(&WindowType::Rectangular, 5);
        assert!(rect.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_windowing_process() {
        let config = WindowingConfig {
            window_size_samples: 4,
            overlap_percent: 50.0,
            window_type: WindowType::Rectangular,
        };

        let mut manager = WindowManager::new(config);
        assert_eq!(manager.hop_size, 2); // 50% overlap

        // Add samples
        for i in 0..6 {
            manager.add_sample(vec![i as f32]);
        }

        // Should have at least one window ready
        assert!(manager.has_windows());

        let window = manager.get_window().unwrap();
        assert_eq!(window.windowed_data.len(), 4);
        assert!(window.window_energy > 0.0);
        assert!(window.rms_amplitude > 0.0);
    }

    #[test]
    fn test_window_analysis() {
        let config = WindowingConfig {
            window_size_samples: 4,
            overlap_percent: 0.0,
            window_type: WindowType::Rectangular,
        };

        let mut manager = WindowManager::new(config);

        // Add known pattern
        for &val in &[1.0, 2.0, 3.0, 4.0] {
            manager.add_sample(vec![val]);
        }

        let analysis = manager.get_window().unwrap();
        assert_eq!(analysis.peak_amplitude, 4.0);
        assert!((analysis.rms_amplitude - (30.0f32 / 4.0).sqrt()).abs() < 0.01);
    }
}