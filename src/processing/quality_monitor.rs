// src/processing/quality_monitor.rs
//! Signal quality monitoring for EMG processing

use crate::config::processing_config::QualityConfig;
use crate::hal::QualityMetrics;
use std::collections::VecDeque;

/// Signal quality monitor
pub struct QualityMonitor {
    config: QualityConfig,
    sample_history: VecDeque<Vec<f32>>,
    quality_history: VecDeque<QualityMetrics>,
    window_size: usize,
}

/// Real-time quality assessment
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    pub current_snr_db: f32,
    pub average_snr_db: f32,
    pub artifact_probability: f32,
    pub contact_quality: Vec<ContactQuality>,
    pub overall_quality: QualityLevel,
}

/// Per-channel contact quality
#[derive(Debug, Clone)]
pub struct ContactQuality {
    pub channel_id: usize,
    pub impedance_kohm: f32,
    pub signal_strength: f32,
    pub quality_level: QualityLevel,
}

/// Quality levels
#[derive(Debug, Clone, PartialEq)]
pub enum QualityLevel {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

impl QualityMonitor {
    /// Create quality monitor with configuration
    pub fn new(config: QualityConfig) -> Self {
        Self {
            config,
            sample_history: VecDeque::new(),
            quality_history: VecDeque::new(),
            window_size: 100, // Keep last 100 samples for analysis
        }
    }

    /// Assess quality of current sample
    pub fn assess_quality(&mut self, channels: &[f32]) -> QualityMetrics {
        // Calculate signal metrics
        let snr_db = self.calculate_snr(channels);
        let contact_impedance = self.estimate_contact_impedance(channels);
        let artifact_detected = self.detect_artifacts(channels);
        let signal_saturation = self.check_saturation(channels);

        let metrics = QualityMetrics {
            snr_db,
            contact_impedance_kohm: contact_impedance,
            artifact_detected,
            signal_saturation,
        };

        // Update history
        self.update_history(channels.to_vec(), metrics.clone());

        metrics
    }

    /// Get comprehensive quality assessment
    pub fn get_assessment(&self) -> QualityAssessment {
        let current_metrics = self.quality_history.back()
            .cloned()
            .unwrap_or_default();

        let average_snr = if !self.quality_history.is_empty() {
            self.quality_history.iter()
                .map(|m| m.snr_db)
                .sum::<f32>() / self.quality_history.len() as f32
        } else {
            current_metrics.snr_db
        };

        let artifact_probability = self.calculate_artifact_probability();
        let contact_quality = self.assess_contact_quality(&current_metrics);
        let overall_quality = self.determine_overall_quality(&current_metrics, average_snr);

        QualityAssessment {
            current_snr_db: current_metrics.snr_db,
            average_snr_db: average_snr,
            artifact_probability,
            contact_quality,
            overall_quality,
        }
    }

    /// Check if quality is acceptable
    pub fn is_quality_acceptable(&self) -> bool {
        if let Some(latest) = self.quality_history.back() {
            latest.snr_db >= self.config.snr_threshold_db &&
                !latest.signal_saturation &&
                latest.contact_impedance_kohm.iter().all(|&imp| imp <= self.config.contact_impedance_max_kohm)
        } else {
            false
        }
    }

    /// Reset quality monitor
    pub fn reset(&mut self) {
        self.sample_history.clear();
        self.quality_history.clear();
    }

    fn calculate_snr(&self, channels: &[f32]) -> f32 {
        if channels.is_empty() {
            return 0.0;
        }

        // Calculate signal power (RMS)
        let signal_power = channels.iter()
            .map(|&x| x * x)
            .sum::<f32>() / channels.len() as f32;

        // Estimate noise floor from recent samples
        let noise_power = if self.sample_history.len() >= 10 {
            let recent_samples: Vec<f32> = self.sample_history
                .iter()
                .flatten()
                .cloned()
                .collect();

            let mean = recent_samples.iter().sum::<f32>() / recent_samples.len() as f32;
            recent_samples.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / recent_samples.len() as f32
        } else {
            0.01 // Default noise floor
        };

        if noise_power > 0.0 {
            10.0 * (signal_power / noise_power).log10()
        } else {
            60.0 // Very high SNR if no noise
        }
    }

    fn estimate_contact_impedance(&self, channels: &[f32]) -> Vec<f32> {
        // Simplified impedance estimation based on signal amplitude
        channels.iter()
            .map(|&signal| {
                let amplitude = signal.abs();
                if amplitude > 0.5 {
                    5.0  // Good contact
                } else if amplitude > 0.1 {
                    15.0 // Fair contact
                } else {
                    50.0 // Poor contact
                }
            })
            .collect()
    }

    fn detect_artifacts(&self, channels: &[f32]) -> bool {
        if !self.config.artifact_detection_enabled {
            return false;
        }

        // Detect sudden amplitude changes
        if let Some(previous) = self.sample_history.back() {
            if previous.len() == channels.len() {
                for (i, &current) in channels.iter().enumerate() {
                    let change = (current - previous[i]).abs();
                    if change > 2.0 { // Threshold for artifact detection
                        return true;
                    }
                }
            }
        }

        // Detect signal clipping/saturation as artifact
        channels.iter().any(|&x| x.abs() > 0.95)
    }

    fn check_saturation(&self, channels: &[f32]) -> bool {
        channels.iter().any(|&x| x.abs() >= self.config.saturation_threshold)
    }

    fn update_history(&mut self, channels: Vec<f32>, metrics: QualityMetrics) {
        self.sample_history.push_back(channels);
        self.quality_history.push_back(metrics);

        // Maintain window size
        while self.sample_history.len() > self.window_size {
            self.sample_history.pop_front();
        }
        while self.quality_history.len() > self.window_size {
            self.quality_history.pop_front();
        }
    }

    fn calculate_artifact_probability(&self) -> f32 {
        if self.quality_history.is_empty() {
            return 0.0;
        }

        let artifact_count = self.quality_history.iter()
            .filter(|m| m.artifact_detected)
            .count();

        artifact_count as f32 / self.quality_history.len() as f32
    }

    fn assess_contact_quality(&self, metrics: &QualityMetrics) -> Vec<ContactQuality> {
        metrics.contact_impedance_kohm.iter()
            .enumerate()
            .map(|(i, &impedance)| {
                let quality_level = if impedance <= 10.0 {
                    QualityLevel::Excellent
                } else if impedance <= 25.0 {
                    QualityLevel::Good
                } else if impedance <= 50.0 {
                    QualityLevel::Fair
                } else {
                    QualityLevel::Poor
                };

                ContactQuality {
                    channel_id: i,
                    impedance_kohm: impedance,
                    signal_strength: 1.0 / (1.0 + impedance / 10.0), // Normalized strength
                    quality_level,
                }
            })
            .collect()
    }

    fn determine_overall_quality(&self, current: &QualityMetrics, average_snr: f32) -> QualityLevel {
        if current.signal_saturation {
            return QualityLevel::Critical;
        }

        if current.snr_db >= 40.0 && average_snr >= 35.0 {
            QualityLevel::Excellent
        } else if current.snr_db >= 25.0 && average_snr >= 20.0 {
            QualityLevel::Good
        } else if current.snr_db >= 15.0 && average_snr >= 12.0 {
            QualityLevel::Fair
        } else if current.snr_db >= 5.0 {
            QualityLevel::Poor
        } else {
            QualityLevel::Critical
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_monitor_creation() {
        let config = QualityConfig::default();
        let monitor = QualityMonitor::new(config);
        assert_eq!(monitor.window_size, 100);
    }

    #[test]
    fn test_quality_assessment() {
        let config = QualityConfig::default();
        let mut monitor = QualityMonitor::new(config);

        let channels = vec![0.5, 0.3, 0.7, 0.2];
        let metrics = monitor.assess_quality(&channels);

        assert!(metrics.snr_db > 0.0);
        assert_eq!(metrics.contact_impedance_kohm.len(), 4);
        assert!(!metrics.signal_saturation);
    }

    #[test]
    fn test_saturation_detection() {
        let config = QualityConfig::default();
        let mut monitor = QualityMonitor::new(config);

        let saturated_channels = vec![0.99, 0.5, 0.3, 0.2];
        let metrics = monitor.assess_quality(&saturated_channels);

        assert!(metrics.signal_saturation);
    }

    #[test]
    fn test_artifact_detection() {
        let config = QualityConfig::default();
        let mut monitor = QualityMonitor::new(config);

        // First sample
        monitor.assess_quality(&vec![0.1, 0.1, 0.1, 0.1]);

        // Second sample with large change (artifact)
        let metrics = monitor.assess_quality(&vec![3.0, 0.1, 0.1, 0.1]);

        assert!(metrics.artifact_detected);
    }
}