// src/processing/pipeline.rs
//! Complete signal processing pipeline

use crate::hal::{EmgSample, QualityMetrics};
use crate::config::processing_config::ProcessingConfig;
use crate::processing::{FilterBank, QualityMonitor, WindowManager, WindowAnalysis};
use crate::acquisition::ProcessedSample;
use std::time::Instant;

/// Complete processing pipeline for EMG signals
pub struct ProcessingPipeline {
    filter_banks: Vec<FilterBank>,
    quality_monitor: QualityMonitor,
    window_manager: WindowManager,
    config: ProcessingConfig,
    sample_rate: f32,
    channel_count: usize,
    is_initialized: bool,
}

/// Processing pipeline error types
#[derive(Debug)]
pub enum PipelineError {
    NotInitialized,
    ConfigurationError(String),
    ProcessingError(String),
    ChannelMismatch(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::NotInitialized => write!(f, "Pipeline not initialized"),
            PipelineError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            PipelineError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
            PipelineError::ChannelMismatch(msg) => write!(f, "Channel mismatch: {}", msg),
        }
    }
}

impl std::error::Error for PipelineError {}

impl ProcessingPipeline {
    /// Create processing pipeline from configuration
    pub fn new(config: ProcessingConfig, sample_rate: f32, channel_count: usize) -> Result<Self, PipelineError> {
        if sample_rate <= 0.0 {
            return Err(PipelineError::ConfigurationError("Invalid sample rate".to_string()));
        }
        if channel_count == 0 {
            return Err(PipelineError::ConfigurationError("Invalid channel count".to_string()));
        }

        Ok(Self {
            filter_banks: Vec::new(),
            quality_monitor: QualityMonitor::new(config.quality_monitoring.clone()),
            window_manager: WindowManager::new(config.windowing.clone()),
            config,
            sample_rate,
            channel_count,
            is_initialized: false,
        })
    }

    /// Initialize pipeline with configuration
    pub fn initialize(&mut self) -> Result<(), PipelineError> {
        // Create filter bank for each channel
        self.filter_banks.clear();
        for channel_id in 0..self.channel_count {
            let filter_bank = FilterBank::from_config(
                self.config.filter_bank.clone(),
                self.sample_rate
            ).map_err(|e| PipelineError::ConfigurationError(format!("Channel {}: {:?}", channel_id, e)))?;

            self.filter_banks.push(filter_bank);
        }

        self.is_initialized = true;
        Ok(())
    }

    /// Process complete EMG sample through pipeline
    pub fn process_sample(&mut self, sample: EmgSample) -> Result<ProcessedSample, PipelineError> {
        if !self.is_initialized {
            return Err(PipelineError::NotInitialized);
        }

        if sample.channels.len() != self.channel_count {
            return Err(PipelineError::ChannelMismatch(
                format!("Expected {} channels, got {}", self.channel_count, sample.channels.len())
            ));
        }

        let processing_start = Instant::now();

        // Step 1: Apply filtering to each channel
        let mut filtered_channels = Vec::with_capacity(self.channel_count);
        for (channel_idx, &channel_value) in sample.channels.iter().enumerate() {
            let filtered_value = self.filter_banks[channel_idx].process_sample(channel_value);
            filtered_channels.push(filtered_value);
        }

        // Step 2: Quality assessment
        let quality_metrics = self.quality_monitor.assess_quality(&filtered_channels);

        // Step 3: Windowing (for feature extraction)
        self.window_manager.add_sample(filtered_channels.clone());

        let processing_latency = processing_start.elapsed().as_nanos() as u64;

        Ok(ProcessedSample {
            timestamp: sample.timestamp,
            sequence: sample.sequence,
            channels: filtered_channels,
            quality_metrics,
            processing_latency_ns: processing_latency,
        })
    }

    /// Process multiple samples in batch
    pub fn process_batch(&mut self, samples: Vec<EmgSample>) -> Result<Vec<ProcessedSample>, PipelineError> {
        let mut processed_samples = Vec::with_capacity(samples.len());

        for sample in samples {
            let processed = self.process_sample(sample)?;
            processed_samples.push(processed);
        }

        Ok(processed_samples)
    }

    /// Get windowed analysis if available
    pub fn get_window_analysis(&mut self) -> Option<WindowAnalysis> {
        self.window_manager.get_window()
    }

    /// Check if pipeline is healthy and processing correctly
    pub fn is_healthy(&self) -> bool {
        self.is_initialized &&
            self.quality_monitor.is_quality_acceptable() &&
            self.filter_banks.iter().all(|bank| bank.is_initialized())
    }

    /// Reset all pipeline components
    pub fn reset(&mut self) {
        for filter_bank in &mut self.filter_banks {
            filter_bank.reset();
        }
        self.quality_monitor.reset();
        self.window_manager.reset();
    }

    /// Get pipeline configuration
    pub fn config(&self) -> &ProcessingConfig {
        &self.config
    }

    /// Get pipeline statistics
    pub fn get_statistics(&self) -> PipelineStatistics {
        let quality_assessment = self.quality_monitor.get_assessment();

        PipelineStatistics {
            sample_rate: self.sample_rate,
            channel_count: self.channel_count,
            is_initialized: self.is_initialized,
            current_snr_db: quality_assessment.current_snr_db,
            average_snr_db: quality_assessment.average_snr_db,
            artifact_probability: quality_assessment.artifact_probability,
            overall_quality: quality_assessment.overall_quality,
            pending_windows: self.window_manager.pending_windows(),
        }
    }

    /// Update configuration (requires reinitialization)
    pub fn update_config(&mut self, new_config: ProcessingConfig) -> Result<(), PipelineError> {
        self.config = new_config;
        self.is_initialized = false;
        self.initialize()
    }
}

/// Pipeline performance statistics
#[derive(Debug, Clone)]
pub struct PipelineStatistics {
    pub sample_rate: f32,
    pub channel_count: usize,
    pub is_initialized: bool,
    pub current_snr_db: f32,
    pub average_snr_db: f32,
    pub artifact_probability: f32,
    pub overall_quality: crate::processing::quality_monitor::QualityLevel,
    pub pending_windows: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::processing_config::ProcessingConfig;

    #[test]
    fn test_pipeline_creation() {
        let config = ProcessingConfig::default();
        let pipeline = ProcessingPipeline::new(config, 2000.0, 8);
        assert!(pipeline.is_ok());

        let pipeline = pipeline.unwrap();
        assert!(!pipeline.is_initialized);
        assert_eq!(pipeline.channel_count, 8);
        assert_eq!(pipeline.sample_rate, 2000.0);
    }

    #[test]
    fn test_pipeline_initialization() {
        let config = ProcessingConfig::default();
        let mut pipeline = ProcessingPipeline::new(config, 2000.0, 4).unwrap();

        assert!(pipeline.initialize().is_ok());
        assert!(pipeline.is_initialized);
        assert_eq!(pipeline.filter_banks.len(), 4);
    }

    #[test]
    fn test_sample_processing() {
        let config = ProcessingConfig::default();
        let mut pipeline = ProcessingPipeline::new(config, 2000.0, 4).unwrap();
        pipeline.initialize().unwrap();

        let sample = EmgSample {
            timestamp: 1234567890,
            sequence: 1,
            channels: vec![0.1, 0.2, 0.3, 0.4],
            quality_indicators: QualityMetrics::default(),
        };

        let result = pipeline.process_sample(sample);
        assert!(result.is_ok());

        let processed = result.unwrap();
        assert_eq!(processed.channels.len(), 4);
        assert!(processed.processing_latency_ns > 0);
    }

    #[test]
    fn test_batch_processing() {
        let config = ProcessingConfig::default();
        let mut pipeline = ProcessingPipeline::new(config, 2000.0, 2).unwrap();
        pipeline.initialize().unwrap();

        let samples = vec![
            EmgSample {
                timestamp: 1,
                sequence: 1,
                channels: vec![0.1, 0.2],
                quality_indicators: QualityMetrics::default(),
            },
            EmgSample {
                timestamp: 2,
                sequence: 2,
                channels: vec![0.3, 0.4],
                quality_indicators: QualityMetrics::default(),
            },
        ];

        let result = pipeline.process_batch(samples);
        assert!(result.is_ok());

        let processed = result.unwrap();
        assert_eq!(processed.len(), 2);
    }

    #[test]
    fn test_pipeline_errors() {
        let config = ProcessingConfig::default();
        let mut pipeline = ProcessingPipeline::new(config, 2000.0, 4).unwrap();

        // Test uninitialized pipeline
        let sample = EmgSample {
            timestamp: 1,
            sequence: 1,
            channels: vec![0.1, 0.2, 0.3, 0.4],
            quality_indicators: QualityMetrics::default(),
        };
        assert!(pipeline.process_sample(sample).is_err());

        // Initialize and test channel mismatch
        pipeline.initialize().unwrap();
        let wrong_sample = EmgSample {
            timestamp: 1,
            sequence: 1,
            channels: vec![0.1, 0.2], // Wrong number of channels
            quality_indicators: QualityMetrics::default(),
        };
        assert!(pipeline.process_sample(wrong_sample).is_err());
    }

    #[test]
    fn test_pipeline_statistics() {
        let config = ProcessingConfig::default();
        let mut pipeline = ProcessingPipeline::new(config, 2000.0, 4).unwrap();
        pipeline.initialize().unwrap();

        let stats = pipeline.get_statistics();
        assert_eq!(stats.sample_rate, 2000.0);
        assert_eq!(stats.channel_count, 4);
        assert!(stats.is_initialized);
    }
}