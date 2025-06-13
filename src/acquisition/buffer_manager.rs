// src/acquisition/buffer_manager.rs
//! Buffer management for EMG signal acquisition

use crate::hal::{EmgSample, QualityMetrics};
use crate::acquisition::ring_buffer::{LockFreeRingBuffer, MpmcRingBuffer};
use crate::acquisition::sample_sync::SampleSynchronizer;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

/// Buffer configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BufferConfig {
    pub raw_buffer_size: Option<usize>,
    pub processed_buffer_size: Option<usize>,
    pub channel_count: usize,
    pub sample_rate_hz: u32,
    pub target_latency_ms: u32,
    pub enable_overflow_protection: bool,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            raw_buffer_size: None,
            processed_buffer_size: None,
            channel_count: 8,
            sample_rate_hz: 2000,
            target_latency_ms: 20,
            enable_overflow_protection: true,
        }
    }
}

/// Processed sample with additional metadata
#[derive(Debug, Clone)]
pub struct ProcessedSample {
    pub timestamp: u64,
    pub sequence: u32,
    pub channels: Vec<f32>,
    pub quality_metrics: QualityMetrics,
    pub processing_latency_ns: u64,
}

/// Buffer utilization metrics
#[derive(Debug, Clone)]
pub struct BufferMetrics {
    pub raw_utilization: f32,
    pub processed_utilization: f32,
    pub samples_processed: u64,
    pub samples_dropped: u64,
    pub average_latency_ns: u64,
    pub max_latency_ns: u64,
    pub underruns: u64,
    pub overruns: u64,
}

/// Main buffer manager coordinating all acquisition buffers
pub struct BufferManager {
    raw_buffer: Arc<Mutex<MpmcRingBuffer<EmgSample>>>,
    processed_buffer: Arc<Mutex<LockFreeRingBuffer<ProcessedSample>>>,
    synchronizer: Arc<Mutex<SampleSynchronizer>>,
    config: BufferConfig,
    metrics: BufferMetrics,

    // Atomic counters for metrics
    samples_processed: AtomicU64,
    samples_dropped: AtomicU64,
    underruns: AtomicU64,
    overruns: AtomicU64,
    latency_sum_ns: AtomicU64,
    latency_count: AtomicU32,
    max_latency_ns: AtomicU64,

    start_time: Instant,
}

/// Buffer manager error types
#[derive(Debug)]
pub enum BufferManagerError {
    ConfigurationError(String),
    BufferFull,
    BufferEmpty,
    SynchronizationError(String),
}

impl std::fmt::Display for BufferManagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferManagerError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            BufferManagerError::BufferFull => write!(f, "Buffer full"),
            BufferManagerError::BufferEmpty => write!(f, "Buffer empty"),
            BufferManagerError::SynchronizationError(msg) => write!(f, "Synchronization error: {}", msg),
        }
    }
}

impl std::error::Error for BufferManagerError {}

impl BufferManager {
    /// Create new buffer manager
    pub fn new(config: BufferConfig) -> Result<Self, BufferManagerError> {
        let raw_buffer_size = config.raw_buffer_size
            .unwrap_or_else(|| Self::calculate_raw_buffer_size(&config));
        let processed_buffer_size = config.processed_buffer_size
            .unwrap_or_else(|| Self::calculate_processed_buffer_size(&config));

        let raw_buffer = Arc::new(Mutex::new(
            MpmcRingBuffer::new(raw_buffer_size)
                .map_err(|e| BufferManagerError::ConfigurationError(format!("Raw buffer: {:?}", e)))?
        ));

        let processed_buffer = Arc::new(Mutex::new(
            LockFreeRingBuffer::new(processed_buffer_size)
                .map_err(|e| BufferManagerError::ConfigurationError(format!("Processed buffer: {:?}", e)))?
        ));

        let sync_config = crate::acquisition::sample_sync::SyncConfig {
            channel_count: config.channel_count,
            sample_rate_hz: config.sample_rate_hz,
            max_jitter_samples: 4,
            sync_timeout_ms: config.target_latency_ms / 2,
        };

        let synchronizer = Arc::new(Mutex::new(
            SampleSynchronizer::new(sync_config)
                .map_err(|e| BufferManagerError::ConfigurationError(e))?
        ));

        Ok(Self {
            raw_buffer,
            processed_buffer,
            synchronizer,
            config,
            metrics: BufferMetrics {
                raw_utilization: 0.0,
                processed_utilization: 0.0,
                samples_processed: 0,
                samples_dropped: 0,
                average_latency_ns: 0,
                max_latency_ns: 0,
                underruns: 0,
                overruns: 0,
            },
            samples_processed: AtomicU64::new(0),
            samples_dropped: AtomicU64::new(0),
            underruns: AtomicU64::new(0),
            overruns: AtomicU64::new(0),
            latency_sum_ns: AtomicU64::new(0),
            latency_count: AtomicU32::new(0),
            max_latency_ns: AtomicU64::new(0),
            start_time: Instant::now(),
        })
    }

    /// Add raw EMG sample to buffer
    pub fn add_raw_sample(&self, sample: EmgSample) -> Result<(), BufferManagerError> {
        let mut raw_buffer = self.raw_buffer.lock().unwrap();
        if self.config.enable_overflow_protection && raw_buffer.utilization() > 0.9 {
            self.overruns.fetch_add(1, Ordering::Relaxed);
            return Err(BufferManagerError::BufferFull);
        }

        match raw_buffer.try_push(sample) {
            Ok(()) => {
                self.samples_processed.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(_) => {
                self.samples_dropped.fetch_add(1, Ordering::Relaxed);
                Err(BufferManagerError::BufferFull)
            }
        }
    }

    /// Get next raw sample for processing
    pub fn get_raw_sample(&self) -> Option<EmgSample> {
        let raw_buffer = self.raw_buffer.lock().unwrap();
        match raw_buffer.try_pop() {
            Some(sample) => Some(sample),
            None => {
                self.underruns.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Add processed sample to output buffer
    pub fn add_processed_sample(&self, sample: ProcessedSample) -> Result<(), BufferManagerError> {
        // Update latency metrics
        self.update_latency_metrics(sample.processing_latency_ns);

        let mut processed_buffer = self.processed_buffer.lock().unwrap();
        match processed_buffer.try_push(sample) {
            Ok(()) => Ok(()),
            Err(_) => {
                self.samples_dropped.fetch_add(1, Ordering::Relaxed);
                Err(BufferManagerError::BufferFull)
            }
        }
    }

    /// Get processed sample for output
    pub fn get_processed_sample(&self) -> Option<ProcessedSample> {
        let processed_buffer = self.processed_buffer.lock().unwrap();
        processed_buffer.try_pop()
    }

    /// Add multi-channel sample via synchronizer
    pub fn add_synchronized_sample(&self, values: &[f32]) -> Result<(), BufferManagerError> {
        let mut synchronizer = self.synchronizer.lock().unwrap();
        synchronizer.add_multi_channel_sample(values)
            .map_err(BufferManagerError::SynchronizationError)?;

        // Try to get synchronized sample and add to raw buffer
        if let Some(sample) = synchronizer.try_get_synchronized_sample() {
            self.add_raw_sample(sample)?;
        }

        Ok(())
    }

    /// Get current buffer metrics
    pub fn get_metrics(&self) -> BufferMetrics {
        let samples_processed = self.samples_processed.load(Ordering::Relaxed);
        let samples_dropped = self.samples_dropped.load(Ordering::Relaxed);
        let latency_count = self.latency_count.load(Ordering::Relaxed);
        let latency_sum = self.latency_sum_ns.load(Ordering::Relaxed);
        let max_latency = self.max_latency_ns.load(Ordering::Relaxed);

        let average_latency_ns = if latency_count > 0 {
            latency_sum / latency_count as u64
        } else {
            0
        };

        BufferMetrics {
            raw_utilization: self.raw_buffer.lock().unwrap().utilization(),
            processed_utilization: self.processed_buffer.lock().unwrap().utilization(),
            samples_processed,
            samples_dropped,
            average_latency_ns,
            max_latency_ns: max_latency,
            underruns: self.underruns.load(Ordering::Relaxed),
            overruns: self.overruns.load(Ordering::Relaxed),
        }
    }

    /// Check if buffer manager is healthy
    pub fn is_healthy(&self) -> bool {
        let metrics = self.get_metrics();

        // Check utilization is reasonable
        let raw_ok = metrics.raw_utilization < 0.95;
        let processed_ok = metrics.processed_utilization < 0.95;

        // Check error rates
        let total_samples = metrics.samples_processed + metrics.samples_dropped;
        let error_rate = if total_samples > 0 {
            metrics.samples_dropped as f64 / total_samples as f64
        } else {
            0.0
        };
        let error_ok = error_rate < 0.01; // Less than 1% error rate

        // Check synchronizer health
        let sync_ok = self.synchronizer.lock().unwrap().is_healthy();

        raw_ok && processed_ok && error_ok && sync_ok
    }

    /// Reset all buffers and metrics
    pub fn reset(&self) {
        // Clear buffers
        let raw_buffer = self.raw_buffer.lock().unwrap();
        while raw_buffer.try_pop().is_some() {}
        let processed_buffer = self.processed_buffer.lock().unwrap();
        while processed_buffer.try_pop().is_some() {}
        let synchronizer = self.synchronizer.lock().unwrap();
        synchronizer.reset();

        // Reset metrics
        self.samples_processed.store(0, Ordering::Relaxed);
        self.samples_dropped.store(0, Ordering::Relaxed);
        self.underruns.store(0, Ordering::Relaxed);
        self.overruns.store(0, Ordering::Relaxed);
        self.latency_sum_ns.store(0, Ordering::Relaxed);
        self.latency_count.store(0, Ordering::Relaxed);
        self.max_latency_ns.store(0, Ordering::Relaxed);
    }

    /// Get raw buffer handle for direct access
    pub fn raw_buffer(&self) -> Arc<Mutex<MpmcRingBuffer<EmgSample>>> {
        self.raw_buffer.clone()
    }

    /// Get processed buffer handle for direct access
    pub fn processed_buffer(&self) -> Arc<Mutex<LockFreeRingBuffer<ProcessedSample>>> {
        self.processed_buffer.clone()
    }

    /// Get synchronizer handle
    pub fn synchronizer(&self) -> Arc<Mutex<SampleSynchronizer>> {
        self.synchronizer.clone()
    }

    /// Get configuration
    pub fn config(&self) -> &BufferConfig {
        &self.config
    }

    fn calculate_raw_buffer_size(config: &BufferConfig) -> usize {
        // Buffer size for target latency + safety margin
        let samples_per_ms = config.sample_rate_hz / 1000;
        let target_samples = samples_per_ms * config.target_latency_ms * 4; // 4x safety margin
        (target_samples as usize).next_power_of_two().max(1024)
    }

    fn calculate_processed_buffer_size(config: &BufferConfig) -> usize {
        // Smaller buffer for processed samples
        let samples_per_ms = config.sample_rate_hz / 1000;
        let target_samples = samples_per_ms * config.target_latency_ms * 2; // 2x safety margin
        (target_samples as usize).next_power_of_two().max(512)
    }

    fn update_latency_metrics(&self, latency_ns: u64) {
        self.latency_sum_ns.fetch_add(latency_ns, Ordering::Relaxed);
        self.latency_count.fetch_add(1, Ordering::Relaxed);

        // Update max latency
        let current_max = self.max_latency_ns.load(Ordering::Relaxed);
        if latency_ns > current_max {
            self.max_latency_ns.compare_exchange_weak(
                current_max,
                latency_ns,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ).ok(); // Ignore CAS failure
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hal::QualityMetrics;

    #[test]
    fn test_buffer_manager_creation() {
        let config = BufferConfig::default();
        let manager = BufferManager::new(config).unwrap();

        assert!(manager.is_healthy());

        let metrics = manager.get_metrics();
        assert_eq!(metrics.samples_processed, 0);
        assert_eq!(metrics.samples_dropped, 0);
    }

    #[test]
    fn test_raw_sample_workflow() {
        let config = BufferConfig::default();
        let manager = BufferManager::new(config).unwrap();

        let sample = EmgSample {
            timestamp: 1234567890,
            sequence: 1,
            channels: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            quality_indicators: QualityMetrics::default(),
        };

        assert!(manager.add_raw_sample(sample.clone()).is_ok());

        let retrieved = manager.get_raw_sample().unwrap();
        assert_eq!(retrieved.sequence, sample.sequence);
        assert_eq!(retrieved.channels, sample.channels);
    }

    #[test]
    fn test_processed_sample_workflow() {
        let config = BufferConfig::default();
        let manager = BufferManager::new(config).unwrap();

        let sample = ProcessedSample {
            timestamp: 1234567890,
            sequence: 1,
            channels: vec![0.1, 0.2, 0.3, 0.4],
            quality_metrics: QualityMetrics::default(),
            processing_latency_ns: 1000000, // 1ms
        };

        assert!(manager.add_processed_sample(sample.clone()).is_ok());

        let retrieved = manager.get_processed_sample().unwrap();
        assert_eq!(retrieved.sequence, sample.sequence);
        assert_eq!(retrieved.processing_latency_ns, sample.processing_latency_ns);
    }

    #[test]
    fn test_synchronized_samples() {
        let config = BufferConfig {
            channel_count: 4,
            ..Default::default()
        };
        let manager = BufferManager::new(config).unwrap();

        let values = [0.1, 0.2, 0.3, 0.4];
        assert!(manager.add_synchronized_sample(&values).is_ok());

        // Should be able to get the synchronized sample
        if let Some(sample) = manager.get_raw_sample() {
            assert_eq!(sample.channels.len(), 4);
        }
    }

    #[test]
    fn test_metrics_update() {
        let config = BufferConfig::default();
        let manager = BufferManager::new(config).unwrap();

        let sample = EmgSample {
            timestamp: 1234567890,
            sequence: 1,
            channels: vec![0.1; 8],
            quality_indicators: QualityMetrics::default(),
        };

        manager.add_raw_sample(sample).unwrap();

        let metrics = manager.get_metrics();
        assert_eq!(metrics.samples_processed, 1);
        assert_eq!(metrics.samples_dropped, 0);
        assert!(metrics.raw_utilization > 0.0);
    }

    #[test]
    fn test_buffer_reset() {
        let config = BufferConfig::default();
        let manager = BufferManager::new(config).unwrap();

        let sample = EmgSample {
            timestamp: 1234567890,
            sequence: 1,
            channels: vec![0.1; 8],
            quality_indicators: QualityMetrics::default(),
        };

        manager.add_raw_sample(sample).unwrap();
        assert!(manager.get_raw_sample().is_some());

        manager.reset();
        assert!(manager.get_raw_sample().is_none());

        let metrics = manager.get_metrics();
        assert_eq!(metrics.samples_processed, 0);
    }
}