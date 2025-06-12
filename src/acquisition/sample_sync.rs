// src/acquisition/sample_sync.rs
//! Multi-channel EMG sample synchronization

use crate::hal::{EmgSample, QualityMetrics};
use crate::acquisition::ring_buffer::LockFreeRingBuffer;
use std::sync::atomic::{AtomicU32, AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

/// Configuration for sample synchronization
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SyncConfig {
    pub channel_count: usize,
    pub sample_rate_hz: u32,
    pub max_jitter_samples: u32,
    pub sync_timeout_ms: u32,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            channel_count: 8,
            sample_rate_hz: 2000,
            max_jitter_samples: 4,
            sync_timeout_ms: 10,
        }
    }
}

/// Multi-channel sample synchronizer
pub struct SampleSynchronizer {
    channel_buffers: Vec<LockFreeRingBuffer<f32>>,
    timestamp_buffer: LockFreeRingBuffer<u64>,
    sequence_counter: AtomicU32,
    config: SyncConfig,
    is_synced: AtomicBool,
}

/// Single channel sample with timing
#[derive(Debug, Clone)]
pub struct ChannelSample {
    pub channel_id: usize,
    pub value: f32,
    pub timestamp: u64,
    pub sequence: u32,
}

/// Synchronization statistics
#[derive(Debug, Clone)]
pub struct SyncStats {
    pub samples_processed: u64,
    pub sync_errors: u64,
    pub max_jitter_samples: u32,
    pub average_jitter_ns: u64,
}

impl SampleSynchronizer {
    /// Create new sample synchronizer
    pub fn new(config: SyncConfig) -> Result<Self, String> {
        if config.channel_count == 0 {
            return Err("Channel count must be greater than 0".to_string());
        }

        let buffer_size = Self::calculate_buffer_size(&config);
        let mut channel_buffers = Vec::with_capacity(config.channel_count);

        for _ in 0..config.channel_count {
            channel_buffers.push(
                LockFreeRingBuffer::new(buffer_size)
                    .map_err(|e| format!("Failed to create channel buffer: {:?}", e))?
            );
        }

        let timestamp_buffer = LockFreeRingBuffer::new(buffer_size)
            .map_err(|e| format!("Failed to create timestamp buffer: {:?}", e))?;

        Ok(Self {
            channel_buffers,
            timestamp_buffer,
            sequence_counter: AtomicU32::new(0),
            config,
            is_synced: AtomicBool::new(false),
        })
    }

    /// Add sample for specific channel
    pub fn add_channel_sample(&self, channel_id: usize, value: f32) -> Result<(), String> {
        if channel_id >= self.config.channel_count {
            return Err(format!("Invalid channel ID: {}", channel_id));
        }

        let timestamp = self.get_current_timestamp();

        // Try to add to channel buffer
        if self.channel_buffers[channel_id].try_push(value).is_err() {
            return Err(format!("Channel {} buffer full", channel_id));
        }

        // Add timestamp
        if self.timestamp_buffer.try_push(timestamp).is_err() {
            return Err("Timestamp buffer full".to_string());
        }

        Ok(())
    }

    /// Try to get synchronized sample across all channels
    pub fn try_get_synchronized_sample(&self) -> Option<EmgSample> {
        // Check if all channels have data
        for buffer in &self.channel_buffers {
            if buffer.is_empty() {
                return None;
            }
        }

        let timestamp = self.timestamp_buffer.try_pop()?;
        let sequence = self.sequence_counter.fetch_add(1, Ordering::Relaxed);

        // Collect channel data
        let mut channels = Vec::with_capacity(self.config.channel_count);
        for buffer in &self.channel_buffers {
            if let Some(value) = buffer.try_pop() {
                channels.push(value);
            } else {
                // This shouldn't happen if we checked above, but handle gracefully
                return None;
            }
        }

        // Generate quality metrics
        let quality_indicators = self.generate_quality_metrics(&channels, timestamp);

        Some(EmgSample {
            timestamp,
            sequence,
            channels,
            quality_indicators,
        })
    }

    /// Add multi-channel sample at once
    pub fn add_multi_channel_sample(&self, values: &[f32]) -> Result<(), String> {
        if values.len() != self.config.channel_count {
            return Err(format!(
                "Expected {} channels, got {}",
                self.config.channel_count,
                values.len()
            ));
        }

        let timestamp = self.get_current_timestamp();

        // Try to add all channels atomically
        for (channel_id, &value) in values.iter().enumerate() {
            if self.channel_buffers[channel_id].try_push(value).is_err() {
                return Err(format!("Channel {} buffer full", channel_id));
            }
        }

        // Add timestamp
        if self.timestamp_buffer.try_push(timestamp).is_err() {
            return Err("Timestamp buffer full".to_string());
        }

        Ok(())
    }

    /// Get buffer utilization for all channels
    pub fn get_buffer_utilization(&self) -> Vec<f32> {
        self.channel_buffers
            .iter()
            .map(|buffer| buffer.utilization())
            .collect()
    }

    /// Check if synchronizer is healthy
    pub fn is_healthy(&self) -> bool {
        // Check buffer utilization is reasonable
        let utilizations = self.get_buffer_utilization();
        let max_util = utilizations.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_util = utilizations.iter().fold(1.0f32, |a, &b| a.min(b));

        // Buffers shouldn't be too full or have too much variance
        max_util < 0.9 && (max_util - min_util) < 0.3
    }

    /// Reset synchronizer state
    pub fn reset(&self) {
        for buffer in &self.channel_buffers {
            while buffer.try_pop().is_some() {}
        }
        while self.timestamp_buffer.try_pop().is_some() {}

        self.sequence_counter.store(0, Ordering::Relaxed);
        self.is_synced.store(false, Ordering::Relaxed);
    }

    /// Get configuration
    pub fn config(&self) -> &SyncConfig {
        &self.config
    }

    fn calculate_buffer_size(config: &SyncConfig) -> usize {
        // Buffer size based on sample rate and timeout
        let samples_per_ms = config.sample_rate_hz / 1000;
        let min_buffer_size = (samples_per_ms * config.sync_timeout_ms * 4) as usize;

        // Round up to next power of 2
        min_buffer_size.next_power_of_two().max(64)
    }

    fn get_current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    }

    fn generate_quality_metrics(&self, channels: &[f32], _timestamp: u64) -> QualityMetrics {
        // Calculate basic signal quality metrics
        let signal_power: f32 = channels.iter().map(|&x| x * x).sum::<f32>() / channels.len() as f32;
        let noise_floor = 0.01; // Estimated noise floor
        let snr_db = 10.0 * (signal_power / noise_floor).log10();

        // Estimate contact impedance (mock implementation)
        let contact_impedance_kohm = channels
            .iter()
            .map(|&x| if x.abs() > 0.1 { 10.0 } else { 50.0 })
            .collect();

        // Check for artifacts and saturation
        let artifact_detected = channels.iter().any(|&x| x.abs() > 0.8);
        let signal_saturation = channels.iter().any(|&x| x.abs() > 0.95);

        QualityMetrics {
            snr_db: snr_db.max(0.0),
            contact_impedance_kohm,
            artifact_detected,
            signal_saturation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synchronizer_creation() {
        let config = SyncConfig::default();
        let sync = SampleSynchronizer::new(config).unwrap();

        assert_eq!(sync.config().channel_count, 8);
        assert!(sync.is_healthy());
    }

    #[test]
    fn test_multi_channel_sample() {
        let config = SyncConfig {
            channel_count: 4,
            ..Default::default()
        };
        let sync = SampleSynchronizer::new(config).unwrap();

        let values = [0.1, 0.2, 0.3, 0.4];
        assert!(sync.add_multi_channel_sample(&values).is_ok());

        let sample = sync.try_get_synchronized_sample().unwrap();
        assert_eq!(sample.channels.len(), 4);
        assert_eq!(sample.channels, values);
        assert!(sample.quality_indicators.snr_db > 0.0);
    }

    #[test]
    fn test_channel_by_channel() {
        let config = SyncConfig {
            channel_count: 3,
            ..Default::default()
        };
        let sync = SampleSynchronizer::new(config).unwrap();

        // Add samples channel by channel
        assert!(sync.add_channel_sample(0, 0.1).is_ok());
        assert!(sync.add_channel_sample(1, 0.2).is_ok());
        assert!(sync.add_channel_sample(2, 0.3).is_ok());

        let sample = sync.try_get_synchronized_sample().unwrap();
        assert_eq!(sample.channels, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_buffer_utilization() {
        let config = SyncConfig {
            channel_count: 2,
            ..Default::default()
        };
        let sync = SampleSynchronizer::new(config).unwrap();

        let utilizations = sync.get_buffer_utilization();
        assert_eq!(utilizations.len(), 2);
        assert!(utilizations.iter().all(|&u| u == 0.0));

        sync.add_multi_channel_sample(&[0.1, 0.2]).unwrap();
        let utilizations = sync.get_buffer_utilization();
        assert!(utilizations.iter().all(|&u| u > 0.0));
    }

    #[test]
    fn test_invalid_channel() {
        let config = SyncConfig {
            channel_count: 2,
            ..Default::default()
        };
        let sync = SampleSynchronizer::new(config).unwrap();

        assert!(sync.add_channel_sample(2, 0.1).is_err()); // Channel 2 doesn't exist
        assert!(sync.add_multi_channel_sample(&[0.1, 0.2, 0.3]).is_err()); // Too many channels
    }

    #[test]
    fn test_reset() {
        let config = SyncConfig {
            channel_count: 2,
            ..Default::default()
        };
        let sync = SampleSynchronizer::new(config).unwrap();

        sync.add_multi_channel_sample(&[0.1, 0.2]).unwrap();
        assert!(sync.try_get_synchronized_sample().is_some());

        sync.reset();
        assert!(sync.try_get_synchronized_sample().is_none());
    }
}