// src/utils/time.rs
//! Time utilities and providers for the EMG system
//!
//! FIXED: Centralized timestamp utilities, dependency injection for testing

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Time provider trait for dependency injection and testing
pub trait TimeProvider: Send + Sync {
    fn now_nanos(&self) -> u64;
    fn now_micros(&self) -> u64 {
        self.now_nanos() / 1000
    }
    fn now_millis(&self) -> u64 {
        self.now_nanos() / 1_000_000
    }
}

/// System time provider using actual system clock
pub struct SystemTimeProvider;

impl TimeProvider for SystemTimeProvider {
    fn now_nanos(&self) -> u64 {
        current_timestamp_nanos()
    }
}

/// Mock time provider for deterministic testing
pub struct MockTimeProvider {
    current_time: AtomicU64,
}

impl MockTimeProvider {
    /// Create new mock time provider with initial time
    pub fn new(initial_time_nanos: u64) -> Self {
        Self {
            current_time: AtomicU64::new(initial_time_nanos),
        }
    }

    /// Advance time by specified nanoseconds
    pub fn advance_by(&self, nanos: u64) {
        self.current_time.fetch_add(nanos, Ordering::Relaxed);
    }

    /// Set absolute time
    pub fn set_time(&self, nanos: u64) {
        self.current_time.store(nanos, Ordering::Relaxed);
    }

    /// Get current mock time without advancing
    pub fn current_time(&self) -> u64 {
        self.current_time.load(Ordering::Relaxed)
    }
}

impl TimeProvider for MockTimeProvider {
    fn now_nanos(&self) -> u64 {
        self.current_time.load(Ordering::Relaxed)
    }
}

/// FIXED: Centralized timestamp utility to eliminate code duplication
pub fn current_timestamp_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Get current timestamp in microseconds
pub fn current_timestamp_micros() -> u64 {
    current_timestamp_nanos() / 1000
}

/// Get current timestamp in milliseconds
pub fn current_timestamp_millis() -> u64 {
    current_timestamp_nanos() / 1_000_000
}

/// Convert nanoseconds to milliseconds
pub fn nanos_to_millis(nanos: u64) -> u64 {
    nanos / 1_000_000
}

/// Convert nanoseconds to microseconds
pub fn nanos_to_micros(nanos: u64) -> u64 {
    nanos / 1000
}

/// Convert milliseconds to nanoseconds
pub fn millis_to_nanos(millis: u64) -> u64 {
    millis * 1_000_000
}

/// Convert microseconds to nanoseconds
pub fn micros_to_nanos(micros: u64) -> u64 {
    micros * 1000
}

/// Calculate sample period in nanoseconds from sample rate
pub fn sample_period_nanos(sample_rate_hz: u32) -> u64 {
    1_000_000_000 / sample_rate_hz as u64
}

/// Calculate sample rate from period in nanoseconds
pub fn sample_rate_from_period_nanos(period_nanos: u64) -> u32 {
    (1_000_000_000 / period_nanos) as u32
}

/// Timestamp statistics for performance monitoring
#[derive(Debug, Clone, Default)]
pub struct TimestampStats {
    pub min_interval_nanos: u64,
    pub max_interval_nanos: u64,
    pub avg_interval_nanos: u64,
    pub jitter_nanos: u64,
    pub sample_count: u64,
}

/// Timestamp monitor for analyzing timing characteristics
pub struct TimestampMonitor {
    last_timestamp: Option<u64>,
    intervals: Vec<u64>,
    max_samples: usize,
}

impl TimestampMonitor {
    /// Create new timestamp monitor
    pub fn new(max_samples: usize) -> Self {
        Self {
            last_timestamp: None,
            intervals: Vec::with_capacity(max_samples),
            max_samples,
        }
    }

    /// Record a new timestamp and return the interval
    pub fn record(&mut self, timestamp: u64) -> Option<u64> {
        let interval = if let Some(last) = self.last_timestamp {
            let interval = timestamp.saturating_sub(last);
            self.intervals.push(interval);

            // Keep only the most recent samples
            if self.intervals.len() > self.max_samples {
                self.intervals.remove(0);
            }

            Some(interval)
        } else {
            None
        };

        self.last_timestamp = Some(timestamp);
        interval
    }

    /// Get timing statistics
    pub fn get_stats(&self) -> TimestampStats {
        if self.intervals.is_empty() {
            return TimestampStats::default();
        }

        let min_interval_nanos = *self.intervals.iter().min().unwrap_or(&0);
        let max_interval_nanos = *self.intervals.iter().max().unwrap_or(&0);
        let sum: u64 = self.intervals.iter().sum();
        let avg_interval_nanos = sum / self.intervals.len() as u64;

        // Calculate jitter (standard deviation)
        let variance: f64 = self.intervals.iter()
            .map(|&x| {
                let diff = x as f64 - avg_interval_nanos as f64;
                diff * diff
            })
            .sum::<f64>() / self.intervals.len() as f64;
        let jitter_nanos = variance.sqrt() as u64;

        TimestampStats {
            min_interval_nanos,
            max_interval_nanos,
            avg_interval_nanos,
            jitter_nanos,
            sample_count: self.intervals.len() as u64,
        }
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.last_timestamp = None;
        self.intervals.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_timestamp_utilities() {
        let nanos = current_timestamp_nanos();
        let micros = current_timestamp_micros();
        let millis = current_timestamp_millis();

        // Basic sanity checks
        assert!(nanos > 0);
        assert!(micros > 0);
        assert!(millis > 0);

        // Check relationships
        assert!(nanos > micros);
        assert!(micros > millis);
    }

    #[test]
    fn test_time_conversions() {
        let millis = 1000u64;
        let micros = 1_000_000u64;
        let nanos = 1_000_000_000u64;

        assert_eq!(millis_to_nanos(millis), nanos);
        assert_eq!(micros_to_nanos(micros), nanos);
        assert_eq!(nanos_to_millis(nanos), millis);
        assert_eq!(nanos_to_micros(nanos), micros);
    }

    #[test]
    fn test_sample_period_calculations() {
        let sample_rate = 1000u32; // 1 kHz
        let expected_period = 1_000_000u64; // 1 ms in nanoseconds

        assert_eq!(sample_period_nanos(sample_rate), expected_period);
        assert_eq!(sample_rate_from_period_nanos(expected_period), sample_rate);
    }

    #[test]
    fn test_mock_time_provider() {
        let initial_time = 1_000_000_000u64;
        let mock_time = MockTimeProvider::new(initial_time);

        assert_eq!(mock_time.now_nanos(), initial_time);
        assert_eq!(mock_time.current_time(), initial_time);

        // Test advancing time
        let advance_by = 500_000u64;
        mock_time.advance_by(advance_by);
        assert_eq!(mock_time.now_nanos(), initial_time + advance_by);

        // Test setting absolute time
        let new_time = 2_000_000_000u64;
        mock_time.set_time(new_time);
        assert_eq!(mock_time.now_nanos(), new_time);
    }

    #[test]
    fn test_system_time_provider() {
        let provider = SystemTimeProvider;
        let time1 = provider.now_nanos();

        // Small delay
        thread::sleep(Duration::from_millis(1));

        let time2 = provider.now_nanos();
        assert!(time2 > time1);
    }

    #[test]
    fn test_timestamp_monitor() {
        let mut monitor = TimestampMonitor::new(10);

        // Record some timestamps with known intervals
        let base_time = 1_000_000_000u64;
        let interval = 1_000_000u64; // 1ms

        for i in 0..5 {
            let timestamp = base_time + i * interval;
            let recorded_interval = monitor.record(timestamp);

            if i > 0 {
                assert_eq!(recorded_interval, Some(interval));
            } else {
                assert_eq!(recorded_interval, None); // First sample has no interval
            }
        }

        let stats = monitor.get_stats();
        assert_eq!(stats.sample_count, 4); // 5 timestamps = 4 intervals
        assert_eq!(stats.min_interval_nanos, interval);
        assert_eq!(stats.max_interval_nanos, interval);
        assert_eq!(stats.avg_interval_nanos, interval);
        assert_eq!(stats.jitter_nanos, 0); // Perfect timing
    }

    #[test]
    fn test_timestamp_monitor_jitter() {
        let mut monitor = TimestampMonitor::new(10);

        // Record timestamps with varying intervals
        let intervals = vec![1_000_000u64, 1_100_000u64, 900_000u64, 1_000_000u64];
        let mut timestamp = 1_000_000_000u64;

        monitor.record(timestamp); // First sample

        for interval in intervals {
            timestamp += interval;
            monitor.record(timestamp);
        }

        let stats = monitor.get_stats();
        assert_eq!(stats.sample_count, 4);
        assert!(stats.jitter_nanos > 0); // Should have some jitter
    }

    #[test]
    fn test_timestamp_monitor_capacity() {
        let max_samples = 3;
        let mut monitor = TimestampMonitor::new(max_samples);

        // Record more samples than capacity
        let base_time = 1_000_000_000u64;
        let interval = 1_000_000u64;

        for i in 0..10 {
            let timestamp = base_time + i * interval;
            monitor.record(timestamp);
        }

        let stats = monitor.get_stats();
        assert_eq!(stats.sample_count, max_samples as u64);
    }

    #[test]
    fn test_trait_object_compatibility() {
        // Test that trait objects work correctly
        let providers: Vec<Box<dyn TimeProvider>> = vec![
            Box::new(SystemTimeProvider),
            Box::new(MockTimeProvider::new(1_000_000_000)),
        ];

        for provider in providers {
            let time = provider.now_nanos();
            assert!(time > 0);

            let time_micros = provider.now_micros();
            let time_millis = provider.now_millis();
            assert!(time >= time_micros);
            assert!(time_micros >= time_millis);
        }
    }
}