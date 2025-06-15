//! Time and timestamp utilities for EMG-Core
//!
//! This module extracts common timestamp generation patterns and provides
//! monotonic clock support, timestamp validation, and testable time providers.
//!
//! Key features:
//! - Unified timestamp generation (replacing repeated patterns)
//! - Monotonic clock support for consistent timing
//! - Timestamp validation and bounds checking
//! - Testable time provider traits
//! - Performance-optimized timestamp functions

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH, Instant, Duration};
use crate::config::constants::{signal, performance, validation};

/// Time-related errors
#[derive(Debug, Clone, PartialEq)]
pub enum TimeError {
    /// System time is before Unix epoch
    SystemTimeBeforeEpoch,
    /// Invalid timestamp value
    InvalidTimestamp(u64),
    /// Monotonic clock not available
    MonotonicClockNotAvailable,
    /// Timestamp validation failed
    ValidationFailed(String),
    /// Time drift detected
    TimeDriftDetected { expected: u64, actual: u64 },
    /// Clock synchronization failed
    SynchronizationFailed(String),
}

impl std::fmt::Display for TimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeError::SystemTimeBeforeEpoch => {
                write!(f, "System time is before Unix epoch")
            }
            TimeError::InvalidTimestamp(ts) => {
                write!(f, "Invalid timestamp: {}", ts)
            }
            TimeError::MonotonicClockNotAvailable => {
                write!(f, "Monotonic clock not available on this system")
            }
            TimeError::ValidationFailed(msg) => {
                write!(f, "Timestamp validation failed: {}", msg)
            }
            TimeError::TimeDriftDetected { expected, actual } => {
                write!(f, "Time drift detected: expected {}, got {}", expected, actual)
            }
            TimeError::SynchronizationFailed(msg) => {
                write!(f, "Clock synchronization failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for TimeError {}

/// Result type for time operations
pub type TimeResult<T> = Result<T, TimeError>;

/// Trait for providing timestamps - enables mocking in tests
pub trait TimeProvider: Send + Sync {
    /// Get current timestamp in nanoseconds since Unix epoch
    fn current_timestamp_nanos(&self) -> TimeResult<u64>;

    /// Get current timestamp in microseconds since Unix epoch
    fn current_timestamp_micros(&self) -> TimeResult<u64>;

    /// Get current timestamp in milliseconds since Unix epoch
    fn current_timestamp_millis(&self) -> TimeResult<u64>;
}

/// System time provider using SystemTime
#[derive(Debug, Default)]
pub struct SystemTimeProvider;

impl TimeProvider for SystemTimeProvider {
    fn current_timestamp_nanos(&self) -> TimeResult<u64> {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .map_err(|_| TimeError::SystemTimeBeforeEpoch)
    }

    fn current_timestamp_micros(&self) -> TimeResult<u64> {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .map_err(|_| TimeError::SystemTimeBeforeEpoch)
    }

    fn current_timestamp_millis(&self) -> TimeResult<u64> {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .map_err(|_| TimeError::SystemTimeBeforeEpoch)
    }
}

/// Monotonic time provider using Instant for relative timing
#[derive(Debug)]
pub struct MonotonicTimeProvider {
    start_instant: Instant,
    start_system_time: SystemTime,
    last_timestamp: AtomicU64,
}

impl MonotonicTimeProvider {
    /// Create a new monotonic time provider
    pub fn new() -> Self {
        Self {
            start_instant: Instant::now(),
            start_system_time: SystemTime::now(),
            last_timestamp: AtomicU64::new(0),
        }
    }

    /// Get monotonic timestamp in nanoseconds (always increasing)
    pub fn monotonic_timestamp_nanos(&self) -> u64 {
        let elapsed = self.start_instant.elapsed();
        let timestamp = elapsed.as_nanos() as u64;

        // Ensure monotonic property
        let last = self.last_timestamp.load(Ordering::Relaxed);
        let new_timestamp = timestamp.max(last + 1);

        self.last_timestamp.store(new_timestamp, Ordering::Relaxed);
        new_timestamp
    }

    /// Convert monotonic timestamp to system timestamp
    pub fn monotonic_to_system_timestamp(&self, monotonic_ns: u64) -> TimeResult<u64> {
        let system_duration = self.start_system_time
            .duration_since(UNIX_EPOCH)
            .map_err(|_| TimeError::SystemTimeBeforeEpoch)?;

        Ok(system_duration.as_nanos() as u64 + monotonic_ns)
    }
}

impl Default for MonotonicTimeProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeProvider for MonotonicTimeProvider {
    fn current_timestamp_nanos(&self) -> TimeResult<u64> {
        let monotonic = self.monotonic_timestamp_nanos();
        self.monotonic_to_system_timestamp(monotonic)
    }

    fn current_timestamp_micros(&self) -> TimeResult<u64> {
        self.current_timestamp_nanos().map(|ns| ns / 1_000)
    }

    fn current_timestamp_millis(&self) -> TimeResult<u64> {
        self.current_timestamp_nanos().map(|ns| ns / 1_000_000)
    }
}

/// Global time providers
static SYSTEM_TIME_PROVIDER: SystemTimeProvider = SystemTimeProvider;
static MONOTONIC_TIME_PROVIDER: std::sync::LazyLock<MonotonicTimeProvider> =
    std::sync::LazyLock::new(MonotonicTimeProvider::new);

/// Mock time provider for testing
#[cfg(test)]
pub struct MockTimeProvider {
    timestamp: Mutex<u64>,
}

#[cfg(test)]
impl MockTimeProvider {
    pub fn new(initial_timestamp: u64) -> Self {
        Self {
            timestamp: Mutex::new(initial_timestamp),
        }
    }

    pub fn advance_time(&self, nanos: u64) {
        let mut ts = self.timestamp.lock().unwrap();
        *ts += nanos;
    }

    pub fn set_time(&self, timestamp: u64) {
        let mut ts = self.timestamp.lock().unwrap();
        *ts = timestamp;
    }
}

#[cfg(test)]
impl TimeProvider for MockTimeProvider {
    fn current_timestamp_nanos(&self) -> TimeResult<u64> {
        Ok(*self.timestamp.lock().unwrap())
    }

    fn current_timestamp_micros(&self) -> TimeResult<u64> {
        Ok(*self.timestamp.lock().unwrap() / 1_000)
    }

    fn current_timestamp_millis(&self) -> TimeResult<u64> {
        Ok(*self.timestamp.lock().unwrap() / 1_000_000)
    }
}

/// Extract the repeated timestamp generation pattern into utility functions
/// This replaces the pattern: SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64

/// Get current timestamp in nanoseconds (primary function used throughout codebase)
pub fn current_timestamp_nanos() -> u64 {
    SYSTEM_TIME_PROVIDER
        .current_timestamp_nanos()
        .unwrap_or_else(|_| {
            // Fallback to monotonic time if system time fails
            MONOTONIC_TIME_PROVIDER.monotonic_timestamp_nanos()
        })
}

/// Get current timestamp in microseconds
pub fn current_timestamp_micros() -> u64 {
    current_timestamp_nanos() / 1_000
}

/// Get current timestamp in milliseconds
pub fn current_timestamp_millis() -> u64 {
    current_timestamp_nanos() / 1_000_000
}

/// Get monotonic timestamp in nanoseconds (always increasing)
pub fn monotonic_timestamp_nanos() -> u64 {
    MONOTONIC_TIME_PROVIDER.monotonic_timestamp_nanos()
}

/// Generate timestamp with specific increment for atomic sequences
/// Fixes the race condition identified in simulator timing
pub fn atomic_timestamp_with_increment(base_timestamp: &AtomicU64, increment_nanos: u64) -> u64 {
    base_timestamp.fetch_add(increment_nanos, Ordering::SeqCst)
}

/// Calculate sample period in nanoseconds from sampling rate
pub fn calculate_sample_period_nanos(sampling_rate_hz: u32) -> u64 {
    if sampling_rate_hz == 0 {
        return 0;
    }
    1_000_000_000 / sampling_rate_hz as u64
}

/// Calculate sampling rate from sample period in nanoseconds
pub fn calculate_sampling_rate_hz(period_nanos: u64) -> u32 {
    if period_nanos == 0 {
        return 0;
    }
    (1_000_000_000 / period_nanos) as u32
}

/// Timestamp validator for ensuring timestamp validity
#[derive(Debug)]
pub struct TimestampValidator {
    min_timestamp: u64,
    max_timestamp: u64,
    max_drift_nanos: u64,
    last_timestamp: AtomicU64,
}

impl TimestampValidator {
    /// Create new timestamp validator
    pub fn new() -> Self {
        let now = current_timestamp_nanos();
        Self {
            min_timestamp: now.saturating_sub(validation::VALIDATION_TIMEOUT_MS as u64 * 1_000_000),
            max_timestamp: now + (365 * 24 * 60 * 60 * 1_000_000_000), // 1 year from now
            max_drift_nanos: performance::MAX_LATENCY_TARGET_MS as u64 * 1_000_000,
            last_timestamp: AtomicU64::new(0),
        }
    }

    /// Create validator with custom bounds
    pub fn with_bounds(min_timestamp: u64, max_timestamp: u64, max_drift_nanos: u64) -> Self {
        Self {
            min_timestamp,
            max_timestamp,
            max_drift_nanos,
            last_timestamp: AtomicU64::new(0),
        }
    }

    /// Validate timestamp is within bounds and monotonic
    pub fn validate(&self, timestamp: u64) -> TimeResult<()> {
        // Check bounds
        if timestamp < self.min_timestamp {
            return Err(TimeError::InvalidTimestamp(timestamp));
        }
        if timestamp > self.max_timestamp {
            return Err(TimeError::InvalidTimestamp(timestamp));
        }

        // Check for excessive drift
        let last = self.last_timestamp.load(Ordering::Relaxed);
        if last > 0 {
            let drift = if timestamp > last {
                timestamp - last
            } else {
                last - timestamp
            };

            if drift > self.max_drift_nanos {
                return Err(TimeError::TimeDriftDetected {
                    expected: last,
                    actual: timestamp,
                });
            }
        }

        // Update last timestamp for drift checking
        self.last_timestamp.store(timestamp, Ordering::Relaxed);

        Ok(())
    }

    /// Reset validator state
    pub fn reset(&self) {
        self.last_timestamp.store(0, Ordering::Relaxed);
    }
}

impl Default for TimestampValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate timestamp is within reasonable bounds
pub fn validate_timestamp(timestamp: u64) -> TimeResult<()> {
    static VALIDATOR: std::sync::LazyLock<TimestampValidator> =
        std::sync::LazyLock::new(TimestampValidator::new);

    VALIDATOR.validate(timestamp)
}

/// Convert duration to nanoseconds with overflow protection
pub fn duration_to_nanos(duration: Duration) -> u64 {
    duration.as_nanos().min(u64::MAX as u128) as u64
}

/// Convert nanoseconds to duration
pub fn nanos_to_duration(nanos: u64) -> Duration {
    Duration::from_nanos(nanos)
}

/// Get elapsed time in nanoseconds since given timestamp
pub fn elapsed_nanos_since(timestamp: u64) -> u64 {
    let now = current_timestamp_nanos();
    now.saturating_sub(timestamp)
}

/// Check if timestamp is within timeout period
pub fn is_within_timeout(timestamp: u64, timeout_nanos: u64) -> bool {
    elapsed_nanos_since(timestamp) <= timeout_nanos
}

/// Sleep for specified nanoseconds (useful for precise timing)
pub fn sleep_nanos(nanos: u64) -> TimeResult<()> {
    if nanos == 0 {
        return Ok(());
    }

    std::thread::sleep(Duration::from_nanos(nanos));
    Ok(())
}

/// High-precision sleep using spin-wait for very short durations
pub fn precision_sleep_nanos(nanos: u64) -> TimeResult<()> {
    if nanos == 0 {
        return Ok(());
    }

    // Use spin-wait for very short durations (< 1ms)
    if nanos < 1_000_000 {
        let start = Instant::now();
        let target = Duration::from_nanos(nanos);

        while start.elapsed() < target {
            std::hint::spin_loop();
        }
    } else {
        std::thread::sleep(Duration::from_nanos(nanos));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_current_timestamp_nanos() {
        let ts1 = current_timestamp_nanos();
        thread::sleep(Duration::from_millis(1));
        let ts2 = current_timestamp_nanos();

        assert!(ts2 > ts1);
        assert!(ts2 - ts1 >= 1_000_000); // At least 1ms difference
    }

    #[test]
    fn test_monotonic_timestamp() {
        let provider = MonotonicTimeProvider::new();
        let ts1 = provider.monotonic_timestamp_nanos();
        let ts2 = provider.monotonic_timestamp_nanos();
        let ts3 = provider.monotonic_timestamp_nanos();

        assert!(ts2 > ts1);
        assert!(ts3 > ts2);
    }

    #[test]
    fn test_timestamp_validator() {
        let validator = TimestampValidator::new();
        let now = current_timestamp_nanos();

        // Valid timestamp
        assert!(validator.validate(now).is_ok());

        // Too old timestamp
        let old_timestamp = now.saturating_sub(validator.max_drift_nanos + 1);
        assert!(validator.validate(old_timestamp).is_err());
    }

    #[test]
    fn test_sample_period_calculation() {
        assert_eq!(calculate_sample_period_nanos(1000), 1_000_000); // 1 kHz = 1ms period
        assert_eq!(calculate_sample_period_nanos(2000), 500_000);   // 2 kHz = 0.5ms period
        assert_eq!(calculate_sample_period_nanos(0), 0);            // Handle zero case
    }

    #[test]
    fn test_sampling_rate_calculation() {
        assert_eq!(calculate_sampling_rate_hz(1_000_000), 1000); // 1ms period = 1 kHz
        assert_eq!(calculate_sampling_rate_hz(500_000), 2000);   // 0.5ms period = 2 kHz
        assert_eq!(calculate_sampling_rate_hz(0), 0);            // Handle zero case
    }

    #[test]
    fn test_mock_time_provider() {
        let mock = MockTimeProvider::new(123_456_789);
        assert_eq!(mock.current_timestamp_nanos().unwrap(), 123_456_789);

        mock.advance_time(1000);
        assert_eq!(mock.current_timestamp_nanos().unwrap(), 123_457_789);

        mock.set_time(999_999_999);
        assert_eq!(mock.current_timestamp_nanos().unwrap(), 999_999_999);
    }

    #[test]
    fn test_atomic_timestamp_increment() {
        let base = AtomicU64::new(1_000_000_000);
        let increment = 500_000; // 0.5ms

        let ts1 = atomic_timestamp_with_increment(&base, increment);
        let ts2 = atomic_timestamp_with_increment(&base, increment);

        assert_eq!(ts2 - ts1, increment);
    }

    #[test]
    fn test_elapsed_time_calculation() {
        let now = current_timestamp_nanos();
        thread::sleep(Duration::from_millis(10));
        let elapsed = elapsed_nanos_since(now);

        assert!(elapsed >= 10_000_000); // At least 10ms
    }

    #[test]
    fn test_timeout_checking() {
        let timestamp = current_timestamp_nanos();

        // Should be within timeout
        assert!(is_within_timeout(timestamp, 1_000_000_000)); // 1 second

        // Simulate old timestamp
        let old_timestamp = timestamp.saturating_sub(2_000_000_000); // 2 seconds ago
        assert!(!is_within_timeout(old_timestamp, 1_000_000_000)); // 1 second timeout
    }

    #[test]
    fn test_duration_conversion() {
        let duration = Duration::from_secs(1);
        let nanos = duration_to_nanos(duration);
        assert_eq!(nanos, 1_000_000_000);

        let back_to_duration = nanos_to_duration(nanos);
        assert_eq!(back_to_duration, duration);
    }

    #[test]
    fn test_time_provider_trait() {
        let system_provider = SystemTimeProvider;
        let monotonic_provider = MonotonicTimeProvider::new();

        let sys_ts = system_provider.current_timestamp_nanos().unwrap();
        let mono_ts = monotonic_provider.current_timestamp_nanos().unwrap();

        // Both should provide reasonable timestamps
        assert!(sys_ts > 1_000_000_000_000_000_000); // After year 2001
        assert!(mono_ts > 0);
    }

    #[test]
    fn test_precision_sleep() {
        let start = Instant::now();
        precision_sleep_nanos(1_000_000).unwrap(); // 1ms
        let elapsed = start.elapsed();

        // Should be close to 1ms (with some tolerance for system timing)
        assert!(elapsed >= Duration::from_millis(1));
        assert!(elapsed < Duration::from_millis(10)); // Reasonable upper bound
    }
}