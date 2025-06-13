use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Time provider trait for dependency injection and testing
pub trait TimeProvider: Send + Sync {
    fn now_nanos(&self) -> u64;
    fn now_micros(&self) -> u64 {
        self.now_nanos() / 1000
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
    pub fn new(initial_time_nanos: u64) -> Self {
        Self {
            current_time: AtomicU64::new(initial_time_nanos),
        }
    }

    pub fn advance_by(&self, nanos: u64) {
        self.current_time.fetch_add(nanos, Ordering::Relaxed);
    }

    pub fn set_time(&self, nanos: u64) {
        self.current_time.store(nanos, Ordering::Relaxed);
    }
}

impl TimeProvider for MockTimeProvider {
    fn now_nanos(&self) -> u64 {
        self.current_time.load(Ordering::Relaxed)
    }
}

/// FIX: Centralized timestamp utility to eliminate code duplication
pub fn current_timestamp_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

pub fn current_timestamp_micros() -> u64 {
    current_timestamp_nanos() / 1000
}