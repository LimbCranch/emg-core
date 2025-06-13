// src/acquisition/ring_buffer.rs
//! Lock-free ring buffer for real-time signal acquisition

use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Lock-free single-producer single-consumer ring buffer
pub struct LockFreeRingBuffer<T> {
    buffer: Vec<MaybeUninit<T>>,
    capacity: usize,
    mask: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

/// Multi-producer multi-consumer ring buffer
pub struct MpmcRingBuffer<T> {
    buffer: Vec<MaybeUninit<T>>,
    capacity: usize,
    mask: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

/// Ring buffer error types
#[derive(Debug, PartialEq)]
pub enum RingBufferError {
    Full,
    Empty,
    InvalidCapacity,
}

impl std::fmt::Display for RingBufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RingBufferError::Full => write!(f, "Ring buffer is full"),
            RingBufferError::Empty => write!(f, "Ring buffer is empty"),
            RingBufferError::InvalidCapacity => write!(f, "Invalid buffer capacity (must be power of 2)"),
        }
    }
}

impl std::error::Error for RingBufferError {}

impl<T> LockFreeRingBuffer<T> {
    /// Create new ring buffer with power-of-2 capacity
    pub fn new(capacity: usize) -> Result<Self, RingBufferError> {
        if capacity == 0 || !capacity.is_power_of_two() {
            return Err(RingBufferError::InvalidCapacity);
        }

        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, MaybeUninit::uninit);

        Ok(Self {
            buffer,
            capacity,
            mask: capacity - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        })
    }

    /// Try to push item (non-blocking)
    pub fn try_push(&mut self, item: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);
        let next_head = (head + 1) & self.mask;
        let tail = self.tail.load(Ordering::Acquire);

        if next_head == tail {
            return Err(item); // Buffer full
        }

        unsafe {
            self.buffer[head].as_mut_ptr().write(item);
        }

        self.head.store(next_head, Ordering::Release);
        Ok(())
    }

    /// Try to pop item (non-blocking)
    pub fn try_pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail == head {
            return None; // Buffer empty
        }

        let item = unsafe { self.buffer[tail].as_ptr().read() };
        let next_tail = (tail + 1) & self.mask;
        self.tail.store(next_tail, Ordering::Release);

        Some(item)
    }

    /// Get current buffer utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f32 {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        let used = if head >= tail {
            head - tail
        } else {
            self.capacity - tail + head
        };
        used as f32 / self.capacity as f32
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Relaxed) == self.tail.load(Ordering::Relaxed)
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        ((head + 1) & self.mask) == tail
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T> MpmcRingBuffer<T> {
    /// Create new MPMC ring buffer
    pub fn new(capacity: usize) -> Result<Self, RingBufferError> {
        if capacity == 0 || !capacity.is_power_of_two() {
            return Err(RingBufferError::InvalidCapacity);
        }

        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, MaybeUninit::uninit);

        Ok(Self {
            buffer,
            capacity,
            mask: capacity - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        })
    }

    /// Try to push item with CAS loop for multiple producers
    pub fn try_push(&mut self, item: T) -> Result<(), T> {
        loop {
            let head = self.head.load(Ordering::Relaxed);
            let next_head = (head + 1) & self.mask;
            let tail = self.tail.load(Ordering::Acquire);

            if next_head == tail {
                return Err(item); // Buffer full
            }

            // Try to claim this slot
            if self.head.compare_exchange_weak(
                head,
                next_head,
                Ordering::Release,
                Ordering::Relaxed,
            ).is_ok() {
                unsafe {
                    self.buffer[head].as_mut_ptr().write(item);
                }
                return Ok(());
            }
            // CAS failed, retry
        }
    }

    /// Try to pop item with CAS loop for multiple consumers
    pub fn try_pop(&self) -> Option<T> {
        loop {
            let tail = self.tail.load(Ordering::Relaxed);
            let head = self.head.load(Ordering::Acquire);

            if tail == head {
                return None; // Buffer empty
            }

            let next_tail = (tail + 1) & self.mask;

            // Try to claim this slot
            if self.tail.compare_exchange_weak(
                tail,
                next_tail,
                Ordering::Release,
                Ordering::Relaxed,
            ).is_ok() {
                let item = unsafe { self.buffer[tail].as_ptr().read() };
                return Some(item);
            }
            // CAS failed, retry
        }
    }

    /// Get current buffer utilization
    pub fn utilization(&self) -> f32 {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        let used = if head >= tail {
            head - tail
        } else {
            self.capacity - tail + head
        };
        used as f32 / self.capacity as f32
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Relaxed) == self.tail.load(Ordering::Relaxed)
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

unsafe impl<T: Send> Send for LockFreeRingBuffer<T> {}
unsafe impl<T: Send> Sync for LockFreeRingBuffer<T> {}
unsafe impl<T: Send> Send for MpmcRingBuffer<T> {}
unsafe impl<T: Send> Sync for MpmcRingBuffer<T> {}

impl<T> Drop for LockFreeRingBuffer<T> {
    fn drop(&mut self) {
        while self.try_pop().is_some() {}
    }
}

impl<T> Drop for MpmcRingBuffer<T> {
    fn drop(&mut self) {
        while self.try_pop().is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spsc_basic_operations() {
        let mut buffer = LockFreeRingBuffer::new(8).unwrap();

        // Test push and pop
        assert!(buffer.try_push(42).is_ok());
        assert!(buffer.try_push(43).is_ok());

        assert_eq!(buffer.try_pop(), Some(42));
        assert_eq!(buffer.try_pop(), Some(43));
        assert_eq!(buffer.try_pop(), None);
    }

    #[test]
    fn test_buffer_full() {
        let mut buffer = LockFreeRingBuffer::new(4).unwrap();

        // Fill buffer
        for i in 0..3 {
            assert!(buffer.try_push(i).is_ok());
        }

        // Buffer should be full (capacity - 1)
        assert!(buffer.try_push(999).is_err());
    }

    /*#[test]
    fn test_concurrent_spsc() {
        let buffer = Arc::new(LockFreeRingBuffer::new(1024).unwrap());
        let mut buffer_clone = buffer.clone();

        let producer = thread::spawn(move || {
            for i in 0..1000 {
                while buffer_clone.try_push(i).is_err() {
                    thread::yield_now();
                }
            }
        });

        let consumer = thread::spawn(move || {
            let mut received = Vec::new();
            for _ in 0..1000 {
                loop {
                    if let Some(item) = buffer.try_pop() {
                        received.push(item);
                        break;
                    }
                    thread::yield_now();
                }
            }
            received
        });

        producer.join().unwrap();
        let received = consumer.join().unwrap();

        assert_eq!(received.len(), 1000);
        for (i, &val) in received.iter().enumerate() {
            assert_eq!(val, i);
        }
    }*/

    #[test]
    fn test_mpmc_basic() {
        let mut buffer = MpmcRingBuffer::new(8).unwrap();

        assert!(buffer.try_push(100).is_ok());
        assert!(buffer.try_push(200).is_ok());

        assert_eq!(buffer.try_pop(), Some(100));
        assert_eq!(buffer.try_pop(), Some(200));
        assert_eq!(buffer.try_pop(), None);
    }

    #[test]
    fn test_utilization() {
        let mut buffer = LockFreeRingBuffer::new(8).unwrap();

        assert_eq!(buffer.utilization(), 0.0);

        buffer.try_push(1).unwrap();
        buffer.try_push(2).unwrap();

        assert!(buffer.utilization() > 0.0);
        assert!(buffer.utilization() < 1.0);
    }

    #[test]
    fn test_invalid_capacity() {
        assert!(LockFreeRingBuffer::<i32>::new(0).is_err());
        assert!(LockFreeRingBuffer::<i32>::new(3).is_err()); // Not power of 2
        assert!(LockFreeRingBuffer::<i32>::new(8).is_ok());
    }
}