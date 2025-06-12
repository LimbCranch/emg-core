// tests/ring_buffer_performance.rs
//! Performance tests for ring buffer implementations

use emg_core::acquisition::ring_buffer::{LockFreeRingBuffer, MpmcRingBuffer};
use std::thread;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
fn test_spsc_latency_under_100ns() {
    let buffer = LockFreeRingBuffer::new(1024).unwrap();
    let iterations = 10000;
    let mut push_times = Vec::with_capacity(iterations);
    let mut pop_times = Vec::with_capacity(iterations);

    // Measure push latency
    for i in 0..iterations {
        let start = Instant::now();
        buffer.try_push(i).unwrap();
        let elapsed = start.elapsed();
        push_times.push(elapsed.as_nanos() as u64);
    }

    // Measure pop latency
    for _ in 0..iterations {
        let start = Instant::now();
        let _item = buffer.try_pop().unwrap();
        let elapsed = start.elapsed();
        pop_times.push(elapsed.as_nanos() as u64);
    }

    // Calculate statistics
    let avg_push = push_times.iter().sum::<u64>() / iterations as u64;
    let avg_pop = pop_times.iter().sum::<u64>() / iterations as u64;
    let max_push = *push_times.iter().max().unwrap();
    let max_pop = *pop_times.iter().max().unwrap();

    println!("SPSC Ring Buffer Performance:");
    println!("  Average push: {} ns", avg_push);
    println!("  Average pop:  {} ns", avg_pop);
    println!("  Max push:     {} ns", max_push);
    println!("  Max pop:      {} ns", max_pop);

    // Verify performance requirements
    assert!(avg_push < 100, "Average push latency {} ns exceeds 100ns", avg_push);
    assert!(avg_pop < 100, "Average pop latency {} ns exceeds 100ns", avg_pop);

    // Allow some outliers but 95th percentile should be under 200ns
    push_times.sort_unstable();
    pop_times.sort_unstable();
    let push_95th = push_times[(iterations as f32 * 0.95) as usize];
    let pop_95th = pop_times[(iterations as f32 * 0.95) as usize];

    assert!(push_95th < 200, "95th percentile push latency {} ns too high", push_95th);
    assert!(pop_95th < 200, "95th percentile pop latency {} ns too high", pop_95th);
}

#[test]
fn test_mpmc_latency() {
    let buffer = MpmcRingBuffer::new(1024).unwrap();
    let iterations = 1000;
    let mut times = Vec::with_capacity(iterations);

    // Measure push/pop latency with contention
    for i in 0..iterations {
        let start = Instant::now();
        buffer.try_push(i).unwrap();
        let _item = buffer.try_pop().unwrap();
        let elapsed = start.elapsed();
        times.push(elapsed.as_nanos() as u64);
    }

    let avg_time = times.iter().sum::<u64>() / iterations as u64;
    let max_time = *times.iter().max().unwrap();

    println!("MPMC Ring Buffer Performance:");
    println!("  Average push+pop: {} ns", avg_time);
    println!("  Max push+pop:     {} ns", max_time);

    // MPMC will be slower due to CAS operations, allow 500ns
    assert!(avg_time < 500, "Average MPMC latency {} ns too high", avg_time);
}

#[test]
fn test_concurrent_spsc_performance() {
    let buffer = Arc::new(LockFreeRingBuffer::new(8192).unwrap());
    let buffer_clone = buffer.clone();
    let iterations = 100000;

    let start_time = Instant::now();

    let producer = thread::spawn(move || {
        for i in 0..iterations {
            while buffer_clone.try_push(i).is_err() {
                thread::yield_now();
            }
        }
    });

    let consumer = thread::spawn(move || {
        let mut received = 0;
        while received < iterations {
            if buffer.try_pop().is_some() {
                received += 1;
            } else {
                thread::yield_now();
            }
        }
    });

    producer.join().unwrap();
    consumer.join().unwrap();

    let elapsed = start_time.elapsed();
    let throughput = iterations as f64 / elapsed.as_secs_f64();

    println!("Concurrent SPSC Performance:");
    println!("  {} items in {:?}", iterations, elapsed);
    println!("  Throughput: {:.0} items/sec", throughput);

    // Should achieve high throughput (>1M items/sec)
    assert!(throughput > 1_000_000.0, "Throughput {} too low", throughput);
}

#[test]
fn test_memory_ordering_stress() {
    // Stress test to verify memory ordering is correct
    let buffer = Arc::new(LockFreeRingBuffer::new(256).unwrap());
    let num_threads = 4;
    let iterations = 10000;

    let mut handles = Vec::new();

    // Start multiple producer threads
    for thread_id in 0..num_threads {
        let buffer_clone = buffer.clone();
        let handle = thread::spawn(move || {
            for i in 0..iterations {
                let value = thread_id * iterations + i;
                while buffer_clone.try_push(value).is_err() {
                    thread::yield_now();
                }
            }
        });
        handles.push(handle);
    }

    // Start consumer thread
    let buffer_consumer = buffer.clone();
    let consumer_handle = thread::spawn(move || {
        let mut received = Vec::new();
        let total_expected = num_threads * iterations;

        while received.len() < total_expected {
            if let Some(item) = buffer_consumer.try_pop() {
                received.push(item);
            } else {
                thread::yield_now();
            }
        }
        received
    });

    // Wait for all producers
    for handle in handles {
        handle.join().unwrap();
    }

    // Wait for consumer
    let received = consumer_handle.join().unwrap();

    // Verify all items received
    assert_eq!(received.len(), num_threads * iterations);

    // Verify no duplicates (memory ordering correctness)
    let mut sorted = received.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), received.len(), "Found duplicate items - memory ordering issue");

    println!("Memory ordering stress test passed: {} items", received.len());
}

#[test]
fn test_emg_realistic_workload() {
    // Simulate realistic EMG data flow
    let buffer = Arc::new(LockFreeRingBuffer::new(4096).unwrap());
    let sample_rate_hz = 2000;
    let duration_seconds = 1;
    let total_samples = sample_rate_hz * duration_seconds;

    let producer_buffer = buffer.clone();
    let consumer_buffer = buffer.clone();

    let start_time = Instant::now();

    // Producer: EMG samples at 2kHz
    let producer = thread::spawn(move || {
        let interval = Duration::from_nanos(1_000_000_000 / sample_rate_hz as u64);
        let mut next_time = Instant::now();

        for i in 0..total_samples {
            // Simulate EMG sample data
            let emg_value = (i as f32 * 0.1).sin(); // Mock EMG signal

            while producer_buffer.try_push(emg_value).is_err() {
                thread::yield_now();
            }

            // Maintain timing
            next_time += interval;
            let now = Instant::now();
            if now < next_time {
                thread::sleep(next_time - now);
            }
        }
    });

    // Consumer: Process samples as they arrive
    let consumer = thread::spawn(move || {
        let mut processed = 0;
        let mut max_gap = Duration::ZERO;
        let mut last_time = Instant::now();

        while processed < total_samples {
            if let Some(_sample) = consumer_buffer.try_pop() {
                let now = Instant::now();
                let gap = now - last_time;
                if gap > max_gap {
                    max_gap = gap;
                }
                last_time = now;
                processed += 1;
            } else {
                thread::yield_now();
            }
        }
        (processed, max_gap)
    });

    producer.join().unwrap();
    let (processed_count, max_gap) = consumer.join().unwrap();

    let total_time = start_time.elapsed();
    let actual_rate = processed_count as f64 / total_time.as_secs_f64();

    println!("EMG Realistic Workload:");
    println!("  Processed {} samples in {:?}", processed_count, total_time);
    println!("  Actual rate: {:.0} Hz", actual_rate);
    println!("  Max processing gap: {:?}", max_gap);

    assert_eq!(processed_count, total_samples);
    assert!(actual_rate >= (sample_rate_hz as f64 * 0.95), "Sample rate too low");
    assert!(max_gap < Duration::from_millis(10), "Processing gap too large");
}

#[test]
fn test_buffer_utilization_tracking() {
    let buffer = LockFreeRingBuffer::new(64).unwrap();

    // Empty buffer
    assert_eq!(buffer.utilization(), 0.0);
    assert!(buffer.is_empty());
    assert!(!buffer.is_full());

    // Fill half buffer
    for i in 0..30 {
        buffer.try_push(i).unwrap();
    }

    let utilization = buffer.utilization();
    assert!(utilization > 0.4 && utilization < 0.6);
    assert!(!buffer.is_empty());
    assert!(!buffer.is_full());

    // Fill to near capacity (capacity - 1 due to ring buffer design)
    for i in 30..63 {
        buffer.try_push(i).unwrap();
    }

    assert!(buffer.utilization() > 0.9);
    assert!(buffer.is_full());

    // Verify can't add more
    assert!(buffer.try_push(999).is_err());
}