// tests/acquisition_integration.rs
//! Integration tests for acquisition pipeline

use emg_core::{
    EmgSample, QualityMetrics,
    acquisition::{
        BufferManager, BufferConfig, SampleSynchronizer,
        sample_sync::SyncConfig, ProcessedSample
    }
};
use std::thread;
use std::time::{Duration, Instant};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

#[test]
fn test_full_acquisition_pipeline() {
    let config = BufferConfig {
        channel_count: 8,
        sample_rate_hz: 2000,
        target_latency_ms: 20,
        ..Default::default()
    };

    let manager = BufferManager::new(config).unwrap();
    assert!(manager.is_healthy());

    // Simulate EMG data acquisition
    let sample = EmgSample {
        timestamp: 1234567890,
        sequence: 1,
        channels: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        quality_indicators: QualityMetrics {
            snr_db: 25.0,
            contact_impedance_kohm: vec![15.0; 8],
            artifact_detected: false,
            signal_saturation: false,
        },
    };

    // Add raw sample
    manager.add_raw_sample(sample.clone()).unwrap();

    // Get and process sample
    let raw_sample = manager.get_raw_sample().unwrap();
    assert_eq!(raw_sample.sequence, sample.sequence);

    // Create processed sample
    let processed = ProcessedSample {
        timestamp: raw_sample.timestamp,
        sequence: raw_sample.sequence,
        channels: raw_sample.channels.clone(),
        quality_metrics: raw_sample.quality_indicators.clone(),
        processing_latency_ns: 1_000_000, // 1ms processing time
    };

    // Add processed sample
    manager.add_processed_sample(processed).unwrap();

    // Get processed sample
    let output = manager.get_processed_sample().unwrap();
    assert_eq!(output.sequence, sample.sequence);
    assert_eq!(output.processing_latency_ns, 1_000_000);

    // Check metrics
    let metrics = manager.get_metrics();
    assert_eq!(metrics.samples_processed, 1);
    assert_eq!(metrics.samples_dropped, 0);
    assert!(metrics.average_latency_ns > 0);
}

#[test]
fn test_synchronized_multi_channel_acquisition() {
    let sync_config = SyncConfig {
        channel_count: 4,
        sample_rate_hz: 1000,
        ..Default::default()
    };

    let synchronizer = SampleSynchronizer::new(sync_config).unwrap();

    // Add samples channel by channel (simulating interleaved acquisition)
    for sample_num in 0..10 {
        for channel in 0..4 {
            let value = (sample_num as f32 + channel as f32) * 0.1;
            synchronizer.add_channel_sample(channel, value).unwrap();
        }

        // Should be able to get synchronized sample
        if let Some(sample) = synchronizer.try_get_synchronized_sample() {
            assert_eq!(sample.channels.len(), 4);
            assert_eq!(sample.sequence, sample_num);

            // Verify channel ordering
            for (i, &val) in sample.channels.iter().enumerate() {
                let expected = (sample_num as f32 + i as f32) * 0.1;
                assert!((val - expected).abs() < 0.001);
            }
        }
    }
}

#[test]
fn test_high_throughput_acquisition() {
    let config = BufferConfig {
        channel_count: 8,
        sample_rate_hz: 2000,
        target_latency_ms: 10,
        ..Default::default()
    };

    let manager = Arc::new(BufferManager::new(config).unwrap());
    let producer_manager = manager.clone();
    let consumer_manager = manager.clone();

    let samples_to_process = 10000;
    let running = Arc::new(AtomicBool::new(true));
    let producer_running = running.clone();
    let consumer_running = running.clone();

    let start_time = Instant::now();

    // Producer thread: Generate EMG samples
    let producer = thread::spawn(move || {
        let mut sequence = 0;
        let sample_interval = Duration::from_nanos(1_000_000_000 / 2000); // 2kHz
        let mut next_time = Instant::now();

        while producer_running.load(Ordering::Relaxed) && sequence < samples_to_process {
            let channels: Vec<f32> = (0..8)
                .map(|i| (sequence as f32 * 0.01 + i as f32 * 0.1).sin())
                .collect();

            let sample = EmgSample {
                timestamp: next_time.elapsed().as_nanos() as u64,
                sequence,
                channels,
                quality_indicators: QualityMetrics {
                    snr_db: 30.0,
                    contact_impedance_kohm: vec![12.0; 8],
                    artifact_detected: false,
                    signal_saturation: false,
                },
            };

            // Non-blocking add with retry
            let mut retries = 0;
            while producer_manager.add_raw_sample(sample.clone()).is_err() && retries < 10 {
                thread::yield_now();
                retries += 1;
            }

            if retries < 10 {
                sequence += 1;
            }

            // Maintain timing
            next_time += sample_interval;
            let now = Instant::now();
            if now < next_time {
                std::thread::sleep(next_time - now);
            }
        }
        sequence
    });

    // Consumer thread: Process samples
    let consumer = thread::spawn(move || {
        let mut processed_count = 0;
        let mut total_latency_ns = 0u64;

        while consumer_running.load(Ordering::Relaxed) && processed_count < samples_to_process {
            if let Some(raw_sample) = consumer_manager.get_raw_sample() {
                let processing_start = Instant::now();

                // Simulate signal processing
                let processed_channels: Vec<f32> = raw_sample.channels
                    .iter()
                    .map(|&x| x * 0.9) // Simple processing
                    .collect();

                let processing_latency = processing_start.elapsed().as_nanos() as u64;
                total_latency_ns += processing_latency;

                let processed_sample = ProcessedSample {
                    timestamp: raw_sample.timestamp,
                    sequence: raw_sample.sequence,
                    channels: processed_channels,
                    quality_metrics: raw_sample.quality_indicators,
                    processing_latency_ns: processing_latency,
                };

                // Try to add processed sample
                if consumer_manager.add_processed_sample(processed_sample).is_ok() {
                    processed_count += 1;
                }
            } else {
                thread::yield_now();
            }
        }
        (processed_count, total_latency_ns / processed_count.max(1) as u64)
    });

    // Let it run for a while
    thread::sleep(Duration::from_millis(100));
    running.store(false, Ordering::Relaxed);

    let produced_count = producer.join().unwrap();
    let (processed_count, avg_processing_latency) = consumer.join().unwrap();
    let total_time = start_time.elapsed();

    println!("High Throughput Test Results:");
    println!("  Produced: {} samples", produced_count);
    println!("  Processed: {} samples", processed_count);
    println!("  Total time: {:?}", total_time);
    println!("  Average processing latency: {} ns", avg_processing_latency);

    // Verify performance
    let actual_rate = processed_count as f64 / total_time.as_secs_f64();
    println!("  Actual processing rate: {:.0} Hz", actual_rate);

    assert!(processed_count > 100, "Should process significant number of samples");
    assert!(actual_rate > 1000.0, "Should maintain >1kHz processing rate");
    assert!(avg_processing_latency < 10_000, "Processing latency should be <10μs");

    // Check final metrics
    let metrics = manager.get_metrics();
    println!("  Buffer metrics:");
    println!("    Samples processed: {}", metrics.samples_processed);
    println!("    Samples dropped: {}", metrics.samples_dropped);
    println!("    Average latency: {} ns", metrics.average_latency_ns);
    println!("    Max latency: {} ns", metrics.max_latency_ns);
    println!("    Underruns: {}", metrics.underruns);
    println!("    Overruns: {}", metrics.overruns);

    // Verify low error rates
    let total_samples = metrics.samples_processed + metrics.samples_dropped;
    if total_samples > 0 {
        let error_rate = metrics.samples_dropped as f64 / total_samples as f64;
        assert!(error_rate < 0.01, "Error rate too high: {:.2}%", error_rate * 100.0);
    }
}

#[test]
fn test_buffer_overflow_protection() {
    let config = BufferConfig {
        channel_count: 4,
        sample_rate_hz: 1000,
        raw_buffer_size: Some(16), // Very small buffer to force overflow
        enable_overflow_protection: true,
        ..Default::default()
    };

    let manager = BufferManager::new(config).unwrap();

    // Fill buffer beyond capacity
    let mut added_count = 0;
    let mut overflow_count = 0;

    for i in 0..100 {
        let sample = EmgSample {
            timestamp: i as u64,
            sequence: i,
            channels: vec![0.1; 4],
            quality_indicators: QualityMetrics::default(),
        };

        match manager.add_raw_sample(sample) {
            Ok(()) => added_count += 1,
            Err(_) => overflow_count += 1,
        }
    }

    println!("Overflow protection test:");
    println!("  Added: {}", added_count);
    println!("  Overflowed: {}", overflow_count);

    // Should have detected overflow
    assert!(overflow_count > 0, "Should have detected buffer overflow");
    assert!(added_count < 100, "Should not have added all samples");

    let metrics = manager.get_metrics();
    assert!(metrics.overruns > 0, "Should have recorded overruns");
}

#[test]
fn test_synchronizer_health_monitoring() {
    let sync_config = SyncConfig {
        channel_count: 8,
        sample_rate_hz: 2000,
        ..Default::default()
    };

    let synchronizer = SampleSynchronizer::new(sync_config).unwrap();
    assert!(synchronizer.is_healthy());

    // Add balanced samples
    for _ in 0..10 {
        for channel in 0..8 {
            synchronizer.add_channel_sample(channel, 0.5).unwrap();
        }
    }

    // Should still be healthy
    assert!(synchronizer.is_healthy());

    // Check utilization
    let utilizations = synchronizer.get_buffer_utilization();
    assert_eq!(utilizations.len(), 8);
    assert!(utilizations.iter().all(|&u| u > 0.0 && u < 1.0));
}

#[test]
fn test_end_to_end_latency() {
    let config = BufferConfig {
        channel_count: 8,
        sample_rate_hz: 2000,
        target_latency_ms: 20,
        ..Default::default()
    };

    let manager = BufferManager::new(config).unwrap();
    let mut end_to_end_latencies = Vec::new();

    for i in 0..100 {
        let start_time = Instant::now();

        // Create sample with current timestamp
        let sample = EmgSample {
            timestamp: start_time.elapsed().as_nanos() as u64,
            sequence: i,
            channels: vec![0.1; 8],
            quality_indicators: QualityMetrics::default(),
        };

        // Add raw sample
        manager.add_raw_sample(sample).unwrap();

        // Get and process immediately
        if let Some(raw_sample) = manager.get_raw_sample() {
            let processed_sample = ProcessedSample {
                timestamp: raw_sample.timestamp,
                sequence: raw_sample.sequence,
                channels: raw_sample.channels,
                quality_metrics: raw_sample.quality_indicators,
                processing_latency_ns: 1000, // 1μs mock processing
            };

            manager.add_processed_sample(processed_sample).unwrap();

            if let Some(_output) = manager.get_processed_sample() {
                let end_to_end_latency = start_time.elapsed();
                end_to_end_latencies.push(end_to_end_latency.as_nanos() as u64);
            }
        }
    }

    let avg_latency_ns = end_to_end_latencies.iter().sum::<u64>() / end_to_end_latencies.len() as u64;
    let max_latency_ns = *end_to_end_latencies.iter().max().unwrap();

    println!("End-to-end latency test:");
    println!("  Average: {} ns ({:.2} μs)", avg_latency_ns, avg_latency_ns as f64 / 1000.0);
    println!("  Maximum: {} ns ({:.2} μs)", max_latency_ns, max_latency_ns as f64 / 1000.0);

    // Should be very low latency for this test
    assert!(avg_latency_ns < 100_000, "Average latency {} ns too high", avg_latency_ns); // <100μs
    assert!(max_latency_ns < 1_000_000, "Max latency {} ns too high", max_latency_ns); // <1ms
}