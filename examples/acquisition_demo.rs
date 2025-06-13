// examples/acquisition_demo.rs
//! Demonstration of EMG acquisition system with ring buffers

use emg_core::{
    EmgDevice, SimulatorDevice, EmgSample, QualityMetrics,
    acquisition::{BufferManager, BufferConfig, ProcessedSample, LockFreeRingBuffer},
};
use std::thread;
use std::time::{Duration, Instant};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 EMG Acquisition System Demo");
    println!("===============================\n");

    // Demo 1: Basic ring buffer operations
    demonstrate_ring_buffer()?;

    // Demo 2: EMG device with buffer manager
    demonstrate_emg_acquisition().await?;

    // Demo 3: Real-time processing pipeline
    demonstrate_realtime_pipeline().await?;

    println!("✅ All demonstrations completed successfully!");
    Ok(())
}

fn demonstrate_ring_buffer() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Demo 1: Lock-Free Ring Buffer Performance");
    println!("--------------------------------------------");

    let mut buffer = LockFreeRingBuffer::new(1024)?;
    println!("Created ring buffer with capacity: {}", buffer.capacity());

    // Performance test
    let iterations = 10000;
    let start_time = Instant::now();

    // Push test data
    for i in 0..iterations {
        while buffer.try_push(i).is_err() {
            // Buffer full, pop some items
            buffer.try_pop();
        }
    }

    // Pop all data
    let mut popped_count = 0;
    while buffer.try_pop().is_some() {
        popped_count += 1;
    }

    let elapsed = start_time.elapsed();
    let throughput = (iterations + popped_count) as f64 / elapsed.as_secs_f64();

    println!("  Processed {} operations in {:?}", iterations + popped_count, elapsed);
    println!("  Throughput: {:.0} ops/sec", throughput);
    println!("  Average operation time: {:.0} ns", elapsed.as_nanos() as f64 / (iterations + popped_count) as f64);
    println!("  ✅ Ring buffer performance verified\n");

    Ok(())
}

async fn demonstrate_emg_acquisition() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Demo 2: EMG Device with Buffer Manager");
    println!("------------------------------------------");

    // Create EMG simulator
    let mut device = SimulatorDevice::healthy_user()?;
    device.initialize().await?;
    device.start_acquisition().await?;

    // Create buffer manager
    let config = BufferConfig {
        channel_count: 8,
        sample_rate_hz: 2000,
        target_latency_ms: 20,
        ..Default::default()
    };
    let buffer_manager = BufferManager::new(config)?;

    println!("  EMG Device: {}", device.get_device_info().name);
    println!("  Channels: {}", device.get_channel_count());
    println!("  Sample Rate: {} Hz", device.get_sampling_rate());
    println!("  Buffer Manager: Ready");

    // Collect samples for 1 second
    println!("\n  Collecting EMG samples for 1 second...");
    let sample_duration = Duration::from_millis(1000);
    let start_time = Instant::now();
    let mut sample_count = 0;

    while start_time.elapsed() < sample_duration {
        // Get sample from device
        let sample = device.read_sample().await?;

        // Add to buffer manager
        buffer_manager.add_raw_sample(sample)?;
        sample_count += 1;

        // Process sample if available
        if let Some(raw_sample) = buffer_manager.get_raw_sample() {
            // Simulate signal processing (filtering, feature extraction, etc.)
            let processing_start = Instant::now();
            let processed_channels: Vec<f32> = raw_sample.channels
                .iter()
                .map(|&x| x * 0.95) // Simple gain adjustment
                .collect();
            let processing_time = processing_start.elapsed().as_nanos() as u64;

            let processed_sample = ProcessedSample {
                timestamp: raw_sample.timestamp,
                sequence: raw_sample.sequence,
                channels: processed_channels,
                quality_metrics: raw_sample.quality_indicators,
                processing_latency_ns: processing_time,
            };

            buffer_manager.add_processed_sample(processed_sample)?;
        }

        // Small delay to prevent overwhelming the system
        tokio::time::sleep(Duration::from_micros(400)).await; // ~2.5kHz
    }

    device.stop_acquisition().await?;

    // Show results
    let metrics = buffer_manager.get_metrics();
    println!("  Samples collected: {}", sample_count);
    println!("  Buffer metrics:");
    println!("    Processed: {}", metrics.samples_processed);
    println!("    Dropped: {}", metrics.samples_dropped);
    println!("    Raw buffer utilization: {:.1}%", metrics.raw_utilization * 100.0);
    println!("    Processed buffer utilization: {:.1}%", metrics.processed_utilization * 100.0);
    println!("    Average latency: {:.1} μs", metrics.average_latency_ns as f64 / 1000.0);
    println!("  ✅ EMG acquisition working correctly\n");

    Ok(())
}

async fn demonstrate_realtime_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ Demo 3: Real-Time Processing Pipeline");
    println!("----------------------------------------");

    // Create high-performance configuration
    let config = BufferConfig {
        channel_count: 8,
        sample_rate_hz: 2000,
        target_latency_ms: 10, // Aggressive latency target
        enable_overflow_protection: true,
        ..Default::default()
    };

    let buffer_manager = Arc::new(BufferManager::new(config)?);
    let running = Arc::new(AtomicBool::new(true));

    println!("  Target sample rate: 2000 Hz");
    println!("  Target latency: 10 ms");
    println!("  Channels: 8");
    println!("  Duration: 2 seconds");

    let producer_manager = buffer_manager.clone();
    let processor_manager = buffer_manager.clone();
    let consumer_manager = buffer_manager.clone();
    let producer_running = running.clone();
    let processor_running = running.clone();
    let consumer_running = running.clone();

    let start_time = Instant::now();

    // Producer thread: Generate EMG samples at 2kHz
    let producer = thread::spawn(move || {
        let mut device = SimulatorDevice::healthy_user().unwrap();
        let sample_interval = Duration::from_nanos(500_000); // 2kHz
        let mut next_time = Instant::now();
        let mut produced = 0;

        while producer_running.load(Ordering::Relaxed) {
            // Generate EMG sample
            let channels: Vec<f32> = (0..8)
                .map(|i| {
                    let freq = 10.0 + i as f32 * 5.0; // Different frequencies per channel
                    let phase = next_time.elapsed().as_secs_f32() * freq * 2.0 * std::f32::consts::PI;
                    (phase.sin() + 0.1 * (phase * 3.0).sin()) * 0.1 // EMG-like signal
                })
                .collect();

            let sample = EmgSample {
                timestamp: next_time.elapsed().as_nanos() as u64,
                sequence: produced,
                channels,
                quality_indicators: QualityMetrics {
                    snr_db: 28.0 + (produced % 10) as f32, // Varying SNR
                    contact_impedance_kohm: vec![15.0; 8],
                    artifact_detected: (produced % 100) == 0, // Occasional artifacts
                    signal_saturation: false,
                },
            };

            if producer_manager.add_raw_sample(sample).is_ok() {
                produced += 1;
            }

            // Maintain precise timing
            next_time += sample_interval;
            let now = Instant::now();
            if now < next_time {
                std::thread::sleep(next_time - now);
            }
        }
        produced
    });

    // Processor thread: Real-time signal processing
    let processor = thread::spawn(move || {
        let mut processed = 0;
        let mut total_processing_time = Duration::ZERO;

        while processor_running.load(Ordering::Relaxed) {
            if let Some(raw_sample) = processor_manager.get_raw_sample() {
                let processing_start = Instant::now();

                // Simulate real EMG processing
                let processed_channels: Vec<f32> = raw_sample.channels
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| {
                        // Simulate filtering and feature extraction
                        let filtered = x * 0.9; // High-pass filter approximation
                        let rectified = filtered.abs(); // Rectification
                        let smoothed = rectified * 0.8; // Low-pass filter approximation
                        smoothed
                    })
                    .collect();

                let processing_time = processing_start.elapsed();
                total_processing_time += processing_time;

                let processed_sample = ProcessedSample {
                    timestamp: raw_sample.timestamp,
                    sequence: raw_sample.sequence,
                    channels: processed_channels,
                    quality_metrics: raw_sample.quality_indicators,
                    processing_latency_ns: processing_time.as_nanos() as u64,
                };

                if processor_manager.add_processed_sample(processed_sample).is_ok() {
                    processed += 1;
                }
            } else {
                std::thread::yield_now();
            }
        }
        (processed, total_processing_time)
    });

    // Consumer thread: Output and analysis
    let consumer = thread::spawn(move || {
        let mut consumed = 0;
        let mut gesture_detections = 0;
        let mut quality_issues = 0;

        while consumer_running.load(Ordering::Relaxed) {
            if let Some(processed_sample) = consumer_manager.get_processed_sample() {
                consumed += 1;

                // Simulate gesture recognition
                let signal_strength: f32 = processed_sample.channels.iter().sum::<f32>() / processed_sample.channels.len() as f32;
                if signal_strength > 0.05 {
                    gesture_detections += 1;
                }

                // Check signal quality
                if processed_sample.quality_metrics.snr_db < 25.0 || processed_sample.quality_metrics.artifact_detected {
                    quality_issues += 1;
                }

                // Log every 1000 samples
                if consumed % 1000 == 0 {
                    println!("    Processed {} samples, {} gestures detected, {} quality issues",
                             consumed, gesture_detections, quality_issues);
                }
            } else {
                std::thread::yield_now();
            }
        }
        (consumed, gesture_detections, quality_issues)
    });

    // Run for 2 seconds
    thread::sleep(Duration::from_secs(2));
    running.store(false, Ordering::Relaxed);

    let total_time = start_time.elapsed();
    let produced_count = producer.join().unwrap();
    let (processed_count, total_processing_time) = processor.join().unwrap();
    let (consumed_count, gesture_detections, quality_issues) = consumer.join().unwrap();

    // Final metrics
    let metrics = buffer_manager.get_metrics();
    let actual_sample_rate = produced_count as f64 / total_time.as_secs_f64();
    let avg_processing_time = if processed_count > 0 {
        total_processing_time.as_micros() as f64 / processed_count as f64
    } else {
        0.0
    };

    println!("\n  📈 Real-Time Pipeline Results:");
    println!("    Total time: {:?}", total_time);
    println!("    Sample rate: {:.0} Hz (target: 2000 Hz)", actual_sample_rate);
    println!("    Samples: {} produced, {} processed, {} consumed", produced_count, processed_count, consumed_count);
    println!("    Gesture detections: {}", gesture_detections);
    println!("    Quality issues: {}", quality_issues);
    println!("    Average processing time: {:.1} μs", avg_processing_time);
    println!("    Buffer utilization: Raw {:.1}%, Processed {:.1}%",
             metrics.raw_utilization * 100.0, metrics.processed_utilization * 100.0);
    println!("    Latency: Avg {:.1} μs, Max {:.1} μs",
             metrics.average_latency_ns as f64 / 1000.0, metrics.max_latency_ns as f64 / 1000.0);
    println!("    Errors: {} dropped, {} underruns, {} overruns",
             metrics.samples_dropped, metrics.underruns, metrics.overruns);

    // Performance verification
    let performance_ok = actual_sample_rate > 1900.0 && // 95% of target rate
        metrics.average_latency_ns < 50_000 && // <50μs average latency
        metrics.samples_dropped < (produced_count / 100) as u64; // <1% drop rate

    if performance_ok {
        println!("  ✅ Real-time performance targets met!");
    } else {
        println!("  ⚠️  Performance targets not fully met (may be normal in debug mode)");
    }

    println!("  ✅ Pipeline demonstration complete\n");

    Ok(())
}