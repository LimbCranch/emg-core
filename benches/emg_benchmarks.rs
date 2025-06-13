
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use emg_core::hal::simulator::{SimulatorDevice, SimulatorConfig};
use emg_core::hal::EmgDevice;
use emg_core::acquisition::ring_buffer::LockFreeRingBuffer;
use emg_core::processing::pipeline::SignalPipeline;
use emg_core::processing::filters::{HighPassFilter, BandPassFilter, NotchFilter, Filter};
use emg_core::utils::time::{SystemTimeProvider, MockTimeProvider};
use std::sync::Arc;
use tokio::runtime::Runtime;
use emg_core::processing::FeatureConfig;

const SAMPLE_RATES: &[u32] = &[1000, 2000, 4000, 8000];
const CHANNEL_COUNTS: &[usize] = &[1, 4, 8, 16, 32];
const BUFFER_SIZES: &[usize] = &[64, 128, 256, 512, 1024, 2048];

fn benchmark_ring_buffer_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("ring_buffer");

    for &buffer_size in BUFFER_SIZES {
        for &channel_count in &[1, 8, 16] {
            group.throughput(Throughput::Elements(1000));

            // Benchmark single producer single consumer
            group.bench_with_input(
                BenchmarkId::new("spsc_push", format!("{}ch_{}buf", channel_count, buffer_size)),
                &(channel_count, buffer_size),
                |b, &(channels, size)| {
                    let buffer = LockFreeRingBuffer::<f32>::new(size);
                    let data = vec![1.0f32; channels];

                    b.iter(|| {
                        for _ in 0..1000 {
                            let _ = buffer.try_push(black_box(&data));
                        }
                    });
                },
            );

            // Benchmark pop operations
            group.bench_with_input(
                BenchmarkId::new("spsc_pop", format!("{}ch_{}buf", channel_count, buffer_size)),
                &(channel_count, buffer_size),
                |b, &(channels, size)| {
                    let buffer = LockFreeRingBuffer::<f32>::new(size);
                    // Pre-fill buffer
                    for _ in 0..size / 2 {
                        let data = vec![1.0f32; channels];
                        let _ = buffer.try_push(&data);
                    }

                    b.iter(|| {
                        for _ in 0..100 {
                            let _ = buffer.try_pop();
                        }
                    });
                },
            );

            // Benchmark concurrent access
            group.bench_with_input(
                BenchmarkId::new("concurrent", format!("{}ch_{}buf", channel_count, buffer_size)),
                &(channel_count, buffer_size),
                |b, &(channels, size)| {
                    use std::thread;
                    use std::sync::Arc;

                    let buffer = Arc::new(LockFreeRingBuffer::<f32>::new(size));
                    let data = vec![1.0f32; channels];

                    b.iter(|| {
                        let buffer_clone = buffer.clone();
                        let data_clone = data.clone();

                        let producer = thread::spawn(move || {
                            for _ in 0..500 {
                                let _ = buffer_clone.try_push(black_box(&data_clone));
                            }
                        });

                        let buffer_clone = buffer.clone();
                        let consumer = thread::spawn(move || {
                            for _ in 0..500 {
                                let _ = buffer_clone.try_pop();
                            }
                        });

                        producer.join().unwrap();
                        consumer.join().unwrap();
                    });
                },
            );
        }
    }

    group.finish();
}

fn benchmark_simulator_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulator");

    for &sample_rate in SAMPLE_RATES {
        for &channel_count in CHANNEL_COUNTS {
            group.throughput(Throughput::Elements(1000));

            group.bench_with_input(
                BenchmarkId::new("sample_generation", format!("{}Hz_{}ch", sample_rate, channel_count)),
                &(sample_rate, channel_count),
                |b, &(rate, channels)| {
                    let rt = Runtime::new().unwrap();
                    let config = SimulatorConfig {
                        channel_count: channels,
                        sample_rate_hz: rate,
                        ..Default::default()
                    };

                    let mut device = SimulatorDevice::new(config).unwrap();
                    rt.block_on(async { device.initialize().await.unwrap() });
                    rt.block_on(async { device.start_acquisition().await.unwrap() });

                    b.to_async(&rt).iter(|| async {
                        for _ in 0..1000 {
                            let _ = device.read_sample().await.unwrap();
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

fn benchmark_signal_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("signal_processing");

    for &channel_count in &[1, 8, 16] {
        // Benchmark individual filters
        group.bench_with_input(
            BenchmarkId::new("highpass_filter", format!("{}ch", channel_count)),
            &channel_count,
            |b, &channels| {
                let mut filter = HighPassFilter::new(20.0, 2000.0).unwrap();
                let input = vec![0.5; channels];

                b.iter(|| {
                    for _ in 0..1000 {
                        let _ = filter.process(black_box(&input)).unwrap();
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bandpass_filter", format!("{}ch", channel_count)),
            &channel_count,
            |b, &channels| {
                let mut filter = BandPassFilter::new(20.0, 500.0, 2000.0).unwrap();
                let input = vec![0.5; channels];

                b.iter(|| {
                    for _ in 0..1000 {
                        let _ = filter.process(black_box(&input)).unwrap();
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("notch_filter", format!("{}ch", channel_count)),
            &channel_count,
            |b, &channels| {
                let mut filter = NotchFilter::new(50.0, 2000.0, 10.0).unwrap();
                let input = vec![0.5; channels];

                b.iter(|| {
                    for _ in 0..1000 {
                        let _ = filter.process(black_box(&input)).unwrap();
                    }
                });
            },
        );

        // Benchmark complete pipeline
        group.bench_with_input(
            BenchmarkId::new("complete_pipeline", format!("{}ch", channel_count)),
            &channel_count,
            |b, &channels| {
                let time_provider = Arc::new(SystemTimeProvider);
                let config = emg_core::processing::pipeline::PipelineConfig::default();
                let mut pipeline = SignalPipeline::new(config,  FeatureConfig::default() , time_provider).unwrap();

                let sample = emg_core::hal::types::EmgSample {
                    timestamp: 1000000,
                    sequence: 1,
                    channels: vec![0.5; channels],
                    quality_indicators: emg_core::hal::types::QualityMetrics {
                        snr_db: 25.0,
                        contact_impedance_kohm: vec![10.0; channels],
                        artifact_detected: false,
                        signal_saturation: false,
                    },
                };

                b.iter(|| {
                    for _ in 0..100 {
                        let _ = pipeline.process_sample(black_box(sample.clone())).unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_end_to_end_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_latency");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    for &channel_count in &[8, 16] {
        group.bench_with_input(
            BenchmarkId::new("acquisition_to_processing", format!("{}ch", channel_count)),
            &channel_count,
            |b, &channels| {
                let rt = Runtime::new().unwrap();

                // Setup complete pipeline
                let config = SimulatorConfig {
                    channel_count: channels,
                    sample_rate_hz: 2000,
                    ..Default::default()
                };

                let mut device = SimulatorDevice::new(config).unwrap();
                let time_provider = Arc::new(SystemTimeProvider);
                let pipeline_config = emg_core::processing::pipeline::PipelineConfig::default();
                let mut pipeline = SignalPipeline::new(pipeline_config,  FeatureConfig::default(), time_provider).unwrap();

                rt.block_on(async {
                    device.initialize().await.unwrap();
                    device.start_acquisition().await.unwrap();
                });

                b.to_async(&rt).iter(&mut || async {
                    let start_time = std::time::Instant::now();

                    // Acquire sample
                    let sample = device.read_sample().await.unwrap();

                    // Process sample
                    let _processed = pipeline.process_sample(sample).unwrap();

                    let end_time = std::time::Instant::now();
                    black_box(end_time.duration_since(start_time))
                });
            },
        );
    }

    group.finish();
}

fn benchmark_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    group.bench_function("zero_copy_operations", |b| {
        let buffer = LockFreeRingBuffer::<f32>::new(1024);
        let data = vec![1.0f32; 8];

        b.iter(|| {
            // Test that we can pass data without unnecessary copying
            for _ in 0..1000 {
                let _ = buffer.try_push(black_box(&data));
                let _ = buffer.try_pop();
            }
        });
    });

    group.bench_function("allocation_free_processing", |b| {
        let mut filter = HighPassFilter::new(20.0, 2000.0).unwrap();
        let mut input = vec![0.5; 8];

        b.iter(|| {
            for _ in 0..1000 {
                // Reuse the same input buffer to avoid allocations
                input[0] = (input[0] + 0.1) % 1.0;
                let _ = filter.process(black_box(&input)).unwrap();
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_ring_buffer_operations,
    benchmark_simulator_performance,
    benchmark_signal_processing,
    benchmark_end_to_end_latency,
    benchmark_memory_efficiency
);
criterion_main!(benches);