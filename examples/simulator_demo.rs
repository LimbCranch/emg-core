//! EMG Simulator Demo
//!
//! This example demonstrates how to use the EMG simulator to generate realistic
//! EMG signals with various activation patterns and noise characteristics.

use emg_core::{
    hal::{DeviceFactory, EmgDevice},
    hal::simulator::{SimulatorConfig, ActivationPattern, TimeProviderType},
    utils::time::current_timestamp_nanos,
};
use std::time::{Duration, Instant};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("EMG-Core Simulator Demo");
    println!("======================");

    // Create simulator configuration
    let config = SimulatorConfig {
        channel_count: 8,
        sampling_rate_hz: 500, // 2 kHz
        signal_amplitude: 0.001, // 1mV
        noise_level: 0.50,       // 5% noise
        enable_artifacts: true,
        artifact_probability: 0.50, // 2% chance per sample
        enable_realistic_emg: true,
        muscle_activation_pattern: ActivationPattern::Sinusoidal { frequency_hz: 20.0 },
        time_provider: TimeProviderType::System,
    };

    println!("Configuration:");
    println!("  Channels: {}", config.channel_count);
    println!("  Sampling Rate: {} Hz", config.sampling_rate_hz);
    println!("  Signal Amplitude: {} V", config.signal_amplitude);
    println!("  Noise Level: {}%", config.noise_level * 100.0);
    println!("  Artifacts Enabled: {}", config.enable_artifacts);
    println!();

    // Create and initialize simulator
    let mut simulator = DeviceFactory::create_simulator(config)?;

    println!("Initializing simulator...");
    simulator.initialize().await?;

    println!("Connecting to simulator...");
    simulator.connect().await?;

    // Get device information
    let device_info = simulator.get_device_info().await?;
    println!("Device Info:");
    println!("  ID: {}", device_info.device_id);
    println!("  Type: {}", device_info.device_type);
    println!("  Firmware: {}", device_info.firmware_version);
    println!("  Serial: {}", device_info.serial_number);
    println!("  Max Sampling Rate: {} Hz", device_info.capabilities.max_sampling_rate_hz);
    println!("  Resolution: {} bits", device_info.capabilities.resolution_bits);
    println!();

    println!("Starting data acquisition...");
    simulator.start_acquisition().await?;

    // Collect samples for analysis
    let mut samples = Vec::new();
    let start_time = Instant::now();
    let acquisition_duration = Duration::from_secs(2);

    println!("Acquiring data for {} seconds...", acquisition_duration.as_secs());

    while start_time.elapsed() < acquisition_duration {
        let sample = simulator.read_sample().await?;
        samples.push(sample);

        // Print progress every 500 samples
        if samples.len() % 10000 == 0 {
            print!(".");
            if samples.len() % 100000 == 0 {
                println!(" {} samples", samples.len());
            }
        }
    }

    if samples.len() % 5000 != 0 {
        println!(" {} samples", samples.len());
    }

    println!("\nData acquisition complete!");
    println!();

    // Analyze collected data
    analyze_samples(&samples);

    // Test different activation patterns
    println!("\nTesting different activation patterns...");
    test_activation_patterns().await?;

    // Stop acquisition and disconnect
    simulator.stop_acquisition().await?;
    simulator.disconnect().await?;

    println!("Demo completed successfully!");
    Ok(())
}

fn analyze_samples(samples: &[emg_core::hal::EmgSample]) {
    if samples.is_empty() {
        println!("No samples to analyze");
        return;
    }

    println!("Sample Analysis:");
    println!("  Total samples: {}", samples.len());

    let channel_count = samples[0].channel_data.len();
    println!("  Channels: {}", channel_count);

    // Calculate statistics for each channel
    for ch in 0..channel_count {
        let channel_values: Vec<f32> = samples.iter()
            .map(|s| s.channel_data[ch])
            .collect();

        let min_val = channel_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = channel_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let avg_val = channel_values.iter().sum::<f32>() / channel_values.len() as f32;

        // Calculate RMS
        let rms = (channel_values.iter().map(|&x| x * x).sum::<f32>() / channel_values.len() as f32).sqrt();

        println!("  Channel {}: min={:.6}V, max={:.6}V, avg={:.6}V, rms={:.6}V",
                 ch, min_val, max_val, avg_val, rms);
    }

    // Analyze timestamps
    let first_timestamp = samples[0].timestamp;
    let last_timestamp = samples.last().unwrap().timestamp;
    let duration_ns = last_timestamp - first_timestamp;
    let duration_ms = duration_ns as f64 / 1_000_000.0;
    let actual_rate = (samples.len() - 1) as f64 / (duration_ms / 1000.0);

    println!("  Duration: {:.2} ms", duration_ms);
    println!("  Actual sampling rate: {:.1} Hz", actual_rate);

    // Check for artifacts in quality metrics
    let mut artifact_count = 0;
    let mut total_snr = 0.0;
    let mut quality_sum = 0.0;

    for sample in samples {
        if let Some(ref metrics) = sample.quality_metrics {
            if metrics.artifact_detected {
                artifact_count += 1;
            }
            total_snr += metrics.snr_db;
            quality_sum += metrics.signal_quality;
        }
    }

    let avg_snr = total_snr / samples.len() as f32;
    let avg_quality = quality_sum / samples.len() as f32;
    let artifact_rate = (artifact_count as f32 / samples.len() as f32) * 100.0;

    println!("  Artifacts detected: {} ({:.2}%)", artifact_count, artifact_rate);
    println!("  Average SNR: {:.1} dB", avg_snr);
    println!("  Average quality: {:.3}", avg_quality);
}

async fn test_activation_patterns() -> Result<(), Box<dyn std::error::Error>> {
    let patterns = vec![
        ("Constant", ActivationPattern::Constant),
        ("Sinusoidal 5Hz", ActivationPattern::Sinusoidal { frequency_hz: 5.0 }),
        ("Burst Pattern", ActivationPattern::Burst { burst_duration_ms: 100, rest_duration_ms: 200 }),
        ("Ramp", ActivationPattern::Ramp { ramp_duration_ms: 500 }),
        ("Random", ActivationPattern::Random),
    ];

    for (name, pattern) in patterns {
        println!("  Testing {} pattern...", name);

        let config = SimulatorConfig {
            channel_count: 4,
            sampling_rate_hz: 1000,
            muscle_activation_pattern: pattern,
            enable_artifacts: false, // Disable for cleaner pattern visualization
            ..Default::default()
        };

        let mut simulator = DeviceFactory::create_simulator(config)?;
        simulator.initialize().await?;
        simulator.connect().await?;
        simulator.start_acquisition().await?;

        // Collect 100 samples
        let mut pattern_samples = Vec::new();
        for _ in 0..100 {
            let sample = simulator.read_sample().await?;
            pattern_samples.push(sample);
        }

        // Analyze pattern
        let ch0_values: Vec<f32> = pattern_samples.iter()
            .map(|s| s.channel_data[0])
            .collect();

        let min_val = ch0_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = ch0_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        println!("    Range: {:.6}V, Pattern visible: {}", range, range > 0.0001);

        simulator.stop_acquisition().await?;
        simulator.disconnect().await?;
    }

    Ok(())
}