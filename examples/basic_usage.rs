// examples/basic_usage.rs
//! Basic usage example for EMG Core HAL

use emg_core::{EmgDevice, SimulatorDevice, config::ConfigLoader};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("EMG Core HAL Basic Usage Example");
    println!("================================");

    // Load configuration
    println!("Loading configuration...");
    let mut config_loader = ConfigLoader::new();
    let config = config_loader.load_system_config()?;
    println!("Configuration loaded successfully");
    println!("  - Sampling rate: {} Hz", config.system.sampling_rate_hz);
    println!("  - Channel count: {}", config.system.channel_count);
    println!("  - Device type: {:?}", config.hal.device_type);
    println!("  - Profile type: {}", config.hal.simulator.unwrap().profile);


    // Create and initialize simulator device
    println!("\nCreating simulator device...");
    let mut device = SimulatorDevice::healthy_user()?;

    // Get device information
    let device_info = device.get_device_info();
    println!("Device: {}", device_info.name);
    println!("Version: {}", device_info.version);
    println!("Serial: {}", device_info.serial_number);
    println!("Channels: {}", device.get_channel_count());
    println!("Sample rate: {} Hz", device.get_sampling_rate());

    // Initialize device
    println!("\nInitializing device...");
    device.initialize().await?;
    println!("Device initialized successfully");

    // Start data acquisition
    println!("\nStarting data acquisition...");
    device.start_acquisition().await?;
    println!("Data acquisition started");

    // Collect some samples
    println!("\nCollecting EMG samples...");
    let mut sample_count = 0;
    let start_time = std::time::Instant::now();

    for i in 0..50 {
        let sample = device.read_sample().await?;
        sample_count += 1;

        if i % 10 == 0 {
            println!("Sample {}: seq={}, channels={}, SNR={:.1} dB",
                     i, sample.sequence, sample.channels.len(), sample.quality_indicators.snr_db);

            // Print first few channel values
            print!("  Channel values: ");
            for (ch, &val) in sample.channels.iter().take(4).enumerate() {
                print!("Ch{}={:.3} ", ch, val);
            }
            println!("...");
        }

        // Small delay to simulate real-time processing
        sleep(Duration::from_millis(1)).await;
    }

    let elapsed = start_time.elapsed();
    println!("\nCollected {} samples in {:.2} ms", sample_count, elapsed.as_millis());
    println!("Average sample rate: {:.1} Hz", sample_count as f64 / elapsed.as_secs_f64());

    // Stop acquisition
    println!("\nStopping data acquisition...");
    device.stop_acquisition().await?;
    println!("Data acquisition stopped");

    // Demonstrate error handling
    println!("\nTesting error handling...");
    match device.read_sample().await {
        Ok(_) => println!("Unexpected: got sample when device stopped"),
        Err(e) => println!("Expected error: {}", e),
    }

    println!("\nExample completed successfully!");
    Ok(())
}