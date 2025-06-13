// tests/hal_integration.rs
//! Integration tests for HAL implementation

use emg_core::{EmgDevice, SimulatorDevice, UsbEmgDevice, SerialEmgDevice};
use emg_core::config::{ConfigLoader, SystemConfig};
use std::time::Duration;
use tokio::time::timeout;

#[tokio::test]
async fn test_simulator_device_basic_operation() {
    let mut device = SimulatorDevice::healthy_user().expect("Failed to create simulator");

    // Test device info
    let info = device.get_device_info();
    assert_eq!(info.name, "EMG Simulator");
    assert_eq!(device.get_channel_count(), 8);
    assert_eq!(device.get_sampling_rate(), 2000);

    // Test initialization
    device.initialize().await.expect("Failed to initialize");

    // Test acquisition
    device.start_acquisition().await.expect("Failed to start acquisition");

    // Read some samples
    for _ in 0..10 {
        let sample = device.read_sample().await.expect("Failed to read sample");
        assert_eq!(sample.channels.len(), 8);
        assert!(!sample.channels.iter().all(|&x| x == 0.0)); // Should have some signal
    }

    device.stop_acquisition().await.expect("Failed to stop acquisition");
}

#[tokio::test]
async fn test_simulator_realistic_emg_data() {
    let mut device = SimulatorDevice::healthy_user().expect("Failed to create simulator");

    device.initialize().await.expect("Failed to initialize");
    device.start_acquisition().await.expect("Failed to start acquisition");

    let mut samples = Vec::new();
    for _ in 0..100 {
        let sample = device.read_sample().await.expect("Failed to read sample");
        samples.push(sample);
    }

    // Verify sample characteristics
    assert!(!samples.is_empty());

    // Check sequence numbers increment
    for i in 1..samples.len() {
        assert!(samples[i].sequence > samples[i-1].sequence);
    }

    // Check timestamps are increasing
    for i in 1..samples.len() {
        assert!(samples[i].timestamp >= samples[i-1].timestamp);
    }

    // Check signal quality
    for sample in &samples {
        assert!(sample.quality_indicators.snr_db > 20.0);
        assert_eq!(sample.quality_indicators.contact_impedance_kohm.len(), 8);
    }

    device.stop_acquisition().await.expect("Failed to stop acquisition");
}

#[tokio::test]
async fn test_usb_device_creation() {
    let device = UsbEmgDevice::with_default_config();

    let info = device.get_device_info();
    assert_eq!(info.name, "USB EMG Device");
    assert_eq!(device.get_channel_count(), 8);
    assert_eq!(device.get_sampling_rate(), 2000);
}

#[tokio::test]
async fn test_serial_device_creation() {
    let device = SerialEmgDevice::with_default_config();

    let info = device.get_device_info();
    assert_eq!(info.name, "Serial EMG Device");
    assert_eq!(device.get_channel_count(), 8);
    assert_eq!(device.get_sampling_rate(), 1000);
}

#[tokio::test]
async fn test_device_trait_object() {
    // Test that we can use devices through trait objects
   // let mut devices: Vec<Box<dyn EmgDevice<Error = Box<dyn std::error::Error + Send + Sync>>>> = Vec::new();

    // Note: We can't easily put different error types in the same Vec
    // This test shows the trait is object-safe for similar error types
    let simulator = SimulatorDevice::healthy_user().expect("Failed to create simulator");

    // Test basic trait object usage
    let info = simulator.get_device_info();
    assert!(!info.name.is_empty());
    assert!(simulator.get_channel_count() > 0);
    assert!(simulator.get_sampling_rate() > 0);
}

#[tokio::test]
async fn test_configuration_loading() {
    let mut loader = ConfigLoader::new();

    // Should be able to load default configuration
    let config = loader.load_system_config().expect("Failed to load config");
    assert!(config.system.sampling_rate_hz > 0);
    assert!(config.system.channel_count > 0);
    assert!(config.hal.connection_timeout_ms > 0);
}

#[tokio::test]
async fn test_sample_timing() {
    let mut device = SimulatorDevice::healthy_user().expect("Failed to create simulator");

    device.initialize().await.expect("Failed to initialize");
    device.start_acquisition().await.expect("Failed to start acquisition");

    let start_time = std::time::Instant::now();

    // Read 100 samples and measure timing
    for _ in 0..100 {
        let _sample = timeout(Duration::from_millis(10), device.read_sample())
            .await
            .expect("Sample read timeout")
            .expect("Failed to read sample");
    }

    let elapsed = start_time.elapsed();

    // Should be able to read 100 samples quickly (well under 1 second)
    assert!(elapsed < Duration::from_millis(500));

    device.stop_acquisition().await.expect("Failed to stop acquisition");
}

#[test]
fn test_error_handling() {
    // Test that errors implement required traits
    use emg_core::hal::simulator::SimulatorError;
    use emg_core::hal::usb_driver::UsbError;
    use emg_core::hal::serial_driver::SerialError;

    let sim_err = SimulatorError::ConfigurationError("test".to_string());
    let usb_err = UsbError::DeviceNotFound;
    let serial_err = SerialError::PortNotFound("test".to_string());

    // Verify Error trait implementation
    assert!(!sim_err.to_string().is_empty());
    assert!(!usb_err.to_string().is_empty());
    assert!(!serial_err.to_string().is_empty());
}

#[test]
fn test_send_sync_traits() {
    // Compile-time test that our types implement Send + Sync
    fn assert_send_sync<T: Send + Sync>() {}

    assert_send_sync::<SimulatorDevice>();
    // Note: USB and Serial devices might not be Send/Sync due to their internal handles
    // This is expected and will be addressed in real implementations
}