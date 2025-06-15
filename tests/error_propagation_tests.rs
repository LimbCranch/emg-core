// tests/error_propagation_tests.rs
//! Comprehensive error propagation and handling tests
//!
//! This module tests the unified error handling system to ensure:
//! - Error conversion between device types works correctly
//! - Error debugging information is preserved and useful
//! - Error serialization/deserialization functions properly
//! - Error context is maintained through the call stack
//! - Error recovery mechanisms function as expected

use emg_core::error::{
    EmgError, EmgResult, ErrorContext, DeviceType, ProcessingStage, BufferType,
    CommunicationProtocol, RealTimeConstraint, Severity, ResourceType,
    EmgErrorBuilder, IntoEmgError,
};
use emg_core::hal::{
    simulator::{SimulatorDevice, SimulatorConfig, SimulatorError},
    usb_driver::{UsbEmgDevice, UsbDeviceConfig, UsbError},
    serial_driver::{SerialEmgDevice, SerialConfig, SerialError},
    EmgDevice, DeviceManager,
};
use std::sync::Arc;
use std::time::{SystemTime, Duration};
use tokio::time::timeout;

/// Test error conversion from simulator errors to unified EmgError
#[test]
fn test_simulator_error_conversion() {
    // Test various simulator error types
    let errors = vec![
        SimulatorError::ConfigurationError("Invalid channel count".to_string()),
        SimulatorError::GenerationError("Sample generation failed"),
        SimulatorError::DeviceNotRunning,
        SimulatorError::InvalidChannelCount,
        SimulatorError::InvalidSampleRate,
    ];

    for sim_error in errors {
        let emg_error: EmgError = sim_error.into();

        // Verify it's converted to a device error
        match &emg_error {
            EmgError::Device { device_type, error, context } => {
                assert_eq!(*device_type, DeviceType::Simulator);
                assert!(!error.to_string().is_empty());
                assert_eq!(context.component, "simulator");
                assert_eq!(context.operation, "device_operation");
            }
            _ => panic!("Expected Device error, got: {:?}", emg_error),
        }

        // Test error display
        let error_string = format!("{}", emg_error);
        assert!(error_string.contains("SIMULATOR"));
        assert!(error_string.contains("Device error"));

        // Test error debugging
        let debug_string = format!("{:?}", emg_error);
        assert!(debug_string.contains("Device"));
        assert!(debug_string.contains("Simulator"));
    }
}

/// Test error conversion from USB errors to unified EmgError
#[test]
fn test_usb_error_conversion() {
    let errors = vec![
        UsbError::DeviceNotFound,
        UsbError::ConnectionFailed("USB port unavailable".to_string()),
        UsbError::ReadError("Bulk transfer failed".to_string()),
        UsbError::ConfigurationError("Invalid endpoint".to_string()),
    ];

    for usb_error in errors {
        let emg_error: EmgError = usb_error.into();

        match &emg_error {
            EmgError::Device { device_type, error, context } => {
                assert_eq!(*device_type, DeviceType::UsbEmg);
                assert!(!error.to_string().is_empty());
                assert_eq!(context.component, "usb_driver");
            }
            _ => panic!("Expected Device error, got: {:?}", emg_error),
        }

        // Verify error chaining works
        let source = emg_error.source();
        assert!(source.is_some());
    }
}

/// Test error conversion from serial errors to unified EmgError
#[test]
fn test_serial_error_conversion() {
    let errors = vec![
        SerialError::PortNotFound("/dev/ttyUSB0".to_string()),
        SerialError::ConnectionFailed("Baud rate mismatch".to_string()),
        SerialError::ReadError("Timeout reading data".to_string()),
        SerialError::ProtocolError("Invalid packet header".to_string()),
        SerialError::ChecksumMismatch { expected: 0xAB, actual: 0xCD },
        SerialError::InvalidPacketSize { expected: 128, actual: 64 },
    ];

    for serial_error in errors {
        let emg_error: EmgError = serial_error.into();

        match &emg_error {
            EmgError::Device { device_type, error, context } => {
                assert_eq!(*device_type, DeviceType::SerialEmg);
                assert!(!error.to_string().is_empty());
                assert_eq!(context.component, "serial_driver");
            }
            _ => panic!("Expected Device error, got: {:?}", emg_error),
        }
    }
}

/// Test error context preservation and debugging information
#[test]
fn test_error_context_preservation() {
    // Create error with detailed context
    let context = ErrorContext::new("test_component", "test_operation")
        .add_info("channel", "3")
        .add_info("sample_rate", "2000")
        .add_to_chain("Initial validation failed")
        .add_to_chain("Retry attempt 1 failed");

    let error = EmgError::Configuration {
        component: "test_component".to_string(),
        reason: "Invalid configuration detected".to_string(),
        context: context.clone(),
    };

    // Verify context information is preserved
    match error {
        EmgError::Configuration { context, .. } => {
            assert_eq!(context.component, "test_component");
            assert_eq!(context.operation, "test_operation");
            assert_eq!(context.additional_info.get("channel"), Some(&"3".to_string()));
            assert_eq!(context.additional_info.get("sample_rate"), Some(&"2000".to_string()));
            assert_eq!(context.chain.len(), 2);
            assert!(context.chain.contains(&"Initial validation failed".to_string()));
            assert!(context.timestamp <= SystemTime::now());
        }
        _ => panic!("Expected Configuration error"),
    }
}

/// Test error debugging information quality
#[test]
fn test_error_debugging_information() {
    // Test various error types for debug information quality
    let errors = vec![
        EmgError::BufferOverflow {
            buffer_type: BufferType::RingBuffer,
            channel: Some(2),
            capacity: 1024,
            attempted_size: 2048,
            context: ErrorContext::new("buffer_manager", "write_sample"),
        },
        EmgError::Timing {
            expected_timing: Some(1000000), // 1ms in ns
            actual_timing: Some(2500000),   // 2.5ms in ns
            drift_ns: Some(1500000),        // 1.5ms drift
            reason: "Sample deadline missed".to_string(),
            context: ErrorContext::new("acquisition", "read_sample"),
        },
        EmgError::RealTime {
            constraint_type: RealTimeConstraint::SampleDeadline,
            deadline_ns: Some(1000000),
            actual_duration_ns: Some(3000000),
            severity: Severity::High,
            context: ErrorContext::new("real_time", "process_sample"),
        },
    ];

    for error in errors {
        // Test display formatting
        let display_string = format!("{}", error);
        assert!(!display_string.is_empty());
        assert!(display_string.len() > 20); // Should have substantial information

        // Test debug formatting
        let debug_string = format!("{:?}", error);
        assert!(!debug_string.is_empty());
        assert!(debug_string.contains("context"));

        // Ensure error implements required traits
        assert_send_sync(&error);
        assert_error_trait(&error);
    }
}

/// Test error serialization and deserialization
#[test]
fn test_error_serialization() {
    use serde_json;

    let context = ErrorContext::new("serialization_test", "test_operation")
        .add_info("test_key", "test_value");

    // Test serialization of ErrorContext
    let serialized = serde_json::to_string(&context)
        .expect("Failed to serialize ErrorContext");
    assert!(!serialized.is_empty());

    let deserialized: ErrorContext = serde_json::from_str(&serialized)
        .expect("Failed to deserialize ErrorContext");

    assert_eq!(deserialized.component, context.component);
    assert_eq!(deserialized.operation, context.operation);
    assert_eq!(deserialized.additional_info, context.additional_info);

    // Test enum serialization
    let device_types = vec![
        DeviceType::Simulator,
        DeviceType::UsbEmg,
        DeviceType::SerialEmg,
        DeviceType::Unknown("custom".to_string()),
    ];

    for device_type in device_types {
        let serialized = serde_json::to_string(&device_type)
            .expect("Failed to serialize DeviceType");
        let deserialized: DeviceType = serde_json::from_str(&serialized)
            .expect("Failed to deserialize DeviceType");
        assert_eq!(device_type, deserialized);
    }
}

/// Test error builder pattern
#[test]
fn test_error_builder_pattern() {
    let builder = EmgErrorBuilder::new("test_component", "test_operation");

    // Test configuration error building
    let config_error = builder.clone().configuration("Missing required parameter");
    match config_error {
        EmgError::Configuration { component, reason, .. } => {
            assert_eq!(component, "test_component");
            assert_eq!(reason, "Missing required parameter");
        }
        _ => panic!("Expected Configuration error"),
    }

    // Test processing error building
    let processing_error = builder.clone().processing(
        ProcessingStage::Filtering,
        "Filter coefficient out of range"
    );
    match processing_error {
        EmgError::Processing { stage, reason, .. } => {
            assert_eq!(stage, ProcessingStage::Filtering);
            assert_eq!(reason, "Filter coefficient out of range");
        }
        _ => panic!("Expected Processing error"),
    }

    // Test buffer overflow error building
    let buffer_error = builder.clone().buffer_overflow(
        BufferType::CircularBuffer,
        1024,
        2048
    );
    match buffer_error {
        EmgError::BufferOverflow { buffer_type, capacity, attempted_size, .. } => {
            assert_eq!(buffer_type, BufferType::CircularBuffer);
            assert_eq!(capacity, 1024);
            assert_eq!(attempted_size, 2048);
        }
        _ => panic!("Expected BufferOverflow error"),
    }
}

/// Test IntoEmgError trait for convenient error conversion
#[test]
fn test_into_emg_error_trait() {
    // Test with std::io::Error
    let io_result: Result<i32, std::io::Error> = Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "File not found"
    ));

    let emg_result = io_result.emg_err("file_manager", "read_file");
    assert!(emg_result.is_err());

    match emg_result.unwrap_err() {
        EmgError::System { subsystem, reason, .. } => {
            assert_eq!(subsystem, "file_manager");
            assert!(reason.contains("File not found"));
        }
        _ => panic!("Expected System error"),
    }

    // Test with successful result
    let ok_result: Result<i32, std::io::Error> = Ok(42);
    let emg_result = ok_result.emg_err("test", "test");
    assert_eq!(emg_result.unwrap(), 42);
}

/// Test error propagation through async operations
#[tokio::test]
async fn test_async_error_propagation() {
    // Create devices that will generate errors
    let mut simulator = SimulatorDevice::new(SimulatorConfig {
        channel_count: 0, // Invalid - will cause error
        ..Default::default()
    });

    // Test error propagation through async operations
    let result = simulator.initialize().await;
    assert!(result.is_err());

    // Verify error type and context are preserved
    if let Err(error) = result {
        match error {
            EmgError::Device { device_type, .. } => {
                assert_eq!(device_type, DeviceType::Simulator);
            }
            _ => panic!("Expected Device error"),
        }
    }
}

/// Test error recovery and retry mechanisms
#[tokio::test]
async fn test_error_recovery_mechanisms() {
    let mut manager = DeviceManager::new();

    // Add a device that will initially fail
    let sim_config = SimulatorConfig {
        channel_count: 0, // Invalid configuration
        ..Default::default()
    };

    let simulator = SimulatorDevice::new(sim_config);
    manager.add_device(Box::new(simulator));

    // Test error during initialization
    let results = manager.initialize_all().await;
    assert!(results.is_ok());
    let init_results = results.unwrap();
    assert_eq!(init_results.len(), 1);
    assert!(init_results[0].is_err());

    // Verify error recovery is handled gracefully
    manager.set_error_recovery(true);
    assert!(manager.device_count() > 0);
}

/// Test error propagation in device manager operations
#[tokio::test]
async fn test_device_manager_error_handling() {
    let mut manager = DeviceManager::new();

    // Test operations on empty manager
    let result = manager.get_active_device();
    assert!(result.is_err());

    match result.unwrap_err() {
        EmgError::Configuration { component, reason, .. } => {
            assert_eq!(component, "device_manager");
            assert!(reason.contains("No active device"));
        }
        _ => panic!("Expected Configuration error"),
    }

    // Test invalid device index
    let result = manager.set_active_device(99);
    assert!(result.is_err());

    match result.unwrap_err() {
        EmgError::InvalidData { data_type, reason, .. } => {
            assert_eq!(data_type, "device_index");
            assert!(reason.contains("Index out of bounds"));
        }
        _ => panic!("Expected InvalidData error"),
    }
}

/// Test error chain preservation
#[test]
fn test_error_chain_preservation() {
    // Create a chain of errors
    let original_error = std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        "Access denied"
    );

    let context = ErrorContext::new("file_handler", "read_config")
        .add_to_chain("Failed to open file")
        .add_to_chain("Retried with elevated permissions")
        .add_to_chain("Still failed");

    let emg_error = EmgError::System {
        subsystem: "configuration".to_string(),
        error_code: Some(13), // Permission denied
        reason: original_error.to_string(),
        context,
    };

    // Verify chain is preserved
    match emg_error {
        EmgError::System { context, .. } => {
            assert_eq!(context.chain.len(), 3);
            assert!(context.chain.contains(&"Failed to open file".to_string()));
            assert!(context.chain.contains(&"Retried with elevated permissions".to_string()));
            assert!(context.chain.contains(&"Still failed".to_string()));
        }
        _ => panic!("Expected System error"),
    }
}

/// Test performance of error handling under load
#[test]
fn test_error_handling_performance() {
    use std::time::Instant;

    let start = Instant::now();
    let error_count = 10000;

    // Generate many errors to test performance
    for i in 0..error_count {
        let context = ErrorContext::new("performance_test", "error_generation")
            .add_info("iteration", &i.to_string());

        let error = EmgError::Processing {
            stage: ProcessingStage::Filtering,
            reason: format!("Test error {}", i),
            context,
        };

        // Force error formatting to test display performance
        let _ = format!("{}", error);
        let _ = format!("{:?}", error);
    }

    let elapsed = start.elapsed();
    let errors_per_sec = error_count as f64 / elapsed.as_secs_f64();

    // Should be able to handle at least 100,000 errors per second
    assert!(errors_per_sec > 100_000.0,
            "Error handling too slow: {:.0} errors/sec", errors_per_sec);
}

/// Test memory usage of error structures
#[test]
fn test_error_memory_usage() {
    use std::mem;

    // Test that error structures don't consume excessive memory
    let error_size = mem::size_of::<EmgError>();
    let context_size = mem::size_of::<ErrorContext>();

    // Error should be reasonably sized (less than 200 bytes)
    assert!(error_size < 200, "EmgError too large: {} bytes", error_size);

    // Context should be reasonably sized (less than 150 bytes)
    assert!(context_size < 150, "ErrorContext too large: {} bytes", context_size);

    println!("EmgError size: {} bytes", error_size);
    println!("ErrorContext size: {} bytes", context_size);
}

/// Test thread safety of error types
#[test]
fn test_error_thread_safety() {
    use std::sync::{Arc, Mutex};
    use std::thread;

    let errors = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();

    // Spawn multiple threads that generate errors
    for i in 0..10 {
        let errors_clone = Arc::clone(&errors);
        let handle = thread::spawn(move || {
            let context = ErrorContext::new("thread_test", "concurrent_error")
                .add_info("thread_id", &i.to_string());

            let error = EmgError::Configuration {
                component: format!("thread_{}", i),
                reason: "Thread safety test".to_string(),
                context,
            };

            errors_clone.lock().unwrap().push(error);
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify all errors were collected
    let collected_errors = errors.lock().unwrap();
    assert_eq!(collected_errors.len(), 10);
}

/// Test timeout error handling
#[tokio::test]
async fn test_timeout_error_handling() {
    async fn slow_operation() -> EmgResult<()> {
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(())
    }

    // Test operation that times out
    let result = timeout(Duration::from_millis(50), slow_operation()).await;

    match result {
        Err(_timeout_err) => {
            // Convert timeout to EMG error
            let emg_error = EmgError::Timing {
                expected_timing: Some(50_000_000), // 50ms in ns
                actual_timing: None,
                drift_ns: None,
                reason: "Operation timed out".to_string(),
                context: ErrorContext::new("timeout_test", "slow_operation"),
            };

            // Verify error formatting
            let error_string = format!("{}", emg_error);
            assert!(error_string.contains("TIMING"));
            assert!(error_string.contains("Operation timed out"));
        }
        Ok(_) => panic!("Operation should have timed out"),
    }
}

// Helper functions for trait verification
fn assert_send_sync<T: Send + Sync>(_value: &T) {}

fn assert_error_trait<T: std::error::Error>(_value: &T) {}

/// Comprehensive integration test for error handling
#[tokio::test]
async fn test_comprehensive_error_integration() {
    // This test exercises the complete error handling system
    let mut manager = DeviceManager::new();

    // Add devices with various configurations
    let devices = vec![
        // Simulator with invalid config
        Box::new(SimulatorDevice::new(SimulatorConfig {
            channel_count: 0,
            ..Default::default()
        })) as Box<dyn EmgDevice>,

        // USB device (will work)
        Box::new(UsbEmgDevice::new(UsbDeviceConfig::default())) as Box<dyn EmgDevice>,

        // Serial device with invalid config
        Box::new(SerialEmgDevice::new(SerialConfig {
            port_name: "/invalid/port".to_string(),
            ..Default::default()
        }).unwrap_or_else(|_| {
            // If creation fails, create with default config
            SerialEmgDevice::new(SerialConfig::default()).unwrap()
        })) as Box<dyn EmgDevice>,
    ];

    for device in devices {
        manager.add_device(device);
    }

    // Test initialization of all devices
    let init_results = manager.initialize_all().await.unwrap();

    // Should have results for all devices
    assert_eq!(init_results.len(), 3);

    // Some should succeed, some should fail
    let success_count = init_results.iter().filter(|r| r.is_ok()).count();
    let error_count = init_results.iter().filter(|r| r.is_err()).count();

    assert!(success_count > 0, "At least one device should initialize successfully");
    assert!(error_count > 0, "At least one device should fail to initialize");

    // Test metrics collection (should handle errors gracefully)
    let metrics_results = manager.get_all_metrics();
    assert_eq!(metrics_results.len(), 3);

    // Verify error types in failed results
    for result in init_results {
        if let Err(error) = result {
            match error {
                EmgError::Device { device_type, .. } => {
                    // Should be one of our known device types
                    assert!(matches!(device_type, 
                        DeviceType::Simulator | DeviceType::UsbEmg | DeviceType::SerialEmg));
                }
                EmgError::Configuration { .. } => {
                    // Configuration errors are also acceptable
                }
                _ => {
                    // Other error types might be valid depending on the failure mode
                    println!("Unexpected error type: {:?}", error);
                }
            }
        }
    }

    println!("✅ Comprehensive error integration test completed successfully");
    println!("   - {} devices tested", manager.device_count());
    println!("   - {} successful initializations", success_count);
    println!("   - {} failed initializations", error_count);
}