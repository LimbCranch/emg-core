// src/hal/tests.rs
//! Unit tests for HAL components
//! 

use crate::hal::{EmgDevice, EmgSample, DeviceInfo, DeviceCapabilities, QualityMetrics , DeviceType};


#[cfg(test)]
mod tests {
    use crate::ThreadPriority;
    use super::*;

    #[test]
    fn test_emg_sample_creation() {
        let sample = EmgSample {
            timestamp: 1234567890,
            sequence: 42,
            channels: vec![0.1, 0.2, 0.3, 0.4],
            quality_indicators: QualityMetrics::default(),
        };

        assert_eq!(sample.timestamp, 1234567890);
        assert_eq!(sample.sequence, 42);
        assert_eq!(sample.channels.len(), 4);
        assert_eq!(sample.channels[0], 0.1);
    }

    #[test]
    fn test_device_info_creation() {
        let info = DeviceInfo {
            name: "Test Device".to_string(),
            version: "1.0.0".to_string(),
            serial_number: "TEST123".to_string(),
            capabilities: DeviceCapabilities::default(),
        };

        assert_eq!(info.name, "Test Device");
        assert_eq!(info.version, "1.0.0");
        assert_eq!(info.serial_number, "TEST123");
    }

    #[test]
    fn test_quality_metrics_default() {
        let metrics = QualityMetrics::default();
        assert_eq!(metrics.snr_db, 0.0);
        assert!(!metrics.artifact_detected);
        assert!(!metrics.signal_saturation);
        assert!(metrics.contact_impedance_kohm.is_empty());
    }

    #[test]
    fn test_device_capabilities_default() {
        let caps = DeviceCapabilities::default();
        assert_eq!(caps.max_channels, 8);
        assert_eq!(caps.max_sample_rate_hz, 4000);
        assert!(!caps.has_builtin_filters);
        assert!(!caps.supports_impedance_check);
        assert!(!caps.supports_calibration);
    }

    /*#[tokio::test]
    async fn test_simulator_device_lifecycle() {
        let mut device = crate::hal::simulator::SimulatorDevice::healthy_user()
            .expect("Failed to create simulator");

        // Test initial state
        assert_eq!(device.get_channel_count(), 8);
        assert_eq!(device.get_sampling_rate(), 2000);

        // Test initialization
        device.initialize().await.expect("Failed to initialize");

        // Test acquisition lifecycle
        device.start_acquisition().await.expect("Failed to start");

        let sample = device.read_sample().await.expect("Failed to read sample");
        assert_eq!(sample.channels.len(), 8);
        assert!(sample.quality_indicators.snr_db > 0.0);

        device.stop_acquisition().await.expect("Failed to stop");

        // Test that reading after stop fails
        let result = device.read_sample().await;
        assert!(result.is_err());
    }*/

    #[test]
    fn test_device_type_serialization() {
        use serde_json;

        let device_type = DeviceType::Simulator;
        let json = serde_json::to_string(&device_type).expect("Failed to serialize");
        let deserialized: DeviceType = serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(device_type, deserialized);
    }

    #[test]
    fn test_thread_priority_serialization() {
        use serde_json;

        let priority = ThreadPriority::RealTime;
        let json = serde_json::to_string(&priority).expect("Failed to serialize");
        let deserialized: ThreadPriority = serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(priority, deserialized);
    }
}