// src/hal/serial_driver.rs
//! Serial EMG device driver implementation

use crate::hal::{EmgDevice, EmgSample, DeviceInfo, DeviceCapabilities, QualityMetrics};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Serial device configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SerialConfig {
    pub port_name: String,
    pub baud_rate: u32,
    pub data_bits: u8,
    pub stop_bits: u8,
    pub parity: Parity,
    pub timeout_ms: u32,
    pub flow_control: FlowControl,
}

/// Serial parity settings
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Parity {
    None,
    Odd,
    Even,
}

/// Serial flow control settings
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FlowControl {
    None,
    Software,
    Hardware,
}

impl Default for SerialConfig {
    fn default() -> Self {
        Self {
            port_name: "/dev/ttyUSB0".to_string(),
            baud_rate: 115200,
            data_bits: 8,
            stop_bits: 1,
            parity: Parity::None,
            timeout_ms: 1000,
            flow_control: FlowControl::None,
        }
    }
}

/// Serial communication protocol handler
pub struct SerialProtocol {
    packet_size: usize,
    header_bytes: Vec<u8>,
    checksum_enabled: bool,
}

impl Default for SerialProtocol {
    fn default() -> Self {
        Self {
            packet_size: 32,
            header_bytes: vec![0xAA, 0x55],
            checksum_enabled: true,
        }
    }
}

/// Serial EMG device implementation
pub struct SerialEmgDevice {
    config: SerialConfig,
    protocol: SerialProtocol,
    port: Option<SerialPort>,
    is_connected: AtomicBool,
    sequence_counter: AtomicU32,
}

/// Mock serial port for compilation
struct SerialPort {
    _port_name: String,
    _baud_rate: u32,
}

#[derive(Debug)]
pub enum SerialError {
    PortNotFound(String),
    ConnectionFailed(String),
    ReadError(String),
    WriteError(String),
    ProtocolError(String),
    ConfigurationError(String),
}

impl fmt::Display for SerialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SerialError::PortNotFound(port) => write!(f, "Serial port not found: {}", port),
            SerialError::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            SerialError::ReadError(msg) => write!(f, "Read error: {}", msg),
            SerialError::WriteError(msg) => write!(f, "Write error: {}", msg),
            SerialError::ProtocolError(msg) => write!(f, "Protocol error: {}", msg),
            SerialError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl Error for SerialError {}

impl SerialEmgDevice {
    /// Create new serial device with configuration
    pub fn new(config: SerialConfig) -> Self {
        Self {
            config,
            protocol: SerialProtocol::default(),
            port: None,
            is_connected: AtomicBool::new(false),
            sequence_counter: AtomicU32::new(0),
        }
    }

    /// Create serial device with custom protocol
    pub fn with_protocol(config: SerialConfig, protocol: SerialProtocol) -> Self {
        Self {
            config,
            protocol,
            port: None,
            is_connected: AtomicBool::new(false),
            sequence_counter: AtomicU32::new(0),
        }
    }

    /// Create serial device with default configuration
    pub fn with_default_config() -> Self {
        Self::new(SerialConfig::default())
    }

    fn open_port(&mut self) -> Result<(), SerialError> {
        // TODO: Implement actual serial port opening
        // This is a stub implementation
        let port = SerialPort {
            _port_name: self.config.port_name.clone(),
            _baud_rate: self.config.baud_rate,
        };

        self.port = Some(port);
        self.is_connected.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn read_packet(&self) -> Result<Vec<u8>, SerialError> {
        if !self.is_connected.load(Ordering::Relaxed) {
            return Err(SerialError::ReadError("Port not connected".to_string()));
        }

        // Mock packet data with proper size calculation
        let mut packet = self.protocol.header_bytes.clone();
        packet.extend_from_slice(&[0u8; 32]); // Mock EMG data

        if self.protocol.checksum_enabled {
            let checksum = packet.iter().fold(0u8, |acc, &b| acc.wrapping_add(b));
            packet.push(checksum);
        }

        // FIX: Validate packet before returning
        let expected_min_size = self.protocol.header_bytes.len() + 32;
        if packet.len() < expected_min_size {
            return Err(SerialError::ProtocolError(
                format!("Generated packet too small: {} < {}", packet.len(), expected_min_size)
            ));
        }

        Ok(packet)
    }

    fn parse_packet(&self, packet: &[u8]) -> Result<Vec<f32>, SerialError> {
        // FIX: Comprehensive bounds checking before any slice operations
        let header_len = self.protocol.header_bytes.len();
        let expected_data_len = 32; // 8 channels * 4 bytes each
        let min_packet_len = header_len + expected_data_len;

        // FIX: Check minimum packet length
        if packet.len() < min_packet_len {
            return Err(SerialError::ProtocolError(
                format!("Packet too short: expected at least {} bytes, got {}",
                        min_packet_len, packet.len())
            ));
        }

        // FIX: Verify header with proper bounds checking
        if packet.len() < header_len || !packet[..header_len].eq(&self.protocol.header_bytes) {
            return Err(SerialError::ProtocolError("Invalid packet header".to_string()));
        }

        // FIX: Safe slice operations with validated bounds
        let data_start = header_len;
        let data_end = data_start + expected_data_len;

        // FIX: Double-check bounds before slicing
        if packet.len() < data_end {
            return Err(SerialError::ProtocolError(
                format!("Insufficient data: need {} bytes, packet has {}",
                        data_end, packet.len())
            ));
        }

        let emg_data = &packet[data_start..data_end];

        // FIX: Safe channel conversion with bounds checking
        let mut channels = Vec::with_capacity(8);
        for i in 0..8 {
            let byte_start = i * 4;
            let byte_end = byte_start + 4;

            // FIX: Verify we have enough bytes for this channel
            if emg_data.len() < byte_end {
                return Err(SerialError::ProtocolError(
                    format!("Insufficient data for channel {}: need {} bytes", i, byte_end)
                ));
            }

            let bytes = &emg_data[byte_start..byte_end];
            // FIX: Safe conversion with error handling
            let channel_value = (bytes[0] as f32 - 128.0) / 128.0;
            channels.push(channel_value);
        }

        Ok(channels)
    }

    fn send_command(&self, command: &[u8]) -> Result<(), SerialError> {
        // TODO: Implement actual command sending
        if !self.is_connected.load(Ordering::Relaxed) {
            return Err(SerialError::WriteError("Port not connected".to_string()));
        }

        // Mock command sending
        let _ = command; // Suppress unused warning
        Ok(())
    }
}

impl EmgDevice for SerialEmgDevice {
    type Error = SerialError;

    async fn initialize(&mut self) -> Result<(), Self::Error> {
        self.open_port()?;

        // Send initialization command
        let init_command = [0x01, 0x00]; // Mock init command
        self.send_command(&init_command)?;

        Ok(())
    }

    async fn start_acquisition(&mut self) -> Result<(), Self::Error> {
        if !self.is_connected.load(Ordering::Relaxed) {
            return Err(SerialError::ConnectionFailed("Port not connected".to_string()));
        }

        // Send start command
        let start_command = [0x02, 0x01]; // Mock start command
        self.send_command(&start_command)?;

        Ok(())
    }

    async fn stop_acquisition(&mut self) -> Result<(), Self::Error> {
        if !self.is_connected.load(Ordering::Relaxed) {
            return Err(SerialError::ConnectionFailed("Port not connected".to_string()));
        }

        // Send stop command
        let stop_command = [0x02, 0x00]; // Mock stop command
        self.send_command(&stop_command)?;

        Ok(())
    }

    async fn read_sample(&mut self) -> Result<EmgSample, Self::Error> {
        let packet = self.read_packet()?;
        let channels = self.parse_packet(&packet)?;
        let sequence = self.sequence_counter.fetch_add(1, Ordering::Relaxed);

        // Generate basic quality metrics
        let quality_indicators = QualityMetrics {
            snr_db: 28.0, // TODO: Calculate from actual signal
            contact_impedance_kohm: vec![12.0; channels.len()],
            artifact_detected: false,
            signal_saturation: channels.iter().any(|&x| x.abs() > 0.9),
        };

        Ok(EmgSample {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64,
            sequence,
            channels,
            quality_indicators,
        })
    }

    fn get_device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: "Serial EMG Device".to_string(),
            version: "1.0.0".to_string(),
            serial_number: format!("SERIAL-{}", self.config.port_name.replace("/", "-")),
            capabilities: DeviceCapabilities {
                max_channels: 8,
                max_sample_rate_hz: 1000,
                has_builtin_filters: false,
                supports_impedance_check: false,
                supports_calibration: true,
            },
        }
    }

    fn get_channel_count(&self) -> usize {
        8 // TODO: Make configurable
    }

    fn get_sampling_rate(&self) -> u32 {
        1000 // TODO: Make configurable based on device
    }
}