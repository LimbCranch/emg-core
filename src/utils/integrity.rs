//! Data integrity utilities for EMG-Core
//!
//! Provides comprehensive data integrity checking functions:
//! - Checksum calculation and verification
//! - CRC-8 and CRC-16 implementations
//! - Packet integrity validation
//! - Data corruption detection
//! - Hash-based integrity verification
//!
//! All functions use constants from config modules for CRC polynomials
//! and other parameters to ensure consistency.

use std::fmt;
use crate::config::constants::serial;

/// Data integrity error types
#[derive(Debug, Clone, PartialEq)]
pub enum IntegrityError {
    /// Checksum mismatch
    ChecksumMismatch {
        expected: u8,
        actual: u8,
        context: String,
    },
    /// CRC mismatch
    CrcMismatch {
        expected: String,
        actual: String,
        algorithm: String,
        context: String,
    },
    /// Data corruption detected
    DataCorruption {
        position: usize,
        reason: String,
        context: String,
    },
    /// Invalid data length for integrity check
    InvalidLength {
        actual: usize,
        expected: usize,
        context: String,
    },
    /// Hash verification failed
    HashMismatch {
        expected: String,
        actual: String,
        algorithm: String,
    },
    /// Unsupported algorithm
    UnsupportedAlgorithm(String),
}

impl fmt::Display for IntegrityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntegrityError::ChecksumMismatch { expected, actual, context } => {
                write!(f, "Checksum mismatch in {}: expected 0x{:02X}, got 0x{:02X}", context, expected, actual)
            }
            IntegrityError::CrcMismatch { expected, actual, algorithm, context } => {
                write!(f, "{} CRC mismatch in {}: expected {}, got {}", algorithm, context, expected, actual)
            }
            IntegrityError::DataCorruption { position, reason, context } => {
                write!(f, "Data corruption at position {} in {}: {}", position, context, reason)
            }
            IntegrityError::InvalidLength { actual, expected, context } => {
                write!(f, "Invalid data length for integrity check in {}: actual {}, expected {}", context, actual, expected)
            }
            IntegrityError::HashMismatch { expected, actual, algorithm } => {
                write!(f, "{} hash mismatch: expected {}, got {}", algorithm, expected, actual)
            }
            IntegrityError::UnsupportedAlgorithm(algo) => {
                write!(f, "Unsupported integrity algorithm: {}", algo)
            }
        }
    }
}

impl std::error::Error for IntegrityError {}

/// Result type for integrity operations
pub type IntegrityResult<T> = Result<T, IntegrityError>;

/// Checksum algorithm types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChecksumType {
    /// Simple 8-bit sum
    Sum8,
    /// XOR-based checksum
    Xor8,
    /// Two's complement checksum
    TwosComplement8,
    /// CRC-8 with CCITT polynomial
    Crc8,
    /// CRC-16 with CCITT polynomial
    Crc16,
    /// CRC-32 (IEEE 802.3)
    Crc32,
}

impl ChecksumType {
    /// Get the size in bytes for this checksum type
    pub fn size_bytes(self) -> usize {
        match self {
            ChecksumType::Sum8 | ChecksumType::Xor8 | ChecksumType::TwosComplement8 | ChecksumType::Crc8 => 1,
            ChecksumType::Crc16 => 2,
            ChecksumType::Crc32 => 4,
        }
    }
}

/// Calculate simple 8-bit sum checksum
pub fn calculate_checksum(data: &[u8]) -> u8 {
    data.iter().fold(0u8, |acc, &byte| acc.wrapping_add(byte))
}

/// Verify 8-bit sum checksum
pub fn verify_checksum(data: &[u8], expected_checksum: u8, context: &str) -> IntegrityResult<()> {
    let actual_checksum = calculate_checksum(data);

    if actual_checksum != expected_checksum {
        return Err(IntegrityError::ChecksumMismatch {
            expected: expected_checksum,
            actual: actual_checksum,
            context: context.to_string(),
        });
    }

    Ok(())
}

/// Calculate XOR checksum
pub fn calculate_xor_checksum(data: &[u8]) -> u8 {
    data.iter().fold(0u8, |acc, &byte| acc ^ byte)
}

/// Calculate two's complement checksum
pub fn calculate_twos_complement_checksum(data: &[u8]) -> u8 {
    let sum = calculate_checksum(data);
    (!sum).wrapping_add(1)
}

/// CRC-8 lookup table (precomputed for performance)
static CRC8_TABLE: [u8; 256] = generate_crc8_table();

/// Generate CRC-8 lookup table at compile time
const fn generate_crc8_table() -> [u8; 256] {
    let mut table = [0u8; 256];
    let mut i = 0;

    while i < 256 {
        let mut crc = i as u8;
        let mut j = 0;

        while j < 8 {
            if (crc & 0x80) != 0 {
                crc = (crc << 1) ^ serial::CRC8_POLYNOMIAL;
            } else {
                crc <<= 1;
            }
            j += 1;
        }

        table[i] = crc;
        i += 1;
    }

    table
}

/// Calculate CRC-8 with CCITT polynomial
pub fn calculate_crc8(data: &[u8]) -> u8 {
    let mut crc = serial::CRC8_INIT_VALUE;

    for &byte in data {
        let table_index = (crc ^ byte) as usize;
        crc = CRC8_TABLE[table_index];
    }

    crc
}

/// CRC-16 lookup table (precomputed for performance)
static CRC16_TABLE: [u16; 256] = generate_crc16_table();

/// Generate CRC-16 lookup table at compile time
const fn generate_crc16_table() -> [u16; 256] {
    let mut table = [0u16; 256];
    let mut i = 0;

    while i < 256 {
        let mut crc = (i as u16) << 8;
        let mut j = 0;

        while j < 8 {
            if (crc & 0x8000) != 0 {
                crc = (crc << 1) ^ serial::CRC16_POLYNOMIAL;
            } else {
                crc <<= 1;
            }
            j += 1;
        }

        table[i] = crc;
        i += 1;
    }

    table
}

/// Calculate CRC-16 with CCITT polynomial
pub fn calculate_crc16(data: &[u8]) -> u16 {
    let mut crc = serial::CRC16_INIT_VALUE;

    for &byte in data {
        let table_index = ((crc >> 8) ^ byte as u16) as usize;
        crc = (crc << 8) ^ CRC16_TABLE[table_index];
    }

    crc
}

/// Calculate checksum using specified algorithm
pub fn calculate_checksum_with_type(data: &[u8], checksum_type: ChecksumType) -> IntegrityResult<Vec<u8>> {
    match checksum_type {
        ChecksumType::Sum8 => Ok(vec![calculate_checksum(data)]),
        ChecksumType::Xor8 => Ok(vec![calculate_xor_checksum(data)]),
        ChecksumType::TwosComplement8 => Ok(vec![calculate_twos_complement_checksum(data)]),
        ChecksumType::Crc8 => Ok(vec![calculate_crc8(data)]),
        ChecksumType::Crc16 => {
            let crc = calculate_crc16(data);
            Ok(crc.to_le_bytes().to_vec())
        }
        ChecksumType::Crc32 => {
            // Use standard CRC-32 implementation
            let crc = crc32fast::hash(data);
            Ok(crc.to_le_bytes().to_vec())
        }
    }
}

/// Verify checksum using specified algorithm
pub fn verify_checksum_with_type(
    data: &[u8],
    expected_checksum: &[u8],
    checksum_type: ChecksumType,
    context: &str,
) -> IntegrityResult<()> {
    let calculated_checksum = calculate_checksum_with_type(data, checksum_type)?;

    if calculated_checksum != expected_checksum {
        return Err(IntegrityError::CrcMismatch {
            expected: format!("{:02X?}", expected_checksum),
            actual: format!("{:02X?}", calculated_checksum),
            algorithm: format!("{:?}", checksum_type),
            context: context.to_string(),
        });
    }

    Ok(())
}

/// Comprehensive packet integrity verification
pub fn verify_packet_integrity<'a>(
    packet: &'a [u8],
    header_pattern: &[u8],
    footer_pattern: Option<&[u8]>,
    checksum_type: Option<ChecksumType>,
    context: &str,
) -> IntegrityResult<&'a [u8]> {
    // Verify minimum packet size
    let min_size = header_pattern.len() +
        footer_pattern.map_or(0, |f| f.len()) +
        checksum_type.map_or(0, |c| c.size_bytes());

    if packet.len() < min_size {
        return Err(IntegrityError::InvalidLength {
            actual: packet.len(),
            expected: min_size,
            context: format!("{}: minimum packet size", context),
        });
    }

    // Verify header pattern
    if !packet.starts_with(header_pattern) {
        return Err(IntegrityError::DataCorruption {
            position: 0,
            reason: "Header pattern mismatch".to_string(),
            context: context.to_string(),
        });
    }

    let mut data_start = header_pattern.len();
    let mut data_end = packet.len();

    // Verify footer pattern if provided
    if let Some(footer) = footer_pattern {
        if !packet.ends_with(footer) {
            return Err(IntegrityError::DataCorruption {
                position: packet.len() - footer.len(),
                reason: "Footer pattern mismatch".to_string(),
                context: context.to_string(),
            });
        }
        data_end -= footer.len();
    }

    // Verify checksum if provided
    if let Some(checksum_type) = checksum_type {
        let checksum_size = checksum_type.size_bytes();

        if data_end < checksum_size {
            return Err(IntegrityError::InvalidLength {
                actual: data_end,
                expected: checksum_size,
                context: format!("{}: checksum extraction", context),
            });
        }

        data_end -= checksum_size;
        let data_payload = &packet[data_start..data_end];
        let checksum_bytes = &packet[data_end..data_end + checksum_size];

        verify_checksum_with_type(data_payload, checksum_bytes, checksum_type, context)?;
    }

    // Return the data payload
    Ok(&packet[data_start..data_end])
}

/// Detect potential data corruption patterns
pub fn detect_data_corruption(data: &[u8], context: &str) -> IntegrityResult<()> {
    // Check for all zeros (common corruption pattern)
    if data.iter().all(|&b| b == 0) && data.len() > 4 {
        return Err(IntegrityError::DataCorruption {
            position: 0,
            reason: "All bytes are zero".to_string(),
            context: context.to_string(),
        });
    }

    // Check for all 0xFF (another common corruption pattern)
    if data.iter().all(|&b| b == 0xFF) && data.len() > 4 {
        return Err(IntegrityError::DataCorruption {
            position: 0,
            reason: "All bytes are 0xFF".to_string(),
            context: context.to_string(),
        });
    }

    // Check for repeating byte patterns that are unlikely in real data
    if data.len() >= 8 {
        let first_byte = data[0];
        if data.iter().all(|&b| b == first_byte) {
            return Err(IntegrityError::DataCorruption {
                position: 0,
                reason: format!("All bytes are identical (0x{:02X})", first_byte),
                context: context.to_string(),
            });
        }
    }

    // Check for incrementing pattern (0x00, 0x01, 0x02, ...)
    if data.len() >= 16 {
        let is_incrementing = data.windows(2).all(|pair| pair[1] == pair[0].wrapping_add(1));
        if is_incrementing {
            return Err(IntegrityError::DataCorruption {
                position: 0,
                reason: "Data appears to be incrementing test pattern".to_string(),
                context: context.to_string(),
            });
        }
    }

    Ok(())
}

/// Validate EMG signal data ranges
pub fn validate_emg_signal_integrity(samples: &[f32], context: &str) -> IntegrityResult<()> {
    for (i, &sample) in samples.iter().enumerate() {
        // Check for NaN or infinite values
        if !sample.is_finite() {
            return Err(IntegrityError::DataCorruption {
                position: i,
                reason: format!("Non-finite sample value: {}", sample),
                context: context.to_string(),
            });
        }

        // Check for unreasonably large values (likely corruption)
        if sample.abs() > 10.0 {  // 10V is unreasonably large for EMG
            return Err(IntegrityError::DataCorruption {
                position: i,
                reason: format!("Sample value {} exceeds reasonable EMG range", sample),
                context: context.to_string(),
            });
        }
    }

    // Check for DC offset issues (all samples very close to same value)
    if samples.len() > 10 {
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance = samples.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / samples.len() as f32;

        if variance < 1e-8 {  // Very low variance indicates possible corruption
            return Err(IntegrityError::DataCorruption {
                position: 0,
                reason: "Signal variance too low, possible DC offset or corruption".to_string(),
                context: context.to_string(),
            });
        }
    }

    Ok(())
}

/// Calculate and append checksum to data
pub fn append_checksum(data: &mut Vec<u8>, checksum_type: ChecksumType) -> IntegrityResult<()> {
    let checksum = calculate_checksum_with_type(data, checksum_type)?;
    data.extend_from_slice(&checksum);
    Ok(())
}

/// Remove and verify checksum from data
pub fn extract_and_verify_checksum(
    data: &mut Vec<u8>,
    checksum_type: ChecksumType,
    context: &str,
) -> IntegrityResult<()> {
    let checksum_size = checksum_type.size_bytes();

    if data.len() < checksum_size {
        return Err(IntegrityError::InvalidLength {
            actual: data.len(),
            expected: checksum_size,
            context: format!("{}: checksum extraction", context),
        });
    }

    let payload_len = data.len() - checksum_size;
    let checksum_bytes = data[payload_len..].to_vec();
    data.truncate(payload_len);

    verify_checksum_with_type(data, &checksum_bytes, checksum_type, context)?;

    Ok(())
}

/// Simple hash function for data integrity (not cryptographic)
pub fn calculate_simple_hash(data: &[u8]) -> u32 {
    let mut hash = 0x811c9dc5u32; // FNV-1a offset basis

    for &byte in data {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x01000193); // FNV-1a prime
    }

    hash
}

/// Verify data integrity using simple hash
pub fn verify_simple_hash(data: &[u8], expected_hash: u32, context: &str) -> IntegrityResult<()> {
    let actual_hash = calculate_simple_hash(data);

    if actual_hash != expected_hash {
        return Err(IntegrityError::HashMismatch {
            expected: format!("0x{:08X}", expected_hash),
            actual: format!("0x{:08X}", actual_hash),
            algorithm: "FNV-1a".to_string(),
        });
    }

    Ok(())
}

/// Comprehensive packet validation combining all checks
pub fn comprehensive_packet_validation<'a>(
    packet: &'a [u8],
    header_pattern: &[u8],
    footer_pattern: Option<&[u8]>,
    checksum_type: Option<ChecksumType>,
    context: &str,
) -> IntegrityResult<&'a [u8]> {
    // First verify packet integrity
    let payload = verify_packet_integrity(packet, header_pattern, footer_pattern, checksum_type, context)?;

    // Then check for corruption patterns
    detect_data_corruption(payload, context)?;

    Ok(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checksum_calculation() {
        let data = [0x01, 0x02, 0x03, 0x04];
        let checksum = calculate_checksum(&data);
        assert_eq!(checksum, 0x0A); // 1 + 2 + 3 + 4 = 10

        assert!(verify_checksum(&data, 0x0A, "test").is_ok());
        assert!(verify_checksum(&data, 0x0B, "test").is_err());
    }

    #[test]
    fn test_xor_checksum() {
        let data = [0x01, 0x02, 0x03, 0x04];
        let checksum = calculate_xor_checksum(&data);
        assert_eq!(checksum, 0x04); // 1 ^ 2 ^ 3 ^ 4 = 4
    }

    #[test]
    fn test_twos_complement_checksum() {
        let data = [0x01, 0x02, 0x03, 0x04];
        let checksum = calculate_twos_complement_checksum(&data);
        assert_eq!(checksum, 0xF6); // Two's complement of 0x0A
    }

    #[test]
    fn test_crc8_calculation() {
        let data = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39]; // "123456789"
        let crc = calculate_crc8(&data);
        // CRC-8-CCITT of "123456789" should be 0xF4
        assert_eq!(crc, 0xF4);
    }

    #[test]
    fn test_crc16_calculation() {
        let data = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39]; // "123456789"
        let crc = calculate_crc16(&data);
        // CRC-16-CCITT of "123456789" should be 0x29B1
        assert_eq!(crc, 0x29B1);
    }

    #[test]
    fn test_checksum_with_type() {
        let data = [0x01, 0x02, 0x03, 0x04];

        let sum8 = calculate_checksum_with_type(&data, ChecksumType::Sum8).unwrap();
        assert_eq!(sum8, vec![0x0A]);

        let xor8 = calculate_checksum_with_type(&data, ChecksumType::Xor8).unwrap();
        assert_eq!(xor8, vec![0x04]);

        let crc16 = calculate_checksum_with_type(&data, ChecksumType::Crc16).unwrap();
        assert_eq!(crc16.len(), 2);
    }

    #[test]
    fn test_packet_integrity_verification() {
        let header = [0xAA, 0x55];
        let footer = [0xDE, 0xAD];
        let data = [0x01, 0x02, 0x03, 0x04];
        let checksum = calculate_checksum(&data);

        // Construct packet: header + data + checksum + footer
        let mut packet = Vec::new();
        packet.extend_from_slice(&header);
        packet.extend_from_slice(&data);
        packet.push(checksum);
        packet.extend_from_slice(&footer);

        // Verify packet integrity
        let extracted_data = verify_packet_integrity(
            &packet,
            &header,
            Some(&footer),
            Some(ChecksumType::Sum8),
            "test"
        ).unwrap();

        assert_eq!(extracted_data, &data);
    }

    #[test]
    fn test_data_corruption_detection() {
        // Test all zeros
        let zeros = vec![0u8; 10];
        assert!(detect_data_corruption(&zeros, "test").is_err());

        // Test all 0xFF
        let ones = vec![0xFFu8; 10];
        assert!(detect_data_corruption(&ones, "test").is_err());

        // Test incrementing pattern
        let incrementing: Vec<u8> = (0..20).collect();
        assert!(detect_data_corruption(&incrementing, "test").is_err());

        // Test normal data
        let normal = [0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0];
        assert!(detect_data_corruption(&normal, "test").is_ok());
    }

    #[test]
    fn test_emg_signal_integrity() {
        // Valid EMG signal
        let valid_signal = vec![0.001, -0.002, 0.0015, -0.0008, 0.0012];
        assert!(validate_emg_signal_integrity(&valid_signal, "test").is_ok());

        // Invalid signal (NaN)
        let invalid_signal = vec![0.001, f32::NAN, 0.0015];
        assert!(validate_emg_signal_integrity(&invalid_signal, "test").is_err());

        // Invalid signal (too large)
        let large_signal = vec![0.001, 15.0, 0.0015];
        assert!(validate_emg_signal_integrity(&large_signal, "test").is_err());

        // Invalid signal (too low variance)
        let flat_signal = vec![0.001; 20];
        assert!(validate_emg_signal_integrity(&flat_signal, "test").is_err());
    }

    #[test]
    fn test_append_and_extract_checksum() {
        let mut data = vec![0x01, 0x02, 0x03, 0x04];
        let original_len = data.len();

        // Append checksum
        append_checksum(&mut data, ChecksumType::Sum8).unwrap();
        assert_eq!(data.len(), original_len + 1);

        // Extract and verify checksum
        extract_and_verify_checksum(&mut data, ChecksumType::Sum8, "test").unwrap();
        assert_eq!(data.len(), original_len);
        assert_eq!(data, vec![0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn test_simple_hash() {
        let data = [0x01, 0x02, 0x03, 0x04];
        let hash = calculate_simple_hash(&data);

        // Verify the same data produces the same hash
        assert_eq!(calculate_simple_hash(&data), hash);

        // Verify hash validation
        assert!(verify_simple_hash(&data, hash, "test").is_ok());
        assert!(verify_simple_hash(&data, hash + 1, "test").is_err());
    }

    #[test]
    fn test_comprehensive_validation() {
        let header = [0xAA, 0x55];
        let data = [0x12, 0x34, 0x56, 0x78]; // Normal looking data
        let checksum = calculate_checksum(&data);

        let mut packet = Vec::new();
        packet.extend_from_slice(&header);
        packet.extend_from_slice(&data);
        packet.push(checksum);

        let payload = comprehensive_packet_validation(
            &packet,
            &header,
            None,
            Some(ChecksumType::Sum8),
            "test"
        ).unwrap();

        assert_eq!(payload, &data);
    }

    #[test]
    fn test_checksum_type_properties() {
        assert_eq!(ChecksumType::Sum8.size_bytes(), 1);
        assert_eq!(ChecksumType::Crc16.size_bytes(), 2);
        assert_eq!(ChecksumType::Crc32.size_bytes(), 4);
    }
}