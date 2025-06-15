//! Conversion utilities for EMG-Core
//!
//! Provides functions for converting between different data formats
//! commonly used in EMG systems:
//! - ADC values to voltage conversions
//! - Sample format conversions
//! - Time/frequency conversions
//! - Unit conversions for biomedical signals
//!
//! All conversion functions use constants from config modules
//! to ensure consistency and avoid magic numbers.

use std::fmt;
use crate::config::constants::{signal, serial};

/// Conversion error types
#[derive(Debug, Clone, PartialEq)]
pub enum ConversionError {
    /// Numeric overflow during conversion
    NumericOverflow {
        operation: String,
        input: String,
    },
    /// Invalid input value for conversion
    InvalidInput {
        function: String,
        input: String,
        reason: String,
    },
    /// Precision loss warning
    PrecisionLoss {
        function: String,
        original_precision: u32,
        result_precision: u32,
    },
    /// Unsupported conversion type
    UnsupportedConversion {
        from: String,
        to: String,
    },
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConversionError::NumericOverflow { operation, input } => {
                write!(f, "Numeric overflow in {}: input {}", operation, input)
            }
            ConversionError::InvalidInput { function, input, reason } => {
                write!(f, "Invalid input for {}: {} ({})", function, input, reason)
            }
            ConversionError::PrecisionLoss { function, original_precision, result_precision } => {
                write!(f, "Precision loss in {}: {} bits -> {} bits", function, original_precision, result_precision)
            }
            ConversionError::UnsupportedConversion { from, to } => {
                write!(f, "Unsupported conversion from {} to {}", from, to)
            }
        }
    }
}

impl std::error::Error for ConversionError {}

/// Result type for conversion operations
pub type ConversionResult<T> = Result<T, ConversionError>;

/// ADC resolution types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdcResolution {
    /// 8-bit ADC (0-255)
    Bits8,
    /// 12-bit ADC (0-4095)  
    Bits12,
    /// 16-bit ADC (0-65535)
    Bits16,
    /// 24-bit ADC (0-16777215)
    Bits24,
    /// 32-bit ADC (signed)
    Bits32,
}

impl AdcResolution {
    /// Get maximum value for this resolution
    pub fn max_value(self) -> u32 {
        match self {
            AdcResolution::Bits8 => 255,
            AdcResolution::Bits12 => 4095,
            AdcResolution::Bits16 => 65535,
            AdcResolution::Bits24 => 16777215,
            AdcResolution::Bits32 => u32::MAX,
        }
    }

    /// Get scale factor for conversion
    pub fn scale_factor(self) -> f32 {
        match self {
            AdcResolution::Bits8 => serial::ADC_8BIT_SCALE,
            AdcResolution::Bits12 => 2048.0,
            AdcResolution::Bits16 => serial::ADC_16BIT_SCALE,
            AdcResolution::Bits24 => serial::ADC_24BIT_SCALE,
            AdcResolution::Bits32 => 2147483648.0,
        }
    }

    /// Get number of bytes per sample
    pub fn bytes_per_sample(self) -> usize {
        match self {
            AdcResolution::Bits8 => 1,
            AdcResolution::Bits12 => 2, // Typically stored in 2 bytes
            AdcResolution::Bits16 => 2,
            AdcResolution::Bits24 => 3,
            AdcResolution::Bits32 => 4,
        }
    }
}

/// Sample format types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleFormat {
    /// Unsigned integer
    UnsignedInt,
    /// Signed integer (two's complement)
    SignedInt,
    /// IEEE 754 floating point
    Float32,
    /// IEEE 754 double precision
    Float64,
}

/// Convert ADC value to voltage
pub fn adc_to_voltage(
    adc_value: u32,
    resolution: AdcResolution,
    reference_voltage: f32,
    gain: f32,
) -> ConversionResult<f32> {
    if adc_value > resolution.max_value() {
        return Err(ConversionError::InvalidInput {
            function: "adc_to_voltage".to_string(),
            input: adc_value.to_string(),
            reason: format!("ADC value exceeds maximum for {:?}", resolution),
        });
    }

    if reference_voltage <= 0.0 {
        return Err(ConversionError::InvalidInput {
            function: "adc_to_voltage".to_string(),
            input: reference_voltage.to_string(),
            reason: "Reference voltage must be positive".to_string(),
        });
    }

    if gain <= 0.0 {
        return Err(ConversionError::InvalidInput {
            function: "adc_to_voltage".to_string(),
            input: gain.to_string(),
            reason: "Gain must be positive".to_string(),
        });
    }

    let normalized = adc_value as f32 / resolution.max_value() as f32;
    let voltage = (normalized * reference_voltage) / gain;

    Ok(voltage)
}

/// Convert voltage to ADC value
pub fn voltage_to_adc(
    voltage: f32,
    resolution: AdcResolution,
    reference_voltage: f32,
    gain: f32,
) -> ConversionResult<u32> {
    if reference_voltage <= 0.0 {
        return Err(ConversionError::InvalidInput {
            function: "voltage_to_adc".to_string(),
            input: reference_voltage.to_string(),
            reason: "Reference voltage must be positive".to_string(),
        });
    }

    if gain <= 0.0 {
        return Err(ConversionError::InvalidInput {
            function: "voltage_to_adc".to_string(),
            input: gain.to_string(),
            reason: "Gain must be positive".to_string(),
        });
    }

    let amplified_voltage = voltage * gain;
    let normalized = amplified_voltage / reference_voltage;

    if normalized < 0.0 || normalized > 1.0 {
        return Err(ConversionError::InvalidInput {
            function: "voltage_to_adc".to_string(),
            input: voltage.to_string(),
            reason: "Voltage out of ADC range after amplification".to_string(),
        });
    }

    let adc_value = (normalized * resolution.max_value() as f32).round() as u32;
    Ok(adc_value.min(resolution.max_value()))
}

/// Convert signed 24-bit ADC value (common in EMG devices)
pub fn signed_24bit_to_voltage(raw_bytes: &[u8], reference_voltage: f32, gain: f32) -> ConversionResult<f32> {
    if raw_bytes.len() != 3 {
        return Err(ConversionError::InvalidInput {
            function: "signed_24bit_to_voltage".to_string(),
            input: format!("{} bytes", raw_bytes.len()),
            reason: "Expected exactly 3 bytes for 24-bit value".to_string(),
        });
    }

    // Convert 3 bytes to signed 24-bit value
    let mut value = ((raw_bytes[2] as u32) << 16) |
        ((raw_bytes[1] as u32) << 8) |
        (raw_bytes[0] as u32);

    // Sign extend if negative (MSB set)
    if (raw_bytes[2] & serial::ADC_24BIT_SIGN_MASK) != 0 {
        value |= serial::ADC_24BIT_SIGN_EXTEND as u32;
    }

    // Convert to signed integer
    let signed_value = value as i32;

    // Convert to voltage
    let normalized = signed_value as f32 / serial::ADC_24BIT_SCALE;
    let voltage = (normalized * reference_voltage) / gain;

    Ok(voltage)
}

/// Convert samples to byte array
pub fn samples_to_bytes(
    samples: &[f32],
    format: SampleFormat,
    resolution: AdcResolution,
) -> ConversionResult<Vec<u8>> {
    let mut bytes = Vec::new();

    for &sample in samples {
        match format {
            SampleFormat::UnsignedInt => {
                let adc_value = voltage_to_adc(sample, resolution, 1.0, 1.0)?;
                match resolution {
                    AdcResolution::Bits8 => bytes.push(adc_value as u8),
                    AdcResolution::Bits16 => {
                        bytes.extend_from_slice(&(adc_value as u16).to_le_bytes());
                    }
                    AdcResolution::Bits24 => {
                        let value_bytes = (adc_value as u32).to_le_bytes();
                        bytes.extend_from_slice(&value_bytes[0..3]);
                    }
                    AdcResolution::Bits32 => {
                        bytes.extend_from_slice(&adc_value.to_le_bytes());
                    }
                    _ => return Err(ConversionError::UnsupportedConversion {
                        from: format!("{:?}", resolution),
                        to: "bytes".to_string(),
                    }),
                }
            }
            SampleFormat::SignedInt => {
                let signed_value = (sample * resolution.scale_factor()) as i32;
                match resolution {
                    AdcResolution::Bits16 => {
                        bytes.extend_from_slice(&(signed_value as i16).to_le_bytes());
                    }
                    AdcResolution::Bits24 => {
                        let value_bytes = signed_value.to_le_bytes();
                        bytes.extend_from_slice(&value_bytes[0..3]);
                    }
                    AdcResolution::Bits32 => {
                        bytes.extend_from_slice(&signed_value.to_le_bytes());
                    }
                    _ => return Err(ConversionError::UnsupportedConversion {
                        from: format!("{:?} signed", resolution),
                        to: "bytes".to_string(),
                    }),
                }
            }
            SampleFormat::Float32 => {
                bytes.extend_from_slice(&sample.to_le_bytes());
            }
            SampleFormat::Float64 => {
                bytes.extend_from_slice(&(sample as f64).to_le_bytes());
            }
        }
    }

    Ok(bytes)
}

/// Convert byte array to samples
pub fn bytes_to_samples(
    bytes: &[u8],
    format: SampleFormat,
    resolution: AdcResolution,
    reference_voltage: f32,
    gain: f32,
) -> ConversionResult<Vec<f32>> {
    let bytes_per_sample = match format {
        SampleFormat::UnsignedInt | SampleFormat::SignedInt => resolution.bytes_per_sample(),
        SampleFormat::Float32 => 4,
        SampleFormat::Float64 => 8,
    };

    if bytes.len() % bytes_per_sample != 0 {
        return Err(ConversionError::InvalidInput {
            function: "bytes_to_samples".to_string(),
            input: format!("{} bytes", bytes.len()),
            reason: format!("Length not multiple of {} bytes per sample", bytes_per_sample),
        });
    }

    let sample_count = bytes.len() / bytes_per_sample;
    let mut samples = Vec::with_capacity(sample_count);

    for i in 0..sample_count {
        let start = i * bytes_per_sample;
        let sample_bytes = &bytes[start..start + bytes_per_sample];

        let sample = match format {
            SampleFormat::UnsignedInt => {
                let adc_value = match resolution {
                    AdcResolution::Bits8 => sample_bytes[0] as u32,
                    AdcResolution::Bits16 => u16::from_le_bytes([sample_bytes[0], sample_bytes[1]]) as u32,
                    AdcResolution::Bits24 => {
                        ((sample_bytes[2] as u32) << 16) |
                            ((sample_bytes[1] as u32) << 8) |
                            (sample_bytes[0] as u32)
                    }
                    AdcResolution::Bits32 => u32::from_le_bytes([
                        sample_bytes[0], sample_bytes[1],
                        sample_bytes[2], sample_bytes[3]
                    ]),
                    _ => return Err(ConversionError::UnsupportedConversion {
                        from: "bytes".to_string(),
                        to: format!("{:?}", resolution),
                    }),
                };
                adc_to_voltage(adc_value, resolution, reference_voltage, gain)?
            }
            SampleFormat::SignedInt => {
                match resolution {
                    AdcResolution::Bits16 => {
                        let signed_value = i16::from_le_bytes([sample_bytes[0], sample_bytes[1]]);
                        (signed_value as f32 / resolution.scale_factor()) * reference_voltage / gain
                    }
                    AdcResolution::Bits24 => {
                        signed_24bit_to_voltage(sample_bytes, reference_voltage, gain)?
                    }
                    AdcResolution::Bits32 => {
                        let signed_value = i32::from_le_bytes([
                            sample_bytes[0], sample_bytes[1],
                            sample_bytes[2], sample_bytes[3]
                        ]);
                        (signed_value as f32 / resolution.scale_factor()) * reference_voltage / gain
                    }
                    _ => return Err(ConversionError::UnsupportedConversion {
                        from: "bytes".to_string(),
                        to: format!("{:?} signed", resolution),
                    }),
                }
            }
            SampleFormat::Float32 => {
                f32::from_le_bytes([
                    sample_bytes[0], sample_bytes[1],
                    sample_bytes[2], sample_bytes[3]
                ])
            }
            SampleFormat::Float64 => {
                f64::from_le_bytes([
                    sample_bytes[0], sample_bytes[1], sample_bytes[2], sample_bytes[3],
                    sample_bytes[4], sample_bytes[5], sample_bytes[6], sample_bytes[7]
                ]) as f32
            }
        };

        samples.push(sample);
    }

    Ok(samples)
}

/// Convert frequency to period in nanoseconds
pub fn frequency_to_period_nanos(frequency_hz: f64) -> ConversionResult<u64> {
    if frequency_hz <= 0.0 {
        return Err(ConversionError::InvalidInput {
            function: "frequency_to_period_nanos".to_string(),
            input: frequency_hz.to_string(),
            reason: "Frequency must be positive".to_string(),
        });
    }

    let period_seconds = 1.0 / frequency_hz;
    let period_nanos = period_seconds * 1_000_000_000.0;

    if period_nanos > u64::MAX as f64 {
        return Err(ConversionError::NumericOverflow {
            operation: "frequency_to_period_nanos".to_string(),
            input: frequency_hz.to_string(),
        });
    }

    Ok(period_nanos as u64)
}

/// Convert period in nanoseconds to frequency
pub fn period_nanos_to_frequency(period_nanos: u64) -> ConversionResult<f64> {
    if period_nanos == 0 {
        return Err(ConversionError::InvalidInput {
            function: "period_nanos_to_frequency".to_string(),
            input: period_nanos.to_string(),
            reason: "Period cannot be zero".to_string(),
        });
    }

    let period_seconds = period_nanos as f64 / 1_000_000_000.0;
    let frequency = 1.0 / period_seconds;

    Ok(frequency)
}

/// Convert voltage to EMG signal in microvolts (common EMG unit)
pub fn voltage_to_microvolts(voltage: f32) -> f32 {
    voltage * 1_000_000.0
}

/// Convert microvolts to voltage
pub fn microvolts_to_voltage(microvolts: f32) -> f32 {
    microvolts / 1_000_000.0
}

/// Convert dB to linear scale
pub fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Convert linear scale to dB
pub fn linear_to_db(linear: f32) -> ConversionResult<f32> {
    if linear <= 0.0 {
        return Err(ConversionError::InvalidInput {
            function: "linear_to_db".to_string(),
            input: linear.to_string(),
            reason: "Linear value must be positive for dB conversion".to_string(),
        });
    }

    Ok(20.0 * linear.log10())
}

/// Normalize signal to range [-1.0, 1.0]
pub fn normalize_signal(samples: &[f32]) -> ConversionResult<Vec<f32>> {
    if samples.is_empty() {
        return Ok(Vec::new());
    }

    let max_abs = samples.iter()
        .map(|&x| x.abs())
        .fold(0.0f32, f32::max);

    if max_abs == 0.0 {
        return Ok(samples.to_vec());
    }

    let normalized: Vec<f32> = samples.iter()
        .map(|&x| x / max_abs)
        .collect();

    Ok(normalized)
}

/// Denormalize signal from [-1.0, 1.0] range
pub fn denormalize_signal(normalized_samples: &[f32], scale_factor: f32) -> Vec<f32> {
    normalized_samples.iter()
        .map(|&x| x * scale_factor)
        .collect()
}

/// Convert time in seconds to sample index
pub fn time_to_sample_index(time_seconds: f64, sampling_rate_hz: u32) -> ConversionResult<usize> {
    if time_seconds < 0.0 {
        return Err(ConversionError::InvalidInput {
            function: "time_to_sample_index".to_string(),
            input: time_seconds.to_string(),
            reason: "Time cannot be negative".to_string(),
        });
    }

    let sample_index = (time_seconds * sampling_rate_hz as f64) as usize;
    Ok(sample_index)
}

/// Convert sample index to time in seconds
pub fn sample_index_to_time(sample_index: usize, sampling_rate_hz: u32) -> ConversionResult<f64> {
    if sampling_rate_hz == 0 {
        return Err(ConversionError::InvalidInput {
            function: "sample_index_to_time".to_string(),
            input: sampling_rate_hz.to_string(),
            reason: "Sampling rate cannot be zero".to_string(),
        });
    }

    let time_seconds = sample_index as f64 / sampling_rate_hz as f64;
    Ok(time_seconds)
}

/// Calculate RMS (Root Mean Square) value of signal
pub fn calculate_rms(samples: &[f32]) -> ConversionResult<f32> {
    if samples.is_empty() {
        return Err(ConversionError::InvalidInput {
            function: "calculate_rms".to_string(),
            input: "0 samples".to_string(),
            reason: "Cannot calculate RMS of empty signal".to_string(),
        });
    }

    let sum_squares: f32 = samples.iter()
        .map(|&x| x * x)
        .sum();

    let mean_square = sum_squares / samples.len() as f32;
    Ok(mean_square.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adc_resolution() {
        assert_eq!(AdcResolution::Bits8.max_value(), 255);
        assert_eq!(AdcResolution::Bits16.max_value(), 65535);
        assert_eq!(AdcResolution::Bits24.max_value(), 16777215);

        assert_eq!(AdcResolution::Bits8.bytes_per_sample(), 1);
        assert_eq!(AdcResolution::Bits16.bytes_per_sample(), 2);
        assert_eq!(AdcResolution::Bits24.bytes_per_sample(), 3);
    }

    #[test]
    fn test_adc_to_voltage_conversion() {
        // Test 16-bit ADC with 3.3V reference and unity gain
        let voltage = adc_to_voltage(32768, AdcResolution::Bits16, 3.3, 1.0).unwrap();
        assert!((voltage - 1.65).abs() < 0.01); // Should be approximately half of 3.3V

        // Test edge cases
        assert!(adc_to_voltage(100000, AdcResolution::Bits8, 3.3, 1.0).is_err()); // Value too high
        assert!(adc_to_voltage(100, AdcResolution::Bits8, 0.0, 1.0).is_err()); // Zero reference
    }

    #[test]
    fn test_voltage_to_adc_conversion() {
        let adc_value = voltage_to_adc(1.65, AdcResolution::Bits16, 3.3, 1.0).unwrap();
        assert!((adc_value as f32 - 32768.0).abs() < 1.0); // Should be approximately half scale

        // Test out of range
        assert!(voltage_to_adc(5.0, AdcResolution::Bits16, 3.3, 1.0).is_err());
    }

    #[test]
    fn test_samples_to_bytes() {
        let samples = vec![0.5, -0.5, 0.0];
        let bytes = samples_to_bytes(&samples, SampleFormat::Float32, AdcResolution::Bits32).unwrap();
        assert_eq!(bytes.len(), 12); // 3 samples * 4 bytes each
    }

    #[test]
    fn test_bytes_to_samples() {
        let bytes = vec![0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0xBF]; // 1.0, -1.0 in IEEE 754
        let samples = bytes_to_samples(&bytes, SampleFormat::Float32, AdcResolution::Bits32, 1.0, 1.0).unwrap();
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_frequency_period_conversion() {
        let period = frequency_to_period_nanos(1000.0).unwrap(); // 1 kHz
        assert_eq!(period, 1_000_000); // 1ms in nanoseconds

        let frequency = period_nanos_to_frequency(1_000_000).unwrap();
        assert!((frequency - 1000.0).abs() < 0.001);

        // Test edge cases
        assert!(frequency_to_period_nanos(0.0).is_err());
        assert!(period_nanos_to_frequency(0).is_err());
    }

    #[test]
    fn test_voltage_microvolt_conversion() {
        assert_eq!(voltage_to_microvolts(0.001), 1000.0);
        assert_eq!(microvolts_to_voltage(1000.0), 0.001);
    }

    #[test]
    fn test_db_conversion() {
        let linear = db_to_linear(20.0); // 20 dB
        assert!((linear - 10.0).abs() < 0.001);

        let db = linear_to_db(10.0).unwrap();
        assert!((db - 20.0).abs() < 0.001);

        // Test invalid input
        assert!(linear_to_db(-1.0).is_err());
    }

    #[test]
    fn test_signal_normalization() {
        let samples = vec![1.0, -2.0, 0.5, -1.5];
        let normalized = normalize_signal(&samples).unwrap();

        // Maximum absolute value should be 1.0
        let max_abs = normalized.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!((max_abs - 1.0).abs() < 0.001);

        // Test empty array
        let empty_normalized = normalize_signal(&[]).unwrap();
        assert!(empty_normalized.is_empty());
    }

    #[test]
    fn test_time_sample_conversion() {
        let sample_index = time_to_sample_index(0.5, 1000).unwrap(); // 0.5 seconds at 1 kHz
        assert_eq!(sample_index, 500);

        let time = sample_index_to_time(500, 1000).unwrap();
        assert!((time - 0.5).abs() < 0.001);

        // Test edge cases
        assert!(time_to_sample_index(-1.0, 1000).is_err());
        assert!(sample_index_to_time(100, 0).is_err());
    }

    #[test]
    fn test_rms_calculation() {
        let samples = vec![1.0, -1.0, 1.0, -1.0];
        let rms = calculate_rms(&samples).unwrap();
        assert!((rms - 1.0).abs() < 0.001);

        // Test empty array
        assert!(calculate_rms(&[]).is_err());
    }

    #[test]
    fn test_signed_24bit_conversion() {
        // Test positive value
        let positive_bytes = [0x00, 0x10, 0x20]; // 0x201000
        let voltage = signed_24bit_to_voltage(&positive_bytes, 3.3, 1.0).unwrap();
        assert!(voltage > 0.0);

        // Test negative value (MSB set)
        let negative_bytes = [0x00, 0x10, 0xE0]; // Negative value
        let voltage = signed_24bit_to_voltage(&negative_bytes, 3.3, 1.0).unwrap();
        assert!(voltage < 0.0);

        // Test wrong byte count
        assert!(signed_24bit_to_voltage(&[0x00, 0x00], 3.3, 1.0).is_err());
    }
}