//! Bounds checking utilities for EMG-Core
//!
//! Provides comprehensive bounds checking to prevent buffer overflows,
//! array out-of-bounds access, and other memory safety issues.
//!
//! This module specifically addresses the bounds checking issues
//! identified in the serial driver and other components.

use std::fmt;
use crate::config::constants::validation;

/// Bounds checking error types
#[derive(Debug, Clone, PartialEq)]
pub enum BoundsError {
    /// Array index out of bounds
    IndexOutOfBounds {
        index: usize,
        length: usize,
        context: String,
    },
    /// Slice bounds invalid
    SliceBoundsInvalid {
        start: usize,
        end: usize,
        length: usize,
        context: String,
    },
    /// Buffer capacity exceeded
    BufferCapacityExceeded {
        required: usize,
        available: usize,
        context: String,
    },
    /// Packet size validation failed
    PacketSizeInvalid {
        actual: usize,
        expected: usize,
        context: String,
    },
    /// Numeric value out of bounds
    NumericOutOfBounds {
        value: String,
        min: String,
        max: String,
        context: String,
    },
    /// Offset calculation overflow
    OffsetOverflow {
        base: usize,
        offset: usize,
        context: String,
    },
}

impl fmt::Display for BoundsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoundsError::IndexOutOfBounds { index, length, context } => {
                write!(f, "Index {} out of bounds for length {} in {}", index, length, context)
            }
            BoundsError::SliceBoundsInvalid { start, end, length, context } => {
                write!(f, "Slice bounds [{}..{}] invalid for length {} in {}", start, end, length, context)
            }
            BoundsError::BufferCapacityExceeded { required, available, context } => {
                write!(f, "Buffer capacity exceeded: required {}, available {} in {}", required, available, context)
            }
            BoundsError::PacketSizeInvalid { actual, expected, context } => {
                write!(f, "Packet size invalid: actual {}, expected {} in {}", actual, expected, context)
            }
            BoundsError::NumericOutOfBounds { value, min, max, context } => {
                write!(f, "Numeric value {} out of bounds [{}, {}] in {}", value, min, max, context)
            }
            BoundsError::OffsetOverflow { base, offset, context } => {
                write!(f, "Offset overflow: base {} + offset {} in {}", base, offset, context)
            }
        }
    }
}

impl std::error::Error for BoundsError {}

/// Result type for bounds checking operations
pub type BoundsResult<T> = Result<T, BoundsError>;

/// Bounds checker for comprehensive validation
#[derive(Debug, Clone)]
pub struct BoundsChecker {
    /// Enable strict mode (fail on any potential issue)
    strict_mode: bool,
    /// Context for error reporting
    context: String,
}

impl BoundsChecker {
    /// Create new bounds checker
    pub fn new(context: &str) -> Self {
        Self {
            strict_mode: false,
            context: context.to_string(),
        }
    }

    /// Enable strict mode
    pub fn strict(mut self) -> Self {
        self.strict_mode = true;
        self
    }

    /// Check array index bounds
    pub fn check_index(&self, index: usize, length: usize) -> BoundsResult<()> {
        if index >= length {
            return Err(BoundsError::IndexOutOfBounds {
                index,
                length,
                context: self.context.clone(),
            });
        }
        Ok(())
    }

    /// Check slice bounds
    pub fn check_slice(&self, start: usize, end: usize, length: usize) -> BoundsResult<()> {
        if start > end {
            return Err(BoundsError::SliceBoundsInvalid {
                start,
                end,
                length,
                context: format!("{}: start > end", self.context),
            });
        }

        if end > length {
            return Err(BoundsError::SliceBoundsInvalid {
                start,
                end,
                length,
                context: self.context.clone(),
            });
        }

        Ok(())
    }

    /// Check buffer capacity
    pub fn check_capacity(&self, required: usize, available: usize) -> BoundsResult<()> {
        if required > available {
            return Err(BoundsError::BufferCapacityExceeded {
                required,
                available,
                context: self.context.clone(),
            });
        }
        Ok(())
    }

    /// Check packet size with expected size
    pub fn check_packet_size(&self, actual: usize, expected: usize) -> BoundsResult<()> {
        if actual != expected {
            return Err(BoundsError::PacketSizeInvalid {
                actual,
                expected,
                context: self.context.clone(),
            });
        }
        Ok(())
    }

    /// Check packet has minimum required size
    pub fn check_min_packet_size(&self, actual: usize, minimum: usize) -> BoundsResult<()> {
        if actual < minimum {
            return Err(BoundsError::PacketSizeInvalid {
                actual,
                expected: minimum,
                context: format!("{}: minimum size check", self.context),
            });
        }
        Ok(())
    }

    /// Check numeric range
    pub fn check_numeric_range<T>(&self, value: T, min: T, max: T) -> BoundsResult<()>
    where
        T: PartialOrd + fmt::Display,
    {
        if value < min || value > max {
            return Err(BoundsError::NumericOutOfBounds {
                value: value.to_string(),
                min: min.to_string(),
                max: max.to_string(),
                context: self.context.clone(),
            });
        }
        Ok(())
    }

    /// Check offset calculation for overflow
    pub fn check_offset(&self, base: usize, offset: usize) -> BoundsResult<usize> {
        base.checked_add(offset).ok_or_else(|| BoundsError::OffsetOverflow {
            base,
            offset,
            context: self.context.clone(),
        })
    }
}

/// Check array bounds with context
pub fn check_array_bounds<T>(array: &[T], index: usize, context: &str) -> BoundsResult<()> {
    if index >= array.len() {
        return Err(BoundsError::IndexOutOfBounds {
            index,
            length: array.len(),
            context: context.to_string(),
        });
    }
    Ok(())
}

/// Check slice bounds with context
pub fn check_slice_bounds<T>(array: &[T], start: usize, end: usize, context: &str) -> BoundsResult<()> {
    if start > end {
        return Err(BoundsError::SliceBoundsInvalid {
            start,
            end,
            length: array.len(),
            context: format!("{}: start > end", context),
        });
    }

    if end > array.len() {
        return Err(BoundsError::SliceBoundsInvalid {
            start,
            end,
            length: array.len(),
            context: context.to_string(),
        });
    }

    Ok(())
}

/// Check numeric range with context
pub fn check_numeric_range<T>(value: T, min: T, max: T, context: &str) -> BoundsResult<()>
where
    T: PartialOrd + fmt::Display,
{
    if value < min || value > max {
        return Err(BoundsError::NumericOutOfBounds {
            value: value.to_string(),
            min: min.to_string(),
            max: max.to_string(),
            context: context.to_string(),
        });
    }
    Ok(())
}

/// Check buffer capacity with context
pub fn check_buffer_capacity(required: usize, available: usize, context: &str) -> BoundsResult<()> {
    if required > available {
        return Err(BoundsError::BufferCapacityExceeded {
            required,
            available,
            context: context.to_string(),
        });
    }
    Ok(())
}

/// Ensure packet has minimum required size (fixes serial driver issue)
pub fn ensure_packet_size(packet: &[u8], required_size: usize, context: &str) -> BoundsResult<()> {
    if packet.len() < required_size {
        return Err(BoundsError::PacketSizeInvalid {
            actual: packet.len(),
            expected: required_size,
            context: context.to_string(),
        });
    }
    Ok(())
}

/// Safe array access with bounds checking
pub fn safe_array_get<'a, T>(array: &'a [T], index: usize, context: &str) -> BoundsResult<&'a T> {
    check_array_bounds(array, index, context)?;
    Ok(&array[index])
}

/// Safe mutable array access with bounds checking
pub fn safe_array_get_mut<'a, T>(array: &'a mut [T], index: usize, context: &str) -> BoundsResult<&'a mut T> {
    if index >= array.len() {
        return Err(BoundsError::IndexOutOfBounds {
            index,
            length: array.len(),
            context: context.to_string(),
        });
    }
    Ok(&mut array[index])
}

/// Safe slice creation with bounds checking
pub fn safe_slice<'a, T>(array: &'a [T], start: usize, end: usize, context: &str) -> BoundsResult<&'a [T]> {
    check_slice_bounds(array, start, end, context)?;
    Ok(&array[start..end])
}

/// Safe mutable slice creation with bounds checking
pub fn safe_slice_mut<'a, T>(array: &'a mut [T], start: usize, end: usize, context: &str) -> BoundsResult<&'a mut [T]> {
    if start > end {
        return Err(BoundsError::SliceBoundsInvalid {
            start,
            end,
            length: array.len(),
            context: format!("{}: start > end", context),
        });
    }

    if end > array.len() {
        return Err(BoundsError::SliceBoundsInvalid {
            start,
            end,
            length: array.len(),
            context: context.to_string(),
        });
    }

    Ok(&mut array[start..end])
}

/// Extract data from packet with comprehensive bounds checking
/// This specifically addresses the serial driver issue: let emg_data = &packet[data_start..data_end];
pub fn extract_packet_data<'a>(packet: &'a [u8], data_start: usize, data_length: usize, context: &str) -> BoundsResult<&'a [u8]> {
    // Check for overflow in end calculation
    let data_end = data_start.checked_add(data_length)
        .ok_or_else(|| BoundsError::OffsetOverflow {
            base: data_start,
            offset: data_length,
            context: format!("{}: data offset calculation", context),
        })?;

    // Ensure packet is large enough
    ensure_packet_size(packet, data_end, context)?;

    // Safe slice extraction
    safe_slice(packet, data_start, data_end, context)
}

/// Validate and extract EMG channel data from packet
pub fn extract_emg_channels<'a>(
    packet: &'a [u8],
    header_size: usize,
    channel_count: usize,
    bytes_per_channel: usize,
    context: &str,
) -> BoundsResult<&'a [u8]> {
    let data_length = channel_count.checked_mul(bytes_per_channel)
        .ok_or_else(|| BoundsError::OffsetOverflow {
            base: channel_count,
            offset: bytes_per_channel,
            context: format!("{}: channel data size calculation", context),
        })?;

    extract_packet_data(packet, header_size, data_length, context)
}

/// Validate buffer write operation won't overflow
pub fn validate_buffer_write(
    buffer: &[u8],
    write_offset: usize,
    write_length: usize,
    context: &str,
) -> BoundsResult<()> {
    let write_end = write_offset.checked_add(write_length)
        .ok_or_else(|| BoundsError::OffsetOverflow {
            base: write_offset,
            offset: write_length,
            context: format!("{}: write operation", context),
        })?;

    if write_end > buffer.len() {
        return Err(BoundsError::BufferCapacityExceeded {
            required: write_end,
            available: buffer.len(),
            context: context.to_string(),
        });
    }

    Ok(())
}

/// Validate ring buffer operation bounds
pub fn validate_ring_buffer_bounds(
    capacity: usize,
    head: usize,
    tail: usize,
    operation: &str,
    context: &str,
) -> BoundsResult<()> {
    // Capacity must be power of 2
    if !capacity.is_power_of_two() {
        return Err(BoundsError::NumericOutOfBounds {
            value: capacity.to_string(),
            min: "power_of_2".to_string(),
            max: "power_of_2".to_string(),
            context: format!("{}: ring buffer capacity", context),
        });
    }

    // Head and tail must be within reasonable bounds
    let max_index = capacity * 2; // Allow for wraparound
    check_numeric_range(head, 0, max_index, &format!("{}: head index in {}", context, operation))?;
    check_numeric_range(tail, 0, max_index, &format!("{}: tail index in {}", context, operation))?;

    Ok(())
}

/// Clamp value to bounds (for graceful degradation)
pub fn clamp_to_bounds<T>(value: T, min: T, max: T) -> T
where
    T: PartialOrd,
{
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Validate array has expected fixed size
pub fn validate_fixed_array_size<T>(array: &[T], expected_size: usize, context: &str) -> BoundsResult<()> {
    if array.len() != expected_size {
        return Err(BoundsError::PacketSizeInvalid {
            actual: array.len(),
            expected: expected_size,
            context: context.to_string(),
        });
    }
    Ok(())
}

/// Validate memory alignment for performance-critical operations
pub fn validate_memory_alignment(ptr: *const u8, alignment: usize, context: &str) -> BoundsResult<()> {
    let addr = ptr as usize;
    if addr % alignment != 0 {
        return Err(BoundsError::NumericOutOfBounds {
            value: format!("0x{:x}", addr),
            min: format!("aligned_to_{}", alignment),
            max: format!("aligned_to_{}", alignment),
            context: format!("{}: memory alignment", context),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounds_checker() {
        let checker = BoundsChecker::new("test_context");

        // Valid index
        assert!(checker.check_index(5, 10).is_ok());

        // Invalid index
        assert!(checker.check_index(10, 10).is_err());

        // Valid slice
        assert!(checker.check_slice(2, 8, 10).is_ok());

        // Invalid slice (start > end)
        assert!(checker.check_slice(8, 2, 10).is_err());

        // Invalid slice (end > length)
        assert!(checker.check_slice(2, 15, 10).is_err());
    }

    #[test]
    fn test_array_bounds_checking() {
        let array = [1, 2, 3, 4, 5];

        // Valid access
        assert!(check_array_bounds(&array, 2, "test").is_ok());

        // Invalid access
        assert!(check_array_bounds(&array, 5, "test").is_err());
        assert!(check_array_bounds(&array, 10, "test").is_err());
    }

    #[test]
    fn test_slice_bounds_checking() {
        let array = [1, 2, 3, 4, 5];

        // Valid slice
        assert!(check_slice_bounds(&array, 1, 4, "test").is_ok());

        // Invalid slice
        assert!(check_slice_bounds(&array, 4, 1, "test").is_err());
        assert!(check_slice_bounds(&array, 1, 10, "test").is_err());
    }

    #[test]
    fn test_safe_array_access() {
        let array = [10, 20, 30, 40, 50];

        // Valid access
        let value = safe_array_get(&array, 2, "test").unwrap();
        assert_eq!(*value, 30);

        // Invalid access
        assert!(safe_array_get(&array, 5, "test").is_err());
    }

    #[test]
    fn test_safe_slice_creation() {
        let array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        // Valid slice
        let slice = safe_slice(&array, 2, 6, "test").unwrap();
        assert_eq!(slice, &[3, 4, 5, 6]);

        // Invalid slice
        assert!(safe_slice(&array, 6, 2, "test").is_err());
        assert!(safe_slice(&array, 2, 15, "test").is_err());
    }

    #[test]
    fn test_packet_data_extraction() {
        let packet = [0xAA, 0x55, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06];

        // Valid extraction (skip 2-byte header, extract 4 bytes)
        let data = extract_packet_data(&packet, 2, 4, "test").unwrap();
        assert_eq!(data, &[0x01, 0x02, 0x03, 0x04]);

        // Invalid extraction (not enough data)
        assert!(extract_packet_data(&packet, 2, 10, "test").is_err());

        // Overflow in offset calculation
        assert!(extract_packet_data(&packet, usize::MAX, 1, "test").is_err());
    }

    #[test]
    fn test_emg_channel_extraction() {
        // Packet: 2-byte header + 4 channels * 2 bytes each = 10 bytes total
        let packet = [0xAA, 0x55, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];

        // Valid extraction
        let channels = extract_emg_channels(&packet, 2, 4, 2, "test").unwrap();
        assert_eq!(channels, &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);

        // Invalid extraction (too many channels)
        assert!(extract_emg_channels(&packet, 2, 8, 2, "test").is_err());
    }

    #[test]
    fn test_buffer_write_validation() {
        let buffer = [0u8; 10];

        // Valid write
        assert!(validate_buffer_write(&buffer, 2, 4, "test").is_ok());

        // Invalid write (overflow)
        assert!(validate_buffer_write(&buffer, 8, 5, "test").is_err());

        // Offset overflow
        assert!(validate_buffer_write(&buffer, usize::MAX, 1, "test").is_err());
    }

    #[test]
    fn test_ring_buffer_bounds() {
        // Valid ring buffer (power of 2 capacity)
        assert!(validate_ring_buffer_bounds(1024, 100, 200, "push", "test").is_ok());

        // Invalid capacity (not power of 2)
        assert!(validate_ring_buffer_bounds(1000, 100, 200, "push", "test").is_err());

        // Invalid head index (too large)
        assert!(validate_ring_buffer_bounds(1024, 3000, 200, "push", "test").is_err());
    }

    #[test]
    fn test_clamp_to_bounds() {
        assert_eq!(clamp_to_bounds(5, 1, 10), 5);
        assert_eq!(clamp_to_bounds(-5, 1, 10), 1);
        assert_eq!(clamp_to_bounds(15, 1, 10), 10);
    }

    #[test]
    fn test_fixed_array_size_validation() {
        let array = [1, 2, 3, 4, 5];

        // Valid size
        assert!(validate_fixed_array_size(&array, 5, "test").is_ok());

        // Invalid size
        assert!(validate_fixed_array_size(&array, 3, "test").is_err());
        assert!(validate_fixed_array_size(&array, 8, "test").is_err());
    }

    #[test]
    fn test_error_display() {
        let error = BoundsError::IndexOutOfBounds {
            index: 10,
            length: 5,
            context: "test_context".to_string(),
        };

        let display = format!("{}", error);
        assert!(display.contains("10"));
        assert!(display.contains("5"));
        assert!(display.contains("test_context"));
    }

    #[test]
    fn test_numeric_range_checking() {
        assert!(check_numeric_range(5, 1, 10, "test").is_ok());
        assert!(check_numeric_range(0, 1, 10, "test").is_err());
        assert!(check_numeric_range(15, 1, 10, "test").is_err());
    }

    #[test]
    fn test_buffer_capacity_checking() {
        assert!(check_buffer_capacity(100, 1024, "test").is_ok());
        assert!(check_buffer_capacity(2048, 1024, "test").is_err());
    }

    #[test]
    fn test_packet_size_checking() {
        let packet = [0u8; 10];

        assert!(ensure_packet_size(&packet, 8, "test").is_ok());
        assert!(ensure_packet_size(&packet, 15, "test").is_err());
    }
}