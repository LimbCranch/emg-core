// src/config/schema_validator.rs
//! Configuration schema validation

use std::collections::HashMap;
use crate::config::constants::*;
use crate::hal::simulator;

/// Configuration validation errors
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub value: String,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Validation error for '{}': {} (value: {})", self.field, self.message, self.value)
    }
}

impl std::error::Error for ValidationError {}

/// Schema validator for configuration
/// #[derive(Debug, Clone)]

#[derive(Debug, Clone)]
pub struct SchemaValidator {
    constraints: HashMap<String, FieldConstraint>,
}

/// Field validation constraints
#[derive(Debug, Clone)]
pub enum FieldConstraint {
    Range { min: f64, max: f64 },
    IntRange { min: i64, max: i64 },
    OneOf(Vec<String>),
    MinLength(usize),
    MaxLength(usize),
    Required,
    Custom(fn(&str) -> bool),
}

impl SchemaValidator {
    /// Create new schema validator with default constraints
    pub fn new() -> Self {
        let mut constraints = HashMap::new();

        // System constraints
        constraints.insert("system.sampling_rate_hz".to_string(),
                           FieldConstraint::IntRange {
                               min: signal::MIN_SAMPLING_RATE_HZ as i64,
                               max: signal::MAX_SAMPLING_RATE_HZ as i64
                           });

        constraints.insert("system.channel_count".to_string(),
                           FieldConstraint::IntRange {
                               min: 1,
                               max: signal::MAX_CHANNEL_COUNT as i64
                           });

        constraints.insert("system.latency_target_ms".to_string(),
                           FieldConstraint::IntRange {
                               min: performance::MIN_LATENCY_TARGET_MS as i64,
                               max: performance::MAX_LATENCY_TARGET_MS as i64
                           });

        // HAL constraints
        constraints.insert("hal.connection_timeout_ms".to_string(),
                           FieldConstraint::IntRange { min: 100, max: 60000 });

        constraints.insert("hal.retry_attempts".to_string(),
                           FieldConstraint::IntRange {
                               min: 0,
                               max: hal::MAX_RETRY_ATTEMPTS as i64
                           });

        // Processing constraints
        constraints.insert("processing.filter_bank.highpass_cutoff_hz".to_string(),
                           FieldConstraint::Range { min: 0.1, max: 1000.0 });

        constraints.insert("processing.filter_bank.lowpass_cutoff_hz".to_string(),
                           FieldConstraint::Range { min: 10.0, max: 5000.0 });

        constraints.insert("processing.filter_bank.filter_order".to_string(),
                           FieldConstraint::IntRange {
                               min: filters::MIN_FILTER_ORDER as i64,
                               max: filters::MAX_FILTER_ORDER as i64
                           });

        constraints.insert("processing.quality_monitoring.snr_threshold_db".to_string(),
                           FieldConstraint::Range {
                               min: quality::MIN_SNR_THRESHOLD_DB as f64,
                               max: quality::MAX_SNR_THRESHOLD_DB as f64
                           });

        constraints.insert("processing.quality_monitoring.saturation_threshold".to_string(),
                           FieldConstraint::Range { min: 0.1, max: 1.0 });

        // Windowing constraints
        constraints.insert("processing.windowing.window_size_samples".to_string(),
                           FieldConstraint::IntRange {
                               min: windowing::MIN_WINDOW_SIZE as i64,
                               max: windowing::MAX_WINDOW_SIZE as i64
                           });

        constraints.insert("processing.windowing.overlap_percent".to_string(),
                           FieldConstraint::Range {
                               min: windowing::MIN_OVERLAP_PERCENT as f64,
                               max: windowing::MAX_OVERLAP_PERCENT as f64
                           });

        /*// Simulator constraints
        constraints.insert("hal.simulator.noise_level".to_string(),
                           FieldConstraint::Range {
                               min: simulator::MIN_NOISE_LEVEL as f64,
                               max: simulator::MAX_NOISE_LEVEL as f64
                           });

        constraints.insert("hal.simulator.artifact_probability".to_string(),
                           FieldConstraint::Range {
                               min: simulator::MIN_ARTIFACT_PROBABILITY as f64,
                               max: simulator::MAX_ARTIFACT_PROBABILITY as f64
                           });*/

        // Device type constraint
        constraints.insert("hal.device_type".to_string(),
                           FieldConstraint::OneOf(vec![
                               "simulator".to_string(),
                               "usb".to_string(),
                               "serial".to_string(),
                               "bluetooth".to_string(),
                           ]));

        // Thread priority constraint
        constraints.insert("system.thread_priority".to_string(),
                           FieldConstraint::OneOf(vec![
                               "normal".to_string(),
                               "high".to_string(),
                               "realtime".to_string(),
                           ]));

        Self { constraints }
    }

    /// Validate configuration value against schema
    pub fn validate_field(&self, field_path: &str, value: &toml::Value) -> Result<(), ValidationError> {
        if let Some(constraint) = self.constraints.get(field_path) {
            self.check_constraint(field_path, value, constraint)
        } else {
            Ok(()) // Unknown fields are allowed for extensibility
        }
    }

    /// Validate entire configuration
    pub fn validate_config(&self, config: &toml::Value) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        self.validate_recursive("", config, &mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Check cross-field dependencies
    pub fn validate_dependencies(&self, config: &toml::Value) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Validate filter frequency relationships
        if let (Some(highpass), Some(lowpass)) = (
            self.get_nested_value(config, "processing.filter_bank.highpass_cutoff_hz"),
            self.get_nested_value(config, "processing.filter_bank.lowpass_cutoff_hz")
        ) {
            if let (Some(hp_val), Some(lp_val)) = (highpass.as_float(), lowpass.as_float()) {
                if hp_val >= lp_val {
                    errors.push(ValidationError {
                        field: "processing.filter_bank".to_string(),
                        message: "Highpass cutoff must be less than lowpass cutoff".to_string(),
                        value: format!("hp: {}, lp: {}", hp_val, lp_val),
                    });
                }
            }
        }

        // Validate sampling rate vs filter frequencies
        if let (Some(sample_rate), Some(lowpass)) = (
            self.get_nested_value(config, "system.sampling_rate_hz"),
            self.get_nested_value(config, "processing.filter_bank.lowpass_cutoff_hz")
        ) {
            if let (Some(sr_val), Some(lp_val)) = (sample_rate.as_integer(), lowpass.as_float()) {
                if lp_val >= (sr_val as f64 / 2.0) {
                    errors.push(ValidationError {
                        field: "processing.filter_bank.lowpass_cutoff_hz".to_string(),
                        message: "Lowpass cutoff must be less than Nyquist frequency".to_string(),
                        value: format!("cutoff: {}, nyquist: {}", lp_val, sr_val as f64 / 2.0),
                    });
                }
            }
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    fn validate_recursive(&self, prefix: &str, value: &toml::Value, errors: &mut Vec<ValidationError>) {
        match value {
            toml::Value::Table(table) => {
                for (key, val) in table {
                    let path = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{}.{}", prefix, key)
                    };

                    if let Err(err) = self.validate_field(&path, val) {
                        errors.push(err);
                    }

                    self.validate_recursive(&path, val, errors);
                }
            }
            _ => {
                if let Err(err) = self.validate_field(prefix, value) {
                    errors.push(err);
                }
            }
        }
    }

    fn check_constraint(&self, field: &str, value: &toml::Value, constraint: &FieldConstraint) -> Result<(), ValidationError> {
        match constraint {
            FieldConstraint::Range { min, max } => {
                if let Some(val) = value.as_float() {
                    if val < *min || val > *max {
                        return Err(ValidationError {
                            field: field.to_string(),
                            message: format!("Value must be between {} and {}", min, max),
                            value: val.to_string(),
                        });
                    }
                }
            }
            FieldConstraint::IntRange { min, max } => {
                if let Some(val) = value.as_integer() {
                    if val < *min || val > *max {
                        return Err(ValidationError {
                            field: field.to_string(),
                            message: format!("Value must be between {} and {}", min, max),
                            value: val.to_string(),
                        });
                    }
                }
            }
            FieldConstraint::OneOf(options) => {
                if let Some(val) = value.as_str() {
                    let val_lower = val.to_lowercase();
                    let options_lower: Vec<String> = options.iter().map(|opt| opt.to_lowercase()).collect();

                    if !options_lower.contains(&val_lower) {
                        return Err(ValidationError {
                            field: field.to_string(),
                            message: format!(
                                "Value must be one of: {}",
                                options.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                            ),
                            value: val.to_string(),
                        });
                    }
                }
            }

            FieldConstraint::MinLength(min_len) => {
                if let Some(val) = value.as_str() {
                    if val.len() < *min_len {
                        return Err(ValidationError {
                            field: field.to_string(),
                            message: format!("Minimum length is {}", min_len),
                            value: val.to_string(),
                        });
                    }
                }
            }
            FieldConstraint::MaxLength(max_len) => {
                if let Some(val) = value.as_str() {
                    if val.len() > *max_len {
                        return Err(ValidationError {
                            field: field.to_string(),
                            message: format!("Maximum length is {}", max_len),
                            value: val.to_string(),
                        });
                    }
                }
            }
            FieldConstraint::Required => {
                if value.as_str().map_or(true, |s| s.is_empty()) {
                    return Err(ValidationError {
                        field: field.to_string(),
                        message: "Field is required".to_string(),
                        value: "empty".to_string(),
                    });
                }
            }
            FieldConstraint::Custom(validator) => {
                if let Some(val) = value.as_str() {
                    if !validator(val) {
                        return Err(ValidationError {
                            field: field.to_string(),
                            message: "Custom validation failed".to_string(),
                            value: val.to_string(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn get_nested_value<'a>(&self, config: &'a toml::Value, path: &str) -> Option<&'a toml::Value> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = config;

        for part in parts {
            if let Some(table) = current.as_table() {
                current = table.get(part)?;
            } else {
                return None;
            }
        }

        Some(current)
    }
}

impl Default for SchemaValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_validator_creation() {
        let validator = SchemaValidator::new();
        assert!(!validator.constraints.is_empty());
    }

    #[test]
    fn test_valid_sampling_rate() {
        let validator = SchemaValidator::new();
        let value = toml::Value::Integer(2000);
        assert!(validator.validate_field("system.sampling_rate_hz", &value).is_ok());
    }

    #[test]
    fn test_invalid_sampling_rate() {
        let validator = SchemaValidator::new();
        let value = toml::Value::Integer(100); // Too low
        assert!(validator.validate_field("system.sampling_rate_hz", &value).is_err());
    }

    #[test]
    fn test_device_type_validation() {
        let validator = SchemaValidator::new();
        let valid_value = toml::Value::String("simulator".to_string());
        assert!(validator.validate_field("hal.device_type", &valid_value).is_ok());

        let invalid_value = toml::Value::String("invalid_device".to_string());
        assert!(validator.validate_field("hal.device_type", &invalid_value).is_err());
    }
}