// src/processing/filters/mod.rs
//! Digital filters for EMG signal processing

pub mod iir;
pub mod fir;
pub mod notch;

pub use iir::*;
pub use fir::*;
pub use notch::*;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FilterType {
    Butterworth,
    Chebyshev1,
    Chebyshev2,
    Elliptic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BandType {
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
}

/// Filter coefficients for IIR filters
#[derive(Debug, Clone)]
pub struct IirCoefficients {
    pub b: Vec<f32>, // Numerator coefficients
    pub a: Vec<f32>, // Denominator coefficients
}

/// Common filter error types
#[derive(Debug)]
pub enum FilterError {
    InvalidParameters(String),
    InvalidCoefficients(String),
    ProcessingError(String),
}

impl std::fmt::Display for FilterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilterError::InvalidParameters(msg) => write!(f, "Invalid parameters: {}", msg),
            FilterError::InvalidCoefficients(msg) => write!(f, "Invalid coefficients: {}", msg),
            FilterError::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
        }
    }
}

impl std::error::Error for FilterError {}