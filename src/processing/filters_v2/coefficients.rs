
use super::{FilterError, BandType, IirCoefficients};
use std::f32::consts::PI;

/// Butterworth filter coefficient calculator
pub struct ButterworthCoefficients;

/// Chebyshev Type I filter coefficient calculator  
pub struct ChebyshevType1Coefficients;

/// Filter coefficient calculation trait
pub trait FilterCoefficients {
    fn calculate_lowpass(order: usize, cutoff: f32, sample_rate: f32) -> Result<IirCoefficients, FilterError>;
    fn calculate_highpass(order: usize, cutoff: f32, sample_rate: f32) -> Result<IirCoefficients, FilterError>;
    fn calculate_bandpass(order: usize, low_cutoff: f32, high_cutoff: f32, sample_rate: f32) -> Result<IirCoefficients, FilterError>;
    fn calculate_bandstop(order: usize, low_cutoff: f32, high_cutoff: f32, sample_rate: f32) -> Result<IirCoefficients, FilterError>;
}

impl FilterCoefficients for ButterworthCoefficients {
    fn calculate_lowpass(order: usize, cutoff: f32, sample_rate: f32) -> Result<IirCoefficients, FilterError> {
        if order == 0 || order > 8 {
            return Err(FilterError::InvalidParameters("Order must be 1-8".to_string()));
        }

        let nyquist = sample_rate / 2.0;
        let normalized_cutoff = cutoff / nyquist;

        if normalized_cutoff >= 1.0 || normalized_cutoff <= 0.0 {
            return Err(FilterError::InvalidParameters("Invalid cutoff frequency".to_string()));
        }

        // Pre-warp frequency for bilinear transform
        let omega_c = (PI * normalized_cutoff).tan();

        match order {
            1 => Self::butterworth_lowpass_1st_order(omega_c),
            2 => Self::butterworth_lowpass_2nd_order(omega_c),
            3 => Self::butterworth_lowpass_3rd_order(omega_c),
            4 => Self::butterworth_lowpass_4th_order(omega_c),
            _ => {
                // For higher orders, cascade 2nd order sections
                Self::butterworth_cascade(order, omega_c, BandType::Lowpass)
            }
        }
    }

    fn calculate_highpass(order: usize, cutoff: f32, sample_rate: f32) -> Result<IirCoefficients, FilterError> {
        let nyquist = sample_rate / 2.0;
        let normalized_cutoff = cutoff / nyquist;
        let omega_c = (PI * normalized_cutoff).tan();

        match order {
            1 => Self::butterworth_highpass_1st_order(omega_c),
            2 => Self::butterworth_highpass_2nd_order(omega_c),
            _ => Self::butterworth_cascade(order, omega_c, BandType::Highpass)
        }
    }

    fn calculate_bandpass(order: usize, low_cutoff: f32, high_cutoff: f32, sample_rate: f32) -> Result<IirCoefficients, FilterError> {
        if low_cutoff >= high_cutoff {
            return Err(FilterError::InvalidParameters("Low cutoff must be less than high cutoff".to_string()));
        }

        // Calculate center frequency and bandwidth
        let center_freq = (low_cutoff * high_cutoff).sqrt();
        let bandwidth = high_cutoff - low_cutoff;

        Self::butterworth_bandpass(order, center_freq, bandwidth, sample_rate)
    }

    fn calculate_bandstop(order: usize, low_cutoff: f32, high_cutoff: f32, sample_rate: f32) -> Result<IirCoefficients, FilterError> {
        let center_freq = (low_cutoff * high_cutoff).sqrt();
        let bandwidth = high_cutoff - low_cutoff;

        Self::butterworth_bandstop(order, center_freq, bandwidth, sample_rate)
    }
}

impl ButterworthCoefficients {
    fn butterworth_lowpass_1st_order(omega_c: f32) -> Result<IirCoefficients, FilterError> {
        let k = omega_c;
        let norm = 1.0 + k;

        Ok(IirCoefficients {
            b: vec![k / norm, k / norm],
            a: vec![1.0, (k - 1.0) / norm],
        })
    }

    fn butterworth_lowpass_2nd_order(omega_c: f32) -> Result<IirCoefficients, FilterError> {
        let k = omega_c;
        let k2 = k * k;
        let sqrt2 = std::f32::consts::SQRT_2;
        let norm = 1.0 + k * sqrt2 + k2;

        Ok(IirCoefficients {
            b: vec![k2 / norm, 2.0 * k2 / norm, k2 / norm],
            a: vec![1.0, (2.0 * k2 - 2.0) / norm, (1.0 - k * sqrt2 + k2) / norm],
        })
    }

    fn butterworth_lowpass_3rd_order(omega_c: f32) -> Result<IirCoefficients, FilterError> {
        // 3rd order = 1st order + 2nd order cascade
        let first_order = Self::butterworth_lowpass_1st_order(omega_c)?;
        let second_order = Self::butterworth_lowpass_2nd_order(omega_c)?;

        Self::cascade_sections(&first_order, &second_order)
    }

    fn butterworth_lowpass_4th_order(omega_c: f32) -> Result<IirCoefficients, FilterError> {
        // 4th order = two 2nd order sections
        let section1 = Self::butterworth_2nd_order_section(omega_c, 0.765367)?;  // Q1
        let section2 = Self::butterworth_2nd_order_section(omega_c, 1.847759)?;  // Q2

        Self::cascade_sections(&section1, &section2)
    }

    fn butterworth_highpass_1st_order(omega_c: f32) -> Result<IirCoefficients, FilterError> {
        let k = omega_c;
        let norm = 1.0 + k;

        Ok(IirCoefficients {
            b: vec![1.0 / norm, -1.0 / norm],
            a: vec![1.0, (k - 1.0) / norm],
        })
    }

    fn butterworth_highpass_2nd_order(omega_c: f32) -> Result<IirCoefficients, FilterError> {
        let k = omega_c;
        let k2 = k * k;
        let sqrt2 = std::f32::consts::SQRT_2;
        let norm = 1.0 + k * sqrt2 + k2;

        Ok(IirCoefficients {
            b: vec![1.0 / norm, -2.0 / norm, 1.0 / norm],
            a: vec![1.0, (2.0 * k2 - 2.0) / norm, (1.0 - k * sqrt2 + k2) / norm],
        })
    }

    fn butterworth_bandpass(order: usize, center_freq: f32, bandwidth: f32, sample_rate: f32) -> Result<IirCoefficients, FilterError> {
        let nyquist = sample_rate / 2.0;
        let wc = 2.0 * PI * center_freq / sample_rate;
        let wb = 2.0 * PI * bandwidth / sample_rate;

        // Use biquad sections for bandpass
        let q = center_freq / bandwidth;
        let cos_wc = wc.cos();
        let sin_wc = wc.sin();
        let alpha = sin_wc / (2.0 * q);

        let norm = 1.0 + alpha;

        Ok(IirCoefficients {
            b: vec![alpha / norm, 0.0, -alpha / norm],
            a: vec![1.0, -2.0 * cos_wc / norm, (1.0 - alpha) / norm],
        })
    }

    fn butterworth_bandstop(order: usize, center_freq: f32, bandwidth: f32, sample_rate: f32) -> Result<IirCoefficients, FilterError> {
        let wc = 2.0 * PI * center_freq / sample_rate;
        let q = center_freq / bandwidth;
        let cos_wc = wc.cos();
        let sin_wc = wc.sin();
        let alpha = sin_wc / (2.0 * q);

        let norm = 1.0 + alpha;

        Ok(IirCoefficients {
            b: vec![1.0 / norm, -2.0 * cos_wc / norm, 1.0 / norm],
            a: vec![1.0, -2.0 * cos_wc / norm, (1.0 - alpha) / norm],
        })
    }

    fn butterworth_2nd_order_section(omega_c: f32, q: f32) -> Result<IirCoefficients, FilterError> {
        let k = omega_c;
        let k2 = k * k;
        let norm = 1.0 + k / q + k2;

        Ok(IirCoefficients {
            b: vec![k2 / norm, 2.0 * k2 / norm, k2 / norm],
            a: vec![1.0, (2.0 * k2 - 2.0) / norm, (1.0 - k / q + k2) / norm],
        })
    }

    fn butterworth_cascade(order: usize, omega_c: f32, band_type: BandType) -> Result<IirCoefficients, FilterError> {
        if order % 2 == 0 {
            // Even order: cascade of 2nd order sections
            let mut result = Self::butterworth_lowpass_2nd_order(omega_c)?;

            for i in 1..(order / 2) {
                let q = 1.0 / (2.0 * (PI * (2.0 * i as f32 + 1.0) / (2.0 * order as f32)).cos());
                let section = Self::butterworth_2nd_order_section(omega_c, q)?;
                result = Self::cascade_sections(&result, &section)?;
            }

            Ok(result)
        } else {
            // Odd order: 1st order + cascade of 2nd order sections
            let mut result = Self::butterworth_lowpass_1st_order(omega_c)?;

            for i in 0..((order - 1) / 2) {
                let q = 1.0 / (2.0 * (PI * (2.0 * i as f32 + 2.0) / (2.0 * order as f32)).cos());
                let section = Self::butterworth_2nd_order_section(omega_c, q)?;
                result = Self::cascade_sections(&result, &section)?;
            }

            Ok(result)
        }
    }

    fn cascade_sections(section1: &IirCoefficients, section2: &IirCoefficients) -> Result<IirCoefficients, FilterError> {
        // Multiply transfer functions: H1(z) * H2(z)
        let mut b = vec![0.0; section1.b.len() + section2.b.len() - 1];
        let mut a = vec![0.0; section1.a.len() + section2.a.len() - 1];

        // Convolve numerator coefficients
        for (i, &b1) in section1.b.iter().enumerate() {
            for (j, &b2) in section2.b.iter().enumerate() {
                b[i + j] += b1 * b2;
            }
        }

        // Convolve denominator coefficients
        for (i, &a1) in section1.a.iter().enumerate() {
            for (j, &a2) in section2.a.iter().enumerate() {
                a[i + j] += a1 * a2;
            }
        }

        Ok(IirCoefficients { b, a })
    }
}