
use crate::error::EmgError;

pub trait Filter {
    fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, EmgError>;
    fn reset(&mut self);
    fn get_name(&self) -> &str;
}

/// Chain of digital filters for EMG processing
pub struct FilterChain {
    filters: Vec<Box<dyn Filter>>,
    intermediate_buffers: Vec<Vec<f32>>,
}

impl FilterChain {
    pub fn new(config: &super::pipeline::PipelineConfig) -> Result<Self, EmgError> {
        let mut filters: Vec<Box<dyn Filter>> = Vec::new();

        // High-pass filter to remove DC offset and low-frequency noise
        filters.push(Box::new(HighPassFilter::new(20.0, 2000.0)?)); // 20 Hz cutoff, 2kHz sampling

        // Band-pass filter for EMG frequency range (20-500 Hz)
        filters.push(Box::new(BandPassFilter::new(20.0, 500.0, 2000.0)?));

        // Notch filter for 50/60 Hz powerline interference
        filters.push(Box::new(NotchFilter::new(50.0, 2000.0, 10.0)?)); // 50 Hz, Q=10

        let intermediate_buffers = vec![Vec::new(); filters.len()];

        Ok(Self {
            filters,
            intermediate_buffers,
        })
    }

    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, EmgError> {
        let mut current_data = input.to_vec();

        for (i, filter) in self.filters.iter_mut().enumerate() {
            current_data = filter.process(&current_data)?;
            self.intermediate_buffers[i] = current_data.clone();
        }

        Ok(current_data)
    }

    pub fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
        for buffer in &mut self.intermediate_buffers {
            buffer.clear();
        }
    }
}

/// Second-order high-pass Butterworth filter
pub struct HighPassFilter {
    name: String,
    cutoff_freq: f32,
    sample_rate: f32,
    // Filter coefficients
    a1: f32,
    a2: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    // Filter memory (per channel)
    x1: Vec<f32>,
    x2: Vec<f32>,
    y1: Vec<f32>,
    y2: Vec<f32>,
}

impl HighPassFilter {
    pub fn new(cutoff_freq: f32, sample_rate: f32) -> Result<Self, EmgError> {
        if cutoff_freq <= 0.0 || cutoff_freq >= sample_rate / 2.0 {
            return Err(EmgError::Configuration(
                format!("Invalid cutoff frequency: {} Hz (sample rate: {} Hz)",
                        cutoff_freq, sample_rate)
            ));
        }

        // Calculate filter coefficients using bilinear transform
        let omega = 2.0 * std::f32::consts::PI * cutoff_freq / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let alpha = sin_omega / (2.0 * 0.707); // Q = 1/sqrt(2) for Butterworth

        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;
        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;

        Ok(Self {
            name: format!("HighPass-{}Hz", cutoff_freq),
            cutoff_freq,
            sample_rate,
            a1: a1 / a0,
            a2: a2 / a0,
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            x1: Vec::new(),
            x2: Vec::new(),
            y1: Vec::new(),
            y2: Vec::new(),
        })
    }
}

impl Filter for HighPassFilter {
    fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, EmgError> {
        let channel_count = input.len();

        // Initialize memory for new channels if needed
        while self.x1.len() < channel_count {
            self.x1.push(0.0);
            self.x2.push(0.0);
            self.y1.push(0.0);
            self.y2.push(0.0);
        }

        let mut output = Vec::with_capacity(channel_count);

        for (ch, &x0) in input.iter().enumerate() {
            // Apply filter difference equation
            let y0 = self.b0 * x0 + self.b1 * self.x1[ch] + self.b2 * self.x2[ch]
                - self.a1 * self.y1[ch] - self.a2 * self.y2[ch];

            output.push(y0);

            // Update filter memory
            self.x2[ch] = self.x1[ch];
            self.x1[ch] = x0;
            self.y2[ch] = self.y1[ch];
            self.y1[ch] = y0;
        }

        Ok(output)
    }

    fn reset(&mut self) {
        self.x1.fill(0.0);
        self.x2.fill(0.0);
        self.y1.fill(0.0);
        self.y2.fill(0.0);
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}

/// Band-pass filter for EMG signal conditioning
pub struct BandPassFilter {
    low_pass: HighPassFilter,
    high_pass: HighPassFilter,
    name: String,
}

impl BandPassFilter {
    pub fn new(low_cutoff: f32, high_cutoff: f32, sample_rate: f32) -> Result<Self, EmgError> {
        if low_cutoff >= high_cutoff {
            return Err(EmgError::Configuration(
                format!("Invalid band-pass range: {} - {} Hz", low_cutoff, high_cutoff)
            ));
        }

        let high_pass = HighPassFilter::new(low_cutoff, sample_rate)?;
        let low_pass = HighPassFilter::new(high_cutoff, sample_rate)?; // Will be modified for low-pass

        Ok(Self {
            low_pass,
            high_pass,
            name: format!("BandPass-{}-{}Hz", low_cutoff, high_cutoff),
        })
    }
}

impl Filter for BandPassFilter {
    fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, EmgError> {
        let high_passed = self.high_pass.process(input)?;
        self.low_pass.process(&high_passed)
    }

    fn reset(&mut self) {
        self.high_pass.reset();
        self.low_pass.reset();
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}

/// Notch filter for powerline interference removal
pub struct NotchFilter {
    name: String,
    center_freq: f32,
    sample_rate: f32,
    q_factor: f32,
    // Filter coefficients
    a1: f32,
    a2: f32,
    b0: f32,
    b1: f32,
    b2: f32,
    // Filter memory (per channel)
    x1: Vec<f32>,
    x2: Vec<f32>,
    y1: Vec<f32>,
    y2: Vec<f32>,
}

impl NotchFilter {
    pub fn new(center_freq: f32, sample_rate: f32, q_factor: f32) -> Result<Self, EmgError> {
        if center_freq <= 0.0 || center_freq >= sample_rate / 2.0 {
            return Err(EmgError::Configuration(
                format!("Invalid center frequency: {} Hz", center_freq)
            ));
        }

        if q_factor <= 0.0 {
            return Err(EmgError::Configuration(
                format!("Invalid Q factor: {}", q_factor)
            ));
        }

        // Calculate notch filter coefficients
        let omega = 2.0 * std::f32::consts::PI * center_freq / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let alpha = sin_omega / (2.0 * q_factor);

        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;
        let b0 = 1.0;
        let b1 = -2.0 * cos_omega;
        let b2 = 1.0;

        Ok(Self {
            name: format!("Notch-{}Hz-Q{}", center_freq, q_factor),
            center_freq,
            sample_rate,
            q_factor,
            a1: a1 / a0,
            a2: a2 / a0,
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            x1: Vec::new(),
            x2: Vec::new(),
            y1: Vec::new(),
            y2: Vec::new(),
        })
    }
}

impl Filter for NotchFilter {
    fn process(&mut self, input: &[f32]) -> Result<Vec<f32>, EmgError> {
        let channel_count = input.len();

        // Initialize memory for new channels if needed
        while self.x1.len() < channel_count {
            self.x1.push(0.0);
            self.x2.push(0.0);
            self.y1.push(0.0);
            self.y2.push(0.0);
        }

        let mut output = Vec::with_capacity(channel_count);

        for (ch, &x0) in input.iter().enumerate() {
            // Apply notch filter difference equation
            let y0 = self.b0 * x0 + self.b1 * self.x1[ch] + self.b2 * self.x2[ch]
                - self.a1 * self.y1[ch] - self.a2 * self.y2[ch];

            output.push(y0);

            // Update filter memory
            self.x2[ch] = self.x1[ch];
            self.x1[ch] = x0;
            self.y2[ch] = self.y1[ch];
            self.y1[ch] = y0;
        }

        Ok(output)
    }

    fn reset(&mut self) {
        self.x1.fill(0.0);
        self.x2.fill(0.0);
        self.y1.fill(0.0);
        self.y2.fill(0.0);
    }

    fn get_name(&self) -> &str {
        &self.name
    }
}
                    