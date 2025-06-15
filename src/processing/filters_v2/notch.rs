
// ================================================================================
// Advanced Notch Filter Implementation
// ================================================================================

/// Advanced notch filter for power line interference removal
pub struct PowerLineNotchFilter {
    sections: Vec<NotchSection>,
    fundamental_freq: f32,
    harmonics: Vec<u32>,
}

struct NotchSection {
    b0: f32, b1: f32, b2: f32,
    a1: f32, a2: f32,
    x1: f32, x2: f32,
    y1: f32, y2: f32,
    center_freq: f32,
    q_factor: f32,
}

impl PowerLineNotchFilter {
    /// Create notch filter for power line interference (50Hz or 60Hz) with harmonics
    pub fn new(fundamental_freq: f32, sample_rate: f32, max_harmonic: u32, q_factor: f32) -> Result<Self, FilterError> {
        let mut sections = Vec::new();
        let mut harmonics = Vec::new();

        // Create notch sections for fundamental and harmonics
        for harmonic in 1..=max_harmonic {
            let freq = fundamental_freq * harmonic as f32;
            if freq < sample_rate / 2.0 {
                let section = NotchSection::new(freq, q_factor, sample_rate)?;
                sections.push(section);
                harmonics.push(harmonic);
            }
        }

        Ok(Self {
            sections,
            fundamental_freq,
            harmonics,
        })
    }

    /// Create adaptive notch filter that can adjust frequency in real-time
    pub fn adaptive(initial_freq: f32, sample_rate: f32, adaptation_rate: f32) -> Result<AdaptiveNotchFilter, FilterError> {
        AdaptiveNotchFilter::new(initial_freq, sample_rate, adaptation_rate)
    }

    /// Process sample through all notch sections
    pub fn process_sample(&mut self, input: f32) -> f32 {
        let mut output = input;
        for section in &mut self.sections {
            output = section.process(output);
        }
        output
    }

    /// Reset all filter states
    pub fn reset(&mut self) {
        for section in &mut self.sections {
            section.reset();
        }
    }

    /// Update filter frequencies (for adaptive operation)
    pub fn update_frequencies(&mut self, new_fundamental: f32, sample_rate: f32) -> Result<(), FilterError> {
        for (i, section) in self.sections.iter_mut().enumerate() {
            let new_freq = new_fundamental * self.harmonics[i] as f32;
            if new_freq < sample_rate / 2.0 {
                section.update_frequency(new_freq, sample_rate)?;
            }
        }
        self.fundamental_freq = new_fundamental;
        Ok(())
    }
}

impl NotchSection {
    fn new(center_freq: f32, q_factor: f32, sample_rate: f32) -> Result<Self, FilterError> {
        let mut section = Self {
            b0: 0.0, b1: 0.0, b2: 0.0,
            a1: 0.0, a2: 0.0,
            x1: 0.0, x2: 0.0,
            y1: 0.0, y2: 0.0,
            center_freq,
            q_factor,
        };

        section.calculate_coefficients(center_freq, q_factor, sample_rate)?;
        Ok(section)
    }

    fn calculate_coefficients(&mut self, center_freq: f32, q_factor: f32, sample_rate: f32) -> Result<(), FilterError> {
        let omega = 2.0 * PI * center_freq / sample_rate;
        let cos_omega = omega.cos();
        let sin_omega = omega.sin();
        let alpha = sin_omega / (2.0 * q_factor);

        let norm = 1.0 + alpha;

        self.b0 = 1.0 / norm;
        self.b1 = -2.0 * cos_omega / norm;
        self.b2 = 1.0 / norm;
        self.a1 = -2.0 * cos_omega / norm;
        self.a2 = (1.0 - alpha) / norm;

        Ok(())
    }

    fn update_frequency(&mut self, new_freq: f32, sample_rate: f32) -> Result<(), FilterError> {
        self.center_freq = new_freq;
        self.calculate_coefficients(new_freq, self.q_factor, sample_rate)
    }

    fn process(&mut self, input: f32) -> f32 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1 - self.a2 * self.y2;

        // Update delay line
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;

        output
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Adaptive notch filter that can track varying interference frequencies
pub struct AdaptiveNotchFilter {
    notch_section: NotchSection,
    frequency_estimator: FrequencyEstimator,
    adaptation_rate: f32,
    sample_rate: f32,
}

struct FrequencyEstimator {
    correlation_buffer: Vec<f32>,
    buffer_index: usize,
    buffer_size: usize,
    estimated_freq: f32,
}

impl AdaptiveNotchFilter {
    fn new(initial_freq: f32, sample_rate: f32, adaptation_rate: f32) -> Result<Self, FilterError> {
        let notch_section = NotchSection::new(initial_freq, 10.0, sample_rate)?;
        let frequency_estimator = FrequencyEstimator::new(initial_freq, sample_rate as usize / 10);

        Ok(Self {
            notch_section,
            frequency_estimator,
            adaptation_rate,
            sample_rate,
        })
    }

    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Update frequency estimate
        self.frequency_estimator.update(input);

        // Adapt notch frequency if estimate has changed significantly
        let estimated_freq = self.frequency_estimator.get_frequency();
        let freq_diff = (estimated_freq - self.notch_section.center_freq).abs();

        if freq_diff > 0.5 {  // 0.5 Hz threshold
            let new_freq = self.notch_section.center_freq +
                self.adaptation_rate * (estimated_freq - self.notch_section.center_freq);
            let _ = self.notch_section.update_frequency(new_freq, self.sample_rate);
        }

        // Process sample
        self.notch_section.process(input)
    }

    pub fn reset(&mut self) {
        self.notch_section.reset();
        self.frequency_estimator.reset();
    }
}

impl FrequencyEstimator {
    fn new(initial_freq: f32, buffer_size: usize) -> Self {
        Self {
            correlation_buffer: vec![0.0; buffer_size],
            buffer_index: 0,
            buffer_size,
            estimated_freq: initial_freq,
        }
    }

    fn update(&mut self, sample: f32) {
        self.correlation_buffer[self.buffer_index] = sample;
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size;

        // Simple autocorrelation-based frequency estimation
        if self.buffer_index == 0 {
            self.estimated_freq = self.estimate_frequency();
        }
    }

    fn estimate_frequency(&self) -> f32 {
        // Simplified frequency estimation using zero crossings
        let mut zero_crossings = 0;
        for i in 1..self.buffer_size {
            if (self.correlation_buffer[i-1] >= 0.0) != (self.correlation_buffer[i] >= 0.0) {
                zero_crossings += 1;
            }
        }

        // Frequency = zero_crossings / (2 * time_window)
        let time_window = self.buffer_size as f32 / 2000.0; // Assuming 2kHz sample rate
        (zero_crossings as f32 / 2.0) / time_window
    }

    fn get_frequency(&self) -> f32 {
        self.estimated_freq
    }

    fn reset(&mut self) {
        self.correlation_buffer.fill(0.0);
        self.buffer_index = 0;
    }
}
