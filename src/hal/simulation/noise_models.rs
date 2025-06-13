//! Realistic noise modeling for EMG simulation
//! Location: src/hal/simulation/noise_models.rs

use super::config::NoiseConfig;

pub struct NoiseModel {
    config: NoiseConfig,
    powerline_phase: f32,
    thermal_generator: ThermalNoiseGenerator,
    powerline_generator: PowerlineNoiseGenerator,
    electrode_noise: ElectrodeNoiseModel,
}

struct ThermalNoiseGenerator {
    power_dbm: f32,
    previous_sample: f32,
    alpha: f32, // Low-pass filter coefficient for colored noise
}

struct PowerlineNoiseGenerator {
    frequency_hz: f32,
    amplitude: f32,
    phase: f32,
    harmonics: Vec<HarmonicComponent>,
}

struct HarmonicComponent {
    frequency_multiplier: f32,
    amplitude_factor: f32,
    phase_offset: f32,
}

struct ElectrodeNoiseModel {
    base_impedance: f32,
    impedance_variance: f32,
    contact_noise_factor: f32,
}

impl NoiseModel {
    pub fn new(config: &NoiseConfig) -> Self {
        Self {
            thermal_generator: ThermalNoiseGenerator::new(config.thermal_noise_power_dbm),
            powerline_generator: PowerlineNoiseGenerator::new(
                config.powerline_frequency_hz,
                config.powerline_amplitude_factor,
            ),
            electrode_noise: ElectrodeNoiseModel::new(
                config.electrode_impedance_base,
                config.electrode_impedance_variance,
            ),
            config: config.clone(),
            powerline_phase: 0.0,
        }
    }

    pub fn add_noise(&mut self, clean_signal: f32) -> f32 {
        let mut noisy_signal = clean_signal;

        // Add thermal noise (Johnson noise)
        noisy_signal += self.thermal_generator.generate_sample();

        // Add powerline interference
        noisy_signal += self.powerline_generator.generate_sample(&mut self.powerline_phase);

        // Add electrode contact noise
        noisy_signal += self.electrode_noise.generate_contact_noise();

        noisy_signal
    }

    pub fn update_powerline_phase(&mut self, sample_rate_hz: u32) {
        self.powerline_phase += 2.0 * std::f32::consts::PI * self.config.powerline_frequency_hz / sample_rate_hz as f32;
        if self.powerline_phase > 2.0 * std::f32::consts::PI {
            self.powerline_phase -= 2.0 * std::f32::consts::PI;
        }
    }
}

impl ThermalNoiseGenerator {
    fn new(power_dbm: f32) -> Self {
        Self {
            power_dbm,
            previous_sample: 0.0,
            alpha: 0.95, // Creates 1/f characteristic
        }
    }

    fn generate_sample(&mut self) -> f32 {
        // Convert dBm to linear power (assuming 50 ohm reference)
        let power_linear = 10_f32.powf(self.power_dbm / 10.0) / 1000.0; // Watts
        let rms_voltage = (power_linear * 50.0).sqrt();

        // Generate white Gaussian noise
        let white_noise = self.box_muller_transform() * rms_voltage;

        // Apply first-order low-pass to create colored noise (1/f characteristic)
        self.previous_sample = self.alpha * self.previous_sample + (1.0 - self.alpha) * white_noise;
        self.previous_sample
    }

    fn box_muller_transform(&self) -> f32 {
        // Box-Muller transform for Gaussian random numbers
        let u1 = rand::random::<f32>();
        let u2 = rand::random::<f32>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

impl PowerlineNoiseGenerator {
    fn new(frequency_hz: f32, amplitude_factor: f32) -> Self {
        let harmonics = vec![
            // Fundamental
            HarmonicComponent {
                frequency_multiplier: 1.0,
                amplitude_factor: 1.0,
                phase_offset: 0.0,
            },
            // Second harmonic (less common but present)
            HarmonicComponent {
                frequency_multiplier: 2.0,
                amplitude_factor: 0.3,
                phase_offset: std::f32::consts::PI / 4.0,
            },
            // Third harmonic
            HarmonicComponent {
                frequency_multiplier: 3.0,
                amplitude_factor: 0.15,
                phase_offset: std::f32::consts::PI / 6.0,
            },
        ];

        Self {
            frequency_hz,
            amplitude: amplitude_factor,
            phase: 0.0,
            harmonics,
        }
    }

    fn generate_sample(&mut self, phase: &mut f32) -> f32 {
        let mut interference = 0.0;

        for harmonic in &self.harmonics {
            let harmonic_phase = *phase * harmonic.frequency_multiplier + harmonic.phase_offset;
            interference += harmonic_phase.sin() * harmonic.amplitude_factor;
        }

        interference * self.amplitude
    }
}

impl ElectrodeNoiseModel {
    fn new(base_impedance: f32, impedance_variance: f32) -> Self {
        Self {
            base_impedance,
            impedance_variance,
            contact_noise_factor: 0.001, // Baseline contact noise level
        }
    }

    fn generate_contact_noise(&self) -> f32 {
        // Simulate impedance-dependent contact noise
        let current_impedance = self.base_impedance +
            (rand::random::<f32>() - 0.5) * self.impedance_variance;

        // Higher impedance = more noise
        let impedance_factor = (current_impedance / self.base_impedance).min(3.0);

        // Generate burst-type contact noise (occasional spikes)
        let burst_probability = 0.001;
        let burst_amplitude = if rand::random::<f32>() < burst_probability {
            (rand::random::<f32>() - 0.5) * 0.1 * impedance_factor
        } else {
            0.0
        };

        // Continuous low-level contact noise
        let continuous_noise = (rand::random::<f32>() - 0.5) *
            self.contact_noise_factor *
            impedance_factor.sqrt();

        burst_amplitude + continuous_noise
    }
}