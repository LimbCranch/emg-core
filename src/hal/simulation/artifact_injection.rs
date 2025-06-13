//! Realistic artifact injection for EMG simulation
//! Location: src/hal/simulation/artifact_injection.rs

use super::config::ArtifactConfig;

pub struct ArtifactInjector {
    config: ArtifactConfig,
    motion_state: MotionArtifactState,
    electrode_state: ElectrodeArtifactState,
    cable_state: CableMovementState,
    baseline_drift: BaselineDriftState,
    last_injection: bool,
    sample_count: u64,
}

struct MotionArtifactState {
    is_active: bool,
    duration_remaining: u32,
    amplitude: f32,
    frequency_hz: f32,
    phase: f32,
    artifact_type: MotionType,
}

struct ElectrodeArtifactState {
    pop_countdown: u32,
    baseline_shift: f32,
    saturation_countdown: u32,
    impedance_drift: f32,
}

struct CableMovementState {
    movement_active: bool,
    movement_duration: u32,
    movement_amplitude: f32,
    spike_pattern: SpikePattern,
}

struct BaselineDriftState {
    current_drift: f32,
    drift_rate: f32,
    target_drift: f32,
    correction_countdown: u32,
}

#[derive(Debug, Clone)]
enum MotionType {
    LimbMovement,      // Large amplitude, low frequency
    MuscleContraction, // Medium amplitude, medium frequency
    Tremor,            // Small amplitude, high frequency
}

#[derive(Debug, Clone)]
enum SpikePattern {
    Single,            // One large spike
    Burst,             // Multiple rapid spikes
    Oscillatory,       // Damped oscillation
}

impl ArtifactInjector {
    pub fn new(config: &ArtifactConfig) -> Self {
        Self {
            config: config.clone(),
            motion_state: MotionArtifactState::new(),
            electrode_state: ElectrodeArtifactState::new(),
            cable_state: CableMovementState::new(),
            baseline_drift: BaselineDriftState::new(),
            last_injection: false,
            sample_count: 0,
        }
    }

    pub fn maybe_inject_artifact(&mut self, clean_signal: f32) -> f32 {
        let mut signal = clean_signal;
        self.last_injection = false;
        self.sample_count += 1;

        // Baseline drift (always active at low level)
        signal += self.update_baseline_drift();

        // Motion artifacts (episodic, high impact)
        if let Some(motion_artifact) = self.generate_motion_artifact() {
            signal += motion_artifact;
            self.last_injection = true;
        }

        // Electrode artifacts (sudden, localized)
        if let Some(electrode_artifact) = self.generate_electrode_artifact() {
            signal += electrode_artifact;
            self.last_injection = true;
        }

        // Cable movement artifacts (transient, electrical)
        if let Some(cable_artifact) = self.generate_cable_movement_artifact() {
            signal += cable_artifact;
            self.last_injection = true;
        }

        // Apply realistic signal bounds with soft clipping
        self.apply_signal_conditioning(signal)
    }

    pub fn last_injection_occurred(&self) -> bool {
        self.last_injection
    }

    pub fn reset_state(&mut self) {
        self.motion_state = MotionArtifactState::new();
        self.electrode_state = ElectrodeArtifactState::new();
        self.cable_state = CableMovementState::new();
        self.baseline_drift = BaselineDriftState::new();
        self.sample_count = 0;
    }

    fn update_baseline_drift(&mut self) -> f32 {
        // Continuous low-frequency baseline drift
        if self.baseline_drift.correction_countdown == 0 {
            // Start new drift cycle
            self.baseline_drift.target_drift = (rand::random::<f32>() - 0.5) * 0.05;
            self.baseline_drift.drift_rate = (self.baseline_drift.target_drift - self.baseline_drift.current_drift) / 5000.0;
            self.baseline_drift.correction_countdown = 5000 + rand::random::<u32>() % 10000; // 2.5-7.5s
        }

        self.baseline_drift.current_drift += self.baseline_drift.drift_rate;
        self.baseline_drift.correction_countdown -= 1;

        self.baseline_drift.current_drift
    }

    fn generate_motion_artifact(&mut self) -> Option<f32> {
        // Check if motion artifact should start
        if !self.motion_state.is_active {
            if rand::random::<f32>() < self.config.motion_artifact_probability {
                self.motion_state.start_motion_artifact(&self.config);
            } else {
                return None;
            }
        }

        if self.motion_state.is_active {
            Some(self.motion_state.generate_sample())
        } else {
            None
        }
    }

    fn generate_electrode_artifact(&mut self) -> Option<f32> {
        let mut total_artifact = 0.0;
        let mut artifact_detected = false;

        // Electrode pop/disconnection
        if rand::random::<f32>() < self.config.electrode_pop_probability {
            self.electrode_state.baseline_shift = (rand::random::<f32>() - 0.5) * 0.8;
            self.electrode_state.pop_countdown = 150 + rand::random::<u32>() % 100; // 75-125ms
            artifact_detected = true;
        }

        // Process ongoing electrode pop
        if self.electrode_state.pop_countdown > 0 {
            let decay_factor = (self.electrode_state.pop_countdown as f32 / 200.0).exp();
            total_artifact += self.electrode_state.baseline_shift * decay_factor;
            self.electrode_state.pop_countdown -= 1;
            artifact_detected = true;
        }

        // Electrode saturation (rare but severe)
        if rand::random::<f32>() < 0.0001 {
            self.electrode_state.saturation_countdown = 50 + rand::random::<u32>() % 50;
            artifact_detected = true;
        }

        if self.electrode_state.saturation_countdown > 0 {
            let saturation_level = 2.0 * (1.0 + 0.5 * (rand::random::<f32>() - 0.5));
            total_artifact += saturation_level;
            self.electrode_state.saturation_countdown -= 1;
            artifact_detected = true;
        }

        // Impedance drift artifacts
        if self.sample_count % 100 == 0 {
            // Update every 50ms
            self.electrode_state.impedance_drift += (rand::random::<f32>() - 0.5) * 0.001;
            self.electrode_state.impedance_drift = self.electrode_state.impedance_drift.clamp(-0.02, 0.02);
        }
        total_artifact += self.electrode_state.impedance_drift;

        if artifact_detected || self.electrode_state.impedance_drift.abs() > 0.001 {
            Some(total_artifact)
        } else {
            None
        }
    }

    fn generate_cable_movement_artifact(&mut self) -> Option<f32> {
        // Check if cable movement should start
        if !self.cable_state.movement_active {
            if rand::random::<f32>() < self.config.cable_movement_probability {
                self.cable_state.start_cable_movement();
            } else {
                return None;
            }
        }

        if self.cable_state.movement_active {
            Some(self.cable_state.generate_sample())
        } else {
            None
        }
    }

    fn apply_signal_conditioning(&self, signal: f32) -> f32 {
        // Realistic amplifier characteristics
        let max_range = 5.0; // ±5mV typical EMG range

        // Soft clipping (more realistic than hard clipping)
        if signal.abs() > max_range {
            let sign = signal.signum();
            let excess = signal.abs() - max_range;
            let clipped = max_range + excess / (1.0 + excess);
            sign * clipped
        } else {
            signal
        }
    }
}

impl MotionArtifactState {
    fn new() -> Self {
        Self {
            is_active: false,
            duration_remaining: 0,
            amplitude: 0.0,
            frequency_hz: 0.0,
            phase: 0.0,
            artifact_type: MotionType::LimbMovement,
        }
    }

    fn start_motion_artifact(&mut self, config: &ArtifactConfig) {
        self.is_active = true;

        // Randomly select motion type
        let motion_type_rand = rand::random::<f32>();
        self.artifact_type = if motion_type_rand < 0.6 {
            MotionType::LimbMovement
        } else if motion_type_rand < 0.9 {
            MotionType::MuscleContraction
        } else {
            MotionType::Tremor
        };

        // Configure based on motion type
        match self.artifact_type {
            MotionType::LimbMovement => {
                self.duration_remaining = 1000 + rand::random::<u32>() % 3000; // 0.5-2s
                self.amplitude = config.motion_artifact_amplitude * (1.0 + rand::random::<f32>() * 2.0);
                self.frequency_hz = 0.5 + rand::random::<f32>() * 3.0; // 0.5-3.5 Hz
            },
            MotionType::MuscleContraction => {
                self.duration_remaining = 200 + rand::random::<u32>() % 600; // 0.1-0.4s
                self.amplitude = config.motion_artifact_amplitude * (0.3 + rand::random::<f32>() * 0.7);
                self.frequency_hz = 8.0 + rand::random::<f32>() * 12.0; // 8-20 Hz
            },
            MotionType::Tremor => {
                self.duration_remaining = 2000 + rand::random::<u32>() % 4000; // 1-3s
                self.amplitude = config.motion_artifact_amplitude * (0.1 + rand::random::<f32>() * 0.3);
                self.frequency_hz = 3.0 + rand::random::<f32>() * 9.0; // 3-12 Hz
            },
        }

        self.phase = rand::random::<f32>() * 2.0 * std::f32::consts::PI;
    }

    fn generate_sample(&mut self) -> f32 {
        if self.duration_remaining == 0 {
            self.is_active = false;
            return 0.0;
        }

        self.duration_remaining -= 1;

        // Update phase
        self.phase += 2.0 * std::f32::consts::PI * self.frequency_hz / 2000.0;
        if self.phase > 2.0 * std::f32::consts::PI {
            self.phase -= 2.0 * std::f32::consts::PI;
        }

        // Generate envelope based on motion type
        let envelope = match self.artifact_type {
            MotionType::LimbMovement => {
                // Gradual onset and offset
                let progress = 1.0 - (self.duration_remaining as f32 / 2000.0);
                if progress < 0.2 {
                    progress / 0.2 // Ramp up
                } else if progress > 0.8 {
                    (1.0 - progress) / 0.2 // Ramp down
                } else {
                    1.0
                }
            },
            MotionType::MuscleContraction => {
                // Sharp onset, exponential decay
                let decay_factor = (self.duration_remaining as f32 / 500.0).exp();
                decay_factor.min(1.0)
            },
            MotionType::Tremor => {
                // Tremor with irregular amplitude
                let base_envelope = (self.duration_remaining as f32 / 3000.0).sqrt();
                let tremor_modulation = 1.0 + 0.3 * (self.phase * 0.1).sin();
                base_envelope * tremor_modulation
            },
        };

        // Generate base waveform
        let base_waveform = match self.artifact_type {
            MotionType::LimbMovement => self.phase.sin(),
            MotionType::MuscleContraction => {
                // More complex waveform with harmonics
                self.phase.sin() + 0.3 * (2.0 * self.phase).sin() + 0.1 * (3.0 * self.phase).sin()
            },
            MotionType::Tremor => {
                // Irregular tremor pattern
                self.phase.sin() + 0.2 * (1.7 * self.phase).sin() + 0.1 * rand::random::<f32>()
            },
        };

        base_waveform * self.amplitude * envelope
    }
}

impl ElectrodeArtifactState {
    fn new() -> Self {
        Self {
            pop_countdown: 0,
            baseline_shift: 0.0,
            saturation_countdown: 0,
            impedance_drift: 0.0,
        }
    }
}

impl CableMovementState {
    fn new() -> Self {
        Self {
            movement_active: false,
            movement_duration: 0,
            movement_amplitude: 0.0,
            spike_pattern: SpikePattern::Single,
        }
    }

    fn start_cable_movement(&mut self) {
        self.movement_active = true;
        self.movement_amplitude = 0.05 + rand::random::<f32>() * 0.4;

        // Select spike pattern
        let pattern_rand = rand::random::<f32>();
        self.spike_pattern = if pattern_rand < 0.5 {
            SpikePattern::Single
        } else if pattern_rand < 0.8 {
            SpikePattern::Burst
        } else {
            SpikePattern::Oscillatory
        };

        self.movement_duration = match self.spike_pattern {
            SpikePattern::Single => 20 + rand::random::<u32>() % 30,      // 10-25ms
            SpikePattern::Burst => 60 + rand::random::<u32>() % 80,       // 30-70ms
            SpikePattern::Oscillatory => 100 + rand::random::<u32>() % 200, // 50-150ms
        };
    }

    fn generate_sample(&mut self) -> f32 {
        if self.movement_duration == 0 {
            self.movement_active = false;
            return 0.0;
        }

        let initial_duration = match self.spike_pattern {
            SpikePattern::Single => 40.0,
            SpikePattern::Burst => 120.0,
            SpikePattern::Oscillatory => 250.0,
        };

        let time_progress = 1.0 - (self.movement_duration as f32 / initial_duration);
        self.movement_duration -= 1;

        let artifact = match self.spike_pattern {
            SpikePattern::Single => {
                // Single exponential decay spike
                let decay = (-time_progress * 8.0).exp();
                (rand::random::<f32>() - 0.5) * decay
            },
            SpikePattern::Burst => {
                // Multiple rapid spikes
                let burst_freq = 200.0; // 200 Hz burst
                let phase = time_progress * 2.0 * std::f32::consts::PI * burst_freq * initial_duration / 2000.0;
                let envelope = (-time_progress * 5.0).exp();
                phase.sin() * envelope * (0.5 + 0.5 * rand::random::<f32>())
            },
            SpikePattern::Oscillatory => {
                // Damped oscillation
                let osc_freq = 50.0 + rand::random::<f32>() * 100.0; // 50-150 Hz
                let phase = time_progress * 2.0 * std::f32::consts::PI * osc_freq * initial_duration / 2000.0;
                let damping = (-time_progress * 3.0).exp();
                phase.sin() * damping
            },
        };

        artifact * self.movement_amplitude
    }
}

impl BaselineDriftState {
    fn new() -> Self {
        Self {
            current_drift: 0.0,
            drift_rate: 0.0,
            target_drift: 0.0,
            correction_countdown: 1000,
        }
    }
}