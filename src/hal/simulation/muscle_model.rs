//! Realistic muscle fiber and activation modeling
//! Location: src/hal/simulation/muscle_model.rs

use crate::hal::types::GestureType;
use super::config::MuscleConfig;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ActivationPattern {
    pub primary_channels: Vec<usize>,
    pub activation_weights: Vec<f32>,
    pub frequency_profile: FrequencyProfile,
    pub onset_delay_ms: f32,
}

#[derive(Debug, Clone)]
pub struct FrequencyProfile {
    pub dominant_frequency_hz: f32,
    pub frequency_spread: f32,
    pub high_frequency_component: f32,
}

#[derive(Debug, Clone)]
pub enum MuscleFiberType {
    TypeI,   // Slow-twitch, low force, fatigue resistant
    TypeIIa, // Fast-twitch, moderate force
    TypeIIx, // Fast-twitch, high force, fatigue quickly
}

pub struct FatigueModel {
    fiber_fatigue_levels: Vec<f32>,
    recovery_rates: Vec<f32>,
    current_mvc_factor: f32,
}

pub struct MuscleModel {
    fiber_types: Vec<MuscleFiberType>,
    activation_patterns: HashMap<GestureType, ActivationPattern>,
    fatigue_model: FatigueModel,
    config: MuscleConfig,
}

impl MuscleModel {
    pub fn new(config: &MuscleConfig) -> Self {
        let activation_patterns = Self::create_gesture_patterns();
        let fiber_types = Self::initialize_fiber_types();
        let fatigue_model = FatigueModel::new(&fiber_types, config);

        Self {
            fiber_types,
            activation_patterns,
            fatigue_model,
            config: config.clone(),
        }
    }

    pub fn generate_activation(&mut self, gesture: GestureType, activation_level: f32, channel_idx: usize) -> f32 {
        let pattern = match self.activation_patterns.get(&gesture) {
            Some(p) => p,
            None => return self.generate_rest_signal(channel_idx),
        };

        // Check if this channel is involved in the gesture
        let channel_weight = pattern.activation_weights.get(channel_idx).copied().unwrap_or(0.0);

        if channel_weight == 0.0 {
            return self.generate_rest_signal(channel_idx);
        }

        // Apply fatigue effects
        let fatigue_factor = self.fatigue_model.get_fatigue_factor(channel_idx);
        let effective_activation = activation_level * channel_weight * fatigue_factor;

        // Generate physiologically realistic EMG signal
        let signal = self.generate_muap_train(effective_activation, &pattern.frequency_profile);

        // Update fatigue state
        self.fatigue_model.update_fatigue(channel_idx, effective_activation);

        signal
    }

    fn create_gesture_patterns() -> HashMap<GestureType, ActivationPattern> {
        let mut patterns = HashMap::new();

        // Hand closure pattern - flexor muscles
        patterns.insert(GestureType::HandClose, ActivationPattern {
            primary_channels: vec![0, 1, 4, 5],
            activation_weights: vec![0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1, 0.1],
            frequency_profile: FrequencyProfile {
                dominant_frequency_hz: 85.0,
                frequency_spread: 25.0,
                high_frequency_component: 0.3,
            },
            onset_delay_ms: 80.0,
        });

        // Hand opening pattern - extensor muscles
        patterns.insert(GestureType::HandOpen, ActivationPattern {
            primary_channels: vec![2, 3, 6, 7],
            activation_weights: vec![0.2, 0.3, 0.9, 0.8, 0.1, 0.2, 0.7, 0.6],
            frequency_profile: FrequencyProfile {
                dominant_frequency_hz: 75.0,
                frequency_spread: 20.0,
                high_frequency_component: 0.25,
            },
            onset_delay_ms: 120.0,
        });

        // Wrist flexion
        patterns.insert(GestureType::WristFlexion, ActivationPattern {
            primary_channels: vec![0, 4],
            activation_weights: vec![0.8, 0.3, 0.1, 0.1, 0.9, 0.2, 0.1, 0.1],
            frequency_profile: FrequencyProfile {
                dominant_frequency_hz: 90.0,
                frequency_spread: 30.0,
                high_frequency_component: 0.4,
            },
            onset_delay_ms: 60.0,
        });

        // Wrist extension
        patterns.insert(GestureType::WristExtension, ActivationPattern {
            primary_channels: vec![2, 6],
            activation_weights: vec![0.1, 0.1, 0.8, 0.3, 0.1, 0.1, 0.9, 0.2],
            frequency_profile: FrequencyProfile {
                dominant_frequency_hz: 80.0,
                frequency_spread: 25.0,
                high_frequency_component: 0.35,
            },
            onset_delay_ms: 70.0,
        });

        patterns
    }

    fn initialize_fiber_types() -> Vec<MuscleFiberType> {
        vec![
            MuscleFiberType::TypeI,
            MuscleFiberType::TypeIIa,
            MuscleFiberType::TypeI,
            MuscleFiberType::TypeIIx,
            MuscleFiberType::TypeI,
            MuscleFiberType::TypeIIa,
            MuscleFiberType::TypeI,
            MuscleFiberType::TypeIIx,
        ]
    }

    fn generate_rest_signal(&self, _channel_idx: usize) -> f32 {
        // Low-level baseline EMG activity
        (rand::random::<f32>() - 0.5) * 0.01
    }

    fn generate_muap_train(&self, activation_level: f32, frequency_profile: &FrequencyProfile) -> f32 {
        // Generate motor unit action potential train
        let firing_rate = 8.0 + activation_level * frequency_profile.dominant_frequency_hz;
        let amplitude_variation = 1.0 + (rand::random::<f32>() - 0.5) * 0.3;

        // Simulate recruitment curve
        let recruitment_factor = activation_level.powf(self.config.fiber_recruitment_curve);

        // Base MUAP signal with realistic characteristics
        let phase = rand::random::<f32>() * 2.0 * std::f32::consts::PI;
        let base_signal = (firing_rate * phase).sin() * recruitment_factor;

        // Add higher frequency components
        let hf_component = (firing_rate * 3.0 * phase).sin() *
            frequency_profile.high_frequency_component * recruitment_factor;

        (base_signal + hf_component) * amplitude_variation * self.config.maximum_voluntary_contraction
    }
}

impl FatigueModel {
    fn new(fiber_types: &[MuscleFiberType], config: &MuscleConfig) -> Self {
        let fiber_count = fiber_types.len();
        Self {
            fiber_fatigue_levels: vec![0.0; fiber_count],
            recovery_rates: fiber_types.iter().map(|ft| match ft {
                MuscleFiberType::TypeI => config.recovery_rate * 1.5,
                MuscleFiberType::TypeIIa => config.recovery_rate,
                MuscleFiberType::TypeIIx => config.recovery_rate * 0.7,
            }).collect(),
            current_mvc_factor: 1.0,
        }
    }

    fn get_fatigue_factor(&self, channel_idx: usize) -> f32 {
        let fatigue_level = self.fiber_fatigue_levels.get(channel_idx).copied().unwrap_or(0.0);
        (1.0 - fatigue_level).max(0.3) // Minimum 30% capacity even when fatigued
    }

    fn update_fatigue(&mut self, channel_idx: usize, activation_level: f32) {
        if let Some(fatigue) = self.fiber_fatigue_levels.get_mut(channel_idx) {
            // Increase fatigue based on activation
            *fatigue += activation_level * 0.001;
            *fatigue = fatigue.min(0.7); // Maximum 70% fatigue

            // Natural recovery
            if let Some(recovery_rate) = self.recovery_rates.get(channel_idx) {
                *fatigue -= recovery_rate * 0.0001;
                *fatigue = fatigue.max(0.0);
            }
        }
    }
}