//! EMG simulation module for realistic device emulation
//! Location: src/hal/simulation/mod.rs

pub mod signal_generator;
pub mod noise_models;
pub mod artifact_injection;
pub mod muscle_model;
pub mod profiles;

pub mod config;


pub use signal_generator::EmgSignalGenerator;
pub use noise_models::NoiseModel;
pub use artifact_injection::ArtifactInjector;
pub use muscle_model::MuscleModel;
pub use profiles::SimulationProfile;