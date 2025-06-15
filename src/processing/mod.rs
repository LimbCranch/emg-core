// src/processing/mod.rs
//! Signal processing pipeline for EMG data

pub mod filters;
pub mod features;
pub mod filter_bank;
pub mod pipeline;
pub mod quality_monitor;
pub mod windowing;
pub mod filters_v1;
pub mod noise_reduction;
pub mod quality;
pub mod artifacts;

pub use filters::*;
pub use features::*;
pub use filter_bank::*;
pub use pipeline::*;
pub use quality_monitor::*;
pub use windowing::*;
pub use quality::*;
pub use artifacts::*;