// src/processing/mod.rs
//! Signal processing pipeline for EMG data

pub mod filters_v1;

pub mod features;
pub mod filter_bank;
pub mod pipeline;
pub mod quality_monitor;
pub mod windowing;
pub mod filters;
mod noise_reduction;


pub use features::*;
pub use filter_bank::*;
pub use pipeline::*;
pub use quality_monitor::*;
pub use windowing::*;