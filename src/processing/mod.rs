// src/processing/mod.rs
//! Signal processing pipeline for EMG data

pub mod filters;
pub mod filter_bank;
pub mod pipeline;
pub mod quality_monitor;
pub mod windowing;

pub use filters::*;
pub use filter_bank::*;
pub use pipeline::*;
pub use quality_monitor::*;
pub use windowing::*;