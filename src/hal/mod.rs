// src/hal/mod.rs
//! Hardware Abstraction Layer for EMG devices

pub mod traits;
pub mod types;
pub mod simulator;
pub mod usb_driver;
pub mod serial_driver;

#[cfg(test)]
mod tests;
mod simulation;

pub use traits::*;
pub use types::*;