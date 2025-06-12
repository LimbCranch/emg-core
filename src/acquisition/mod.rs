// src/acquisition/mod.rs
//! Signal acquisition and buffering components

pub mod ring_buffer;
pub mod sample_sync;
pub mod buffer_manager;

pub use ring_buffer::*;
pub use sample_sync::*;
pub use buffer_manager::*;