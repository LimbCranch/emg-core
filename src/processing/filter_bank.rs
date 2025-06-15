// src/processing/filter_bank.rs
//! Filter bank combining multiple filters_v1

use std::sync::Arc;
use std::time::{Duration, Instant};
use rayon::prelude::*;
use crate::error::EmgError;
use crate::processing::filters::{IIRFilter, FIRFilter, NotchFilter, FilterType, WindowType};
use crate::processing::quality_monitor::ChannelQualityMonitor;

/// Configuration for a single filter in the chain
#[derive(Debug, Clone)]
pub struct FilterConfig {
    pub filter_type: FilterType,
    pub enabled: bool,
    pub parameters: FilterParameters,
}

/// Filter parameters for different filter types
#[derive(Debug, Clone)]
pub enum FilterParameters {
    IIR {
        cutoff_freq: f32,
        bandwidth: Option<f32>,
        order: usize,
    },
    FIR {
        order: usize,
        window_type: WindowType,
        frequency_response: Vec<f32>,
    },
    Notch {
        center_freq: f32,
        bandwidth: f32,
        q_factor: f32,
        harmonics: Vec<f32>,
    },
}

/// Channel-specific filter configuration
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    pub channel_id: usize,
    pub filters: Vec<FilterConfig>,
    pub bypass: bool,
}

/// Filter bank configuration
#[derive(Debug, Clone)]
pub struct FilterBankConfig {
    pub channels: Vec<ChannelConfig>,
    pub sample_rate: f32,
    pub enable_simd: bool,
    pub enable_parallel: bool,
}

/// Performance metrics for filter bank
#[derive(Debug, Default)]
pub struct FilterBankMetrics {
    pub processing_time: Duration,
    pub latency: Duration,
    pub channel_quality: Vec<f32>,
}

/// Enhanced filter bank implementation
pub struct FilterBank {
    config: FilterBankConfig,
    filters: Vec<Vec<Box<dyn DigitalFilter>>>,
    quality_monitors: Vec<ChannelQualityMonitor>,
    metrics: FilterBankMetrics,
    last_process_time: Instant,
}

impl FilterBank {
    /// Create a new filter bank from configuration
    pub fn new(config: FilterBankConfig) -> Result<Self, EmgError> {
        let mut filters = Vec::with_capacity(config.channels.len());
        let mut quality_monitors = Vec::with_capacity(config.channels.len());

        // Initialize filters for each channel
        for channel_config in &config.channels {
            let mut channel_filters = Vec::new();
            
            for filter_config in &channel_config.filters {
                if !filter_config.enabled {
                    continue;
                }

                let filter: Box<dyn DigitalFilter> = match &filter_config.parameters {
                    FilterParameters::IIR { cutoff_freq, bandwidth, order } => {
                        Box::new(IIRFilter::new(
                            filter_config.filter_type,
                            config.sample_rate,
                            *cutoff_freq,
                            *bandwidth,
                            *order,
                        )?)
                    }
                    FilterParameters::FIR { order, window_type, frequency_response } => {
                        Box::new(FIRFilter::new(
                            *order,
                            *window_type,
                            frequency_response,
                        )?)
                    }
                    FilterParameters::Notch { center_freq, bandwidth, q_factor, harmonics } => {
                        let mut notch = NotchFilter::new(
                            *center_freq,
                            config.sample_rate,
                            *bandwidth,
                            *q_factor,
                        )?;
                        
                        for &harmonic in harmonics {
                            notch.add_harmonic(harmonic)?;
                        }
                        
                        Box::new(notch)
                    }
                };
                
                channel_filters.push(filter);
            }
            
            filters.push(channel_filters);
            quality_monitors.push(ChannelQualityMonitor::new(config.sample_rate));
        }

        Ok(Self {
            config,
            filters,
            quality_monitors,
            metrics: FilterBankMetrics::default(),
            last_process_time: Instant::now(),
        })
    }

    /// Process a block of multi-channel data
    pub fn process(&mut self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, EmgError> {
        let start_time = Instant::now();
        
        if input.len() != self.config.channels.len() {
            return Err(EmgError::Configuration(
                format!("Expected {} channels, got {}", self.config.channels.len(), input.len())
            ));
        }

        let mut output = Vec::with_capacity(input.len());
        
        // Process channels in parallel if enabled
        if self.config.enable_parallel {
            output = self.process_parallel(input)?;
        } else {
            output = self.process_sequential(input)?;
        }

        // Update metrics
        self.metrics.processing_time = start_time.elapsed();
        self.metrics.latency = self.last_process_time.elapsed();
        self.last_process_time = Instant::now();

        // Update quality metrics
        for (i, monitor) in self.quality_monitors.iter_mut().enumerate() {
            self.metrics.channel_quality[i] = monitor.update_quality(&output[i]);
        }

        Ok(output)
    }

    /// Process channels sequentially
    fn process_sequential(&mut self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, EmgError> {
        let mut output = Vec::with_capacity(input.len());
        
        for (channel_idx, channel_data) in input.iter().enumerate() {
            if self.config.channels[channel_idx].bypass {
                output.push(channel_data.clone());
                continue;
            }

            let mut channel_output = channel_data.clone();
            
            for filter in &mut self.filters[channel_idx] {
                channel_output = filter.process(&channel_output);
            }
            
            output.push(channel_output);
        }
        
        Ok(output)
    }

    /// Process channels in parallel using rayon
    fn process_parallel(&mut self, input: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, EmgError> {
        let output: Vec<Vec<f32>> = input.par_iter()
            .enumerate()
            .map(|(channel_idx, channel_data)| {
                if self.config.channels[channel_idx].bypass {
                    return channel_data.clone();
                }

                let mut channel_output = channel_data.clone();
                
                for filter in &mut self.filters[channel_idx] {
                    channel_output = filter.process(&channel_output);
                }
                
                channel_output
            })
            .collect();
            
        Ok(output)
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> &FilterBankMetrics {
        &self.metrics
    }

    /// Reset all filter states
    pub fn reset(&mut self) {
        for channel_filters in &mut self.filters {
            for filter in channel_filters {
                filter.reset();
            }
        }
        
        for monitor in &mut self.quality_monitors {
            monitor.reset();
        }
        
        self.metrics = FilterBankMetrics::default();
    }

    /// Enable/disable SIMD acceleration
    pub fn set_simd_enabled(&mut self, enabled: bool) {
        self.config.enable_simd = enabled;
    }

    /// Enable/disable parallel processing
    pub fn set_parallel_enabled(&mut self, enabled: bool) {
        self.config.enable_parallel = enabled;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_bank_creation() {
        let config = FilterBankConfig {
            channels: vec![
                ChannelConfig {
                    channel_id: 0,
                    filters: vec![
                        FilterConfig {
                            filter_type: FilterType::LowPass,
                            enabled: true,
                            parameters: FilterParameters::IIR {
                                cutoff_freq: 500.0,
                                bandwidth: None,
                                order: 4,
                            },
                        }
                    ],
                    bypass: false,
                }
            ],
            sample_rate: 2000.0,
            enable_simd: true,
            enable_parallel: true,
        };

        let bank = FilterBank::new(config);
        assert!(bank.is_ok());
    }

    #[test]
    fn test_filter_bank_processing() {
        let config = FilterBankConfig {
            channels: vec![
                ChannelConfig {
                    channel_id: 0,
                    filters: vec![
                        FilterConfig {
                            filter_type: FilterType::LowPass,
                            enabled: true,
                            parameters: FilterParameters::IIR {
                                cutoff_freq: 500.0,
                                bandwidth: None,
                                order: 4,
                            },
                        }
                    ],
                    bypass: false,
                }
            ],
            sample_rate: 2000.0,
            enable_simd: true,
            enable_parallel: true,
        };

        let mut bank = FilterBank::new(config).unwrap();
        
        // Test processing
        let input = vec![vec![1.0; 1000]; 1];
        let output = bank.process(&input);
        assert!(output.is_ok());
        
        let output = output.unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].len(), 1000);
    }
}