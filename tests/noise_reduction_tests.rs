
// ================================================================================
// Integration and Testing
// File: tests/noise_reduction_tests.rs
// ================================================================================

use emg_core::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_adaptive_filter_basic() {
        let config = NoiseReductionConfig::default();
        let mut filter = AdaptiveFilter::new(&config, 2).unwrap();

        // Test with simple sinusoidal input
        for i in 0..100 {
            let t = i as f32 / 100.0;
            let clean_signal = (2.0 * PI * 10.0 * t).sin();
            let noise = 0.1 * (2.0 * PI * 100.0 * t).sin();
            let noisy_input = vec![clean_signal + noise, clean_signal + noise * 0.8];

            let output = filter.filter(&noisy_input).unwrap();
            assert_eq!(output.len(), 2);

            // After some adaptation, output should be closer to clean signal
            if i > 50 {
                assert!((output[0] - clean_signal).abs() < (noisy_input[0] - clean_signal).abs());
            }
        }
    }

    #[test]
    fn test_spectral_subtraction() {
        let config = NoiseReductionConfig::default();
        let mut subtractor = SpectralSubtraction::new(&config).unwrap();

        let input = vec![0.1, 0.05, 0.02, 0.15]; // Mix of signal and noise levels
        let output = subtractor.process(&input).unwrap();

        assert_eq!(output.len(), input.len());
        // Small values should be attenuated more than large values
        assert!(output[2].abs() < input[2].abs()); // Small value more attenuated
    }

    #[test]
    fn test_wiener_filter() {
        let config = NoiseReductionConfig::default();
        let mut filter = WienerFilter::new(&config, 1).unwrap();

        // Test with noisy signal
        for i in 0..50 {
            let clean = 0.5;
            let noise = 0.1 * (i as f32 / 10.0).sin();
            let input = vec![clean + noise];

            let output = filter.filter(&input).unwrap();
            assert_eq!(output.len(), 1);

            // Output should gradually approach clean signal
            if i > 20 {
                assert!((output[0] - clean).abs() < (input[0] - clean).abs());
            }
        }
    }

    #[test]
    fn test_noise_reducer_integration() {
        let pipeline_config = crate::processing::pipeline::PipelineConfig::default();
        let mut reducer = NoiseReducer::new(&pipeline_config).unwrap();

        let input = vec![0.8, 0.6, 0.9, 0.7]; // 4-channel input
        let output = reducer.reduce_noise(&input).unwrap();

        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_algorithm_switching() {
        let mut config = NoiseReductionConfig::default();
        config.algorithm = NoiseReductionAlgorithm::Wiener;

        let mut reducer = NoiseReducer::with_config(config).unwrap();

        // Test Wiener filter
        let input = vec![0.5, 0.3];
        let output1 = reducer.reduce_noise(&input).unwrap();

        // Switch to adaptive filter
        reducer.set_algorithm(NoiseReductionAlgorithm::Adaptive).unwrap();
        let output2 = reducer.reduce_noise(&input).unwrap();

        // Outputs should be different (different algorithms)
        assert_ne!(output1, output2);
    }
}