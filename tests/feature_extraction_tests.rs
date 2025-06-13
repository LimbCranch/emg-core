
// ================================================================================
// Integration tests for feature extraction
// File: tests/feature_extraction_tests.rs (NEW)
// ================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use emg_core::processing::features::{FeatureExtractor, FeatureConfig};
    use emg_core::hal::types::EmgSample;
    use std::f32::consts::PI;
    use emg_core::processing::FeatureVector;

    #[test]
    fn test_time_domain_features() {
        let config = FeatureConfig {
            window_size: 128,
            enable_time_domain: true,
            enable_frequency_domain: false,
            enable_wavelets: false,
            ..Default::default()
        };

        let mut extractor = FeatureExtractor::new(&config).unwrap();

        // Generate test signal: 50Hz sine wave with noise
        let window: Vec<Vec<f32>> = (0..128)
            .map(|i| {
                let t = i as f32 / 2000.0; // 2kHz sampling
                let signal = (2.0 * PI * 50.0 * t).sin() + 0.1 * (2.0 * PI * 200.0 * t).sin();
                vec![signal, signal * 0.8] // 2 channels
            })
            .collect();

        let features = extractor.extract_features_from_window().unwrap();

        // Verify basic properties
        assert_eq!(features.rms.len(), 2);
        assert_eq!(features.mean_absolute_value.len(), 2);
        assert!(features.rms[0] > 0.0);
        assert!(features.mean_absolute_value[0] > 0.0);
        assert!(features.zero_crossings[0] > 0);
    }

    #[test]
    fn test_frequency_domain_features() {
        let config = FeatureConfig {
            window_size: 256,
            enable_time_domain: false,
            enable_frequency_domain: true,
            enable_wavelets: false,
            sample_rate_hz: 2000,
            ..Default::default()
        };

        let mut extractor = FeatureExtractor::new(&config).unwrap();

        // Generate 100Hz sine wave
        let window: Vec<Vec<f32>> = (0..256)
            .map(|i| {
                let t = i as f32 / 2000.0;
                let signal = (2.0 * PI * 100.0 * t).sin();
                vec![signal] // 1 channel
            })
            .collect();

        let features = extractor.extract_features_from_window().unwrap();

        // Check that spectral centroid is around 100 Hz
        assert!(features.spectral_centroid[0] > 80.0);
        assert!(features.spectral_centroid[0] < 120.0);
        assert!(features.power_spectral_density[0].len() > 0);
    }

    #[test]
    fn test_streaming_feature_extraction() {
        let config = FeatureConfig {
            window_size: 64,
            overlap_ratio: 0.5,
            ..Default::default()
        };

        let mut extractor = FeatureExtractor::new(&config).unwrap();

        // Feed samples one by one
        for i in 0..100 {
            let t = i as f32 / 2000.0;
            let sample = vec![(2.0 * PI * 50.0 * t).sin()];

            let features = extractor.extract_features(&sample).unwrap();

            // After enough samples, we should get meaningful features
            if i > 64 {
                assert!(features.rms[0] >= 0.0);
            }
        }
    }

    #[test]
    fn test_feature_vector_flattening() {
        let mut feature_vector = FeatureVector::default_for_channels(2);
        feature_vector.rms = vec![0.5, 0.6];
        feature_vector.mean_absolute_value = vec![0.3, 0.4];

        let flat = feature_vector.to_flat_vector();
        let names = feature_vector.feature_names();

        assert!(flat.len() > 0);
        assert_eq!(flat.len(), names.len());
        assert_eq!(flat[0], 0.5); // First RMS value
        assert_eq!(flat[1], 0.6); // Second RMS value
    }

    #[test]
    fn test_performance_requirements() {
        use std::time::Instant;

        let config = FeatureConfig::default();
        let mut extractor = FeatureExtractor::new(&config).unwrap();

        // Generate realistic EMG data
        let window: Vec<Vec<f32>> = (0..256)
            .map(|i| {
                let t = i as f32 / 2000.0;
                // Simulate EMG: multiple frequency components + noise
                let emg = (2.0 * PI * 80.0 * t).sin() * 0.5 +
                    (2.0 * PI * 150.0 * t).sin() * 0.3 +
                    (2.0 * PI * 200.0 * t).sin() * 0.2 +
                    0.1 * (rand::random::<f32>() - 0.5);
                vec![emg; 8] // 8 channels
            })
            .collect();

        let start = Instant::now();
        let _features = extractor.extract_features_from_window().unwrap();
        let duration = start.elapsed();

        // Should complete in less than 2ms
        assert!(duration.as_micros() < 2000,
                "Feature extraction took {}μs, target is <2000μs",
                duration.as_micros());
    }
}