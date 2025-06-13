
use crate::error::EmgError;

/// Wavelet coefficients and energy distribution
#[derive(Debug, Clone)]
pub struct WaveletFeatures {
    pub coefficients: Vec<Vec<f32>>,      // Wavelet coefficients per channel
    pub energy_distribution: Vec<Vec<f32>>, // Energy in each frequency band
    pub detail_coefficients: Vec<Vec<Vec<f32>>>, // Detail coefficients per level
    pub approximation_coefficients: Vec<Vec<f32>>, // Final approximation coefficients
}

impl WaveletFeatures {
    pub fn new(channel_count: usize, levels: usize, coeff_per_level: usize) -> Self {
        Self {
            coefficients: vec![vec![0.0; levels * coeff_per_level]; channel_count],
            energy_distribution: vec![vec![0.0; levels + 1]; channel_count], // +1 for approximation
            detail_coefficients: vec![vec![vec![0.0; coeff_per_level]; levels]; channel_count],
            approximation_coefficients: vec![vec![0.0; coeff_per_level]; channel_count],
        }
    }
}

/// Wavelet transform extractor using Daubechies wavelets
pub struct WaveletExtractor {
    levels: usize,
    wavelet_type: WaveletType,
    filter_coeffs: WaveletFilters,
}

#[derive(Debug, Clone)]
pub enum WaveletType {
    Daubechies4,  // db4 - good for EMG signals
    Haar,         // Simplest wavelet
    Biorthogonal, // Good for signal reconstruction
}

#[derive(Debug, Clone)]
struct WaveletFilters {
    low_pass: Vec<f32>,   // Scaling function coefficients
    high_pass: Vec<f32>,  // Wavelet function coefficients
}

impl WaveletExtractor {
    pub fn new(_config: &super::FeatureConfig) -> Result<Self, EmgError> {
        let wavelet_type = WaveletType::Daubechies4;
        let filter_coeffs = Self::create_wavelet_filters(&wavelet_type)?;

        Ok(Self {
            levels: 5, // 5 levels of decomposition
            wavelet_type,
            filter_coeffs,
        })
    }

    /// Extract wavelet features from a window of samples
    pub fn extract(&mut self, window: &[Vec<f32>]) -> Result<WaveletFeatures, EmgError> {
        if window.is_empty() {
            return Err(EmgError::Processing("Empty window provided".to_string()));
        }

        let channel_count = window[0].len();
        let window_size = window.len();
        let mut features = WaveletFeatures::new(channel_count, self.levels, window_size / (2_usize.pow(self.levels as u32)));

        for ch in 0..channel_count {
            // Extract channel data
            let channel_data: Vec<f32> = window.iter()
                .map(|sample| sample[ch])
                .collect();

            // Perform wavelet decomposition
            let (detail_coeffs, approx_coeffs) = self.wavelet_decomposition(&channel_data)?;

            // Store coefficients
            features.detail_coefficients[ch] = detail_coeffs.clone();
            features.approximation_coefficients[ch] = approx_coeffs.clone();

            // Calculate energy distribution
            features.energy_distribution[ch] = self.calculate_energy_distribution(&detail_coeffs, &approx_coeffs);

            // Flatten coefficients for feature vector
            features.coefficients[ch] = self.flatten_coefficients(&detail_coeffs, &approx_coeffs);
        }

        Ok(features)
    }

    pub fn reset(&mut self) {
        // No state to reset
    }

    // Private methods
    fn create_wavelet_filters(wavelet_type: &WaveletType) -> Result<WaveletFilters, EmgError> {
        match wavelet_type {
            WaveletType::Daubechies4 => {
                // Daubechies 4-tap filter coefficients
                let sqrt3 = 3.0f32.sqrt();
                let denom = 4.0 * 2.0f32.sqrt();

                let low_pass = vec![
                    (1.0 + sqrt3) / denom,
                    (3.0 + sqrt3) / denom,
                    (3.0 - sqrt3) / denom,
                    (1.0 - sqrt3) / denom,
                ];

                let high_pass = vec![
                    low_pass[3],
                    -low_pass[2],
                    low_pass[1],
                    -low_pass[0],
                ];

                Ok(WaveletFilters { low_pass, high_pass })
            }

            WaveletType::Haar => {
                let sqrt2_inv = 1.0 / 2.0f32.sqrt();
                Ok(WaveletFilters {
                    low_pass: vec![sqrt2_inv, sqrt2_inv],
                    high_pass: vec![sqrt2_inv, -sqrt2_inv],
                })
            }

            WaveletType::Biorthogonal => {
                // Biorthogonal 2.2 wavelet coefficients
                Ok(WaveletFilters {
                    low_pass: vec![-0.1767767, 0.3535534, 1.0606602, 0.3535534, -0.1767767],
                    high_pass: vec![0.0, 0.3535534, -0.7071068, 0.3535534, 0.0],
                })
            }
        }
    }

    fn wavelet_decomposition(&self, signal: &[f32]) -> Result<(Vec<Vec<f32>>, Vec<f32>), EmgError> {
        let mut current_signal = signal.to_vec();
        let mut detail_coefficients = Vec::new();

        // Perform multi-level decomposition
        for level in 0..self.levels {
            if current_signal.len() < 2 {
                break;
            }

            let (details, approximation) = self.single_level_decomposition(&current_signal)?;
            detail_coefficients.push(details);
            current_signal = approximation;
        }

        Ok((detail_coefficients, current_signal))
    }

    fn single_level_decomposition(&self, signal: &[f32]) -> Result<(Vec<f32>, Vec<f32>), EmgError> {
        let n = signal.len();
        let filter_len = self.filter_coeffs.low_pass.len();

        let mut details = Vec::new();
        let mut approximation = Vec::new();

        // Convolution with downsampling by 2
        for i in (0..n).step_by(2) {
            let mut low_sum = 0.0;
            let mut high_sum = 0.0;

            for (j, (&low_coeff, &high_coeff)) in self.filter_coeffs.low_pass.iter()
                .zip(self.filter_coeffs.high_pass.iter()).enumerate() {

                let signal_idx = (i + j) % n; // Circular boundary condition
                low_sum += signal[signal_idx] * low_coeff;
                high_sum += signal[signal_idx] * high_coeff;
            }

            approximation.push(low_sum);
            details.push(high_sum);
        }

        Ok((details, approximation))
    }

    fn calculate_energy_distribution(&self, detail_coeffs: &[Vec<f32>], approx_coeffs: &[f32]) -> Vec<f32> {
        let mut energy_dist = Vec::new();

        // Energy in each detail level
        for details in detail_coeffs {
            let energy: f32 = details.iter().map(|&x| x * x).sum();
            energy_dist.push(energy);
        }

        // Energy in approximation coefficients
        let approx_energy: f32 = approx_coeffs.iter().map(|&x| x * x).sum();
        energy_dist.push(approx_energy);

        // Normalize by total energy
        let total_energy: f32 = energy_dist.iter().sum();
        if total_energy > 0.0 {
            energy_dist.iter_mut().for_each(|e| *e /= total_energy);
        }

        energy_dist
    }

    fn flatten_coefficients(&self, detail_coeffs: &[Vec<f32>], approx_coeffs: &[f32]) -> Vec<f32> {
        let mut flattened = Vec::new();

        // Add detail coefficients (taking only a subset to limit feature count)
        for details in detail_coeffs {
            let max_coeffs = 16.min(details.len()); // Limit to 16 coefficients per level
            flattened.extend_from_slice(&details[..max_coeffs]);
        }

        // Add approximation coefficients
        let max_approx = 16.min(approx_coeffs.len());
        flattened.extend_from_slice(&approx_coeffs[..max_approx]);

        flattened
    }
}


