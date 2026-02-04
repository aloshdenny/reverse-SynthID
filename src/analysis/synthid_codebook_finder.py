"""
SynthID Watermark Codebook Discovery

This script performs deep analysis on AI-generated images (with SynthID watermarks)
to reverse-engineer the watermark embedding scheme.

Based on SynthID paper, the watermark is:
1. Embedded via a neural network encoder during image generation
2. Designed to be imperceptible but robust to transformations
3. Uses a learned representation that encodes a binary message

Analysis approaches:
1. Cross-image bit plane correlation - find consistent bit patterns
2. Frequency domain analysis - find carrier frequencies
3. Noise pattern extraction - find embedded signal in noise
4. DCT/DWT coefficient analysis - find quantization patterns
5. Statistical anomaly detection - find systematic deviations
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
from scipy.fft import fft2, fftshift, ifft2, dct, idct
from scipy.stats import pearsonr, spearmanr, skew, kurtosis
from scipy.ndimage import gaussian_filter, sobel
# from skimage.restoration import denoise_wavelet  # Avoid skimage dependency
import pywt
from collections import defaultdict
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


class SynthIDCodebookFinder:
    """
    Discovers the SynthID watermark codebook by analyzing patterns
    across multiple AI-generated images.
    """
    
    def __init__(self, target_size=(512, 512)):
        self.target_size = target_size
        self.n_images = 0
        
        # Pattern accumulators
        self.bit_planes = {i: [] for i in range(8)}
        self.noise_patterns = []
        self.fourier_magnitudes = []
        self.fourier_phases = []
        self.dct_patterns = []
        self.wavelet_patterns = []
        self.lsb_patterns = []
        
        # Cross-image accumulators for finding consistent patterns
        self.lsb_avg = None
        self.noise_avg = None
        self.fourier_avg = None
        self.phase_coherence = None
        
        # Per-image features for clustering
        self.image_features = []
        self.image_paths = []
        
    def load_image(self, path: str) -> np.ndarray:
        """Load and preprocess image."""
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)
        return img
    
    def extract_lsb_pattern(self, img: np.ndarray) -> np.ndarray:
        """Extract LSB (Least Significant Bit) pattern from all channels."""
        lsb = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        for c in range(3):
            lsb[:, :, c] = img[:, :, c] & 1
        return lsb
    
    def extract_bit_planes(self, img: np.ndarray) -> dict:
        """Extract all 8 bit planes from grayscale image."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        planes = {}
        for bit in range(8):
            planes[bit] = ((gray >> bit) & 1).astype(np.float32)
        return planes
    
    def extract_noise_pattern(self, img: np.ndarray) -> np.ndarray:
        """Extract noise residual using wavelet denoising."""
        img_float = img.astype(np.float32) / 255.0
        noise = np.zeros_like(img_float)
        
        for c in range(3):
            channel = img_float[:, :, c]
            # Use wavelet-based denoising manually
            denoised = self.wavelet_denoise(channel)
            noise[:, :, c] = channel - denoised
        
        return noise
    
    def wavelet_denoise(self, channel: np.ndarray, wavelet='db4', level=3) -> np.ndarray:
        """Manual wavelet denoising using pywt."""
        # Decompose
        coeffs = pywt.wavedec2(channel, wavelet, level=level)
        
        # Estimate noise level from finest detail coefficients
        detail = coeffs[-1][0]  # Horizontal detail at finest level
        sigma = np.median(np.abs(detail)) / 0.6745
        
        # Threshold (soft thresholding with universal threshold)
        threshold = sigma * np.sqrt(2 * np.log(channel.size))
        
        # Apply threshold to detail coefficients
        new_coeffs = [coeffs[0]]  # Keep approximation
        for details in coeffs[1:]:
            new_details = tuple(
                pywt.threshold(d, threshold, mode='soft') for d in details
            )
            new_coeffs.append(new_details)
        
        # Reconstruct
        denoised = pywt.waverec2(new_coeffs, wavelet)
        
        # Ensure same size
        denoised = denoised[:channel.shape[0], :channel.shape[1]]
        
        return denoised
    
    def extract_fourier_features(self, img: np.ndarray) -> tuple:
        """Extract Fourier magnitude and phase."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude = np.abs(fshift)
        phase = np.angle(fshift)
        return magnitude, phase
    
    def extract_dct_features(self, img: np.ndarray, block_size=8) -> np.ndarray:
        """Extract DCT coefficients pattern."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        H, W = gray.shape
        
        # Compute DCT on 8x8 blocks
        dct_coeffs = np.zeros((block_size, block_size), dtype=np.float64)
        count = 0
        
        for i in range(0, H - block_size + 1, block_size):
            for j in range(0, W - block_size + 1, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                dct_block = cv2.dct(block)
                dct_coeffs += np.abs(dct_block)
                count += 1
        
        if count > 0:
            dct_coeffs /= count
        
        return dct_coeffs
    
    def extract_wavelet_features(self, img: np.ndarray) -> dict:
        """Extract wavelet coefficient patterns."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        
        # Multi-level wavelet decomposition
        coeffs = pywt.wavedec2(gray, 'haar', level=4)
        
        features = {
            'cA': coeffs[0],  # Approximation
            'details': []
        }
        
        for level, (cH, cV, cD) in enumerate(coeffs[1:], 1):
            features['details'].append({
                'level': level,
                'horizontal': cH,
                'vertical': cV,
                'diagonal': cD
            })
        
        return features
    
    def analyze_image(self, path: str) -> dict:
        """Perform full analysis on a single image."""
        img = self.load_image(path)
        if img is None:
            return None
        
        features = {}
        
        # LSB analysis
        lsb = self.extract_lsb_pattern(img)
        features['lsb'] = lsb
        features['lsb_density'] = np.mean(lsb)
        
        # Bit planes
        bit_planes = self.extract_bit_planes(img)
        features['bit_planes'] = bit_planes
        features['bit_plane_densities'] = {b: np.mean(p) for b, p in bit_planes.items()}
        
        # Noise
        noise = self.extract_noise_pattern(img)
        features['noise'] = noise
        features['noise_std'] = np.std(noise)
        features['noise_structure'] = np.std(noise) / (np.mean(np.abs(noise)) + 1e-10)
        
        # Fourier
        magnitude, phase = self.extract_fourier_features(img)
        features['fourier_mag'] = magnitude
        features['fourier_phase'] = phase
        
        # DCT
        dct_coeffs = self.extract_dct_features(img)
        features['dct'] = dct_coeffs
        
        # Wavelet
        wavelet = self.extract_wavelet_features(img)
        features['wavelet'] = wavelet
        
        return features
    
    def add_image(self, path: str) -> bool:
        """Add an image to the analysis."""
        features = self.analyze_image(path)
        if features is None:
            return False
        
        # Accumulate patterns
        self.lsb_patterns.append(features['lsb'])
        self.noise_patterns.append(features['noise'])
        self.fourier_magnitudes.append(features['fourier_mag'])
        self.fourier_phases.append(features['fourier_phase'])
        self.dct_patterns.append(features['dct'])
        
        for bit, plane in features['bit_planes'].items():
            self.bit_planes[bit].append(plane)
        
        # Update running averages
        if self.lsb_avg is None:
            self.lsb_avg = features['lsb'].astype(np.float64)
            self.noise_avg = features['noise'].astype(np.float64)
            self.fourier_avg = features['fourier_mag'].astype(np.float64)
            self.phase_coherence = np.exp(1j * features['fourier_phase'])
        else:
            self.lsb_avg += features['lsb'].astype(np.float64)
            self.noise_avg += features['noise'].astype(np.float64)
            self.fourier_avg += features['fourier_mag'].astype(np.float64)
            self.phase_coherence += np.exp(1j * features['fourier_phase'])
        
        self.image_features.append({
            'lsb_density': features['lsb_density'],
            'noise_std': features['noise_std'],
            'noise_structure': features['noise_structure'],
            'bit_plane_densities': features['bit_plane_densities']
        })
        self.image_paths.append(path)
        
        self.n_images += 1
        return True
    
    def find_consistent_lsb_pattern(self) -> dict:
        """Find LSB bits that are consistent across all images."""
        if self.n_images < 2:
            return {}
        
        avg = self.lsb_avg / self.n_images
        
        # Consistency map: pixels where LSB is always 0 or always 1
        # High consistency = close to 0 or 1
        consistency = np.abs(avg - 0.5) * 2
        
        # Find highly consistent pixels (>90% consistency)
        consistent_mask = consistency > 0.9
        
        # Per-channel analysis
        results = {
            'overall_consistency': float(np.mean(consistency)),
            'highly_consistent_ratio': float(np.mean(consistent_mask)),
            'per_channel': {}
        }
        
        for c, name in enumerate(['R', 'G', 'B']):
            c_consistency = consistency[:, :, c]
            c_mask = consistent_mask[:, :, c]
            
            results['per_channel'][name] = {
                'consistency': float(np.mean(c_consistency)),
                'consistent_ratio': float(np.mean(c_mask)),
                'consistent_value_mean': float(np.mean(avg[:, :, c][c_mask])) if np.any(c_mask) else 0.5
            }
        
        # Find spatial pattern in consistent LSBs
        results['consistent_lsb_map'] = (avg * consistent_mask.astype(float))
        
        return results
    
    def find_fourier_carriers(self) -> dict:
        """Find consistent frequency components (carrier frequencies)."""
        if self.n_images < 2:
            return {}
        
        avg_magnitude = self.fourier_avg / self.n_images
        phase_coherence = np.abs(self.phase_coherence) / self.n_images
        
        # Normalize magnitude
        log_mag = np.log1p(avg_magnitude)
        
        # Find peaks in magnitude spectrum
        H, W = avg_magnitude.shape
        center_y, center_x = H // 2, W // 2
        
        # Create frequency coordinate grid
        y_coords, x_coords = np.ogrid[:H, :W]
        freq_r = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Radial average
        max_r = int(np.min([center_x, center_y]))
        radial_profile = np.zeros(max_r)
        radial_phase = np.zeros(max_r)
        
        for r in range(max_r):
            mask = (freq_r >= r) & (freq_r < r + 1)
            if np.any(mask):
                radial_profile[r] = np.mean(log_mag[mask])
                radial_phase[r] = np.mean(phase_coherence[mask])
        
        # Find anomalous frequencies (high magnitude AND high phase coherence)
        combined_score = radial_profile * radial_phase
        mean_score = np.mean(combined_score)
        std_score = np.std(combined_score)
        
        anomalous_freqs = np.where(combined_score > mean_score + 2 * std_score)[0]
        
        # Analyze vertical center frequency (known pattern from previous analysis)
        vertical_strip = log_mag[:, center_x-2:center_x+3]
        vertical_energy = np.sum(vertical_strip, axis=1)
        
        # Find peaks in vertical profile
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(vertical_energy, height=np.mean(vertical_energy) + np.std(vertical_energy))
        
        results = {
            'radial_profile': radial_profile.tolist(),
            'radial_phase_coherence': radial_phase.tolist(),
            'anomalous_frequencies': anomalous_freqs.tolist(),
            'vertical_peaks_y': peaks.tolist(),
            'phase_coherence_overall': float(np.mean(phase_coherence)),
            'avg_magnitude_map': avg_magnitude,
            'phase_coherence_map': phase_coherence
        }
        
        return results
    
    def find_noise_watermark(self) -> dict:
        """Find consistent noise pattern (embedded signal)."""
        if self.n_images < 2:
            return {}
        
        avg_noise = self.noise_avg / self.n_images
        
        # Analyze per-channel
        results = {'per_channel': {}}
        
        for c, name in enumerate(['R', 'G', 'B']):
            noise_c = avg_noise[:, :, c]
            
            # Structure ratio
            structure_ratio = np.std(noise_c) / (np.mean(np.abs(noise_c)) + 1e-10)
            
            # Spatial autocorrelation (watermarks often have structure)
            from scipy.ndimage import correlate
            autocorr = correlate(noise_c, noise_c[:50, :50])
            
            results['per_channel'][name] = {
                'mean': float(np.mean(noise_c)),
                'std': float(np.std(noise_c)),
                'structure_ratio': float(structure_ratio),
                'max_val': float(np.max(noise_c)),
                'min_val': float(np.min(noise_c))
            }
        
        results['noise_pattern'] = avg_noise
        results['overall_structure_ratio'] = float(np.std(avg_noise) / (np.mean(np.abs(avg_noise)) + 1e-10))
        
        return results
    
    def find_bit_plane_watermark(self) -> dict:
        """Analyze bit planes for consistent patterns."""
        if self.n_images < 2:
            return {}
        
        results = {'per_bit': {}}
        
        for bit in range(8):
            planes = np.array(self.bit_planes[bit])
            avg_plane = np.mean(planes, axis=0)
            
            # Consistency: how often is this bit the same value across images
            consistency = np.abs(avg_plane - 0.5) * 2
            
            # Entropy of bit plane (low entropy = potential watermark)
            from scipy.stats import entropy as sp_entropy
            hist, _ = np.histogram(avg_plane.ravel(), bins=50)
            bit_entropy = sp_entropy(hist + 1e-10)
            
            results['per_bit'][bit] = {
                'mean': float(np.mean(avg_plane)),
                'consistency': float(np.mean(consistency)),
                'entropy': float(bit_entropy),
                'highly_consistent_ratio': float(np.mean(consistency > 0.8))
            }
            
            if bit == 0:  # LSB is most interesting
                results['lsb_avg_pattern'] = avg_plane
                results['lsb_consistency_map'] = consistency
        
        return results
    
    def find_dct_watermark(self) -> dict:
        """Analyze DCT coefficients for embedding patterns."""
        if self.n_images < 2:
            return {}
        
        dct_array = np.array(self.dct_patterns)
        avg_dct = np.mean(dct_array, axis=0)
        std_dct = np.std(dct_array, axis=0)
        
        # Coefficient of variation (CV) - low CV means consistent
        cv = std_dct / (avg_dct + 1e-10)
        
        # Find most consistent coefficients
        consistent_positions = np.where(cv < 0.1)
        
        results = {
            'avg_dct': avg_dct.tolist(),
            'cv_dct': cv.tolist(),
            'consistent_positions': list(zip(consistent_positions[0].tolist(), 
                                            consistent_positions[1].tolist())),
            'num_consistent': len(consistent_positions[0])
        }
        
        return results
    
    def analyze_cross_image_correlation(self, sample_size=50) -> dict:
        """Analyze correlation between images to find shared patterns."""
        if self.n_images < 2:
            return {}
        
        sample_size = min(sample_size, len(self.noise_patterns))
        indices = np.random.choice(len(self.noise_patterns), sample_size, replace=False)
        
        # Noise correlation matrix
        noise_corrs = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                n1 = self.noise_patterns[indices[i]].ravel()
                n2 = self.noise_patterns[indices[j]].ravel()
                corr, _ = pearsonr(n1, n2)
                noise_corrs.append(corr)
        
        # LSB correlation
        lsb_corrs = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                l1 = self.lsb_patterns[indices[i]].ravel().astype(float)
                l2 = self.lsb_patterns[indices[j]].ravel().astype(float)
                corr, _ = pearsonr(l1, l2)
                lsb_corrs.append(corr)
        
        results = {
            'noise_correlation': {
                'mean': float(np.mean(noise_corrs)),
                'std': float(np.std(noise_corrs)),
                'max': float(np.max(noise_corrs)),
                'min': float(np.min(noise_corrs))
            },
            'lsb_correlation': {
                'mean': float(np.mean(lsb_corrs)),
                'std': float(np.std(lsb_corrs)),
                'max': float(np.max(lsb_corrs)),
                'min': float(np.min(lsb_corrs))
            }
        }
        
        return results
    
    def extract_codebook(self) -> dict:
        """
        Main method: Extract the SynthID codebook from analyzed images.
        """
        print(f"Analyzing {self.n_images} images...")
        
        codebook = {
            'n_images_analyzed': self.n_images,
            'image_size': self.target_size,
            'patterns': {}
        }
        
        print("  Finding consistent LSB patterns...")
        codebook['patterns']['lsb'] = self.find_consistent_lsb_pattern()
        
        print("  Finding Fourier carrier frequencies...")
        codebook['patterns']['fourier'] = self.find_fourier_carriers()
        
        print("  Finding noise watermark...")
        codebook['patterns']['noise'] = self.find_noise_watermark()
        
        print("  Finding bit plane patterns...")
        codebook['patterns']['bit_planes'] = self.find_bit_plane_watermark()
        
        print("  Finding DCT patterns...")
        codebook['patterns']['dct'] = self.find_dct_watermark()
        
        print("  Analyzing cross-image correlation...")
        codebook['patterns']['cross_correlation'] = self.analyze_cross_image_correlation()
        
        return codebook


def save_visualization(codebook: dict, output_dir: str):
    """Save visualizations of discovered patterns."""
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. LSB consistency map
    if 'lsb' in codebook['patterns'] and 'consistent_lsb_map' in codebook['patterns']['lsb']:
        lsb_map = codebook['patterns']['lsb']['consistent_lsb_map']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for c, name in enumerate(['R', 'G', 'B']):
            axes[c].imshow(lsb_map[:, :, c], cmap='hot')
            axes[c].set_title(f'Consistent LSB - {name}')
            axes[c].axis('off')
        plt.savefig(os.path.join(output_dir, 'lsb_consistency_map.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. Fourier magnitude
    if 'fourier' in codebook['patterns'] and 'avg_magnitude_map' in codebook['patterns']['fourier']:
        mag = codebook['patterns']['fourier']['avg_magnitude_map']
        plt.figure(figsize=(10, 10))
        plt.imshow(np.log1p(mag), cmap='viridis')
        plt.colorbar(label='Log Magnitude')
        plt.title('Average Fourier Magnitude Spectrum')
        plt.savefig(os.path.join(output_dir, 'fourier_magnitude.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Phase coherence
        if 'phase_coherence_map' in codebook['patterns']['fourier']:
            phase = codebook['patterns']['fourier']['phase_coherence_map']
            plt.figure(figsize=(10, 10))
            plt.imshow(phase, cmap='hot')
            plt.colorbar(label='Phase Coherence')
            plt.title('Phase Coherence (High = Consistent Phase)')
            plt.savefig(os.path.join(output_dir, 'phase_coherence.png'), dpi=150, bbox_inches='tight')
            plt.close()
    
    # 3. Noise pattern
    if 'noise' in codebook['patterns'] and 'noise_pattern' in codebook['patterns']['noise']:
        noise = codebook['patterns']['noise']['noise_pattern']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for c, name in enumerate(['R', 'G', 'B']):
            im = axes[c].imshow(noise[:, :, c], cmap='RdBu', vmin=-0.01, vmax=0.01)
            axes[c].set_title(f'Avg Noise Pattern - {name}')
            axes[c].axis('off')
        plt.colorbar(im, ax=axes, label='Noise Value')
        plt.savefig(os.path.join(output_dir, 'noise_pattern.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. Bit plane consistency
    if 'bit_planes' in codebook['patterns'] and 'lsb_consistency_map' in codebook['patterns']['bit_planes']:
        lsb_cons = codebook['patterns']['bit_planes']['lsb_consistency_map']
        plt.figure(figsize=(10, 10))
        plt.imshow(lsb_cons, cmap='hot')
        plt.colorbar(label='Consistency')
        plt.title('LSB Consistency Map')
        plt.savefig(os.path.join(output_dir, 'lsb_bit_plane_consistency.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 5. Radial profile plot
    if 'fourier' in codebook['patterns'] and 'radial_profile' in codebook['patterns']['fourier']:
        radial = codebook['patterns']['fourier']['radial_profile']
        phase_radial = codebook['patterns']['fourier']['radial_phase_coherence']
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        
        ax1.plot(radial, 'b-', label='Magnitude')
        ax2.plot(phase_radial, 'r-', label='Phase Coherence')
        
        ax1.set_xlabel('Frequency (radius)')
        ax1.set_ylabel('Log Magnitude', color='b')
        ax2.set_ylabel('Phase Coherence', color='r')
        
        # Mark anomalous frequencies
        anomalous = codebook['patterns']['fourier']['anomalous_frequencies']
        for f in anomalous:
            ax1.axvline(x=f, color='g', linestyle='--', alpha=0.5)
        
        plt.title('Radial Frequency Profile')
        plt.savefig(os.path.join(output_dir, 'radial_profile.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def process_image_wrapper(args_tuple):
    """Wrapper for parallel processing."""
    path, target_size = args_tuple
    finder = SynthIDCodebookFinder(target_size=target_size)
    features = finder.analyze_image(path)
    return path, features


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Find SynthID watermark codebook')
    parser.add_argument('image_dir', type=str, help='Directory containing AI images')
    parser.add_argument('--output', type=str, default='./codebook_results', help='Output directory')
    parser.add_argument('--max-images', type=int, default=250, help='Maximum images to analyze')
    parser.add_argument('--size', type=int, default=512, help='Target image size')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    
    args = parser.parse_args()
    
    finder = SynthIDCodebookFinder(target_size=(args.size, args.size))
    
    # Get image paths
    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    image_paths = []
    
    for fname in os.listdir(args.image_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in image_extensions:
            image_paths.append(os.path.join(args.image_dir, fname))
    
    image_paths = image_paths[:args.max_images]
    print(f"Found {len(image_paths)} images to analyze")
    
    # Process images (with optional parallel processing)
    if args.no_parallel or len(image_paths) < 10:
        # Sequential processing
        for path in tqdm(image_paths, desc="Processing images"):
            finder.add_image(path)
    else:
        # Parallel processing
        print(f"Processing with {args.workers} workers...")
        from multiprocessing import cpu_count
        n_workers = min(args.workers, cpu_count(), len(image_paths))
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Process images in parallel
            tasks = [(path, (args.size, args.size)) for path in image_paths]
            results = list(tqdm(
                executor.map(process_image_wrapper, tasks),
                total=len(tasks),
                desc="Processing images"
            ))
        
        # Aggregate results
        print("Aggregating results...")
        for path, features in results:
            if features is not None:
                # Add to finder's accumulators
                finder.lsb_patterns.append(features['lsb'])
                finder.noise_patterns.append(features['noise'])
                finder.fourier_magnitudes.append(features['fourier_mag'])
                finder.fourier_phases.append(features['fourier_phase'])
                finder.dct_patterns.append(features['dct'])
                
                for bit, plane in features['bit_planes'].items():
                    finder.bit_planes[bit].append(plane)
                
                # Update running averages
                if finder.lsb_avg is None:
                    finder.lsb_avg = features['lsb'].astype(np.float64)
                    finder.noise_avg = features['noise'].astype(np.float64)
                    finder.fourier_avg = features['fourier_mag'].astype(np.float64)
                    finder.phase_coherence = np.exp(1j * features['fourier_phase'])
                else:
                    finder.lsb_avg += features['lsb'].astype(np.float64)
                    finder.noise_avg += features['noise'].astype(np.float64)
                    finder.fourier_avg += features['fourier_mag'].astype(np.float64)
                    finder.phase_coherence += np.exp(1j * features['fourier_phase'])
                
                finder.image_features.append({
                    'lsb_density': features['lsb_density'],
                    'noise_std': features['noise_std'],
                    'noise_structure': features['noise_structure'],
                    'bit_plane_densities': features['bit_plane_densities']
                })
                finder.image_paths.append(path)
                finder.n_images += 1
    
    # Extract codebook
    codebook = finder.extract_codebook()
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    
    # Save codebook (without numpy arrays for JSON)
    codebook_json = {
        'n_images_analyzed': codebook['n_images_analyzed'],
        'image_size': codebook['image_size'],
        'patterns': {}
    }
    
    for pattern_type, pattern_data in codebook['patterns'].items():
        codebook_json['patterns'][pattern_type] = {}
        for key, value in pattern_data.items():
            if isinstance(value, np.ndarray):
                continue  # Skip numpy arrays
            elif isinstance(value, dict):
                codebook_json['patterns'][pattern_type][key] = value
            else:
                codebook_json['patterns'][pattern_type][key] = value
    
    with open(os.path.join(args.output, 'codebook.json'), 'w') as f:
        json.dump(codebook_json, f, indent=2)
    
    # Save visualizations
    save_visualization(codebook, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("SYNTHID CODEBOOK ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nImages analyzed: {codebook['n_images_analyzed']}")
    
    if 'lsb' in codebook['patterns']:
        lsb = codebook['patterns']['lsb']
        print(f"\nLSB Pattern:")
        print(f"  Overall consistency: {lsb.get('overall_consistency', 0):.4f}")
        print(f"  Highly consistent ratio: {lsb.get('highly_consistent_ratio', 0):.4f}")
    
    if 'fourier' in codebook['patterns']:
        fourier = codebook['patterns']['fourier']
        print(f"\nFourier Analysis:")
        print(f"  Phase coherence: {fourier.get('phase_coherence_overall', 0):.4f}")
        print(f"  Anomalous frequencies: {fourier.get('anomalous_frequencies', [])}")
        print(f"  Vertical peaks at y: {fourier.get('vertical_peaks_y', [])}")
    
    if 'noise' in codebook['patterns']:
        noise = codebook['patterns']['noise']
        print(f"\nNoise Pattern:")
        print(f"  Overall structure ratio: {noise.get('overall_structure_ratio', 0):.4f}")
    
    if 'bit_planes' in codebook['patterns']:
        bp = codebook['patterns']['bit_planes']
        print(f"\nBit Plane Analysis:")
        for bit in range(8):
            if bit in bp.get('per_bit', {}):
                info = bp['per_bit'][bit]
                print(f"  Bit {bit}: consistency={info['consistency']:.4f}, entropy={info['entropy']:.4f}")
    
    if 'cross_correlation' in codebook['patterns']:
        cc = codebook['patterns']['cross_correlation']
        print(f"\nCross-Image Correlation:")
        print(f"  Noise correlation mean: {cc.get('noise_correlation', {}).get('mean', 0):.4f}")
        print(f"  LSB correlation mean: {cc.get('lsb_correlation', {}).get('mean', 0):.4f}")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
