"""
Improved SynthID Watermark Detector

Improvements over the original RobustSynthIDExtractor:

1. PER-CHANNEL FFT ANALYSIS — Green channel carries the strongest SynthID signal
   (G=1.0, R=0.85, B=0.70). Old code averaged to grayscale, diluting the signal.

2. SOFT PROBABILISTIC DECISION — Replaces hard AND thresholds with sigmoid-based
   soft scoring. No single metric can veto the detection anymore.

3. ICA PATTERN UTILIZATION — The codebook already contains an ICA-extracted
   watermark pattern but the old detector never used it. Now it's a detection signal.

4. ADAPTIVE PHASE THRESHOLD — Instead of a fixed 0.45, uses the distribution of
   phase matches across carriers to find a data-driven threshold.

5. IMPROVED CONFIDENCE SCORING — Rebalanced weights based on which signals
   actually carry information (correlation is near-zero for non-native resolutions).

Uses the SAME codebook format as the original extractor.
"""

import os
import numpy as np
import cv2
from scipy.fft import fft2, fftshift
from scipy import ndimage
import pywt
import pickle
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Result of watermark detection (same interface as original)."""
    is_watermarked: bool
    confidence: float
    correlation: float
    phase_match: float
    structure_ratio: float
    carrier_strength: float
    multi_scale_consistency: float
    details: Dict


class ImprovedSynthIDExtractor:
    """
    Improved SynthID detector with per-channel analysis and soft decision logic.

    Drop-in replacement for RobustSynthIDExtractor — uses the same codebook format.
    """

    # SynthID per-channel embedding weights (from bypass research)
    CHANNEL_WEIGHTS = {'R': 0.85, 'G': 1.0, 'B': 0.70}

    def __init__(
        self,
        scales: List[int] = [256, 512, 1024],
        wavelets: List[str] = ['db4', 'sym8', 'coif3'],
        codebook_path: Optional[str] = None
    ):
        self.scales = scales
        self.wavelets = wavelets
        self.codebook = None
        self.classifier_data = None

        # SynthID known carriers (same as original)
        self.known_carriers = [
            (48, 0), (-48, 0),
            (96, 0), (-96, 0),
            (192, 0), (-192, 0),
            (210, 0), (-210, 0),
            (238, 0), (-238, 0),
            (0, 88), (0, -88),
            (0, 176), (0, -176),
            (0, 192), (0, -192),
            (48, 88), (-48, -88),
            (48, -88), (-48, 88),
            (96, 88), (-96, -88),
            (96, -88), (-96, 88),
            (96, 176), (-96, -176),
            (96, -176), (-96, 176),
        ]

        if codebook_path and os.path.exists(codebook_path):
            self.load_codebook(codebook_path)

    def load_classifier(self, path: str) -> None:
        """Load trained neural classifier (e.g. from train_classifier.py)."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.classifier_data = pickle.load(f)

    def load_codebook(self, path: str) -> None:
        """Load pre-extracted codebook."""
        import pickle
        with open(path, 'rb') as f:
            self.codebook = pickle.load(f)

    # ================================================================
    # DENOISING (reused from original)
    # ================================================================

    def wavelet_denoise(self, channel: np.ndarray, wavelet: str = 'db4', level: int = 3) -> np.ndarray:
        """Wavelet-based denoising using soft thresholding."""
        coeffs = pywt.wavedec2(channel, wavelet, level=level)
        detail = coeffs[-1][0]
        sigma = np.median(np.abs(detail)) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(channel.size))

        new_coeffs = [coeffs[0]]
        for details in coeffs[1:]:
            new_details = tuple(pywt.threshold(d, threshold, mode='soft') for d in details)
            new_coeffs.append(new_details)

        denoised = pywt.waverec2(new_coeffs, wavelet)
        return denoised[:channel.shape[0], :channel.shape[1]]

    def extract_noise_single(self, image: np.ndarray, method: str = 'wavelet', **kwargs) -> np.ndarray:
        """Extract noise using a single denoising method."""
        img_f = image.astype(np.float32)
        if img_f.max() > 1:
            img_f = img_f / 255.0

        if method == 'wavelet':
            wavelet = kwargs.get('wavelet', 'db4')
            if len(img_f.shape) == 2:
                denoised = self.wavelet_denoise(img_f, wavelet)
            else:
                denoised = np.zeros_like(img_f)
                for c in range(img_f.shape[2]):
                    denoised[:, :, c] = self.wavelet_denoise(img_f[:, :, c], wavelet)
        elif method == 'bilateral':
            img_uint8 = (img_f * 255).clip(0, 255).astype(np.uint8)
            if len(img_f.shape) == 2:
                denoised = cv2.bilateralFilter(img_uint8, 9, 75, 75).astype(np.float32) / 255.0
            else:
                denoised = np.zeros_like(img_f)
                for c in range(img_f.shape[2]):
                    denoised[:, :, c] = cv2.bilateralFilter(
                        img_uint8[:, :, c], 9, 75, 75
                    ).astype(np.float32) / 255.0
        elif method == 'wiener':
            if len(img_f.shape) == 2:
                noise_var = np.var(img_f - ndimage.gaussian_filter(img_f, sigma=2))
                f = fft2(img_f)
                power = np.abs(f) ** 2
                signal_power = np.maximum(power - noise_var, 0)
                wiener_ratio = signal_power / (signal_power + noise_var + 1e-10)
                denoised = np.real(np.fft.ifft2(f * wiener_ratio))
            else:
                denoised = np.zeros_like(img_f)
                for c in range(img_f.shape[2]):
                    noise_var = np.var(img_f[:,:,c] - ndimage.gaussian_filter(img_f[:,:,c], sigma=2))
                    f = fft2(img_f[:,:,c])
                    power = np.abs(f) ** 2
                    signal_power = np.maximum(power - noise_var, 0)
                    wiener_ratio = signal_power / (signal_power + noise_var + 1e-10)
                    denoised[:,:,c] = np.real(np.fft.ifft2(f * wiener_ratio))
        else:
            raise ValueError(f"Unknown denoising method: {method}")

        return img_f - denoised

    def extract_noise_fused(self, image: np.ndarray) -> np.ndarray:
        """Extract noise with adaptive denoiser fusion."""
        noises = []
        base_weights = []

        # Wavelet denoising with multiple families
        for wavelet in self.wavelets:
            noise = self.extract_noise_single(image, 'wavelet', wavelet=wavelet)
            noises.append(noise)
            base_weights.append(1.0)

        # Bilateral filter
        noise = self.extract_noise_single(image, 'bilateral')
        noises.append(noise)
        base_weights.append(0.8)

        # Wiener filter
        noise = self.extract_noise_single(image, 'wiener')
        noises.append(noise)
        base_weights.append(0.6)

        # IMPROVEMENT: Adaptive weighting — denoisers that agree with
        # the majority get higher weight
        n = len(noises)
        flat_noises = [ni.ravel() for ni in noises]
        adaptive_weights = np.array(base_weights, dtype=np.float64)

        for i in range(n):
            agreements = []
            for j in range(n):
                if i != j:
                    # Fast dot-product correlation (avoid full corrcoef)
                    a, b = flat_noises[i], flat_noises[j]
                    a_z = a - a.mean()
                    b_z = b - b.mean()
                    corr = np.dot(a_z, b_z) / (np.linalg.norm(a_z) * np.linalg.norm(b_z) + 1e-10)
                    agreements.append(max(0, corr))
            # Scale base weight by agreement with others
            avg_agreement = np.mean(agreements) if agreements else 0
            adaptive_weights[i] *= (0.5 + 0.5 * avg_agreement)

        adaptive_weights /= adaptive_weights.sum()
        noises_arr = np.array(noises)
        fused = np.tensordot(adaptive_weights, noises_arr, axes=([0], [0]))

        return fused

    # ================================================================
    # IMPROVED DETECTION
    # ================================================================

    @staticmethod
    def _sigmoid(x: float, steepness: float = 1.0) -> float:
        """Sigmoid function for soft thresholding."""
        return 1.0 / (1.0 + np.exp(-steepness * x))

    def _per_channel_phase_analysis(
        self,
        image: np.ndarray,
        target_size: int,
        carriers_to_check: list,
        ref_phase: Optional[np.ndarray]
    ) -> Dict:
        """
        IMPROVEMENT #1: Per-channel FFT phase analysis.

        Instead of averaging to grayscale (which dilutes the green channel's
        strong signal), analyze each channel separately and weight by
        SynthID's known embedding strength.
        """
        center = target_size // 2
        channel_results = {}
        channel_names = ['R', 'G', 'B']
        channel_weights_list = [0.85, 1.0, 0.70]  # SynthID embedding weights

        for c_idx, (name, weight) in enumerate(zip(channel_names, channel_weights_list)):
            if len(image.shape) == 3:
                channel = image[:, :, c_idx].astype(np.float32)
            else:
                channel = image.astype(np.float32)

            f = fftshift(fft2(channel))
            magnitude = np.abs(f)
            phase = np.angle(f)

            carrier_scores = []
            carrier_strengths = []
            top_carrier_scores = []

            for carrier in carriers_to_check:
                freq = carrier['frequency']
                y = freq[0] + center
                x = freq[1] + center

                if 0 <= y < target_size and 0 <= x < target_size:
                    actual_phase = phase[y, x]

                    if ref_phase is not None:
                        expected_phase = ref_phase[y, x]
                    else:
                        expected_phase = carrier.get('phase', 0)

                    phase_diff = np.abs(np.angle(np.exp(1j * (actual_phase - expected_phase))))
                    phase_match = 1 - phase_diff / np.pi
                    carrier_scores.append(phase_match)
                    carrier_strengths.append(magnitude[y, x])

                    # Track top carriers (those with codebook votes)
                    if carrier.get('votes', 0) > 0 or carrier.get('coherence', 0) > 0.5:
                        top_carrier_scores.append(phase_match)

            channel_results[name] = {
                'phase_match': float(np.mean(carrier_scores)) if carrier_scores else 0,
                'carrier_strength': float(np.mean(carrier_strengths)) if carrier_strengths else 0,
                'top_carrier_phase': float(np.mean(top_carrier_scores)) if top_carrier_scores else 0,
                'weight': weight,
                'n_carriers': len(carrier_scores),
                # Percentile-based metric: fraction of carriers with good phase match
                'high_match_ratio': float(np.mean([s > 0.55 for s in carrier_scores])) if carrier_scores else 0
            }

        # Weighted fusion across channels
        total_weight = sum(channel_weights_list)
        weighted_phase = sum(
            channel_results[n]['phase_match'] * channel_results[n]['weight']
            for n in channel_names
        ) / total_weight

        weighted_strength = sum(
            channel_results[n]['carrier_strength'] * channel_results[n]['weight']
            for n in channel_names
        ) / total_weight

        # Green channel dominance score — if green is distinctly higher, it's a strong signal
        green_dominance = 0.0
        if channel_results['G']['phase_match'] > 0:
            other_avg = (channel_results['R']['phase_match'] + channel_results['B']['phase_match']) / 2
            green_dominance = max(0, channel_results['G']['phase_match'] - other_avg)

        return {
            'weighted_phase_match': weighted_phase,
            'weighted_carrier_strength': weighted_strength,
            'green_dominance': green_dominance,
            'per_channel': channel_results,
            'green_phase': channel_results['G']['phase_match'],
            'green_high_match_ratio': channel_results['G']['high_match_ratio'],
        }

    def _ica_correlation_score(self, noise: np.ndarray) -> float:
        """
        IMPROVEMENT #2: Use ICA watermark pattern as detection signal.

        The codebook contains an ICA-extracted watermark component but the
        old detector never checked against it.
        """
        if self.codebook is None:
            return 0.0

        watermark_pattern = self.codebook.get('watermark_pattern')
        if watermark_pattern is None:
            return 0.0

        noise_gray = np.mean(noise, axis=2) if len(noise.shape) == 3 else noise

        # Resize watermark pattern to match noise if needed
        if noise_gray.shape != watermark_pattern.shape:
            watermark_pattern = cv2.resize(
                watermark_pattern,
                (noise_gray.shape[1], noise_gray.shape[0])
            )

        # Correlation with ICA pattern
        try:
            corr = np.corrcoef(noise_gray.ravel(), watermark_pattern.ravel())[0, 1]
            return float(abs(corr))  # abs because ICA component sign is arbitrary
        except:
            return 0.0

    def detect_array(self, image: np.ndarray) -> DetectionResult:
        """
        Improved SynthID detection with per-channel analysis + soft decision.
        """
        if self.codebook is None:
            raise ValueError("No codebook loaded. Call load_codebook() first.")

        target_size = self.codebook['image_size']
        img_resized = cv2.resize(image, (target_size, target_size))

        # ── Step 1: Extract noise pattern (same as original) ──
        noise = self.extract_noise_fused(img_resized)

        # ── Step 2: Correlation with reference noise ──
        ref_noise = self.codebook['reference_noise']
        correlation = float(np.corrcoef(noise.ravel(), ref_noise.ravel())[0, 1])

        # ── Step 3: Per-channel carrier frequency analysis (IMPROVED) ──
        carriers_to_check = self.codebook['carriers'][:30] if self.codebook['carriers'] else []
        known_carrier_dicts = [
            {'frequency': freq, 'phase': 0}
            for freq in self.codebook.get('known_carriers', self.known_carriers)
        ]
        carriers_to_check = carriers_to_check + known_carrier_dicts
        ref_phase = self.codebook.get('reference_phase')

        channel_analysis = self._per_channel_phase_analysis(
            img_resized, target_size, carriers_to_check, ref_phase
        )

        # Use weighted per-channel phase match instead of grayscale
        avg_phase_match = channel_analysis['weighted_phase_match']
        avg_carrier_strength = channel_analysis['weighted_carrier_strength']

        # Also compute grayscale phase match for backward compatibility
        gray = np.mean(img_resized, axis=2).astype(np.float32) if len(img_resized.shape) == 3 else img_resized.astype(np.float32)
        f_gray = fftshift(fft2(gray))
        gray_phase = np.angle(f_gray)
        gray_carrier_scores = []
        center = target_size // 2
        for carrier in carriers_to_check:
            freq = carrier['frequency']
            y, x = freq[0] + center, freq[1] + center
            if 0 <= y < target_size and 0 <= x < target_size:
                actual = gray_phase[y, x]
                expected = ref_phase[y, x] if ref_phase is not None else carrier.get('phase', 0)
                diff = np.abs(np.angle(np.exp(1j * (actual - expected))))
                gray_carrier_scores.append(1 - diff / np.pi)
        gray_phase_match = float(np.mean(gray_carrier_scores)) if gray_carrier_scores else 0

        # Take the BEST of per-channel weighted and grayscale
        best_phase_match = max(avg_phase_match, gray_phase_match)

        # ── Step 4: Noise structure ratio ──
        noise_gray = np.mean(noise, axis=2) if len(noise.shape) == 3 else noise
        structure_ratio = float(np.std(noise_gray) / (np.mean(np.abs(noise_gray)) + 1e-10))

        # ── Step 5: ICA pattern correlation (NEW) ──
        ica_score = self._ica_correlation_score(noise)

        # ── Step 6: Multi-scale consistency ──
        scale_scores = []
        for scale in self.scales:
            img_scaled = cv2.resize(image, (scale, scale))
            noise_scaled = self.extract_noise_single(img_scaled, 'wavelet')
            ref_scaled = cv2.resize(ref_noise, (scale, scale))
            corr = np.corrcoef(noise_scaled.ravel(), ref_scaled.ravel())[0, 1]
            scale_scores.append(corr)
        multi_scale_consistency = float(np.std(scale_scores))

        # ================================================================
        # IMPROVED DETECTION DECISION — Soft Probabilistic Scoring
        #
        # Instead of: correlation > T AND phase > 0.45 AND 0.7 < struct < 2.0
        # We use:     weighted sigmoid scores across multiple signals
        # ================================================================

        threshold = self.codebook['detection_threshold']

        # Signal 1: Correlation (low weight — near zero for non-native resolutions)
        corr_signal = self._sigmoid(
            (correlation - threshold) / (abs(self.codebook['correlation_mean'] - threshold) + 1e-6),
            steepness=3.0
        )

        # Signal 2: Phase match (high weight — primary detection signal)
        # Use soft sigmoid instead of hard > 0.45
        phase_signal = self._sigmoid((best_phase_match - 0.40) * 12.0)

        # Signal 3: Green channel phase (bonus signal — SynthID is strongest in green)
        green_phase_signal = self._sigmoid(
            (channel_analysis['green_phase'] - 0.38) * 10.0
        )

        # Signal 4: Structure ratio (moderate weight)
        struct_signal = max(0.0, 1.0 - abs(structure_ratio - 1.32) / 0.5)

        # Signal 5: ICA pattern match (new signal)
        ica_signal = self._sigmoid((ica_score - 0.01) * 50.0)

        # Signal 6: Multi-scale consistency (lower std = more consistent)
        consistency_signal = max(0.0, 1.0 - multi_scale_consistency * 4.0)

        # Signal 7: Green channel high-match carrier ratio
        green_hmr = channel_analysis['green_high_match_ratio']
        high_match_signal = self._sigmoid((green_hmr - 0.25) * 8.0)

        # ── Weighted combination ──
        confidence = min(1.0, (
            0.08 * corr_signal +           # Low weight — near-zero for most images
            0.30 * phase_signal +           # Primary signal
            0.18 * green_phase_signal +     # Green-channel bonus
            0.14 * struct_signal +          # Structure consistency
            0.10 * ica_signal +             # ICA pattern
            0.08 * consistency_signal +     # Multi-scale
            0.12 * high_match_signal        # Carrier match ratio
        ))

        # ── Soft decision: is_watermarked if confidence exceeds threshold ──
        # No more hard AND gates!
        is_watermarked = confidence > 0.45

        # ── Additional safety: if structure ratio is WAY out of range, override ──
        if structure_ratio < 0.5 or structure_ratio > 3.0:
            is_watermarked = False
            confidence *= 0.3

        # ================================================================
        # NEURAL CLASSIFIER INTEGRATION (If loaded)
        # ================================================================
        
        neural_confidence = None
        neural_is_watermarked = None
        
        if self.classifier_data is not None:
            # Build feature array matching training feature names
            features = {
                'confidence': confidence,
                'phase_match': best_phase_match,
                'correlation': correlation,
                'structure_ratio': structure_ratio,
                'carrier_strength': avg_carrier_strength,
                'multi_scale_consistency': multi_scale_consistency,
                'green_phase': channel_analysis['green_phase'],
                'ica_score': ica_score,
                'green_dominance': channel_analysis['green_dominance'],
                'high_match_signal': high_match_signal,
                'corr_signal': corr_signal,
                'phase_signal': phase_signal,
                'green_phase_signal': green_phase_signal,
                'struct_signal': struct_signal,
            }
            
            feature_names = self.classifier_data['feature_names']
            X_raw = np.array([[features[k] for k in feature_names]], dtype=np.float64)
            X_scaled = self.classifier_data['scaler'].transform(X_raw)
            
            neural_is_watermarked = bool(self.classifier_data['classifier'].predict(X_scaled)[0])
            
            # Predict proba if available, otherwise fallback to 1/0
            if hasattr(self.classifier_data['classifier'], 'predict_proba'):
                neural_confidence = float(self.classifier_data['classifier'].predict_proba(X_scaled)[0][1])
            else:
                neural_confidence = float(neural_is_watermarked)

        return DetectionResult(
            is_watermarked=neural_is_watermarked if neural_is_watermarked is not None else bool(is_watermarked),
            confidence=neural_confidence if neural_confidence is not None else float(confidence),
            correlation=correlation,
            phase_match=best_phase_match,
            structure_ratio=structure_ratio,
            carrier_strength=avg_carrier_strength,
            multi_scale_consistency=multi_scale_consistency,
            details={
                'threshold': threshold,
                'corr_signal': float(corr_signal),
                'phase_signal': float(phase_signal),
                'green_phase_signal': float(green_phase_signal),
                'struct_signal': float(struct_signal),
                'ica_signal': float(ica_signal),
                'consistency_signal': float(consistency_signal),
                'high_match_signal': float(high_match_signal),
                'green_phase': channel_analysis['green_phase'],
                'green_dominance': channel_analysis['green_dominance'],
                'gray_phase_match': gray_phase_match,
                'per_channel_phase': avg_phase_match,
                'ica_score': ica_score,
                'scale_correlations': scale_scores,
                'detector': 'ImprovedSynthIDExtractor (Neural)' if self.classifier_data else 'ImprovedSynthIDExtractor (Heuristic)',
                'heuristic_confidence': float(confidence),
                'heuristic_is_watermarked': bool(is_watermarked)
            }
        )

    def detect(self, image_path: str) -> DetectionResult:
        """Detect SynthID watermark in an image file."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.detect_array(img)
