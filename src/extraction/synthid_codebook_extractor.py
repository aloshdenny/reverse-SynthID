"""
SynthID Codebook Extractor

Based on analysis of 250 Gemini images, this script extracts and saves
the discovered SynthID watermark codebook for detection purposes.

KEY FINDINGS:
1. Carrier frequencies at specific locations (±14, ±14), (±126, ±14), etc.
2. High phase coherence (0.99+) at carrier frequencies
3. Noise correlation of ~0.21 between watermarked images
4. Noise structure ratio of ~1.32

The codebook consists of:
1. Reference noise pattern (average across all images)
2. Carrier frequency locations and expected phases
3. Detection thresholds
"""

import os
import sys
import numpy as np
import cv2
from scipy.fft import fft2, fftshift
import pywt
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.utils import load_config, setup_logging, validate_image, validate_codebook, secure_pickle_load

logger = logging.getLogger('reverse_synthid.extraction')


def wavelet_denoise(
    channel: np.ndarray,
    wavelet: str = 'db4',
    level: int = 3
) -> np.ndarray:
    """
    Wavelet-based denoising using soft thresholding.
    
    Args:
        channel: 2D array representing a single image channel
        wavelet: Wavelet type (default: 'db4')
        level: Decomposition level (default: 3)
    
    Returns:
        Denoised channel as 2D array
    
    Raises:
        ValueError: If channel is invalid or empty
    """
    if channel.size == 0:
        raise ValueError("Cannot denoise empty channel")
    
    if channel.ndim != 2:
        raise ValueError(f"Channel must be 2D, got {channel.ndim}D")
    
    try:
        # Decompose
        coeffs = pywt.wavedec2(channel, wavelet, level=level)
        
        # Estimate noise level from finest detail coefficients
        detail = coeffs[-1][0]
        sigma = np.median(np.abs(detail)) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(channel.size))
        
        # Apply threshold to detail coefficients
        new_coeffs = [coeffs[0]]
        for details in coeffs[1:]:
            new_details = tuple(pywt.threshold(d, threshold, mode='soft') for d in details)
            new_coeffs.append(new_details)
        
        # Reconstruct
        denoised = pywt.waverec2(new_coeffs, wavelet)
        
        # Handle shape mismatch (fix bug)
        if denoised.shape != channel.shape:
            # Crop if larger
            if denoised.shape[0] >= channel.shape[0] and denoised.shape[1] >= channel.shape[1]:
                denoised = denoised[:channel.shape[0], :channel.shape[1]]
            # Pad if smaller
            else:
                padded = np.zeros(channel.shape, dtype=denoised.dtype)
                padded[:denoised.shape[0], :denoised.shape[1]] = denoised
                denoised = padded
        
        return denoised
    
    except Exception as e:
        logger.error(f"Wavelet denoising failed: {e}")
        raise ValueError(f"Wavelet denoising error: {e}")


def extract_codebook(
    image_dir: str,
    output_path: str,
    max_images: int = 250,
    size: int = 512
) -> Dict[str, Any]:
    """
    Extract SynthID codebook from a collection of watermarked images.
    
    Args:
        image_dir: Directory containing watermarked images
        output_path: Path to save codebook pickle file
        max_images: Maximum number of images to analyze
        size: Target image size (will resize to size×size)
    
    Returns:
        Extracted codebook dictionary
    
    Raises:
        FileNotFoundError: If image directory doesn't exist
        ValueError: If no valid images found or max_images < 10
    """
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    
    if max_images < 10:
        raise ValueError(f"max_images must be at least 10, got {max_images}")
    
    logger.info(f"Loading images from {image_dir}...")
    
    # Load images with validation
    extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    images = []
    failed_count = 0
    
    for fname in sorted(os.listdir(image_dir)):
        if len(images) >= max_images:
            break
        
        if os.path.splitext(fname)[1].lower() not in extensions:
            continue
        
        path = os.path.join(image_dir, fname)
        
        try:
            # Validate before loading
            validate_image(path)
            
            img = cv2.imread(path)
            if img is None:
                logger.warning(f"Failed to load: {fname}")
                failed_count += 1
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
            images.append(img)
            
        except Exception as e:
            logger.warning(f"Skipping {fname}: {e}")
            failed_count += 1
            continue
    
    if len(images) == 0:
        raise ValueError(f"No valid images found in {image_dir}")
    
    if len(images) < 10:
        raise ValueError(
            f"Too few images loaded ({len(images)}). Need at least 10 for reliable codebook."
        )
    
    logger.info(f"Loaded {len(images)} images ({failed_count} failed)")
    images = np.array(images)
    
    # ================================================================
    # 1. EXTRACT REFERENCE NOISE PATTERN
    # ================================================================
    print("Extracting reference noise pattern...")
    
    noise_sum = np.zeros((size, size, 3), dtype=np.float64)
    
    for img in images:
        img_f = img.astype(np.float32) / 255.0
        for c in range(3):
            denoised = wavelet_denoise(img_f[:, :, c])
            noise_sum[:, :, c] += img_f[:, :, c] - denoised
    
    reference_noise = noise_sum / len(images)
    
    # ================================================================
    # 2. EXTRACT CARRIER FREQUENCIES
    # ================================================================
    print("Extracting carrier frequencies...")
    
    magnitude_sum = None
    phase_sum = None
    
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        f = fft2(gray)
        fshift = fftshift(f)
        
        if magnitude_sum is None:
            magnitude_sum = np.abs(fshift)
            phase_sum = np.exp(1j * np.angle(fshift))
        else:
            magnitude_sum += np.abs(fshift)
            phase_sum += np.exp(1j * np.angle(fshift))
    
    avg_magnitude = magnitude_sum / len(images)
    phase_coherence = np.abs(phase_sum) / len(images)
    avg_phase = np.angle(phase_sum)
    
    # Find carrier frequencies (high coherence, significant magnitude)
    log_mag = np.log1p(avg_magnitude)
    combined_score = log_mag * phase_coherence
    
    # Get top carriers
    threshold = np.percentile(combined_score, 99.5)
    carrier_mask = combined_score > threshold
    carrier_locs = np.where(carrier_mask)
    
    center = size // 2
    carriers = []
    for y, x in zip(carrier_locs[0], carrier_locs[1]):
        freq_y, freq_x = y - center, x - center
        # Skip DC
        if abs(freq_y) < 5 and abs(freq_x) < 5:
            continue
        carriers.append({
            'position': (int(y), int(x)),
            'frequency': (int(freq_y), int(freq_x)),
            'magnitude': float(avg_magnitude[y, x]),
            'phase': float(avg_phase[y, x]),
            'coherence': float(phase_coherence[y, x])
        })
    
    carriers.sort(key=lambda c: c['coherence'] * np.log1p(c['magnitude']), reverse=True)
    carriers = carriers[:100]  # Top 100 carriers
    
    # ================================================================
    # 3. COMPUTE DETECTION THRESHOLDS
    # ================================================================
    print("Computing detection thresholds...")
    
    # Compute noise correlations for threshold calibration
    correlations = []
    for i in range(min(50, len(images))):
        for j in range(i+1, min(50, len(images))):
            img1 = images[i].astype(np.float32) / 255.0
            img2 = images[j].astype(np.float32) / 255.0
            
            noise1 = np.zeros((size, size, 3))
            noise2 = np.zeros((size, size, 3))
            
            for c in range(3):
                noise1[:, :, c] = img1[:, :, c] - wavelet_denoise(img1[:, :, c])
                noise2[:, :, c] = img2[:, :, c] - wavelet_denoise(img2[:, :, c])
            
            corr = np.corrcoef(noise1.ravel(), noise2.ravel())[0, 1]
            correlations.append(corr)
    
    correlation_mean = float(np.mean(correlations))
    correlation_std = float(np.std(correlations))
    
    # Detection threshold: if correlation > mean - 2*std, likely watermarked
    detection_threshold = correlation_mean - 2 * correlation_std
    
    # ================================================================
    # 4. CREATE CODEBOOK
    # ================================================================
    print("Creating codebook...")
    
    codebook = {
        'version': '1.0',
        'source': 'Gemini/SynthID',
        'n_images_analyzed': len(images),
        'image_size': size,
        
        # Reference patterns
        'reference_noise': reference_noise,
        'reference_magnitude': avg_magnitude,
        'reference_phase': avg_phase,
        'phase_coherence': phase_coherence,
        
        # Carrier frequencies
        'carriers': carriers,
        'n_carriers': len(carriers),
        
        # Detection parameters
        'correlation_mean': correlation_mean,
        'correlation_std': correlation_std,
        'detection_threshold': detection_threshold,
        'noise_structure_ratio': 1.32,  # From previous analysis
        
        # Key carrier frequencies (simplified)
        'key_frequencies': [
            {'freq': (14, 14), 'coherence': 0.9996},
            {'freq': (-14, -14), 'coherence': 0.9996},
            {'freq': (126, 14), 'coherence': 0.9996},
            {'freq': (-126, -14), 'coherence': 0.9996},
            {'freq': (98, -14), 'coherence': 0.9994},
            {'freq': (-98, 14), 'coherence': 0.9994},
            {'freq': (128, 128), 'coherence': 0.9925},
            {'freq': (-128, -128), 'coherence': 0.9925},
        ]
    }
    
    # Save codebook
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save as pickle (includes numpy arrays)
    with open(output_path, 'wb') as f:
        pickle.dump(codebook, f)
    
    # Save metadata as JSON
    json_path = output_path.replace('.pkl', '_meta.json')
    meta = {
        'version': codebook['version'],
        'source': codebook['source'],
        'n_images_analyzed': codebook['n_images_analyzed'],
        'image_size': codebook['image_size'],
        'n_carriers': codebook['n_carriers'],
        'correlation_mean': codebook['correlation_mean'],
        'correlation_std': codebook['correlation_std'],
        'detection_threshold': codebook['detection_threshold'],
        'key_frequencies': codebook['key_frequencies'],
        'carriers': codebook['carriers'][:20]  # Top 20 for reference
    }
    
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nCodebook saved to {output_path}")
    print(f"Metadata saved to {json_path}")
    
    return codebook


def detect_synthid(
    image_path: str,
    codebook_path: str
) -> Dict[str, Any]:
    """
    Detect SynthID watermark in an image using the extracted codebook.
    
    Args:
        image_path: Path to image to check
        codebook_path: Path to codebook pickle file
    
    Returns:
        Dictionary with detection results:
        - is_watermarked: bool
        - confidence: float (0-1)
        - correlation: float
        - phase_match: float (0-1)
        - structure_ratio: float
        - threshold: float
        - reference_correlation_mean: float
    
    Raises:
        FileNotFoundError: If image or codebook not found
        ValueError: If image or codebook is invalid
    """
    # Validate and load codebook securely
    try:
        codebook = secure_pickle_load(codebook_path)
        validate_codebook(codebook)
    except Exception as e:
        logger.error(f"Failed to load codebook: {e}")
        raise ValueError(f"Invalid codebook: {e}")
    
    # Validate and load image
    try:
        validate_image(image_path)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        raise
    
    # Preprocess image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    size = codebook['image_size']
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    img_f = img.astype(np.float32) / 255.0
    
    # Extract noise pattern
    noise = np.zeros((size, size, 3))
    for c in range(3):
        noise[:, :, c] = img_f[:, :, c] - wavelet_denoise(img_f[:, :, c])
    
    # Method 1: Correlation with reference noise
    ref_noise = codebook['reference_noise']
    correlation = np.corrcoef(noise.ravel(), ref_noise.ravel())[0, 1]
    
    # Method 2: Check carrier frequencies
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = fft2(gray)
    fshift = fftshift(f)
    magnitude = np.abs(fshift)
    phase = np.angle(fshift)
    
    center = size // 2
    carrier_scores = []
    for carrier in codebook['carriers'][:20]:
        y, x = carrier['position']
        expected_phase = carrier['phase']
        actual_phase = phase[y, x]
        
        # Phase difference (accounting for wrap-around)
        phase_diff = np.abs(np.angle(np.exp(1j * (actual_phase - expected_phase))))
        phase_match = 1 - phase_diff / np.pi
        
        carrier_scores.append(phase_match)
    
    avg_phase_match = float(np.mean(carrier_scores))
    
    # Method 3: Noise structure ratio
    noise_gray = np.mean(noise, axis=2)
    noise_mean = np.mean(np.abs(noise_gray))
    
    # Prevent division by zero
    if noise_mean < 1e-10:
        structure_ratio = 1.0
        logger.warning("Noise mean too small, defaulting structure_ratio to 1.0")
    else:
        structure_ratio = float(np.std(noise_gray) / noise_mean)
    
    # Detection decision
    threshold = codebook['detection_threshold']
    phase_threshold = codebook.get('phase_match_threshold', 0.5)
    
    is_watermarked = (
        correlation > threshold and
        avg_phase_match > phase_threshold and
        0.8 < structure_ratio < 1.8
    )
    
    # Confidence score (fixed to handle edge cases)
    correlation_component = 0.0
    if codebook['correlation_mean'] > threshold:
        correlation_component = max(0.0, min(1.0,
            (correlation - threshold) / (codebook['correlation_mean'] - threshold)
        )) * 0.4
    
    phase_component = avg_phase_match * 0.4
    
    # Structure component (handle deviations gracefully)
    structure_deviation = abs(structure_ratio - 1.32)
    structure_component = max(0.0, 1.0 - min(1.0, structure_deviation / 0.5)) * 0.2
    
    confidence = min(1.0, max(0.0, 
        correlation_component + phase_component + structure_component
    ))
    
    return {
        'is_watermarked': bool(is_watermarked),
        'confidence': float(confidence),
        'correlation': float(correlation),
        'phase_match': float(avg_phase_match),
        'structure_ratio': float(structure_ratio),
        'threshold': float(threshold),
        'reference_correlation_mean': float(codebook['correlation_mean'])
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SynthID Codebook Extractor and Detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract codebook from images
  %(prog)s extract /path/to/images --output codebook.pkl --max-images 250
  
  # Detect watermark in single image
  %(prog)s detect image.png --codebook codebook.pkl
  
  # Batch detect with custom config
  %(prog)s detect /path/to/images/ --batch --config config.yaml
"""
    )
    
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str, help='Log file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract codebook from images')
    extract_parser.add_argument('image_dir', type=str, help='Directory with watermarked images')
    extract_parser.add_argument('--output', type=str, default='./synthid_codebook.pkl', 
                               help='Output path (default: ./synthid_codebook.pkl)')
    extract_parser.add_argument('--max-images', type=int, default=250, 
                               help='Max images to analyze (default: 250)')
    extract_parser.add_argument('--size', type=int, default=512, 
                               help='Target image size (default: 512)')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect watermark in image(s)')
    detect_parser.add_argument('image', type=str, help='Image file or directory to check')
    detect_parser.add_argument('--codebook', type=str, default='./synthid_codebook.pkl', 
                              help='Codebook path (default: ./synthid_codebook.pkl)')
    detect_parser.add_argument('--batch', action='store_true', 
                              help='Process directory in batch mode')
    detect_parser.add_argument('--output', type=str, help='Output file for batch results (JSON)')
    
    args = parser.parse_args()
    
    # Setup logging
    try:
        if args.config:
            config = load_config(args.config)
        else:
            config = load_config()
    except FileNotFoundError:
        config = None
        logger.warning("No config file found, using defaults")
    
    setup_logging(level=args.log_level, log_file=args.log_file)
    
    try:
        if args.command == 'extract':
            logger.info("Starting codebook extraction...")
            codebook = extract_codebook(args.image_dir, args.output, args.max_images, args.size)
            logger.info(f"Codebook extraction complete: {args.output}")
            
        elif args.command == 'detect':
            if args.batch and os.path.isdir(args.image):
                # Batch detection
                logger.info(f"Batch detection on directory: {args.image}")
                results = []
                
                for fname in sorted(os.listdir(args.image)):
                    if os.path.splitext(fname)[1].lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                        path = os.path.join(args.image, fname)
                        try:
                            result = detect_synthid(path, args.codebook)
                            result['filename'] = fname
                            results.append(result)
                            print(f"✓ {fname}: {'WATERMARKED' if result['is_watermarked'] else 'CLEAN'} "
                                  f"(confidence: {result['confidence']:.3f})")
                        except Exception as e:
                            logger.error(f"Failed to process {fname}: {e}")
                            results.append({'filename': fname, 'error': str(e)})
                
                # Save results if output specified
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(results, f, indent=2)
                    logger.info(f"Batch results saved to {args.output}")
                
                # Summary
                watermarked = sum(1 for r in results if r.get('is_watermarked', False))
                print(f"\nSummary: {watermarked}/{len(results)} images watermarked")
                
            else:
                # Single image detection
                result = detect_synthid(args.image, args.codebook)
                
                print("\n" + "="*60)
                print("SynthID Watermark Detection Results")
                print("="*60)
                print(f"Image:            {args.image}")
                print(f"Watermarked:      {'YES' if result['is_watermarked'] else 'NO'}")
                print(f"Confidence:       {result['confidence']:.4f}")
                print(f"Correlation:      {result['correlation']:.4f} (threshold: {result['threshold']:.4f})")
                print(f"Phase Match:      {result['phase_match']:.4f}")
                print(f"Structure Ratio:  {result['structure_ratio']:.4f} (expected: ~1.32)")
                print("="*60)
                
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)
