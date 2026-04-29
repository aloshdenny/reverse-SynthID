"""
Calibration Pipeline — Use Bypass to Create Clean Samples

Strategy:
1. Run V3 spectral bypass on all 88 watermarked images -> creates 88 "cleaned" images
2. Run improved detector on BOTH sets (watermarked + cleaned)
3. Find the optimal decision threshold that maximizes separation
4. Update the improved extractor with calibrated thresholds
"""

import os
import sys
import time
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'extraction'))

from improved_extractor import ImprovedSynthIDExtractor


def create_cleaned_images(input_dir, output_dir, codebook_v3_path, limit=None):
    """Run V3 bypass on all watermarked images to create clean samples."""
    from synthid_bypass import SynthIDBypass, SpectralCodebook

    os.makedirs(output_dir, exist_ok=True)

    # Load spectral codebook
    print("Loading SpectralCodebook V3...")
    codebook = SpectralCodebook()
    codebook.load(codebook_v3_path)
    print(f"  Loaded {len(codebook.profiles)} resolution profiles")

    # Create bypass instance (no extractor for speed — we'll verify ourselves)
    bypass = SynthIDBypass(extractor=None)

    # Get image files
    extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])
    if limit:
        image_files = image_files[:limit]

    print(f"\nProcessing {len(image_files)} images with V3 bypass (aggressive)...")
    print("-" * 70)

    results = []
    for i, fname in enumerate(image_files):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        try:
            img = cv2.imread(input_path)
            if img is None:
                print(f"  [{i+1:3d}/{len(image_files)}] SKIP {fname[:50]} (could not load)")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Run V3 bypass
            result = bypass.bypass_v3(
                img_rgb, codebook=codebook,
                strength='aggressive', verify=False
            )

            # Save cleaned image
            cv2.imwrite(output_path, cv2.cvtColor(result.cleaned_image, cv2.COLOR_RGB2BGR))

            results.append({
                'file': fname,
                'psnr': round(result.psnr, 1),
                'ssim': round(result.ssim, 4),
            })

            print(f"  [{i+1:3d}/{len(image_files)}] OK  PSNR={result.psnr:.1f}dB  SSIM={result.ssim:.4f}  {fname[:45]}")

        except Exception as e:
            print(f"  [{i+1:3d}/{len(image_files)}] ERROR {fname[:50]} -> {e}")
            import traceback
            traceback.print_exc()

    print(f"\nCleaned {len(results)} images saved to {output_dir}/")
    return results


def run_calibration():
    # Paths
    codebook_path = os.path.join('artifacts', 'codebook', 'robust_codebook.pkl')
    codebook_v3_path = os.path.join('artifacts', 'spectral_codebook_v3.npz')
    watermarked_dir = 'gemini_random'
    cleaned_dir = 'gemini_cleaned_v3'

    print("=" * 70)
    print("  CALIBRATION PIPELINE")
    print("  Step 1: Create clean samples via V3 bypass")
    print("  Step 2: Test detector on both sets")
    print("  Step 3: Find optimal threshold")
    print("=" * 70)

    # ── Step 1: Create cleaned images ──
    if not os.path.exists(cleaned_dir) or len(os.listdir(cleaned_dir)) < 10:
        create_cleaned_images(watermarked_dir, cleaned_dir, codebook_v3_path)
    else:
        print(f"\nUsing existing cleaned images in {cleaned_dir}/ ({len(os.listdir(cleaned_dir))} files)")

    # ── Step 2: Run detector on BOTH sets ──
    print("\n" + "=" * 70)
    print("  Step 2: Running detector on watermarked + cleaned images")
    print("=" * 70)

    extractor = ImprovedSynthIDExtractor()
    extractor.load_codebook(codebook_path)

    extensions = {'.png', '.jpg', '.jpeg', '.webp'}

    # Test watermarked images
    watermarked_files = sorted([
        os.path.join(watermarked_dir, f) for f in os.listdir(watermarked_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])
    # Test cleaned images
    cleaned_files = sorted([
        os.path.join(cleaned_dir, f) for f in os.listdir(cleaned_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])

    print(f"\n  Watermarked images: {len(watermarked_files)}")
    print(f"  Cleaned images:    {len(cleaned_files)}")

    # Collect confidence scores for both sets
    wm_confidences = []
    wm_phases = []
    cl_confidences = []
    cl_phases = []

    print(f"\n  --- Scanning watermarked images ---")
    for i, path in enumerate(watermarked_files):
        try:
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = extractor.detect_array(img_rgb)
            wm_confidences.append(result.confidence)
            wm_phases.append(result.phase_match)
            if (i + 1) % 20 == 0 or i == 0:
                print(f"    [{i+1:3d}/{len(watermarked_files)}] conf={result.confidence:.4f}  phase={result.phase_match:.4f}")
        except Exception as e:
            print(f"    [{i+1:3d}] ERROR: {e}")

    print(f"\n  --- Scanning cleaned images ---")
    for i, path in enumerate(cleaned_files):
        try:
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = extractor.detect_array(img_rgb)
            cl_confidences.append(result.confidence)
            cl_phases.append(result.phase_match)
            if (i + 1) % 20 == 0 or i == 0:
                print(f"    [{i+1:3d}/{len(cleaned_files)}] conf={result.confidence:.4f}  phase={result.phase_match:.4f}")
        except Exception as e:
            print(f"    [{i+1:3d}] ERROR: {e}")

    wm_conf = np.array(wm_confidences)
    cl_conf = np.array(cl_confidences)
    wm_ph = np.array(wm_phases)
    cl_ph = np.array(cl_phases)

    # ── Step 3: Distribution analysis ──
    print("\n" + "=" * 70)
    print("  Step 3: Distribution Analysis")
    print("=" * 70)

    print(f"\n  {'Metric':<25} {'Watermarked':>15} {'Cleaned':>15}  {'Gap':>10}")
    print(f"  {'-'*25} {'-'*15} {'-'*15}  {'-'*10}")
    print(f"  {'Avg Confidence':<25} {np.mean(wm_conf):>15.4f} {np.mean(cl_conf):>15.4f}  {np.mean(wm_conf)-np.mean(cl_conf):>+10.4f}")
    print(f"  {'Min Confidence':<25} {np.min(wm_conf):>15.4f} {np.min(cl_conf):>15.4f}")
    print(f"  {'Max Confidence':<25} {np.max(wm_conf):>15.4f} {np.max(cl_conf):>15.4f}")
    print(f"  {'Avg Phase Match':<25} {np.mean(wm_ph):>15.4f} {np.mean(cl_ph):>15.4f}  {np.mean(wm_ph)-np.mean(cl_ph):>+10.4f}")
    print(f"  {'Min Phase Match':<25} {np.min(wm_ph):>15.4f} {np.min(cl_ph):>15.4f}")
    print(f"  {'Max Phase Match':<25} {np.max(wm_ph):>15.4f} {np.max(cl_ph):>15.4f}")

    # ── Overlap analysis ──
    overlap_conf = max(0, min(np.max(cl_conf), np.max(wm_conf)) - max(np.min(cl_conf), np.min(wm_conf)))
    conf_separable = np.min(wm_conf) > np.max(cl_conf)

    print(f"\n  Confidence distributions:")
    print(f"    Watermarked range: [{np.min(wm_conf):.4f}, {np.max(wm_conf):.4f}]")
    print(f"    Cleaned range:     [{np.min(cl_conf):.4f}, {np.max(cl_conf):.4f}]")
    print(f"    Perfectly separable: {'YES' if conf_separable else 'NO (overlap exists)'}")

    # ── Find optimal threshold ──
    print(f"\n  --- Optimal Threshold Search ---")
    best_threshold = 0.5
    best_accuracy = 0
    best_tp = 0
    best_fp = 0

    for t in np.arange(0.1, 0.95, 0.01):
        tp = np.sum(wm_conf > t)  # True positives
        fn = np.sum(wm_conf <= t)  # False negatives
        tn = np.sum(cl_conf <= t)  # True negatives
        fp = np.sum(cl_conf > t)   # False positives

        accuracy = (tp + tn) / (tp + fn + tn + fp)
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t
            best_tp = tp
            best_tn = tn
            best_fn = fn
            best_fp = fp
            best_tpr = tpr
            best_fpr = fpr

    print(f"\n  Optimal confidence threshold: {best_threshold:.2f}")
    print(f"  At this threshold:")
    print(f"    True Positives:  {best_tp}/{len(wm_conf)} ({100*best_tp/len(wm_conf):.1f}%)")
    print(f"    False Negatives: {best_fn}/{len(wm_conf)} ({100*best_fn/len(wm_conf):.1f}%)")
    print(f"    True Negatives:  {best_tn}/{len(cl_conf)} ({100*best_tn/len(cl_conf):.1f}%)")
    print(f"    False Positives: {best_fp}/{len(cl_conf)} ({100*best_fp/len(cl_conf):.1f}%)")
    print(f"    Overall Accuracy: {100*best_accuracy:.1f}%")
    print(f"    TPR (Sensitivity): {100*best_tpr:.1f}%")
    print(f"    FPR:               {100*best_fpr:.1f}%")

    # ── Also try phase match threshold ──
    print(f"\n  --- Phase Match Threshold Search ---")
    best_ph_threshold = 0.45
    best_ph_accuracy = 0

    for t in np.arange(0.30, 0.60, 0.005):
        tp = np.sum(wm_ph > t)
        fn = np.sum(wm_ph <= t)
        tn = np.sum(cl_ph <= t)
        fp = np.sum(cl_ph > t)

        accuracy = (tp + tn) / (tp + fn + tn + fp)

        if accuracy > best_ph_accuracy:
            best_ph_accuracy = accuracy
            best_ph_threshold = t
            best_ph_tp = tp
            best_ph_tn = tn

    print(f"  Optimal phase threshold: {best_ph_threshold:.3f}")
    print(f"  At this threshold:")
    print(f"    True Positives:  {best_ph_tp}/{len(wm_ph)}")
    print(f"    True Negatives:  {best_ph_tn}/{len(cl_ph)}")
    print(f"    Overall Accuracy: {100*best_ph_accuracy:.1f}%")

    print("\n" + "=" * 70)
    print("  CALIBRATION COMPLETE")
    print("=" * 70)

    # Save calibration report
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'watermarked_count': len(wm_conf),
        'cleaned_count': len(cl_conf),
        'watermarked_confidence': {
            'mean': round(float(np.mean(wm_conf)), 4),
            'min': round(float(np.min(wm_conf)), 4),
            'max': round(float(np.max(wm_conf)), 4),
            'std': round(float(np.std(wm_conf)), 4),
        },
        'cleaned_confidence': {
            'mean': round(float(np.mean(cl_conf)), 4),
            'min': round(float(np.min(cl_conf)), 4),
            'max': round(float(np.max(cl_conf)), 4),
            'std': round(float(np.std(cl_conf)), 4),
        },
        'watermarked_phase': {
            'mean': round(float(np.mean(wm_ph)), 4),
            'min': round(float(np.min(wm_ph)), 4),
            'max': round(float(np.max(wm_ph)), 4),
        },
        'cleaned_phase': {
            'mean': round(float(np.mean(cl_ph)), 4),
            'min': round(float(np.min(cl_ph)), 4),
            'max': round(float(np.max(cl_ph)), 4),
        },
        'optimal_confidence_threshold': round(float(best_threshold), 2),
        'optimal_phase_threshold': round(float(best_ph_threshold), 3),
        'best_accuracy': round(float(best_accuracy) * 100, 1),
        'distributions': {
            'watermarked_confidences': [round(float(c), 4) for c in wm_conf],
            'cleaned_confidences': [round(float(c), 4) for c in cl_conf],
        }
    }

    with open('calibration_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nCalibration results saved to: calibration_results.json")

    return report


if __name__ == '__main__':
    run_calibration()
