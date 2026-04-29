"""
Improved Detection Test — New Detector
Tests the ImprovedSynthIDExtractor against the same 88 watermarked images
and compares with the baseline results.
"""

import os
import sys
import time
import json
import numpy as np
import cv2

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'extraction'))

from improved_extractor import ImprovedSynthIDExtractor

def run_improved_test():
    # Paths
    codebook_path = os.path.join('artifacts', 'codebook', 'robust_codebook.pkl')
    test_dir = 'gemini_random'

    print("=" * 70)
    print("  IMPROVED DETECTION TEST -- New Detector")
    print("=" * 70)

    # Load improved extractor
    print(f"\nLoading codebook from: {codebook_path}")
    extractor = ImprovedSynthIDExtractor()
    extractor.load_codebook(codebook_path)
    print(f"  Codebook loaded. Image size: {extractor.codebook['image_size']}")
    print(f"  Improvements: per-channel FFT, soft scoring, ICA pattern, adaptive denoise")

    # List test images
    extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    test_images = []
    for fname in sorted(os.listdir(test_dir)):
        if os.path.splitext(fname)[1].lower() in extensions:
            test_images.append(os.path.join(test_dir, fname))

    print(f"\nTest images: {len(test_images)} (all Gemini-generated, ALL should be detected)")
    print("-" * 70)

    # Run detection
    results = []
    detected_count = 0
    errors = 0
    start_time = time.time()

    for i, img_path in enumerate(test_images):
        fname = os.path.basename(img_path)
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"  [{i+1:3d}/{len(test_images)}] SKIP  {fname[:50]}... (could not load)")
                errors += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            result = extractor.detect_array(img_rgb)

            status = "[+] DETECTED" if result.is_watermarked else "[-] MISSED "
            if result.is_watermarked:
                detected_count += 1

            results.append({
                'file': fname,
                'resolution': f"{w}x{h}",
                'is_watermarked': result.is_watermarked,
                'confidence': round(result.confidence, 4),
                'correlation': round(result.correlation, 4),
                'phase_match': round(result.phase_match, 4),
                'structure_ratio': round(result.structure_ratio, 4),
                'carrier_strength': round(result.carrier_strength, 2),
                'multi_scale_consistency': round(result.multi_scale_consistency, 4),
                'green_phase': round(result.details.get('green_phase', 0), 4),
                'ica_score': round(result.details.get('ica_score', 0), 4),
            })

            print(f"  [{i+1:3d}/{len(test_images)}] {status}  conf={result.confidence:.4f}  "
                  f"phase={result.phase_match:.4f}  green={result.details.get('green_phase', 0):.4f}  "
                  f"ica={result.details.get('ica_score', 0):.4f}  "
                  f"struct={result.structure_ratio:.4f}  [{w}x{h}]")

        except Exception as e:
            print(f"  [{i+1:3d}/{len(test_images)}] ERROR {fname[:50]}... -> {e}")
            import traceback
            traceback.print_exc()
            errors += 1

    elapsed = time.time() - start_time

    # Summary
    total_valid = len(results)
    missed = [r for r in results if not r['is_watermarked']]
    detection_rate = (detected_count / total_valid * 100) if total_valid > 0 else 0

    print("\n" + "=" * 70)
    print("  IMPROVED DETECTOR RESULTS")
    print("=" * 70)
    print(f"  Total images tested:    {total_valid}")
    print(f"  Correctly detected:     {detected_count}")
    print(f"  Missed (false neg):     {len(missed)}")
    print(f"  Errors:                 {errors}")
    print(f"  Detection rate:         {detection_rate:.1f}%")
    print(f"  Time:                   {elapsed:.1f}s ({elapsed/max(total_valid,1):.2f}s per image)")

    if results:
        confs = [r['confidence'] for r in results]
        phases = [r['phase_match'] for r in results]
        structs = [r['structure_ratio'] for r in results]
        greens = [r['green_phase'] for r in results]
        icas = [r['ica_score'] for r in results]

        print(f"\n  Avg confidence:         {np.mean(confs):.4f} (min={np.min(confs):.4f}, max={np.max(confs):.4f})")
        print(f"  Avg phase match:        {np.mean(phases):.4f} (min={np.min(phases):.4f}, max={np.max(phases):.4f})")
        print(f"  Avg green phase:        {np.mean(greens):.4f} (min={np.min(greens):.4f}, max={np.max(greens):.4f})")
        print(f"  Avg ICA score:          {np.mean(icas):.4f} (min={np.min(icas):.4f}, max={np.max(icas):.4f})")
        print(f"  Avg structure ratio:    {np.mean(structs):.4f} (min={np.min(structs):.4f}, max={np.max(structs):.4f})")

    if missed:
        print(f"\n  --- MISSED IMAGES (False Negatives) ---")
        for r in missed:
            print(f"    {r['file'][:55]}")
            print(f"      conf={r['confidence']:.4f}  phase={r['phase_match']:.4f}  "
                  f"green={r['green_phase']:.4f}  ica={r['ica_score']:.4f}  "
                  f"struct={r['structure_ratio']:.4f}")
    else:
        print(f"\n  >>> NO MISSED IMAGES! 100% DETECTION RATE <<<")

    # Load baseline for comparison
    print("\n" + "=" * 70)
    print("  COMPARISON: OLD vs IMPROVED")
    print("=" * 70)

    baseline_path = 'baseline_detection_results.json'
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)

        old_rate = baseline['detection_rate_percent']
        new_rate = detection_rate
        old_detected = baseline['detected']
        new_detected = detected_count
        old_missed = baseline['missed']
        new_missed = len(missed)

        print(f"  {'Metric':<25} {'Old Detector':>15} {'Improved':>15} {'Change':>12}")
        print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")
        print(f"  {'Detection Rate':<25} {old_rate:>14.1f}% {new_rate:>14.1f}% {new_rate - old_rate:>+11.1f}%")
        print(f"  {'Detected':<25} {old_detected:>15} {new_detected:>15} {new_detected - old_detected:>+12}")
        print(f"  {'Missed':<25} {old_missed:>15} {new_missed:>15} {new_missed - old_missed:>+12}")
        print(f"  {'Avg Confidence':<25} {baseline['avg_confidence']:>15.4f} {np.mean(confs):>15.4f}")
        print(f"  {'Avg Phase Match':<25} {baseline['avg_phase_match']:>15.4f} {np.mean(phases):>15.4f}")

        # Show which previously-missed images are now detected
        old_missed_files = set(r['file'] for r in baseline['results'] if not r['is_watermarked'])
        new_missed_files = set(r['file'] for r in results if not r['is_watermarked'])
        newly_detected = old_missed_files - new_missed_files

        if newly_detected:
            print(f"\n  --- Previously missed, NOW DETECTED ({len(newly_detected)} images) ---")
            for f in sorted(newly_detected):
                print(f"    [+] {f[:60]}")

        still_missed = old_missed_files & new_missed_files
        if still_missed:
            print(f"\n  --- Still missed ({len(still_missed)} images) ---")
            for f in sorted(still_missed):
                print(f"    [-] {f[:60]}")

        new_misses = new_missed_files - old_missed_files
        if new_misses:
            print(f"\n  --- NEW regressions ({len(new_misses)} images) ---")
            for f in sorted(new_misses):
                print(f"    [!] {f[:60]}")
    else:
        print("  (No baseline results found for comparison)")

    print("=" * 70)

    # Save results
    report = {
        'test_type': 'improved_detector',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_images': total_valid,
        'detected': detected_count,
        'missed': len(missed),
        'detection_rate_percent': round(detection_rate, 1),
        'elapsed_seconds': round(elapsed, 1),
        'avg_confidence': round(float(np.mean(confs)), 4) if results else 0,
        'avg_phase_match': round(float(np.mean(phases)), 4) if results else 0,
        'improvements': [
            'per-channel FFT (G>R>B weighting)',
            'soft probabilistic decision (no hard thresholds)',
            'ICA watermark pattern as detection signal',
            'adaptive denoiser fusion',
            'green channel dominance bonus',
        ],
        'results': results
    }

    report_path = 'improved_detection_results.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nFull results saved to: {report_path}")

    return report


if __name__ == '__main__':
    run_improved_test()
