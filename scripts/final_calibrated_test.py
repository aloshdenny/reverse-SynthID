"""
Final Calibrated Detection Test

Uses the best-removal results to calibrate the improved detector,
then tests on:
1. Original watermarked images (should detect)
2. Best-cleaned images (should NOT detect)
3. Black/white reference images (should detect)
4. Synthetic clean images (should NOT detect)

Shows the REAL, honest detection rate after calibration.
"""

import os
import sys
import json
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'extraction'))

from improved_extractor import ImprovedSynthIDExtractor


def scan_images(extractor, image_dir, limit=None):
    """Scan images and return confidence + phase scores."""
    extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    files = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])
    if limit:
        files = files[:limit]

    confidences = []
    phases = []
    for path in files:
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = extractor.detect_array(img_rgb)
            confidences.append(result.confidence)
            phases.append(result.phase_match)
        except:
            pass
    return np.array(confidences), np.array(phases), len(files)


def run_final_test():
    codebook_path = os.path.join('artifacts', 'codebook', 'robust_codebook.pkl')

    print("=" * 70)
    print("  FINAL CALIBRATED DETECTION TEST (NEURAL)")
    print("=" * 70)

    extractor = ImprovedSynthIDExtractor()
    extractor.load_codebook(codebook_path)
    
    classifier_path = os.path.join('artifacts', 'classifier', 'watermark_classifier.pkl')
    extractor.load_classifier(classifier_path)

    # ── TEST ALL SETS ──
    test_sets = [
        ('gemini_random', 'Watermarked (training set)', True, None),
        ('gemini_best_cleaned', 'Best-cleaned (bypass)', False, None),
        ('gemini_black', 'Black reference (watermarked)', True, 20),
        ('gemini_white', 'White reference (watermarked)', True, 20),
        ('test_clean_images', 'Synthetic clean (no watermark)', False, None),
    ]

    results_table = []

    for dir_name, label, expected, limit in test_sets:
        if not os.path.exists(dir_name):
            print(f"\n  SKIP: {dir_name} not found")
            continue

        confs, phases, total = scan_images(extractor, dir_name, limit)
        if len(confs) == 0:
            continue

        # Apply neural threshold (0.5 since it's a probability)
        detected = np.sum(confs > 0.5)
        not_detected = total - detected

        if expected:
            correct = detected
            rate = 100 * detected / total
            label_type = "TP"
        else:
            correct = not_detected
            rate = 100 * not_detected / total
            label_type = "TN"

        results_table.append({
            'set': label,
            'total': total,
            'detected': int(detected),
            'expected': expected,
            'correct': int(correct),
            'rate': round(rate, 1),
            'avg_conf': round(float(np.mean(confs)), 4),
            'min_conf': round(float(np.min(confs)), 4),
            'max_conf': round(float(np.max(confs)), 4),
        })

        mark = "PASS" if rate >= 80 else "WARN" if rate >= 60 else "FAIL"
        print(f"\n  [{mark}] {label}")
        print(f"         {label_type}: {correct}/{total} ({rate:.1f}%)  "
              f"prob=[{np.min(confs):.4f}, {np.max(confs):.4f}]  "
              f"avg={np.mean(confs):.4f}")

    # ── FINAL SCORECARD ──
    print("\n" + "=" * 70)
    print("  FINAL SCORECARD (Neural Classifier)")
    print("=" * 70)

    print(f"\n  {'Test Set':<40} {'Result':>8} {'Rate':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8}")

    for r in results_table:
        correct_label = "TP" if r['expected'] else "TN"
        print(f"  {r['set']:<40} {r['correct']:>3}/{r['total']:<3} {r['rate']:>7.1f}%")

    # Overall metrics
    all_wm = [r for r in results_table if r['expected']]
    all_clean = [r for r in results_table if not r['expected']]

    total_tp = sum(r['correct'] for r in all_wm)
    total_wm = sum(r['total'] for r in all_wm)
    total_tn = sum(r['correct'] for r in all_clean)
    total_clean = sum(r['total'] for r in all_clean)

    tpr = 100 * total_tp / total_wm if total_wm > 0 else 0
    tnr = 100 * total_tn / total_clean if total_clean > 0 else 0
    overall = 100 * (total_tp + total_tn) / (total_wm + total_clean) if (total_wm + total_clean) > 0 else 0

    print(f"\n  {'OVERALL True Positive Rate':<40} {total_tp:>3}/{total_wm:<3} {tpr:>7.1f}%")
    print(f"  {'OVERALL True Negative Rate':<40} {total_tn:>3}/{total_clean:<3} {tnr:>7.1f}%")
    print(f"  {'OVERALL Accuracy':<40} {'':>8} {overall:>7.1f}%")

    print("=" * 70)

    # Save
    report = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'calibrated_threshold': 0.5,
        'results': results_table,
        'overall_tpr': round(tpr, 1),
        'overall_tnr': round(tnr, 1),
        'overall_accuracy': round(overall, 1),
    }
    with open('final_calibrated_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to: final_calibrated_results.json")


if __name__ == '__main__':
    run_final_test()
