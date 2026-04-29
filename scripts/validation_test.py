"""
HONEST Validation Test — Testing for Real Legitimacy

Tests the improved detector in 3 ways:
1. POSITIVE TEST: Black/white reference images (watermarked, different content from training)
2. FALSE POSITIVE TEST: Synthetically created clean images (no watermark)
3. FALSE POSITIVE TEST: Downloaded/camera-like images (if available)

This tests whether the detector is ACTUALLY detecting watermarks or just
overfitting to artifacts in the training data.
"""

import os
import sys
import time
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'extraction'))

from improved_extractor import ImprovedSynthIDExtractor
from robust_extractor import RobustSynthIDExtractor


def create_clean_test_images(output_dir='test_clean_images'):
    """Create a set of images that are DEFINITELY not watermarked."""
    os.makedirs(output_dir, exist_ok=True)

    images_created = []

    # 1. Pure solid colors (not from Gemini)
    for name, color in [('pure_red', [255,0,0]), ('pure_green', [0,255,0]),
                         ('pure_blue', [0,0,255]), ('pure_gray', [128,128,128]),
                         ('pure_black_local', [0,0,0]), ('pure_white_local', [255,255,255])]:
        img = np.full((1024, 1024, 3), color, dtype=np.uint8)
        path = os.path.join(output_dir, f'{name}.png')
        cv2.imwrite(path, img)
        images_created.append(path)

    # 2. Gradient images
    for name, axis in [('gradient_h', 1), ('gradient_v', 0)]:
        img = np.zeros((1024, 1024, 3), dtype=np.uint8)
        for i in range(1024):
            val = int(255 * i / 1024)
            if axis == 1:
                img[:, i, :] = val
            else:
                img[i, :, :] = val
        path = os.path.join(output_dir, f'{name}.png')
        cv2.imwrite(path, img)
        images_created.append(path)

    # 3. Random noise (definitely not watermarked)
    for i in range(5):
        img = np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8)
        path = os.path.join(output_dir, f'random_noise_{i}.png')
        cv2.imwrite(path, img)
        images_created.append(path)

    # 4. Checkerboard patterns
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    for y in range(0, 1024, 64):
        for x in range(0, 1024, 64):
            if (y // 64 + x // 64) % 2 == 0:
                img[y:y+64, x:x+64, :] = 255
    path = os.path.join(output_dir, 'checkerboard.png')
    cv2.imwrite(path, img)
    images_created.append(path)

    # 5. Natural-looking synthetic images (smooth gradients with noise)
    for i in range(5):
        # Create a "photo-like" image with smooth regions and edges
        img = np.zeros((1024, 1024, 3), dtype=np.float32)
        # Random smooth blobs
        for _ in range(10):
            cx, cy = np.random.randint(0, 1024, 2)
            radius = np.random.randint(50, 300)
            color = np.random.uniform(0, 1, 3)
            y_grid, x_grid = np.ogrid[:1024, :1024]
            dist = np.sqrt((y_grid - cy)**2 + (x_grid - cx)**2)
            mask = np.exp(-dist**2 / (2 * radius**2))
            for c in range(3):
                img[:, :, c] += mask * color[c]
        img = np.clip(img, 0, 1)
        # Add realistic noise
        img += np.random.normal(0, 0.02, img.shape)
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        path = os.path.join(output_dir, f'synthetic_photo_{i}.png')
        cv2.imwrite(path, img)
        images_created.append(path)

    # 6. Resize/crop from non-watermarked images (JPEG artifacts)
    for i in range(3):
        img = np.random.randint(50, 200, (2048, 2048, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (31, 31), 0)  # Make it smooth
        # JPEG compress to add artifacts
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (1024, 1024))
        path = os.path.join(output_dir, f'jpeg_artifact_{i}.png')
        cv2.imwrite(path, img)
        images_created.append(path)

    print(f"Created {len(images_created)} clean test images in {output_dir}/")
    return images_created


def test_on_images(extractor, image_paths, label, expected_watermarked):
    """Test a set of images and return results."""
    results = []
    detected = 0

    for i, img_path in enumerate(image_paths):
        fname = os.path.basename(img_path)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            result = extractor.detect_array(img_rgb)

            if result.is_watermarked:
                detected += 1

            correct = (result.is_watermarked == expected_watermarked)
            status = "OK" if correct else "WRONG!"

            results.append({
                'file': fname,
                'resolution': f"{w}x{h}",
                'is_watermarked': result.is_watermarked,
                'confidence': round(result.confidence, 4),
                'phase_match': round(result.phase_match, 4),
                'correct': correct,
            })

            mark = "[+]" if result.is_watermarked else "[-]"
            flag = "" if correct else " *** WRONG ***"
            print(f"  [{i+1:3d}/{len(image_paths)}] {mark} conf={result.confidence:.4f}  "
                  f"phase={result.phase_match:.4f}  [{w}x{h}]  {fname[:45]}{flag}")

        except Exception as e:
            print(f"  [{i+1:3d}/{len(image_paths)}] ERROR: {e}")

    return results, detected


def run_validation():
    codebook_path = os.path.join('artifacts', 'codebook', 'robust_codebook.pkl')

    print("=" * 70)
    print("  HONEST VALIDATION TEST")
    print("  Testing both detection AND false positive rate")
    print("=" * 70)

    # Load both extractors for comparison
    print("\nLoading extractors...")
    improved = ImprovedSynthIDExtractor()
    improved.load_codebook(codebook_path)

    old = RobustSynthIDExtractor()
    old.load_codebook(codebook_path)

    all_results = {}

    # ── TEST 1: Black reference images (watermarked, not in training set for detection) ──
    print("\n" + "=" * 70)
    print("  TEST 1: Black Reference Images (watermarked, 1024x1024)")
    print("  These are Gemini-generated but different content/resolution from training")
    print("=" * 70)

    black_dir = 'gemini_black'
    extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    black_images = [os.path.join(black_dir, f) for f in sorted(os.listdir(black_dir))
                    if os.path.splitext(f)[1].lower() in extensions][:20]  # Test 20

    print(f"\n  --- IMPROVED detector on {len(black_images)} black images ---")
    imp_black_results, imp_black_detected = test_on_images(improved, black_images, "black", True)

    print(f"\n  --- OLD detector on {len(black_images)} black images ---")
    old_black_results, old_black_detected = test_on_images(old, black_images, "black", True)

    all_results['black_images'] = {
        'improved': {'detected': imp_black_detected, 'total': len(black_images)},
        'old': {'detected': old_black_detected, 'total': len(black_images)},
    }

    # ── TEST 2: White reference images (watermarked, not in training set for detection) ──
    print("\n" + "=" * 70)
    print("  TEST 2: White Reference Images (watermarked, 1024x1024)")
    print("=" * 70)

    white_dir = 'gemini_white'
    white_images = [os.path.join(white_dir, f) for f in sorted(os.listdir(white_dir))
                    if os.path.splitext(f)[1].lower() in extensions][:20]

    print(f"\n  --- IMPROVED detector on {len(white_images)} white images ---")
    imp_white_results, imp_white_detected = test_on_images(improved, white_images, "white", True)

    print(f"\n  --- OLD detector on {len(white_images)} white images ---")
    old_white_results, old_white_detected = test_on_images(old, white_images, "white", True)

    all_results['white_images'] = {
        'improved': {'detected': imp_white_detected, 'total': len(white_images)},
        'old': {'detected': old_white_detected, 'total': len(white_images)},
    }

    # ── TEST 3: Clean synthetic images (NOT watermarked) ──
    print("\n" + "=" * 70)
    print("  TEST 3: Clean (Non-Watermarked) Images")
    print("  These should NOT be detected. Any detection = FALSE POSITIVE")
    print("=" * 70)

    clean_images = create_clean_test_images()

    print(f"\n  --- IMPROVED detector on {len(clean_images)} clean images ---")
    imp_clean_results, imp_clean_false_pos = test_on_images(improved, clean_images, "clean", False)

    print(f"\n  --- OLD detector on {len(clean_images)} clean images ---")
    old_clean_results, old_clean_false_pos = test_on_images(old, clean_images, "clean", False)

    all_results['clean_images'] = {
        'improved': {'false_positives': imp_clean_false_pos, 'total': len(clean_images)},
        'old': {'false_positives': old_clean_false_pos, 'total': len(clean_images)},
    }

    # ── FINAL SUMMARY ──
    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)

    n_black = len(black_images)
    n_white = len(white_images)
    n_clean = len(clean_images)

    print(f"\n  {'Test':<35} {'Old Detector':>15} {'Improved':>15}")
    print(f"  {'-'*35} {'-'*15} {'-'*15}")

    # Watermarked detection rates
    print(f"  {'Black imgs (watermarked)':<35} {old_black_detected:>3}/{n_black:>3} ({100*old_black_detected/n_black:.0f}%) "
          f"{imp_black_detected:>3}/{n_black:>3} ({100*imp_black_detected/n_black:.0f}%)")
    print(f"  {'White imgs (watermarked)':<35} {old_white_detected:>3}/{n_white:>3} ({100*old_white_detected/n_white:.0f}%) "
          f"{imp_white_detected:>3}/{n_white:>3} ({100*imp_white_detected/n_white:.0f}%)")

    # False positive rates
    print(f"  {'Clean imgs (NOT watermarked)':<35} {old_clean_false_pos:>3}/{n_clean:>3} FP ({100*old_clean_false_pos/n_clean:.0f}%)  "
          f"{imp_clean_false_pos:>3}/{n_clean:>3} FP ({100*imp_clean_false_pos/n_clean:.0f}%)")

    # Overall
    total_wm = n_black + n_white
    old_wm_detected = old_black_detected + old_white_detected
    imp_wm_detected = imp_black_detected + imp_white_detected

    print(f"\n  {'OVERALL True Positive Rate':<35} {100*old_wm_detected/total_wm:>14.1f}% {100*imp_wm_detected/total_wm:>14.1f}%")
    print(f"  {'OVERALL False Positive Rate':<35} {100*old_clean_false_pos/n_clean:>14.1f}% {100*imp_clean_false_pos/n_clean:>14.1f}%")

    is_legit = imp_wm_detected >= total_wm * 0.8 and imp_clean_false_pos <= n_clean * 0.1
    print(f"\n  VERDICT: {'LEGIT - Real watermark detection' if is_legit else 'SUSPICIOUS - May have issues'}")
    print("=" * 70)

    # Save report
    report_path = 'validation_results.json'
    with open(report_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to: {report_path}")


if __name__ == '__main__':
    run_validation()
