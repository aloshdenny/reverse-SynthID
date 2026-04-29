"""
Best-of-All Removal Pipeline

For each watermarked image, tries MULTIPLE removal strategies and keeps
the one that produces the lowest detector confidence (best removal).

Strategies:
1. V3 Maximum strength (spectral codebook subtraction)
2. V2 Combined worst-case aggressive (stacked multi-category)
3. V2 Combined worst-case maximum (strongest possible)
4. Chained: V3 aggressive -> V2 aggressive (double pass)
5. Nuclear: V3 max + heavy JPEG Q30 + Gaussian noise

Then re-calibrates the improved detector with the best-cleaned set.
"""

import os
import sys
import time
import json
import io
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'extraction'))

from improved_extractor import ImprovedSynthIDExtractor
from synthid_bypass import SynthIDBypass, SpectralCodebook


def nuclear_clean(image_rgb, bypass, codebook):
    """
    Nuclear option: V3 max + JPEG Q30 + WebP Q25 + Gaussian noise.
    Trades quality for maximum watermark destruction.
    """
    # Step 1: V3 maximum
    result = bypass.bypass_v3(image_rgb, codebook=codebook,
                               strength='maximum', verify=False)
    cleaned = result.cleaned_image.copy()

    # Step 2: Heavy JPEG compression
    pil = Image.fromarray(cleaned)
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=30)
    buf.seek(0)
    cleaned = np.array(Image.open(buf).convert('RGB'))

    # Step 3: WebP compression (different transform basis)
    pil2 = Image.fromarray(cleaned)
    buf2 = io.BytesIO()
    pil2.save(buf2, format='WEBP', quality=25)
    buf2.seek(0)
    cleaned = np.array(Image.open(buf2).convert('RGB'))

    # Step 4: Add Gaussian noise
    noise = np.random.normal(0, 8, cleaned.shape).astype(np.float32)
    cleaned = np.clip(cleaned.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Step 5: Light denoise to reduce noise artifacts
    cleaned = cv2.bilateralFilter(cleaned, 5, 30, 30)

    return cleaned


def double_pass(image_rgb, bypass, codebook):
    """
    Double pass: V3 aggressive -> V2 aggressive.
    Two different removal philosophies stacked.
    """
    # Pass 1: V3 spectral subtraction
    r1 = bypass.bypass_v3(image_rgb, codebook=codebook,
                           strength='aggressive', verify=False)
    intermediate = r1.cleaned_image.copy()

    # Pass 2: V2 combined worst-case
    r2 = bypass.bypass_v2(intermediate, strength='aggressive', verify=False)
    return r2.cleaned_image


def run_best_removal():
    codebook_det_path = os.path.join('artifacts', 'codebook', 'robust_codebook.pkl')
    codebook_v3_path = os.path.join('artifacts', 'spectral_codebook_v3.npz')
    input_dir = 'gemini_random'
    output_dir = 'gemini_best_cleaned'

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  BEST-OF-ALL REMOVAL PIPELINE")
    print("  Tries 5 methods per image, keeps the best removal")
    print("=" * 70)

    # Load tools
    print("\nLoading components...")
    detector = ImprovedSynthIDExtractor()
    detector.load_codebook(codebook_det_path)

    spectral_cb = SpectralCodebook()
    spectral_cb.load(codebook_v3_path)

    bypass = SynthIDBypass(extractor=None)

    # Get image files
    extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in extensions
    ])

    print(f"Processing {len(image_files)} images x 5 methods each")
    print("-" * 70)

    all_results = []
    method_wins = {'v3_max': 0, 'v2_agg': 0, 'v2_max': 0, 'double': 0, 'nuclear': 0}
    start_time = time.time()

    for i, fname in enumerate(image_files):
        input_path = os.path.join(input_dir, fname)
        img = cv2.imread(input_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get original detection score
        orig_result = detector.detect_array(img_rgb)
        orig_conf = orig_result.confidence
        orig_phase = orig_result.phase_match

        candidates = {}

        # Method 1: V3 Maximum
        try:
            r = bypass.bypass_v3(img_rgb, codebook=spectral_cb,
                                  strength='maximum', verify=False)
            det = detector.detect_array(r.cleaned_image)
            candidates['v3_max'] = {
                'image': r.cleaned_image, 'conf': det.confidence,
                'phase': det.phase_match, 'psnr': r.psnr
            }
        except Exception as e:
            pass

        # Method 2: V2 Aggressive
        try:
            r = bypass.bypass_v2(img_rgb, strength='aggressive', verify=False)
            det = detector.detect_array(r.cleaned_image)
            candidates['v2_agg'] = {
                'image': r.cleaned_image, 'conf': det.confidence,
                'phase': det.phase_match, 'psnr': r.psnr
            }
        except Exception as e:
            pass

        # Method 3: V2 Maximum
        try:
            r = bypass.bypass_v2(img_rgb, strength='maximum', verify=False)
            det = detector.detect_array(r.cleaned_image)
            candidates['v2_max'] = {
                'image': r.cleaned_image, 'conf': det.confidence,
                'phase': det.phase_match, 'psnr': r.psnr
            }
        except Exception as e:
            pass

        # Method 4: Double pass (V3 -> V2)
        try:
            cleaned = double_pass(img_rgb, bypass, spectral_cb)
            det = detector.detect_array(cleaned)
            # Compute PSNR manually
            mse = np.mean((img_rgb.astype(float) - cleaned.astype(float)) ** 2)
            psnr = float('inf') if mse == 0 else float(10 * np.log10(255**2 / mse))
            candidates['double'] = {
                'image': cleaned, 'conf': det.confidence,
                'phase': det.phase_match, 'psnr': psnr
            }
        except Exception as e:
            pass

        # Method 5: Nuclear
        try:
            cleaned = nuclear_clean(img_rgb, bypass, spectral_cb)
            det = detector.detect_array(cleaned)
            mse = np.mean((img_rgb.astype(float) - cleaned.astype(float)) ** 2)
            psnr = float('inf') if mse == 0 else float(10 * np.log10(255**2 / mse))
            candidates['nuclear'] = {
                'image': cleaned, 'conf': det.confidence,
                'phase': det.phase_match, 'psnr': psnr
            }
        except Exception as e:
            pass

        # Select BEST: lowest confidence = best removal
        if not candidates:
            print(f"  [{i+1:3d}/{len(image_files)}] ALL FAILED for {fname[:40]}")
            continue

        best_method = min(candidates, key=lambda m: candidates[m]['conf'])
        best = candidates[best_method]
        method_wins[best_method] += 1

        # Save best result
        output_path = os.path.join(output_dir, fname)
        cv2.imwrite(output_path, cv2.cvtColor(best['image'], cv2.COLOR_RGB2BGR))

        conf_drop = orig_conf - best['conf']
        phase_drop = orig_phase - best['phase']

        all_results.append({
            'file': fname,
            'orig_conf': round(orig_conf, 4),
            'orig_phase': round(orig_phase, 4),
            'best_method': best_method,
            'best_conf': round(best['conf'], 4),
            'best_phase': round(best['phase'], 4),
            'best_psnr': round(best['psnr'], 1),
            'conf_drop': round(conf_drop, 4),
            'phase_drop': round(phase_drop, 4),
            'all_methods': {m: round(candidates[m]['conf'], 4) for m in candidates},
        })

        print(f"  [{i+1:3d}/{len(image_files)}] BEST={best_method:<10s} "
              f"conf: {orig_conf:.4f}->{best['conf']:.4f} ({conf_drop:+.4f})  "
              f"phase: {orig_phase:.4f}->{best['phase']:.4f} ({phase_drop:+.4f})  "
              f"PSNR={best['psnr']:.1f}dB")

    elapsed = time.time() - start_time

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  REMOVAL RESULTS SUMMARY")
    print("=" * 70)

    orig_confs = [r['orig_conf'] for r in all_results]
    best_confs = [r['best_conf'] for r in all_results]
    orig_phases = [r['orig_phase'] for r in all_results]
    best_phases = [r['best_phase'] for r in all_results]
    psnrs = [r['best_psnr'] for r in all_results]
    conf_drops = [r['conf_drop'] for r in all_results]

    print(f"\n  Images processed: {len(all_results)}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/max(len(all_results),1):.1f}s per image)")

    print(f"\n  {'Metric':<25} {'Original':>12} {'Best Cleaned':>12}  {'Avg Drop':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}  {'-'*10}")
    print(f"  {'Avg Confidence':<25} {np.mean(orig_confs):>12.4f} {np.mean(best_confs):>12.4f}  {np.mean(conf_drops):>+10.4f}")
    print(f"  {'Min Confidence':<25} {np.min(orig_confs):>12.4f} {np.min(best_confs):>12.4f}")
    print(f"  {'Max Confidence':<25} {np.max(orig_confs):>12.4f} {np.max(best_confs):>12.4f}")
    print(f"  {'Avg Phase Match':<25} {np.mean(orig_phases):>12.4f} {np.mean(best_phases):>12.4f}  {np.mean([r['phase_drop'] for r in all_results]):>+10.4f}")
    print(f"  {'Avg PSNR':<25} {'':>12} {np.mean(psnrs):>11.1f}dB")

    # Method wins
    print(f"\n  --- Best Method Distribution ---")
    for method, wins in sorted(method_wins.items(), key=lambda x: -x[1]):
        pct = 100 * wins / len(all_results) if all_results else 0
        bar = '#' * int(pct / 2)
        print(f"    {method:<12s} {wins:>3d} wins ({pct:>5.1f}%)  {bar}")

    # Separation check
    overlap = max(0, min(np.max(best_confs), np.max(orig_confs)) - max(np.min(best_confs), np.min(orig_confs)))
    separable = np.min(orig_confs) > np.max(best_confs)
    print(f"\n  Distributions separable: {'YES!' if separable else 'NO (overlap)'}")
    print(f"  Originals range:  [{np.min(orig_confs):.4f}, {np.max(orig_confs):.4f}]")
    print(f"  Cleaned range:    [{np.min(best_confs):.4f}, {np.max(best_confs):.4f}]")

    # Find optimal threshold
    if all_results:
        best_acc = 0
        best_t = 0.5
        wm = np.array(orig_confs)
        cl = np.array(best_confs)
        for t in np.arange(0.1, 0.95, 0.005):
            tp = np.sum(wm > t)
            tn = np.sum(cl <= t)
            acc = (tp + tn) / (len(wm) + len(cl))
            if acc > best_acc:
                best_acc = acc
                best_t = t
                best_tp, best_tn = tp, tn

        print(f"\n  Optimal threshold: {best_t:.3f}")
        print(f"    TP: {best_tp}/{len(wm)} ({100*best_tp/len(wm):.1f}%)")
        print(f"    TN: {best_tn}/{len(cl)} ({100*best_tn/len(cl):.1f}%)")
        print(f"    Overall accuracy: {100*best_acc:.1f}%")

    print("=" * 70)

    # Save
    with open('best_removal_results.json', 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_images': len(all_results),
            'method_wins': method_wins,
            'avg_conf_drop': round(float(np.mean(conf_drops)), 4),
            'avg_psnr': round(float(np.mean(psnrs)), 1),
            'results': all_results,
        }, f, indent=2)
    print(f"\nResults saved to: best_removal_results.json")


if __name__ == '__main__':
    run_best_removal()
