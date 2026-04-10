"""
Baseline Detection Test — Old (Original) Detector
Tests the existing RobustSynthIDExtractor against all 88 watermarked Gemini images
to establish a baseline detection rate before improvements.
"""

import os
import sys
import time
import json
import numpy as np
import cv2

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'extraction'))

from robust_extractor import RobustSynthIDExtractor

def run_baseline_test():
    # Paths
    codebook_path = os.path.join('artifacts', 'codebook', 'robust_codebook.pkl')
    test_dir = 'gemini_random'  # 88 watermarked Gemini images (ALL should be detected)
    
    print("=" * 70)
    print("  BASELINE DETECTION TEST — Original (Old) Detector")
    print("=" * 70)
    
    # Load extractor
    print(f"\nLoading codebook from: {codebook_path}")
    extractor = RobustSynthIDExtractor()
    extractor.load_codebook(codebook_path)
    print(f"  Codebook loaded. Image size: {extractor.codebook['image_size']}")
    print(f"  Codebook version: {extractor.codebook.get('version', 'unknown')}")
    print(f"  Detection threshold: {extractor.codebook['detection_threshold']:.6f}")
    
    # List test images
    extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    test_images = []
    for fname in sorted(os.listdir(test_dir)):
        if os.path.splitext(fname)[1].lower() in extensions:
            test_images.append(os.path.join(test_dir, fname))
    
    print(f"\nTest images: {len(test_images)} (all are Gemini-generated, ALL should be detected)")
    print("-" * 70)
    
    # Run detection on each image
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
            })
            
            print(f"  [{i+1:3d}/{len(test_images)}] {status}  conf={result.confidence:.4f}  "
                  f"corr={result.correlation:.4f}  phase={result.phase_match:.4f}  "
                  f"struct={result.structure_ratio:.4f}  [{w}x{h}]  {fname[:40]}")
                  
        except Exception as e:
            print(f"  [{i+1:3d}/{len(test_images)}] ERROR {fname[:50]}... -> {e}")
            errors += 1
    
    elapsed = time.time() - start_time
    
    # Summary
    total_valid = len(results)
    missed = [r for r in results if not r['is_watermarked']]
    detection_rate = (detected_count / total_valid * 100) if total_valid > 0 else 0
    
    print("\n" + "=" * 70)
    print("  BASELINE RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total images tested:    {total_valid}")
    print(f"  Correctly detected:     {detected_count}")
    print(f"  Missed (false neg):     {len(missed)}")
    print(f"  Errors:                 {errors}")
    print(f"  Detection rate:         {detection_rate:.1f}%")
    print(f"  Time:                   {elapsed:.1f}s ({elapsed/max(total_valid,1):.2f}s per image)")
    
    if results:
        confs = [r['confidence'] for r in results]
        corrs = [r['correlation'] for r in results]
        phases = [r['phase_match'] for r in results]
        structs = [r['structure_ratio'] for r in results]
        
        print(f"\n  Avg confidence:         {np.mean(confs):.4f} (min={np.min(confs):.4f}, max={np.max(confs):.4f})")
        print(f"  Avg correlation:        {np.mean(corrs):.4f} (min={np.min(corrs):.4f}, max={np.max(corrs):.4f})")
        print(f"  Avg phase match:        {np.mean(phases):.4f} (min={np.min(phases):.4f}, max={np.max(phases):.4f})")
        print(f"  Avg structure ratio:    {np.mean(structs):.4f} (min={np.min(structs):.4f}, max={np.max(structs):.4f})")
    
    if missed:
        print(f"\n  --- MISSED IMAGES (False Negatives) ---")
        for r in missed:
            print(f"    {r['file'][:55]}")
            print(f"      conf={r['confidence']:.4f}  corr={r['correlation']:.4f}  "
                  f"phase={r['phase_match']:.4f}  struct={r['structure_ratio']:.4f}")
    
    print("=" * 70)
    
    # Save results to JSON
    report = {
        'test_type': 'baseline_old_detector',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_images': total_valid,
        'detected': detected_count,
        'missed': len(missed),
        'detection_rate_percent': round(detection_rate, 1),
        'elapsed_seconds': round(elapsed, 1),
        'avg_confidence': round(float(np.mean(confs)), 4) if results else 0,
        'avg_correlation': round(float(np.mean(corrs)), 4) if results else 0,
        'avg_phase_match': round(float(np.mean(phases)), 4) if results else 0,
        'results': results
    }
    
    report_path = 'baseline_detection_results.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nFull results saved to: {report_path}")
    
    return report


if __name__ == '__main__':
    run_baseline_test()
