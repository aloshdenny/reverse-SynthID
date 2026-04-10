import os
import sys
import argparse
import cv2

# Add src to path to allow direct execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'extraction'))

from improved_extractor import ImprovedSynthIDExtractor
from synthid_bypass import V3SpectralBypass, SpectralCodebook

def find_codebook_and_model():
    """Finds the required artifact paths."""
    base_dir = os.path.dirname(__file__)
    
    codebook_path = os.path.join(base_dir, 'artifacts', 'codebook', 'robust_codebook.pkl')
    classifier_path = os.path.join(base_dir, 'artifacts', 'classifier', 'watermark_classifier.pkl')
    
    return codebook_path, classifier_path

def detect(args):
    codebook_path, classifier_path = find_codebook_and_model()
    
    if not os.path.exists(codebook_path):
        print(f"[!] Error: Codebook not found at {codebook_path}.")
        return

    print("Loading AI Watermark Detector...")
    extractor = ImprovedSynthIDExtractor()
    extractor.load_codebook(codebook_path)
    
    if os.path.exists(classifier_path):
        extractor.load_classifier(classifier_path)
        print("  [+] Neural classifier loaded (high accuracy mode)")
    else:
        print("  [!] Neural classifier not found, falling back to heuristic scoring.")

    if not os.path.exists(args.image):
        print(f"[!] Error: Target image not found: {args.image}")
        return

    print(f"\nAnalyzing: {args.image}")
    
    try:
        res = extractor.detect(args.image)
        
        print("\n" + "="*40)
        print("  DETECTION RESULT")
        print("="*40)
        
        status = "DETECTED" if res.is_watermarked else "CLEAN"
        color = "\033[91m" if res.is_watermarked else "\033[92m" # Red if detected, Green if clean
        reset = "\033[0m"
        
        print(f"  Status:       {color}{status}{reset}")
        print(f"  Confidence:   {res.confidence * 100:.1f}%")
        print("")
        print("  --- Technical Signals ---")
        print(f"  Phase Match:  {res.phase_match:.3f}")
        print(f"  Struct Ratio: {res.structure_ratio:.2f}")
        print(f"  ICA Score:    {res.details.get('ica_score', 0):.3f}")
        print("="*40)
        
    except Exception as e:
        print(f"[!] Failed to analyze image: {e}")

def bypass(args):
    codebook_path, _ = find_codebook_and_model()
    
    if not os.path.exists(codebook_path):
        print(f"[!] Error: Codebook not found at {codebook_path}.")
        return

    if not os.path.exists(args.input):
        print(f"[!] Error: Input image not found: {args.input}")
        return

    print("Loading V3 Spectral Bypass Module...")
    cb = SpectralCodebook.load(codebook_path)
    bypasser = V3SpectralBypass(cb, mode=args.mode)
    
    print(f"\nProcessing: {args.input}")
    print(f"Mode: {args.mode}")
    print("This may take a moment to perform spectral scattering...")
    
    try:
        img_bgr = cv2.imread(args.input)
        if img_bgr is None:
            print("[!] Could not decode input image format.")
            return
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        cleaned = bypasser.remove(img_rgb)
        
        cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output, cleaned_bgr)
        
        print(f"\n[+] Success! Cleaned image saved to: {args.output}")
        print("    Run 'python detect.py analyze -i <output>' to verify removal.")
        
    except Exception as e:
        print(f"[!] Failed to process image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SynthID Watermark Detector & Bypass CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Analyze Command
    parser_analyze = subparsers.add_parser('analyze', help='Detect watermarks in an image')
    parser_analyze.add_argument('-i', '--image', required=True, help='Path to target image')
    
    # Bypass Command
    parser_bypass = subparsers.add_parser('bypass', help='Attempt to scatter and remove AI watermark')
    parser_bypass.add_argument('-i', '--input', required=True, help='Path to watermarked image')
    parser_bypass.add_argument('-o', '--output', required=True, help='Path to save output image')
    parser_bypass.add_argument('-m', '--mode', choices=['max', 'agg', 'standard', 'nuclear'], default='max', 
                              help='Bypass strength (nuclear combines aggressive + denoising)')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        detect(args)
    elif args.command == 'bypass':
        bypass(args)
