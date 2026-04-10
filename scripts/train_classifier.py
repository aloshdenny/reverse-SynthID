"""
Neural Classifier Training Pipeline

Extracts features from all available images, augments the dataset,
trains multiple scikit-learn classifiers, picks the best one,
and integrates it into the final detector.

Features extracted per image (10 dimensions):
  1. confidence         6. multi_scale_consistency
  2. phase_match        7. green_phase
  3. correlation        8. ica_score
  4. structure_ratio    9. green_dominance
  5. carrier_strength   10. high_match_signal
"""

import os
import sys
import io
import time
import json
import pickle
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'extraction'))

from improved_extractor import ImprovedSynthIDExtractor


# ================================================================
# FEATURE EXTRACTION
# ================================================================

def extract_features(extractor, image_rgb):
    """Extract a feature vector from a single image."""
    result = extractor.detect_array(image_rgb)
    return {
        'confidence': result.confidence,
        'phase_match': result.phase_match,
        'correlation': result.correlation,
        'structure_ratio': result.structure_ratio,
        'carrier_strength': result.carrier_strength,
        'multi_scale_consistency': result.multi_scale_consistency,
        'green_phase': result.details.get('green_phase', 0),
        'ica_score': result.details.get('ica_score', 0),
        'green_dominance': result.details.get('green_dominance', 0),
        'high_match_signal': result.details.get('high_match_signal', 0),
        'corr_signal': result.details.get('corr_signal', 0),
        'phase_signal': result.details.get('phase_signal', 0),
        'green_phase_signal': result.details.get('green_phase_signal', 0),
        'struct_signal': result.details.get('struct_signal', 0),
    }


FEATURE_NAMES = [
    'confidence', 'phase_match', 'correlation', 'structure_ratio',
    'carrier_strength', 'multi_scale_consistency', 'green_phase',
    'ica_score', 'green_dominance', 'high_match_signal',
    'corr_signal', 'phase_signal', 'green_phase_signal', 'struct_signal',
]


def features_to_vector(feat_dict):
    """Convert feature dict to numpy array."""
    return np.array([feat_dict[k] for k in FEATURE_NAMES], dtype=np.float64)


def scan_directory(extractor, directory, label, limit=None):
    """Extract features from all images in a directory."""
    extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    files = sorted([
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in extensions
    ])
    if limit:
        files = files[:limit]

    features = []
    labels = []
    filenames = []

    for i, fname in enumerate(files):
        path = os.path.join(directory, fname)
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            feat = extract_features(extractor, img_rgb)
            features.append(features_to_vector(feat))
            labels.append(label)
            filenames.append(fname)

            if (i + 1) % 20 == 0:
                print(f"    [{i+1}/{len(files)}] processed")
        except Exception as e:
            print(f"    [{i+1}/{len(files)}] ERROR: {e}")

    return features, labels, filenames


# ================================================================
# DATA AUGMENTATION
# ================================================================

def augment_image(img_rgb):
    """Create augmented versions of an image for more training data."""
    augmented = []

    # 1. JPEG compression at various qualities
    for q in [50, 70, 85]:
        pil = Image.fromarray(img_rgb)
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        aug = np.array(Image.open(buf).convert('RGB'))
        augmented.append(('jpeg_q{}'.format(q), aug))

    # 2. Resize down and back up
    h, w = img_rgb.shape[:2]
    for scale in [0.5, 0.75]:
        small = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))
        restored = cv2.resize(small, (w, h))
        augmented.append(('resize_{}'.format(scale), restored))

    # 3. Center crop 90% and resize back
    ch, cw = int(h * 0.9), int(w * 0.9)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    cropped = img_rgb[y0:y0+ch, x0:x0+cw]
    cropped = cv2.resize(cropped, (w, h))
    augmented.append(('crop_90', cropped))

    return augmented


def scan_with_augmentation(extractor, directory, label, limit=None, aug_per_image=3):
    """Extract features from images + augmented versions."""
    extensions = {'.png', '.jpg', '.jpeg', '.webp'}
    files = sorted([
        f for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in extensions
    ])
    if limit:
        files = files[:limit]

    features = []
    labels = []

    for i, fname in enumerate(files):
        path = os.path.join(directory, fname)
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Original
            feat = extract_features(extractor, img_rgb)
            features.append(features_to_vector(feat))
            labels.append(label)

            # Augmented (limited for speed)
            if aug_per_image > 0:
                augs = augment_image(img_rgb)
                for j, (aug_name, aug_img) in enumerate(augs[:aug_per_image]):
                    feat = extract_features(extractor, aug_img)
                    features.append(features_to_vector(feat))
                    labels.append(label)

            if (i + 1) % 10 == 0:
                print(f"    [{i+1}/{len(files)}] processed ({1 + min(aug_per_image, 6)} variants each)")
        except Exception as e:
            pass

    return features, labels


# ================================================================
# TRAINING
# ================================================================

def train_classifier():
    codebook_path = os.path.join('artifacts', 'codebook', 'robust_codebook.pkl')

    print("=" * 70)
    print("  CLASSIFIER TRAINING PIPELINE")
    print("=" * 70)

    extractor = ImprovedSynthIDExtractor()
    extractor.load_codebook(codebook_path)

    # ── Collect training data ──
    print("\n  Phase 1: Feature extraction")
    print("  " + "-" * 60)

    all_features = []
    all_labels = []

    # POSITIVE: Watermarked images
    datasets_pos = [
        ('gemini_random', 88, 3),     # 88 images x 4 variants = 352
        ('gemini_black', 50, 2),      # 50 images x 3 variants = 150
        ('gemini_white', 50, 2),      # 50 images x 3 variants = 150
    ]

    for dir_name, limit, aug in datasets_pos:
        if not os.path.exists(dir_name):
            continue
        print(f"\n  [+] Scanning {dir_name} (watermarked, limit={limit}, aug={aug})...")
        feats, lbls = scan_with_augmentation(extractor, dir_name, 1, limit=limit, aug_per_image=aug)
        all_features.extend(feats)
        all_labels.extend(lbls)
        print(f"      -> {len(feats)} samples")

    # NEGATIVE: Clean images
    datasets_neg = [
        ('gemini_best_cleaned', 88, 3),  # 88 bypass-cleaned x 4 = 352
        ('test_clean_images', None, 3),   # 22 synthetic x 4 = 88
    ]

    for dir_name, limit, aug in datasets_neg:
        if not os.path.exists(dir_name):
            continue
        print(f"\n  [-] Scanning {dir_name} (clean, limit={limit}, aug={aug})...")
        feats, lbls = scan_with_augmentation(extractor, dir_name, 0, limit=limit, aug_per_image=aug)
        all_features.extend(feats)
        all_labels.extend(lbls)
        print(f"      -> {len(feats)} samples")

    X = np.array(all_features)
    y = np.array(all_labels)

    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    print(f"\n  Total dataset: {len(X)} samples ({n_pos} watermarked, {n_neg} clean)")
    print(f"  Features: {X.shape[1]} dimensions")

    # ── Train multiple classifiers ──
    print("\n  Phase 2: Training classifiers")
    print("  " + "-" * 60)

    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report, confusion_matrix

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=1000, C=1.0),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        'SVM_RBF': SVC(kernel='rbf', C=10, gamma='scale', probability=True),
        'MLP': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42),
    }

    results = {}
    for name, clf in classifiers.items():
        try:
            scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')
            f1_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='f1')
            results[name] = {
                'accuracy': scores,
                'f1': f1_scores,
                'mean_acc': float(np.mean(scores)),
                'mean_f1': float(np.mean(f1_scores)),
            }
            print(f"    {name:<25s} Acc={np.mean(scores):.3f} (+/-{np.std(scores):.3f})  "
                  f"F1={np.mean(f1_scores):.3f} (+/-{np.std(f1_scores):.3f})")
        except Exception as e:
            print(f"    {name:<25s} ERROR: {e}")

    # ── Pick best model ──
    best_name = max(results, key=lambda k: results[k]['mean_f1'])
    best_clf = classifiers[best_name]
    print(f"\n  Best model: {best_name} (F1={results[best_name]['mean_f1']:.3f})")

    # Train on full dataset
    best_clf.fit(X_scaled, y)

    # Full training set report
    y_pred = best_clf.predict(X_scaled)
    print(f"\n  Full training set performance:")
    print(f"    {classification_report(y, y_pred, target_names=['Clean', 'Watermarked'], digits=3)}")

    cm = confusion_matrix(y, y_pred)
    print(f"    Confusion Matrix:")
    print(f"      TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"      FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    # ── Feature importance ──
    if hasattr(best_clf, 'feature_importances_'):
        importances = best_clf.feature_importances_
        print(f"\n  Feature importance ({best_name}):")
        sorted_idx = np.argsort(importances)[::-1]
        for idx in sorted_idx:
            bar = '#' * int(importances[idx] * 50)
            print(f"    {FEATURE_NAMES[idx]:<28s} {importances[idx]:.3f}  {bar}")
    elif hasattr(best_clf, 'coef_'):
        coefs = np.abs(best_clf.coef_[0])
        print(f"\n  Feature coefficients ({best_name}):")
        sorted_idx = np.argsort(coefs)[::-1]
        for idx in sorted_idx:
            bar = '#' * int(coefs[idx] / coefs.max() * 30)
            print(f"    {FEATURE_NAMES[idx]:<28s} {coefs[idx]:.3f}  {bar}")

    # ── Save model ──
    model_dir = os.path.join('artifacts', 'classifier')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'watermark_classifier.pkl')
    model_data = {
        'classifier': best_clf,
        'scaler': scaler,
        'feature_names': FEATURE_NAMES,
        'model_name': best_name,
        'cv_accuracy': results[best_name]['mean_acc'],
        'cv_f1': results[best_name]['mean_f1'],
        'n_train_pos': int(n_pos),
        'n_train_neg': int(n_neg),
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n  Model saved to: {model_path}")

    # ── Save all CV results ──
    cv_report = {
        'best_model': best_name,
        'dataset_size': len(X),
        'n_positive': int(n_pos),
        'n_negative': int(n_neg),
        'n_features': X.shape[1],
        'feature_names': FEATURE_NAMES,
        'cv_results': {
            name: {
                'mean_accuracy': round(r['mean_acc'], 4),
                'mean_f1': round(r['mean_f1'], 4),
            }
            for name, r in results.items()
        }
    }
    with open('classifier_training_results.json', 'w') as f:
        json.dump(cv_report, f, indent=2)

    return model_data


if __name__ == '__main__':
    train_classifier()
