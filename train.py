import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Enable UTF-8 encoding for standard output and error on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models

from features import normalize_sequence
from labels import CLASSES, LABELS

# Ensure working directory is the script root
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
SEQ_LEN = 30
FEATURE_DIM = 126
NORMALIZED_FEATURE_DIM = 132


def load_dataset():
    """Load and validate all recorded .npy sequences from the data directory."""
    X, y, signers = [], [], []
    class_counts = {}

    print("\n--- Scanning Dataset Directory ---")
    for ci, cls in enumerate(CLASSES):
        cls_dir = DATA_DIR / cls
        files = sorted(cls_dir.glob("*.npy")) if cls_dir.exists() else []
        class_counts[cls] = 0

        for file_path in files:
            try:
                seq = np.load(file_path)
                if seq.shape != (SEQ_LEN, FEATURE_DIM):
                    print(
                        f"Warning: Skipping '{file_path.name}' due to invalid shape "
                        f"{seq.shape} (expected ({SEQ_LEN}, {FEATURE_DIM}))."
                    )
                    continue

                norm_seq = normalize_sequence(seq)
                X.append(norm_seq)
                y.append(ci)
                signer = file_path.stem.split("_")[0]  # e.g. "s01"
                signers.append(signer)
                class_counts[cls] += 1
            except Exception as e:
                print(f"Warning: Failed to load '{file_path.name}': {e}. Skipping.")

    print("\nDataset Summary by Class:")
    for cls, count in class_counts.items():
        bengali = LABELS.get(cls, "")
        status = f"{count} samples" if count > 0 else "0 samples (empty)"
        print(f"  - {cls:<12} ({bengali}): {status}")

    if not X:
        raise RuntimeError(
            f"No valid training samples found in '{DATA_DIR}'.\n"
            "Please run 'python collect_data.py' to collect samples first."
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    signers = np.asarray(signers)

    print(f"\nTotal loaded samples: {len(X)} | Feature shape: {X.shape[1:]}")
    print(f"Unique signers: {sorted(set(signers))}")
    return X, y, signers


def split_dataset(X, y, signers):
    """Split data into train, val, and test partitions with resilient fallback."""
    uniq_signers = sorted(set(signers))

    # Signer-independent split if at least 3 distinct signers are available
    if len(uniq_signers) >= 3:
        test_s, val_s = uniq_signers[-1], uniq_signers[-2]
        te = signers == test_s
        va = signers == val_s
        tr = ~(te | va)

        if np.sum(tr) > 0 and np.sum(va) > 0 and np.sum(te) > 0:
            print(f"\nUsing Signer-Independent Split:")
            print(f"  - Test signer: '{test_s}' ({np.sum(te)} samples)")
            print(f"  - Validation signer: '{val_s}' ({np.sum(va)} samples)")
            print(f"  - Training signers: {sorted(set(signers[tr]))} ({np.sum(tr)} samples)")
            return X[tr], y[tr], X[va], y[va], X[te], y[te]
        else:
            print("\nWarning: Signer partition left one set empty. Falling back to random split.")

    # Random split
    print("\nUsing Random Stratified/Proportional Split:")
    n_samples = len(X)
    if n_samples < 3:
        raise RuntimeError(
            f"At least 3 total samples are required to create train/val/test splits (found {n_samples})."
        )

    # For very small datasets (3-5 samples)
    if n_samples < 6:
        perm = np.random.RandomState(42).permutation(n_samples)
        te_idx, va_idx, tr_idx = perm[:1], perm[1:2], perm[2:]
        return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], X[te_idx], y[te_idx]

    # Check if stratification is safe
    n_unique_classes = len(np.unique(y))
    class_counts = np.bincount(y)
    present_counts = [class_counts[c] for c in np.unique(y)]
    min_count = min(present_counts) if present_counts else 0

    test_len = max(1, int(round(n_samples * 0.3)))
    train_len = n_samples - test_len
    can_stratify_1 = (
        min_count >= 2
        and test_len >= n_unique_classes
        and train_len >= n_unique_classes
    )

    Xtr, Xtmp, ytr, ytmp = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y if can_stratify_1 else None,
        random_state=42,
    )

    n_tmp = len(Xtmp)
    n_unique_tmp = len(np.unique(ytmp))
    tmp_counts = np.bincount(ytmp)
    present_tmp = [tmp_counts[c] for c in np.unique(ytmp)] if n_tmp > 0 else []
    min_tmp = min(present_tmp) if present_tmp else 0

    tmp_test_len = max(1, int(round(n_tmp * 0.5)))
    tmp_train_len = n_tmp - tmp_test_len
    can_stratify_2 = (
        n_unique_tmp > 1
        and min_tmp >= 2
        and tmp_test_len >= n_unique_tmp
        and tmp_train_len >= n_unique_tmp
    )

    Xva, Xte, yva, yte = train_test_split(
        Xtmp,
        ytmp,
        test_size=0.5,
        stratify=ytmp if can_stratify_2 else None,
        random_state=42,
    )

    print(f"  - Train: {len(Xtr)} samples")
    print(f"  - Validation: {len(Xva)} samples")
    print(f"  - Test: {len(Xte)} samples")
    return Xtr, ytr, Xva, yva, Xte, yte


def augment_data(Xs: np.ndarray, ys: np.ndarray, copies: int = 3):
    """Apply slight Gaussian jitter to landmarks for training set augmentation."""
    if len(Xs) == 0:
        return Xs, ys

    Xa, ya = [Xs], [ys]
    rng = np.random.RandomState(42)
    for _ in range(copies):
        noise = rng.normal(0.0, 0.01, size=Xs.shape).astype(np.float32)
        Xa.append(Xs + noise)
        ya.append(ys)

    return np.concatenate(Xa, axis=0), np.concatenate(ya, axis=0)


def build_lstm_model(input_shape, num_classes):
    """Construct an LSTM classifier for sign landmark sequences."""
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(64),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    X, y, signers = load_dataset()
    Xtr, ytr, Xva, yva, Xte, yte = split_dataset(X, y, signers)

    print(f"\nAugmenting training set (original {len(Xtr)} samples)...")
    Xtr_aug, ytr_aug = augment_data(Xtr, ytr, copies=3)
    print(f"Augmented training set size: {len(Xtr_aug)} samples.")

    # ---------- Baseline: Random Forest ----------
    print("\n==============================")
    print("  Training Random Forest Baseline")
    print("==============================")
    rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    rf.fit(Xtr_aug.reshape(len(Xtr_aug), -1), ytr_aug)
    rf_pred = rf.predict(Xte.reshape(len(Xte), -1))
    rf_acc = float(accuracy_score(yte, rf_pred))
    print(f"Random Forest Test Accuracy: {rf_acc * 100:.2f}%")

    # ---------- LSTM Training ----------
    print("\n==============================")
    print("  Training LSTM Neural Network")
    print("==============================")
    num_classes = len(CLASSES)
    model = build_lstm_model(input_shape=Xtr.shape[1:], num_classes=num_classes)

    callbacks_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    batch_size = min(16, max(2, len(Xtr_aug) // 4))
    epochs = 100

    model.fit(
        Xtr_aug,
        ytr_aug,
        validation_data=(Xva, yva),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=1,
    )

    # ---------- Evaluation ----------
    print("\n==============================")
    print("  Evaluating LSTM on Held-out Test Set")
    print("==============================")
    probs = model.predict(Xte)
    pred = probs.argmax(axis=1)
    lstm_acc = float(accuracy_score(yte, pred))
    print(f"LSTM Test Accuracy: {lstm_acc * 100:.2f}%\n")

    print("Detailed Classification Report:")
    print(
        classification_report(
            yte,
            pred,
            labels=range(num_classes),
            target_names=CLASSES,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(yte, pred, labels=range(num_classes)))

    # ---------- Save Models and Configuration ----------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    keras_model_path = MODEL_DIR / "isl_lstm.keras"
    model.save(keras_model_path)
    print(f"\nSaved LSTM model to: {keras_model_path}")

    rf_model_path = MODEL_DIR / "isl_rf.joblib"
    joblib.dump(rf, rf_model_path)
    print(f"Saved Random Forest baseline to: {rf_model_path}")

    metadata = {
        "classes": CLASSES,
        "labels": LABELS,
        "sequence_length": SEQ_LEN,
        "feature_dim": NORMALIZED_FEATURE_DIM,
        "num_classes": num_classes,
        "rf_test_accuracy": rf_acc,
        "lstm_test_accuracy": lstm_acc,
        "trained_at": datetime.now().isoformat(),
    }
    config_path = MODEL_DIR / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved model metadata to: {config_path}")
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
