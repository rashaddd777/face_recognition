#!/usr/bin/env python3
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import face_recognition
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# CONFIGURATION
MODEL_PATH = Path("face_recognizer.pkl")
DEFAULT_TEST_DIR = Path("data/public/train")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )


def load_model(path: Path):
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        logging.info(f"Loaded model from {path}")
        return data["model"]
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None


def load_test_data(test_dir: Path):
    X, y = [], []
    if not test_dir.exists() or not test_dir.is_dir():
        logging.error(f"Test directory {test_dir} not found.")
        return np.array(X), y

    for class_dir in test_dir.iterdir():
        if not class_dir.is_dir():
            continue
        for img in class_dir.glob('*'):
            if img.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            try:
                image = face_recognition.load_image_file(str(img))
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    X.append(encodings[0])
                    y.append(class_dir.name)
                else:
                    logging.warning(f"No face found in {img.name}")
            except Exception as e:
                logging.error(f"Error processing {img}: {e}")
    return np.array(X), y


def plot_confusion(cm, classes, out_path: Path):
    fig, ax = plt.subplots()
    cax = ax.imshow(cm, interpolation='nearest')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(out_path)
    logging.info(f"Saved confusion matrix to {out_path}")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Evaluate face recognition model.")
    parser.add_argument("--test-dir", type=str, default=str(DEFAULT_TEST_DIR), help="Path to test data (subfolders by label)")
    parser.add_argument("--output", type=str, default="confusion_matrix.png", help="Output path for confusion matrix image")
    args = parser.parse_args()

    model = load_model(MODEL_PATH)
    if model is None:
        return

    test_dir = Path(args.test_dir)
    X_test, y_test = load_test_data(test_dir)
    if X_test.size == 0:
        logging.error("No test data found.")
        return

    logging.info(f"Testing on {len(X_test)} samples from {len(set(y_test))} classes.")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
    plot_confusion(cm, sorted(set(y_test)), Path(args.output))


if __name__ == "__main__":
    main()
