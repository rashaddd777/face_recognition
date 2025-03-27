#!/usr/bin/env python3
import pickle
import logging
import time
from pathlib import Path

import numpy as np
import face_recognition
from sklearn.svm import SVC

# CONFIGURATION
TRAIN_DIR = Path("data/public/train")  
MODEL_PATH = Path("face_recognizer.pkl")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )


def load_embeddings(train_dir: Path):
    embeddings = []
    labels = []
    
    if not train_dir.exists() or not train_dir.is_dir():
        logging.error(f"Training directory '{train_dir}' does not exist or is not a directory.")
        return np.array(embeddings), labels

    # Iterate over each person's folder
    for person_dir in train_dir.iterdir():
        if not person_dir.is_dir():
            continue
        # Process each image in the folder
        for img_path in person_dir.glob("*"):
            if img_path.suffix.lower() in ALLOWED_EXTENSIONS:
                try:
                    image = face_recognition.load_image_file(str(img_path))
                    encoding_list = face_recognition.face_encodings(image)
                    if encoding_list:
                        embeddings.append(encoding_list[0])
                        labels.append(person_dir.name)
                    else:
                        logging.warning(f"No face found in image {img_path}.")
                except Exception as e:
                    logging.error(f"Error processing {img_path}: {e}")

    return np.array(embeddings), labels


def train_classifier(embeddings, labels):
    clf = SVC(kernel="linear", probability=True)
    clf.fit(embeddings, labels)
    return clf


def save_model(model, model_path: Path):
    try:
        with open(model_path, "wb") as f:
            pickle.dump({"model": model}, f)
        logging.info(f"Saved trained model to {model_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")


def main():
    setup_logging()
    start_time = time.time()

    logging.info("Loading embeddings and labels...")
    embeddings, labels = load_embeddings(TRAIN_DIR)
    if embeddings.size == 0:
        logging.error("No embeddings found. Exiting.")
        return

    logging.info(f"Collected {len(embeddings)} embeddings from {len(set(labels))} identities.")

    logging.info("Training SVM classifier...")
    clf = train_classifier(embeddings, labels)
    logging.info(f"Model trained on {len(embeddings)} samples.")

    save_model(clf, MODEL_PATH)

    elapsed = time.time() - start_time
    logging.info(f"Training completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
