#!/usr/bin/env python3
import pickle
import logging
from pathlib import Path
import face_recognition
import numpy as np
import argparse

# CONFIGURATION
MODEL_PATH = Path("face_recognizer.pkl")
# Set the default path to your training folder instead of test
DEFAULT_TEST_PATH = Path("data/public/train")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO,
    )


def load_model(model_path: Path):
    """Load the pre-trained SVM model from disk."""
    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        logging.info(f"Loaded model from {model_path}")
        return data["model"]
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None


def test_single_image(model, image_path: Path):
    """Test the model on a single image file."""
    if not image_path.exists() or not image_path.is_file():
        logging.error(f"Test image {image_path} does not exist or is not a file.")
        return

    try:
        image = face_recognition.load_image_file(str(image_path))
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return

    encodings = face_recognition.face_encodings(image)
    if not encodings:
        logging.warning(f"No faces found in {image_path.name}.")
        return

    for i, encoding in enumerate(encodings):
        prediction = model.predict([encoding])
        prediction_proba = model.predict_proba([encoding])
        confidence = np.max(prediction_proba)
        logging.info(
            f"Image: {image_path.name} | Face {i+1}: Predicted as '{prediction[0]}' with confidence {confidence:.2f}"
        )


def test_images(model, test_path: Path):
    """
    Test the model on a given path which can be a file or a directory.
    In this case, test_path is expected to be your training folder.
    """
    if not test_path.exists():
        logging.error(f"Test path {test_path} does not exist.")
        return

    if test_path.is_file():
        if test_path.suffix.lower() in ALLOWED_EXTENSIONS:
            test_single_image(model, test_path)
        else:
            logging.error(f"File {test_path} is not a supported image file.")
    elif test_path.is_dir():
        # Recursively test all images in subfolders (assuming train folder structure)
        for subfolder in test_path.iterdir():
            if subfolder.is_dir():
                image_files = [p for p in subfolder.iterdir() if p.suffix.lower() in ALLOWED_EXTENSIONS]
                if not image_files:
                    logging.warning(f"No supported image files found in {subfolder}.")
                    continue
                for img_path in image_files:
                    test_single_image(model, img_path)
    else:
        logging.error(f"Test path {test_path} is neither a file nor a directory.")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Test a pre-trained face recognition model on images from a folder."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=str(DEFAULT_TEST_PATH),
        help="Path to a test image or directory of test images (default is the train folder)",
    )
    args = parser.parse_args()

    model = load_model(MODEL_PATH)
    if model is None:
        logging.error("Could not load the model. Exiting.")
        return

    test_path = Path(args.path)
    test_images(model, test_path)


if __name__ == "__main__":
    main()
