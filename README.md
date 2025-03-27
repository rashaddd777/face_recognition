# Face Recognition Toolkit

This repository provides a simple pipeline for training and testing a face recognition model using the `face_recognition` library and a linear SVM classifier. You can use the provided scripts to train on your own dataset, evaluate accuracy, and make predictions on either entire directories of images or individual photos.

---
## 📂 Repository Structure
```
├── data
│   ├── public
│   │   ├── train/              # Training images organized by identity (folder names like n000006)
│   │   └── test/               # Test images (same structure as train/ for directory-wide testing)
├── face_recognizer.pkl         # Trained SVM model
├── train2.py                   # Train script (produces face_recognizer.pkl)
├── test1.py                    # Tests entire directory (data/public/test by default)
├── test2.py                    # Tests a single image file
├── Classification_Report.txt   # Sample evaluation report (9058 images, 100 classes)
├── confusion_matrix.png        # Sample confusion matrix visualization
└── README.md                   # This document
```

---
## 🚀 Quickstart

### 1. Train your model (train2.py)
By default, this script reads `data/public/train`, extracts face embeddings, and trains a linear SVM. It outputs `face_recognizer.pkl`.
```bash
python train2.py
```

### 2. Test on a directory (test1.py)
Evaluate the saved model on **all** images under `data/public/test`. It prints predictions with confidence scores.
```bash
python test1.py --path data/public/test
```

### 3. Test on a single image (test2.py)
Run inference on one picture.
```bash
python test2.py --path data/public/test/sample.jpg
```

---
## 📊 Model Accuracy
- **Test set size:** 9,058 images across 100 identities
- **Overall accuracy:** 98.44% 

Full classification report is saved in `Classification_Report.txt` and confusion matrix in `confusion_matrix.png`.

---
## 🗂️ Data Format
Each subfolder under `data/public/train` and `data/public/test` represents a unique person (e.g. `n000006`, `n000007`, etc.). All folders contain only JPEG/PNG images of human faces.


Feel free to fork and customize with your own face dataset — just mirror the folder structure and update `--path` arguments when running the scripts.

