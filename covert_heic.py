import os
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

DATA_DIR = "data"

for root, _, files in os.walk(DATA_DIR):
    for name in files:
        if name.lower().endswith(".heic"):
            heic_path = os.path.join(root, name)
            jpg_path = os.path.splitext(heic_path)[0] + ".jpg"
            try:
                Image.open(heic_path).save(jpg_path, format="JPEG")
                print(f"✔ {heic_path} → {jpg_path}")
          
            except Exception as e:
                print(f"❌ Failed {heic_path}: {e}")
