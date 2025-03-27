import os

DATA_DIR = "data"

for root, _, files in os.walk(DATA_DIR):
    for name in files:
        if name.lower().endswith(".heic"):
            path = os.path.join(root, name)
            try:
                os.remove(path)
                print(f"Deleted {path}")
            except Exception as e:
                print(f"Couldn’t delete {path}: {e}")
