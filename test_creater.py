#!/usr/bin/env python3
from pathlib import Path

def create_test_directory():
    test_dir = Path("data/public/test")
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"Directory '{test_dir}' has been created or already exists.")

if __name__ == "__main__":
    create_test_directory()
