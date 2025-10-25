"""
Script to download and setup YOLOv8 model for drowning detection

Usage:
    - Make sure the virtual environment is activated: source .venv/bin/activate
    - Run from the backend directory: python scripts/setup_model.py
    - Or use the main startup script from root: ../start.sh
"""

import os
from pathlib import Path
from ultralytics import YOLO

def setup_model():
    """Download and setup YOLOv8 model"""
    
    print("=" * 60)
    print("YOLOv8 Model Setup for Drowning Detection")
    print("=" * 60)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("\n[1/3] Downloading YOLOv8n base model...")
    try:
        # Download base YOLOv8n model
        model = YOLO("yolov8n.pt")
        print("✓ Base model downloaded successfully")
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return
    
    print("\n[2/3] Saving model to models directory...")
    try:
        model_path = models_dir / "drowning_detection.pt"
        # Copy the downloaded model to our models directory
        import shutil
        shutil.copy("yolov8n.pt", str(model_path))
        print(f"✓ Model saved to: {model_path}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return
    
    print("\n[3/3] Testing model...")
    try:
        # Test the model
        test_model = YOLO(str(model_path))
        print("✓ Model loaded and tested successfully")
    except Exception as e:
        print(f"✗ Error testing model: {e}")
        return
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nNOTE: This is a base YOLOv8n model.")
    print("For actual drowning detection, you should:")
    print("  1. Collect drowning/swimming dataset")
    print("  2. Train a custom YOLOv8 model")
    print("  3. Replace the model at:", model_path)
    print("\nYou can now start the API server with:")
    print("  python scripts/api_server.py")
    print()

if __name__ == "__main__":
    setup_model()
