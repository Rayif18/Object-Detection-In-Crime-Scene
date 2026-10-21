"""
Crime Scene Object Detection - Model Training
Uses YOLOv8 for detecting forensic evidence
"""
import shutil
from pathlib import Path

import yaml
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = ROOT / "data" / "dataset"
DEFAULT_MODELS_DIR = ROOT / "models"
DEFAULT_RUNS_DIR = ROOT / "runs" / "detect"
BASE_MODEL_NAME = "yolov8s.pt"
DEFAULT_EPOCHS = 80
DEFAULT_IMGSZ = 768
DEFAULT_BATCH_SIZE = 8


def resolve_device(requested_device: str | None = None) -> str:
    if requested_device:
        return requested_device
    return "0" if torch.cuda.is_available() else "cpu"

def train_model(dataset_path=None, epochs=DEFAULT_EPOCHS, imgsz=DEFAULT_IMGSZ, batch_size=DEFAULT_BATCH_SIZE, device=None):
    """
    Train YOLOv8 model on crime scene dataset
    """
    print("=" * 60)
    print("CRIME SCENE DETECTION - MODEL TRAINING")
    print("=" * 60)
    
    # Check if dataset exists
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_PATH
    else:
        dataset_path = Path(dataset_path)
        if not dataset_path.is_absolute():
            dataset_path = (ROOT / dataset_path).resolve()

    yaml_path = dataset_path / "data.yaml"
    
    if not yaml_path.exists():
        print(f"\n❌ Error: data.yaml not found at {yaml_path}")
        print("\n📝 Please setup your dataset first:")
        print("1. Make sure dataset is in data/dataset/")
        print("\nExpected structure:")
        print("data/dataset/")
        print("├── train/")
        print("│   ├── images/")
        print("│   └── labels/")
        print("├── valid/")
        print("│   ├── images/")
        print("│   └── labels/")
        print("└── data.yaml")
        return None
    
    # Load and display dataset info
    print(f"\n📂 Loading dataset from: {dataset_path}")
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\n📊 Dataset Configuration:")
    print(f"   Classes: {data_config.get('names', [])}")
    
    train_images = sum(1 for item in (dataset_path / "train" / "images").iterdir() if item.is_file())
    valid_images = sum(1 for item in (dataset_path / "valid" / "images").iterdir() if item.is_file())
    print(f"   Training images: {train_images}")
    print(f"   Validation images: {valid_images}")
    
    # Create models directory
    models_dir = DEFAULT_MODELS_DIR
    models_dir.mkdir(exist_ok=True)
    
    # Load pre-trained YOLOv8 model
    print("\n🚀 Loading YOLOv8 model...")
    pretrained_model = ROOT / BASE_MODEL_NAME
    model = YOLO(str(pretrained_model if pretrained_model.exists() else BASE_MODEL_NAME))
    
    # Train the model
    print("\n🎯 Starting training...")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch_size}")
    selected_device = resolve_device(device)
    print(f"   Device: {selected_device}")
    print("\n⏳ This may take 30-60 minutes depending on your hardware...\n")
    
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name='crime_scene_detector',
        project=str(DEFAULT_RUNS_DIR),
        save=True,
        save_period=10,
        patience=30,
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        label_smoothing=0.0,
        nbs=64,
        cache=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        amp=True,
        device=selected_device,
        workers=0,
        seed=42,
        verbose=True
    )
    
    # Get the best model path
    save_dir = Path(model.trainer.save_dir)
    best_model_path = save_dir / "weights" / "best.pt"
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED!")
    print("=" * 60)
    print(f"\n📦 Best model saved to: {best_model_path}")
    print(f"\n📊 Training results saved to: {save_dir}")
    
    # Copy best model to models directory
    if best_model_path.exists():
        final_model_path = models_dir / "best_crime_scene.pt"
        shutil.copy(best_model_path, final_model_path)
        print(f"\n💾 Model copied to: {final_model_path}")
    else:
        print(f"\n⚠️ Best model not found at: {best_model_path}")
    
    return best_model_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Crime Scene Detection Model')
    parser.add_argument('--dataset', type=str, default='data/dataset', 
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, 
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=DEFAULT_IMGSZ, 
                       help='Image size')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH_SIZE, 
                       help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                       help='Training device, for example 0, cpu, or cuda:0')
    
    args = parser.parse_args()
    
    train_model(
        dataset_path=args.dataset,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device
    )