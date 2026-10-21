# Object Detection in Crime Scene

A local YOLOv8-based object detection project for training on merged Roboflow datasets and running a Streamlit web app on Windows.

## What This Project Does

This project trains a custom object detection model for crime-scene images and serves it in a local web interface. The final model can detect crime-scene classes such as blood, handgun, knife, victim, human body, and any other classes you kept in your merged dataset.

## Final Project Structure

```text
crime_scene_detector/
├── data/
│   └── dataset/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── valid/
│       │   ├── images/
│       │   └── labels/
│       ├── test/
│       │   ├── images/
│       │   └── labels/
│       └── data.yaml
├── models/
│   └── best_crime_scene.pt
├── app.py
├── train.py
├── requirements.txt
├── train_model.bat
├── run_app.bat

```

## Final Model

The app uses only this final trained model:

- [models/best_crime_scene.pt](models/best_crime_scene.pt)

The old `runs/detect/...` folders are training history only and are not needed by the app.

## Optional Roboflow API Mode

The web app can also use a Roboflow workflow as an optional detection mode.

Use this only if you want to compare results or add a cloud-based backup. The local trained model remains the default and recommended mode for your final project.

To use the optional Roboflow mode, provide:

- Roboflow API key
- Workspace name
- Workflow ID
- Class list

The app reads these values from the sidebar, or from environment variables if you set them:

- `ROBOFLOW_API_URL`
- `ROBOFLOW_API_KEY`
- `ROBOFLOW_WORKSPACE`
- `ROBOFLOW_WORKFLOW_ID`
- `ROBOFLOW_CLASSES`

## Dataset

The merged dataset should live in:

- [data/dataset](data/dataset)

This folder must contain:

- `train/images`
- `train/labels`
- `valid/images`
- `valid/labels`
- `test/images` and `test/labels` if you keep a test split
- `data.yaml`

### Label Format

YOLO label files use this format:

```text
class_id x_center y_center width height
```

Example:

```text
2 0.490625 0.50625 0.9796875 0.734375
```

The first number is the class ID. It must match the order in `data.yaml`.

## Current Class Setup

Your merged dataset currently uses 15 classes. The exact class order is defined in `data/dataset/data.yaml` and must match the label IDs used in the `.txt` files.

If you change class IDs, update both the label files and `data.yaml` together.

## Requirements

- Windows
- Python 3.12+ recommended
- NVIDIA GPU recommended for training
- CUDA-enabled PyTorch installed in the project virtual environment

## Install Dependencies

Open a terminal inside `crime_scene_detector` and install the packages in the virtual environment.

In **CMD**:

```cmd
venv\Scripts\activate
pip install -r requirements.txt
```

In **PowerShell**:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train the Model

The training script uses:

- `yolov8s.pt` as the base model
- `imgsz=768`
- `batch=8`
- GPU automatically when CUDA is available

### Recommended Training Command

Run training from the folder that contains `crime_scene_detector`:

```cmd
train_model.bat
```

This batch file runs:

```cmd
python train.py --dataset "data\dataset"
```

### What Training Produces

After training completes, the best model is copied to:

- [models/best_crime_scene.pt](models/best_crime_scene.pt)0

## Run the Web App

Run the app from the folder that contains `crime_scene_detector`:

```cmd
run_app.bat
```

Or run manually:

```cmd
cd /d "crime_scene_detector"
venv\Scripts\activate
python -m streamlit run app.py
```

The app loads the final model from:

- [models/best_crime_scene.pt](models/best_crime_scene.pt)

## Utilities

## Model Selection in the App

The app uses the final local model by default:

- `models/best_crime_scene.pt`

The old run folders are no longer required by the app. If you switch to Roboflow mode, the local model is bypassed and the cloud workflow is used instead.

## Troubleshooting

### The app does not detect anything

- Confirm [models/best_crime_scene.pt](models/best_crime_scene.pt) exists.
- Check that `data/dataset/data.yaml` matches the label IDs.
- Try lowering the confidence threshold in the app.

### Training stops with memory errors

- Keep `cache=False`.
- Keep `workers=0`.
- Lower `batch` from 8 to 4 if needed.
- Lower `imgsz` from 768 to 640 if needed.

### Training is too slow

- Make sure CUDA is working in the virtual environment.
- Confirm the app prints `CUDA:0` during training.
- Close other heavy applications while training.

## Notes

- This project is now organized around one final merged dataset and one final model.
- The important files to keep are the dataset folder, `train.py`, `app.py`, `train_model.bat`, `run_app.bat`, and `models/best_crime_scene.pt`.
