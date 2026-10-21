@echo off
setlocal
cd /d "%~dp0crime_scene_detector"
call venv\Scripts\activate.bat
python train.py --dataset "data\dataset"
