# YOLO Model Comparison for Rock-Paper-Scissors Detection

This project compares three YOLO models (YOLOv8, YOLOv9, and YOLOv11) for detecting hand gestures in Rock-Paper-Scissors game.

## Dataset

- **Source**: Rock-Paper-Scissors v1i from Roboflow
- **Total Images**: 3,129
  - Train: 2,196 images (70%)
  - Validation: 604 images (20%)
  - Test: 329 images (10%)
- **Classes**: 3 (Paper, Rock, Scissors)

## 🚀 Quick Start (Windows)

### 🚀 Easy Setup with Batch Files

**For Windows users, we provide convenient batch scripts:**

#### 1. First-time Setup
Double-click `install.bat` or run:
```cmd
install.bat
```
This will automatically:
- Check Python installation
- Upgrade pip
- Install all dependencies
- Verify installation

#### 1.5. GPU Acceleration (Recommended!) ⚡
If you have NVIDIA GPU, enable GPU for **3x faster performance**:
```cmd
setup_gpu.bat
```
This will:
- Install CUDA-enabled PyTorch
- Enable GPU acceleration
- Boost FPS from 8-15 to 25-35!

#### 2. Run the Application
Double-click `run.bat` or run:
```cmd
run.bat
```
This will automatically:
- Check dependencies
- Verify model weights
- Start Flask server
- Open browser at http://localhost:5000

#### 3. Run Model Evaluation
Double-click `evaluate.bat` or run:
```cmd
evaluate.bat
```
This will run the evaluation script and open results folder.

---

## Manual Installation

### Requirements

Install the required packages:

```bash
pip install ultralytics
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
# For CUDA 13.0 support:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Project Structure

```
NMMH_LAB3/
├── data/
│   ├── train/           # Training dataset (2,196 images)
│   ├── validation/      # Validation dataset (604 images)
│   └── test/            # Test dataset (329 images)
├── runs/
│   └── detect/
│       ├── yolov8/      # YOLOv8s training results
│       ├── yolov9/      # YOLOv9s training results
│       └── yolov11/     # YOLOv11s training results
├── evaluation_results/  # Model evaluation results and visualizations
├── report/              # LaTeX report files
├── data.yaml            # Dataset configuration
├── evaluate_models.py   # Model evaluation script
└── train_val_split.py   # Dataset splitting script
```

## Training

Models were trained for 100 epochs with the following results:

| Model    | Training Time | F1-Score | Precision | Recall | mAP@50 | mAP@50-95 |
|----------|--------------|----------|-----------|--------|--------|-----------|
| YOLOv8s  | 34.4 min     | 0.9333   | 0.9459    | 0.9069 | 0.9541 | 0.8084    |
| YOLOv9s  | 51.4 min     | 0.9340   | 0.9644    | 0.9144 | 0.9496 | 0.8089    |
| YOLOv11s | 39.2 min     | 0.9360   | 0.9482    | 0.9241 | 0.9548 | 0.8078    |

**Best Model**: YOLOv11s achieved the highest F1-Score (0.9360) with balanced precision and recall.

## Evaluation

Run the evaluation script to compare all models on the test set:

```bash
python evaluate_models.py
```

This will generate:
- Comparison metrics CSV file
- Model comparison visualizations
- Confusion matrices
- Precision-Recall curves
- Prediction examples

## Results

The evaluation results show:
- **YOLOv11s** achieves the best overall performance with F1-Score of 93.60%
- **YOLOv8s** is the fastest to train (34.4 minutes)
- **YOLOv9s** has the highest precision (96.44%)
- All models exceed 93% F1-Score, demonstrating YOLO's effectiveness for gesture recognition

## Report

A comprehensive LaTeX report is available in the `report/` directory, containing:
- Literature review
- Methodology
- Experimental setup
- Results and analysis
- Conclusions and future work

Compile the report:
```bash
cd report
pdflatex main.tex
pdflatex main.tex  # Run twice for references
```

## Pre-trained Models

The trained models are available in:
- `runs/detect/yolov8/weights/best.pt`
- `runs/detect/yolov9/weights/best.pt`
- `runs/detect/yolov11/weights/best.pt`

## Demo Application

### Prerequisites
- Webcam connected to your computer
- Python 3.8 or higher
- All dependencies installed (see Installation section above)

### Web Application (Recommended)

**Quick Start (Windows):**
Simply double-click `run.bat` - it will handle everything automatically!

**Manual Start:**
```bash
python app.py
```

The server will start on `http://localhost:5000`

**Access the web interface:**
1. Open your browser and navigate to `http://localhost:5000`
2. Allow webcam access when prompted
3. Click "Start Detection" to begin real-time detection
4. Switch between YOLOv8, YOLOv9, and YOLOv11 models using the model selector
5. View real-time FPS and detection count

**Features:**
- Real-time object detection via webcam
- Switch between 3 YOLO models (v8, v9, v11)
- Live FPS monitoring
- Object detection statistics
- Responsive web UI with modern design

**Performance:**
- **CPU Mode:** 8-15 FPS (optimized settings applied automatically)
- **GPU Mode:** 25-35 FPS (requires CUDA-enabled PyTorch)
- **See `FPS_OPTIMIZATION.md` for detailed performance guide**

**Troubleshooting:**
- If webcam is not detected, check camera permissions in your OS
- If models fail to load, verify model weights exist in `runs/detect/{model}/weights/best.pt`
- For GPU acceleration (3-5x faster), see `FPS_OPTIMIZATION.md` for CUDA installation

### Command Line Interface

For command-line detection without the web interface:

```bash
# YOLOv11 with webcam
python yolo_detect.py --model runs/detect/yolov11/weights/best.pt --source usb0

# With recording
python yolo_detect.py --model runs/detect/yolov11/weights/best.pt --source usb0 --resolution 640x480 --record
```

## Recent Improvements

### Ease of Use (New! 🎉)
- ✅ **Windows Batch Scripts** for one-click setup and launch
  - `install.bat` - Auto-installs all dependencies
  - `run.bat` - Starts app and opens browser automatically
  - `evaluate.bat` - Runs model evaluation with one click
- ✅ Quick start guide (`QUICKSTART.md`)
- ✅ Batch scripts documentation (`BATCH_SCRIPTS.md`)

### Code Quality Enhancements
- ✅ Enhanced error handling in `app.py` for robust camera and model initialization
- ✅ Improved error messages with troubleshooting tips
- ✅ Added graceful error recovery in frame generation loop
- ✅ Better logging for debugging

### Documentation
- ✅ Comprehensive application demo instructions
- ✅ Video recording guide (`VIDEO_DEMO_GUIDE.md`)
- ✅ Submission checklist (`SUBMISSION_CHECKLIST.md`)
- ✅ Project summary (`PROJECT_SUMMARY.md`)

### Application Features
- ✅ Real-time FPS monitoring
- ✅ Object counting display
- ✅ Smooth model switching between YOLOv8/9/11
- ✅ Responsive web interface
- ✅ Camera testing before streaming

## Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Paper Analysis | ✅ Complete | 18-page report |
| Model Training | ✅ Complete | 3 models, 100 epochs |
| Evaluation | ✅ Complete | Full metrics + visualization |
| Web Application | ✅ Complete | Functional with error handling |
| Documentation | ✅ Complete | Comprehensive guides |
| Video Demo | ⚠️ Pending | See VIDEO_DEMO_GUIDE.md |

**Overall Completion**: 98% (Only video demo remaining)

## License

This project is for educational purposes.
