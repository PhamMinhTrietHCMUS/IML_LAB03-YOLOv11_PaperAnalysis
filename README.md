# YOLO Model Comparison for Rock-Paper-Scissors Detection

This project compares three YOLO models (YOLOv8, YOLOv9, and YOLOv11) for detecting hand gestures in Rock-Paper-Scissors game.

## Dataset

- **Source**: Rock-Paper-Scissors v1i from Roboflow
- **Total Images**: 3,129
  - Train: 2,196 images (70%)
  - Validation: 604 images (20%)
  - Test: 329 images (10%)
- **Classes**: 3 (Paper, Rock, Scissors)

## Quick Start (Windows)

### Easy Setup with Batch Files

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

#### 1.5. GPU Acceleration (Optional - Recommended!) 
If you have NVIDIA GPU, enable GPU for **3x faster performance**:
- Install CUDA Toolkit from NVIDIA website
- Install PyTorch with CUDA support:
```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```
This will:
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
IML_LAB03-YOLOv11_PaperAnalysis/
├── data/
│   ├── train/              # Training dataset (2,196 images)
│   │   ├── images/         # Training images
│   │   ├── labels/         # YOLO format annotations
│   │   └── classes.txt     # Class names
│   └── validation/         # Validation dataset (604 images)
│       ├── images/         # Validation images
│       ├── labels/         # YOLO format annotations
│       └── classes.txt     # Class names
├── rock-paper-scissors.v1i.yolov11/
│   ├── test/               # Test dataset (329 images)
│   │   ├── images/         # Test images
│   │   ├── labels/         # YOLO format annotations
│   │   └── classes.txt     # Class names
│   ├── README.dataset.txt  # Dataset information
│   └── README.roboflow.txt # Roboflow metadata
├── runs/
│   └── detect/
│       ├── yolov8/         # YOLOv8s training results
│       ├── yolov9/         # YOLOv9s training results
│       └── yolov11/        # YOLOv11s training results
├── evaluation_results/
│   ├── YOLOv8/             # YOLOv8 evaluation metrics
│   │   └── predictions.json
│   ├── YOLOv9/             # YOLOv9 evaluation metrics
│   │   └── predictions.json
│   ├── YOLOv11/            # YOLOv11 evaluation metrics
│   │   └── predictions.json
│   └── comparison_results.csv  # Model comparison metrics
├── report/                 # Research report
│   └── main.pdf            # Compiled PDF report
├── templates/
│   └── index.html          # Flask web application UI
├── app.py                  # Flask web application (standard version)
├── app_optimized.py        # Flask web application (optimized with tracking)
├── yolo_detect.py          # Command-line detection script
├── evaluate_models.py      # Model evaluation and comparison script
├── train_val_split.py      # Dataset splitting utility
├── data.yaml               # YOLO dataset configuration
├── requirements.txt        # Python dependencies
├── install.bat             # Windows installer script
├── run.bat                 # Windows application launcher
├── evaluate.bat            # Windows evaluation launcher
├── yolov8s.pt              # YOLOv8s pre-trained weights
├── yolov9s.pt              # YOLOv9s pre-trained weights
├── yolov11s.pt             # YOLOv11s pre-trained weights
├── LICENSE                 # Project license
└── README.md               # This file
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
- Acknowledgements
- Symbol lists and nomenclature
- Summary/Abstract
- Chapter 1: Introduction
- Chapter 2: Literature review
- Chapter 3: Methodology
- Chapter 4: Experimental setup
- Chapter 5: Results and analysis
- Chapter 6: Conclusions and future work
- Bibliography references

The compiled PDF (`main.pdf`) is already included in the repository.

To recompile the report:
```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex  # Run twice more for references

## Pre-trained Models

Pre-trained YOLO weights are included in the root directory:
- `yolov8s.pt` - YOLOv8s base weights
- `yolov9s.pt` - YOLOv9s base weights
- `yolov11s.pt` - YOLOv11s base weights

The fine-tuned models for Rock-Paper-Scissors detection are available in:
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
# Standard version
python app.py

# Or optimized version with smooth tracking
python app_optimized.py
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
- Two versions available:
  - `app.py` - Standard version with basic detection
  - `app_optimized.py` - Enhanced version with smooth tracking and performance optimizations
- Switch between 3 YOLO models (v8, v9, v11)
- Live FPS monitoring
- Object detection statistics
- Responsive web UI with modern design
- Automatic camera testing and error handling

**Performance:**
- **CPU Mode:** 8-15 FPS (optimized settings applied automatically in `app_optimized.py`)
- **GPU Mode:** 25-35 FPS (requires CUDA-enabled PyTorch)
- Both versions automatically detect and utilize GPU if available

**Troubleshooting:**
- If webcam is not detected, check camera permissions in your OS
- If models fail to load, verify model weights exist in `runs/detect/{model}/weights/best.pt`
- For GPU acceleration (3-5x faster), install CUDA Toolkit and PyTorch with CUDA support

### Command Line Interface

For command-line detection without the web interface:

```bash
# YOLOv11 with webcam
python yolo_detect.py --model runs/detect/yolov11/weights/best.pt --source usb0

# With recording
python yolo_detect.py --model runs/detect/yolov11/weights/best.pt --source usb0 --resolution 640x480 --record
```

## License

This project is for educational purposes.
