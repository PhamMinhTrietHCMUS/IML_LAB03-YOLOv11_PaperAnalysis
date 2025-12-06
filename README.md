# YOLO Model Comparison for Rock-Paper-Scissors Detection

This project compares three YOLO models (YOLOv8, YOLOv9, and YOLOv11) for detecting hand gestures in Rock-Paper-Scissors game.

## Dataset

- **Source**: Rock-Paper-Scissors v1i from Roboflow
- **Total Images**: 3,129
  - Train: 2,196 images (70%)
  - Validation: 604 images (20%)
  - Test: 329 images (10%)
- **Classes**: 3 (Paper, Rock, Scissors)

## Installation

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

### Web Application (Recommended)
Start the interactive web interface:
```bash
python app.py
```
Then open http://localhost:5000 to use the app with your webcam.

### Command Line
```bash
# YOLOv11 with webcam
python yolo_detect.py --model runs/detect/yolov11/weights/best.pt --source usb0

# With recording
python yolo_detect.py --model runs/detect/yolov11/weights/best.pt --source usb0 --resolution 640x480 --record
```

## License

This project is for educational purposes.
