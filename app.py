"""
YOLO Real-time Detection Web Application
Supports YOLOv8, YOLOv9, and YOLOv11 models with webcam streaming
"""
import os
import sys
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import time
import torch

app = Flask(__name__)

# Detect device (GPU/CPU)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n{'='*70}")
print(f"🖥️  Device: {DEVICE.upper()}")
if DEVICE == 'cuda':
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"⚡ CUDA Version: {torch.version.cuda}")
else:
    print(f"⚠️  Running on CPU - Performance may be limited")
    print(f"💡 For better FPS, install CUDA-enabled PyTorch")
print(f"{'='*70}\n")

# Global variables
current_model = None
model_name = "YOLOv11"
cap = None
is_running = False

# Performance settings
IMG_SIZE = 416 if DEVICE == 'cpu' else 640  # Smaller size for CPU
CONF_THRESHOLD = 0.5  # Confidence threshold
JPEG_QUALITY = 70  # JPEG encoding quality (70 = good balance of speed/quality)
USE_HALF = DEVICE == 'cuda'  # Use FP16 for faster GPU inference

# Model paths - check multiple possible locations
def find_model_path(model_name):
    """Find model path, checking multiple possible locations"""
    possible_paths = [
        f"runs/detect/{model_name.lower()}/weights/best.pt",  # New structure
        f"runs/detect/train/weights/best.pt",  # Old structure (same for all)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Return first path as default (will show error later)
    return possible_paths[0]

MODELS = {
    "YOLOv8": find_model_path("YOLOv8"),
    "YOLOv9": find_model_path("YOLOv9"),
    "YOLOv11": find_model_path("YOLOv11")
}

# Bounding box colors (Tableau 10 color scheme)
BBOX_COLORS = [
    (164, 120, 87), (68, 148, 228), (93, 97, 209), 
    (178, 182, 133), (88, 159, 106), (96, 202, 231),
    (159, 124, 168), (169, 162, 241), (98, 118, 150), 
    (172, 176, 184)
]

def load_model(model_key):
    """Load YOLO model"""
    global current_model, model_name
    
    if model_key not in MODELS:
        print(f"Error: Unknown model '{model_key}'. Available: {list(MODELS.keys())}")
        return False
    
    model_path = MODELS[model_key]
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        print(f"Please ensure you have trained the model or the weights file exists.")
        return False
    
    try:
        current_model = YOLO(model_path, task='detect')
        # Move model to appropriate device
        current_model.to(DEVICE)
        # Use half precision for faster GPU inference
        if USE_HALF:
            current_model.model.half()
            print(f"🚀 FP16 (half precision) enabled for faster inference")
        model_name = model_key
        print(f"Successfully loaded {model_key} model from {model_path}")
        print(f"Model running on: {DEVICE.upper()}")
        return True
    except Exception as e:
        print(f"Error loading model {model_key}: {str(e)}")
        print(f"This might be due to corrupted weights or incompatible YOLO version.")
        return False

def initialize_camera(camera_id=0):
    """Initialize webcam"""
    global cap
    
    try:
        if cap is not None:
            cap.release()
            time.sleep(0.2)  # Give time for camera to release
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            print(f"Troubleshooting:")
            print(f"  - Check if camera is connected")
            print(f"  - Check camera permissions in OS settings")
            print(f"  - Try a different camera_id (0, 1, 2...)")
            print(f"  - Close other applications using the camera")
            return False
        
        # Set camera properties (lower resolution for CPU)
        width = 640 if DEVICE == 'cuda' else 480
        height = 480 if DEVICE == 'cuda' else 360
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        print(f"Camera resolution: {width}x{height}")
        
        # Test read a frame
        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print(f"Error: Camera opened but cannot read frames")
            cap.release()
            cap = None
            return False
        
        print(f"Camera {camera_id} initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing camera: {str(e)}")
        if cap is not None:
            cap.release()
            cap = None
        return False

def generate_frames():
    """Generate video frames with YOLO detection"""
    global cap, current_model, is_running
    
    print(f"[GENERATE] Starting frame generation. is_running={is_running}, cap={cap is not None}, model={current_model is not None}")
    
    frame_count = 0
    fps_buffer = []
    fps_avg_len = 15  # Smaller buffer for faster FPS update
    
    while is_running:
        try:
            if cap is None or current_model is None:
                print(f"[GENERATE] Missing resources - cap: {cap is not None}, model: {current_model is not None}")
                break
            
            start_time = time.perf_counter()
            
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[GENERATE] Failed to read frame, retrying...")
                time.sleep(0.1)
                continue
            
            # Run YOLO detection with optimizations
            results = current_model(frame, imgsz=IMG_SIZE, verbose=False, device=DEVICE, half=USE_HALF)
            detections = results[0].boxes
            labels = current_model.names
        except Exception as e:
            print(f"[GENERATE] Error during frame processing: {str(e)}")
            time.sleep(0.1)
            continue
        
        # Count objects
        object_count = 0
        
        # Draw detections
        for i in range(len(detections)):
            # Get bounding box coordinates
            xyxy_tensor = detections[i].xyxy.cpu()
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            
            # Get class and confidence
            classidx = int(detections[i].cls.item())
            classname = labels[classidx]
            conf = detections[i].conf.item()
            
            # Draw if confidence > threshold
            if conf > CONF_THRESHOLD:
                color = BBOX_COLORS[classidx % len(BBOX_COLORS)]
                
                # Draw bounding box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                
                # Prepare label
                label = f'{classname}: {int(conf*100)}%'
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, label_size[1] + 10)
                
                # Draw label background
                cv2.rectangle(frame, 
                            (xmin, label_ymin - label_size[1] - 10),
                            (xmin + label_size[0], label_ymin + baseline - 10),
                            color, cv2.FILLED)
                
                # Draw label text
                cv2.putText(frame, label, (xmin, label_ymin - 7),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                object_count += 1
        
        # Calculate FPS
        end_time = time.perf_counter()
        frame_time = end_time - start_time
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        
        # Update FPS buffer
        fps_buffer.append(current_fps)
        if len(fps_buffer) > fps_avg_len:
            fps_buffer.pop(0)
        avg_fps = np.mean(fps_buffer)
        
        # Draw info on frame
        cv2.putText(frame, f'Model: {model_name}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'FPS: {avg_fps:.1f}', (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f'Objects: {object_count}', (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Encode frame with optimized quality for faster streaming
        try:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            ret, buffer = cv2.imencode('.jpg', frame, encode_param)
            if not ret:
                print("[GENERATE] Failed to encode frame")
                continue
            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            frame_count += 1
        except Exception as e:
            print(f"[GENERATE] Error encoding frame: {str(e)}")
            continue

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', models=list(MODELS.keys()))

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    response = Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    # Add headers to reduce browser overhead
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/start', methods=['POST'])
def start_detection():
    """Start detection with selected model"""
    global is_running, cap
    
    try:
        data = request.get_json()
        selected_model = data.get('model', 'YOLOv11')
        camera_id = data.get('camera', 0)
        
        print(f"[START] Received request - Model: {selected_model}, Camera: {camera_id}, is_running: {is_running}")
        
        # Stop previous detection if running
        if is_running:
            print("[START] Stopping previous detection...")
            is_running = False
            if cap is not None:
                cap.release()
                cap = None
            time.sleep(0.5)  # Give time to release resources
        
        # Load model
        print(f"[START] Loading model: {selected_model}")
        if not load_model(selected_model):
            return jsonify({
                'status': 'error', 
                'message': f'Failed to load {selected_model}. Check console for details.'
            })
        
        # Initialize camera
        print(f"[START] Initializing camera: {camera_id}")
        if not initialize_camera(camera_id):
            return jsonify({
                'status': 'error', 
                'message': 'Failed to open camera. Check permissions and connection.'
            })
        
        is_running = True
        print(f"[START] Detection started successfully with {selected_model}")
        return jsonify({'status': 'success', 'message': f'Started {selected_model}'})
        
    except Exception as e:
        print(f"[START] Unexpected error: {str(e)}")
        is_running = False
        return jsonify({
            'status': 'error',
            'message': f'Unexpected error: {str(e)}'
        })

@app.route('/stop', methods=['POST'])
def stop_detection():
    """Stop detection"""
    global is_running, cap
    
    is_running = False
    
    if cap is not None:
        cap.release()
        cap = None
    
    return jsonify({'status': 'success', 'message': 'Detection stopped'})

@app.route('/status')
def get_status():
    """Get current status"""
    return jsonify({
        'is_running': is_running,
        'model': model_name,
        'available_models': list(MODELS.keys())
    })

if __name__ == '__main__':
    # Load default model
    load_model('YOLOv11')
    
    print("=" * 70)
    print("YOLO Real-time Detection Web Application")
    print("=" * 70)
    print(f"Device: {DEVICE.upper()}")
    if DEVICE == 'cpu':
        print(f"⚠️  CPU Mode - Expected FPS: 5-15")
        print(f"💡 For better performance (25-30 FPS), use GPU")
    else:
        print(f"⚡ GPU Mode - Expected FPS: 25-35")
    print(f"Image Size: {IMG_SIZE}px")
    print("=" * 70)
    print("Available models:")
    for model_key, model_path in MODELS.items():
        exists = "✓" if os.path.exists(model_path) else "✗"
        print(f"  {exists} {model_key}: {model_path}")
    print("=" * 70)
    print("Starting server at http://localhost:5000")
    print("=" * 70)
    
    # Use threaded=True and disable reloader to prevent multiple instances
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
