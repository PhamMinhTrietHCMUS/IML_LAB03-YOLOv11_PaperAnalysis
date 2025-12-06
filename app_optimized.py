"""
YOLO Real-time Detection - Optimized Flask App with Smooth Tracking
Performance improvements + Stable bounding boxes
"""
import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from ultralytics import YOLO
import time
import torch
from threading import Thread, Lock
from collections import deque, defaultdict

app = Flask(__name__)

# Device detection
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n{'='*70}")
print(f"🖥️  Device: {DEVICE.upper()}")
if DEVICE == 'cuda':
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
print(f"{'='*70}\n")

# Global variables
current_model = None
model_name = "YOLOv11"
is_running = False

# Performance settings - OPTIMIZED & STABLE
IMG_SIZE = 480
CONF_THRESHOLD = 0.45  # Slightly lower for stability
IOU_THRESHOLD = 0.45   # For NMS
JPEG_QUALITY = 75
USE_HALF = DEVICE == 'cuda'
FRAME_SKIP = 1  # Process every frame for smooth tracking (changed from 2)

# Smoothing settings
SMOOTH_FACTOR = 0.7  # Higher = smoother but slower response (0-1)
MIN_DETECTION_FRAMES = 2  # Minimum frames before showing detection

# Thread-safe frame buffer
frame_lock = Lock()
output_frame = None
fps_value = 0.0
object_count = 0

# Detection tracker for smoothing
class DetectionTracker:
    """Track and smooth detections across frames"""
    def __init__(self, smooth_factor=0.7, min_frames=2):
        self.smooth_factor = smooth_factor
        self.min_frames = min_frames
        self.tracked_objects = {}  # id: {bbox, class, conf, frames_seen}
        self.next_id = 0
        self.max_disappeared = 5  # Remove after N frames not seen
        
    def update(self, detections):
        """Update tracked objects with new detections"""
        current_objects = {}
        
        if len(detections) == 0:
            # Age out old detections
            for obj_id in list(self.tracked_objects.keys()):
                self.tracked_objects[obj_id]['disappeared'] += 1
                if self.tracked_objects[obj_id]['disappeared'] > self.max_disappeared:
                    del self.tracked_objects[obj_id]
            return []
        
        # Match new detections with existing tracks
        for det in detections:
            bbox = det['bbox']
            class_id = det['class_id']
            conf = det['conf']
            
            # Find best matching tracked object (by IoU)
            best_match_id = None
            best_iou = 0.3  # Minimum IoU threshold
            
            for obj_id, tracked in self.tracked_objects.items():
                iou = self._calculate_iou(bbox, tracked['bbox'])
                if iou > best_iou and tracked['class_id'] == class_id:
                    best_iou = iou
                    best_match_id = obj_id
            
            if best_match_id is not None:
                # Update existing track with smoothing
                tracked = self.tracked_objects[best_match_id]
                smoothed_bbox = self._smooth_bbox(tracked['bbox'], bbox)
                
                tracked['bbox'] = smoothed_bbox
                tracked['conf'] = conf
                tracked['frames_seen'] += 1
                tracked['disappeared'] = 0
                current_objects[best_match_id] = tracked
            else:
                # New detection
                new_id = self.next_id
                self.next_id += 1
                current_objects[new_id] = {
                    'bbox': bbox,
                    'class_id': class_id,
                    'conf': conf,
                    'frames_seen': 1,
                    'disappeared': 0
                }
        
        # Update tracked objects
        self.tracked_objects = current_objects
        
        # Return only stable detections
        stable_detections = []
        for obj_id, obj in self.tracked_objects.items():
            if obj['frames_seen'] >= self.min_frames:
                stable_detections.append(obj)
        
        return stable_detections
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _smooth_bbox(self, old_bbox, new_bbox):
        """Smooth bounding box transition"""
        smoothed = []
        for old, new in zip(old_bbox, new_bbox):
            smoothed.append(
                int(old * self.smooth_factor + new * (1 - self.smooth_factor))
            )
        return smoothed
    
    def reset(self):
        """Reset tracker"""
        self.tracked_objects = {}
        self.next_id = 0

# Global tracker
detection_tracker = DetectionTracker(
    smooth_factor=SMOOTH_FACTOR,
    min_frames=MIN_DETECTION_FRAMES
)

class CameraThread:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.running = False
        self.thread = None
        self.frame_queue = deque(maxlen=2)
        
    def start(self):
        if self.running:
            return True
            
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            return False
            
        # Optimize camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.running = True
        self.thread = Thread(target=self._capture_frames, daemon=True)
        self.thread.start()
        return True
    
    def _capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_queue.append(frame)
            time.sleep(0.01)
    
    def read(self):
        if len(self.frame_queue) > 0:
            return True, self.frame_queue[-1]
        return False, None
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.frame_queue.clear()

camera_thread = CameraThread()

def find_model_path(model_name):
    possible_paths = [
        f"runs/detect/{model_name.lower()}/weights/best.pt",
        f"runs/detect/train/weights/best.pt",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return possible_paths[0]

MODELS = {
    "YOLOv8": find_model_path("YOLOv8"),
    "YOLOv9": find_model_path("YOLOv9"),
    "YOLOv11": find_model_path("YOLOv11")
}

BBOX_COLORS = [
    (164, 120, 87), (68, 148, 228), (93, 97, 209), 
    (178, 182, 133), (88, 159, 106), (96, 202, 231),
    (159, 124, 168), (169, 162, 241), (98, 118, 150), 
    (172, 176, 184)
]

def load_model(model_key):
    global current_model, model_name
    
    if model_key not in MODELS:
        return False
    
    model_path = MODELS[model_key]
    if not os.path.exists(model_path):
        return False
    
    try:
        current_model = YOLO(model_path, task='detect')
        current_model.to(DEVICE)
        if USE_HALF:
            current_model.model.half()
        model_name = model_key
        
        # Reset tracker when changing models
        detection_tracker.reset()
        
        print(f"✅ Loaded {model_key} on {DEVICE.upper()}")
        return True
    except Exception as e:
        print(f"❌ Error loading {model_key}: {e}")
        return False

def inference_thread():
    global output_frame, fps_value, object_count, is_running
    
    frame_count = 0
    fps_buffer = deque(maxlen=10)
    
    while True:
        if not is_running or current_model is None:
            time.sleep(0.1)
            continue
        
        start_time = time.perf_counter()
        
        ret, frame = camera_thread.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
        
        # Process every frame (no skipping for smooth tracking)
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue
        
        try:
            # YOLO inference with NMS
            results = current_model(
                frame, 
                imgsz=IMG_SIZE, 
                verbose=False, 
                device=DEVICE, 
                half=USE_HALF,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD  # NMS threshold
            )
            detections = results[0].boxes
            labels = current_model.names
            
            # Convert to tracker format
            raw_detections = []
            for i in range(len(detections)):
                xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                xmin, ymin, xmax, ymax = xyxy
                
                classidx = int(detections[i].cls.item())
                conf = detections[i].conf.item()
                
                if conf > CONF_THRESHOLD:
                    raw_detections.append({
                        'bbox': [xmin, ymin, xmax, ymax],
                        'class_id': classidx,
                        'conf': conf
                    })
            
            # Update tracker (smooth detections)
            stable_detections = detection_tracker.update(raw_detections)
            obj_count = len(stable_detections)
            
            # Draw stable detections
            for det in stable_detections:
                xmin, ymin, xmax, ymax = det['bbox']
                classidx = det['class_id']
                conf = det['conf']
                classname = labels[classidx]
                
                color = BBOX_COLORS[classidx % len(BBOX_COLORS)]
                
                # Draw bbox with thicker lines for stability
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
                
                # Draw label with background
                label = f'{classname}: {int(conf*100)}%'
                label_size, baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                label_ymin = max(ymin, label_size[1] + 10)
                
                cv2.rectangle(
                    frame,
                    (xmin, label_ymin - label_size[1] - 10),
                    (xmin + label_size[0], label_ymin + baseline - 10),
                    color, cv2.FILLED
                )
                cv2.putText(
                    frame, label, (xmin, label_ymin - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
            
            # Calculate FPS
            end_time = time.perf_counter()
            frame_time = end_time - start_time
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_buffer.append(current_fps)
            avg_fps = np.mean(fps_buffer)
            
            # Draw info overlay
            info_bg = np.zeros((120, 300, 3), dtype=np.uint8)
            info_bg[:] = (40, 40, 40)
            cv2.putText(info_bg, f'Model: {model_name}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(info_bg, f'FPS: {avg_fps:.1f}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(info_bg, f'Objects: {obj_count}', (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Overlay info on frame
            alpha = 0.7
            frame[0:120, 0:300] = cv2.addWeighted(
                frame[0:120, 0:300], 1-alpha, info_bg, alpha, 0
            )
            
            # Update global frame (thread-safe)
            with frame_lock:
                output_frame = frame.copy()
                fps_value = avg_fps
                object_count = obj_count
                
        except Exception as e:
            print(f"⚠️ Inference error: {e}")
            time.sleep(0.01)

def generate_frames():
    global output_frame
    
    encode_param = [
        int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY,
        int(cv2.IMWRITE_JPEG_OPTIMIZE), 1,
        int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1
    ]
    
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            frame = output_frame.copy()
        
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', models=list(MODELS.keys()))

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/start', methods=['POST'])
def start_detection():
    global is_running
    
    try:
        data = request.get_json()
        selected_model = data.get('model', 'YOLOv11')
        camera_id = data.get('camera', 0)
        
        if is_running:
            is_running = False
            camera_thread.stop()
            detection_tracker.reset()
            time.sleep(0.3)
        
        if not load_model(selected_model):
            return jsonify({
                'status': 'error',
                'message': f'Failed to load {selected_model}'
            })
        
        if not camera_thread.start():
            return jsonify({
                'status': 'error',
                'message': 'Failed to open camera'
            })
        
        is_running = True
        print(f"✅ Started {selected_model}")
        return jsonify({
            'status': 'success',
            'message': f'Started {selected_model}'
        })
        
    except Exception as e:
        is_running = False
        return jsonify({
            'status': 'error',
            'message': f'Error: {str(e)}'
        })

@app.route('/stop', methods=['POST'])
def stop_detection():
    global is_running
    is_running = False
    camera_thread.stop()
    detection_tracker.reset()
    return jsonify({'status': 'success', 'message': 'Detection stopped'})

@app.route('/status')
def get_status():
    return jsonify({
        'is_running': is_running,
        'model': model_name,
        'fps': round(fps_value, 1),
        'objects': object_count,
        'available_models': list(MODELS.keys())
    })

if __name__ == '__main__':
    load_model('YOLOv11')
    
    inference_worker = Thread(target=inference_thread, daemon=True)
    inference_worker.start()
    
    print("=" * 70)
    print("🚀 OPTIMIZED YOLO with SMOOTH TRACKING")
    print("=" * 70)
    print(f"Device: {DEVICE.upper()}")
    print(f"Expected FPS: {'30-40' if DEVICE == 'cuda' else '15-25'}")
    print(f"Optimizations:")
    print(f"  ✓ Temporal smoothing (factor: {SMOOTH_FACTOR})")
    print(f"  ✓ Detection tracking & stabilization")
    print(f"  ✓ NMS with IoU threshold: {IOU_THRESHOLD}")
    print(f"  ✓ Multi-threading (camera + inference)")
    print("=" * 70)
    
    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000,
        threaded=True,
        use_reloader=False
    )