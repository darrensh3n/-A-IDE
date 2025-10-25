"""
FastAPI server for drowning detection using YOLOv8
Run with: python scripts/api_server.py
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pathlib import Path
import shutil
import tempfile
from datetime import datetime
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import os
from dotenv import load_dotenv

# Import alert services
from services.llm_service import LLMService
from services.voice_alert_service import VoiceAlertService
from services.alert_manager import AlertManager

# Load environment variables
load_dotenv()

app = FastAPI(title="Drowning Detection API", version="1.0.0")

# Configure CORS to allow requests from Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
MODEL_PATH = "models/drowning_detection.pt"

# Global alert manager
alert_manager = None

def load_model():
    """Load YOLOv8 model for drowning detection"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Loading custom model from {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
        else:
            print("Custom model not found. Using YOLOv8n as fallback.")
            print("To use a custom drowning detection model, place it at:", MODEL_PATH)
            model = YOLO("yolov8n.pt")  # Fallback to pretrained model
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def initialize_alert_system():
    """Initialize the alert system with Fish Audio and Groq"""
    global alert_manager
    
    try:
        # Get API keys from environment
        fish_key = os.getenv("FISH_AUDIO_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        
        if not fish_key or fish_key == "your_fish_audio_api_key_here":
            print("⚠️  FISH_AUDIO_API_KEY not set - Alert system disabled")
            return
        
        if not groq_key or groq_key == "your_groq_api_key_here":
            print("⚠️  GROQ_API_KEY not set - Alert system disabled")
            return
        
        # Initialize services
        print("\nInitializing alert system...")
        llm_service = LLMService(groq_key)
        voice_service = VoiceAlertService(fish_key, output_dir="audio_alerts")
        alert_manager = AlertManager(
            llm_service=llm_service,
            voice_service=voice_service,
            confidence_threshold=0.7,
            repeat_alert_delay=6.0
        )
        
        print("✓ Alert system initialized successfully!\n")
        
    except Exception as e:
        print(f"✗ Failed to initialize alert system: {e}")
        print("   Detection will continue without alerts\n")
        alert_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize model and alert system on startup"""
    load_model()
    initialize_alert_system()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "Drowning Detection API is running",
        "model_loaded": model is not None
    }

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if model else "not loaded",
        "model_path": MODEL_PATH if os.path.exists(MODEL_PATH) else "using fallback"
    }

def process_image(image_path: str, confidence_threshold: float = 0.25):
    """
    Process a single image for drowning detection
    
    Args:
        image_path: Path to the image file
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        Dictionary with detections and annotated image
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Run inference
    results = model(img, conf=confidence_threshold)
    
    # Extract detections
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Get confidence and class
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]
            
            detections.append({
                "class": class_name,
                "confidence": confidence,
                "bbox": [float(x1), float(y1), float(x2), float(y2)]
            })
    
    # Draw bounding boxes on image
    annotated_img = img.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(coord) for coord in detection["bbox"]]
        class_name = detection["class"]
        confidence = detection["confidence"]
        
        # Determine color based on class (red for drowning-related classes)
        drowning_keywords = ["drowning", "distress", "struggle", "person"]
        color = (0, 0, 255) if any(kw in class_name.lower() for kw in drowning_keywords) else (0, 255, 0)
        
        # Draw rectangle
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name} {confidence:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated_img, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
        cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Convert annotated image to base64
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    result = {
        "detections": detections,
        "image": f"data:image/jpeg;base64,{img_base64}",
        "timestamp": datetime.now().isoformat()
    }
    
    # Check for danger and trigger alerts
    if alert_manager:
        alert_status = alert_manager.check_and_alert(detections)
        result["alert_status"] = alert_status
    
    return result

def process_video(video_path: str, confidence_threshold: float = 0.25, sample_rate: int = 30):
    """
    Process a video file for drowning detection
    
    Args:
        video_path: Path to the video file
        confidence_threshold: Minimum confidence for detections
        sample_rate: Process every Nth frame
        
    Returns:
        Dictionary with aggregated detections and sample frame
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Invalid video file")
    
    all_detections = []
    frame_count = 0
    sample_frame = None
    sample_frame_detections = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every Nth frame
        if frame_count % sample_rate == 0:
            results = model(frame, conf=confidence_threshold)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    
                    detection = {
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "frame": frame_count
                    }
                    all_detections.append(detection)
                    
                    # Save first frame with detections
                    if sample_frame is None and len(boxes) > 0:
                        sample_frame = frame.copy()
                        sample_frame_detections = [detection]
    
    cap.release()
    
    # If we have a sample frame, annotate it
    img_base64 = None
    if sample_frame is not None:
        annotated_frame = sample_frame.copy()
        for detection in sample_frame_detections:
            x1, y1, x2, y2 = [int(coord) for coord in detection["bbox"]]
            class_name = detection["class"]
            confidence = detection["confidence"]
            
            drowning_keywords = ["drowning", "distress", "struggle", "person"]
            color = (0, 0, 255) if any(kw in class_name.lower() for kw in drowning_keywords) else (0, 255, 0)
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Aggregate detections by class
    detection_summary = {}
    for det in all_detections:
        class_name = det["class"]
        if class_name not in detection_summary:
            detection_summary[class_name] = {
                "count": 0,
                "max_confidence": 0,
                "avg_confidence": 0,
                "confidences": []
            }
        detection_summary[class_name]["count"] += 1
        detection_summary[class_name]["confidences"].append(det["confidence"])
        detection_summary[class_name]["max_confidence"] = max(
            detection_summary[class_name]["max_confidence"],
            det["confidence"]
        )
    
    # Calculate averages
    for class_name in detection_summary:
        confidences = detection_summary[class_name]["confidences"]
        detection_summary[class_name]["avg_confidence"] = sum(confidences) / len(confidences)
        del detection_summary[class_name]["confidences"]
    
    result = {
        "detections": all_detections[:10],  # Return first 10 detections
        "summary": detection_summary,
        "total_frames": frame_count,
        "processed_frames": frame_count // sample_rate,
        "image": f"data:image/jpeg;base64,{img_base64}" if img_base64 else None,
        "timestamp": datetime.now().isoformat()
    }
    
    # Check for danger and trigger alerts (use all detections for analysis)
    if alert_manager:
        alert_status = alert_manager.check_and_alert(all_detections)
        result["alert_status"] = alert_status
    
    return result

@app.post("/api/detect-drowning")
async def detect_drowning(
    file: UploadFile = File(...),
    confidence: float = 0.25
):
    """
    Detect drowning in uploaded image or video
    
    Args:
        file: Image or video file
        confidence: Confidence threshold (0-1)
        
    Returns:
        Detection results with bounding boxes and confidence scores
    """
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov', '.webm'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name
    
    try:
        # Process based on file type
        if file_ext in {'.jpg', '.jpeg', '.png'}:
            result = process_image(tmp_path, confidence)
        else:
            result = process_video(tmp_path, confidence)
        
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass

@app.post("/api/detect-frame")
async def detect_frame(
    file: UploadFile = File(...),
    confidence: float = 0.25
):
    """
    Detect drowning in a single frame (optimized for real-time camera feed)
    
    Args:
        file: Image file (frame from camera)
        confidence: Confidence threshold (0-1)
        
    Returns:
        Detection results with bounding boxes
    """
    return await detect_drowning(file, confidence)

@app.get("/api/alert-status")
async def get_alert_status():
    """
    Get current alert system status
    
    Returns:
        Alert status information
    """
    if not alert_manager:
        return {
            "enabled": False,
            "message": "Alert system not initialized"
        }
    
    status = alert_manager.get_status()
    status["enabled"] = True
    return status

@app.post("/api/alert-reset")
async def reset_alert():
    """
    Manually reset alert state
    
    Returns:
        Confirmation message
    """
    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert system not initialized")
    
    alert_manager.reset()
    return {
        "message": "Alert state reset successfully",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("=" * 60)
    print("Drowning Detection API Server")
    print("=" * 60)
    print("\nStarting server on http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("\nNote: Place your custom YOLOv8 model at:", MODEL_PATH)
    print("      Or the server will use YOLOv8n as fallback\n")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
