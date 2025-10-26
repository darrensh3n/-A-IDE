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
pose_model = None  # YOLOv8 pose estimation model
MODEL_PATH = "models/drowning_detection.pt"

# Global state for tracking people across frames (for drowning behavior analysis)
person_positions = {}  # {person_id: [(frame_number, x, y, width, height), ...]}
frame_counter = 0

# Global alert manager
alert_manager = None

def load_model():
    """Load YOLOv8 model for drowning detection"""
    global model, pose_model
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Loading custom model from {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
        else:
            print("Custom model not found. Using YOLOv8n as fallback.")
            print("To use a custom drowning detection model, place it at:", MODEL_PATH)
            model = YOLO("yolov8n.pt")  # Fallback to pretrained model
        print("Model loaded successfully!")
        
        # Load YOLOv8 pose estimation model for distress detection
        print("Loading YOLOv8 pose estimation model...")
        pose_model = YOLO("yolov8n-pose.pt")
        print("✓ Pose estimation model loaded successfully!")
        
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
            confidence_threshold=0.5,
            repeat_alert_delay=2.0,
            live_mode=True
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

def analyze_drowning_behavior(detections, frame_number):
    """
    Analyze person movements to detect drowning behavior
    
    Drowning indicators:
    - Person remains in similar vertical position (no swimming movement)
    - Erratic/vertical hand movements
    - Person is in water for extended period without horizontal movement
    - Multiple people clustered in one area (possible rescue scene)
    
    Args:
        detections: List of detected objects
        frame_number: Current frame number
        
    Returns:
        Dictionary with drowning risk assessment
    """
    global person_positions, frame_counter
    frame_counter += 1
    
    drowning_risk = "none"
    risk_score = 0.0
    indicators = []
    
    # Filter for person detections
    person_detections = [d for d in detections if d.get("class", "").lower() == "person"]
    
    if not person_detections:
        return {
            "drowning_risk": "none",
            "risk_score": 0.0,
            "indicators": [],
            "people_detected": 0
        }
    
    # Track each person
    for i, detection in enumerate(person_detections):
        person_id = i
        bbox = detection["bbox"]
        x, y, x2, y2 = bbox
        center_x = (x + x2) / 2
        center_y = (y + y2) / 2
        width = x2 - x
        height = y2 - y
        
        # Initialize tracking for new person
        if person_id not in person_positions:
            person_positions[person_id] = []
        
        # Add current position
        person_positions[person_id].append({
            "frame": frame_number,
            "x": center_x,
            "y": center_y,
            "width": width,
            "height": height
        })
        
        # Keep only last 30 frames of history
        person_positions[person_id] = person_positions[person_id][-30:]
    
    # Analyze each person's movement pattern
    for person_id, positions in person_positions.items():
        if len(positions) < 10:  # Need minimum frames for analysis
            continue
        
        # Calculate movement statistics
        y_positions = [p["y"] for p in positions[-15:]]  # Last 15 frames
        x_positions = [p["x"] for p in positions[-15:]]
        
        y_variance = np.var(y_positions)
        x_variance = np.var(x_positions)
        avg_y = np.mean(y_positions)
        avg_x = np.mean(x_positions)
        
        # Indicator 1: Lack of horizontal movement (staying in same spot)
        if x_variance < 100:  # Person not moving horizontally
            risk_score += 0.3
            indicators.append(f"Person {person_id}: Minimal horizontal movement")
        
        # Indicator 2: Vertical position not changing (not swimming up)
        if y_variance < 50:
            risk_score += 0.3
            indicators.append(f"Person {person_id}: Stuck at same vertical position")
        
        # Indicator 3: Extended time without significant movement
        if len(positions) > 20:
            recent_y_variance = np.var([p["y"] for p in positions[-10:]])
            if recent_y_variance < 30:
                risk_score += 0.4
                indicators.append(f"Person {person_id}: Extended time without vertical movement")
    
    # Determine risk level
    if risk_score >= 0.7:
        drowning_risk = "high"
    elif risk_score >= 0.4:
        drowning_risk = "medium"
    elif risk_score >= 0.2:
        drowning_risk = "low"
    else:
        drowning_risk = "none"
    
    return {
        "drowning_risk": drowning_risk,
        "risk_score": round(min(risk_score, 1.0), 2),
        "indicators": indicators,
        "people_detected": len(person_detections)
    }

def analyze_pose_distress(image, detections):
    """
    Analyze pose keypoints to detect distress/drowning behavior
    
    Distress indicators from pose:
    - Arms raised above head (sinking)
    - Head below shoulders (sinking)
    - Vertical body orientation (in water)
    - Arms flailing (erratic keypoints)
    - Asymmetric arm positions (struggling)
    
    Args:
        image: Input image (numpy array)
        detections: List of person detections with bboxes
        
    Returns:
        Dictionary with pose-based distress analysis
    """
    global pose_model
    
    if pose_model is None:
        return None
    
    pose_results = []
    distress_scores = []
    
    for detection in detections:
        if detection.get("class", "").lower() != "person":
            continue
            
        bbox = detection["bbox"]
        x1, y1, x2, y2 = bbox
        
        # Extract person region from image
        person_roi = image[int(y1):int(y2), int(x1):int(x2)]
        
        if person_roi.size == 0:
            continue
        
        # Run pose estimation on person region
        try:
            results = pose_model(person_roi, verbose=False)
            
            if len(results) == 0 or results[0].keypoints is None:
                print("    No pose keypoints detected for this person")
                continue
            
            print(f"    ✓ Detected {len(results[0].keypoints.xy[0])} keypoints for person")
                
            keypoints = results[0].keypoints.xy.cpu().numpy()[0]  # Get first person's keypoints
            confidences = results[0].keypoints.conf.cpu().numpy()[0]
            
            # YOLO pose keypoint indices (17 keypoints):
            # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
            # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
            # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
            # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
            
            distress_score = 0.0
            distress_indicators = []
            
            # Calculate distress indicators
            valid_keypoints = []
            for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
                if conf > 0.5:  # Only use confident keypoints
                    valid_keypoints.append((i, kp, conf))
            
            if len(valid_keypoints) < 5:  # Need minimum keypoints
                continue
            
            # Indicator 1: Arms raised above head
            if all(conf > 0.5 for conf in [confidences[0], confidences[9], confidences[10]]):
                nose_y = keypoints[0][1]
                left_wrist_y = keypoints[9][1]
                right_wrist_y = keypoints[10][1]
                
                if left_wrist_y < nose_y or right_wrist_y < nose_y:
                    distress_score += 0.3
                    distress_indicators.append("Arms raised (sinking distress)")
            
            # Indicator 2: Head below shoulders (vertical sinking)
            if all(conf > 0.5 for conf in [confidences[0], confidences[5], confidences[6]]):
                nose_y = keypoints[0][1]
                left_shoulder_y = keypoints[5][1]
                right_shoulder_y = keypoints[6][1]
                avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2
                
                if nose_y > avg_shoulder_y + 30:  # Head significantly below shoulders
                    distress_score += 0.3
                    distress_indicators.append("Head below shoulders (vertical orientation)")
            
            # Indicator 3: Asymmetric arm positions (flailing)
            if all(conf > 0.5 for conf in [confidences[7], confidences[8], confidences[9], confidences[10]]):
                left_elbow_y = keypoints[7][1]
                right_elbow_y = keypoints[8][1]
                left_wrist_y = keypoints[9][1]
                right_wrist_y = keypoints[10][1]
                
                arm_asymmetry = abs((left_elbow_y - left_wrist_y) - (right_elbow_y - right_wrist_y))
                if arm_asymmetry > 50:  # Significant asymmetry
                    distress_score += 0.2
                    distress_indicators.append("Asymmetric arm positions (flailing)")
            
            # Indicator 4: Erratic movement (jump detection from recent frames)
            # This would require frame history, simplified here
            distress_scores.append(distress_score)
            
            pose_results.append({
                "person_id": len(pose_results),
                "distress_score": distress_score,
                "indicators": distress_indicators,
                "keypoints": keypoints.tolist() if len(keypoints) > 0 else []
            })
            
        except Exception as e:
            print(f"Error in pose estimation: {e}")
            continue
    
    if not distress_scores:
        return None
    
    max_distress = max(distress_scores)
    
    # Normalize distress score
    if max_distress >= 0.6:
        distress_level = "high"
    elif max_distress >= 0.3:
        distress_level = "medium"
    else:
        distress_level = "low"
    
    return {
        "distress_level": distress_level,
        "distress_score": round(max_distress, 2),
        "people_analyzed": len(pose_results),
        "pose_results": pose_results
    }

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

def process_image(image_path: str, confidence_threshold: float = 0.30):
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
    
    # Filter for person detections only
    person_detections = [d for d in detections if d.get("class", "").lower() == "person"]
    
    # Run pose estimation on person detections
    pose_analysis = None
    if person_detections and pose_model is not None:
        print(f"Running pose estimation on {len(person_detections)} person(s)...")
        pose_analysis = analyze_pose_distress(img, person_detections)
        if pose_analysis:
            print(f"Pose analysis complete: {pose_analysis.get('distress_level', 'unknown')} distress")
        else:
            print("No pose analysis results")
    elif person_detections and pose_model is None:
        print("⚠️ Pose model not loaded - skipping pose estimation")
    
    # Draw bounding boxes and keypoints for all detections
    for detection in detections:
        x1, y1, x2, y2 = [int(coord) for coord in detection["bbox"]]
        class_name = detection["class"]
        confidence = detection["confidence"]
        is_person = class_name.lower() == "person"
        
        # Determine color based on class (red for drowning-related classes)
        drowning_keywords = ["drowning", "distress", "struggle", "person"]
        color = (0, 0, 255) if any(kw in class_name.lower() for kw in drowning_keywords) else (0, 255, 0)
        
        # Draw rectangle
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        
        # If this is a person and we have pose analysis, draw keypoints
        if is_person and pose_analysis:
            # Find corresponding pose result for this person
            person_index = next((i for i, det in enumerate(person_detections) if det == detection), None)
            if person_index is not None and person_index < len(pose_analysis.get("pose_results", [])):
                pose_result = pose_analysis["pose_results"][person_index]
                keypoints = pose_result.get("keypoints", [])
                
                if keypoints:
                    # Draw skeleton connections
                    skeleton_connections = [
                        (0, 1), (0, 2), (1, 3), (2, 4),  # head
                        (5, 6),  # shoulders
                        (5, 7), (7, 9),  # left arm
                        (6, 8), (8, 10),  # right arm
                        (5, 11), (6, 12),  # torso
                        (11, 12),  # hips
                        (11, 13), (13, 15),  # left leg
                        (12, 14), (14, 16),  # right leg
                    ]
                    
                    for start_idx, end_idx in skeleton_connections:
                        if (start_idx < len(keypoints) and end_idx < len(keypoints) and
                            all(keypoints[start_idx][i] > 0 and keypoints[end_idx][i] > 0 for i in range(2))):
                            # Adjust keypoint coordinates (they're relative to ROI)
                            pt1 = (int(keypoints[start_idx][0] + x1), int(keypoints[start_idx][1] + y1))
                            pt2 = (int(keypoints[end_idx][0] + x1), int(keypoints[end_idx][1] + y1))
                            cv2.line(annotated_img, pt1, pt2, (255, 255, 0), 2)
                    
                    # Draw keypoints
                    for i, kp in enumerate(keypoints):
                        if all(kp[j] > 0 for j in range(2)):  # Valid keypoint
                            center = (int(kp[0] + x1), int(kp[1] + y1))
                            # Color keypoints based on importance for distress
                            if i in [0, 9, 10]:  # nose, left_wrist, right_wrist (important for distress)
                                kp_color = (0, 255, 255)  # Yellow
                                cv2.circle(annotated_img, center, 4, kp_color, -1)
                            elif i in [5, 6]:  # shoulders
                                kp_color = (255, 0, 255)  # Magenta
                                cv2.circle(annotated_img, center, 4, kp_color, -1)
                            else:
                                kp_color = (255, 255, 255)  # White
                                cv2.circle(annotated_img, center, 3, kp_color, -1)
        
        # Draw label
        label = f"{class_name} {confidence:.2f}"
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated_img, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
        cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Analyze for drowning behavior
    global frame_counter
    behavior_analysis = analyze_drowning_behavior(detections, frame_counter)
    
    # Combine pose analysis with behavior analysis for comprehensive risk assessment
    if pose_analysis:
        # Use the higher risk level from pose analysis
        pose_risk_scores = {"low": 0.3, "medium": 0.6, "high": 0.9}
        pose_risk_score = pose_risk_scores.get(pose_analysis.get("distress_level", "low"), 0.3)
        
        # Combine with movement-based analysis (weighted average)
        combined_risk_score = (behavior_analysis["risk_score"] * 0.4) + (pose_risk_score * 0.6)
        
        # Update drowning risk based on combined analysis
        if combined_risk_score >= 0.7:
            behavior_analysis["drowning_risk"] = "high"
        elif combined_risk_score >= 0.4:
            behavior_analysis["drowning_risk"] = "medium"
        elif combined_risk_score >= 0.2:
            behavior_analysis["drowning_risk"] = "low"
        
        behavior_analysis["risk_score"] = round(combined_risk_score, 2)
        behavior_analysis["pose_analysis"] = pose_analysis
    
    # Highlight high-risk detections
    if behavior_analysis["drowning_risk"] in ["high", "medium"]:
        # Draw warning overlay
        cv2.putText(annotated_img, f"ALERT: {behavior_analysis['drowning_risk'].upper()} DROWNING RISK", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(annotated_img, f"Risk Score: {behavior_analysis['risk_score']}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add pose-based distress info if available
        if pose_analysis and pose_analysis.get("distress_level") in ["medium", "high"]:
            cv2.putText(annotated_img, f"Pose Distress: {pose_analysis['distress_level']}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Convert annotated image to base64
    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    result = {
        "detections": detections,
        "image": f"data:image/jpeg;base64,{img_base64}",
        "timestamp": datetime.now().isoformat(),
        "drowning_analysis": behavior_analysis
    }
    
    # Check for danger and trigger alerts
    if alert_manager:
        alert_status = alert_manager.check_and_alert(detections, behavior_analysis)
        result["alert_status"] = alert_status
    
    return result

def process_video(video_path: str, confidence_threshold: float = 0.30, sample_rate: int = 30):
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
    
    # Analyze for drowning behavior (use recent detections)
    global frame_counter
    recent_detections = all_detections[-10:] if all_detections else []  # Last 10 detections
    behavior_analysis = analyze_drowning_behavior(recent_detections, frame_counter)
    result["drowning_analysis"] = behavior_analysis
    
    # Check for danger and trigger alerts (use all detections for analysis)
    if alert_manager:
        alert_status = alert_manager.check_and_alert(all_detections, behavior_analysis)
        result["alert_status"] = alert_status
    
    return result

@app.post("/api/detect-drowning")
async def detect_drowning(
    file: UploadFile = File(...),
    confidence: float = 0.30
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
    confidence: float = 0.30
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

@app.post("/api/generate-voice-alert")
async def generate_voice_alert():
    """
    Manually trigger a test voice alert
    
    Returns:
        Status of the voice alert generation
    """
    if not alert_manager:
        raise HTTPException(status_code=503, detail="Alert system not initialized")
    
    # Create mock detection data for testing
    mock_detections = [
        {
            "class": "person",
            "confidence": 0.85,
            "bbox": [100, 100, 200, 300]
        }
    ]
    
    try:
        # Generate and send a voice alert
        alert_manager._send_alert(mock_detections, is_repeat=False)
        
        return {
            "success": True,
            "message": "Voice alert generated and sent successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate voice alert: {str(e)}"
        )

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
