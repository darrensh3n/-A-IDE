"""
Alert Manager
Manages alert state and triggers voice alerts based on danger detection.
"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from .llm_service import LLMService
from .voice_alert_service import VoiceAlertService


class AlertManager:
    """
    Manages alert state and triggers voice alerts when danger is detected.
    
    Alert Logic:
    - First dangerous detection → immediate alert
    - Danger persists for 6+ seconds → second alert
    - Danger clears → reset state
    """
    
    # Classes that should trigger alerts (when confidence > threshold)
    DANGEROUS_CLASSES = ["drowning", "distress", "struggle", "person"]
    
    def __init__(
        self,
        llm_service: LLMService,
        voice_service: VoiceAlertService,
        confidence_threshold: float = 0.5,
        repeat_alert_delay: float = 2.0,
        live_mode: bool = True
    ):
        """
        Initialize the alert manager
        
        Args:
            llm_service: LLM service for generating alert messages
            voice_service: Voice service for TTS
            confidence_threshold: Minimum confidence to trigger alert (lowered for live detection)
            repeat_alert_delay: Seconds before repeat alert (reduced for live mode)
            live_mode: Enable live detection mode (alerts on every detection)
        """
        self.llm_service = llm_service
        self.voice_service = voice_service
        self.confidence_threshold = confidence_threshold
        self.repeat_alert_delay = repeat_alert_delay
        self.live_mode = live_mode
        
        # Alert state
        self.danger_active = False
        self.first_alert_time: Optional[float] = None
        self.second_alert_sent = False
        self.last_dangerous_detections: List[Dict[str, Any]] = []
        self.last_alert_time: Optional[float] = None
        
        print(f"✓ Alert Manager initialized (threshold: {confidence_threshold}, repeat delay: {repeat_alert_delay}s, live mode: {live_mode})")
    
    def check_and_alert(self, detections: List[Dict[str, Any]], drowning_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check detections for danger and trigger alerts if needed
        
        Args:
            detections: List of detection dictionaries
            drowning_analysis: Optional drowning behavior analysis results
            
        Returns:
            Dictionary with alert status information
        """
        # Check if there are any dangerous detections
        dangerous = self._filter_dangerous_detections(detections)
        
        # Only trigger Fish Audio alerts if person is detected AND there's drowning risk
        if dangerous and self._should_trigger_voice_alert(dangerous, drowning_analysis):
            return self._handle_danger_detected(dangerous)
        else:
            return self._handle_no_danger()
    
    def _filter_dangerous_detections(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter detections for dangerous classes above threshold"""
        dangerous = []
        
        for det in detections:
            class_name = det.get("class", "").lower()
            confidence = det.get("confidence", 0)
            
            # Check if class is dangerous and confidence is high enough
            if any(dc in class_name for dc in self.DANGEROUS_CLASSES):
                if confidence >= self.confidence_threshold:
                    dangerous.append(det)
        
        return dangerous
    
    def _should_trigger_voice_alert(self, dangerous_detections: List[Dict[str, Any]], drowning_analysis: Optional[Dict[str, Any]]) -> bool:
        """
        Determine if Fish Audio alert should be triggered
        
        Args:
            dangerous_detections: List of dangerous detections
            drowning_analysis: Optional drowning behavior analysis results
            
        Returns:
            True if voice alert should be triggered
        """
        # Check if any person detections are present
        has_person = any(det.get("class", "").lower() == "person" for det in dangerous_detections)
        
        if not has_person:
            return False
        
        # In live mode, be more aggressive with alerts
        if self.live_mode:
            # Trigger on any person detection with medium/high risk, or any person if no analysis
            if drowning_analysis:
                drowning_risk = drowning_analysis.get("drowning_risk", "none")
                return drowning_risk in ["medium", "high", "low"]  # Include low risk in live mode
            else:
                return True  # Trigger on any person detection in live mode
        
        # Standard mode: only trigger on medium/high risk
        if drowning_analysis:
            drowning_risk = drowning_analysis.get("drowning_risk", "none")
            return drowning_risk in ["medium", "high"]
        
        # If no drowning analysis, trigger on any person detection (fallback)
        return True
    
    def _handle_danger_detected(self, dangerous_detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle when danger is detected"""
        current_time = time.time()
        self.last_dangerous_detections = dangerous_detections
        
        # Live mode: Send alert on every detection (with minimal throttling)
        if self.live_mode:
            # Throttle alerts to prevent spam (minimum 1 second between alerts)
            if self.last_alert_time and (current_time - self.last_alert_time) < 1.0:
                return {
                    "danger_active": True,
                    "alert_sent": False,
                    "alert_type": "throttled",
                    "time_since_last_alert": current_time - self.last_alert_time,
                    "dangerous_detections": len(dangerous_detections)
                }
            
            print("\n🚨 LIVE ALERT - Person detected with drowning risk!")
            self.last_alert_time = current_time
            
            # Generate and send alert
            self._send_alert(dangerous_detections, is_repeat=False)
            
            return {
                "danger_active": True,
                "alert_sent": True,
                "alert_type": "live",
                "time_since_last_alert": 0,
                "dangerous_detections": len(dangerous_detections)
            }
        
        # Original logic for non-live mode
        # Case 1: First danger detection - send immediate alert
        if not self.danger_active:
            print("\n⚠️  DANGER DETECTED - Sending first alert")
            self.danger_active = True
            self.first_alert_time = current_time
            self.second_alert_sent = False
            
            # Generate and send first alert
            self._send_alert(dangerous_detections, is_repeat=False)
            
            return {
                "danger_active": True,
                "alert_sent": True,
                "alert_type": "first",
                "time_since_first_alert": 0,
                "dangerous_detections": len(dangerous_detections)
            }
        
        # Case 2: Danger persists - check if it's time for repeat alert
        elif self.danger_active and not self.second_alert_sent:
            time_elapsed = current_time - self.first_alert_time
            
            if time_elapsed >= self.repeat_alert_delay:
                print(f"\n⚠️  DANGER PERSISTS ({time_elapsed:.1f}s) - Sending repeat alert")
                self.second_alert_sent = True
                
                # Generate and send repeat alert
                self._send_alert(dangerous_detections, is_repeat=True)
                
                return {
                    "danger_active": True,
                    "alert_sent": True,
                    "alert_type": "repeat",
                    "time_since_first_alert": time_elapsed,
                    "dangerous_detections": len(dangerous_detections)
                }
            else:
                # Still in danger but not time for repeat yet
                return {
                    "danger_active": True,
                    "alert_sent": False,
                    "alert_type": "monitoring",
                    "time_since_first_alert": time_elapsed,
                    "dangerous_detections": len(dangerous_detections),
                    "time_until_repeat": self.repeat_alert_delay - time_elapsed
                }
        
        # Case 3: Both alerts already sent
        else:
            time_elapsed = current_time - self.first_alert_time
            return {
                "danger_active": True,
                "alert_sent": False,
                "alert_type": "both_sent",
                "time_since_first_alert": time_elapsed,
                "dangerous_detections": len(dangerous_detections)
            }
    
    def _handle_no_danger(self) -> Dict[str, Any]:
        """Handle when no danger is detected"""
        # Reset state if danger was previously active
        if self.danger_active:
            print("\n✓ Danger cleared - Resetting alert state")
            self.danger_active = False
            self.first_alert_time = None
            self.second_alert_sent = False
            self.last_dangerous_detections = []
        
        return {
            "danger_active": False,
            "alert_sent": False,
            "alert_type": "clear",
            "dangerous_detections": 0
        }
    
    def _send_alert(self, dangerous_detections: List[Dict[str, Any]], is_repeat: bool):
        """
        Generate and send voice alert
        
        Args:
            dangerous_detections: List of dangerous detection dictionaries
            is_repeat: Whether this is a repeat alert
        """
        try:
            # Generate alert message using LLM
            alert_text = self.llm_service.generate_alert_message(
                dangerous_detections,
                is_repeat=is_repeat
            )
            
            # Fallback if LLM fails
            if not alert_text:
                print("⚠️  LLM generation failed, using fallback message")
                alert_text = self.llm_service.get_fallback_message(is_repeat)
            
            # Generate and play voice alert
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            alert_type = "repeat" if is_repeat else "first"
            filename = f"alert_{alert_type}_{timestamp}.mp3"
            
            self.voice_service.generate_and_play(
                text=alert_text,
                save=True,
                play=True
            )
            
        except Exception as e:
            print(f"✗ Error sending alert: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current alert manager status"""
        status = {
            "danger_active": self.danger_active,
            "second_alert_sent": self.second_alert_sent,
            "dangerous_detections": len(self.last_dangerous_detections)
        }
        
        if self.first_alert_time:
            time_elapsed = time.time() - self.first_alert_time
            status["time_since_first_alert"] = time_elapsed
            
            if not self.second_alert_sent:
                status["time_until_repeat"] = max(0, self.repeat_alert_delay - time_elapsed)
        
        return status
    
    def reset(self):
        """Manually reset alert state"""
        print("🔄 Manually resetting alert state")
        self.danger_active = False
        self.first_alert_time = None
        self.second_alert_sent = False
        self.last_dangerous_detections = []


# Example usage
if __name__ == "__main__":
    import sys
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    fish_key = os.getenv("FISH_AUDIO_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not fish_key or fish_key == "your_fish_audio_api_key_here":
        print("Please set FISH_AUDIO_API_KEY in .env file")
        sys.exit(1)
    
    if not groq_key or groq_key == "your_groq_api_key_here":
        print("Please set GROQ_API_KEY in .env file")
        sys.exit(1)
    
    # Initialize services
    print("\nInitializing services...")
    llm_service = LLMService(groq_key)
    voice_service = VoiceAlertService(fish_key)
    alert_manager = AlertManager(llm_service, voice_service)
    
    # Simulate detection sequence
    print("\n" + "="*60)
    print("SIMULATION: Testing Alert Manager")
    print("="*60)
    
    # Test 1: First danger detection
    print("\n[T=0s] Simulating first danger detection...")
    dangerous_detections = [
        {"class": "person", "confidence": 0.85, "bbox": [100, 100, 200, 300]},
        {"class": "drowning", "confidence": 0.78, "bbox": [150, 120, 220, 280]}
    ]
    
    result = alert_manager.check_and_alert(dangerous_detections)
    print(f"Result: {result}")
    
    # Test 2: Danger still present (after 3 seconds)
    print("\n[T=3s] Simulating continued danger (3s later)...")
    time.sleep(3)
    result = alert_manager.check_and_alert(dangerous_detections)
    print(f"Result: {result}")
    
    # Test 3: Danger still present (after 4 more seconds - total 7s)
    print("\n[T=7s] Simulating continued danger (4s later, 7s total)...")
    time.sleep(4)
    result = alert_manager.check_and_alert(dangerous_detections)
    print(f"Result: {result}")
    
    print("\n✓ Simulation complete")

