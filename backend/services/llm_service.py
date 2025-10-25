"""
LLM Service using Groq
Generates contextual alert messages based on detection data.
"""

import os
from typing import List, Dict, Any, Optional
from groq import Groq


class LLMService:
    """Service for generating alert messages using Groq LLM"""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize the LLM service
        
        Args:
            api_key: Groq API key
            model: Groq model to use (default: llama-3.3-70b-versatile)
        """
        self.api_key = api_key
        self.model = model
        
        try:
            self.client = Groq(api_key=api_key)
            print(f"✓ Groq LLM initialized (model: {model})")
        except Exception as e:
            print(f"✗ Failed to initialize Groq: {e}")
            self.client = None
    
    def generate_alert_message(
        self,
        detections: List[Dict[str, Any]],
        is_repeat: bool = False
    ) -> Optional[str]:
        """
        Generate contextual alert message based on detection data
        
        Args:
            detections: List of detection dictionaries containing:
                - class: Detection class name
                - confidence: Confidence score (0-1)
                - bbox: Bounding box coordinates (optional)
            is_repeat: Whether this is a repeat alert (danger persisting)
            
        Returns:
            Generated alert message, or None if generation failed
        """
        if not self.client:
            print("✗ Groq client not initialized")
            return None
        
        try:
            # Prepare detection summary
            detection_summary = self._format_detections(detections)
            
            # Create prompt for Groq
            prompt = self._create_prompt(detection_summary, is_repeat)
            
            print(f"🤖 Generating alert with Groq...")
            
            # Call Groq API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=150,
                temperature=0.7
            )
            
            # Extract text from response
            alert_text = completion.choices[0].message.content.strip()
            
            # Remove quotes if LLM added them
            if alert_text.startswith('"') and alert_text.endswith('"'):
                alert_text = alert_text[1:-1]
            
            print(f"✓ Alert generated: '{alert_text}'")
            return alert_text
            
        except Exception as e:
            print(f"✗ Error generating alert: {e}")
            return None
    
    def _format_detections(self, detections: List[Dict[str, Any]]) -> str:
        """Format detections into a readable summary"""
        if not detections:
            return "No specific detections"
        
        # Group detections by class
        class_counts = {}
        max_confidence = {}
        
        for det in detections:
            cls = det.get("class", "unknown")
            conf = det.get("confidence", 0)
            
            class_counts[cls] = class_counts.get(cls, 0) + 1
            max_confidence[cls] = max(max_confidence.get(cls, 0), conf)
        
        # Format summary
        summary_parts = []
        for cls, count in class_counts.items():
            conf = max_confidence[cls]
            summary_parts.append(f"{cls} ({count}x, {conf:.0%} confidence)")
        
        return ", ".join(summary_parts)
    
    def _create_prompt(self, detection_summary: str, is_repeat: bool) -> str:
        """Create prompt for Groq"""
        if is_repeat:
            prompt = f"""You are an AI safety alert system for drowning detection. Generate a SECOND urgent voice alert (15-25 words) because danger is still present after 6 seconds.

Detection details: {detection_summary}

Requirements:
- Urgent and direct tone
- Mention this is a continuing/persistent danger
- Clear call to action
- Natural for text-to-speech
- NO quotes, NO punctuation except periods and exclamation marks

Example: "Critical alert! Danger still detected in the pool area. Immediate intervention required. Check the pool now!"

Generate the alert:"""
        else:
            prompt = f"""You are an AI safety alert system for drowning detection. Generate an urgent voice alert (15-25 words) based on these detections.

Detection details: {detection_summary}

Requirements:
- Urgent and direct tone
- Clearly state the danger
- Clear call to action
- Natural for text-to-speech
- NO quotes, NO punctuation except periods and exclamation marks

Example: "Warning! Potential drowning detected in pool area. Please check immediately. This is an emergency alert!"

Generate the alert:"""
        
        return prompt
    
    def get_fallback_message(self, is_repeat: bool = False) -> str:
        """
        Get a fallback message if LLM generation fails
        
        Args:
            is_repeat: Whether this is a repeat alert
            
        Returns:
            Fallback alert message
        """
        if is_repeat:
            return "Critical alert! Danger is still present. Immediate action required. Check the area now!"
        else:
            return "Warning! Potential danger detected. Please check the monitored area immediately!"


# Example usage
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key or api_key == "your_groq_api_key_here":
        print("Please set GROQ_API_KEY in .env file")
        sys.exit(1)
    
    # Create service
    service = LLMService(api_key)
    
    # Test with sample detections
    test_detections = [
        {"class": "person", "confidence": 0.89},
        {"class": "drowning", "confidence": 0.76}
    ]
    
    # Generate first alert
    print("\n--- First Alert ---")
    message1 = service.generate_alert_message(test_detections, is_repeat=False)
    print(f"Result: {message1}")
    
    # Generate repeat alert
    print("\n--- Repeat Alert ---")
    message2 = service.generate_alert_message(test_detections, is_repeat=True)
    print(f"Result: {message2}")

