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
        drowning_analysis: Optional[Dict[str, Any]] = None,
        is_repeat: bool = False
    ) -> Optional[str]:
        """
        Generate contextual alert message based on detection data
        
        Args:
            detections: List of detection dictionaries containing:
                - class: Detection class name
                - confidence: Confidence score (0-1)
                - bbox: Bounding box coordinates (optional)
            drowning_analysis: Optional drowning behavior analysis with:
                - drowning_risk: Risk level (low/medium/high)
                - risk_score: Numeric risk score (0-1)
                - indicators: List of behavior indicators
                - people_detected: Number of people detected
                - pose_analysis: Optional pose-based distress analysis
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
            
            # Create prompt for Groq (with drowning analysis for detailed descriptions)
            prompt = self._create_prompt(detection_summary, drowning_analysis, is_repeat)
            
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
    
    def _create_prompt(self, detection_summary: str, drowning_analysis: Optional[Dict[str, Any]], is_repeat: bool) -> str:
        """Create prompt for Groq with detailed person description"""
        
        # Build detailed context from drowning analysis
        context_parts = [f"Detection details: {detection_summary}"]
        
        if drowning_analysis:
            risk_level = drowning_analysis.get("drowning_risk", "unknown").upper()
            risk_score = drowning_analysis.get("risk_score", 0.0)
            people_count = drowning_analysis.get("people_detected", 0)
            indicators = drowning_analysis.get("indicators", [])
            pose_analysis = drowning_analysis.get("pose_analysis")
            
            context_parts.append(f"Risk level: {risk_level} (score: {risk_score})")
            context_parts.append(f"People detected: {people_count}")
            
            if indicators:
                indicators_text = "; ".join(indicators)
                context_parts.append(f"Behavioral indicators: {indicators_text}")
            
            if pose_analysis:
                distress_level = pose_analysis.get("distress_level", "unknown")
                pose_indicators = pose_analysis.get("pose_results", [])
                if pose_indicators:
                    pose_details = []
                    for result in pose_indicators:
                        pose_ind = result.get("indicators", [])
                        if pose_ind:
                            pose_details.extend(pose_ind)
                    if pose_details:
                        context_parts.append(f"Pose distress ({distress_level}): {'; '.join(pose_details)}")
        
        full_context = "\n".join(context_parts)
        
        if is_repeat:
            prompt = f"""You are an AI safety alert system for drowning detection. Generate a SECOND urgent voice alert (15-30 words) because danger is still present.

{full_context}

Requirements:
- Focus on describing EXACTLY what the person is doing based on the indicators (e.g., "arms raised above head", "not moving horizontally", "vertical body position", "head below water level")
- This is a continuing danger so emphasize the persistence of these behaviors
- Use the severity level (LOW/MEDIUM/HIGH) to adjust urgency
- Be specific about the physical actions and position, NOT generic phrases like "check the area"
- Natural for text-to-speech
- NO quotes, NO punctuation except periods and exclamation marks

Example for HIGH risk: "Critical alert continues! Person with arms raised above head and no horizontal movement detected. Vertical sinking position observed. Immediate rescue required!"
Example for MEDIUM risk: "Alert persists! Individual not moving horizontally with reduced vertical motion. Possible struggling behavior continues. Urgent intervention needed!"
Example for LOW risk: "Ongoing alert. Person showing minimal movement in water with limited position changes. Continue monitoring closely!"

Generate the alert:"""
        else:
            prompt = f"""You are an AI safety alert system for drowning detection. Generate an urgent voice alert (15-30 words) based on these detections.

{full_context}

Requirements:
- Focus on describing EXACTLY what the person is doing based on the indicators (e.g., "arms raised above head", "not moving horizontally", "stuck at same vertical position", "head below shoulders")
- Use the severity level (LOW/MEDIUM/HIGH) to adjust urgency and tone
- Be specific about the physical actions and body position the person is exhibiting
- Do NOT use generic phrases like "check the area" or "check the pool" - instead describe what you observe about the person
- Natural for text-to-speech
- NO quotes, NO punctuation except periods and exclamation marks

Example for HIGH risk: "Emergency! Person detected with arms raised above head and no horizontal movement. Vertical body position with head below shoulders. Severe drowning indicators present!"
Example for MEDIUM risk: "Warning! Individual showing minimal horizontal movement and reduced vertical motion. Person stuck at same position. Possible distress detected!"
Example for LOW risk: "Alert! Person detected with limited movement patterns in water. No significant position changes observed. Monitor this individual!"

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
            return "Critical alert continues! Person showing no movement recovery. Distress behavior persisting. Immediate rescue action required!"
        else:
            return "Warning! Person detected with drowning indicators. Individual showing distress signals. Immediate attention required!"


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

