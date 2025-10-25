"""
Voice Alert Service using Fish Audio TTS
Converts text alerts to speech and plays them.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional
from fish_audio_sdk import Session, TTSRequest


class VoiceAlertService:
    """Service for generating voice alerts using Fish Audio TTS"""
    
    def __init__(self, api_key: str, output_dir: str = "audio_alerts"):
        """
        Initialize the voice alert service
        
        Args:
            api_key: Fish Audio API key
            output_dir: Directory to save generated audio files
        """
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize Fish Audio session
        try:
            self.session = Session(api_key)
            print(f"✓ Fish Audio TTS initialized")
        except Exception as e:
            print(f"✗ Failed to initialize Fish Audio: {e}")
            self.session = None
    
    def generate_audio(self, text: str, voice: str = "s1") -> Optional[bytes]:
        """
        Generate audio from text using Fish Audio TTS
        
        Args:
            text: The text to convert to speech
            voice: Voice model to use (default: "s1")
            
        Returns:
            Audio data as bytes, or None if generation failed
        """
        if not self.session:
            print("✗ Fish Audio session not initialized")
            return None
        
        try:
            print(f"🎤 Generating audio: '{text}'")
            
            # Generate speech using Fish Audio
            audio_chunks = []
            for chunk in self.session.tts(
                TTSRequest(text=text, backend=voice)
            ):
                audio_chunks.append(chunk)
            
            # Combine all chunks
            audio_data = b''.join(audio_chunks)
            print(f"✓ Audio generated successfully ({len(audio_data)} bytes)")
            return audio_data
            
        except Exception as e:
            print(f"✗ Error generating audio: {e}")
            return None
    
    def save_audio(self, audio_data: bytes, filename: Optional[str] = None) -> Optional[str]:
        """
        Save audio data to a file
        
        Args:
            audio_data: Audio data as bytes
            filename: Optional filename (will generate timestamp-based name if not provided)
            
        Returns:
            Path to saved file, or None if save failed
        """
        if not audio_data:
            return None
        
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"alert_{timestamp}.mp3"
            
            filepath = self.output_dir / filename
            
            with open(filepath, "wb") as f:
                f.write(audio_data)
            
            print(f"✓ Audio saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"✗ Error saving audio: {e}")
            return None
    
    def play_audio(self, filepath: str) -> bool:
        """
        Play audio file using system audio player
        
        Args:
            filepath: Path to audio file
            
        Returns:
            True if playback started successfully, False otherwise
        """
        try:
            import platform
            import subprocess
            
            system = platform.system()
            
            if system == "Darwin":  # macOS
                subprocess.Popen(["afplay", filepath])
            elif system == "Linux":
                subprocess.Popen(["mpg123", filepath])
            elif system == "Windows":
                subprocess.Popen(["start", filepath], shell=True)
            else:
                print(f"✗ Unsupported platform for audio playback: {system}")
                return False
            
            print(f"🔊 Playing audio: {filepath}")
            return True
            
        except Exception as e:
            print(f"✗ Error playing audio: {e}")
            return False
    
    def generate_and_play(self, text: str, save: bool = True, play: bool = True) -> Optional[str]:
        """
        Generate audio from text and optionally save/play it
        
        Args:
            text: The text to convert to speech
            save: Whether to save the audio file
            play: Whether to play the audio
            
        Returns:
            Path to saved audio file if saved, None otherwise
        """
        # Generate audio
        audio_data = self.generate_audio(text)
        if not audio_data:
            return None
        
        # Save audio
        filepath = None
        if save:
            filepath = self.save_audio(audio_data)
        
        # Play audio
        if play and filepath:
            self.play_audio(filepath)
        
        return filepath


# Example usage
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("FISH_AUDIO_API_KEY")
    
    if not api_key or api_key == "your_fish_audio_api_key_here":
        print("Please set FISH_AUDIO_API_KEY in .env file")
        sys.exit(1)
    
    # Create service
    service = VoiceAlertService(api_key)
    
    # Generate test alert
    test_message = "Warning! Potential drowning detected. Please check the pool immediately."
    service.generate_and_play(test_message)

