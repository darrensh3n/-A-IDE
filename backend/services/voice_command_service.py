"""
Voice Command Service for handling speech recognition and command execution
"""

import speech_recognition as sr
import threading
import time
from typing import Optional, Callable, Dict, Any
import logging

logger = logging.getLogger(__name__)

class VoiceCommandService:
    """Service for processing voice commands"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False
        self.command_callbacks: Dict[str, Callable] = {}
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Define available commands
        self.commands = {
            "start camera": self._start_camera,
            "start monitoring": self._start_camera,
            "begin monitoring": self._start_camera,
            "turn on camera": self._start_camera,
            "stop camera": self._stop_camera,
            "stop monitoring": self._stop_camera,
            "end monitoring": self._stop_camera,
            "turn off camera": self._stop_camera,
            "emergency": self._emergency,
            "help": self._emergency,
            "stop alerts": self._stop_alerts,
            "silence alerts": self._stop_alerts,
            "clear alerts": self._clear_alerts,
            "system status": self._system_status,
        }
        
        logger.info("Voice Command Service initialized")
    
    def register_callback(self, command: str, callback: Callable):
        """Register a callback function for a specific command"""
        self.command_callbacks[command] = callback
        logger.info(f"Registered callback for command: {command}")
    
    def _start_camera(self):
        """Handle start camera command"""
        logger.info("Voice command: Starting camera")
        if "start_camera" in self.command_callbacks:
            self.command_callbacks["start_camera"]()
    
    def _stop_camera(self):
        """Handle stop camera command"""
        logger.info("Voice command: Stopping camera")
        if "stop_camera" in self.command_callbacks:
            self.command_callbacks["stop_camera"]()
    
    def _emergency(self):
        """Handle emergency command"""
        logger.info("Voice command: Emergency triggered")
        if "emergency" in self.command_callbacks:
            self.command_callbacks["emergency"]()
    
    def _stop_alerts(self):
        """Handle stop alerts command"""
        logger.info("Voice command: Stopping alerts")
        if "stop_alerts" in self.command_callbacks:
            self.command_callbacks["stop_alerts"]()
    
    def _clear_alerts(self):
        """Handle clear alerts command"""
        logger.info("Voice command: Clearing alerts")
        if "clear_alerts" in self.command_callbacks:
            self.command_callbacks["clear_alerts"]()
    
    def _system_status(self):
        """Handle system status command"""
        logger.info("Voice command: System status requested")
        if "system_status" in self.command_callbacks:
            self.command_callbacks["system_status"]()
    
    def listen_for_command(self, timeout: int = 5) -> Optional[str]:
        """
        Listen for a voice command
        
        Args:
            timeout: Maximum time to listen in seconds
            
        Returns:
            Detected command string or None
        """
        try:
            with self.microphone as source:
                logger.info("Listening for voice command...")
                # Listen for audio with timeout
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=3)
                
            # Recognize speech using Google's service
            text = self.recognizer.recognize_google(audio).lower()
            logger.info(f"Heard: {text}")
            
            # Check if any command matches
            for command, handler in self.commands.items():
                if command in text:
                    logger.info(f"Command detected: {command}")
                    handler()
                    return command
            
            logger.info("No recognized command found")
            return None
            
        except sr.WaitTimeoutError:
            logger.info("No speech detected within timeout")
            return None
        except sr.UnknownValueError:
            logger.info("Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in voice recognition: {e}")
            return None
    
    def start_continuous_listening(self, callback: Optional[Callable] = None):
        """
        Start continuous listening for voice commands in a separate thread
        
        Args:
            callback: Optional callback function to call when command is detected
        """
        if self.is_listening:
            logger.warning("Already listening for voice commands")
            return
        
        self.is_listening = True
        
        def listen_loop():
            while self.is_listening:
                try:
                    command = self.listen_for_command(timeout=1)
                    if command and callback:
                        callback(command)
                    time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                except Exception as e:
                    logger.error(f"Error in continuous listening: {e}")
                    time.sleep(1)
        
        # Start listening in a separate thread
        self.listening_thread = threading.Thread(target=listen_loop, daemon=True)
        self.listening_thread.start()
        logger.info("Started continuous voice command listening")
    
    def stop_continuous_listening(self):
        """Stop continuous listening for voice commands"""
        if not self.is_listening:
            logger.warning("Not currently listening for voice commands")
            return
        
        self.is_listening = False
        if hasattr(self, 'listening_thread'):
            self.listening_thread.join(timeout=2)
        logger.info("Stopped continuous voice command listening")
    
    def get_available_commands(self) -> list:
        """Get list of available voice commands"""
        return list(self.commands.keys())
    
    def test_microphone(self) -> bool:
        """Test if microphone is working"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            logger.info("Microphone test successful")
            return True
        except Exception as e:
            logger.error(f"Microphone test failed: {e}")
            return False
