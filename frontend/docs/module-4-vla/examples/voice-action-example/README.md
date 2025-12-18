# Voice-to-Action Example

This directory contains a minimal runnable example demonstrating voice-to-action mapping for humanoid robots.

## Files Included

- `whisper_integration.py`: Example implementation of OpenAI Whisper for speech recognition
- `intent_mapper.py`: Example of intent classification and mapping
- `robot_action_mapper.py`: Example of mapping intents to robot actions
- `voice_interface_demo.py`: Complete demonstration of the voice-to-action pipeline
- `README.md`: This file

## How to Run

1. Install required dependencies: `pip install openai-whisper pyaudio pyttsx3`
2. Run the complete example: `python voice_interface_demo.py`
3. Speak voice commands when prompted
4. The system will recognize the command and map it to a simulated robot action

## Purpose

This example demonstrates:
- Speech recognition using OpenAI Whisper
- Intent classification from voice commands
- Mapping intents to robot actions
- Providing feedback to the user