# Speech Recognition with OpenAI Whisper

## Introduction

This chapter focuses on implementing speech recognition for the Vision-Language-Action (VLA) system using OpenAI Whisper. Whisper is a state-of-the-art automatic speech recognition (ASR) system developed by OpenAI that enables accurate conversion of spoken language into text format that can be processed by our VLA system.

## Understanding OpenAI Whisper

### What is Whisper

Whisper is a general-purpose speech recognition model developed by OpenAI. It is trained on a large dataset of diverse audio and is known for its robustness and accuracy across different languages and accents. The model is particularly effective for:

- Automatic speech recognition (ASR)
- Voice activity detection
- Language identification
- Speech-to-text translation

### Whisper Architecture

Whisper uses a transformer-based encoder-decoder architecture:

- **Encoder**: Processes audio input and extracts acoustic features
- **Decoder**: Converts acoustic representations to text tokens
- **Multilingual capability**: Trained on 98 languages
- **Robustness**: Performs well even in challenging acoustic environments

### Whisper Models

OpenAI provides several Whisper models with different sizes and performance characteristics:

- **tiny**: Fastest with lower accuracy, ~39M parameters
- **base**: Good balance of speed and accuracy, ~74M parameters
- **small**: Better accuracy, ~244M parameters
- **medium**: High accuracy, ~769M parameters
- **large**: Highest accuracy, ~1550M parameters

## Implementing Whisper in Robotics Context

### Robot Voice Command Recognition

When integrating Whisper into our humanoid robot system, we'll need to consider several factors:

1. **Audio Input Quality**: Microphone placement and quality will affect recognition performance
2. **Real-time Processing**: Latency requirements for responsive voice interaction
3. **Language Selection**: Specifying the language to optimize recognition accuracy
4. **Contextual Understanding**: Using robot-specific vocabulary and commands

### Basic Whisper Implementation

Here's a basic implementation of Whisper for robot voice command recognition:

```python
import torch
import whisper
import pyaudio
import wave
import threading
import queue

class VoiceCommandRecognizer:
    def __init__(self, model_size="base", device="cpu"):
        """
        Initialize the Whisper-based voice command recognizer
        """
        self.model_size = model_size
        self.device = device
        self.model = whisper.load_model(model_size).to(device)
        
        # Audio recording parameters
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.record_seconds = 5  # Record 5-second chunks
        
        self.audio_queue = queue.Queue()
        
    def record_audio(self, duration=5):
        """
        Record audio from microphone for the specified duration
        """
        p = pyaudio.PyAudio()
        
        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk)
        
        print(f"Recording for {duration} seconds...")
        frames = []
        
        for i in range(0, int(self.rate / self.chunk * duration)):
            data = stream.read(self.chunk)
            frames.append(data)
        
        print("Finished recording.")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save audio to temporary file
        filename = "temp_recording.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(p.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        return filename
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio file using Whisper
        """
        result = self.model.transcribe(audio_path)
        return result["text"].strip()
    
    def process_voice_command(self, duration=5):
        """
        Complete process of recording and transcribing a voice command
        """
        audio_file = self.record_audio(duration)
        transcription = self.transcribe_audio(audio_file)
        
        # Clean up temporary file
        import os
        os.remove(audio_file)
        
        return transcription

# Example usage
if __name__ == "__main__":
    recognizer = VoiceCommandRecognizer(model_size="base")
    command = recognizer.process_voice_command(duration=3)
    print(f"Recognized command: '{command}'")
```

## Optimizing Whisper for Robot Applications

### Model Selection

For embedded robotics applications, consider the trade-off between:

- **Accuracy**: Larger models provide better accuracy but require more computation
- **Latency**: Smaller models process faster but may have lower accuracy
- **Hardware Requirements**: Larger models need more RAM and processing power

### Audio Preprocessing

Improve recognition accuracy with proper audio preprocessing:

```python
import librosa
import numpy as np

def preprocess_audio_for_whisper(audio_path, target_sr=16000):
    """
    Preprocess audio to optimize for Whisper recognition
    """
    # Load audio at Whisper's expected sample rate
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # Normalize audio amplitude
    audio = audio / np.max(np.abs(audio))
    
    # Remove silence at beginning and end
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
    
    # Save processed audio
    processed_path = "processed_" + audio_path
    librosa.output.write_wav(processed_path, audio_trimmed, sr)
    
    return processed_path
```

### Context-Specific Vocabulary

While Whisper doesn't support traditional keyword spotting, you can implement post-processing to match recognized text with predefined robot commands:

```python
class CommandMatcher:
    def __init__(self):
        # Define robot-specific commands
        self.commands = {
            "move_forward": ["go forward", "move ahead", "move forward", "go straight"],
            "move_backward": ["go backward", "move back", "go back"],
            "turn_left": ["turn left", "rotate left", "go left"],
            "turn_right": ["turn right", "rotate right", "go right"],
            "pick_up": ["pick up", "grasp", "grab", "take"],
            "put_down": ["put down", "release", "place", "drop"],
            "stop": ["stop", "halt", "cease", "pause"]
        }
    
    def match_command(self, transcription):
        """
        Match transcription to known robot commands
        """
        transcription_lower = transcription.lower()
        
        for command, patterns in self.commands.items():
            for pattern in patterns:
                if pattern in transcription_lower:
                    return command
        
        return "unknown"
```

## Integration with Robot Control

### Voice Command Processing Pipeline

The complete pipeline for processing voice commands using Whisper:

```
Voice Input → Audio Recording → Whisper Transcription → Command Matching → Robot Action
```

### Real-Time Considerations

For real-time applications, consider implementing streaming recognition using:

- Microphone audio stream
- Continuous buffering of audio chunks
- Periodic recognition on buffered audio
- Wake word detection to activate recognition

## Quality Assurance

### Recognition Accuracy

Monitor and improve recognition accuracy by:

- Testing with different speakers and accents
- Evaluating in noisy environments
- Tracking command-to-intent mapping accuracy
- Collecting unrecognized commands for model improvement

### Error Handling

Implement robust error handling:

```python
def safe_process_command(self, duration=5):
    """
    Safely process voice command with error handling
    """
    try:
        command_text = self.process_voice_command(duration)
        if not command_text:
            return {"status": "error", "message": "Empty transcription"}
        
        matched_command = self.command_matcher.match_command(command_text)
        return {
            "status": "success", 
            "command": matched_command, 
            "raw_text": command_text
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

## Troubleshooting Common Issues

### Poor Recognition Quality

- Check microphone quality and placement
- Reduce background noise during recording
- Use preprocessing to normalize audio
- Select appropriate Whisper model size

### Latency Issues

- Use appropriate model size for your hardware
- Optimize audio processing pipeline
- Consider chunk-based processing for real-time performance

### Language-Specific Issues

- Ensure correct language identification
- Consider fine-tuning for domain-specific vocabulary
- Account for linguistic variations in robot commands

## Summary

In this chapter, we explored the implementation of speech recognition using OpenAI Whisper in the context of Vision-Language-Action systems for humanoid robots. We covered the architecture of Whisper models, best practices for integration with robotic systems, and strategies for optimizing recognition accuracy in real-world applications.

The next chapter will focus on mapping these recognized voice commands to specific robot intents and actions, building the bridge between speech recognition and robot control.