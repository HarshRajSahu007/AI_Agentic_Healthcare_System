import os
import logging
import numpy as np
import torch
import librosa
import whisper
from typing import Dict , Any , List , Optional , Tuple
from transformers import pipeline
from pydub import AudioSegment

logger=logging.getLogger(__name__)

class AudioAgent:
    """
    Agent responsible for processing and analyzing audio inputs 
    such as cough sounds , breathing patterns , voice analysis, etc .
    Using OpenAI's Whisper Model for Transcription and custom PyTorch models
    for health-related audio analysis. 
    """

    def __init__(self,config: Dict[str,Any]):
        """
        Initialize the audio agent with configuration

        Args:
            config: Configuration dictionary with audio processing parameters 
        """
        self.config=config
        self.device=torch.device("cuda" if torch.cuda.is_available() and config.get("use_gpu",True) else "cpu")
        self.sample_rate=config.get("sample_rate",16000)
        self.models={}

        whisper_model_size=config.get("whisper_model_size","base")
        self.whisper_model=whisper.load_model(whisper_model_size,device=self.device)

        self._load_models()

        logger.info(f"AudioAgent initialized pn {self.device} with Whisper {whisper_model_size}")

    
    def _load_models(self):
        """load the required audio models based on configuration"""
        model_configs=self.config.get("models",{})

        for model_name, model_config in model_configs.items():
            try:
                if model_name == "cough_classifier":
                    self.models[model_name]=self._load_cough_classifier(model_config)

                elif model_name == "breathing_analyzer":
                    self.models[model_name]=self._load_breathing_analyzer(model_config)

                elif model_name == "voice_analyzer":
                    self.models[model_name]==self._load_voice_ananlyzer(model_config)

                elif model_name == "emotion_detector":
                    self.models[model_name]== pipeline(
                        "audio-classification",
                         model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                    )

                logger.info(f"Successfully loaded audio model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load audio model {model_name}: {str(e)}") 


    def _load_cough_classifier(self,config):
        """load cough sound classifier model using PyTorch"""

        model_path= config.get("model_path","models/audio_models/cough_classifier.pt")

        class CoughClassifier(torch.nn.Module):
            def __init__(self):
                super(CoughClassifier, self).__init__()
                self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=10, stride=5)
                self.pool = torch.nn.MaxPool1d(2)
                self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=10, stride=5)
                self.flatten = torch.nn.Flatten()
                self.fc1 = torch.nn.Linear(128 * 99, 128)  
                self.fc2 = torch.nn.Linear(128, 64)
                self.fc3 = torch.nn.Linear(64, 4) 
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.flatten(x)
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        model = CoughClassifier().to(self.device)
        

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
        else:
            logger.warning(f"Cough classifier model not found at {model_path}. Using untrained model.")
        
        return model

    def _load_breathing_analyzer(self,config):
        """Load breathing pattern analyzer model"""

        class BreathingAnalyzer(torch.nn.Module):
            def __init__(self):
                super(BreathingAnalyzer,self).__init__()
                self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=10, stride=5)
                self.pool = torch.nn.MaxPool1d(2)
                self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=10, stride=5)
                self.flatten = torch.nn.Flatten()
                self.fc1 = torch.nn.Linear(128 * 99, 128)
                self.fc2 = torch.nn.Linear(128, 64)
                self.fc3 = torch.nn.Linear(64, 5)
                self.relu = torch.nn.ReLU()
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.flatten(x)
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        model = BreathingAnalyzer().to(self.device)
        model_path = config.get("model_path", "models/audio_models/breathing_analyzer.pt")
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
        else:
            logger.warning(f"Breathing analyzer model not found at {model_path}. Using untrained model.")
            
        return model
    
    def _load_voice_analyzer(self, config):
        """Load voice characteristics analyzer model"""
        model_path = config.get("model_path", "models/audio_models/voice_analyzer.pt")
        
        # Placeholder for a real model implementation
        class VoiceAnalyzer(torch.nn.Module):
            def __init__(self):
                super(VoiceAnalyzer, self).__init__()
                self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=10, stride=5)
                self.pool = torch.nn.MaxPool1d(2)
                self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=10, stride=5)
                self.lstm = torch.nn.LSTM(128, 64, batch_first=True, bidirectional=True)
                self.fc = torch.nn.Linear(128, 3)  # 3 outputs: tremor, hoarseness, clarity
                
            def forward(self, x):
                x = self.pool(torch.nn.functional.relu(self.conv1(x)))
                x = self.pool(torch.nn.functional.relu(self.conv2(x)))
                x = x.permute(0, 2, 1)  # Reshape for LSTM
                x, _ = self.lstm(x)
                x = x[:, -1, :]  # Take the last output
                x = self.fc(x)
                return x
                
        model = VoiceAnalyzer().to(self.device)
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
        else:
            logger.warning(f"Voice analyzer model not found at {model_path}. Using untrained model.")
            
        return model