import torch
import torchaudio
from transformers import WhisperProcessor , WhisperForConditionalGeneration

class AudioProcessingAgent:
    def __init__(self,config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor=WhisperProcessor.from_pretrained(config["whisper_model"])
        self.model=WhisperForConditionalGeneration.from_pretrained(config["whisper_model"]).to(self.device)
        self.sample_rate=config["sample_rate"]
    
    def process_audio(self,audio_path):
        waveform,sample_rate=torchaudio.load(audio_path)
        if sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate,self.sample_rate)
            waveform = resampler(waveform)
        input_features=self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        ).input_features.to(self.device)

        prediction_ids=self.model.generate(input_features)
        transcription=self.processor.batch_decode(prediction_ids,skip_special_tokens=True)[0]
        return {
            "transciption":transcription,
            "audio_features":input_features.cpu().numpy()
        }