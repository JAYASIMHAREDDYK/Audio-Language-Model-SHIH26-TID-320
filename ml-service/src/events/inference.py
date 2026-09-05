import os
import torch
from src.events.encoder import SoundEventEncoder
from src.events.classifier import SoundEventClassifier

class SoundEventInferencePipeline:
    """
    Inference Pipeline for Sound Event Perception Model.
    """
    def __init__(self, checkpoint_path: str = None, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = SoundEventEncoder(embed_dim=256).to(self.device)
        self.classifier = SoundEventClassifier(embed_dim=256).to(self.device)
        self.encoder.eval()
        self.classifier.eval()

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
                self.encoder.load_state_dict(ckpt.get("encoder_state_dict", ckpt), strict=False)
                self.classifier.load_state_dict(ckpt.get("classifier_state_dict", ckpt), strict=False)
            except Exception as e:
                print(f"[SoundEventInferencePipeline] Checkpoint load note: {e}")

    def run(self, audio_tensor: torch.Tensor, duration: float = 5.0) -> dict:
        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        
        with torch.no_grad():
            embeddings = self.encoder(audio_tensor)
            events = self.classifier.detect_events(embeddings, total_duration=duration)
            
        return {
            "events": events,
            "embeddings": embeddings
        }
