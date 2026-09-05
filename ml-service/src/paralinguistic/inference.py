import os
import torch
from src.paralinguistic.encoder import ParalinguisticEncoder
from src.paralinguistic.emotion import EmotionClassifier

class ParalinguisticInferencePipeline:
    """
    Inference Pipeline for Paralinguistic Perception Model.
    """
    def __init__(self, checkpoint_path: str = None, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = ParalinguisticEncoder(embed_dim=256).to(self.device)
        self.classifier = EmotionClassifier(embed_dim=256).to(self.device)
        self.encoder.eval()
        self.classifier.eval()

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                self.encoder.load_state_dict(ckpt.get("encoder_state_dict", ckpt), strict=False)
                self.classifier.load_state_dict(ckpt.get("classifier_state_dict", ckpt), strict=False)
            except Exception as e:
                print(f"[ParalinguisticInferencePipeline] Checkpoint load note: {e}")

    def run(self, audio_tensor: torch.Tensor) -> dict:
        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        
        with torch.no_grad():
            embeddings = self.encoder(audio_tensor)
            pred = self.classifier.predict(embeddings)
            
        return {
            "emotion": pred["emotion"],
            "confidence": pred["confidence"],
            "arousal": pred["arousal"],
            "speaking_style": pred["speaking_style"],
            "urgency": pred["urgency"],
            "embeddings": embeddings
        }
