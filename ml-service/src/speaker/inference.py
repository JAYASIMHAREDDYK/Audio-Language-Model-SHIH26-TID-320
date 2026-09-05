import os
import torch
from src.speaker.encoder import SpeakerEncoder
from src.speaker.diarization import SpeakerDiarizer

class SpeakerInferencePipeline:
    """
    Inference Pipeline for Speaker Perception Model.
    """
    def __init__(self, checkpoint_path: str = None, device: str = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = SpeakerEncoder(embed_dim=256).to(self.device)
        self.diarizer = SpeakerDiarizer(embed_dim=256).to(self.device)
        self.encoder.eval()

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                self.encoder.load_state_dict(ckpt.get("encoder_state_dict", ckpt), strict=False)
            except Exception as e:
                print(f"[SpeakerInferencePipeline] Checkpoint load note: {e}")

    def run(self, audio_tensor: torch.Tensor, duration: float = 5.0) -> dict:
        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = audio_tensor.to(self.device)
        
        with torch.no_grad():
            speaker_embeddings = self.encoder(audio_tensor)
            diar_result = self.diarizer.diarize_embeddings(speaker_embeddings, total_duration=duration)
            
        return {
            "speakers": diar_result["speakers"],
            "speaker_count": diar_result["speaker_count"],
            "embeddings": speaker_embeddings
        }
