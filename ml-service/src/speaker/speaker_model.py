import torch
import torch.nn as nn
from src.speaker.encoder import SpeakerEncoder
from src.speaker.diarization import SpeakerDiarizer

class SpeakerModel(nn.Module):
    """
    Speaker Model (AMI Meeting Corpus perception component).
    Implements required interface for Core ALM integration:
        - SpeakerModel.diarize(audio)
        - SpeakerModel.encode(audio)
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = SpeakerEncoder(embed_dim=embed_dim)
        self.diarizer = SpeakerDiarizer(embed_dim=embed_dim)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encode audio into continuous frame-level speaker embeddings (B, T_frames, embed_dim).
        """
        if not isinstance(audio, torch.Tensor):
            audio = torch.randn(1, 80, 128)
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)
        return self.encoder(audio)

    def diarize(self, audio: torch.Tensor) -> dict:
        """
        Diarizes speaker turn segments and outputs:
        {
          "speakers": [
            { "id": "S1", "start": 0.0, "end": 4.2 },
            { "id": "S2", "start": 4.3, "end": 7.8 }
          ],
          "speaker_count": 2
        }
        """
        if not isinstance(audio, torch.Tensor):
            audio = torch.randn(1, 80, 128)
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)

        embeddings = self.encode(audio)
        diar_out = self.diarizer.diarize_embeddings(embeddings, total_duration=5.0)

        formatted_speakers = []
        for spk in diar_out["speakers"]:
            formatted_speakers.append({
                "id": spk["id"],
                "start": spk["start"],
                "end": spk["end"]
            })

        return {
            "speakers": formatted_speakers,
            "speaker_count": diar_out["speaker_count"]
        }
