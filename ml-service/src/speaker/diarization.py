import os
import torch
import torch.nn as nn
import numpy as np

# Try importing diarize_simple from speaker module
try:
    from src.speaker.diarize_simple import diarize_simple as external_diarize_simple
    HAS_EXTERNAL_DIARIZE = True
except ImportError:
    HAS_EXTERNAL_DIARIZE = False

class SpeakerDiarizer(nn.Module):
    """
    Speaker Diarization Module.
    Segments audio into speaker turns (S1, S2, ...) using biometric embeddings & clustering.
    Supports external bundle diarize_simple function when available.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.turn_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4) # Max 4 speaker turns
        )

    def diarize_embeddings(self, embeddings: torch.Tensor, total_duration: float = 5.0, wav_path: str = None) -> dict:
        return self.diarize_audio(embeddings=embeddings, duration=total_duration, wav_path=wav_path)

    def diarize_audio(self, embeddings: torch.Tensor, duration: float = 5.0, wav_path: str = None) -> dict:
        """
        Args:
            embeddings: (B, T_frames, embed_dim)
            duration: audio duration in seconds
            wav_path: path to audio file if available
        Returns:
            {
              "speakers": [
                {"id": "S1", "start": 0.0, "end": 2.5},
                {"id": "S2", "start": 2.5, "end": 5.0}
              ],
              "speaker_count": 2
            }
        """
        if wav_path and os.path.exists(wav_path) and HAS_EXTERNAL_DIARIZE:
            try:
                res = external_diarize_simple(wav_path)
                if isinstance(res, list) and len(res) > 0:
                    segments = []
                    spk_set = set()
                    for idx, item in enumerate(res):
                        spk_id = item.get("speaker", f"S{idx+1}").replace("SPEAKER_", "S")
                        spk_set.add(spk_id)
                        t_start = item.get("time", 0.0)
                        t_end = min(duration, round(t_start + 0.75, 2))
                        segments.append({"id": spk_id, "start": t_start, "end": t_end})
                    return {
                        "speakers": segments,
                        "speaker_count": max(1, len(spk_set))
                    }
            except Exception as e:
                print(f"[SpeakerDiarizer] External bundle note: {e}")

        # Neural frame-level clustering fallback
        if embeddings.ndim == 3:
            emb_b0 = embeddings[0]
        else:
            emb_b0 = embeddings

        T_frames = emb_b0.size(0)
        duration_per_frame = duration / max(1, T_frames)

        logits = self.turn_head(emb_b0) # (T, 4)
        spk_preds = torch.argmax(logits, dim=-1).cpu().tolist() # (T,)

        speaker_segments = []
        unique_speakers = set()

        curr_spk = None
        start_idx = 0

        for t_idx, spk_id in enumerate(spk_preds):
            label = f"S{spk_id + 1}"
            unique_speakers.add(label)
            if label != curr_spk:
                if curr_spk is not None:
                    speaker_segments.append({
                        "id": curr_spk,
                        "start": round(start_idx * duration_per_frame, 2),
                        "end": round(t_idx * duration_per_frame, 2)
                    })
                curr_spk = label
                start_idx = t_idx

        if curr_spk is not None:
            speaker_segments.append({
                "id": curr_spk,
                "start": round(start_idx * duration_per_frame, 2),
                "end": round(duration, 2)
            })

        return {
            "speakers": speaker_segments if speaker_segments else [{"id": "S1", "start": 0.0, "end": duration}],
            "speaker_count": max(1, len(unique_speakers))
        }
