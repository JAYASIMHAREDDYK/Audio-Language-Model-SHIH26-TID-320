import torch
import torch.nn as nn
import torch.nn.functional as F
from src.alm.audio_encoder import AudioEncoder
from src.alm.projector import ModalityProjector
from src.alm.fusion_adapter import ALMFusionAdapter
from src.alm.reasoning_model import CoreALMReasoningModel
from src.fusion.temporal_alignment import TemporalAligner
from src.fusion.multimodal_projection import MultimodalProjector
from src.fusion.temporal_fusion import TemporalFusionAdapter

from src.asr.asr_model import ASRModel
from src.events.event_model import SoundEventModel
from src.speaker.speaker_model import SpeakerModel
from src.paralinguistic.paralinguistic_model import ParalinguisticModel

class CoreALM(nn.Module):
    """
    Core Audio Language Model (Core ALM) with Learned Latent Multimodal Fusion & Autoregressive Decoder.
    """
    def __init__(self, embed_dim=256, vocab_size=5000):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Base Specialized Audio Modality Encoders
        self.audio_encoder = AudioEncoder(embed_dim=embed_dim)
        self.asr_model = ASRModel(embed_dim=embed_dim)
        self.event_model = SoundEventModel(embed_dim=embed_dim)
        self.speaker_model = SpeakerModel(embed_dim=embed_dim)
        self.para_model = ParalinguisticModel(embed_dim=embed_dim)

        # 2. Temporal Alignment & Multimodal Projection Pipeline
        self.temporal_aligner = TemporalAligner(target_len=64, embed_dim=embed_dim)
        self.multimodal_projector = MultimodalProjector(fusion_dim=embed_dim)
        self.temporal_fusion_adapter = TemporalFusionAdapter(embed_dim=embed_dim)

        # 3. Core ALM Projector & Question Fusion Adapter
        self.modality_projector = ModalityProjector(fused_dim=embed_dim, question_dim=embed_dim, model_dim=embed_dim)
        self.alm_fusion_adapter = ALMFusionAdapter(embed_dim=embed_dim)

        # 4. Core ALM Question-Conditioned Reasoning Head
        self.reasoning_model = CoreALMReasoningModel(embed_dim=embed_dim, vocab_size=vocab_size)

    def encode_multimodal(self, audio_tensor: torch.Tensor, disable_modalities: list = None) -> torch.Tensor:
        disable_modalities = disable_modalities or []
        device = audio_tensor.device

        # ALWAYS run real encoders — using random noise during training
        # caused mode collapse because the decoder never learned audio-conditioned responses.
        speech_emb = self.asr_model.encode(audio_tensor).to(device)
        event_emb = self.event_model.encode(audio_tensor).to(device)
        speaker_emb = self.speaker_model.encode(audio_tensor).to(device)
        para_emb = self.para_model.encode(audio_tensor).to(device)
        audio_emb = self.audio_encoder(audio_tensor).to(device)

        if "speech" in disable_modalities or "asr" in disable_modalities:
            speech_emb = torch.zeros_like(speech_emb)
        if "events" in disable_modalities or "event" in disable_modalities:
            event_emb = torch.zeros_like(event_emb)
        if "speaker" in disable_modalities or "speakers" in disable_modalities:
            speaker_emb = torch.zeros_like(speaker_emb)
        if "paralinguistic" in disable_modalities or "para" in disable_modalities:
            para_emb = torch.zeros_like(para_emb)
        if "audio" in disable_modalities:
            audio_emb = torch.zeros_like(audio_emb)

        aligned_dict = self.temporal_aligner(
            speech_emb=speech_emb,
            event_emb=event_emb,
            speaker_emb=speaker_emb,
            para_emb=para_emb,
            audio_emb=audio_emb
        )

        projected_dict = self.multimodal_projector(
            speech_emb=aligned_dict["speech"],
            event_emb=aligned_dict["events"],
            speaker_emb=aligned_dict["speakers"],
            para_emb=aligned_dict["paralinguistic"],
            audio_emb=aligned_dict["audio"]
        )

        fused_multimodal = self.temporal_fusion_adapter(projected_dict)
        return fused_multimodal, projected_dict

    def forward(self, audio_tensor: torch.Tensor, question_text: str = "Where is the speaker likely to be?", disable_modalities: list = None, tgt_tokens: torch.Tensor = None) -> dict:
        device = audio_tensor.device

        fused_multimodal, projected_dict = self.encode_multimodal(audio_tensor, disable_modalities=disable_modalities)

        q_emb = self.modality_projector.embed_question_text(question_text, device=device)
        if q_emb.shape[0] == 1 and audio_tensor.shape[0] > 1:
            q_emb = q_emb.repeat(audio_tensor.shape[0], 1, 1)

        proj_fused, proj_question = self.modality_projector(fused_multimodal, q_emb)

        joint_context = self.alm_fusion_adapter(proj_fused, proj_question)

        reasoning_output = self.reasoning_model(joint_context, tgt_tokens=tgt_tokens)

        if self.training or tgt_tokens is not None:
            asr_out, events_out, speaker_out, para_out = {}, {}, {}, {}
        else:
            disable_modalities = disable_modalities or []
            asr_out = {} if ("speech" in disable_modalities or "asr" in disable_modalities) else self.asr_model.transcribe(audio_tensor)
            events_out = {"events": []} if ("events" in disable_modalities or "event" in disable_modalities or "audio" in disable_modalities) else self.event_model.detect(audio_tensor)
            speaker_out = {"speakers": []} if ("speaker" in disable_modalities or "speakers" in disable_modalities) else self.speaker_model.diarize(audio_tensor)
            para_out = {"emotion": "neutral", "arousal": "low"} if ("paralinguistic" in disable_modalities or "para" in disable_modalities) else self.para_model.analyze(audio_tensor)

        reasoning_output.update({
            "fused_multimodal_tensor": fused_multimodal,
            "joint_context_tensor": joint_context,
            "projected_modalities": projected_dict,
            "speech": asr_out,
            "audio_events": events_out,
            "speakers": speaker_out,
            "paralinguistic": para_out
        })

        return reasoning_output
