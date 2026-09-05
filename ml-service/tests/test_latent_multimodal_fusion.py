import os
import sys
import torch
import pytest

# Ensure repository root and ml-service are in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(BASE_DIR, "ml-service"))
sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.asr.asr_model import ASRModel
from src.events.event_model import SoundEventModel
from src.speaker.speaker_model import SpeakerModel
from src.paralinguistic.paralinguistic_model import ParalinguisticModel
from src.fusion.temporal_alignment import TemporalAligner
from src.fusion.temporal_fusion import TemporalFusionAdapter
from src.alm.alm_model import CoreALM
from src.alm.inference import ALMInferencePipeline

def test_submodel_tensor_encoders():
    print("\n--- Test 1: Specialized Submodel Tensor Encoders ---")
    audio = torch.randn(1, 80, 128)
    
    asr = ASRModel(embed_dim=256)
    events = SoundEventModel(embed_dim=256)
    speaker = SpeakerModel(embed_dim=256)
    para = ParalinguisticModel(embed_dim=256)
    
    asr_emb = asr.encode(audio)
    event_emb = events.encode(audio)
    spk_emb = speaker.encode(audio)
    para_emb = para.encode(audio)
    
    assert isinstance(asr_emb, torch.Tensor), "ASR encode must return torch.Tensor"
    assert isinstance(event_emb, torch.Tensor), "SoundEvent encode must return torch.Tensor"
    assert isinstance(spk_emb, torch.Tensor), "Speaker encode must return torch.Tensor"
    assert isinstance(para_emb, torch.Tensor), "Paralinguistic encode must return torch.Tensor"
    
    assert asr_emb.ndim == 3 and asr_emb.size(-1) == 256
    assert event_emb.ndim == 3 and event_emb.size(-1) == 256
    assert spk_emb.ndim == 3 and spk_emb.size(-1) == 256
    assert para_emb.ndim == 3 and para_emb.size(-1) == 256
    print("✔ All 4 specialized submodels return valid continuous latent feature tensors (B, T, 256).")

def test_temporal_alignment_and_fusion():
    print("\n--- Test 2: Temporal Alignment & Cross-Modal Fusion ---")
    aligner = TemporalAligner(target_len=64, embed_dim=256)
    fusion_adapter = TemporalFusionAdapter(embed_dim=256)
    
    s = torch.randn(1, 30, 256)
    e = torch.randn(1, 40, 256)
    sp = torch.randn(1, 20, 256)
    p = torch.randn(1, 10, 256)
    a = torch.randn(1, 64, 256)
    
    aligned = aligner(speech_emb=s, event_emb=e, speaker_emb=sp, para_emb=p, audio_emb=a, target_len=64)
    assert aligned["speech"].shape == (1, 64, 256)
    assert aligned["events"].shape == (1, 64, 256)
    assert aligned["speakers"].shape == (1, 64, 256)
    assert aligned["paralinguistic"].shape == (1, 64, 256)
    
    fused = fusion_adapter(aligned)
    assert fused.shape == (1, 64, 256)
    print("✔ Temporal alignment resamples multi-rate feature streams to unified timeline and fuses via Transformer.")

def test_core_alm_latent_fusion_and_ablation():
    print("\n--- Test 3: Core ALM Latent Multimodal Fusion & Modality Sensitivity ---")
    model = CoreALM(embed_dim=256)
    model.eval()
    
    audio = torch.randn(1, 80, 128)
    question = "Where is the speaker likely to be?"
    
    # 1. Full multimodal fused tensor
    fused_full = model.encode_multimodal(audio)
    if isinstance(fused_full, tuple):
        fused_full = fused_full[0]
    out_full = model(audio, question_text=question)
    
    # 2. Ablated multimodal tensor (events modality removed)
    fused_no_events = model.encode_multimodal(audio, disable_modalities=["events"])
    if isinstance(fused_no_events, tuple):
        fused_no_events = fused_no_events[0]
    out_no_events = model(audio, question_text=question, disable_modalities=["events"])
    
    # 3. Ablated multimodal tensor (speech modality removed)
    fused_no_speech = model.encode_multimodal(audio, disable_modalities=["speech"])
    if isinstance(fused_no_speech, tuple):
        fused_no_speech = fused_no_speech[0]
    out_no_speech = model(audio, question_text=question, disable_modalities=["speech"])
    
    diff_events = (fused_full - fused_no_events).abs().sum().item()
    diff_speech = (fused_full - fused_no_speech).abs().sum().item()
    
    print(f"  Fused full vs no_events abs diff: {diff_events:.4f}")
    print(f"  Fused full vs no_speech abs diff : {diff_speech:.4f}")
    
    assert diff_events > 1e-4, "Removing events modality MUST change fused representation mathematically!"
    assert diff_speech > 1e-4, "Removing speech modality MUST change fused representation mathematically!"
    
    assert out_full["pooled_rep"].shape == (1, 256)
    assert out_full["evidence_scores"].shape[1] > 0
    assert out_full["confidence"].shape == (1, 1)
    print("✔ Verified Core ALM consumes latent multimodal feature tensors; modality ablation produces mathematically distinct representations.")

def test_end_to_end_inference_pipeline():
    print("\n--- Test 4: End-to-End ALMInferencePipeline Execution ---")
    pipeline = ALMInferencePipeline(checkpoint_path=os.path.join(BASE_DIR, "ml-service", "checkpoints", "best_checkpoint.pt"))
    
    res = pipeline.analyze(torch.randn(1, 80, 128), question="What sound occurred immediately after the announcement?")
    
    assert "answer" in res and len(res["answer"]) > 0
    assert "confidence" in res and 0.5 <= res["confidence"] <= 1.0
    assert "speech" in res
    assert "speakers" in res
    assert "audio_events" in res
    assert "paralinguistic" in res
    assert "scene" in res
    assert "evidence" in res
    print("✔ End-to-end ALMInferencePipeline analysis verified successfully.")

if __name__ == "__main__":
    test_submodel_tensor_encoders()
    test_temporal_alignment_and_fusion()
    test_core_alm_latent_fusion_and_ablation()
    test_end_to_end_inference_pipeline()
    print("\n" + "="*70)
    print("ALL LATENT MULTIMODAL FUSION TESTS PASSED CLEANLY!")
    print("="*70)
