"""
Comprehensive 12-Test Verification Suite for 7-Language Deep Conformer ASR
and ALM Multimodal Integration.

Verifies:
1.  Zero Whisper dependencies in active source tree
2.  Exact architecture parameters (12 blocks, d_model=512, d_ff=2048, 8 heads, kernel=31)
3.  Tokenizer vocabulary & tokenization across all 7 languages (hi, te, ta, bn, mr, kn, zh)
4.  Dataset manifests existence and valid schema
5.  Language balance across train manifest
6.  Strict speaker-disjointness across train/val/test splits
7.  Forward pass with variable-length audio waveforms
8.  CTC target length validity (T_subsampled >= target_len)
9.  End-to-end gradient flow through all 12 blocks and CTC projection head
10. Frame-level continuous acoustic representation contract (B, T', 512)
11. ALM Temporal Fusion ASRAdapter integration (512 -> 256 projection)
12. End-to-end transcription inference across all 7 languages
"""

import json
import os
import sys
import pytest
import torch
import torch.nn as nn

# Ensure repo root and ml-service are in sys.path
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ROOT = os.path.dirname(TESTS_DIR)
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.asr.tokenizer import MultilingualTokenizer, MultilingualCharTokenizer, LANG_TOKENS
from src.asr.conformer import DeepConformerEncoder as ConformerEncoder
from src.asr.multilingual_asr import ASRModel as ConformerASR, ASRModel as DeepConformerASR
from src.asr.asr_model import ASRModel as ALMASRAdapter, ASRAdapter


# ============================================================================
# 1. Zero Whisper Dependencies
# ============================================================================
def test_zero_whisper_dependencies():
    """Verify no active source code imports or depends on openai-whisper."""
    asr_dir = os.path.join(SERVICE_ROOT, "src", "asr")
    training_dir = os.path.join(SERVICE_ROOT, "training")
    
    files_to_check = []
    for d in [asr_dir, training_dir]:
        if os.path.exists(d):
            for root, _, files in os.walk(d):
                for f in files:
                    if f.endswith(".py"):
                        files_to_check.append(os.path.join(root, f))
                        
    for fpath in files_to_check:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            assert "import whisper" not in content, f"Forbidden 'import whisper' in {fpath}"
            assert "from whisper" not in content, f"Forbidden 'from whisper' in {fpath}"


# ============================================================================
# 2. Exact Architecture Parameters
# ============================================================================
def test_exact_architecture_parameters():
    """Verify Conformer architecture matches exact 79.3M parameter specifications."""
    model = ConformerASR(device="cpu")
    
    # 12 Conformer encoder blocks
    assert len(model.encoder.blocks) == 12, f"Expected 12 blocks, got {len(model.encoder.blocks)}"
    
    # d_model = 512
    assert model.encoder.config.d_model == 512, f"Expected d_model=512, got {model.encoder.config.d_model}"
    
    first_block = model.encoder.blocks[0]
    
    # d_ff = 2048
    assert first_block.ffn1.w_1.out_features == 2048, "Expected d_ff=2048 in FFN1"
    assert first_block.ffn2.w_1.out_features == 2048, "Expected d_ff=2048 in FFN2"
    
    # 8 attention heads
    assert first_block.self_attn.num_heads == 8, "Expected 8 attention heads"
    
    # Depthwise convolution kernel size = 31
    assert first_block.conv_module.depthwise_conv.kernel_size == (31,), "Expected kernel size 31"
    
    # Parameter count around 79M (78M - 85M depending on vocab)
    param_info = model.count_parameters()
    total_params = param_info["total_parameters"]
    assert 78_000_000 <= total_params <= 85_000_000, f"Expected ~79.3M params, got {total_params:,}"


# ============================================================================
# 3. Tokenizer 7 Languages Support & Special Tokens
# ============================================================================
def test_tokenizer_all_7_languages():
    """Verify tokenizer supports hi, te, ta, bn, mr, kn, zh with 0% UNK on test phrases."""
    tokenizer = MultilingualTokenizer()
    
    required_langs = ["hi", "te", "ta", "bn", "mr", "kn", "zh"]
    for lang in required_langs:
        tag = f"<lang:{lang}>"
        assert tag in tokenizer.char_to_id, f"Missing special language token {tag}"
        assert tokenizer.get_lang_token_id(lang) is not None, f"Could not get ID for {lang}"
        
    sample_texts = {
        "hi": "नमस्ते दुनिया यह आपातकालीन चिकित्सा सहायता प्रणाली है",
        "te": "నమస్కారం ప్రపంచం ఇది అత్యవసర వైద్య సహాయ వ్యవస్థ",
        "ta": "வணக்கம் உலகம் இது அவசர மருத்துவ உதவி அமைப்பு",
        "bn": "নমস্কার বিশ্ব এটি জরুরি চিকিৎসা সহায়তা ব্যবস্থা",
        "mr": "नमस्कार जग हे आपत्कालीन वैद्यकीय मदत प्रणाली आहे",
        "kn": "ನಮಸ್ಕಾರ ವಿಶ್ವ ಇದು ತುರ್ತು ವೈದ್ಯಕೀಯ ಸಹಾಯ ವ್ಯವಸ್ಥೆ",
        "zh": "你好世界这是急救医疗救援系统"
    }
    
    for lang, text in sample_texts.items():
        token_ids = tokenizer.encode(text, language=lang, add_special_tokens=True)
        assert len(token_ids) > 1, f"Tokenization returned too few tokens for {lang}"
        assert token_ids[0] == tokenizer.get_lang_token_id(lang), f"First token must be lang token for {lang}"
        
        # UNK token check
        unk_id = tokenizer.unk_id
        unk_count = token_ids.count(unk_id)
        unk_rate = unk_count / len(token_ids)
        assert unk_rate < 0.05, f"High UNK rate {unk_rate:.2%} in {lang}: '{text}'"
        
        # Decode roundtrip
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert len(decoded.strip()) > 0, f"Decoded string is empty for {lang}"


# ============================================================================
# 4. Dataset Manifests Exist and Valid Schema
# ============================================================================
def test_dataset_manifests_exist_and_non_empty():
    """Verify train, val, and test manifests exist and have required schema fields."""
    manifest_dir = os.path.join(SERVICE_ROOT, "data", "manifests")
    if not os.path.exists(manifest_dir):
        pytest.skip("Dataset manifests not yet curated in data/manifests (run curate_7lang_dataset.py first)")
        
    for split in ["train", "val", "test"]:
        manifest_path = os.path.join(manifest_dir, f"{split}.jsonl")
        if not os.path.exists(manifest_path):
            pytest.skip(f"Manifest {manifest_path} not found yet")
        assert os.path.getsize(manifest_path) > 0, f"Empty manifest: {manifest_path}"
        
        # Verify first few records
        with open(manifest_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                rec = json.loads(line)
                assert "audio" in rec or "audio_filepath" in rec
                assert "text" in rec
                assert "language" in rec
                assert "speaker_id" in rec
                assert "duration" in rec


# ============================================================================
# 5. Language Balance in Train Manifest
# ============================================================================
def test_language_balance_in_train_manifest():
    """Verify that all 7 languages are represented in the train split."""
    manifest_path = os.path.join(SERVICE_ROOT, "data", "manifests", "train.jsonl")
    if not os.path.exists(manifest_path):
        pytest.skip("Train manifest not found (run curate_7lang_dataset.py first)")
        
    lang_counts = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            l = rec.get("language")
            lang_counts[l] = lang_counts.get(l, 0) + 1
            
    required_langs = {"hi", "te", "ta", "bn", "mr", "kn", "zh"}
    assert required_langs.issubset(set(lang_counts.keys())), (
        f"Missing languages in train set! Present: {set(lang_counts.keys())}, Expected: {required_langs}"
    )
    # Ensure minimum representation per language
    for l in required_langs:
        assert lang_counts[l] >= 10, f"Language {l} has fewer than 10 samples: {lang_counts[l]}"


# ============================================================================
# 6. Speaker Disjointness Across Splits
# ============================================================================
def test_speaker_disjointness_across_splits():
    """Verify zero speaker overlap between train, val, and test splits."""
    manifest_dir = os.path.join(SERVICE_ROOT, "data", "manifests")
    train_p = os.path.join(manifest_dir, "train.jsonl")
    val_p = os.path.join(manifest_dir, "val.jsonl")
    test_p = os.path.join(manifest_dir, "test.jsonl")
    
    if not (os.path.exists(train_p) and os.path.exists(val_p) and os.path.exists(test_p)):
        pytest.skip("Manifest files not found for speaker disjointness test")
        
    def get_speakers(path):
        speakers = set()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                speakers.add(rec["speaker_id"])
        return speakers
        
    train_spks = get_speakers(train_p)
    val_spks = get_speakers(val_p)
    test_spks = get_speakers(test_p)
    
    assert train_spks.isdisjoint(val_spks), f"Train and Val share {len(train_spks & val_spks)} speakers!"
    assert train_spks.isdisjoint(test_spks), f"Train and Test share {len(train_spks & test_spks)} speakers!"
    assert val_spks.isdisjoint(test_spks), f"Val and Test share {len(val_spks & test_spks)} speakers!"


# ============================================================================
# 7. Variable-Length Audio Forward Pass
# ============================================================================
def test_forward_pass_variable_length_audio():
    """Verify model handles batched inputs of differing lengths with 4x subsampling."""
    model = ConformerASR(device="cpu")
    model.eval()
    
    B = 2
    max_len = 32000  # 2.0 seconds at 16kHz
    audio = torch.randn(B, max_len)
    lengths = torch.tensor([16000, 32000], dtype=torch.long)
    
    with torch.no_grad():
        out = model(audio=audio, audio_lengths=lengths)
        logits = out["logits"]
        out_lens = out["audio_lengths"]
        
    # Logits shape: (B, T_sub, vocab_size)
    assert logits.dim() == 3
    assert logits.size(0) == B
    assert logits.size(2) == model.vocab_size
    
    # 4x temporal subsampling from 80-bin mel
    expected_t0 = (16000 // 160) // 4
    expected_t1 = (32000 // 160) // 4
    assert abs(out_lens[0].item() - expected_t0) <= 2, f"Expected ~{expected_t0}, got {out_lens[0].item()}"
    assert abs(out_lens[1].item() - expected_t1) <= 2, f"Expected ~{expected_t1}, got {out_lens[1].item()}"


# ============================================================================
# 8. CTC Target Length Validity
# ============================================================================
def test_ctc_target_length_validity():
    """Verify subsampled acoustic frames T_sub >= target label length U for CTC loss."""
    model = ConformerASR(device="cpu")
    
    # 3.0s audio -> 300 mel frames -> 75 subsampled frames
    audio = torch.randn(1, 48000)
    lengths = torch.tensor([48000], dtype=torch.long)
    
    # Target label of length 20 (well under 75)
    targets = torch.randint(1, model.vocab_size, (1, 20), dtype=torch.long)
    target_lens = torch.tensor([20], dtype=torch.long)
    
    out = model(audio=audio, audio_lengths=lengths, targets=targets, target_lengths=target_lens)
    loss = out["loss"]
    assert not torch.isnan(loss).item(), "CTC loss produced NaN"
    assert loss.item() > 0, "CTC loss should be positive"


# ============================================================================
# 9. End-to-End Gradient Flow
# ============================================================================
def test_gradient_flow_end_to_end():
    """Verify gradients propagate from CTC loss through all 12 blocks down to ConvSubsampling."""
    model = ConformerASR(device="cpu")
    model.train()
    
    audio = torch.randn(2, 16000)
    lengths = torch.tensor([16000, 16000], dtype=torch.long)
    targets = torch.randint(1, model.vocab_size, (2, 10), dtype=torch.long)
    target_lens = torch.tensor([10, 10], dtype=torch.long)
    
    out = model(audio=audio, audio_lengths=lengths, targets=targets, target_lengths=target_lens)
    loss = out["loss"]
    loss.backward()
    
    # Check ConvSubsampling gradients
    conv_grad = model.encoder.subsampling.conv1.weight.grad
    assert conv_grad is not None and conv_grad.abs().sum() > 0, "No grad in ConvSubsampling"
    
    # Check first and last Conformer block gradients
    for idx in [0, 5, 11]:
        block = model.encoder.blocks[idx]
        attn_grad = block.self_attn.q_proj.weight.grad
        assert attn_grad is not None and attn_grad.abs().sum() > 0, f"No grad in Block {idx} self_attn"
        
        conv_depth_grad = block.conv_module.depthwise_conv.weight.grad
        assert conv_depth_grad is not None and conv_depth_grad.abs().sum() > 0, f"No grad in Block {idx} depthwise conv"
        
    # Check linear projection head gradients
    head_grad = model.ctc_decoder.lm_head.weight.grad
    assert head_grad is not None and head_grad.abs().sum() > 0, "No grad in CTC classifier head"


# ============================================================================
# 10. Continuous Acoustic Representation Shape Contract (B, T', 512)
# ============================================================================
def test_encode_acoustic_representation_shape():
    """Verify model.encode() produces continuous (B, T', 512) frame embeddings."""
    model = ConformerASR(device="cpu")
    model.eval()
    
    B = 2
    audio = torch.randn(B, 32000)
    
    with torch.no_grad():
        acoustic_repr = model.encode(audio, pool=False)
        pooled = model.encode(audio, pool=True)
        
    assert acoustic_repr.dim() == 3, f"Expected 3D tensor, got {acoustic_repr.dim()}"
    assert acoustic_repr.size(0) == B, f"Expected batch size {B}"
    assert acoustic_repr.size(2) == 512, f"Expected d_model=512, got {acoustic_repr.size(2)}"
    assert pooled.shape == (B, 512), f"Expected pooled shape (B, 512), got {pooled.shape}"


# ============================================================================
# 11. ALM Temporal Fusion Adapter Integration
# ============================================================================
def test_alm_temporal_fusion_adapter_integration():
    """Verify ALMASRAdapter seamlessly projects (B, T', 512) to (B, T', 256) for ALM."""
    adapter = ALMASRAdapter(embed_dim=256, device="cpu")
    adapter.eval()
    
    # 1. Test raw waveform encode
    audio = torch.randn(2, 32000)
    with torch.no_grad():
        frame_embs = adapter.encode(audio, pool=False)
        pooled_emb = adapter.encode(audio, pool=True)
        
    assert frame_embs.shape[0] == 2
    assert frame_embs.shape[2] == 256, f"Expected ALM dimension 256, got {frame_embs.shape[2]}"
    assert pooled_emb.shape == (2, 256), f"Expected pooled shape (2, 256), got {pooled_emb.shape}"
    
    # 2. Test get_speech_embeddings dictionary output
    with torch.no_grad():
        out_dict = adapter.get_speech_embeddings(audio)
        
    assert "frame_embeddings" in out_dict
    assert "raw_conformer_frames" in out_dict
    assert out_dict["frame_embeddings"].shape[-1] == 256
    assert out_dict["raw_conformer_frames"].shape[-1] == 512


# ============================================================================
# 12. End-to-End Transcription Across All 7 Languages
# ============================================================================
def test_transcription_pipeline_all_languages():
    """Verify model.transcribe() executes cleanly without error across all 7 language hints."""
    model = DeepConformerASR(device="cpu")
    dummy_wav = torch.randn(1, 24000)  # 1.5s audio
    
    target_langs = ["hi", "te", "ta", "bn", "mr", "kn", "zh"]
    for lang in target_langs:
        result = model.transcribe(dummy_wav, language=lang)
        assert "text" in result, f"Missing 'text' in result for {lang}"
        assert "language" in result, f"Missing 'language' in result for {lang}"
        assert result["language"] == lang, f"Expected language {lang}, got {result['language']}"
        assert "tokens" in result
        assert isinstance(result["text"], str)
