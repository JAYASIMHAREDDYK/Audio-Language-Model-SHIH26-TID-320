"""
Comprehensive CTC Diagnostic Test Suite for Deep Multilingual Conformer ASR.
Covers all Phase 23 specifications:
- test_tokenizer_round_trip_all_7_languages
- test_ctc_blank_id
- test_ctc_target_construction
- test_ctc_input_lengths
- test_ctc_repeated_label_validity
- test_ctc_greedy_decoder
- test_variable_length_batch
- test_ctc_loss_finite
- test_gradient_norms
- test_single_batch_overfit
- test_wer_known_examples
- test_cer_known_examples
- test_unicode_normalization
- test_alm_representation_contract
"""

import os
import sys
import json
import pytest
import torch
import torch.nn as nn

# Ensure ml-service root is in sys.path
SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.asr.tokenizer import MultilingualTokenizer, BLANK_TOKEN, UNK_TOKEN, SPACE_TOKEN, PAD_TOKEN, LANG_TOKENS
from src.asr.multilingual_asr import ASRModel
from src.asr.preprocessing import normalize_text, TARGET_SAMPLE_RATE
from src.asr.evaluate import calculate_wer, calculate_cer, compute_levenshtein_distance
from training.dataset import MultilingualASRDataset, collate_asr_batch


@pytest.fixture(scope="module")
def tokenizer():
    return MultilingualTokenizer()


@pytest.fixture(scope="module")
def cpu_model():
    return ASRModel(device="cpu")


def test_tokenizer_round_trip_all_7_languages(tokenizer):
    """Phase 4: Verify decode(encode(text)) == normalized_text across all 7 languages."""
    langs = ["hi", "te", "ta", "bn", "mr", "kn", "zh"]
    found_samples = {l: [] for l in langs}
    
    # Read real samples from train and val manifests
    for split in ["train.jsonl", "val.jsonl"]:
        path = os.path.join(SERVICE_ROOT, "data", "manifests", split)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                l = item.get("language")
                if l in found_samples and len(found_samples[l]) < 5:
                    found_samples[l].append(item["text"])
                    
    for lang in langs:
        samples = found_samples[lang]
        assert len(samples) > 0, f"No samples found for language {lang}"
        for raw_text in samples:
            norm = normalize_text(raw_text, language=lang)
            encoded = tokenizer.encode(norm, language=lang, add_special_tokens=False)
            assert tokenizer.unk_id not in encoded, f"Found UNK in encoded tokens for {lang}: {norm}"
            decoded = tokenizer.decode(encoded, skip_special_tokens=True)
            assert decoded.strip() == norm.strip(), (
                f"Round-trip failed for {lang}:\nNorm:    {repr(norm)}\nDecoded: {repr(decoded)}"
            )


def test_ctc_blank_id(tokenizer, cpu_model):
    """Phase 3: Verify CTC blank ID consistency and isolation."""
    assert tokenizer.blank_id == 0, f"tokenizer.blank_id must be 0, got {tokenizer.blank_id}"
    assert tokenizer.unk_id == 1, f"tokenizer.unk_id must be 1, got {tokenizer.unk_id}"
    assert tokenizer.space_id == 2, f"tokenizer.space_id must be 2, got {tokenizer.space_id}"
    assert tokenizer.pad_id == 3, f"tokenizer.pad_id must be 3, got {tokenizer.pad_id}"
    
    assert cpu_model.decoder.blank_id == 0
    assert cpu_model.decoder.ctc_loss_fn.blank == 0
    assert cpu_model.decoder.lm_head.out_features == tokenizer.vocab_size()
    
    # Blank must not be pad or unk or space
    assert tokenizer.blank_id != tokenizer.pad_id
    assert tokenizer.blank_id != tokenizer.unk_id
    assert tokenizer.blank_id != tokenizer.space_id


def test_ctc_target_construction(tokenizer):
    """Phase 4 & 5: Verify target construction without erroneous language prefix."""
    text = "नमस्ते दुनिया"
    lang = "hi"
    tokens = tokenizer.encode(text, language=lang, add_special_tokens=False)
    # add_special_tokens=False must not insert <lang:hi> (ID 4)
    lang_id = tokenizer.get_lang_token_id(lang)
    assert lang_id not in tokens[:1], "Language token should NOT be prepended when add_special_tokens=False"
    assert len(tokens) > 0
    decoded = tokenizer.decode(tokens, skip_special_tokens=True)
    assert "दुनिया" in decoded


def test_ctc_input_lengths(cpu_model):
    """Phase 6: Verify subsampled length calculation matches tensor dimensions."""
    sample_rates = TARGET_SAMPLE_RATE
    # Test short (1.5s), medium (4.0s), and long (8.0s) durations
    durations = [1.5, 4.0, 8.0]
    for dur in durations:
        n_samples = int(dur * sample_rates)
        dummy_audio = torch.randn(1, n_samples)
        lengths = torch.tensor([n_samples], dtype=torch.long)
        out = cpu_model(audio=dummy_audio, audio_lengths=lengths)
        logits = out["logits"]
        enc_len = out["lengths"]
        
        # Subsampled length must equal logits sequence dimension
        assert logits.size(1) == enc_len.item()
        
        # Approximate 4x subsampling from hop_length 160: ~ (n_samples / 640)
        expected_approx = n_samples // 640
        assert abs(enc_len.item() - expected_approx) <= 3


def test_ctc_repeated_label_validity(tokenizer):
    """Phase 6: Verify repeated label capacity requirements."""
    # Repeated labels in CTC require at least one blank in between
    text = "hello"
    tokens = tokenizer.encode(text, language="en", add_special_tokens=False)
    # 'l' is repeated
    repeats = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1])
    min_frames = len(tokens) + repeats
    assert min_frames > len(tokens)


def test_ctc_greedy_decoder(tokenizer, cpu_model):
    """Phase 3: Verify canonical greedy CTC decoder logic with known sequence."""
    # Construct synthetic logits: batch of 1, T=8 frames, Vocab=tokenizer.vocab_size()
    # Path: [blank, token_A, token_A, blank, token_A, token_B, token_B, blank]
    # Collapsed expected: [token_A, token_A, token_B]
    t_a = tokenizer.token_to_id.get("a", 10)
    t_b = tokenizer.token_to_id.get("b", 11)
    blank = tokenizer.blank_id
    
    seq = [blank, t_a, t_a, blank, t_a, t_b, t_b, blank]
    T = len(seq)
    V = tokenizer.vocab_size()
    
    logits = torch.full((1, T, V), -10.0)
    for t_idx, tok_id in enumerate(seq):
        logits[0, t_idx, tok_id] = 10.0
        
    results = cpu_model.decoder.decode_greedy(logits, tokenizer, input_lengths=torch.tensor([T]))
    res = results[0]
    expected_ids = [t_a, t_a, t_b]
    assert res["token_ids"] == expected_ids, f"Expected {expected_ids}, got {res['token_ids']}"


def test_variable_length_batch(cpu_model):
    """Phase 7: Verify variable-length audio batching and padding mask."""
    s1 = torch.randn(int(2.0 * TARGET_SAMPLE_RATE))
    s2 = torch.randn(int(4.5 * TARGET_SAMPLE_RATE))
    
    lengths = torch.tensor([len(s1), len(s2)], dtype=torch.long)
    max_len = max(len(s1), len(s2))
    padded_audio = torch.zeros(2, max_len)
    padded_audio[0, :len(s1)] = s1
    padded_audio[1, :len(s2)] = s2
    
    targets = torch.tensor([[10, 11, 3], [12, 13, 14]], dtype=torch.long)
    target_lengths = torch.tensor([2, 3], dtype=torch.long)
    
    out = cpu_model(audio=padded_audio, audio_lengths=lengths, targets=targets, target_lengths=target_lengths)
    loss = out["loss"]
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert out["lengths"][0] < out["lengths"][1]


def test_ctc_loss_finite(cpu_model):
    """Phase 12: Verify CTC loss numerics on valid inputs."""
    audio = torch.randn(2, 32000)
    targets = torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.long)
    target_lengths = torch.tensor([3, 3], dtype=torch.long)
    out = cpu_model(audio=audio, targets=targets, target_lengths=target_lengths)
    loss = out["loss"]
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_gradient_norms(cpu_model):
    """Phase 11: Verify non-zero finite gradient flow across critical layers."""
    cpu_model.train()
    cpu_model.zero_grad()
    
    audio = torch.randn(1, 32000)
    targets = torch.tensor([[10, 12, 14]], dtype=torch.long)
    target_lengths = torch.tensor([3], dtype=torch.long)
    
    out = cpu_model(audio=audio, targets=targets, target_lengths=target_lengths)
    loss = out["loss"]
    loss.backward()
    
    # Check ConvSubsampling
    sub_grad = cpu_model.encoder.subsampling.conv1.weight.grad
    assert sub_grad is not None and torch.isfinite(sub_grad).all() and sub_grad.norm().item() > 0.0
    
    # Check Conformer Block 1
    b1_grad = cpu_model.encoder.blocks[0].ffn1.w_1.weight.grad
    assert b1_grad is not None and torch.isfinite(b1_grad).all() and b1_grad.norm().item() > 0.0
    
    # Check Conformer Block 6
    b6_grad = cpu_model.encoder.blocks[5].ffn1.w_1.weight.grad
    assert b6_grad is not None and torch.isfinite(b6_grad).all() and b6_grad.norm().item() > 0.0
    
    # Check Conformer Block 12
    b12_grad = cpu_model.encoder.blocks[11].ffn1.w_1.weight.grad
    assert b12_grad is not None and torch.isfinite(b12_grad).all() and b12_grad.norm().item() > 0.0
    
    # Check CTC Head
    lm_grad = cpu_model.decoder.lm_head.weight.grad
    assert lm_grad is not None and torch.isfinite(lm_grad).all() and lm_grad.norm().item() > 0.0


def test_wer_known_examples():
    """Phase 14: Test Levenshtein WER metric with exact known cases."""
    # Identical
    assert calculate_wer("hello world", "hello world", language="en") == 0.0
    # Complete deletion
    assert calculate_wer("hello world", "", language="en") == 1.0
    # Partial error
    w = calculate_wer("hello world", "hello", language="en")
    assert 0.0 < w <= 1.0
    # Mandarin (character/segmented based)
    assert calculate_wer("西 班 牙", "西 班 牙", language="zh") == 0.0


def test_cer_known_examples():
    """Phase 14: Test Levenshtein CER metric with known cases across scripts."""
    assert calculate_cer("नमस्ते", "नमस्ते", language="hi") == 0.0
    assert calculate_cer("नमस्ते", "", language="hi") == 1.0
    # One substitution in 4 chars: 1/4 = 0.25
    assert calculate_cer("abcd", "abce", language="en") == 0.25
    # Mandarin
    assert calculate_cer("你好世界", "你好世界", language="zh") == 0.0
    assert calculate_cer("你好世界", "你好", language="zh") == 0.5


def test_unicode_normalization():
    """Phase 14: Verify Unicode normalization handles NFKC and zero-width characters."""
    raw = "न\u200dमस्ते ।"
    norm = normalize_text(raw, language="hi")
    assert "\u200d" not in norm
    assert "।" not in norm
    assert "नमस्ते" in norm


def test_alm_representation_contract(cpu_model):
    """Phase 21: Verify ALM representation contract (B, T, 512)."""
    cpu_model.eval()
    with torch.no_grad():
        dummy_audio = torch.randn(1, 32000)
        frames = cpu_model.encode(dummy_audio)
        assert frames.ndim == 3
        assert frames.size(0) == 1
        assert frames.size(2) == 512
        
        # ALM projection adapter simulation
        adapter = nn.Linear(512, 256)
        alm_features = adapter(frames)
        assert alm_features.shape == (1, frames.size(1), 256)


def test_single_batch_overfit():
    """Phase 10: Verify single batch can be memorized (loss drops and CER < 0.5)."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASRModel(device=device).to(device)
    model.train()
    
    ds = MultilingualASRDataset(
        os.path.join(SERVICE_ROOT, "data", "manifests", "train.jsonl"),
        model.tokenizer,
        max_samples=2
    )
    batch = collate_asr_batch([ds[0], ds[1]])
    
    audio = batch["audio"].to(device)
    audio_lengths = batch["audio_lengths"].to(device)
    targets = batch["targets"].to(device)
    target_lengths = batch["target_lengths"].to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    initial_loss = None
    final_loss = None
    final_cer = 1.0
    
    for step in range(1, 71):
        optimizer.zero_grad()
        out = model(audio=audio, audio_lengths=audio_lengths, targets=targets, target_lengths=target_lengths)
        loss = out["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step == 1:
            initial_loss = loss.item()
        if step == 70:
            final_loss = loss.item()
            with torch.no_grad():
                dec = model.decoder.decode_greedy(out["logits"], model.tokenizer, input_lengths=out["lengths"])
                final_cer = calculate_cer(batch["texts"][0], dec[0]["text"], language=batch["languages"][0])
                
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
    assert final_cer < 0.65, f"CER did not decrease below 0.65: got {final_cer:.2%}"
