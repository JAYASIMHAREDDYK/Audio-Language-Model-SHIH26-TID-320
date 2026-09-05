"""
Canonical Training Script for Core Audio Language Model (Core ALM).
Trains CoreALM using multi-modal audio representations and question conditioning.
"""
import os, sys, math, json, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from collections import defaultdict
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.alm.alm_model import CoreALM
from src.alm.reasoning_model import ALMTokenizer
from src.audio.spectrogram import LogMelSpectrogramExtractor

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for training.")
device = torch.device("cuda")

mel_extractor = LogMelSpectrogramExtractor(sample_rate=16000, hop_length=625).to(device)
tokenizer = ALMTokenizer(vocab_size=5000)

def load_audio_mel(audio_path, device):
    if isinstance(audio_path, str):
        full_path = audio_path
        if not os.path.isabs(full_path):
            full_path = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "alm_nhce", audio_path)
        if not os.path.exists(full_path):
            full_path = os.path.join(os.path.dirname(__file__), "..", "..", audio_path)
        
        if os.path.exists(full_path):
            try:
                waveform, sr = torchaudio.load(full_path)
                if waveform.ndim == 2:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                    waveform = resampler(waveform)
                waveform = waveform.to(device)
                mel = mel_extractor(waveform)
                if mel.size(2) > 128: mel = mel[:, :, :128]
                elif mel.size(2) < 128:
                    mel = torch.cat([mel, torch.zeros(1, 80, 128 - mel.size(2), device=device)], dim=2)
                return mel
            except Exception as e:
                print(f"Warning loading {full_path}: {e}")
                
    # Fallback to noise mel
    return torch.randn(1, 80, 128, device=device)

def train_core_alm(epochs=15, batch_size=16, lr=1e-3, checkpoint_out=None):
    if checkpoint_out is None:
        checkpoint_out = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints", "core_alm_latest.pt")
        
    print("=" * 80)
    print("CANONICAL CORE ALM TRAINING PIPELINE")
    print("=" * 80)

    # 1. Load real training dataset from train_rich.jsonl
    train_rich_path = os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "alm_nhce", "train_rich.jsonl")
    
    real_train_by_cat = defaultdict(list)
    if os.path.exists(train_rich_path):
        with open(train_rich_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    cat = item.get("category", "")
                    audio_file = item.get("audio", "")
                    question = item.get("question", "")
                    answer = item.get("clean_answer") or item.get("ground_truth_answer") or item.get("answer", "")
                    if cat and audio_file and question and answer:
                        real_train_by_cat[cat].append((audio_file, question, answer, cat))

    print("Training dataset category breakdown:")
    for c, lst in real_train_by_cat.items():
        print(f"  {c:<25s}: {len(lst)} items")

    # Select balanced dataset per category
    train_samples = []
    for cat in ["sound_event", "speaker_count", "speech_cooccurrence", "emotion", "scene", "temporal"]:
        lst = real_train_by_cat.get(cat, [])
        if len(lst) >= 250:
            indices = np.random.choice(len(lst), size=250, replace=False)
            train_samples.extend([lst[i] for i in indices])
        elif len(lst) > 0:
            reps = (250 + len(lst) - 1) // len(lst)
            full_lst = (lst * reps)[:250]
            train_samples.extend(full_lst)

    np.random.shuffle(train_samples)
    print(f"Balanced training dataset size: {len(train_samples)} samples across 6 categories.")

    model = CoreALM(embed_dim=256, vocab_size=5000).to(device)
    with torch.no_grad():
        _ = model(torch.randn(1, 80, 128, device=device), question_text="Init")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    num_batches = (len(train_samples) + batch_size - 1) // batch_size

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        
        np.random.shuffle(train_samples)
        
        for b in range(num_batches):
            batch = train_samples[b*batch_size : (b+1)*batch_size]
            if not batch: continue
            
            b_mels = torch.cat([load_audio_mel(s[0], device) for s in batch], dim=0)
            b_qs = [s[1] for s in batch]
            b_ans = [s[2] for s in batch]
            
            tgt_tensors = [tokenizer.encode(ans, max_len=20, device=device) for ans in b_ans]
            tgt_batch = torch.stack(tgt_tensors, dim=0)
            
            b_tgt_in = tgt_batch[:, :-1]
            b_tgt_lbl = tgt_batch[:, 1:]
            
            # 30% Question masking to prevent question-prior domination
            masked_qs = []
            for q in b_qs:
                if np.random.rand() < 0.3:
                    masked_qs.append("")
                else:
                    masked_qs.append(q)
                    
            optimizer.zero_grad()
            out = model(b_mels, question_text=masked_qs, tgt_tokens=b_tgt_in)
            logits = out["answer_logits"]
            
            loss = F.cross_entropy(logits.reshape(-1, 5000), b_tgt_lbl.reshape(-1), ignore_index=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(batch)
            preds = torch.argmax(logits, dim=-1)
            mask = (b_tgt_lbl != 0)
            total_correct += (preds[mask] == b_tgt_lbl[mask]).sum().item()
            total_tokens += mask.sum().item()
            
        scheduler.step()
        avg_loss = total_loss / len(train_samples)
        token_acc = total_correct / max(1, total_tokens) * 100
        print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {avg_loss:.4f} | Token Acc: {token_acc:6.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")

    os.makedirs(os.path.dirname(checkpoint_out), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_out)
    print(f"\nCanonical Core ALM checkpoint saved to {checkpoint_out}")

if __name__ == "__main__":
    train_core_alm()
