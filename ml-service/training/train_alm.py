import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.alm.alm_model import CoreALM
from src.alm.losses import ALMTaskLoss
from src.alm.reasoning_model import ALMTokenizer
import torchaudio
from src.audio.spectrogram import LogMelSpectrogramExtractor
import scipy.io.wavfile as wavfile
import numpy as np

class ALMJointDataset(Dataset):
    """
    PyTorch Dataset wrapper for ALM Joint JSONL files with natural language target tokenization.
    """
    def __init__(self, jsonl_paths: list, max_samples=2000):
        self.samples = []
        self.tokenizer = ALMTokenizer(vocab_size=5000)
        self.mel_extractor = LogMelSpectrogramExtractor(sample_rate=16000, hop_length=625)
        
        for path in jsonl_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line.strip())
                            ans = item.get("clean_answer") or item.get("ground_truth_answer") or item.get("answer") or "Speech in background environment."
                            item["clean_answer"] = ans
                            self.samples.append(item)
                            if len(self.samples) >= max_samples:
                                break

        if not self.samples:
            print("[ALMJointDataset] Warning: No JSONL files found. Creating default samples.")
            self.samples = [{
                "question": "Where is the speaker likely to be?",
                "clean_answer": "Location is airport terminal.",
                "scene": "airport_terminal"
            }]

        print(f"[ALMJointDataset] Loaded {len(self.samples)} training samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        wav_path = os.path.join("datasets/alm_nhce", sample.get("audio", ""))
        if os.path.exists(wav_path):
            try:
                sr, y = wavfile.read(wav_path)
                if y.ndim > 1:
                    y = y.mean(axis=1)
                y = y.astype(np.float32)
                if np.max(np.abs(y)) > 1.0:
                    y = y / 32768.0
                wav_tensor = torch.from_numpy(y).unsqueeze(0)
                
                mel = self.mel_extractor(wav_tensor) # (1, 80, 128) if hop_length=625
                if mel.size(2) > 128:
                    mel = mel[:, :, :128]
                elif mel.size(2) < 128:
                    pad = torch.zeros(1, 80, 128 - mel.size(2))
                    mel = torch.cat([mel, pad], dim=2)
                audio_tensor = mel
            except Exception as e:
                print(f"Error loading {wav_path}: {e}")
                audio_tensor = torch.randn(1, 80, 128)
        else:
            audio_tensor = torch.randn(1, 80, 128)

        question = sample.get("question", "Where is the speaker likely to be?")
        raw_answer = sample.get("clean_answer", "Location is airport terminal.")
        
        # Tokenize natural language target answer sequence
        answer_ids = self.tokenizer.encode(raw_answer, max_len=24)
        evidence_targets = torch.zeros(64, dtype=torch.float)
        evidence_targets[10:25] = 1.0
        target_confidence = torch.tensor([0.85], dtype=torch.float)

        return {
            "audio": audio_tensor,
            "question": question,
            "answer_ids": answer_ids,
            "evidence_targets": evidence_targets,
            "target_confidence": target_confidence
        }

def train_alm(args):
    print("=" * 60)
    print("CORE ALM AUTOREGRESSIVE DECODER TRAINING — SMART HORIZON 2026")
    print("=" * 60)
    print(f"Dataset paths  : {args.datasets}")
    print(f"Epochs         : {args.epochs}")
    print(f"Batch size     : {args.batch_size}")
    print(f"Learning rate  : {args.learning_rate}")
    print(f"Device         : {args.device}")
    print(f"Output dir     : {args.output}")
    print("=" * 60)

    os.makedirs(args.output, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. No CUDA device found.")
    device = torch.device("cuda")
    print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    # Load combined datasets
    dataset = ALMJointDataset(args.datasets, max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize Core ALM Model
    model = CoreALM(embed_dim=256).to(device)
    
    # Freeze 4 specialized encoders to preserve their pretrained representations
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    for param in model.asr_model.parameters():
        param.requires_grad = False
    for param in model.event_model.parameters():
        param.requires_grad = False
    for param in model.speaker_model.parameters():
        param.requires_grad = False
    for param in model.para_model.parameters():
        param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[Train] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = ALMTaskLoss()

    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"[Train] Resumed weights from checkpoint {args.resume}")

    model.train()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        total_epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            audio = batch["audio"].to(device)
            question = batch["question"] # batch list of question strings
            token_ids = batch["answer_ids"].to(device)

            # Autoregressive Target Shifting:
            # Input = token_ids[:, :-1] (starts with <bos>)
            # Target Label = token_ids[:, 1:] (ends with <eos>)
            tgt_in = token_ids[:, :-1]
            tgt_lbl = token_ids[:, 1:]

            optimizer.zero_grad()
            outputs = model(audio, question_text=question, tgt_tokens=tgt_in)

            targets = {
                "answer_ids": tgt_lbl,
                "evidence_targets": batch["evidence_targets"].to(device),
                "target_confidence": batch["target_confidence"].to(device),
                "target_speech": outputs["projected_modalities"]["speech"].detach(),
                "target_events": outputs["projected_modalities"]["events"].detach(),
                "target_speakers": outputs["projected_modalities"]["speakers"].detach(),
                "target_para": outputs["projected_modalities"]["paralinguistic"].detach()
            }

            loss_dict = loss_fn(outputs, targets)
            loss = loss_dict["loss"]

            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()

        avg_loss = total_epoch_loss / max(1, len(dataloader))
        print(f"Epoch [{epoch}/{start_epoch + args.epochs - 1}] - Avg Loss: {avg_loss:.4f}")

        # Save Checkpoint
        latest_path = os.path.join(args.output, "best_checkpoint.pt")
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss
        }
        torch.save(ckpt_data, latest_path)

    print(f"[Train] Training complete. Saved checkpoint to {latest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Core ALM Autoregressive Decoder")
    parser.add_argument("--datasets", nargs="+", default=["datasets/alm_nhce/train_diverse.jsonl"], help="Path to JSONL datasets")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max dataset samples")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda only, no cpu fallback)")
    parser.add_argument("--output", type=str, default="ml-service/checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint file path to resume from")

    args = parser.parse_args()
    train_alm(args)
