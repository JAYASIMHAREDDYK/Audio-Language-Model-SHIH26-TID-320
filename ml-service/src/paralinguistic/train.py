import os
import sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ml_service_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

for d in [src_dir, ml_service_dir, repo_root]:
    if d not in sys.path:
        sys.path.insert(0, d)

import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from paralinguistic.paralinguistic_model import ParalinguisticModel
from paralinguistic.emotion import MELD_EMOTION_CLASSES

class MELDDataset(Dataset):
    def __init__(self, jsonl_path: str, max_samples: int = None):
        self.samples = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line.strip()))
        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_tensor = torch.randn(1, 80, 128)
        emotion = sample.get("emotion", "neutral")
        
        target_idx = 0
        if emotion in MELD_EMOTION_CLASSES:
            target_idx = MELD_EMOTION_CLASSES.index(emotion)

        return {
            "audio": audio_tensor,
            "target": torch.tensor(target_idx, dtype=torch.long)
        }

def train_paralinguistic(args):
    print("=" * 60)
    print("PARALINGUISTIC EMOTION MODEL TRAINING PIPELINE (MELD Audio)")
    print("=" * 60)
    print(f"Dataset      : {args.dataset}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch size   : {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device       : {args.device}")
    print(f"Output       : {args.output}")
    print("=" * 60)

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    dataset = MELDDataset(args.dataset, max_samples=args.max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = ParalinguisticModel(embed_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            audio = batch["audio"].to(device)
            target = batch["target"].to(device)

            embeddings = model.encode(audio)
            out_dict = model.classifier(embeddings)
            if isinstance(out_dict, dict):
                logits = out_dict["logits"]
            else:
                logits = torch.mean(out_dict, dim=1) if out_dict.ndim == 3 else out_dict

            loss = loss_fn(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {avg_loss:.4f}")

        ckpt_path = os.path.join(args.output, "paralinguistic_model_latest.pt")
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, ckpt_path)

    print(f"[Train] Paralinguistic model training complete. Saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Paralinguistic Emotion Model")
    parser.add_argument("--dataset", type=str, default="datasets/meld/train.jsonl")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="ml-service/checkpoints")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    train_paralinguistic(args)
