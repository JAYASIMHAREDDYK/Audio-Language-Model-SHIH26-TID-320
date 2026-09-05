import os
import sys

# Ensure ml-service/src and ml-service are in sys.path BEFORE any imports
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ml_service_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

for d in [src_dir, ml_service_dir, repo_root]:
    if d not in sys.path:
        sys.path.insert(0, d)

from paralinguistic.train import train_paralinguistic, argparse

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
    
    # Resolve relative dataset path if needed
    if not os.path.exists(args.dataset):
        alt_path = os.path.join(repo_root, args.dataset)
        if os.path.exists(alt_path):
            args.dataset = alt_path

    train_paralinguistic(args)
