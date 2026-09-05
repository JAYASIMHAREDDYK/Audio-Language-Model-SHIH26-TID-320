"""
Section 9: Learning-Rate Sanity Experiment on 1 GB Subset.
Compares: 3e-5, 5e-5, 1e-4, 2e-4, 4e-4 on a fixed subset of the 1 GB dataset.
Measures: loss descent, CER, WER, blank percentage, and gradient norm.
"""

import os
import sys
import json
import torch
from torch.utils.data import DataLoader

SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.asr.multilingual_asr import ASRModel
from src.asr.tokenizer import MultilingualTokenizer
from training.dataset import MultilingualASRDataset, collate_asr_batch
from src.asr.evaluate import calculate_cer, calculate_wer


def run_lr_sanity():
    print("=" * 70, flush=True)
    print("SECTION 9: LEARNING-RATE SANITY EXPERIMENT (1 GB Subset)", flush=True)
    print("Testing candidate LRs: [3e-5, 5e-5, 1e-4, 2e-4, 4e-4]", flush=True)
    print("=" * 70, flush=True)

    tokenizer = MultilingualTokenizer()
    manifest_path = os.path.join(SERVICE_ROOT, "data", "manifests", "train_1gb.jsonl")
    dataset = MultilingualASRDataset(manifest_path, tokenizer=tokenizer, max_samples=48)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_asr_batch)
    
    val_batch = next(iter(dataloader))
    candidate_lrs = [3e-5, 5e-5, 1e-4, 2e-4, 4e-4]
    results = []

    for lr in candidate_lrs:
        torch.manual_seed(42)
        model = ASRModel()
        device = next(model.parameters()).device
        model.encoder.use_spec_augment = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        model.train()

        losses = []
        grad_norms = []
        num_steps = 25

        for step in range(1, num_steps + 1):
            step_loss = 0.0
            for batch in dataloader:
                optimizer.zero_grad()
                audio = batch["audio"].to(device)
                audio_lengths = batch["audio_lengths"].to(device)
                targets = batch["targets"].to(device)
                target_lengths = batch["target_lengths"].to(device)

                outputs = model(audio=audio, audio_lengths=audio_lengths, targets=targets, target_lengths=target_lengths)
                loss = outputs["loss"]
                loss.backward()

                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_norms.append(gn.item())
                optimizer.step()
                step_loss += loss.item()
            losses.append(step_loss / len(dataloader))

        # Evaluate on validation batch
        model.eval()
        with torch.no_grad():
            v_audio = val_batch["audio"].to(device)
            v_lens = val_batch["audio_lengths"].to(device)
            v_out = model(audio=v_audio, audio_lengths=v_lens)
            logits = v_out["logits"]
            lengths = v_out["lengths"]

            preds = logits.argmax(dim=-1)
            total_frames = sum(lengths.tolist())
            blank_frames = sum((preds[i, :lengths[i]] == 0).sum().item() for i in range(len(lengths)))
            blank_pct = (blank_frames / max(1, total_frames)) * 100.0

            dec_res = model.decoder.decode_greedy(logits, tokenizer, input_lengths=lengths)
            cers = [calculate_cer(val_batch["texts"][i], dec_res[i]["text"], language=val_batch["languages"][i]) for i in range(len(val_batch["texts"]))]
            wers = [calculate_wer(val_batch["texts"][i], dec_res[i]["text"], language=val_batch["languages"][i]) for i in range(len(val_batch["texts"]))]

            mean_cer = sum(cers) / len(cers)
            mean_wer = sum(wers) / len(wers)
            avg_grad_norm = sum(grad_norms[-8:]) / 8

            res_entry = {
                "lr": lr,
                "initial_loss": round(losses[0], 4),
                "final_loss": round(losses[-1], 4),
                "val_cer": round(mean_cer, 4),
                "val_wer": round(mean_wer, 4),
                "blank_percentage": round(blank_pct, 2),
                "avg_grad_norm": round(avg_grad_norm, 4),
                "sample_hypothesis": dec_res[0]["text"][:40]
            }
            results.append(res_entry)

            print(
                f"LR: {lr:7.1e} | "
                f"Loss: {losses[0]:.2f} -> {losses[-1]:.4f} | "
                f"CER: {mean_cer:.2%} | "
                f"WER: {mean_wer:.2%} | "
                f"Blank: {blank_pct:5.1f}% | "
                f"GradNorm: {avg_grad_norm:.3f} | "
                f"Sample: {repr(dec_res[0]['text'][:30])}",
                flush=True
            )

    out_file = os.path.join(SERVICE_ROOT, "artifacts", "lr_sanity_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {out_file}", flush=True)

    best_lr = min(results, key=lambda x: (x["val_cer"], x["final_loss"]))
    print(f"\n[OPTIMAL LR SELECTED]: {best_lr['lr']} (Final Loss: {best_lr['final_loss']}, CER: {best_lr['val_cer']:.2%})", flush=True)
    return best_lr


if __name__ == "__main__":
    run_lr_sanity()
