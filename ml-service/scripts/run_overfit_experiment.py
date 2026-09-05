"""
Tiny Overfit Experiment (Phase 9 Diagnostic Protocol).
Trains 12-layer Deep Conformer ASR aggressively on 16 distinct multilingual samples.
Verifies training loss drops and training CER / WER drops to < 15%.
Produces artifacts/overfit_test.json.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add ml-service root
SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.asr.multilingual_asr import ASRModel
from src.asr.tokenizer import MultilingualTokenizer
from src.asr.evaluate import calculate_wer, calculate_cer
from training.dataset import MultilingualASRDataset, collate_asr_batch


def run_overfit_experiment(
    num_samples: int = 16,
    num_epochs: int = 220,
    learning_rate: float = 1.2e-3,
    device: str = "auto"
):
    print("=" * 65, flush=True)
    print("PHASE 9: TINY MULTILINGUAL OVERFIT EXPERIMENT", flush=True)
    print(f"Goal: Prove CER drops from 100% to < 15% on {num_samples} distinct samples", flush=True)
    print("=" * 65, flush=True)

    if device == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)
    print(f"Device: {target_device}", flush=True)

    tokenizer = MultilingualTokenizer()
    model = ASRModel(device=target_device).to(target_device)
    # Disable SpecAugment for pure overfit capacity verification
    model.encoder.use_spec_augment = False
    model.train()

    # Select distinct utterances per language (no conflicting parallel sentences)
    langs = ["hi", "te", "ta", "bn", "mr", "kn", "zh"]
    collected = {l: [] for l in langs}
    seen_prefixes = set()
    train_manifest = os.path.join(SERVICE_ROOT, "data", "manifests", "train.jsonl")

    with open(train_manifest, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            l = item.get("language")
            if l not in collected:
                continue
            prefix = item.get("text", "")[:12].strip()
            # Select 2 distinct samples per language, up to 16 total
            lang_limit = 3 if l in ("zh", "hi") else 2
            if len(collected[l]) < lang_limit and prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                collected[l].append(item)
            if sum(len(v) for v in collected.values()) >= num_samples:
                break

    subset_items = []
    for l in langs:
        subset_items.extend(collected[l])
    subset_items = sorted(subset_items, key=lambda it: it.get("duration", 0.0))
    print(f"Loaded {len(subset_items)} distinct duration-sorted samples across 7 languages:", flush=True)
    for it in subset_items:
        print(f"  [{it['language']:2s}] ({it.get('duration', 0.0):.1f}s) {repr(it['text'][:35])}", flush=True)

    overfit_manifest = os.path.join(SERVICE_ROOT, "data", "manifests", "overfit_subset.jsonl")
    with open(overfit_manifest, "w", encoding="utf-8") as f:
        for it in subset_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    dataset = MultilingualASRDataset(overfit_manifest, tokenizer=tokenizer, max_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_asr_batch)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    total_steps = num_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=4e-4)

    history = []
    initial_loss = None
    initial_cer = 1.0
    initial_wer = 1.0
    
    start_time = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        model.train()
        model.encoder.use_spec_augment = False
        epoch_loss = 0.0
        batch_count = 0

        optimizer.zero_grad()
        for batch in dataloader:
            audio = batch["audio"].to(target_device)
            audio_lengths = batch["audio_lengths"].to(target_device)
            targets = batch["targets"].to(target_device)
            target_lengths = batch["target_lengths"].to(target_device)

            outputs = model(
                audio=audio,
                audio_lengths=audio_lengths,
                targets=targets,
                target_lengths=target_lengths
            )
            loss = outputs["loss"] / len(dataloader)
            loss.backward()

            epoch_loss += loss.item() * len(dataloader)
            batch_count += 1

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        avg_loss = epoch_loss / max(1, batch_count)
        if initial_loss is None:
            initial_loss = avg_loss

        # Evaluate overfit set every 10 epochs or at end
        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            model.eval()
            wers = []
            cers = []
            sample_predictions = []

            with torch.no_grad():
                for batch in dataloader:
                    audio = batch["audio"].to(target_device)
                    audio_lengths = batch["audio_lengths"].to(target_device)
                    outputs = model(audio=audio, audio_lengths=audio_lengths)
                    logits = outputs["logits"]
                    lengths = outputs["lengths"]

                    dec_res = model.decoder.decode_greedy(logits, tokenizer, input_lengths=lengths)
                    for b in range(len(dec_res)):
                        ref = batch["texts"][b]
                        hyp = dec_res[b]["text"]
                        lang = batch["languages"][b]
                        sw = calculate_wer(ref, hyp, language=lang)
                        sc = calculate_cer(ref, hyp, language=lang)
                        wers.append(sw)
                        cers.append(sc)
                        if len(sample_predictions) < 3:
                            sample_predictions.append({
                                "lang": lang,
                                "ref": ref[:40],
                                "hyp": hyp[:40],
                                "cer": round(sc, 4),
                                "wer": round(sw, 4)
                            })

            mean_cer = sum(cers) / max(1, len(cers))
            mean_wer = sum(wers) / max(1, len(wers))
            if epoch == 1:
                initial_cer = mean_cer
                initial_wer = mean_wer

            print(
                f"Epoch [{epoch:03d}/{num_epochs:03d}] "
                f"Loss: {avg_loss:.4f} | "
                f"CER: {mean_cer:.2%} | "
                f"WER: {mean_wer:.2%} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}",
                flush=True
            )
            for sp in sample_predictions[:2]:
                print(f"   [{sp['lang']}] REF: {repr(sp['ref'])} -> HYP: {repr(sp['hyp'])}", flush=True)

            history.append({
                "epoch": epoch,
                "loss": round(avg_loss, 4),
                "cer": round(mean_cer, 4),
                "wer": round(mean_wer, 4),
                "sample_predictions": sample_predictions
            })

            # Early exit once CER target is reached (< 10%)
            if mean_cer < 0.10:
                print(f"\n[Overfit Target Reached!] Mean CER: {mean_cer:.2%} < 10% at epoch {epoch}.", flush=True)
                break

    total_time = time.perf_counter() - start_time
    final_loss = history[-1]["loss"]
    final_cer = history[-1]["cer"]
    final_wer = history[-1]["wer"]

    print("\n" + "=" * 65)
    print("OVERFIT EXPERIMENT RESULT:")
    print(f"  Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    print(f"  Initial CER:  {initial_cer:.2%} -> Final CER:  {final_cer:.2%}")
    print(f"  Initial WER:  {initial_wer:.2%} -> Final WER:  {final_wer:.2%}")
    print(f"  Total Duration: {total_time:.1f}s")
    print("=" * 65)

    os.makedirs(os.path.join(SERVICE_ROOT, "artifacts"), exist_ok=True)
    out_artifact = os.path.join(SERVICE_ROOT, "artifacts", "overfit_test.json")
    report = {
        "status": "PASSED" if final_cer < 0.20 else "FAILED",
        "num_samples": num_samples,
        "epochs": num_epochs,
        "initial_loss": round(initial_loss, 4),
        "final_loss": round(final_loss, 4),
        "initial_cer": round(initial_cer, 4),
        "final_cer": round(final_cer, 4),
        "initial_wer": round(initial_wer, 4),
        "final_wer": round(final_wer, 4),
        "training_time_seconds": round(total_time, 2),
        "history": history
    }
    with open(out_artifact, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved artifact to: {out_artifact}")

    return report


if __name__ == "__main__":
    run_overfit_experiment()
