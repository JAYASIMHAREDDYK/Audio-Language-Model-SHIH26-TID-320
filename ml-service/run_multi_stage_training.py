"""
Multi-Stage Training Pipeline Runner for 7-Language Deep Conformer ASR.
Executes:
Stage 1: GPU Smoke Test (1 batch forward/backward/eval)
Stage 2: Overfit Verification Test (32 samples, 5 epochs, verifies gradient propagation & loss descent)
Stage 3: Experiment B (~1.0 GB corpus training, 3 epochs)
Stage 4: Experiment C (Full 4.0–4.5 GB corpus training with DynamicBatchSampler)
"""

import os
import sys
import time
import json
from typing import Dict, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from training.train_asr import train


def run_stage(stage_name: str, fn, *args, **kwargs) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print(f"STARTING {stage_name.upper()}")
    print("=" * 70)
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"\n[Finished {stage_name}] in {elapsed:.2f}s ({elapsed/60.0:.2f} mins)")
    return {
        "stage": stage_name,
        "elapsed_seconds": round(elapsed, 2),
        "result": result
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run 7-Language Multi-Stage Conformer Training")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "smoke", "overfit", "exp_b", "exp_c"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    pipeline_report = {}

    # Stage 1: Smoke Test
    if args.stage in ["all", "smoke"]:
        s1 = run_stage(
            "Stage 1: Smoke Test",
            train,
            train_manifest="data/manifests/train.jsonl",
            val_manifest="data/manifests/val.jsonl",
            smoke_test=True,
            device=args.device,
            output_dir="checkpoints/smoke_test"
        )
        pipeline_report["stage1_smoke"] = s1

    # Stage 2: Overfit Test
    if args.stage in ["all", "overfit"]:
        s2 = run_stage(
            "Stage 2: Overfit Test (32 samples, 5 epochs)",
            train,
            train_manifest="data/manifests/train.jsonl",
            val_manifest="data/manifests/val.jsonl",
            max_samples=32,
            epochs=5,
            batch_size=2,
            grad_accum_steps=2,
            learning_rate=3e-4,
            device=args.device,
            output_dir="checkpoints/overfit_test"
        )
        pipeline_report["stage2_overfit"] = s2

    # Stage 3: Experiment B (~1.0 GB)
    if args.stage in ["all", "exp_b"]:
        manifest_1gb = "data/manifests/train_1gb.jsonl"
        if not os.path.exists(manifest_1gb):
            manifest_1gb = "data/manifests/train.jsonl"
        s3 = run_stage(
            "Stage 3: Experiment B (~1.0 GB Corpus)",
            train,
            train_manifest=manifest_1gb,
            val_manifest="data/manifests/val.jsonl",
            test_manifest="data/manifests/test.jsonl",
            epochs=25,
            max_batch_audio_seconds=80.0,
            grad_accum_steps=2,
            learning_rate=5e-4,
            device=args.device,
            output_dir="checkpoints/exp_b_1gb"
        )
        pipeline_report["stage3_exp_b"] = s3

    # Stage 4: Experiment C (Full 4.0–4.5 GB)
    if args.stage in ["all", "exp_c"]:
        # Resume from Exp B best model if available
        resume_ckpt_c = "checkpoints/exp_b_1gb/best_model.pt"
        if not os.path.exists(resume_ckpt_c):
            resume_ckpt_c = None
        s4 = run_stage(
            "Stage 4: Experiment C (Full 4.0–4.5 GB Corpus)",
            train,
            train_manifest="data/manifests/train.jsonl",
            val_manifest="data/manifests/val.jsonl",
            test_manifest="data/manifests/test.jsonl",
            epochs=15,
            max_batch_audio_seconds=80.0,
            grad_accum_steps=4,
            learning_rate=1e-4,
            device=args.device,
            output_dir="checkpoints/exp_c_4gb",
            resume_checkpoint=resume_ckpt_c
        )
        pipeline_report["stage4_exp_c"] = s4

    # Save summary report
    summary_path = os.path.join(SCRIPT_DIR, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"ALL REQUESTED STAGES COMPLETED. SUMMARY SAVED TO {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
