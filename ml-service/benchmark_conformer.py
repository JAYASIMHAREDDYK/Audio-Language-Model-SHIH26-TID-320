"""
Benchmark and Profiling Script for Deep Multilingual Conformer ASR.
Measures on NVIDIA GeForce RTX 4060 Laptop GPU:
1. Exact parameter breakdown (Frontend, Encoder, CTC Head, Total)
2. Peak VRAM consumption during FP16 forward and backward passes
3. Real-Time Factor (RTF) and inference latency across multiple audio lengths
"""

import os
import sys
import time
import json
from typing import Dict, Any
import torch

# Ensure ml-service is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.asr.multilingual_asr import ASRModel


def run_benchmark(device_str: str = "auto") -> Dict[str, Any]:
    if device_str == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device_str)

    use_cuda = target_device.type == "cuda"
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if use_cuda else 0.0

    print("=" * 65)
    print("DEEP CONFORMER ASR BENCHMARK & PROFILING")
    print(f"Device: {target_device} ({device_name}, {vram_total_gb:.2f} GB VRAM)")
    print("=" * 65)

    # 1. Instantiate Model
    model = ASRModel(device=target_device)
    model.to(target_device)

    # 2. Measure Parameter Counts
    param_info = model.count_parameters()
    print("\n[Parameter Count Breakdown]")
    print(f"  Acoustic Frontend (Conv2D Subsampling): {param_info['frontend_parameters']:,}")
    print(f"  12-Layer Conformer Encoder:            {param_info['encoder_parameters']:,}")
    print(f"  Multilingual CTC Projection Head:      {param_info['decoder_parameters']:,}")
    print(f"  Total Model Parameters:                {param_info['total_parameters']:,}")
    print(f"  Trainable Parameters:                  {param_info['trainable_parameters']:,}")

    # 3. Measure Peak VRAM Consumption
    vram_metrics = {}
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 10.0 seconds of 16kHz audio for batch size 2
        dummy_audio = torch.randn(2, 160000, device=target_device, dtype=torch.float32)
        dummy_targets = torch.randint(1, 300, (2, 40), device=target_device, dtype=torch.long)
        dummy_lengths = torch.tensor([40, 35], device=target_device, dtype=torch.long)

        # Forward + Backward under FP16 AMP
        model.train()
        scaler = torch.amp.GradScaler('cuda')
        with torch.amp.autocast('cuda'):
            outputs = model(audio=dummy_audio, targets=dummy_targets, target_lengths=dummy_lengths)
            loss = outputs["loss"]
        scaler.scale(loss).backward()

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved_vram_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        vram_metrics = {
            "peak_allocated_mb": round(peak_vram_mb, 2),
            "peak_reserved_mb": round(reserved_vram_mb, 2),
            "vram_utilization_pct": round((peak_vram_mb / (vram_total_gb * 1024)) * 100.0, 2)
        }
        print("\n[Peak VRAM Utilization (10s audio, Batch=2, FP16 AMP)]")
        print(f"  Peak Allocated: {vram_metrics['peak_allocated_mb']} MB")
        print(f"  Peak Reserved:  {vram_metrics['peak_reserved_mb']} MB")
        print(f"  VRAM Usage:     {vram_metrics['vram_utilization_pct']:.2f}% of 8.0 GB")

    # 4. Measure Inference Latency & Real-Time Factor (RTF)
    model.eval()
    durations_tested = [2.0, 5.0, 10.0]
    rtf_results = {}

    print("\n[Inference Latency & Real-Time Factor (RTF)]")
    with torch.no_grad():
        for dur in durations_tested:
            audio_samples = int(dur * 16000)
            test_tensor = torch.randn(1, audio_samples, device=target_device, dtype=torch.float32)

            # Warmup
            for _ in range(5):
                _ = model.encode(test_tensor)

            # Timed iterations
            iterations = 30
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            for _ in range(iterations):
                if use_cuda:
                    with torch.amp.autocast('cuda'):
                        _ = model.encode(test_tensor)
                else:
                    _ = model.encode(test_tensor)

            if use_cuda:
                torch.cuda.synchronize()
            total_elapsed = time.perf_counter() - t0
            avg_latency_sec = total_elapsed / iterations
            rtf = avg_latency_sec / dur

            rtf_results[f"{dur}s_audio"] = {
                "audio_duration_sec": dur,
                "latency_sec": round(avg_latency_sec, 4),
                "latency_ms": round(avg_latency_sec * 1000, 2),
                "rtf": round(rtf, 4),
                "realtime_capable": rtf < 1.0
            }
            print(
                f"  Duration: {dur:4.1f}s | "
                f"Latency: {avg_latency_sec * 1000:6.2f} ms | "
                f"RTF: {rtf:.4f} (Real-time: {rtf < 1.0})"
            )

    # Compile Benchmark Report
    benchmark_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": str(target_device),
            "gpu_name": device_name,
            "total_vram_gb": round(vram_total_gb, 2)
        },
        "parameters": param_info,
        "vram_profile": vram_metrics,
        "rtf_latency": rtf_results
    }

    out_path = os.path.join(SCRIPT_DIR, "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, ensure_ascii=False, indent=2)
    print(f"\nBenchmark report saved at: {out_path}")
    print("=" * 65)

    return benchmark_report


if __name__ == "__main__":
    run_benchmark()
