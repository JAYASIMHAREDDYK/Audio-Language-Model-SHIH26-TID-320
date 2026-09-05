"""
Performance Benchmark for Deep Conformer ASR on RTX 4060 Laptop GPU.
Tests inference latency and RTF for 2s, 5s, and 10s audio.
"""
import os
import sys
import time
import torch
import numpy as np
import json
import wave

SERVICE_ROOT = os.path.dirname(os.path.abspath(__file__))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")



def create_test_wav(path, duration_sec, sr=16000):
    t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
    waveform = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((waveform * 32767).astype(np.int16).tobytes())


def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("DEEP CONFORMER ASR PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM: {total_vram:.2f} GB")

    from src.asr.multilingual_asr import ASRModel
    model = ASRModel(device=device)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    durations = [2, 5, 10]
    results = {}
    tmp_dir = os.path.join(SERVICE_ROOT, "artifacts")
    os.makedirs(tmp_dir, exist_ok=True)

    # Warmup
    warmup_wav = os.path.join(tmp_dir, "_warmup.wav")
    create_test_wav(warmup_wav, 1.0)
    with torch.no_grad():
        model.encode(warmup_wav, pool=False)
    if device.type == "cuda":
        torch.cuda.synchronize()

    for dur in durations:
        wav_path = os.path.join(tmp_dir, f"_bench_{dur}s.wav")
        create_test_wav(wav_path, dur)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        latencies = []
        n_runs = 5
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.no_grad():
                feats = model.encode(wav_path, pool=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        avg_lat = np.mean(latencies)
        rtf = avg_lat / dur

        peak_alloc = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**3) if device.type == "cuda" else 0

        results[f"{dur}s"] = {
            "duration_sec": dur,
            "avg_latency_ms": round(avg_lat * 1000, 1),
            "rtf": round(rtf, 4),
            "output_shape": list(feats.shape),
            "peak_allocated_gb": round(peak_alloc, 3),
            "peak_reserved_gb": round(peak_reserved, 3)
        }

        print(f"\n--- {dur}s Audio ---")
        print(f"  Latency: {avg_lat*1000:.1f}ms (avg over {n_runs} runs)")
        print(f"  RTF: {rtf:.4f} ({1/rtf:.0f}× real-time)")
        print(f"  Output: {list(feats.shape)}")
        if device.type == "cuda":
            print(f"  Peak VRAM allocated: {peak_alloc:.3f} GB")
            print(f"  Peak VRAM reserved: {peak_reserved:.3f} GB")

        os.remove(wav_path)

    os.remove(warmup_wav)

    # Save results
    bench_path = os.path.join(tmp_dir, "benchmark_results.json")
    bench_data = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "parameters": params,
        "results": results
    }
    with open(bench_path, "w") as f:
        json.dump(bench_data, f, indent=2)

    print(f"\n✓ Benchmark saved to {bench_path}")
    print("=" * 60)


if __name__ == "__main__":
    benchmark()
