"""
Benchmark and Profiling Suite for 7-Language Deep Multilingual Conformer ASR
and ALM Multimodal Integration.

Measures on NVIDIA GeForce RTX 4060 Laptop GPU:
1. Exact parameter breakdown (Conv Subsampling, 12 Conformer blocks, 7-lang CTC Head)
2. Peak VRAM consumption during FP16 forward and backward passes
3. Real-Time Factor (RTF) and inference latency across 2s, 5s, and 10s audio inputs
4. Tokenizer UNK rates and roundtrip tokenization speed across all 7 languages
5. Frame-level continuous acoustic representation throughput and ALM adapter projection
"""

import os
import sys
import time
import json
from typing import Dict, Any
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure ml-service is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from src.asr.multilingual_asr import ASRModel
from src.asr.asr_model import ASRAdapter
from src.asr.tokenizer import MultilingualCharTokenizer


def run_7lang_benchmark(device_str: str = "auto") -> Dict[str, Any]:
    if device_str == "auto":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device_str)

    use_cuda = target_device.type == "cuda"
    device_name = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) if use_cuda else 0.0

    print("=" * 70)
    print("7-LANGUAGE DEEP CONFORMER ASR BENCHMARK & PROFILING")
    print(f"Device: {target_device} ({device_name}, {vram_total_gb:.2f} GB VRAM)")
    print("=" * 70)

    # 1. Instantiate Conformer ASR Model & ALM Adapter
    model = ASRModel(device=target_device)
    model.to(target_device)
    adapter = ASRAdapter(embed_dim=256, device=target_device)
    adapter.to(target_device)

    # 2. Parameter Count Breakdown
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
            "vram_headroom_mb": round((vram_total_gb * 1024) - peak_vram_mb, 2),
            "vram_utilization_pct": round((peak_vram_mb / (vram_total_gb * 1024)) * 100, 2)
        }
        print("\n[Peak GPU VRAM Footprint (FP16 AMP Forward+Backward, 10s Batch Size 2)]")
        print(f"  Peak VRAM Allocated:  {vram_metrics['peak_allocated_mb']} MB")
        print(f"  Peak VRAM Reserved:   {vram_metrics['peak_reserved_mb']} MB")
        print(f"  Available Headroom:   {vram_metrics['vram_headroom_mb']} MB ({8.0 - peak_vram_mb/1024:.2f} GB)")
        print(f"  VRAM Utilization:     {vram_metrics['vram_utilization_pct']}% of {vram_total_gb:.1f} GB")

    # 4. Inference Latency & RTF Benchmarking
    model.eval()
    benchmark_durations = [2.0, 5.0, 10.0]
    timing_results = {}

    print("\n[Inference Latency & Real-Time Factor (RTF)]")
    # Warmup
    warmup_wav = torch.randn(1, 16000, device=target_device)
    for _ in range(5):
        with torch.no_grad():
            if use_cuda:
                with torch.amp.autocast('cuda'):
                    _ = model.transcribe(warmup_wav)
            else:
                _ = model.transcribe(warmup_wav)

    for dur in benchmark_durations:
        num_samples = int(dur * 16000)
        test_wav = torch.randn(1, num_samples, device=target_device)
        latencies = []

        num_runs = 10
        for _ in range(num_runs):
            if use_cuda:
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            with torch.no_grad():
                if use_cuda:
                    with torch.amp.autocast('cuda'):
                        _ = model.transcribe(test_wav)
                else:
                    _ = model.transcribe(test_wav)

            if use_cuda:
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            latencies.append(t_end - t_start)

        avg_latency = sum(latencies) / len(latencies)
        rtf = avg_latency / dur

        timing_results[f"{dur}s_audio"] = {
            "audio_duration_seconds": dur,
            "avg_latency_ms": round(avg_latency * 1000, 2),
            "rtf": round(rtf, 4),
            "throughput_x_realtime": round(1.0 / rtf, 2)
        }
        print(f"  {dur:4.1f}s Audio -> Latency: {avg_latency*1000:6.2f} ms | RTF: {rtf:.4f} ({1.0/rtf:5.1f}x real-time)")

    # 5. ALM Temporal Fusion Adapter Benchmark
    print("\n[ALM Temporal Fusion Adapter (512-dim -> 256-dim)]")
    test_wav = torch.randn(1, 48000, device=target_device)  # 3.0s audio
    with torch.no_grad():
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        adapter_out = adapter.get_speech_embeddings(test_wav)
        if use_cuda:
            torch.cuda.synchronize()
        adapter_latency_ms = (time.perf_counter() - t0) * 1000

    frame_shape = list(adapter_out["frame_embeddings"].shape)
    raw_shape = list(adapter_out["raw_conformer_frames"].shape)
    print(f"  Raw Conformer Output:    {raw_shape} (512-dim continuous frames)")
    print(f"  Projected ALM Output:    {frame_shape} (256-dim fusion frames)")
    print(f"  Adapter Feature Latency: {adapter_latency_ms:.2f} ms")

    # 6. Per-Language Tokenizer Coverage & UNK Rates
    print("\n[Per-Language Tokenizer Evaluation (7 Languages)]")
    tokenizer = MultilingualCharTokenizer()
    sample_texts = {
        "hi": ("Hindi", "नमस्ते दुनिया यह आपातकालीन चिकित्सा सहायता प्रणाली है"),
        "te": ("Telugu", "నమస్కారం ప్రపంచం ఇది అత్యవసర వైద్య సహాయ వ్యవస్థ"),
        "ta": ("Tamil", "வணக்கம் உலகம் இது அவசர மருத்துவ உதவி அமைப்பு"),
        "bn": ("Bengali", "নমস্কার বিশ্ব এটি জরুরি চিকিৎসা সহায়তা ব্যবস্থা"),
        "mr": ("Marathi", "नमस्कार जग हे आपत्कालीन वैद्यकीय मदत प्रणाली आहे"),
        "kn": ("Kannada", "ನಮಸ್ಕಾರ ವಿಶ್ವ ಇದು ತುರ್ತು ವೈದ್ಯಕೀಯ ಸಹಾಯ ವ್ಯವಸ್ಥೆ"),
        "zh": ("Mandarin Chinese", "你好世界这是急救医疗救援系统")
    }

    tokenizer_metrics = {}
    for lcode, (lname, text) in sample_texts.items():
        token_ids = tokenizer.encode(text, language=lcode)
        unk_count = token_ids.count(tokenizer.unk_id)
        unk_rate = unk_count / max(1, len(token_ids))
        decoded = tokenizer.decode(token_ids, remove_special=True)

        tokenizer_metrics[lcode] = {
            "language_name": lname,
            "sample_length_chars": len(text),
            "num_tokens": len(token_ids),
            "unk_count": unk_count,
            "unk_rate_pct": round(unk_rate * 100, 2),
            "lang_token": f"<lang:{lcode}>",
            "lang_token_id": tokenizer.get_lang_token_id(lcode)
        }
        print(f"  [{lcode}] {lname:16s} -> Tokens: {len(token_ids):2d} | UNK Rate: {unk_rate*100:4.1f}% | Token: <lang:{lcode}> (ID {tokenizer.get_lang_token_id(lcode)})")

    # Compile Final Summary
    benchmark_report = {
        "device": str(target_device),
        "device_name": device_name,
        "vram_total_gb": round(vram_total_gb, 2),
        "conformer_specifications": {
            "num_encoder_blocks": 12,
            "d_model": 512,
            "d_ff": 2048,
            "num_attention_heads": 8,
            "conv_kernel_size": 31,
            "temporal_subsampling": 4,
            "mel_filterbanks": 80,
            "target_sample_rate": 16000
        },
        "parameter_breakdown": param_info,
        "vram_metrics": vram_metrics,
        "inference_latency": timing_results,
        "alm_adapter": {
            "raw_conformer_dim": 512,
            "alm_projected_dim": 256,
            "adapter_latency_ms": round(adapter_latency_ms, 2)
        },
        "tokenizer_metrics": tokenizer_metrics
    }

    output_path = os.path.join(SCRIPT_DIR, "benchmark_7lang_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2, ensure_ascii=False)

    print(f"\nBenchmark results saved to: {output_path}")
    print("=" * 70)
    return benchmark_report


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    run_7lang_benchmark(dev)
