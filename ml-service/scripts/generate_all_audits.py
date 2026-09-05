"""
Systematic Audit and Diagnostic Artifact Generator for Deep Multilingual Conformer ASR.
Generates:
1. artifacts/tokenizer_roundtrip.json
2. artifacts/ctc_length_audit.json
3. artifacts/gradient_audit.json
4. artifacts/data_alignment_audit.json
5. artifacts/ctc_debug_predictions.json
"""

import os
import sys
import json
import torch
import torch.nn as nn

SERVICE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SERVICE_ROOT not in sys.path:
    sys.path.insert(0, SERVICE_ROOT)

from src.asr.multilingual_asr import ASRModel
from src.asr.tokenizer import MultilingualTokenizer
from src.asr.preprocessing import normalize_text, load_and_preprocess_audio, TARGET_SAMPLE_RATE
from src.asr.evaluate import calculate_cer, calculate_wer
from training.dataset import MultilingualASRDataset, collate_asr_batch

ARTIFACTS_DIR = os.path.join(SERVICE_ROOT, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def audit_tokenizer_roundtrip(tokenizer):
    print("\n[1/5] Generating tokenizer_roundtrip.json...")
    langs = ["hi", "te", "ta", "bn", "mr", "kn", "zh"]
    results = {}
    total_checked = 0
    total_passed = 0

    manifest_path = os.path.join(SERVICE_ROOT, "data", "manifests", "val.jsonl")
    collected_samples = {l: [] for l in langs}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line.strip())
            l = it.get("language")
            if l in collected_samples and len(collected_samples[l]) < 5:
                collected_samples[l].append(it["text"])

    for lang in langs:
        lang_samples = collected_samples[lang]
        lang_results = []
        for text in lang_samples:
            total_checked += 1
            norm = normalize_text(text, language=lang)
            enc = tokenizer.encode(norm, language=lang, add_special_tokens=False)
            dec = tokenizer.decode(enc, skip_special_tokens=True)
            passed = (norm.strip() == dec.strip())
            if passed:
                total_passed += 1
            lang_results.append({
                "raw_text": text[:60],
                "normalized_text": norm[:60],
                "encoded_ids_sample": enc[:15],
                "token_count": len(enc),
                "decoded_text": dec[:60],
                "roundtrip_exact": passed,
                "unk_count": enc.count(tokenizer.unk_id)
            })
        results[lang] = {
            "samples_checked": len(lang_samples),
            "roundtrip_accuracy": round(sum(1 for r in lang_results if r["roundtrip_exact"]) / max(1, len(lang_results)), 4),
            "samples": lang_results
        }

    report = {
        "status": "PASSED" if total_passed == total_checked else "FAILED",
        "vocabulary_size": tokenizer.vocab_size(),
        "total_samples_checked": total_checked,
        "total_samples_passed": total_passed,
        "roundtrip_pass_rate": round(total_passed / max(1, total_checked), 4),
        "per_language": results
    }

    out_path = os.path.join(ARTIFACTS_DIR, "tokenizer_roundtrip.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved {out_path} (Pass rate: {report['roundtrip_pass_rate']:.2%})")
    return report


def audit_ctc_lengths(model, tokenizer):
    print("\n[2/5] Generating ctc_length_audit.json...")
    manifest_path = os.path.join(SERVICE_ROOT, "data", "manifests", "val.jsonl")
    dataset = MultilingualASRDataset(manifest_path, tokenizer, max_samples=40)
    
    samples_audit = []
    hop_length = 160
    total_valid = 0

    device = model.target_device
    model.eval()

    with torch.no_grad():
        for i in range(len(dataset)):
            it = dataset[i]
            w = it["waveform"]
            audio_samples = len(w)
            duration_s = round(audio_samples / float(TARGET_SAMPLE_RATE), 3)
            
            # Mel frames
            mel_frames = (audio_samples // hop_length) + 1
            
            # Forward model to get actual subsampled frames
            out = model(w.unsqueeze(0).to(device))
            subsampled_frames = int(out["lengths"].item())
            
            target_ids = it["token_ids"].tolist()
            u_target = len(target_ids)
            
            repeats = sum(1 for idx in range(1, len(target_ids)) if target_ids[idx] == target_ids[idx-1])
            min_required = u_target + repeats
            is_valid = subsampled_frames >= min_required
            if is_valid:
                total_valid += 1
                
            samples_audit.append({
                "sample_idx": i,
                "language": it["language"],
                "audio_samples": audio_samples,
                "duration_seconds": duration_s,
                "mel_frames": mel_frames,
                "subsampled_ctc_frames": subsampled_frames,
                "target_tokens": u_target,
                "repeated_consecutive_tokens": repeats,
                "min_required_frames": min_required,
                "valid_for_ctc": is_valid
            })

    report = {
        "status": "PASSED" if total_valid == len(samples_audit) else "WARNING",
        "total_audited": len(samples_audit),
        "valid_ctc_samples": total_valid,
        "valid_ratio": round(total_valid / max(1, len(samples_audit)), 4),
        "samples": samples_audit
    }

    out_path = os.path.join(ARTIFACTS_DIR, "ctc_length_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved {out_path} (Valid ratio: {report['valid_ratio']:.2%})")
    return report


def audit_gradients(model):
    print("\n[3/5] Generating gradient_audit.json...")
    device = model.target_device
    model.train()
    model.zero_grad()

    # Create synthetic variable-length batch
    s1 = torch.randn(int(3.0 * TARGET_SAMPLE_RATE))
    s2 = torch.randn(int(5.5 * TARGET_SAMPLE_RATE))
    batch_audio = torch.zeros(2, len(s2), device=device)
    batch_audio[0, :len(s1)] = s1.to(device)
    batch_audio[1, :] = s2.to(device)
    lengths = torch.tensor([len(s1), len(s2)], dtype=torch.long, device=device)
    
    targets = torch.tensor([[10, 20, 30, 3], [40, 50, 60, 70]], dtype=torch.long, device=device)
    target_lengths = torch.tensor([3, 4], dtype=torch.long, device=device)

    out = model(audio=batch_audio, audio_lengths=lengths, targets=targets, target_lengths=target_lengths)
    loss = out["loss"]
    loss.backward()

    gradient_report = {}
    
    # Check groups
    groups = {
        "ConvSubsampling": model.encoder.subsampling,
        "ConformerBlock_01": model.encoder.blocks[0],
        "ConformerBlock_06": model.encoder.blocks[5],
        "ConformerBlock_12": model.encoder.blocks[11],
        "CTC_LM_Head": model.decoder.lm_head
    }

    all_valid = True
    for name, module in groups.items():
        norms = []
        has_none = False
        has_nan = False
        has_inf = False
        total_p = 0
        for p in module.parameters():
            total_p += p.numel()
            if p.grad is None:
                has_none = True
            else:
                gnorm = p.grad.norm().item()
                if torch.isnan(p.grad).any():
                    has_nan = True
                if torch.isinf(p.grad).any():
                    has_inf = True
                norms.append(gnorm)

        avg_norm = sum(norms) / max(1, len(norms)) if norms else 0.0
        is_healthy = (not has_none) and (not has_nan) and (not has_inf) and (avg_norm > 0.0)
        if not is_healthy:
            all_valid = False

        gradient_report[name] = {
            "parameter_count": total_p,
            "average_grad_norm": round(avg_norm, 6),
            "max_grad_norm": round(max(norms) if norms else 0.0, 6),
            "has_none": has_none,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "healthy": is_healthy
        }

    report = {
        "status": "PASSED" if all_valid else "FAILED",
        "loss_value": round(loss.item(), 4),
        "parameter_groups": gradient_report
    }

    out_path = os.path.join(ARTIFACTS_DIR, "gradient_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved {out_path} (All healthy: {all_valid})")
    return report


def audit_data_alignment():
    print("\n[4/5] Generating data_alignment_audit.json...")
    langs = ["hi", "te", "ta", "bn", "mr", "kn", "zh"]
    counts = {l: 0 for l in langs}
    inspected_samples = []

    for split in ["train.jsonl", "val.jsonl"]:
        path = os.path.join(SERVICE_ROOT, "data", "manifests", split)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                it = json.loads(line.strip())
                l = it.get("language")
                if l in counts and counts[l] < 20:
                    counts[l] += 1
                    audio_p = it.get("audio") or it.get("audio_filepath", "")
                    exists = os.path.exists(audio_p)
                    sz = os.path.getsize(audio_p) if exists else 0
                    inspected_samples.append({
                        "id": it.get("id", "unknown"),
                        "language": l,
                        "speaker_id": it.get("speaker_id", "unknown"),
                        "audio_path": audio_p,
                        "file_exists": exists,
                        "file_size_bytes": sz,
                        "duration": it.get("duration", 0.0),
                        "text": it.get("text", "")[:50],
                        "passed": exists and sz > 44 and len(it.get("text", "")) > 0
                    })

    passed_count = sum(1 for s in inspected_samples if s["passed"])
    report = {
        "status": "PASSED" if passed_count == len(inspected_samples) else "FAILED",
        "total_audited": len(inspected_samples),
        "total_passed": passed_count,
        "pass_rate": round(passed_count / max(1, len(inspected_samples)), 4),
        "per_language_counts": counts,
        "samples": inspected_samples
    }

    out_path = os.path.join(ARTIFACTS_DIR, "data_alignment_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved {out_path} (Pass rate: {report['pass_rate']:.2%})")
    return report


def audit_ctc_predictions(model, tokenizer):
    print("\n[5/5] Generating ctc_debug_predictions.json...")
    manifest_path = os.path.join(SERVICE_ROOT, "data", "manifests", "val.jsonl")
    langs = ["hi", "te", "ta", "bn", "mr", "kn", "zh"]
    collected = {l: [] for l in langs}
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line.strip())
            l = it.get("language")
            if l in collected and len(collected[l]) < 3:
                collected[l].append(it)
            if all(len(v) == 3 for v in collected.values()):
                break
                
    device = model.target_device
    model.eval()
    predictions = []
    
    sample_idx = 0
    with torch.no_grad():
        for lang in langs:
            print(f"\n--- DEBUG PREDICTIONS: {lang.upper()} (3 samples) ---")
            for it in collected[lang]:
                audio_path = it.get("audio") or it.get("audio_filepath")
                waveform, _ = load_and_preprocess_audio(audio_path, TARGET_SAMPLE_RATE)
                w = waveform.to(device)
                ref_text = normalize_text(it["text"], language=lang)
                ref_ids = tokenizer.encode(ref_text, language=lang, add_special_tokens=False)
                
                out = model(w.unsqueeze(0) if w.ndim == 1 else w)
                logits = out["logits"]
                lengths = out["lengths"]
                
                raw_argmax_ids = logits.argmax(dim=-1)[0, :lengths[0]].tolist()
                dec_results = model.decoder.decode_greedy(logits, tokenizer, input_lengths=lengths)
                hyp_text = dec_results[0]["text"]
                pred_ids = dec_results[0]["token_ids"]
                
                cer = calculate_cer(ref_text, hyp_text, language=lang)
                wer = calculate_wer(ref_text, hyp_text, language=lang)
                
                print(f"REFERENCE:     {repr(ref_text[:40])}")
                print(f"REFERENCE IDS: {ref_ids[:15]}...")
                print(f"RAW ARGMAX:    {raw_argmax_ids[:15]}... (total frames: {len(raw_argmax_ids)})")
                print(f"COLLAPSED IDS: {pred_ids}")
                print(f"DECODED:       {repr(hyp_text[:40])}")
                print(f"CER: {cer:.2%} | WER: {wer:.2%}\n")
                
                predictions.append({
                    "sample_idx": sample_idx,
                    "language": lang,
                    "reference": ref_text,
                    "hypothesis": hyp_text,
                    "reference_ids": ref_ids,
                    "raw_argmax": raw_argmax_ids[:30],
                    "collapsed_ids": pred_ids,
                    "prediction_ids": pred_ids,
                    "raw_argmax_length": len(raw_argmax_ids),
                    "collapsed_token_count": len(pred_ids),
                    "sample_cer": round(cer, 4),
                    "sample_wer": round(wer, 4)
                })
                sample_idx += 1

    out_path = os.path.join(ARTIFACTS_DIR, "ctc_debug_predictions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"  -> Saved {out_path} ({len(predictions)} predictions evaluated across all 7 languages)")
    return predictions


def main():
    print("=" * 65)
    print("RUNNING ALL CTC AUDITS & PRODUCING REQUIRED DEBUG ARTIFACTS")
    print("=" * 65)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MultilingualTokenizer()
    model = ASRModel(device=device).to(device)

    audit_tokenizer_roundtrip(tokenizer)
    audit_ctc_lengths(model, tokenizer)
    audit_gradients(model)
    audit_data_alignment()
    audit_ctc_predictions(model, tokenizer)

    print("\nAll audits successfully executed!")


if __name__ == "__main__":
    main()
