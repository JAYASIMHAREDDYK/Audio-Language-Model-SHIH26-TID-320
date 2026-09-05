import argparse
import json
import os
import sys
from typing import List, Dict, Any, Optional

from src.asr.preprocessing import SUPPORTED_LANGUAGES, normalize_language_code, normalize_text


def scan_local_directory(audio_dir: str, default_lang: str = "en") -> List[Dict[str, Any]]:
    """Scan local directory for audio files (.wav, .mp3, .flac) and pair with transcript if available."""
    samples = []
    supported_exts = {".wav", ".mp3", ".flac", ".ogg"}

    for root, _, files in os.walk(audio_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in supported_exts:
                audio_path = os.path.join(root, f)
                txt_path = os.path.splitext(audio_path)[0] + ".txt"
                text = ""
                if os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as tf:
                        text = tf.read().strip()
                
                samples.append({
                    "audio_filepath": os.path.abspath(audio_path),
                    "text": text,
                    "language": default_lang,
                    "duration": 0.0
                })

    return samples


def stream_indicvoices_subset(
    languages: List[str] = ["hi", "te", "ta", "bn", "ur"],
    max_samples_per_lang: int = 100,
    output_dir: str = "datasets/indicvoices_subset"
) -> List[Dict[str, Any]]:
    """
    Stream and materialize a small subset of IndicVoices adhering to the < 2.5 GB budget.
    """
    os.makedirs(output_dir, exist_ok=True)
    samples = []

    try:
        from datasets import load_dataset
    except ImportError:
        print("HuggingFace 'datasets' library not installed. Generating manifest entries only.")
        return samples

    for lang in languages:
        print(f"Streaming IndicVoices subset for {lang} (max {max_samples_per_lang} samples)...")
        try:
            ds = load_dataset(
                "ai4bharat/IndicVoices",
                lang,
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            count = 0
            for item in ds:
                if count >= max_samples_per_lang:
                    break
                text = item.get("normalized_text") or item.get("transcription") or item.get("text", "")
                audio_info = item.get("audio", {})
                
                audio_filename = f"indicvoices_{lang}_{count:04d}.wav"
                audio_save_path = os.path.join(output_dir, audio_filename)
                
                if isinstance(audio_info, dict) and "array" in audio_info:
                    try:
                        import scipy.io.wavfile as wavfile
                        import numpy as np
                        arr = (np.array(audio_info["array"]) * 32767).astype(np.int16)
                        wavfile.write(audio_save_path, audio_info.get("sampling_rate", 16000), arr)
                    except Exception:
                        pass
                
                samples.append({
                    "audio_filepath": os.path.abspath(audio_save_path),
                    "text": normalize_text(text, language=lang),
                    "language": lang,
                    "duration": float(len(audio_info.get("array", []))) / float(audio_info.get("sampling_rate", 16000)) if isinstance(audio_info, dict) and "array" in audio_info else 3.0
                })
                count += 1
        except Exception as e:
            print(f"Warning: Could not stream IndicVoices for {lang}: {e}")

    return samples


def generate_dataset_manifest(
    output_manifest_path: str,
    sources: Optional[List[Dict[str, str]]] = None,
    max_samples: Optional[int] = None
) -> str:
    """
    Generate unified JSON dataset manifest file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_manifest_path)), exist_ok=True)
    all_samples = []

    if sources:
        for src in sources:
            path = src.get("path")
            lang = src.get("language", "en")
            if path and os.path.exists(path):
                all_samples.extend(scan_local_directory(path, default_lang=lang))

    # If no local sources, write standardized sample template
    if not all_samples:
        sample_entries = [
            {"audio_filepath": "datasets/samples/hindi_sample_01.wav", "text": "पुलिस को जल्दी बुलाओ यहां आपातकाल है", "language": "hi", "duration": 3.2},
            {"audio_filepath": "datasets/samples/telugu_sample_01.wav", "text": "దయచేసి మాకు సహాయం చేయండి ఇక్కడ అత్యవసర పరిస్థితి ఉంది", "language": "te", "duration": 4.1},
            {"audio_filepath": "datasets/samples/tamil_sample_01.wav", "text": "தயவுசெய்து எங்களுக்கு உதவுங்கள் இங்கே அவசரநிலை உள்ளது", "language": "ta", "duration": 3.8},
            {"audio_filepath": "datasets/samples/bengali_sample_01.wav", "text": "দয়া করে আমাদের সাহায্য করুন এখানে জরুরী অবস্থা", "language": "bn", "duration": 3.5},
            {"audio_filepath": "datasets/samples/urdu_sample_01.wav", "text": "برائے مہربانی مدد کریں یہاں ایمرجنسی ہے", "language": "ur", "duration": 3.6},
            {"audio_filepath": "datasets/samples/mandarin_sample_01.wav", "text": "请帮帮我们这里有紧急情况快报警", "language": "zh", "duration": 4.0},
            {"audio_filepath": "datasets/samples/english_sample_01.wav", "text": "please send help immediately there is an emergency", "language": "en", "duration": 3.9},
        ]
        all_samples.extend(sample_entries)

    if max_samples and max_samples > 0:
        all_samples = all_samples[:max_samples]

    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    print(f"Generated dataset manifest with {len(all_samples)} samples at: {output_manifest_path}")
    return output_manifest_path


def main():
    parser = argparse.ArgumentParser(description="Multilingual ASR Dataset Manifest Generator (< 2.5 GB Budget)")
    parser.add_argument("--output", "-o", type=str, default="datasets/manifests/multilingual_asr_manifest.json", help="Output JSON manifest path")
    parser.add_argument("--dir", "-d", type=str, default=None, help="Local directory to scan")
    parser.add_argument("--lang", "-l", type=str, default="en", choices=SUPPORTED_LANGUAGES, help="Language code")
    parser.add_argument("--stream-indicvoices", action="store_true", help="Stream small subset from IndicVoices (Hindi, Telugu, Tamil, Bengali, Urdu)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples limit")

    args = parser.parse_args()

    if args.stream_indicvoices:
        samples = stream_indicvoices_subset(max_samples_per_lang=args.max_samples or 50)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"Saved streamed subset manifest to {args.output}")
    else:
        sources = [{"path": args.dir, "language": args.lang}] if args.dir else None
        generate_dataset_manifest(args.output, sources=sources, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
