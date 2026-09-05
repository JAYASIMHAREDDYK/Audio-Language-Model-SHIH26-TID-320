import argparse
import json
import os
import sys
from typing import Optional, Dict, Any

from src.asr.multilingual_asr import ASRModel
from src.asr.preprocessing import normalize_language_code, SUPPORTED_LANGUAGES


def run_inference(
    audio_path: str,
    language: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    extract_embeddings: bool = False
) -> Dict[str, Any]:
    """
    Run Multilingual Non-Whisper ASR inference on an audio file.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Determine device
    if device == "auto":
        import torch
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device

    model = ASRModel(model_name_or_path=checkpoint_path, device=device_str)

    lang_code = normalize_language_code(language) if language else None
    result = model.transcribe(audio_path, language=lang_code)

    if extract_embeddings:
        embs = model.encode(audio_path)
        pooled = model.encode(audio_path, pool=True)
        result["speech_embeddings"] = {
            "frame_shape": list(embs.shape),
            "pooled_shape": list(pooled.shape),
            "embedding_dim": int(embs.size(-1))
        }

    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Lahari Multilingual Non-Whisper ASR Inference CLI (7 Languages)"
    )
    parser.add_argument("--audio", "-a", type=str, required=True, help="Path to input audio file (.wav, .mp3, etc.)")
    parser.add_argument("--language", "-l", type=str, default=None, choices=SUPPORTED_LANGUAGES + [None], help="Language code (hi, te, ta, bn, ur, zh, en)")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--device", "-d", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Inference device")
    parser.add_argument("--extract-embeddings", action="store_true", help="Extract and output speech embeddings metadata for Core ALM")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save JSON output")

    args = parser.parse_args()

    try:
        res = run_inference(
            audio_path=args.audio,
            language=args.language,
            checkpoint_path=args.checkpoint,
            device=args.device,
            extract_embeddings=args.extract_embeddings
        )

        output_str = json.dumps(res, indent=2, ensure_ascii=False)
        print(output_str)

        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_str)

    except Exception as e:
        print(f"Error running ASR inference: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
