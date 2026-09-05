from typing import List, Dict, Any, Tuple, Optional
import argparse
import json
import os
import sys

from src.asr.preprocessing import normalize_text, normalize_language_code, SUPPORTED_LANGUAGES


def compute_levenshtein_distance(ref: List[Any], hyp: List[Any]) -> Tuple[int, int, int, int]:
    """
    Standard dynamic programming Levenshtein distance computation.
    Returns:
        Tuple of (substitutions, deletions, insertions, total_ref_tokens)
    """
    r_len = len(ref)
    h_len = len(hyp)

    if r_len == 0:
        return 0, 0, h_len, 0
    if h_len == 0:
        return 0, r_len, 0, r_len

    # Distance matrix (r_len + 1, h_len + 1)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]

    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub = dp[i - 1][j - 1] + 1
                ins = dp[i][j - 1] + 1
                dlt = dp[i - 1][j] + 1
                dp[i][j] = min(sub, ins, dlt)

    # Traceback to count exact S, D, I
    i, j = r_len, h_len
    substitutions = 0
    deletions = 0
    insertions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            insertions += 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            break

    return substitutions, deletions, insertions, r_len


def calculate_wer(reference: str, hypothesis: str, language: str = "en") -> float:
    """Calculate Word Error Rate (WER) after language text normalization."""
    norm_ref = normalize_text(reference, language=language)
    norm_hyp = normalize_text(hypothesis, language=language)

    # For Mandarin, words are characters or segmented tokens
    if language == "zh":
        ref_words = list(norm_ref.replace(" ", ""))
        hyp_words = list(norm_hyp.replace(" ", ""))
    else:
        ref_words = norm_ref.split()
        hyp_words = norm_hyp.split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    s, d, i, n = compute_levenshtein_distance(ref_words, hyp_words)
    return min(1.0, float(s + d + i) / float(n))


def calculate_cer(reference: str, hypothesis: str, language: str = "en") -> float:
    """Calculate Character Error Rate (CER) after language text normalization."""
    norm_ref = normalize_text(reference, language=language)
    norm_hyp = normalize_text(hypothesis, language=language)

    ref_chars = list(norm_ref.replace(" ", ""))
    hyp_chars = list(norm_hyp.replace(" ", ""))

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    s, d, i, n = compute_levenshtein_distance(ref_chars, hyp_chars)
    return min(1.0, float(s + d + i) / float(n))


def evaluate_dataset(
    references: List[str],
    hypotheses: List[str],
    languages: List[str]
) -> Dict[str, Any]:
    """
    Evaluate ASR performance reporting WER and CER per target language:
    Hindi, Telugu, Tamil, Bengali, Urdu, Mandarin, and English.
    """
    assert len(references) == len(hypotheses) == len(languages), "Inputs must have equal length"

    lang_stats: Dict[str, Dict[str, Any]] = {
        lang: {
            "samples": 0,
            "word_errors": 0,
            "word_total": 0,
            "char_errors": 0,
            "char_total": 0,
            "wer": 0.0,
            "cer": 0.0
        }
        for lang in SUPPORTED_LANGUAGES
    }

    for ref, hyp, lang in zip(references, hypotheses, languages):
        l_code = normalize_language_code(lang)
        if l_code not in lang_stats:
            l_code = "en"

        norm_ref = normalize_text(ref, language=l_code)
        norm_hyp = normalize_text(hyp, language=l_code)

        # Word tokens
        if l_code == "zh":
            ref_words = list(norm_ref.replace(" ", ""))
            hyp_words = list(norm_hyp.replace(" ", ""))
        else:
            ref_words = norm_ref.split()
            hyp_words = norm_hyp.split()

        ref_chars = list(norm_ref.replace(" ", ""))
        hyp_chars = list(norm_hyp.replace(" ", ""))

        s_w, d_w, i_w, n_w = compute_levenshtein_distance(ref_words, hyp_words)
        s_c, d_c, i_c, n_c = compute_levenshtein_distance(ref_chars, hyp_chars)

        stats = lang_stats[l_code]
        stats["samples"] += 1
        stats["word_errors"] += (s_w + d_w + i_w)
        stats["word_total"] += max(1, n_w)
        stats["char_errors"] += (s_c + d_c + i_c)
        stats["char_total"] += max(1, n_c)

    # Compute final ratios
    total_samples = 0
    total_w_errors = 0
    total_w_count = 0
    total_c_errors = 0
    total_c_count = 0

    per_language_results = {}
    for lang, stats in lang_stats.items():
        if stats["samples"] > 0:
            stats["wer"] = round(float(stats["word_errors"]) / float(stats["word_total"]), 4)
            stats["cer"] = round(float(stats["char_errors"]) / float(stats["char_total"]), 4)
            total_samples += stats["samples"]
            total_w_errors += stats["word_errors"]
            total_w_count += stats["word_total"]
            total_c_errors += stats["char_errors"]
            total_c_count += stats["char_total"]
            per_language_results[lang] = stats

    overall_wer = round(float(total_w_errors) / max(1, total_w_count), 4) if total_w_count > 0 else 0.0
    overall_cer = round(float(total_c_errors) / max(1, total_c_count), 4) if total_c_count > 0 else 0.0

    return {
        "overall": {
            "total_samples": total_samples,
            "macro_wer": overall_wer,
            "macro_cer": overall_cer
        },
        "per_language": per_language_results
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Evaluate Non-Whisper ASR WER/CER on 7 Languages")
    parser.add_argument("--eval-file", "-f", type=str, required=True, help="JSON file with list of {reference, hypothesis, language}")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save evaluation report to JSON file")

    args = parser.parse_args()

    if not os.path.exists(args.eval_file):
        print(f"File not found: {args.eval_file}", file=sys.stderr)
        sys.exit(1)

    with open(args.eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    references = [item["reference"] for item in data]
    hypotheses = [item["hypothesis"] for item in data]
    languages = [item.get("language", "en") for item in data]

    results = evaluate_dataset(references, hypotheses, languages)
    output_str = json.dumps(results, indent=2, ensure_ascii=False)
    print(output_str)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_str)


if __name__ == "__main__":
    main()
