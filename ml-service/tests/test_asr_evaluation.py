import pytest
from src.asr.evaluate import calculate_wer, calculate_cer, evaluate_dataset, compute_levenshtein_distance


class TestASREvaluation:
    """Test suite for WER and CER metrics computation."""

    def test_levenshtein_exact_cases(self):
        # Exact match
        s, d, i, n = compute_levenshtein_distance(["a", "b"], ["a", "b"])
        assert (s, d, i, n) == (0, 0, 0, 2)

        # Substitution
        s, d, i, n = compute_levenshtein_distance(["a", "b"], ["a", "c"])
        assert s == 1 and d == 0 and i == 0

        # Deletion
        s, d, i, n = compute_levenshtein_distance(["a", "b", "c"], ["a", "c"])
        assert d == 1 and s == 0 and i == 0

        # Insertion
        s, d, i, n = compute_levenshtein_distance(["a", "c"], ["a", "b", "c"])
        assert i == 1 and s == 0 and d == 0

    def test_wer_exact_values(self):
        ref = "help me please"
        hyp = "help us please"
        wer = calculate_wer(ref, hyp, language="en")
        # 1 substitution out of 3 words = 1/3 ~ 0.3333
        assert pytest.approx(wer, 0.01) == 0.3333

        # Identical
        assert calculate_wer(ref, ref, language="en") == 0.0

    def test_cer_exact_values(self):
        ref = "help"
        hyp = "held"
        cer = calculate_cer(ref, hyp, language="en")
        # 1 char error out of 4 = 0.25
        assert pytest.approx(cer, 0.01) == 0.25

    def test_multilingual_dataset_evaluation(self):
        references = [
            "पुलिस को जल्दी बुलाओ",                  # hi
            "సహాయం చేయండి",                       # te
            "உதவுங்கள்",                           # ta
            "সাহায্য করুন",                          # bn
            "مدد کریں",                            # ur
            "请快点救人",                          # zh
            "please send help immediately",       # en
        ]
        hypotheses = [
            "पुलिस को तुरंत बुलाओ",                 # hi (1 sub)
            "సహాయం చేయండి",                       # te (exact)
            "உதவுங்கள்",                           # ta (exact)
            "সাহায্য করুন",                          # bn (exact)
            "مدد کرو",                            # ur (1 sub)
            "请快救人",                            # zh (1 del)
            "please send assistance immediately", # en (1 sub)
        ]
        languages = ["hi", "te", "ta", "bn", "ur", "zh", "en"]

        report = evaluate_dataset(references, hypotheses, languages)

        assert "overall" in report
        assert "per_language" in report
        assert report["overall"]["total_samples"] == 7

        per_lang = report["per_language"]
        for l in languages:
            assert l in per_lang
            assert "wer" in per_lang[l]
            assert "cer" in per_lang[l]
            assert 0.0 <= per_lang[l]["wer"] <= 1.0
            assert 0.0 <= per_lang[l]["cer"] <= 1.0
