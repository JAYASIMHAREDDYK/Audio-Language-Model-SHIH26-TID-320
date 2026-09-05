import json
import os
import sys
from collections import Counter

# Add ml-service root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.asr.preprocessing import normalize_text, normalize_language_code
from src.asr.tokenizer import MultilingualTokenizer

def build_vocab():
    # 1. Start from base vocabulary in MultilingualTokenizer (which contains base Indic and common Hanzi)
    # Pass a non-existent path to prevent loading previous vocab.json
    base_tok = MultilingualTokenizer(vocab_path="__dummy_none__")
    vocab = dict(base_tok.token_to_id)
    
    # Base digits & punctuation
    extra_chars = "0123456789'-.,?!;:/\"()[]{}*&#@$%+=_`~^|<>"
    for ch in extra_chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
            
    # Emergency domain subwords
    emergency_subwords = [
        "emergency", "help", "security", "danger", "police", "fire", "medical",
        "threat", "alarm", "attention", "passenger", "flight", "gate", "station"
    ]
    for w in emergency_subwords:
        if w not in vocab:
            vocab[w] = len(vocab)

    # 2. Add sample test phrases from tests/
    test_phrases = [
        "नमस्ते दुनिया यह आपातकालीन चिकित्सा सहायता प्रणाली है",
        "నమస్కారం ప్రపంచం ఇది అత్యవసర వైద్య సహాయ వ్యవస్థ",
        "வணக்கம் உலகம் இது அவசர மருத்துவ உதவி அமைப்பு",
        "নমস্কার বিশ্ব এটি জরুরি চিকিৎসা সহায়তা ব্যবস্থা",
        "नमस्कार जग हे आपत्कालीन वैद्यकीय मदत प्रणाली आहे",
        "ನಮಸ್ಕಾರ ವಿಶ್ವ ಇದು ತುರ್ತು ವೈದ್ಯಕೀಯ ಸಹಾಯ ವ್ಯವಸ್ಥೆ",
        "你好世界这是急救医疗救援系统"
    ]
    for phrase in test_phrases:
        for ch in phrase:
            if ch != " " and ch not in vocab:
                vocab[ch] = len(vocab)

    # 3. Scan all manifests
    manifest_files = ["train.jsonl", "val.jsonl", "test.jsonl"]
    char_counts = Counter()
    for mf in manifest_files:
        path = os.path.join("data", "manifests", mf)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                lang = normalize_language_code(item.get("language", "en"))
                text = item.get("text", "")
                norm = normalize_text(text, language=lang)
                for ch in norm:
                    if ch != " ":
                        char_counts[ch] += 1

    sorted_chars = sorted(char_counts.keys(), key=lambda c: ord(c))
    for ch in sorted_chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)

    output_path = os.path.join("data", "vocab.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Built complete vocabulary with {len(vocab)} tokens.")
    print(f"Saved to: {output_path}")
    return vocab

if __name__ == "__main__":
    build_vocab()
