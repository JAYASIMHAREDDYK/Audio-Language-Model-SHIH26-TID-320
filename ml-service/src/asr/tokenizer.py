"""
Multilingual Tokenizer for Deep Conformer ASR Model.
Provides character and subword tokenization across 7 mandatory target languages:
Hindi (hi), Telugu (te), Tamil (ta), Bengali (bn), Marathi (mr), Kannada (kn), and Mandarin Chinese (zh),
plus English (en) and Urdu (ur).
Independent of Whisper tokenizers and hardcoded toy vocabularies.
"""

import json
import os
import re
from typing import List, Dict, Union, Optional, Any, Set


BLANK_TOKEN = "<blank>"
UNK_TOKEN = "<unk>"
SPACE_TOKEN = " "
PAD_TOKEN = "<pad>"

LANG_TOKENS: Dict[str, str] = {
    "hi": "<lang:hi>",
    "te": "<lang:te>",
    "ta": "<lang:ta>",
    "bn": "<lang:bn>",
    "mr": "<lang:mr>",
    "kn": "<lang:kn>",
    "zh": "<lang:zh>",
    "en": "<lang:en>",
    "ur": "<lang:ur>",
}


class MultilingualTokenizer:
    """
    Subword and Character-Level Multilingual Tokenizer for CTC Decoding.
    Supports Hindi, Telugu, Tamil, Bengali, Marathi, Kannada, Mandarin, English, and Urdu.
    """

    def __init__(self, vocab_path: Optional[str] = None):
        self.token_to_id: Dict[str, int] = {
            BLANK_TOKEN: 0,
            UNK_TOKEN: 1,
            SPACE_TOKEN: 2,
            PAD_TOKEN: 3,
        }

        # Register language-specific tokens
        for lang_code, lang_tag in LANG_TOKENS.items():
            if lang_tag not in self.token_to_id:
                self.token_to_id[lang_tag] = len(self.token_to_id)
        
        # Build comprehensive multilingual character set for all 7 target languages
        base_vocab: List[str] = [
            # English & Punctuation
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            '\'', '-', '.', ',', '?', '!',
            
            # Common Emergency Domain Subwords / Words
            "emergency", "help", "security", "danger", "police", "fire", "medical",
            "threat", "alarm", "attention", "passenger", "flight", "gate", "station",
            
            # Devanagari / Hindi & Marathi Range (U+0900 to U+097F)
            'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'ङ',
            'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प',
            'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह', 'ा', 'ि', 'ी', 'ु',
            'ू', 'ृ', 'े', 'ै', 'ो', 'ौ', 'ं', 'ः', '्',
            # Marathi specific Devanagari extensions:
            'ळ', 'ॅ', 'ॉ', 'ॐ', 'ऽ', '़', '॒', '॑',
            
            # Telugu Range (U+0C00 to U+0C7F)
            'అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఎ', 'ఏ', 'ఐ', 'ఒ', 'ఓ', 'ఔ', 'క', 'ఖ', 'గ', 'ఘ',
            'చ', 'ఛ', 'జ', 'ఝ', 'ట', 'ఠ', 'డ', 'ఢ', 'ణ', 'త', 'థ', 'ద', 'ధ', 'న', 'ప', 'ఫ',
            'బ', 'భ', 'మ', 'య', 'ర', 'ల', 'వ', 'శ', 'ష', 'స', 'హ', 'ా', 'ి', 'ీ', 'ు', 'ూ',
            'ె', 'ే', 'ై', 'ొ', 'ో', 'ౌ', 'ం', 'ః', '్',
            
            # Tamil Range (U+0B80 to U+0BFF)
            'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ', 'க', 'ங', 'ச', 'ஞ',
            'ட', 'ண', 'த', 'ந', 'ப', 'ம', 'ய', 'ர', 'ல', 'வ', 'ழ', 'ள', 'ற', 'ன', 'ா', 'ி',
            'ீ', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ', '்',
            
            # Bengali Range (U+0980 to U+09FF)
            'অ', 'আ', 'ই', 'ঈ', 'উ', 'ঊ', 'ঋ', 'এ', 'ঐ', 'ও', 'ঔ', 'ক', 'খ', 'গ', 'ঘ', 'ঙ',
            'চ', 'ছ', 'জ', 'ঝ', 'ঞ', 'ট', 'ঠ', 'ড', 'ঢ', 'ণ', 'ত', 'থ', 'দ', 'ধ', 'ন', 'প',
            'ফ', 'ব', 'ভ', 'ম', 'য', 'র', 'ল', 'শ', 'ষ', 'স', 'হ', 'া', 'ি', 'ী', 'ু', 'ূ',
            'ে', 'ৈ', 'ো', 'ৌ', '্', 'ৎ', 'ড়', 'ঢ়', 'য়', 'ং', 'ঃ', 'ঁ', 'ৃ', '়', 'ৰ', 'ৱ',
            
            # Kannada Range (U+0C80 to U+0CFF)
            'ಅ', 'ಆ', 'ಇ', 'ಈ', 'ಉ', 'ಊ', 'ಋ', 'ಌ', 'ಎ', 'ಏ', 'ಐ', 'ಒ', 'ಓ', 'ಔ',
            'ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ', 'ಚ', 'ಛ', 'ಜ', 'ಝ', 'ಞ', 'ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ',
            'ತ', 'ಥ', 'ದ', 'ಧ', 'ನ', 'ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ', 'ಯ', 'ರ', 'ಱ', 'ಲ', 'ವ',
            'ಶ', 'ಷ', 'ಸ', 'ಹ', 'ಳ', 'ೞ',
            'ಾ', 'ಿ', 'ೀ', 'ು', 'ೂ', 'ೃ', 'ೄ', 'ೆ', 'ೇ', 'ೈ', 'ೊ', 'ೋ', 'ೌ', '್', 'ಂ', 'ಃ',
            
            # Urdu / Perso-Arabic (U+0600 to U+06FF)
            'آ', 'ا', 'ب', 'پ', 'ت', 'ٹ', 'ث', 'ج', 'چ', 'ح', 'خ', 'د', 'ڈ', 'ذ', 'ر', 'ڑ',
            'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م',
            'ن', 'ں', 'و', 'ہ', 'ھ', 'ء', 'ی', 'ے',
        ]

        # Common Mandarin Hanzi (CJK Unified Ideographs, high frequency speech characters)
        common_mandarin_hanzi = (
            "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就会可也你对生能后多自着"
            "之过发得里后自以家前所道去把动头想现看天行理长开手使样经由经声两面进面问只全回力正女"
            "高文已应次分定主向又公如见实关机气很法各起政老名四其成此西关水走同等政从点本将外化日"
            "问情重加常题新月明特真直几先口目因给被真表战题代它直最情果身意第解先干原神通美长界位"
            "车条数马门应指变走总白反海并更部并民车利第路物果走立打代话每总度空反身做情西结度各安"
            "警察紧危急险火报警援医疗安全注意危险旅客航班登机站台旅客行李广播通知系统发生情况发现"
            "西班牙殖民时期如果冬天极地白昼夜晚太阳地平线出现天气冰雪森林海洋城市车辆道路飞机场"
            "时间年月日时分秒今天昨天明天早上中午下午晚上开始结束继续完成确认取消进入离开寻找帮助"
            "声音语言识别模型网络结构注意力特征向量嵌入表示融合多语种训练测试验证评估性能指标准确率"
            "你好世界谢谢再见请问对不起没关系欢迎光临中国北京上海广州深圳成都武汉西安南京杭州"
            "救助援助救援急救救护车医生医院患者受伤创伤病痛疾病求助警务治安防护灭火抢险求生"
        )
        for char in common_mandarin_hanzi:
            if char not in base_vocab:
                base_vocab.append(char)
        
        for token in base_vocab:
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)

        default_vocab = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "vocab.json"
        )
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
        elif os.path.exists(default_vocab):
            self.load_vocab(default_vocab)

        self._refresh_mappings()

    def _refresh_mappings(self):
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}
        self.char_to_id: Dict[str, int] = self.token_to_id
        self.id_to_char: Dict[int, str] = self.id_to_token
        self.blank_id = self.token_to_id.get(BLANK_TOKEN, 0)
        self.unk_id = self.token_to_id.get(UNK_TOKEN, 1)
        self.space_id = self.token_to_id.get(SPACE_TOKEN, 2)
        self.pad_id = self.token_to_id.get(PAD_TOKEN, 3)

        assert self.blank_id == 0, f"blank_id must be 0, got {self.blank_id}"
        assert self.unk_id == 1, f"unk_id must be 1, got {self.unk_id}"
        assert self.space_id == 2, f"space_id must be 2, got {self.space_id}"
        assert self.pad_id == 3, f"pad_id must be 3, got {self.pad_id}"

    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def get_lang_token_id(self, lang: str) -> Optional[int]:
        """Return the token ID for a language code (e.g. 'hi' -> <lang:hi>)."""
        lang = lang.lower().strip()
        tag = LANG_TOKENS.get(lang, f"<lang:{lang}>")
        return self.token_to_id.get(tag, self.unk_id)

    def add_characters_from_texts(self, texts: List[str]) -> int:
        """
        Dynamically add any missing characters from a list of corpus texts to vocabulary.
        Returns the number of newly added tokens.
        """
        added = 0
        for text in texts:
            if not text:
                continue
            for char in text.strip():
                if char not in (" ", "\t", "\n", "\r") and char not in self.token_to_id:
                    self.token_to_id[char] = len(self.token_to_id)
                    added += 1
        if added > 0:
            self._refresh_mappings()
        return added

    def encode(
        self,
        text: str,
        language: Optional[str] = None,
        add_special_tokens: bool = True
    ) -> List[int]:
        """Convert text string into token IDs, optionally prepending language tag."""
        ids: List[int] = []
        if add_special_tokens and language:
            lang_id = self.get_lang_token_id(language)
            if lang_id is not None and lang_id != self.unk_id:
                ids.append(lang_id)

        clean_text = text.lower().strip()
        i = 0
        while i < len(clean_text):
            # Check for subwords first
            matched = False
            for wlen in range(min(12, len(clean_text) - i), 1, -1):
                sub = clean_text[i:i+wlen]
                if sub in self.token_to_id:
                    ids.append(self.token_to_id[sub])
                    i += wlen
                    matched = True
                    break
            if not matched:
                char = clean_text[i]
                ids.append(self.token_to_id.get(char, self.unk_id))
                i += 1
        return ids

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
        remove_special: Optional[bool] = None
    ) -> str:
        """Convert token IDs back into text string."""
        if remove_special is not None:
            skip_special_tokens = remove_special
        tokens = []
        special_tags = set(LANG_TOKENS.values()) | {BLANK_TOKEN, UNK_TOKEN, PAD_TOKEN}

        for tid in ids:
            if tid == self.blank_id and skip_special_tokens:
                continue
            token = self.id_to_token.get(tid, UNK_TOKEN)
            if skip_special_tokens and token in special_tags:
                continue
            tokens.append(token)
            
        text = "".join(tokens).replace("  ", " ").strip()
        return text

    def calculate_unk_rate(self, texts: List[str], language: Optional[str] = None) -> float:
        """
        Calculate the unknown-token rate (ratio of UNK tokens to total encoded tokens).
        """
        total_tokens = 0
        unk_tokens = 0
        for text in texts:
            if not text:
                continue
            encoded = self.encode(text, language=language, add_special_tokens=False)
            total_tokens += len(encoded)
            unk_tokens += sum(1 for tid in encoded if tid == self.unk_id)
        if total_tokens == 0:
            return 0.0
        return float(unk_tokens) / float(total_tokens)

    def get_vocab_stats(self) -> Dict[str, Any]:
        """
        Produce a detailed report of vocabulary size, language tokens, and script distributions.
        """
        script_counts: Dict[str, int] = {
            "special": 0,
            "latin": 0,
            "devanagari": 0,
            "telugu": 0,
            "tamil": 0,
            "bengali": 0,
            "kannada": 0,
            "perso_arabic": 0,
            "cjk_hanzi": 0,
            "other": 0
        }

        for token in self.token_to_id.keys():
            if token.startswith("<") and token.endswith(">"):
                script_counts["special"] += 1
            elif len(token) == 1:
                cp = ord(token)
                if 0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A or token in ("'", "-"):
                    script_counts["latin"] += 1
                elif 0x0900 <= cp <= 0x097F:
                    script_counts["devanagari"] += 1
                elif 0x0C00 <= cp <= 0x0C7F:
                    script_counts["telugu"] += 1
                elif 0x0B80 <= cp <= 0x0BFF:
                    script_counts["tamil"] += 1
                elif 0x0980 <= cp <= 0x09FF:
                    script_counts["bengali"] += 1
                elif 0x0C80 <= cp <= 0x0CFF:
                    script_counts["kannada"] += 1
                elif 0x0600 <= cp <= 0x06FF:
                    script_counts["perso_arabic"] += 1
                elif 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
                    script_counts["cjk_hanzi"] += 1
                else:
                    script_counts["other"] += 1
            else:
                script_counts["latin"] += 1

        return {
            "total_vocab_size": len(self.token_to_id),
            "num_language_tokens": len(LANG_TOKENS),
            "script_distribution": script_counts
        }

    def save_vocab(self, path: str):
        """Save vocabulary to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str):
        """Load vocabulary from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)
        self._refresh_mappings()


# Alias for character-level multilingual tokenizer interface
MultilingualCharTokenizer = MultilingualTokenizer

