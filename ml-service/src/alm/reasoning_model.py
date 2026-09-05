import torch
import torch.nn as nn
import torch.nn.functional as F

class ALMTokenizer:
    """
    Bi-directional 5000-token natural language vocabulary tokenizer for Core ALM.
    Maps discrete subword/word strings <-> integer token IDs.
    """
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        
        base_words = [
            "<pad>", "<bos>", "<eos>", "<unk>",
            ".", ",", "?", "!", ":", ";", "-", "&", "(", ")", "\"", "'",
            "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "there", "this", "that", "it", "in", "on", "at", "to", "from", "with",
            "by", "for", "of", "and", "or", "but", "not", "no", "yes", "detected",
            "present", "active", "occurring", "co-occurring", "simultaneous",
            "simultaneously", "speech", "speaker", "speakers", "speaker's", "speaker(s)",
            "one", "two", "three", "four", "five", "sound", "sounds", "sounding", "acoustic",
            "event", "events", "noise", "non-speech", "level", "background", "environment",
            "scene", "context", "recording", "aircraft", "helicopter", "car", "bus",
            "train", "vehicle", "car_horn", "siren", "sirens", "alarm", "warning",
            "emergency", "dog", "crowd", "crowded", "footsteps", "engine", "machinery",
            "music", "rain", "thunder", "airport", "terminal", "station", "market",
            "office", "discussion", "hospital", "ward", "factory", "industrial",
            "street", "traffic", "neutral", "happy", "sad", "angry", "fearful",
            "surprised", "disgusted", "emotion", "emotional", "vocal", "tone",
            "voices", "voice", "cues", "distress", "change", "changed", "shift",
            "transition", "transitions", "immediately", "before", "after", "during",
            "while", "evaluation", "indicates", "suggests", "observed", "recorded",
            "confidence", "high", "moderate", "low", "quiet", "busy", "loudest",
            "dominant", "primary", "secondary", "evidence", "grounding", "timestamp",
            "segment", "activity", "number", "count", "turn", "state", "situation",
            "consistent", "inferred", "together", "combined", "location", "place",
            "person", "people", "talking", "speaking", "announcement", "room", "area",
            "building", "outside", "inside", "indoor", "outdoor", "clear", "faint",
            "strong", "distorted", "intermittent", "continuous", "steady", "sudden",
            "abrupt", "gradual", "rising", "falling", "pitch", "energy", "volume",
            "intense", "calm", "urgent", "agitated", "relaxed", "tense", "panic",
            "shout", "whisper", "cry", "laughter", "gasp", "sigh", "cough", "applause",
            "cheering", "screaming", "whistle", "bells", "chimes", "explosion", "bang",
            "clatter", "thud", "click", "beep", "ring", "hum", "buzz", "hiss", "rattle",
            "rumble", "splash", "drip", "wind", "storm", "birds", "chirping", "barking",
            "howling", "meow", "roar", "squeak", "creak", "slam", "knock", "impact",
            "collision", "brakes", "screech", "honk", "siren_alarm", "public_address",
            "pa_system", "broadcast", "call", "conversation", "dialogue", "monologue",
            "overlap", "interruption", "pause", "silence", "quiet_background", "ambient",
            "reverb", "echo", "clean", "noisy", "distorted_audio", "compressed",
            "construction", "stop", "zone", "full", "happened", "happening", "heard",
            "identified", "identify", "including", "infer", "interacting", "likely",
            "occurred", "occurs", "overall", "playing", "type", "would", "you",
            "begins", "did", "how", "what", "where", "any", "around", "audible", "audio",
            "describe", "detect", "distinct", "expressed"
        ]

        self.word2id = {}
        self.id2word = {}

        for idx, w in enumerate(base_words):
            self.word2id[w] = idx
            self.id2word[idx] = w

        for i in range(len(base_words), vocab_size):
            token_str = f"tok_{i}"
            self.word2id[token_str] = i
            self.id2word[i] = token_str

        self.next_dynamic_id = len(base_words)

    def _tokenize_text(self, text) -> list:
        if isinstance(text, (list, tuple)):
            text = " ".join([str(t) for t in text])
        t = str(text).lower()
        for p in [".", ",", "?", "!", ":", ";", "(", ")", "\"", "&"]:
            t = t.replace(p, f" {p} ")
        return t.split()

    def encode(self, text, max_len=64, device=None) -> torch.Tensor:
        words = self._tokenize_text(text)
        ids = [self.bos_id]
        for w in words:
            if w in self.word2id:
                ids.append(self.word2id[w])
            else:
                if self.next_dynamic_id < self.vocab_size:
                    tid = self.next_dynamic_id
                    self.word2id[w] = tid
                    self.id2word[tid] = w
                    self.next_dynamic_id += 1
                    ids.append(tid)
                else:
                    ids.append(self.unk_id)
        ids.append(self.eos_id)
        
        if len(ids) < max_len:
            ids += [self.pad_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len-1] + [self.eos_id]

        tensor = torch.tensor(ids, dtype=torch.long)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def decode(self, token_ids) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        words = []
        for tid in token_ids:
            if tid in [self.pad_id, self.bos_id, self.unk_id]:
                continue
            if tid == self.eos_id:
                break
            w = self.id2word.get(tid, "")
            if w and not w.startswith("tok_"):
                if not words or words[-1] != w:
                    words.append(w)

        if not words:
            return "No response generated"

        out = ""
        for w in words:
            if w in [".", ",", "?", "!", ":", ";"]:
                out = out.rstrip() + w + " "
            else:
                out += w + " "

        out = out.strip()
        if not out:
            return "No response generated"
            
        return out[0].upper() + out[1:]


class CoreALMReasoningModel(nn.Module):
    """
    Core ALM Question-Conditioned Multimodal Reasoning Engine with Autoregressive Decoder.
    """
    def __init__(self, embed_dim=256, vocab_size=5000, num_layers=3, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.tokenizer = ALMTokenizer(vocab_size=vocab_size)

        # Autoregressive Token Embedding & Positional Encoding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 128, embed_dim))
        nn.init.normal_(self.pos_encoder, std=0.02)

        # Deep Multimodal Autoregressive Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.reasoning_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 1. Output Head
        self.answer_head = nn.Linear(embed_dim, vocab_size)

        # 2. Evidence Grounding Head
        self.evidence_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # 3. Confidence Head
        self.confidence_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 4. Auxiliary Task-Specific Supervision (Modality Reconstruction Heads)
        self.recon_speech = nn.Linear(embed_dim, embed_dim)
        self.recon_events = nn.Linear(embed_dim, embed_dim)
        self.recon_speakers = nn.Linear(embed_dim, embed_dim)
        self.recon_para = nn.Linear(embed_dim, embed_dim)

    def decode_answer_from_logits(self, answer_logits: torch.Tensor) -> str:
        """
        Decodes natural language answer directly from model's predicted logits.
        """
        probs = F.softmax(answer_logits[0], dim=-1)
        pred_token_ids = torch.argmax(probs, dim=-1).cpu().tolist()
        return self.tokenizer.decode(pred_token_ids)

    def forward(self, joint_context: torch.Tensor, tgt_tokens: torch.Tensor = None) -> dict:
        """
        Args:
            joint_context: (B, T_ctx, embed_dim)
            tgt_tokens: Optional target token ID sequence (B, L_tgt) for teacher forcing during training
        """
        B = joint_context.size(0)
        device = joint_context.device

        if tgt_tokens is not None:
            # Teacher forcing mode during training
            L_tgt = tgt_tokens.size(1)
            tgt_embed = self.token_embedding(tgt_tokens) + self.pos_encoder[:, :L_tgt, :]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(L_tgt, device=device)
            pad_mask = (tgt_tokens == 0)

            reasoned = self.reasoning_decoder(
                tgt=tgt_embed,
                memory=joint_context,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=pad_mask
            )
            answer_logits = self.answer_head(reasoned)
        else:
            # Direct feature evaluation mode
            L_ctx = joint_context.size(1)
            tgt_embed = joint_context + self.pos_encoder[:, :L_ctx, :]
            reasoned = self.reasoning_decoder(tgt=tgt_embed, memory=joint_context)
            answer_logits = self.answer_head(reasoned)

        pooled_rep = torch.mean(reasoned, dim=1)
        evidence_scores = self.evidence_head(joint_context).squeeze(-1)
        
        # Dynamic confidence combining logit certainty and confidence projection
        probs = F.softmax(answer_logits, dim=-1)
        max_token_probs = torch.max(probs, dim=-1)[0]
        mean_token_cert = torch.mean(max_token_probs, dim=-1, keepdim=True)
        pred_conf = self.confidence_head(pooled_rep)
        dynamic_conf = 0.5 * pred_conf + 0.5 * mean_token_cert

        # Auxiliary Reconstructions
        recon_speech = self.recon_speech(joint_context)
        recon_events = self.recon_events(joint_context)
        recon_speakers = self.recon_speakers(joint_context)
        recon_para = self.recon_para(joint_context)

        return {
            "answer_logits": answer_logits,
            "pooled_rep": pooled_rep,
            "evidence_scores": evidence_scores,
            "confidence": dynamic_conf,
            "recon_speech": recon_speech,
            "recon_events": recon_events,
            "recon_speakers": recon_speakers,
            "recon_para": recon_para
        }

    def generate_answer(self, joint_context: torch.Tensor, max_len=32) -> tuple:
        """
        Autoregressive sequence decoding loop for inference.
        Returns: (decoded_text, full_answer_logits, calibrated_confidence)
        """
        device = joint_context.device
        B = joint_context.size(0)
        
        generated_ids = torch.full((B, 1), self.tokenizer.bos_id, dtype=torch.long, device=device)
        logits_list = []

        for step in range(max_len):
            L_seq = generated_ids.size(1)
            tgt_embed = self.token_embedding(generated_ids) + self.pos_encoder[:, :L_seq, :]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(L_seq, device=device)

            dec_out = self.reasoning_decoder(
                tgt=tgt_embed,
                memory=joint_context,
                tgt_mask=causal_mask
            )
            step_logits = self.answer_head(dec_out[:, -1:, :]) # (B, 1, vocab_size)

            # Suppress pad (0), bos (1), unk (3) at generation step
            step_logits[:, :, 0] = -1e9
            step_logits[:, :, 1] = -1e9
            step_logits[:, :, 3] = -1e9

            # Repetition penalty to prevent word loops
            for b in range(B):
                for prev_id in generated_ids[b].tolist():
                    if prev_id not in [0, 1, 2]: # Don't penalize pad, bos, eos
                        step_logits[b, 0, prev_id] -= 3.0

            logits_list.append(step_logits)

            next_token = torch.argmax(step_logits, dim=-1) # (B, 1)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if (next_token == self.tokenizer.eos_id).all():
                break

        full_logits = torch.cat(logits_list, dim=1)
        decoded_tokens = generated_ids[0].cpu().tolist()
        text = self.tokenizer.decode(decoded_tokens)

        # Dynamic confidence calculation
        probs = F.softmax(full_logits, dim=-1)
        mean_token_cert = torch.mean(torch.max(probs, dim=-1)[0])
        pooled_rep = torch.mean(joint_context, dim=1)
        pred_conf = self.confidence_head(pooled_rep)[0, 0]
        confidence = 0.5 * pred_conf + 0.5 * mean_token_cert

        return text, full_logits, confidence
