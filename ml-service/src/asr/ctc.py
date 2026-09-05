"""
Multilingual Connectionist Temporal Classification (CTC) Projection & Decoding Head.
Maps frame-level Conformer acoustic representations (B, T, 512) to token probabilities over
the multilingual vocabulary.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.asr.tokenizer import MultilingualTokenizer, BLANK_TOKEN


class CTCDecoder(nn.Module):
    """
    Multilingual CTC Projection and Decoding Head.
    
    Tensor Flow:
        Frame Embeddings: (B, T_frames, d_model=512)
        Linear Projection: -> (B, T_frames, Vocab_Size)
        LogSoftmax: -> (B, T_frames, Vocab_Size)
    """

    def __init__(
        self,
        d_model: int = 512,
        vocab_size: int = 500,
        dropout: float = 0.1,
        blank_id: int = 0,
        hidden_dim: Optional[int] = None
    ):
        super().__init__()
        if hidden_dim is not None:
            d_model = hidden_dim
        self.d_model = d_model
        self.hidden_dim = d_model
        self.vocab_size = vocab_size
        self.blank_id = blank_id

        self.dropout = nn.Dropout(dropout)
        self.lm_head = nn.Linear(d_model, vocab_size)

        # PyTorch Connectionist Temporal Classification (CTC) Loss
        self.ctc_loss_fn = nn.CTCLoss(
            blank=self.blank_id,
            reduction="mean",
            zero_infinity=True
        )

        assert self.blank_id == 0, f"blank_id must be 0, got {self.blank_id}"
        assert self.ctc_loss_fn.blank == 0, f"ctc_loss_fn blank must be 0, got {self.ctc_loss_fn.blank}"
        assert self.lm_head.out_features == self.vocab_size, (
            f"lm_head out_features ({self.lm_head.out_features}) != vocab_size ({self.vocab_size})"
        )

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute logits over vocabulary.
        Args:
            frame_embeddings: (B, T_frames, d_model=512)
        Returns:
            logits: (B, T_frames, Vocab_Size)
        """
        x = self.dropout(frame_embeddings)
        logits = self.lm_head(x)
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CTC loss using PyTorch CTCLoss.
        Args:
            logits: (B, T, Vocab_Size)
            targets: (B, L_target) or 1D unpadded concatenated tensor
            input_lengths: (B,) sequence frame lengths
            target_lengths: (B,) target label lengths
        """
        # PyTorch CTCLoss expects log_probs shape: (T_frames, Batch, Vocab_Size)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        return self.ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

    def decode_greedy(
        self,
        logits: torch.Tensor,
        tokenizer: MultilingualTokenizer,
        input_lengths: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        """
        Greedy CTC Decoding over sequence frames:
        1. argmax over vocabulary logits
        2. collapse repeating identical consecutive tokens
        3. filter out blank tokens (ID 0)
        4. compute frame confidence scores and word alignment timestamps
        """
        probs = F.softmax(logits, dim=-1)             # (B, T, Vocab_Size)
        max_probs, preds = torch.max(probs, dim=-1)   # (B, T), (B, T)

        batch_results = []
        batch_size = logits.size(0)

        for b in range(batch_size):
            t_len = int(input_lengths[b].item()) if input_lengths is not None else logits.size(1)
            seq_preds = preds[b, :t_len].tolist()
            seq_probs = max_probs[b, :t_len].tolist()

            collapsed_ids: List[int] = []
            collapsed_probs: List[float] = []
            word_timestamps: List[Dict[str, Any]] = []
            
            prev_token = None
            duration_per_frame = 5.0 / max(1, t_len)

            for t_idx, (token_id, p) in enumerate(zip(seq_preds, seq_probs)):
                if token_id != prev_token:
                    if token_id != self.blank_id and token_id != tokenizer.pad_id:
                        collapsed_ids.append(token_id)
                        collapsed_probs.append(p)
                        token_str = tokenizer.id_to_token.get(token_id, "")
                        if token_str and token_str not in ("<pad>", "<unk>", "<blank>") and not token_str.startswith("<lang:"):
                            word_timestamps.append({
                                "word": token_str,
                                "start": round(t_idx * duration_per_frame, 2),
                                "end": round((t_idx + 1) * duration_per_frame, 2),
                                "confidence": round(float(p), 4)
                            })
                    prev_token = token_id

            decoded_text = tokenizer.decode(collapsed_ids, skip_special_tokens=True)
            avg_conf = float(sum(collapsed_probs) / max(1, len(collapsed_probs))) if collapsed_probs else 0.0

            batch_results.append({
                "text": decoded_text,
                "token_ids": collapsed_ids,
                "confidence": round(max(0.40, min(0.99, avg_conf)), 4),
                "word_timestamps": word_timestamps
            })

        return batch_results
