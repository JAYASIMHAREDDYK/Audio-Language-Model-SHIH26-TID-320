import torch
import torch.nn as nn
import torch.nn.functional as F
from src.alm.reasoning_model import ALMTokenizer

class ModalityProjector(nn.Module):
    """
    Core ALM Modality Projector with Tokenizer Question Embedding.
    """
    def __init__(self, fused_dim=256, question_dim=256, model_dim=256, vocab_size=5000):
        super().__init__()
        self.vocab_size = vocab_size
        self.tokenizer = ALMTokenizer(vocab_size=vocab_size)

        self.multimodal_proj = nn.Sequential(
            nn.Linear(fused_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU()
        )
        self.question_proj = nn.Sequential(
            nn.Linear(question_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU()
        )
        self.text_embedding = nn.Embedding(self.vocab_size, question_dim)

    def embed_question_text(self, question_text, device=None) -> torch.Tensor:
        """
        Encodes question text using ALMTokenizer -> embedding lookup.
        Supports single string or list of strings.
        """
        if isinstance(question_text, (list, tuple)):
            ids_list = [self.tokenizer.encode(q, max_len=16, device=device) for q in question_text]
            ids_tensor = torch.stack(ids_list, dim=0)
        else:
            ids_tensor = self.tokenizer.encode(question_text, max_len=16, device=device).unsqueeze(0)
        embeds = self.text_embedding(ids_tensor)
        return embeds

    def forward(self, fused_tensor: torch.Tensor, question_tensor: torch.Tensor) -> tuple:
        proj_fused = self.multimodal_proj(fused_tensor)
        proj_question = self.question_proj(question_tensor)
        return proj_fused, proj_question
