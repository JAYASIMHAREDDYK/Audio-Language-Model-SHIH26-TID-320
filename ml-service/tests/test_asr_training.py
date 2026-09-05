import os
import tempfile
import pytest
import torch
from torch.utils.data import DataLoader

from src.asr.multilingual_asr import ASRModel
from src.asr.tokenizer import MultilingualTokenizer
from training.dataset import MultilingualASRDataset, collate_asr_batch
from training.train_asr import train


class TestASRTraining:
    """Test suite for training pipeline, loss computation, and checkpointing."""

    @pytest.fixture
    def setup_data(self):
        tokenizer = MultilingualTokenizer()
        dataset = MultilingualASRDataset(
            manifest_path="non_existent.json",
            tokenizer=tokenizer,
            max_samples=4,
            synthetic_fallback=True
        )
        return tokenizer, dataset

    def test_dataset_and_collate_fn(self, setup_data):
        tokenizer, dataset = setup_data
        assert len(dataset) > 0

        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_asr_batch)
        batch = next(iter(loader))

        assert "audio" in batch
        assert "targets" in batch
        assert "audio_lengths" in batch
        assert "target_lengths" in batch
        assert batch["audio"].ndim == 2
        assert batch["targets"].ndim == 2

    def test_forward_pass_and_loss_backward(self, setup_data):
        tokenizer, dataset = setup_data
        model = ASRModel(device="cpu")
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_asr_batch)
        batch = next(iter(loader))

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()

        outputs = model(
            audio=batch["audio"],
            targets=batch["targets"],
            input_lengths=None,
            target_lengths=batch["target_lengths"]
        )

        assert "loss" in outputs
        loss = outputs["loss"]
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

        loss.backward()
        optimizer.step()

    def test_checkpoint_save_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = ASRModel(device="cpu")
            ckpt_path = os.path.join(tmp_dir, "test_ckpt.pt")

            model.save_checkpoint(ckpt_path)
            assert os.path.exists(ckpt_path)

            new_model = ASRModel(device="cpu")
            new_model.load_checkpoint(ckpt_path)
            assert new_model.hidden_dim == model.hidden_dim
