"""
Training and Data Subsystem for Multilingual Non-Whisper ASR.
"""

from training.dataset import MultilingualASRDataset, collate_asr_batch
try:
    from training.manifest_generator import generate_dataset_manifest
except ImportError:
    generate_dataset_manifest = None

__all__ = [
    "MultilingualASRDataset",
    "collate_asr_batch",
    "generate_dataset_manifest",
]
