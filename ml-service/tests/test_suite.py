"""
Comprehensive Post-Refactor Verification Test Suite for Core ALM-NHCE.
Verifies all 9 mandatory architectural checkpoints before declaring consolidation complete.
"""
import os, sys, unittest, torch

ml_service_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ml_service_dir not in sys.path:
    sys.path.insert(0, ml_service_dir)

from src.alm.alm_model import CoreALM
from src.alm.reasoning_model import ALMTokenizer, CoreALMReasoningModel
from src.alm.inference import ALMInferencePipeline
from src.audio.spectrogram import LogMelSpectrogramExtractor

class TestCoreALMCodebase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.tokenizer = ALMTokenizer(vocab_size=5000)

    def test_01_imports(self):
        """Verify clean import of all canonical Core ALM modules."""
        from src.asr.asr_model import ASRModel
        from src.events.event_model import SoundEventModel
        from src.speaker.speaker_model import SpeakerModel
        from src.paralinguistic.paralinguistic_model import ParalinguisticModel
        from src.fusion.temporal_alignment import TemporalAligner
        from src.fusion.multimodal_projection import MultimodalProjector
        from src.fusion.temporal_fusion import TemporalFusionAdapter
        self.assertTrue(True)

    def test_02_model_construction(self):
        """Verify CoreALM model construction and LazyLinear initialization."""
        model = CoreALM(embed_dim=256, vocab_size=5000).to(self.device)
        dummy_audio = torch.randn(1, 80, 128, device=self.device)
        with torch.no_grad():
            out = model(dummy_audio, question_text="What sound event is present?")
        self.assertIn("answer_logits", out)
        self.assertIn("joint_context_tensor", out)
        self.assertEqual(list(out["joint_context_tensor"].shape), [1, 80, 256])

    def test_03_tokenizer_roundtrip(self):
        """Verify tokenizer encode/decode roundtrip."""
        text = "aircraft sound is present in the recording."
        tokens = self.tokenizer.encode(text, max_len=16, device=self.device)
        decoded = self.tokenizer.decode(tokens)
        self.assertIn("aircraft", decoded.lower())
        self.assertIn("present", decoded.lower())

    def test_04_audio_preprocessing(self):
        """Verify LogMelSpectrogramExtractor tensor output shape."""
        extractor = LogMelSpectrogramExtractor(sample_rate=16000, hop_length=625).to(self.device)
        signal = torch.randn(1, 16000, device=self.device)
        mel = extractor(signal)
        self.assertEqual(mel.ndim, 3)
        self.assertEqual(mel.size(1), 80)

    def test_05_specialized_models_smoke(self):
        """Verify specialized encoder feature extraction shapes."""
        model = CoreALM(embed_dim=256).to(self.device)
        audio = torch.randn(1, 80, 128, device=self.device)
        with torch.no_grad():
            speech_emb = model.asr_model.encode(audio)
            event_emb = model.event_model.encode(audio)
            speaker_emb = model.speaker_model.encode(audio)
            para_emb = model.para_model.encode(audio)
        self.assertEqual(speech_emb.shape[-1], 256)
        self.assertEqual(event_emb.shape[-1], 256)
        self.assertEqual(speaker_emb.shape[-1], 256)
        self.assertEqual(para_emb.shape[-1], 256)

    def test_06_fusion_tensor_shapes(self):
        """Verify spatio-temporal fusion tensor shape and alignment."""
        model = CoreALM(embed_dim=256).to(self.device)
        audio = torch.randn(2, 80, 128, device=self.device) # Batch size 2
        with torch.no_grad():
            fused, proj_dict = model.encode_multimodal(audio)
        self.assertEqual(list(fused.shape), [2, 64, 256])

    def test_07_decoder_target_shift(self):
        """Verify autoregressive target shift behavior (position 0 predicts target content, NOT BOS)."""
        tgt = self.tokenizer.encode("aircraft", max_len=6, device=self.device).unsqueeze(0)
        tgt_in = tgt[:, :-1]
        tgt_lbl = tgt[:, 1:]
        self.assertNotEqual(tgt_lbl[0, 0].item(), self.tokenizer.bos_id)
        self.assertEqual(self.tokenizer.id2word.get(tgt_lbl[0, 0].item()), "aircraft")

    def test_08_inference_pipeline_smoke(self):
        """Verify ALMInferencePipeline end-to-end JSON response schema."""
        ckpt_path = os.path.join(ml_service_dir, "checkpoints", "core_alm_latest.pt")
        pipeline = ALMInferencePipeline(checkpoint_path=ckpt_path if os.path.exists(ckpt_path) else None)
        dummy_mel = torch.randn(1, 80, 128)
        res = pipeline.analyze(dummy_mel, question="What sound event is present?")
        self.assertIn("answer", res)
        self.assertIn("confidence", res)
        self.assertIn("speech", res)
        self.assertIn("audio_events", res)
        self.assertIn("speakers", res)
        self.assertIn("paralinguistic", res)

if __name__ == "__main__":
    unittest.main()
