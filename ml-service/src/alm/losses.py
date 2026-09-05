import torch
import torch.nn as nn
import torch.nn.functional as F

class ALMTaskLoss(nn.Module):
    """
    Core ALM Multi-Task Loss Function.
    Combines:
    1. Language Generation / Answer Cross-Entropy Loss
    2. Evidence Temporal Grounding Binary Cross-Entropy Loss
    3. Calibrated Confidence Mean Squared Error Loss
    """
    def __init__(self, answer_weight=1.0, evidence_weight=0.5, confidence_weight=0.3, recon_weight=0.5):
        super().__init__()
        self.answer_weight = answer_weight
        self.evidence_weight = evidence_weight
        self.confidence_weight = confidence_weight
        self.recon_weight = recon_weight

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Args:
            predictions:
                - answer_logits: (B, Seq_len, Vocab_size)
                - evidence_scores: (B, Seq_len)
                - confidence: (B, 1)
            targets:
                - answer_ids: (B, Target_len)
                - evidence_targets: (B, Target_len)
                - target_confidence: (B, 1)
        Returns:
            loss_dict containing total_loss and components
        """
        answer_logits = predictions["answer_logits"]
        evidence_scores = predictions["evidence_scores"]
        confidence = predictions["confidence"]

        target_answer_ids = targets.get("answer_ids", None)
        target_evidence = targets.get("evidence_targets", None)
        target_conf = targets.get("target_confidence", torch.ones_like(confidence))

        # 1. Answer Loss (with sequence length alignment)
        if target_answer_ids is not None:
            B, L_pred, V = answer_logits.shape
            L_tgt = target_answer_ids.shape[1]
            if L_tgt < L_pred:
                padding = torch.zeros((B, L_pred - L_tgt), dtype=target_answer_ids.dtype, device=target_answer_ids.device)
                aligned_target_ids = torch.cat([target_answer_ids, padding], dim=1)
            else:
                aligned_target_ids = target_answer_ids[:, :L_pred]
            
            l_ans = self.ce_loss(answer_logits.reshape(-1, V), aligned_target_ids.reshape(-1))
        else:
            l_ans = torch.tensor(0.0, device=answer_logits.device)

        # 2. Evidence Grounding Loss (with sequence length alignment)
        if target_evidence is not None:
            L_pred = evidence_scores.size(1)
            L_tgt = target_evidence.size(1)
            if L_tgt < L_pred:
                padding = torch.zeros((evidence_scores.size(0), L_pred - L_tgt), dtype=target_evidence.dtype, device=target_evidence.device)
                aligned_evidence = torch.cat([target_evidence, padding], dim=1)
            else:
                aligned_evidence = target_evidence[:, :L_pred]
            # Cast to float32 for AMP safety (BCELoss is unsafe with fp16)
            l_ev = self.bce_loss(evidence_scores.float(), aligned_evidence.float())
        else:
            l_ev = torch.tensor(0.0, device=evidence_scores.device)

        # 3. Confidence Calibration Loss
        l_conf = self.mse_loss(confidence, target_conf)

        # 4. Modality Reconstruction Loss
        if "target_speech" in targets:
            T_audio = targets["target_speech"].size(1)
            pred_speech = predictions["recon_speech"][:, -T_audio:, :]
            pred_events = predictions["recon_events"][:, -T_audio:, :]
            pred_speakers = predictions["recon_speakers"][:, -T_audio:, :]
            pred_para = predictions["recon_para"][:, -T_audio:, :]

            l_recon = (
                self.mse_loss(pred_speech, targets["target_speech"]) +
                self.mse_loss(pred_events, targets["target_events"]) +
                self.mse_loss(pred_speakers, targets["target_speakers"]) +
                self.mse_loss(pred_para, targets["target_para"])
            ) / 4.0
        else:
            l_recon = torch.tensor(0.0, device=confidence.device)

        total_loss = (self.answer_weight * l_ans) + (self.evidence_weight * l_ev) + (self.confidence_weight * l_conf) + (self.recon_weight * l_recon)

        return {
            "loss": total_loss,
            "answer_loss": l_ans.item() if isinstance(l_ans, torch.Tensor) else l_ans,
            "evidence_loss": l_ev.item() if isinstance(l_ev, torch.Tensor) else l_ev,
            "confidence_loss": l_conf.item() if isinstance(l_conf, torch.Tensor) else l_conf,
            "recon_loss": l_recon.item() if isinstance(l_recon, torch.Tensor) else l_recon
        }
