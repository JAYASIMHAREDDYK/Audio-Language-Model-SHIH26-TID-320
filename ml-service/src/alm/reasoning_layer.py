import os
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

class SceneReasoningLayer:
    """
    Scene Reasoning Layer using Phi-3 Instruction Prompting or Core ALM neural synthesis.
    """
    def __init__(self, load_llm: bool = False):
        self.tokenizer = None
        self.reasoning_model = None
        if load_llm and HAS_TRANSFORMERS:
            try:
                bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
                self.reasoning_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
                print(f"[SceneReasoningLayer] Successfully loaded {MODEL_NAME}")
            except Exception as e:
                print(f"[SceneReasoningLayer] LLM load note: {e}")

    def reason_about_scene(self, fused_data: dict, question: str = "What can be inferred from this audio?") -> str:
        transcript_summary = " | ".join(
            f'{seg.get("speaker", "S1")} ({seg.get("language","hi")}): "{seg.get("text","")}"'
            for seg in fused_data.get("transcript_segments", [])
            if isinstance(seg, dict) and seg.get("text")
        )
        bg_summary = ", ".join(fused_data.get("background_sounds", [])) or "none detected"

        if self.reasoning_model is not None and self.tokenizer is not None:
            try:
                prompt = f"""<|user|>
You are an audio scene understanding assistant. Given detected signals from an audio clip, reason about the situation like a human listening would.

Speech transcript: {transcript_summary or "no speech detected"}
Background sounds detected: {bg_summary}
Dominant emotion: {fused_data.get("emotion", "unknown")}
Background noise class: {fused_data.get("noise_class", "unknown")}

Question: {question}
Answer concisely in 2-3 sentences. Quote any numbers, flight codes, or names from the transcript exactly as given.<|end|>
<|assistant|>"""
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.reasoning_model.device)
                output = self.reasoning_model.generate(**inputs, max_new_tokens=80, do_sample=False)
                return self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            except Exception as e:
                print(f"[SceneReasoningLayer] Fallback reason note: {e}")

        # Deterministic dynamic synthesis fallback
        emotion = fused_data.get("emotion", "neutral")
        noise_cls = fused_data.get("noise_class", bg_summary)
        
        q_lower = question.lower()
        if "where" in q_lower or "location" in q_lower or "scene" in q_lower:
            return f"Based on acoustic environment analysis, the audio indicates a {noise_cls} setting with spoken transcript ('{transcript_summary or 'speech'}')."
        elif "emotion" in q_lower or "tone" in q_lower:
            return f"Vocal analysis indicates a dominant emotion of {emotion} with {noise_cls} acoustic context."
        else:
            return f"Audio analysis detected {noise_cls} alongside speech ('{transcript_summary or 'speech'}') exhibiting {emotion} emotion."

def reason_about_scene(fused_data: dict, question: str = "What can be inferred from this audio?") -> str:
    layer = SceneReasoningLayer(load_llm=False)
    return layer.reason_about_scene(fused_data, question)
