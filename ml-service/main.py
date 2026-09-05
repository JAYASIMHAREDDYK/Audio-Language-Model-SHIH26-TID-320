import os
import sys
import tempfile
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# Ensure repository root and ml-service are on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.alm.inference import ALMInferencePipeline
from src.api.schemas import AnalyzeResponse

pipeline_instance: Optional[ALMInferencePipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance
    ckpt_path = os.path.join(REPO_ROOT, "checkpoints", "core_alm_latest.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(BASE_DIR, "checkpoints", "core_alm_latest.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = None

    print(f"[ML Service] Initializing Core ALM Pipeline ONCE on startup. Checkpoint: {ckpt_path}")
    try:
        pipeline_instance = ALMInferencePipeline(checkpoint_path=ckpt_path)
        print("[ML Service] Pipeline initialized successfully and ready for inference.")
    except Exception as e:
        print(f"[ML Service] Error initializing pipeline: {e}")
        pipeline_instance = None
    yield
    print("[ML Service] Shutting down Core ALM Service.")

app = FastAPI(
    title="Smart Horizon 2026 - Core ALM ML Service",
    description="Core Audio Language Model & Multimodal Temporal Fusion Engine API",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
frontend_origin_env = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173,http://localhost:3000,http://localhost:4000,http://127.0.0.1:5173,http://127.0.0.1:3000,http://127.0.0.1:4000")
origins = [o.strip() for o in frontend_origin_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB max audio upload limit

# Lazy-loaded ASR model for /predict endpoint
_asr_model_instance = None

def _get_asr_model():
    global _asr_model_instance
    if _asr_model_instance is None:
        try:
            from src.asr.multilingual_asr import ASRModel
            _asr_model_instance = ASRModel()
            _asr_model_instance.eval()
            print("[ML Service] Deep Conformer ASR model loaded for /predict endpoint.")
        except Exception as e:
            print(f"[ML Service] ASR model load error: {e}")
    return _asr_model_instance

@app.get("/health")
async def health_check():
    import torch
    alm_loaded = pipeline_instance is not None and getattr(pipeline_instance, "model", None) is not None
    asr_loaded = _asr_model_instance is not None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "status": "healthy" if alm_loaded else "degraded",
        "service": "Core Audio Language Model (Core ALM)",
        "model_loaded": alm_loaded,
        "asr_loaded": asr_loaded,
        "device": device,
        "gpu": gpu_name,
        "team": "SH-DST-02 (Team ID: SHIH26-TID-320)",
        "architecture": {
            "asr": "12-layer Deep Multilingual Conformer (80.24M params)",
            "alm": "Core ALM Temporal Fusion (embed_dim=256)",
            "languages": ["hi", "te", "ta", "bn", "mr", "kn", "zh"],
            "vocab_size": 2122
        }
    }

@app.post("/predict")
async def predict(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form("hi")
):
    """
    POST /predict
    Returns Deep Conformer ASR features and experimental transcription.
    Clearly separates acoustic representation (B,T',256) from CTC transcription.
    """
    import torch

    temp_audio_path = None
    try:
        contents = await audio_file.read()
        if len(contents) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(status_code=413, detail="Audio file exceeds 50 MB limit.")
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file.")

        ext = os.path.splitext(audio_file.filename)[1].lower() if audio_file.filename else ".wav"
        if not ext or len(ext) > 6:
            ext = ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(contents)
            temp_audio_path = tmp.name

        asr = _get_asr_model()
        if asr is None:
            raise HTTPException(status_code=503, detail="ASR model not available.")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Get Conformer acoustic representations (B, T', 512)
        with torch.no_grad():
            conformer_features = asr.encode(temp_audio_path, pool=False)  # (B, T', 512)

        # 2. Get ALM-projected features (B, T', 256) via ASRAdapter
        try:
            from src.asr.asr_model import ASRAdapter
            adapter = ASRAdapter(embed_dim=256).to(conformer_features.device)
            adapter.eval()
            with torch.no_grad():
                alm_features = adapter.encode(temp_audio_path, pool=False)  # (B, T', 256)
            alm_shape = list(alm_features.shape)
        except Exception:
            alm_shape = None

        # 3. Experimental CTC transcription
        try:
            result = asr.transcribe(temp_audio_path, language=language)
            transcript = result.get("text", "")
        except Exception:
            transcript = ""

        return {
            "status": "success",
            "language": language,
            "transcript": transcript if transcript else "(experimental — full-data CTC not converged)",
            "transcription_status": "experimental",
            "conformer_features_shape": list(conformer_features.shape),
            "alm_features_shape": alm_shape,
            "model": "12-layer Deep Multilingual Conformer (80.24M params)",
            "device": device,
            "note": "Full-data CTC training did not converge within hackathon budget. "
                    "Conformer acoustic representations (B,T',512) → ASRAdapter → (B,T',256) are functional."
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ML Service] /predict error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(
    audio_file: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
    question: Optional[str] = Form(""),
    language_hint: Optional[str] = Form("auto"),
    spoken_transcript: Optional[str] = Form(None),
    llm_model: Optional[str] = Form("gpt-4o-mini")
):
    """
    POST /analyze
    Accepts multipart/form-data with binary audio buffer or audio file and text question.
    Runs canonical Core ALM inference pipeline and returns validated AnalyzeResponse JSON schema.
    """
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Core ALM model service is not initialized or unavailable.")

    # Sanitize and default question
    final_question = question.strip() if question and question.strip() else ""
    if len(final_question) > 500:
        raise HTTPException(status_code=400, detail="Question string exceeds maximum allowed length (500 characters).")

    temp_audio_path = None
    try:
        audio_source = None

        if audio_file is not None:
            contents = await audio_file.read()
            if len(contents) > MAX_FILE_SIZE_BYTES:
                raise HTTPException(status_code=413, detail="Uploaded audio file exceeds maximum limit of 50 MB.")

            if len(contents) > 0:
                ext = os.path.splitext(audio_file.filename)[1].lower() if (audio_file and audio_file.filename) else ".webm"
                if not ext or len(ext) > 6:
                    ext = ".webm"
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(contents)
                    temp_audio_path = tmp.name
                audio_source = temp_audio_path

        print(f"[ML Service] Processing POST /analyze. Question: '{final_question}', Audio file: {audio_file.filename if audio_file else 'None'}, Spoken: '{spoken_transcript}', LLM: '{llm_model}'")

        # Run canonical Core ALM inference
        analysis_result = pipeline_instance.analyze(
            audio_source=audio_source,
            question=final_question,
            language_hint=language_hint or "auto",
            spoken_transcript=spoken_transcript,
            llm_model=llm_model or "gpt-4o-mini"
        )

        return analysis_result

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ML Service] Exception during /analyze processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Core ALM Inference Error: {str(e)}")
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception as cleanup_err:
                print(f"[ML Service] Temporary file cleanup warning: {cleanup_err}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

