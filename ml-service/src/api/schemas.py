from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any

class SpeechResult(BaseModel):
    transcript: str = Field(default="", description="Decoded speech transcript text")
    language: str = Field(default="en", description="Detected language code (e.g. en, hi, te)")
    confidence: float = Field(default=0.90, ge=0.0, le=1.0, description="Speech recognition confidence")
    word_timestamps: List[Dict[str, Any]] = Field(default_factory=list, description="Word-level start/end timestamps")

class SpeakerSegment(BaseModel):
    speaker_id: str = Field(default="Speaker 1", description="Speaker identifier")
    start: float = Field(default=0.0, ge=0.0, description="Start timestamp in seconds")
    end: float = Field(default=0.0, ge=0.0, description="End timestamp in seconds")
    text: Optional[str] = Field(default=None, description="Speaker spoken text snippet")

class SoundEventItem(BaseModel):
    label: str = Field(default="event", description="Sound event class label")
    confidence: float = Field(default=0.90, ge=0.0, le=1.0, description="Event classification confidence")
    start: float = Field(default=0.0, ge=0.0, description="Event start timestamp in seconds")
    end: float = Field(default=5.0, ge=0.0, description="Event end timestamp in seconds")

class ParalinguisticResult(BaseModel):
    emotion: str = Field(default="neutral", description="Detected vocal emotion state")
    arousal: str = Field(default="low", description="Vocal arousal level (low/medium/high)")
    valence: str = Field(default="neutral", description="Vocal valence (positive/negative/neutral)")
    confidence: float = Field(default=0.90, ge=0.0, le=1.0, description="Paralinguistic emotion confidence")

class SceneResult(BaseModel):
    environment: str = Field(default="Acoustic Environment", description="Acoustic scene environment classification")
    confidence: float = Field(default=0.90, ge=0.0, le=1.0, description="Scene classification confidence")

class AnalyzeResponse(BaseModel):
    answer: str = Field(..., description="Generative natural language answer from Core ALM reasoning decoder")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Dynamic confidence score of the generated answer")
    speech: SpeechResult = Field(default_factory=SpeechResult, description="ASR speech recognition results")
    speakers: List[Union[SpeakerSegment, Dict[str, Any]]] = Field(default_factory=list, description="Speaker diarization segments")
    audio_events: List[Union[SoundEventItem, Dict[str, Any]]] = Field(default_factory=list, description="Detected non-speech sound events")
    paralinguistic: Union[ParalinguisticResult, Dict[str, Any]] = Field(default_factory=dict, description="Paralinguistic emotion & tone features")
    scene: Union[SceneResult, Dict[str, Any]] = Field(default_factory=dict, description="Acoustic environment classification")
    evidence: List[str] = Field(default_factory=list, description="Acoustic grounding evidence timestamps & attention peaks")
