import React from 'react';
import { VoiceAIOrbVisualizer } from './VoiceAIOrbVisualizer';
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  FileText,
  Users,
  Mic,
  Activity,
  Layers,
  CheckCircle2,
  Clock,
  Info,
  Sparkles,
  Volume2,
  HelpCircle,
  Radio
} from 'lucide-react';

export function VerdictPanel({ activeSession, isLoading, isCapturing, realtimeTelemetry = [] }) {
  if (isCapturing) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 md:p-8 bg-[#0d0d0d] text-[#ececec]">
        <div className="flex flex-col items-center gap-4 animate-in fade-in duration-200">
          <VoiceAIOrbVisualizer telemetry={realtimeTelemetry} isCapturing={true} />
          <div className="flex flex-col items-center gap-1.5 text-center">
            <span className="text-sm font-medium text-white tracking-wide">
              Listening for audio input...
            </span>
            <span className="text-xs text-[#8e8ea0]">
              Real-time Web Audio FFT telemetry active
            </span>
          </div>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 md:p-8 bg-[#0d0d0d] text-[#ececec]">
        <div className="flex flex-col items-center gap-4 animate-in fade-in duration-200">
          <VoiceAIOrbVisualizer telemetry={realtimeTelemetry} isCapturing={false} />
          <div className="flex flex-col items-center gap-1.5 text-center">
            <span className="text-sm font-medium text-white tracking-wide animate-pulse">
              Running Core ALM Multimodal Reasoning Pass...
            </span>
            <span className="text-xs text-[#8e8ea0]">
              Fusing ASR, Sound Events, Speakers, & Paralinguistics
            </span>
          </div>
        </div>
      </div>
    );
  }

  if (!activeSession || !activeSession.result) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-6 md:p-12 bg-[#0d0d0d] text-[#ececec] overflow-y-auto select-none">
        <div className="max-w-3xl w-full flex flex-col items-center gap-8 animate-in fade-in duration-300 my-auto">
          <div className="flex flex-col items-center text-center gap-3">
            <h1 className="text-2xl md:text-3xl font-medium tracking-tight text-[#ececec]">
              Smart Horizon — Core ALM Engine
            </h1>
            <p className="text-sm md:text-base text-[#8e8ea0] max-w-xl leading-relaxed italic">
              "Multimodal Audio Language Model for continuous acoustic perception, speaker diarization, vocal affect, and natural language question answering."
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full pt-2">
            <div className="bg-[#181818] border border-[#262626] rounded-xl p-5 flex flex-col gap-2.5">
              <div className="flex items-center gap-2 text-white">
                <Mic className="w-5 h-5 text-[#ececec]" />
                <h3 className="text-sm font-medium">Multilingual Speech & ASR</h3>
              </div>
              <p className="text-xs text-[#8e8ea0] leading-relaxed">
                Extracts speech latent representations and transcriptions across multiple languages using the Lahari encoder.
              </p>
            </div>

            <div className="bg-[#181818] border border-[#262626] rounded-xl p-5 flex flex-col gap-2.5">
              <div className="flex items-center gap-2 text-white">
                <Volume2 className="w-5 h-5 text-[#ececec]" />
                <h3 className="text-sm font-medium">Sound Event Perception</h3>
              </div>
              <p className="text-xs text-[#8e8ea0] leading-relaxed">
                Perceives non-speech acoustic events such as sirens, barks, bus noise, crowd chatter, and rain.
              </p>
            </div>

            <div className="bg-[#181818] border border-[#262626] rounded-xl p-5 flex flex-col gap-2.5">
              <div className="flex items-center gap-2 text-white">
                <Users className="w-5 h-5 text-[#ececec]" />
                <h3 className="text-sm font-medium">Speaker Diarization & Count</h3>
              </div>
              <p className="text-xs text-[#8e8ea0] leading-relaxed">
                Identifies active speaker count, turn-taking boundaries, and individual speaker representations.
              </p>
            </div>

            <div className="bg-[#181818] border border-[#262626] rounded-xl p-5 flex flex-col gap-2.5">
              <div className="flex items-center gap-2 text-white">
                <Sparkles className="w-5 h-5 text-[#ececec]" />
                <h3 className="text-sm font-medium">Core ALM Generative Decoder</h3>
              </div>
              <p className="text-xs text-[#8e8ea0] leading-relaxed">
                Autoregressively decodes precise natural language answers over a 5,000-token acoustic domain vocabulary.
              </p>
            </div>
          </div>

          <span className="text-xs text-[#8e8ea0] font-mono pt-2">
            Click the <strong className="text-white">Microphone</strong> or <strong className="text-white">Upload</strong> icon below to ask a question
          </span>
        </div>
      </div>
    );
  }

  const result = activeSession.result;
  const capture = activeSession.capture || {};

  const answerText = result.answer || "No response generated.";
  const confidence = result.confidence !== undefined ? result.confidence : 0.85;
  const confPct = Math.round(confidence * 100);
  const evidenceList = result.evidence || [];

  const speechData = result.speech || {};
  const transcript = speechData.transcript || capture.transcript || "";
  const language = speechData.language || "en";

  const eventsList = Array.isArray(result.audio_events) ? result.audio_events : (result.audio_events?.events || []);
  const speakersList = Array.isArray(result.speakers) ? result.speakers : (result.speakers?.diarization || []);
  const paralinguistic = result.paralinguistic || {};
  const sceneData = result.scene || {};

  return (
    <div className="flex-1 overflow-y-auto px-6 md:px-12 py-8 flex flex-col gap-6 w-full max-w-4xl mx-auto select-text bg-[#0d0d0d] text-[#ececec]">
      {/* 1. USER QUESTION CARD */}
      <div className="flex items-start justify-end gap-3 animate-in fade-in duration-150">
        <div className="bg-[#181818] border border-[#262626] text-[#ececec] rounded-2xl px-4 py-3 max-w-xl text-sm leading-relaxed shadow-xs">
          <div className="text-[11px] text-[#8e8ea0] mb-1 font-medium flex items-center gap-1.5">
            <HelpCircle className="w-3.5 h-3.5 text-[#8e8ea0]" />
            <span>Question</span>
          </div>
          <div>"{capture.question || 'Where is the speaker likely to be?'}"</div>
        </div>
        <div className="w-8 h-8 rounded-full bg-[#262626] flex items-center justify-center text-white text-xs font-semibold shrink-0">
          User
        </div>
      </div>

      {/* 2. CORE ALM ANSWER CARD */}
      <div className="flex items-start gap-4 pt-1 animate-in fade-in duration-200">
        <div className="w-8 h-8 rounded-full bg-white text-black flex items-center justify-center text-xs font-bold shrink-0 shadow-sm">
          ALM
        </div>

        <div className="flex-1 flex flex-col gap-6 text-sm leading-relaxed text-[#ececec] bg-[#181818] border border-[#262626] rounded-xl p-6 shadow-sm">
          
          {/* Main Answer Header */}
          <div className="flex items-center justify-between border-b border-[#262626] pb-4">
            <div className="flex flex-col gap-1">
              <span className="text-[11px] font-mono uppercase tracking-wider text-emerald-400 font-bold flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5" />
                CORE ALM REASONING
              </span>
              <h2 className="text-xl font-bold text-white tracking-tight">
                {answerText}
              </h2>
            </div>

            <div className="flex items-center gap-1.5 bg-[#222222] border border-[#333333] px-3.5 py-1.5 rounded-lg text-xs font-mono shrink-0">
              <span className="text-[#8e8ea0]">Confidence:</span>
              <span className="text-emerald-400 font-bold text-sm">{confPct}%</span>
            </div>
          </div>

          {/* Structured Evidence Decomposition */}
          <div className="flex flex-col gap-3">
            <h4 className="text-xs uppercase tracking-wider font-semibold text-[#8e8ea0] flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-[#8e8ea0]" />
              Reasoning Evidence & Modality Decomposition
            </h4>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {/* Speech Evidence */}
              <div className="bg-[#1e1e1e] border border-[#262626] rounded-xl p-3.5 flex flex-col gap-1.5">
                <span className="text-xs font-semibold text-white flex items-center gap-1.5">
                  <Mic className="w-3.5 h-3.5 text-emerald-400" />
                  Speech Evidence
                </span>
                <p className="text-xs text-[#cccccc] font-mono">
                  {result.speech_evidence || `Transcript: "${transcript}" | Language: ${language}`}
                </p>
              </div>

              {/* Non-Speech Evidence */}
              <div className="bg-[#1e1e1e] border border-[#262626] rounded-xl p-3.5 flex flex-col gap-1.5">
                <span className="text-xs font-semibold text-white flex items-center gap-1.5">
                  <Volume2 className="w-3.5 h-3.5 text-cyan-400" />
                  Non-Speech Evidence
                </span>
                <p className="text-xs text-[#cccccc] font-mono">
                  {result.non_speech_evidence || `Events: ${eventsList.map(e => typeof e === 'string' ? e : e.label).join(', ')}`}
                </p>
              </div>

              {/* Speaker Evidence */}
              <div className="bg-[#1e1e1e] border border-[#262626] rounded-xl p-3.5 flex flex-col gap-1.5">
                <span className="text-xs font-semibold text-white flex items-center gap-1.5">
                  <Users className="w-3.5 h-3.5 text-purple-400" />
                  Speaker Evidence
                </span>
                <p className="text-xs text-[#cccccc] font-mono">
                  {result.speaker_evidence || `${speakersList.length || 1} speaker turn(s) active`}
                </p>
              </div>

              {/* Paralinguistic Evidence */}
              <div className="bg-[#1e1e1e] border border-[#262626] rounded-xl p-3.5 flex flex-col gap-1.5">
                <span className="text-xs font-semibold text-white flex items-center gap-1.5">
                  <Activity className="w-3.5 h-3.5 text-amber-400" />
                  Paralinguistic Evidence
                </span>
                <p className="text-xs text-[#cccccc] font-mono">
                  {result.paralinguistic_evidence || `Emotion: ${paralinguistic.emotion || 'neutral'} | Arousal: ${paralinguistic.arousal || 'low'}`}
                </p>
              </div>
            </div>

            {/* Joint & Temporal Inference Evidence */}
            {result.reasoning_evidence && (
              <div className="bg-[#1c241f] border border-emerald-500/20 rounded-xl p-3.5 flex flex-col gap-1">
                <span className="text-xs font-semibold text-emerald-400 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5" />
                  Integrated Multimodal Inference
                </span>
                <p className="text-xs text-[#d1d5db]">
                  {result.reasoning_evidence}
                </p>
              </div>
            )}
          </div>

          {/* Model Grounding Evidence Section */}
          {evidenceList.length > 0 && (
            <div className="flex flex-col gap-2.5 pt-2 border-t border-[#262626]">
              <h4 className="text-xs uppercase tracking-wider font-semibold text-[#8e8ea0] flex items-center gap-1.5">
                <Clock className="w-3.5 h-3.5" />
                Acoustic Attention & Grounding Evidence
              </h4>
              <div className="flex flex-col gap-2">
                {evidenceList.map((item, idx) => (
                  <div key={idx} className="bg-[#1e1e1e] border border-[#292929] rounded-lg p-2.5 px-3 text-xs text-[#cccccc] flex items-center gap-2">
                    <Clock className="w-3.5 h-3.5 text-[#8e8ea0] shrink-0" />
                    <span>{item}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Acoustic Modalities Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
            
            {/* Speech & Transcript Card */}
            <div className="bg-[#1e1e1e] border border-[#262626] rounded-xl p-4 flex flex-col gap-2">
              <div className="flex items-center justify-between text-xs text-[#8e8ea0] font-medium">
                <span className="flex items-center gap-1.5 text-white">
                  <Mic className="w-4 h-4 text-[#ececec]" />
                  ASR Transcript
                </span>
                <span className="uppercase text-[10px] bg-emerald-500/10 border border-emerald-500/30 px-2.5 py-0.5 rounded-full text-emerald-400 font-semibold tracking-wide flex items-center gap-1.5 shadow-sm">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  Lang: {language?.toUpperCase()} ({(() => {
                    const l = (language || 'en').toLowerCase();
                    const map = { hi: 'Hindi', te: 'Telugu', ta: 'Tamil', bn: 'Bengali', mr: 'Marathi', kn: 'Kannada', zh: 'Mandarin', en: 'English' };
                    return map[l] || l.toUpperCase();
                  })()})
                </span>
              </div>
              <p className="text-xs text-[#cccccc] italic">
                {transcript ? `"${transcript}"` : "No speech transcript decoded."}
              </p>
            </div>

            {/* Paralinguistic / Emotion Card */}
            <div className="bg-[#1e1e1e] border border-[#262626] rounded-xl p-4 flex flex-col gap-2">
              <div className="flex items-center justify-between text-xs text-[#8e8ea0] font-medium">
                <span className="flex items-center gap-1.5 text-white">
                  <Activity className="w-4 h-4 text-[#ececec]" />
                  Vocal Affect / Paralinguistics
                </span>
              </div>
              <div className="text-xs text-[#cccccc] flex items-center justify-between pt-1">
                <span>Emotion: <strong className="text-white capitalize">{paralinguistic.emotion || 'neutral'}</strong></span>
                <span>Arousal: <strong className="text-white capitalize">{paralinguistic.arousal || 'low'}</strong></span>
              </div>
            </div>

            {/* Sound Events Card */}
            <div className="bg-[#1e1e1e] border border-[#262626] rounded-xl p-4 flex flex-col gap-2">
              <div className="flex items-center justify-between text-xs text-[#8e8ea0] font-medium">
                <span className="flex items-center gap-1.5 text-white">
                  <Volume2 className="w-4 h-4 text-[#ececec]" />
                  Sound Events
                </span>
                <span className="text-[10px] text-[#8e8ea0]">
                  {eventsList.length} detected
                </span>
              </div>
              {eventsList.length > 0 ? (
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {eventsList.map((ev, idx) => {
                    const label = typeof ev === 'string' ? ev : (ev.label || ev.event || 'event');
                    return (
                      <span key={idx} className="bg-[#262626] border border-[#333333] px-2 py-0.5 rounded text-[11px] text-white">
                        {label}
                      </span>
                    );
                  })}
                </div>
              ) : (
                <span className="text-xs text-[#8e8ea0] italic">No discrete sound events detected.</span>
              )}
            </div>

            {/* Scene Environment Card */}
            <div className="bg-[#1e1e1e] border border-[#262626] rounded-xl p-4 flex flex-col gap-2">
              <div className="flex items-center justify-between text-xs text-[#8e8ea0] font-medium">
                <span className="flex items-center gap-1.5 text-white">
                  <Radio className="w-4 h-4 text-[#ececec]" />
                  Acoustic Scene
                </span>
              </div>
              <div className="text-xs text-[#cccccc] pt-1">
                Environment: <strong className="text-white capitalize">{sceneData.environment || 'Ambient Context'}</strong>
              </div>
            </div>

          </div>

          {/* Speaker Diarization Segments */}
          {speakersList.length > 0 && (
            <div className="flex flex-col gap-2.5 pt-2 border-t border-[#262626]">
              <h4 className="text-xs uppercase tracking-wider font-semibold text-[#8e8ea0]">
                Speaker Diarization ({speakersList.length} turns)
              </h4>
              <div className="flex flex-col gap-2">
                {speakersList.map((spk, idx) => {
                  const spkId = typeof spk === 'object' ? (spk.speaker_id || spk.speaker || `Speaker ${idx+1}`) : `Speaker ${idx+1}`;
                  const st = typeof spk === 'object' ? (spk.start ?? 0.0) : 0.0;
                  const en = typeof spk === 'object' ? (spk.end ?? 5.0) : 5.0;
                  const txt = typeof spk === 'object' ? spk.text : String(spk);
                  return (
                    <div key={idx} className="bg-[#1e1e1e] border-l-4 border-[#565869] rounded-r-xl p-3 text-xs leading-relaxed">
                      <span className="font-medium text-[#ececec] block mb-0.5">
                        {spkId} [{Number(st).toFixed(1)}s - {Number(en).toFixed(1)}s]
                      </span>
                      {txt && <p className="italic text-[#8e8ea0]">"{txt}"</p>}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
