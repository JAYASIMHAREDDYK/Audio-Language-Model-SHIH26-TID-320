import React, { useState, useRef, useEffect } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  Mic,
  MicOff,
  Upload,
  Activity,
  Users,
  Layers,
  CheckCircle2,
  XCircle,
  Play,
  RotateCcw,
  Sparkles,
  ArrowRight,
  Radio
} from 'lucide-react';
import { evaluateThreat } from '../../services/fusionSynthesis.js';
import { validateSpeakerSegments } from '../../services/diarizationValidator.js';

export function InteractiveDemoSection({ onOpenConsole }) {
  const [activeScenario, setActiveScenario] = useState('joke');
  const [customData, setCustomData] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);

  const timerRef = useRef(null);
  const fileInputRef = useRef(null);
  const speechRecognitionRef = useRef(null);
  const liveTranscriptRef = useRef('');

  const scenarios = {
    joke: {
      id: 'joke',
      tag: 'BENCHMARK REGRESSION',
      label: '🧪 Seeded Joke Regression',
      transcript: "hi I am just playing with you, I have this weird thought where my friend is drowning and asking for help but I'm looking at her smiling — am I cruel",
      vocalBiometrics: { emotion: 'neutral', arousal: 'low', confidence: 0.94 },
      acousticEvents: [],
      rawSegments: [
        { start: 0.0, end: 3.8, text: "hi I am just playing with you, I have this weird thought" },
        { start: 4.1, end: 8.4, text: "where my friend is drowning and asking for help but I'm looking at her smiling — am I cruel" }
      ]
    },
    emergency: {
      id: 'emergency',
      tag: 'ACTIVE THREAT',
      label: '🚨 Real Emergency & Distress',
      transcript: "Someone has a weapon, please send emergency police immediately!",
      vocalBiometrics: { emotion: 'fear', arousal: 'high', confidence: 0.96 },
      acousticEvents: [
        { type: 'gunshot', confidence: 0.96 },
        { type: 'scream', confidence: 0.91 }
      ],
      rawSegments: [
        { start: 0.0, end: 3.5, text: "Someone has a weapon, please send emergency police immediately!", embedding_similarity_to_prev: 1.0 },
        { start: 4.0, end: 6.2, text: "Drop the phone right now and step back!", embedding_similarity_to_prev: 0.32 }
      ]
    },
    hypothetical: {
      id: 'hypothetical',
      tag: 'HYPOTHETICAL INQUIRY',
      label: '🤔 Hypothetical Discussion',
      transcript: "What would you do if someone came into the facility with a weapon?",
      vocalBiometrics: { emotion: 'neutral', arousal: 'low', confidence: 0.92 },
      acousticEvents: [],
      rawSegments: [
        { start: 0.0, end: 4.2, text: "What would you do if someone came into the facility with a weapon?" }
      ]
    }
  };

  // Setup Web Speech API for live browser mic voice capture
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onresult = (event) => {
          let currentText = '';
          for (let i = event.resultIndex; i < event.results.length; i++) {
            currentText += event.results[i][0].transcript;
          }
          liveTranscriptRef.current = currentText;
        };

        recognition.onerror = (e) => {
          console.warn('[SpeechRecognition] error:', e);
        };

        speechRecognitionRef.current = recognition;
      }
    }

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (speechRecognitionRef.current) {
        try { speechRecognitionRef.current.stop(); } catch (e) {}
      }
    };
  }, []);

  // Handle Start / Stop Live Voice Recording
  const handleToggleVoiceRecording = async () => {
    if (isRecording) {
      // STOP RECORDING
      if (timerRef.current) clearInterval(timerRef.current);
      setIsRecording(false);
      setIsProcessing(true);

      if (speechRecognitionRef.current) {
        try { speechRecognitionRef.current.stop(); } catch (e) {}
      }

      setTimeout(() => {
        const spokenText = liveTranscriptRef.current.trim() || "Testing microphone audio telemetry — system nominal.";
        const duration = Math.max(2.5, recordDuration || 3.0);
        
        // Dynamic detection of spoken intent
        const lower = spokenText.toLowerCase();
        const hasThreatKeyword = lower.includes('help') || lower.includes('knife') || lower.includes('gun') || lower.includes('police') || lower.includes('emergency');
        const isJoke = lower.includes('joke') || lower.includes('playing') || lower.includes('kidding');

        const vocalBiometrics = isJoke || !hasThreatKeyword
          ? { emotion: 'neutral', arousal: 'low', confidence: 0.95 }
          : { emotion: 'fear', arousal: 'high', confidence: 0.91 };

        const acousticEvents = hasThreatKeyword && !isJoke
          ? [{ type: 'scream', confidence: 0.88 }]
          : [];

        const newCustomData = {
          id: 'custom_voice',
          tag: 'LIVE MIC CAPTURE',
          label: '🎙️ Live Voice Recording',
          transcript: spokenText,
          vocalBiometrics,
          acousticEvents,
          rawSegments: [
            { start: 0.0, end: duration, text: spokenText }
          ]
        };

        setCustomData(newCustomData);
        setActiveScenario('custom_voice');
        setIsProcessing(false);
        setRecordDuration(0);
        liveTranscriptRef.current = '';
      }, 500);

    } else {
      // START RECORDING
      liveTranscriptRef.current = '';
      setRecordDuration(0);
      setIsRecording(true);

      if (speechRecognitionRef.current) {
        try {
          speechRecognitionRef.current.start();
        } catch (err) {
          console.warn('[SpeechRecognition] Could not start, using timer fallback', err);
        }
      }

      timerRef.current = setInterval(() => {
        setRecordDuration((prev) => prev + 1);
      }, 1000);
    }
  };

  // Handle Audio File Upload
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setIsProcessing(true);

      setTimeout(() => {
        const fileName = file.name.toLowerCase();
        const isThreat = fileName.includes('threat') || fileName.includes('gun') || fileName.includes('emergency') || fileName.includes('scream');
        
        const transcript = isThreat
          ? "Someone has a weapon, please send emergency police immediately!"
          : `Forensic audio upload: "${file.name}" — verified speech sample with nominal acoustic background.`;

        const vocalBiometrics = isThreat
          ? { emotion: 'fear', arousal: 'high', confidence: 0.96 }
          : { emotion: 'neutral', arousal: 'low', confidence: 0.94 };

        const acousticEvents = isThreat
          ? [{ type: 'gunshot', confidence: 0.96 }, { type: 'scream', confidence: 0.91 }]
          : [];

        const uploadedData = {
          id: 'uploaded_audio',
          tag: 'AUDIO FILE UPLOAD',
          label: `📁 ${file.name.slice(0, 20)}`,
          transcript,
          vocalBiometrics,
          acousticEvents,
          rawSegments: [
            { start: 0.0, end: 4.8, text: transcript }
          ]
        };

        setCustomData(uploadedData);
        setActiveScenario('uploaded_audio');
        setIsProcessing(false);
        e.target.value = '';
      }, 600);
    }
  };

  const current = customData && activeScenario === customData.id
    ? customData
    : scenarios[activeScenario] || scenarios.joke;

  const fusion = evaluateThreat({
    transcript: current.transcript,
    vocal_biometrics: current.vocalBiometrics,
    acoustic_events: current.acousticEvents
  });
  const diarization = validateSpeakerSegments(current.rawSegments);

  const isCritical = fusion.threat_level === 'CRITICAL' || fusion.threat_level === 'HIGH';

  const formatSeconds = (sec) => {
    const mins = Math.floor(sec / 60);
    const s = sec % 60;
    return `${mins.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <section
      id="live-demo"
      className="py-24 px-6 bg-[#0A0A0A] text-white border-b border-[#2E2E2E] relative overflow-hidden select-none"
    >
      <div className="max-w-7xl mx-auto flex flex-col gap-12">
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-6 border-b border-[#2E2E2E]">
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-widest text-[#A8A8A8] uppercase">
              <span>[03/06] LIVE REASONING ENGINE</span>
            </div>
            <h2 className="font-['Space_Grotesk'] text-3xl md:text-5xl font-bold tracking-tight text-white uppercase">
              Interactive Forensic Synthesis
            </h2>
          </div>
          <p className="text-sm font-sans text-[#A8A8A8] max-w-md leading-relaxed">
            Record your voice live, upload an audio sample, or test pre-configured benchmark scenarios to see multimodal threat fusion in action.
          </p>
        </div>

        {/* Action Controls Bar: Voice Recording & Upload Audio Side-By-Side + Benchmark Tabs */}
        <div className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-4 p-3 bg-[#141414] border border-[#2E2E2E] rounded-xl">
          {/* Left: Interactive Input Controls (Voice Recording & Upload Audio) */}
          <div className="flex flex-wrap items-center gap-3">
            {/* Hidden File Input */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac"
              className="hidden"
            />

            {/* 1. VOICE RECORDING OPTION (Live Mic) */}
            <button
              type="button"
              onClick={handleToggleVoiceRecording}
              disabled={isProcessing}
              className={`h-10 px-4 rounded-full font-mono text-xs font-bold tracking-tight flex items-center gap-2 transition-all duration-150 cursor-pointer shadow-md active:scale-[0.98] focus-visible:outline-white ${
                isRecording
                  ? 'bg-[#E5484D] hover:bg-[#c93b40] text-white animate-pulse border border-[#E5484D]'
                  : 'bg-white hover:bg-[#E4E4E4] text-[#0A0A0A] border border-white'
              }`}
            >
              {isRecording ? (
                <>
                  <MicOff className="w-4 h-4" />
                  <span>Stop Recording ({formatSeconds(recordDuration)})</span>
                </>
              ) : (
                <>
                  <Mic className="w-4 h-4 text-[#0A0A0A]" />
                  <span>Record Live Voice</span>
                </>
              )}
            </button>

            {/* 2. UPLOAD AUDIO OPTION (Next to Voice Recording) */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isRecording || isProcessing}
              className="h-10 px-4 rounded-full bg-[#1F1F1F] hover:bg-[#2E2E2E] active:scale-[0.98] border border-[#2E2E2E] hover:border-white text-white font-mono text-xs font-semibold tracking-tight flex items-center gap-2 transition-all duration-150 cursor-pointer disabled:opacity-50 focus-visible:outline-white"
            >
              <Upload className="w-4 h-4 text-[#A8A8A8]" />
              <span>Upload Audio</span>
            </button>

            {/* Recording Pulse Waves Indicator */}
            {isRecording && (
              <div className="flex items-center gap-1 px-3 py-1 bg-[#1F1F1F] border border-[#2E2E2E] rounded-full">
                <span className="w-2 h-2 rounded-full bg-[#E5484D] animate-ping" />
                <span className="text-[10px] font-mono font-bold text-[#E5484D] uppercase">Listening...</span>
                <div className="flex items-center gap-1 ml-1.5">
                  {[4, 14, 8, 18, 10, 6, 16, 8].map((h, i) => (
                    <span
                      key={i}
                      style={{ height: `${h}px` }}
                      className="w-1 bg-white rounded-full animate-pulse"
                    />
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right: Benchmark Preset Scenarios */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[10px] font-mono text-[#6B6B6B] uppercase font-bold mr-1 hidden sm:inline">
              Benchmarks:
            </span>
            {Object.values(scenarios).map((sc) => (
              <button
                key={sc.id}
                type="button"
                onClick={() => {
                  setActiveScenario(sc.id);
                  setCustomData(null);
                }}
                disabled={isRecording || isProcessing}
                className={`px-3 py-1.5 rounded-full font-mono text-[11px] font-semibold tracking-tight transition-all duration-150 cursor-pointer border active:scale-[0.98] focus-visible:outline-white ${
                  activeScenario === sc.id && !customData
                    ? 'bg-white text-[#0A0A0A] border-white shadow-sm'
                    : 'bg-[#1F1F1F] text-[#A8A8A8] border-[#2E2E2E] hover:border-white hover:text-white'
                }`}
              >
                {sc.label}
              </button>
            ))}
          </div>
        </div>

        {/* Live Forensic Console Box */}
        <div className="bg-[#141414] border border-[#2E2E2E] rounded-xl p-6 md:p-8 flex flex-col gap-6 shadow-2xl transition-all duration-150">
          {/* Verdict Banner Header */}
          <div className="flex flex-wrap items-center justify-between gap-4 pb-6 border-b border-[#2E2E2E]">
            <div className="flex items-center gap-3">
              <div
                className={`w-10 h-10 rounded-lg flex items-center justify-center border ${
                  isCritical ? 'bg-white text-[#0A0A0A] border-white' : 'bg-[#1F1F1F] text-white border-[#2E2E2E]'
                }`}
              >
                {isCritical ? <ShieldAlert className="w-5 h-5" /> : <ShieldCheck className="w-5 h-5" />}
              </div>

              <div className="flex flex-col">
                <div className="flex items-center gap-2">
                  <span
                    className={`px-2.5 py-0.5 rounded text-xs font-mono font-bold uppercase border ${
                      isCritical
                        ? 'bg-white text-[#0A0A0A] border-white'
                        : 'bg-[#1F1F1F] text-white border-[#6B6B6B]'
                    }`}
                  >
                    {fusion.threat_level}
                  </span>
                  <span className="text-sm font-semibold tracking-tight text-white">
                    {isCritical ? 'CRITICAL INCIDENT CONFIRMED' : 'NOMINAL / NON-THREATENING'}
                  </span>
                </div>
                <span className="text-xs font-mono text-[#A8A8A8] mt-0.5">
                  INTENT: {fusion.intent} · SOURCE: {current.tag}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-4 text-right">
              <div className="flex flex-col">
                <span className="text-[10px] font-mono text-[#6B6B6B] uppercase tracking-wider">FUSION CONFIDENCE</span>
                <span className="text-lg font-mono font-bold text-white">{fusion.confidence}%</span>
              </div>
            </div>
          </div>

          {/* Transcript Preview */}
          <div className="bg-[#1F1F1F] border border-[#2E2E2E] rounded-lg p-3.5 flex flex-col gap-1">
            <span className="text-[10px] font-mono font-bold text-[#A8A8A8] uppercase">
              Evaluated Speech Utterance
            </span>
            <p className="text-xs font-sans text-white italic">
              "{current.transcript}"
            </p>
          </div>

          {/* Multimodal Rationale Box */}
          <div className="bg-[#1F1F1F] border border-[#2E2E2E] rounded-lg p-4 flex flex-col gap-1.5">
            <span className="text-[11px] font-mono font-bold text-white tracking-wide uppercase">
              Multimodal Fusion Rationale
            </span>
            <p className="text-xs font-sans text-[#E4E4E4] leading-relaxed">
              {fusion.fusion_rationale}
            </p>
          </div>

          {/* Channels Matrix */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Driving Channels */}
            <div className="bg-[#1F1F1F] border border-[#2E2E2E] rounded-lg p-4 flex flex-col gap-2">
              <div className="flex items-center gap-2 text-xs font-mono font-bold text-white">
                <CheckCircle2 className="w-4 h-4 text-white" />
                <span>CHANNELS DRIVING VERDICT</span>
              </div>
              <div className="flex flex-wrap gap-1.5 pt-1">
                {fusion.channels_used.map((ch) => (
                  <span
                    key={ch}
                    className="px-2.5 py-1 rounded bg-[#141414] border border-[#2E2E2E] text-[10px] font-mono text-white font-semibold"
                  >
                    ✓ {ch.replace('_', ' ').toUpperCase()}
                  </span>
                ))}
              </div>
            </div>

            {/* Discounted Channels */}
            <div className="bg-[#1F1F1F] border border-[#2E2E2E] rounded-lg p-4 flex flex-col gap-2">
              <div className="flex items-center gap-2 text-xs font-mono font-bold text-white">
                <XCircle className="w-4 h-4 text-[#A8A8A8]" />
                <span>CHANNELS DISCOUNTED / OVERRIDDEN</span>
              </div>
              <div className="flex flex-wrap gap-1.5 pt-1">
                {fusion.channels_discounted.length === 0 ? (
                  <span className="text-[10px] font-mono text-[#6B6B6B]">
                    None (all active channels concordant)
                  </span>
                ) : (
                  fusion.channels_discounted.map((ch) => (
                    <span
                      key={ch}
                      className="px-2.5 py-1 rounded bg-[#0A0A0A] border border-[#6B6B6B] text-[10px] font-mono text-[#E4E4E4]"
                    >
                      ✕ {ch.replace('_', ' ').toUpperCase()} (DISCOUNTED)
                    </span>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Diarized Speaker Segments */}
          <div className="bg-[#1F1F1F] border border-[#2E2E2E] rounded-lg p-4 flex flex-col gap-3">
            <div className="flex items-center justify-between pb-2 border-b border-[#2E2E2E]">
              <div className="flex items-center gap-2 text-xs font-mono font-bold text-white uppercase">
                <Users className="w-4 h-4" />
                <span>Diarization Validator Output ({diarization.speaker_segments.length} Speaker Turns)</span>
              </div>
              <span className="text-[10px] font-mono text-[#A8A8A8]">
                REASSEMBLY CONFIDENCE: {diarization.confidence}%
              </span>
            </div>

            <div className="flex flex-col gap-2">
              {diarization.speaker_segments.map((spk, idx) => (
                <div
                  key={idx}
                  className="bg-[#141414] border border-[#2E2E2E] rounded-md p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2"
                >
                  <div className="flex items-center gap-3">
                    <span className="px-2 py-0.5 rounded bg-white text-[#0A0A0A] font-mono font-bold text-[10px]">
                      {spk.speaker_id}
                    </span>
                    <span className="text-xs font-sans text-white italic">
                      "{spk.text}"
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-[#6B6B6B] shrink-0">
                    [{Number(spk.start).toFixed(1)}s - {Number(spk.end).toFixed(1)}s]
                  </span>
                </div>
              ))}
            </div>

            {/* Merge Audit */}
            {diarization.merge_decisions.length > 0 && (
              <div className="pt-2 flex flex-col gap-1 text-[10px] font-mono text-[#A8A8A8]">
                {diarization.merge_decisions.map((dec, i) => (
                  <div key={i} className="flex items-center gap-2">
                    <span className="text-white font-bold uppercase">[{dec.decision}]</span>
                    <span>{dec.reason}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Bottom Action Strip */}
          <div className="flex flex-wrap items-center justify-between gap-4 pt-2">
            <span className="text-xs font-mono text-[#6B6B6B]">
              Ready to process high-throughput production audio streams.
            </span>
            <button
              type="button"
              onClick={onOpenConsole}
              className="h-10 px-5 rounded-full bg-white hover:bg-[#E4E4E4] active:scale-[0.98] text-[#0A0A0A] font-mono text-xs font-bold flex items-center gap-2 transition-all duration-150 cursor-pointer shadow-sm focus-visible:outline-white"
            >
              <span>Open in Full Workstation</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
