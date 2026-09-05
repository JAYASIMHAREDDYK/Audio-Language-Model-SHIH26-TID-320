import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Upload, Sparkles, AlertCircle, Play, Volume2, AlertTriangle, Send, ChevronUp, Bot, Globe } from 'lucide-react';
import { calculatePitchFromBuffer } from '../services/audioAnalyzer';

const SUPPORTED_LANGUAGES = [
  { code: 'auto', label: 'Auto Detect', flag: '🌐' },
  { code: 'hi', label: 'Hindi', locale: 'hi-IN', flag: '🇮🇳' },
  { code: 'te', label: 'Telugu', locale: 'te-IN', flag: '🇮🇳' },
  { code: 'ta', label: 'Tamil', locale: 'ta-IN', flag: '🇮🇳' },
  { code: 'bn', label: 'Bengali', locale: 'bn-IN', flag: '🇮🇳' },
  { code: 'mr', label: 'Marathi', locale: 'mr-IN', flag: '🇮🇳' },
  { code: 'kn', label: 'Kannada', locale: 'kn-IN', flag: '🇮🇳' },
  { code: 'zh', label: 'Mandarin', locale: 'zh-CN', flag: '🇨🇳' },
  { code: 'en', label: 'English', locale: 'en-US', flag: '🇬🇧' },
];

/**
 * ComposerBar Component
 * Provides microphone recording, file upload, and natural language question input.
 * Connects directly to the Core ALM ML Service API.
 */
export function ComposerBar({
  onLiveCaptureStart,
  onLiveCaptureStop,
  onAudioUploaded,
  onAskQuestion,
  onLoadScenario,
  onAudioTelemetryUpdate,
  isCapturing,
  isLoading
}) {
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [liveTranscript, setLiveTranscript] = useState('');
  const [micVolume, setMicVolume] = useState([]);
  const [errorMessage, setErrorMessage] = useState(null);
  const [measuredPitch, setMeasuredPitch] = useState(null);
  const [measuredRms, setMeasuredRms] = useState(null);
  const [questionText, setQuestionText] = useState('');
  const [llmModel, setLlmModel] = useState('gpt-4o-mini');
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState('auto');
  const [showLangDropdown, setShowLangDropdown] = useState(false);

  const timerRef = useRef(null);
  const fileInputRef = useRef(null);
  const recognitionRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animFrameRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const mediaChunksRef = useRef([]);
  const liveTranscriptAccumulatorRef = useRef('');

  const formatSeconds = (sec) => {
    if (!Number.isFinite(sec) || isNaN(sec) || sec < 0) return '00:00';
    const mins = Math.floor(sec / 60);
    const secs = Math.floor(sec % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Example PS-aligned question suggestions
  const questionSuggestions = [
    "What can be inferred from speech and background sounds together?",
    "What sound event is present?",
    "How many speakers are present?",
    "What is the emotion of the speaker?",
    "Where is the speaker likely to be?"
  ];

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onresult = (event) => {
          let fullText = '';
          for (let i = 0; i < event.results.length; i++) {
            fullText += event.results[i][0].transcript;
          }
          liveTranscriptAccumulatorRef.current = fullText;
          setLiveTranscript(fullText);
        };

        recognition.onerror = (event) => {
          console.warn('[ComposerBar SpeechRecognition] error:', event.error);
          if (event.error === 'not-allowed') {
            setErrorMessage('Microphone permission denied. Please enable microphone access in browser settings.');
          }
        };

        recognitionRef.current = recognition;
      }
    }

    return () => {
      stopAudioCaptureResources();
    };
  }, []);

  const stopAudioCaptureResources = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);

    if (recognitionRef.current) {
      try { recognitionRef.current.stop(); } catch (e) {}
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try { mediaRecorderRef.current.stop(); } catch (e) {}
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      try { audioContextRef.current.close(); } catch (e) {}
      audioContextRef.current = null;
    }
  };

  const handleToggleCapture = async () => {
    setErrorMessage(null);

    if (isCapturing) {
      const finalRecordedText = liveTranscriptAccumulatorRef.current.trim() || liveTranscript.trim();
      const duration = Math.max(1.5, recordSeconds || 3.0);
      const pitch = measuredPitch;
      const rms = measuredRms;

      // Safely stop MediaRecorder and wait for onstop event to ensure all audio chunks are captured
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        await new Promise((resolve) => {
          mediaRecorderRef.current.onstop = () => resolve();
          try {
            mediaRecorderRef.current.stop();
          } catch (e) {
            resolve();
          }
        });
      }

      const mimeType = mediaRecorderRef.current?.mimeType || 'audio/webm';
      const recordedBlob = mediaChunksRef.current.length > 0
        ? new Blob(mediaChunksRef.current, { type: mimeType })
        : null;

      stopAudioCaptureResources();
      setRecordSeconds(0);
      setLiveTranscript('');
      liveTranscriptAccumulatorRef.current = '';
      if (onAudioTelemetryUpdate) onAudioTelemetryUpdate([]);

      const defaultText = finalRecordedText || '';
      onLiveCaptureStop(defaultText, duration, { measuredPitch: pitch, measuredRms: rms }, recordedBlob, questionText, llmModel, selectedLanguage);

    } else {
      setRecordSeconds(0);
      setLiveTranscript('');
      liveTranscriptAccumulatorRef.current = '';
      mediaChunksRef.current = [];

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setErrorMessage('Microphone audio capture is not supported in this browser.');
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
        });
        mediaStreamRef.current = stream;

        try {
          const mediaRecorder = new MediaRecorder(stream);
          mediaRecorder.ondataavailable = (e) => {
            if (e.data && e.data.size > 0) {
              mediaChunksRef.current.push(e.data);
            }
          };
          mediaRecorder.start(100);
          mediaRecorderRef.current = mediaRecorder;
        } catch (e) {
          console.warn('[ComposerBar] MediaRecorder init warning:', e);
        }

        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (AudioContext) {
          const audioCtx = new AudioContext();
          audioContextRef.current = audioCtx;
          const analyser = audioCtx.createAnalyser();
          analyser.fftSize = 1024;
          analyserRef.current = analyser;

          const source = audioCtx.createMediaStreamSource(stream);
          source.connect(analyser);

          const freqArray = new Uint8Array(analyser.frequencyBinCount);
          const timeArray = new Float32Array(analyser.fftSize);

          const updateLiveMeter = () => {
            if (!analyserRef.current) return;
            analyserRef.current.getByteFrequencyData(freqArray);
            analyserRef.current.getFloatTimeDomainData(timeArray);

            let sum = 0;
            for (let i = 0; i < timeArray.length; i++) {
              sum += timeArray[i] * timeArray[i];
            }
            const currentRms = Math.sqrt(sum / timeArray.length);
            setMeasuredRms(currentRms);

            const detectedPitch = calculatePitchFromBuffer(timeArray, audioCtx.sampleRate);
            if (detectedPitch) {
              setMeasuredPitch(detectedPitch);
            }

            const numBars = 48;
            const step = Math.max(1, Math.floor((analyser.frequencyBinCount * 0.7) / numBars));
            const realSpectrogramBars = [];

            for (let b = 0; b < numBars; b++) {
              const rawVal = freqArray[b * step] || 0;
              const pct = Math.max(5, Math.min(98, Math.floor((rawVal / 255) * 100)));
              realSpectrogramBars.push(pct);
            }

            setMicVolume(realSpectrogramBars);
            if (onAudioTelemetryUpdate) {
              onAudioTelemetryUpdate(realSpectrogramBars);
            }

            animFrameRef.current = requestAnimationFrame(updateLiveMeter);
          };
          updateLiveMeter();
        }

        onLiveCaptureStart();

        if (recognitionRef.current) {
          const matchLang = SUPPORTED_LANGUAGES.find(l => l.code === selectedLanguage);
          const targetLocale = matchLang?.locale || (selectedLanguage === 'auto' ? 'en-US' : 'en-US');
          try {
            recognitionRef.current.lang = targetLocale;
          } catch (e) {}
          try { recognitionRef.current.start(); } catch (err) {}
        }

        timerRef.current = setInterval(() => {
          setRecordSeconds((s) => s + 1);
        }, 1000);

      } catch (err) {
        console.error('[ComposerBar] Microphone permission error:', err);
        setErrorMessage('Microphone access was denied or no input device found. Please allow microphone permissions.');
      }
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onAudioUploaded(file, questionText, llmModel, selectedLanguage);
      e.target.value = '';
    }
  };

  return (
    <div className="w-full bg-[#121212] border-t border-[#262626] p-4 shrink-0 flex flex-col gap-3 items-center justify-center select-none">
      {/* Error Banner */}
      {errorMessage && (
        <div className="bg-[#1e1e1e] border border-red-500/30 rounded-xl p-3 px-4 flex items-center justify-between gap-3 text-xs text-[#f87171] shadow-sm w-full max-w-3xl">
          <div className="flex items-center gap-2.5">
            <AlertTriangle className="w-4 h-4 shrink-0" />
            <span>{errorMessage}</span>
          </div>
          <button
            type="button"
            onClick={() => setErrorMessage(null)}
            className="text-[11px] font-medium underline cursor-pointer text-[#8e8ea0] hover:text-white"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Live Voice Transcription & Live Metrics */}
      {isCapturing && (
        <div className="bg-[#1e1e1e] border border-red-500/40 rounded-xl p-3 flex flex-col gap-2 w-full max-w-3xl animate-in fade-in duration-150 shadow-lg">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3 min-w-0 flex-1">
              <div className="w-3 h-3 rounded-full bg-red-500 animate-ping shrink-0" />
              <div className="flex items-center gap-2 min-w-0 flex-1">
                <span className="text-xs font-bold text-red-400 shrink-0">RECORDING [{formatSeconds(recordSeconds)}]:</span>
                <span className="text-xs text-[#ececec] italic truncate">
                  {liveTranscript ? `"${liveTranscript}"` : 'Listening for spoken voice...'}
                </span>
              </div>
            </div>

            {/* Prominent Stop Recording Button inside Banner */}
            <button
              type="button"
              onClick={handleToggleCapture}
              className="px-4 py-1.5 rounded-lg bg-red-600 hover:bg-red-700 active:scale-95 text-white font-bold text-xs flex items-center gap-1.5 shadow-md transition-all cursor-pointer shrink-0"
            >
              <MicOff className="w-4 h-4" />
              <span>STOP RECORDING</span>
            </button>
          </div>

          <div className="flex items-center gap-4 pl-6 pt-1 border-t border-[#333] mt-1">
            <div className="flex items-center gap-1.5">
              <Volume2 className="w-3.5 h-3.5 text-emerald-500" />
              <span className="text-[11px] text-[#a1a1aa] font-mono">
                {measuredRms > 0 ? (20 * Math.log10(measuredRms)).toFixed(1) : '-∞'} dB
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-blue-500" />
              <span className="text-[11px] text-[#a1a1aa] font-mono">
                {measuredPitch ? `${measuredPitch.toFixed(1)} Hz` : '-- Hz'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Natural Language Question Input Field */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (questionText && onAskQuestion) {
            onAskQuestion(questionText, llmModel);
          }
        }}
        className="w-full max-w-3xl flex flex-col gap-2"
      >
        <div className="flex items-center gap-2 bg-[#181818] border border-[#262626] rounded-xl px-4 py-2 focus-within:border-[#404040] transition-colors relative">
          <span className="text-xs text-[#8e8ea0] font-mono shrink-0">Question:</span>
          <input
            type="text"
            value={questionText}
            onChange={(e) => setQuestionText(e.target.value)}
            placeholder="Ask Core ALM a question about the audio..."
            disabled={isLoading || isCapturing}
            className="flex-1 bg-transparent text-xs text-white placeholder-[#525252] focus:outline-none"
          />
          
          {/* AI Model Selector */}
          <div className="relative">
            <button
              type="button"
              onClick={() => { setShowModelDropdown(!showModelDropdown); setShowLangDropdown(false); }}
              disabled={isLoading || isCapturing}
              className="flex items-center gap-1.5 px-2 py-1.5 rounded-lg bg-[#262626] hover:bg-[#333] text-xs text-[#d1d5db] font-medium transition-colors border border-[#404040] disabled:opacity-50 cursor-pointer"
            >
              <Bot className="w-3.5 h-3.5 text-blue-400" />
              <span>{llmModel === 'gpt-4o-mini' ? 'GPT-4o-Mini' : 'GPT-3.5-Turbo'}</span>
              <ChevronUp className="w-3 h-3 text-[#8e8ea0]" />
            </button>
            
            {showModelDropdown && (
              <div className="absolute bottom-full mb-2 right-0 w-36 bg-[#1e1e1e] border border-[#333] rounded-xl shadow-xl overflow-hidden z-50">
                <button
                  type="button"
                  onClick={() => { setLlmModel('gpt-4o-mini'); setShowModelDropdown(false); }}
                  className="w-full text-left px-3 py-2 text-xs text-[#d1d5db] hover:bg-[#262626] hover:text-white transition-colors cursor-pointer"
                >
                  GPT-4o-Mini
                </button>
                <button
                  type="button"
                  onClick={() => { setLlmModel('gpt-3.5-turbo'); setShowModelDropdown(false); }}
                  className="w-full text-left px-3 py-2 text-xs text-[#d1d5db] hover:bg-[#262626] hover:text-white transition-colors cursor-pointer"
                >
                  GPT-3.5-Turbo
                </button>
              </div>
            )}
          </div>

          {/* Language Selector */}
          <div className="relative">
            <button
              type="button"
              onClick={() => { setShowLangDropdown(!showLangDropdown); setShowModelDropdown(false); }}
              disabled={isLoading || isCapturing}
              className="flex items-center gap-1.5 px-2 py-1.5 rounded-lg bg-[#262626] hover:bg-[#333] text-xs text-[#d1d5db] font-medium transition-colors border border-[#404040] disabled:opacity-50 cursor-pointer"
              title="Select Audio / Speech Language"
            >
              <Globe className="w-3.5 h-3.5 text-emerald-400" />
              <span>{SUPPORTED_LANGUAGES.find(l => l.code === selectedLanguage)?.label || 'Auto'}</span>
              <ChevronUp className="w-3 h-3 text-[#8e8ea0]" />
            </button>
            
            {showLangDropdown && (
              <div className="absolute bottom-full mb-2 right-0 w-44 bg-[#1e1e1e] border border-[#333] rounded-xl shadow-xl overflow-hidden z-50 max-h-64 overflow-y-auto">
                <div className="px-3 py-1.5 text-[10px] text-[#8e8ea0] uppercase tracking-wider font-semibold border-b border-[#2a2a2a] bg-[#161616]">
                  Speech Language
                </div>
                {SUPPORTED_LANGUAGES.map((lang) => (
                  <button
                    key={lang.code}
                    type="button"
                    onClick={() => {
                      setSelectedLanguage(lang.code);
                      setShowLangDropdown(false);
                      if (recognitionRef.current && lang.locale) {
                        try { recognitionRef.current.lang = lang.locale; } catch(e) {}
                      }
                    }}
                    className={`w-full text-left px-3 py-2 text-xs flex items-center justify-between transition-colors cursor-pointer ${
                      selectedLanguage === lang.code
                        ? 'bg-emerald-500/10 text-emerald-400 font-medium'
                        : 'text-[#d1d5db] hover:bg-[#262626] hover:text-white'
                    }`}
                  >
                    <span className="flex items-center gap-1.5">
                      <span>{lang.flag}</span>
                      <span>{lang.label}</span>
                    </span>
                    <span className="text-[10px] text-[#737373] uppercase font-mono">{lang.code}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          <button
            type="submit"
            disabled={isLoading || isCapturing || !questionText.trim()}
            className="p-1.5 rounded-lg bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 disabled:opacity-30 disabled:hover:bg-emerald-500/20 cursor-pointer transition-colors"
            title="Ask Core ALM"
          >
            <Send className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Question Suggestion Chips */}
        <div className="flex flex-wrap items-center justify-center gap-1.5 text-xs pt-1">
          <span className="text-[11px] text-[#737373] mr-1">Suggestions:</span>
          {questionSuggestions.map((q, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => {
                setQuestionText(q);
                if (onAskQuestion) {
                  onAskQuestion(q, llmModel);
                }
              }}
              disabled={isLoading || isCapturing}
              className={`text-[11px] px-2.5 py-0.5 rounded-full border transition-colors cursor-pointer disabled:opacity-50 ${
                questionText === q
                  ? 'bg-[#262626] border-[#404040] text-white font-medium'
                  : 'bg-[#181818] border-[#262626] text-[#8e8ea0] hover:text-white hover:border-[#333333]'
              }`}
            >
              {q}
            </button>
          ))}
        </div>
      </form>

      {/* ACTION BAR: Centered Microphone & Upload Buttons */}
      <div className="flex items-center justify-center gap-5 py-1">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac"
          className="hidden"
        />

        {/* Microphone Button */}
        {isCapturing ? (
          <button
            type="button"
            onClick={handleToggleCapture}
            disabled={isLoading}
            className="px-5 py-2.5 rounded-full bg-red-600 hover:bg-red-700 active:scale-95 text-white font-bold text-xs flex items-center gap-2 shadow-xl transition-all cursor-pointer animate-pulse"
            title="Stop Voice Recording"
          >
            <MicOff className="w-4 h-4" />
            <span>STOP RECORDING ({formatSeconds(recordSeconds)})</span>
          </button>
        ) : (
          <button
            type="button"
            onClick={handleToggleCapture}
            disabled={isLoading}
            className="p-3 rounded-full bg-transparent text-white hover:text-neutral-300 hover:bg-[#1e1e1e]/50 transition-all duration-150 active:scale-95 cursor-pointer"
            title="Start Voice Recording"
          >
            <Mic className="w-6 h-6" />
          </button>
        )}

        {/* Upload Audio File Button */}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={isCapturing || isLoading}
          className="p-3 rounded-full bg-transparent text-[#8e8ea0] hover:text-white hover:bg-[#1e1e1e]/50 transition-all duration-150 active:scale-95 cursor-pointer disabled:opacity-50"
          title="Upload Audio File for Core ALM Analysis"
        >
          <Upload className="w-6 h-6" />
        </button>
      </div>
    </div>
  );
}
