import React, { useState, useRef, useEffect, useCallback } from 'react';

export function RecordOrUpload({ sessionId, onSessionChange, onAudioReady, onSimulateSample, activeScenario }) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0);
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [peakDb, setPeakDb] = useState('-∞ dBFS');
  const [rmsDb, setRmsDb] = useState('-∞ dBFS');

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);
  const audioPlayerRef = useRef(null);
  const mediaElementSourceRef = useRef(null);
  const fileInputRef = useRef(null);
  const speechRecognitionRef = useRef(null);
  const liveTranscriptRef = useRef('');
  const peakVolumeRef = useRef(0);
  const idlePhaseRef = useRef(0);

  const stopVisualizer = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    if (sourceRef.current) {
      try { sourceRef.current.disconnect(); } catch (e) {}
      sourceRef.current = null;
    }
    setPeakDb('-∞ dBFS');
    setRmsDb('-∞ dBFS');
  }, []);

  const startIdleVisualizer = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const renderIdle = () => {
      idlePhaseRef.current += 0.03;
      const phase = idlePhaseRef.current;
      const width = canvas.width;
      const height = canvas.height;

      ctx.fillStyle = '#0d0c08';
      ctx.fillRect(0, 0, width, height);

      // Grid Lines
      ctx.strokeStyle = '#1a1812';
      ctx.lineWidth = 1;

      for (let y = 20; y < height; y += 30) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      for (let x = 60; x < width; x += 100) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }

      // Subtle Ambient FFT Spectrum Floor
      const barCount = 72;
      const barWidth = width / barCount;

      for (let i = 0; i < barCount; i++) {
        const ambientAmp = Math.sin(i * 0.15 + phase) * 6 + Math.cos(i * 0.08 - phase * 0.5) * 4 + 8;
        const grad = ctx.createLinearGradient(0, height, 0, height - ambientAmp);
        grad.addColorStop(0, 'rgba(192, 133, 50, 0.05)');
        grad.addColorStop(1, 'rgba(192, 133, 50, 0.25)');

        ctx.fillStyle = grad;
        ctx.fillRect(i * barWidth + 1, height - ambientAmp, barWidth - 2, ambientAmp);
      }

      // Ambient Oscilloscope Baseline Drift
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = 'rgba(192, 133, 50, 0.45)';
      ctx.beginPath();

      for (let x = 0; x < width; x += 4) {
        const y = height / 2 + Math.sin(x * 0.015 + phase) * 3 + Math.sin(x * 0.04 - phase * 1.5) * 2;
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      animationFrameRef.current = requestAnimationFrame(renderIdle);
    };

    renderIdle();
  }, []);

  const runActiveAnalysis = useCallback((analyser) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const bufferLength = analyser.frequencyBinCount;
    const freqData = new Uint8Array(bufferLength);
    const timeData = new Uint8Array(bufferLength);

    let frameCounter = 0;

    const drawActive = () => {
      animationFrameRef.current = requestAnimationFrame(drawActive);
      analyser.getByteFrequencyData(freqData);
      analyser.getByteTimeDomainData(timeData);

      const width = canvas.width;
      const height = canvas.height;

      ctx.fillStyle = '#0d0c08';
      ctx.fillRect(0, 0, width, height);

      // Grid
      ctx.strokeStyle = '#1a1812';
      ctx.lineWidth = 1;
      for (let y = 20; y < height; y += 30) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Fluid FFT Spectrum Bars
      const barCount = 80;
      const barWidth = width / barCount;
      let peakVal = 0;
      let sumSquares = 0;

      for (let i = 0; i < barCount; i++) {
        const index = Math.floor(Math.pow(i / barCount, 1.8) * bufferLength);
        const value = freqData[index] || 0;
        if (value > peakVal) peakVal = value;

        const barHeight = (value / 255) * height * 0.88;

        const grad = ctx.createLinearGradient(0, height, 0, height - barHeight);
        grad.addColorStop(0, 'rgba(192, 133, 50, 0.2)');
        grad.addColorStop(0.7, 'rgba(192, 133, 50, 0.7)');
        grad.addColorStop(1, 'rgba(245, 245, 243, 0.95)');

        ctx.fillStyle = grad;
        ctx.fillRect(i * barWidth + 1, height - barHeight, barWidth - 1.5, barHeight);
      }

      // Track max peak
      if (peakVal > peakVolumeRef.current) {
        peakVolumeRef.current = peakVal;
      }

      // Overlay Oscilloscope Trace Line
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = '#f5f5f3';
      ctx.beginPath();

      const sliceWidth = width / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = (timeData[i] - 128) / 128.0;
        sumSquares += v * v;
        const y = v * (height * 0.45) + height / 2;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);

        x += sliceWidth;
      }

      ctx.stroke();

      // Peak & RMS calculation
      frameCounter++;
      if (frameCounter % 6 === 0) {
        const peakDbVal = peakVal > 0 ? (20 * Math.log10(peakVal / 255)).toFixed(1) : '-90.0';
        const rmsVal = Math.sqrt(sumSquares / bufferLength);
        const rmsDbVal = rmsVal > 0 ? (20 * Math.log10(rmsVal)).toFixed(1) : '-90.0';

        setPeakDb(`${peakDbVal} dBFS`);
        setRmsDb(`${rmsDbVal} dBFS`);
      }
    };

    drawActive();
  }, []);

  const startStreamVisualizer = useCallback((stream) => {
    try {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
        audioContextRef.current = new AudioCtx();
      }
      const audioCtx = audioContextRef.current;
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }

      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 512;
      analyser.minDecibels = -90;
      analyser.maxDecibels = -10;
      analyser.smoothingTimeConstant = 0.85;
      analyserRef.current = analyser;

      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);
      sourceRef.current = source;

      runActiveAnalysis(analyser);
    } catch (err) {
      // visualizer fallback
    }
  }, [runActiveAnalysis]);

  const attachAudioElementVisualizer = useCallback(() => {
    try {
      if (!audioPlayerRef.current) return;
      const audioEl = audioPlayerRef.current;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
        audioContextRef.current = new AudioCtx();
      }
      const audioCtx = audioContextRef.current;
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }

      if (!analyserRef.current) {
        const analyser = audioCtx.createAnalyser();
        analyser.fftSize = 512;
        analyser.minDecibels = -90;
        analyser.maxDecibels = -10;
        analyser.smoothingTimeConstant = 0.85;
        analyserRef.current = analyser;
      }

      if (!mediaElementSourceRef.current) {
        try {
          const source = audioCtx.createMediaElementSource(audioEl);
          source.connect(analyserRef.current);
          analyserRef.current.connect(audioCtx.destination);
          mediaElementSourceRef.current = source;
        } catch (e) {
          // already connected
        }
      }

      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      runActiveAnalysis(analyserRef.current);
    } catch (err) {
      // visualizer fallback
    }
  }, [runActiveAnalysis]);

  const startRecording = async () => {
    try {
      if (isPlaying && audioPlayerRef.current) {
        audioPlayerRef.current.pause();
      }

      audioChunksRef.current = [];
      liveTranscriptRef.current = '';
      peakVolumeRef.current = 0;

      // Start Web Speech Recognition if available in browser
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        try {
          const recognition = new SpeechRecognition();
          recognition.continuous = true;
          recognition.interimResults = true;
          recognition.lang = 'en-US';

          recognition.onresult = (event) => {
            let fullText = '';
            for (let i = 0; i < event.results.length; i++) {
              fullText += event.results[i][0].transcript + ' ';
            }
            liveTranscriptRef.current = fullText.trim();
          };

          recognition.onerror = (e) => {
            console.log('Speech recognition notice:', e.error);
          };

          recognition.start();
          speechRecognitionRef.current = recognition;
        } catch (recErr) {
          console.log('Speech recognition start note:', recErr);
        }
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);
        setAudioFile({ name: `capture-${Date.now()}.wav`, size: audioBlob.size });

        if (speechRecognitionRef.current) {
          try { speechRecognitionRef.current.stop(); } catch (e) {}
          speechRecognitionRef.current = null;
        }

        const transcript = liveTranscriptRef.current;
        const peakVol = peakVolumeRef.current;
        const isLowVol = peakVol > 0 && peakVol < 150;

        if (onAudioReady) {
          onAudioReady({
            blob: audioBlob,
            url,
            fileName: `capture-${Date.now()}.wav`,
            recognizedTranscript: transcript,
            peakVolume: peakVol,
            isLowVolume: isLowVol
          });
        }
        stream.getTracks().forEach((track) => track.stop());
        stopVisualizer();
        startIdleVisualizer();
      };

      mediaRecorder.start(200);
      setIsRecording(true);
      setRecordDuration(0);

      timerRef.current = setInterval(() => {
        setRecordDuration((prev) => prev + 1);
      }, 1000);

      startStreamVisualizer(stream);
    } catch (err) {
      setIsRecording(false);
      alert('Microphone access is required for live audio capture.');
    }
  };

  const stopRecording = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsRecording(false);

    if (mediaRecorderRef.current) {
      try {
        if (mediaRecorderRef.current.state !== 'inactive') {
          mediaRecorderRef.current.stop();
        }
      } catch (e) {
        console.warn('Error stopping MediaRecorder:', e);
      }
    }

    if (speechRecognitionRef.current) {
      try { speechRecognitionRef.current.stop(); } catch (e) {}
      speechRecognitionRef.current = null;
    }
  };

  const processSelectedAudioFile = (file) => {
    if (!file) return;

    // Check MIME or extension
    const isAudioType = file.type && file.type.startsWith('audio/');
    const isAudioExt = /\.(wav|mp3|ogg|m4a|flac|aac|wma|weba|mp4|webm)$/i.test(file.name);

    if (!isAudioType && !isAudioExt) {
      alert('Please select an audio file (.wav, .mp3, .ogg, .m4a, .flac, .aac, .weba).');
      return;
    }

    if (isPlaying && audioPlayerRef.current) {
      audioPlayerRef.current.pause();
    }

    setAudioFile(file);
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    setCurrentTime(0);

    if (onAudioReady) {
      onAudioReady({ file, url, fileName: file.name });
    }
  };

  const handleTogglePlay = () => {
    if (!audioPlayerRef.current || !audioUrl) return;

    const player = audioPlayerRef.current;
    if (isPlaying) {
      player.pause();
    } else {
      attachAudioElementVisualizer();
      player.play().catch((err) => {
        console.warn('Playback error:', err);
      });
    }
  };

  const handleSeek = (e) => {
    if (!audioPlayerRef.current) return;
    const target = parseFloat(e.target.value);
    audioPlayerRef.current.currentTime = target;
    setCurrentTime(target);
  };

  const handleEject = () => {
    if (isPlaying && audioPlayerRef.current) {
      audioPlayerRef.current.pause();
    }
    setAudioUrl(null);
    setAudioFile(null);
    setCurrentTime(0);
    setDuration(0);
    setIsPlaying(false);
    stopVisualizer();
    startIdleVisualizer();
  };

  useEffect(() => {
    startIdleVisualizer();
    return () => {
      stopVisualizer();
      if (timerRef.current) clearInterval(timerRef.current);
      if (speechRecognitionRef.current) {
        try { speechRecognitionRef.current.stop(); } catch (e) {}
      }
    };
  }, [startIdleVisualizer, stopVisualizer]);

  const formatSeconds = (sec) => {
    if (!Number.isFinite(sec) || isNaN(sec) || sec < 0) return '00:00';
    const mins = Math.floor(sec / 60);
    const secs = Math.floor(sec % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <section
      className={`cursor-hero-agent-window ${isDragOver ? 'drag-over' : ''}`}
      id="hero-acoustic-monitor"
      onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragOver(false);
        if (e.dataTransfer.files?.[0]) {
          processSelectedAudioFile(e.dataTransfer.files[0]);
        }
      }}
    >
      {/* Hidden Audio Player Tag */}
      {audioUrl && (
        <audio
          ref={audioPlayerRef}
          src={audioUrl}
          crossOrigin="anonymous"
          onPlay={() => {
            setIsPlaying(true);
            attachAudioElementVisualizer();
          }}
          onPause={() => {
            setIsPlaying(false);
            stopVisualizer();
            startIdleVisualizer();
          }}
          onEnded={() => {
            setIsPlaying(false);
            stopVisualizer();
            startIdleVisualizer();
          }}
          onTimeUpdate={() => {
            if (audioPlayerRef.current) {
              setCurrentTime(audioPlayerRef.current.currentTime);
            }
          }}
          onLoadedMetadata={() => {
            if (audioPlayerRef.current) {
              setDuration(audioPlayerRef.current.duration || 0);
            }
          }}
        />
      )}

      {/* Cursor Prompt Bar Header */}
      <div className="cursor-prompt-bar">
        <div className="cursor-prompt-left">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: 'var(--cursor-copper)' }}>
            <path d="M2 12h3l3 8 4-16 4 16 3-8h3" />
          </svg>
          <span className="cursor-prompt-placeholder">
            Ask Acoustic Agent to monitor or analyze live sound field...
          </span>
        </div>

        <div className="cursor-prompt-selectors">
          <div className="cursor-agent-select-chip">
            <span style={{ color: 'var(--cursor-copper)' }}>●</span>
            <span>Agent</span>
            <span className="cursor-select-caret">▾</span>
          </div>

          <div className="cursor-agent-select-chip">
            <span>Grok-DSP</span>
            <span className="cursor-select-caret">▾</span>
          </div>

          {isRecording ? (
            <button
              type="button"
              onClick={stopRecording}
              className="cursor-chip-pill"
              style={{
                marginLeft: '6px',
                backgroundColor: '#ef4444',
                color: '#ffffff',
                fontWeight: 'bold',
                cursor: 'pointer',
                border: 'none',
                padding: '4px 10px'
              }}
            >
              <span>⏹ STOP MIC</span>
            </button>
          ) : (
            <span className="cursor-chip-pill" style={{ marginLeft: '6px' }}>
              <span>{isPlaying ? 'Playing Audio' : 'Ready'}</span>
            </span>
          )}
        </div>
      </div>

      {/* Wide Hero Visualizer Viewport */}
      <div className="cursor-canvas-viewport" style={{ position: 'relative' }}>
        {/* Floating Active Recording Overlay Banner */}
        {isRecording && (
          <div
            style={{
              position: 'absolute',
              top: '14px',
              left: '50%',
              transform: 'translateX(-50%)',
              zIndex: 40,
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              backgroundColor: 'rgba(225, 29, 72, 0.95)',
              boxShadow: '0 0 25px rgba(225, 29, 72, 0.75)',
              padding: '8px 20px',
              borderRadius: '30px',
              border: '1px solid #fca5a5'
            }}
          >
            <span
              style={{
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                backgroundColor: '#ffffff',
                display: 'inline-block'
              }}
            />
            <span
              style={{
                fontFamily: 'monospace',
                fontWeight: 'bold',
                fontSize: '0.85rem',
                color: '#ffffff',
                letterSpacing: '0.05em'
              }}
            >
              LIVE MIC STREAMING [{formatSeconds(recordDuration)}]
            </span>
            <button
              type="button"
              onClick={stopRecording}
              style={{
                backgroundColor: '#ffffff',
                color: '#e11d48',
                border: 'none',
                padding: '6px 16px',
                borderRadius: '20px',
                fontWeight: 'bold',
                fontSize: '0.8rem',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                boxShadow: '0 2px 10px rgba(0,0,0,0.4)',
                transition: 'transform 0.1s'
              }}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                <rect x="4" y="4" width="16" height="16" rx="2" />
              </svg>
              STOP RECORDING
            </button>
          </div>
        )}

        <canvas
          ref={canvasRef}
          className="cursor-canvas-element"
          width={1200}
          height={200}
        />
        <div className="cursor-canvas-hud-overlay">
          <span>0 dBFS</span>
          <span>-24 dBFS</span>
          <span>-48 dBFS</span>
        </div>
      </div>

      {/* Frequency Hz Ruler */}
      <div className="cursor-frequency-ruler">
        <span>20 Hz</span>
        <span>100 Hz</span>
        <span>500 Hz</span>
        <span>1.0 kHz</span>
        <span>4.0 kHz</span>
        <span>10.0 kHz</span>
        <span>20.0 kHz</span>
      </div>

      {/* Cursor Controls & Verification Modes Bar */}
      <div className="cursor-controls-bar">
        {/* Primary Acquisition Buttons */}
        <div className="cursor-primary-actions">
          {!isRecording ? (
            <button
              className="btn-cursor-primary"
              id="btn-start-record"
              onClick={startRecording}
            >
              <span>●</span>
              <span>Start Live Capture</span>
            </button>
          ) : (
            <button
              className="btn-cursor-primary recording"
              id="btn-stop-record"
              onClick={stopRecording}
              style={{ backgroundColor: '#ef4444', borderColor: '#dc2626', color: '#ffffff' }}
            >
              <span>⏹</span>
              <span>Stop Recording ({formatSeconds(recordDuration)})</span>
            </button>
          )}

          <button
            className="btn-cursor-secondary"
            id="btn-upload-trigger"
            onClick={() => fileInputRef.current?.click()}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            <span>Upload Audio</span>
          </button>
          <input
            ref={fileInputRef}
            type="file"
            id="hero-audio-file-input"
            accept="audio/*,.wav,.mp3,.ogg,.m4a,.flac,.aac,.weba,.wma,.mp4,.webm"
            style={{ display: 'none' }}
            onChange={(e) => {
              if (e.target.files?.[0]) {
                processSelectedAudioFile(e.target.files[0]);
                e.target.value = '';
              }
            }}
          />

          <button
            className="btn-cursor-secondary"
            id="btn-load-airport-benchmark"
            onClick={() => {
              fetch('/samples/airport_concourse_benchmark.wav')
                .then((res) => res.blob())
                .then((blob) => {
                  const file = new File([blob], 'airport_concourse_benchmark.wav', { type: 'audio/wav' });
                  processSelectedAudioFile(file);
                })
                .catch(() => {
                  alert('Airport benchmark audio file ready in samples directory.');
                });
            }}
            title="Load Problem Statement Airport Concourse Benchmark audio track"
          >
            <span>✈ Load PS Airport Track</span>
          </button>
        </div>

        {/* Cursor Verification Test Modes */}
        {onSimulateSample && (
          <div className="cursor-verification-modes">
            <span className="cursor-verification-label">Modes:</span>

            <div
              className={`cursor-mode-card ${activeScenario === 'airport' ? 'active-warning' : ''}`}
              onClick={() => onSimulateSample('airport')}
            >
              <span className="cursor-mode-bullet amber"></span>
              <span className="cursor-mode-title">PS Airport Concourse</span>
              <span className="cursor-mode-tag">PS Benchmark</span>
            </div>

            <div
              className={`cursor-mode-card ${activeScenario === 'threat' ? 'active-threat' : ''}`}
              onClick={() => onSimulateSample('threat')}
            >
              <span className="cursor-mode-bullet rose"></span>
              <span className="cursor-mode-title">Gunfire + Screaming</span>
              <span className="cursor-mode-tag">Critical</span>
            </div>

            <div
              className={`cursor-mode-card ${activeScenario === 'covert' ? 'active-threat' : ''}`}
              onClick={() => onSimulateSample('covert')}
            >
              <span className="cursor-mode-bullet rose"></span>
              <span className="cursor-mode-title">Whisper / Covert Distress</span>
              <span className="cursor-mode-tag">Low-Vol Threat</span>
            </div>

            <div
              className={`cursor-mode-card ${activeScenario === 'fire' ? 'active-warning' : ''}`}
              onClick={() => onSimulateSample('fire')}
            >
              <span className="cursor-mode-bullet amber"></span>
              <span className="cursor-mode-title">Smoke + Alarm</span>
              <span className="cursor-mode-tag">Warning</span>
            </div>

            <div
              className={`cursor-mode-card ${activeScenario === 'nominal' ? 'active-nominal' : ''}`}
              onClick={() => onSimulateSample('nominal')}
            >
              <span className="cursor-mode-bullet emerald"></span>
              <span className="cursor-mode-title">Conversational</span>
              <span className="cursor-mode-tag">Normal</span>
            </div>
          </div>
        )}
      </div>

      {/* Interactive Cursor Audio Playback Bar */}
      {audioUrl && (
        <div className="cursor-playback-strip" id="audio-playback-strip">
          <div className="cursor-playback-left">
            <button
              className="btn-playback-playpause"
              onClick={handleTogglePlay}
              title={isPlaying ? 'Pause Audio' : 'Play Audio'}
            >
              {isPlaying ? (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="4" width="4" height="16" />
                  <rect x="14" y="4" width="4" height="16" />
                </svg>
              ) : (
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" style={{ transform: 'translateX(1px)' }}>
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
              )}
            </button>

            <div className="playback-file-meta">
              <span className="playback-filename">
                {audioFile ? audioFile.name : 'Stream Capture'}
              </span>
              <span className="playback-badge-tag">
                {audioFile?.size ? `${Math.round(audioFile.size / 1024)} KB` : 'DSP Capture'} · 48 kHz
              </span>
            </div>
          </div>

          <div className="cursor-playback-scrubber-box">
            <span className="playback-timecode">{formatSeconds(currentTime)}</span>
            <input
              type="range"
              className="playback-seek-slider"
              min="0"
              max={duration || 1}
              step="0.01"
              value={currentTime}
              onChange={handleSeek}
            />
            <span className="playback-timecode">{formatSeconds(duration)}</span>
          </div>

          <button
            className="btn-playback-eject"
            onClick={handleEject}
            title="Eject audio"
          >
            ✕ Eject
          </button>
        </div>
      )}
    </section>
  );
}

export default RecordOrUpload;
