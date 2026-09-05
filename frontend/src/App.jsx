import React, { useState, useEffect } from 'react';
import { LandingPage } from './components/landing/LandingPage';
import { AuthScreen } from './components/AuthScreen';
import { Sidebar } from './components/Sidebar';
import { ConsoleHeader } from './components/ConsoleHeader';
import { VerdictPanel } from './components/VerdictPanel';
import { ComposerBar } from './components/ComposerBar';
import {
  createInitialSessions,
  saveSessions,
  getSeededJokeSession,
  getEmergencySession,
  getHypotheticalSession
} from './services/sessionStore';
import { analyzeAudio, checkHealth } from './services/api';
import { ArrowLeft, Home, Shield, Activity } from 'lucide-react';

/**
 * Smart Horizon Main Application
 * Integrates Landing Page and Operational Core ALM Console.
 */
export function App() {
  const [currentView, setCurrentView] = useState('landing');
  const [user, setUser] = useState({
    name: 'Alex Rivera',
    email: 'alex.rivera@horizon.sec',
    avatar: 'AR',
    role: 'Lead Forensic Analyst'
  });

  const [sessions, setSessions] = useState(() => createInitialSessions());
  const [activeSessionId, setActiveSessionId] = useState(() => sessions[0]?.id || 'sess_seeded_01');
  const [isLoadingAnalysis, setIsLoadingAnalysis] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const [realtimeTelemetry, setRealtimeTelemetry] = useState([]);
  const [apiErrorMessage, setApiErrorMessage] = useState(null);

  useEffect(() => {
    saveSessions(sessions);
  }, [sessions]);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth < 900) {
        setIsSidebarCollapsed(true);
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const activeSession = sessions.find((s) => s.id === activeSessionId) || sessions[0];

  const handleAuthSuccess = (authenticatedUser) => {
    setUser(authenticatedUser);
    setCurrentView('console');
  };

  const handleSignOut = () => {
    setCurrentView('landing');
  };

  const handleNewSession = () => {
    const newId = `sess_${Math.random().toString(36).substring(2, 9)}`;
    const newSession = {
      id: newId,
      label: `Core ALM Session #${sessions.length + 1}`,
      createdAt: new Date().toISOString(),
      capture: null,
      result: null
    };

    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newId);
  };

  const handleClearHistory = () => {
    const newId = `sess_${Math.random().toString(36).substring(2, 9)}`;
    const freshSession = {
      id: newId,
      label: 'Core ALM Session #1',
      createdAt: new Date().toISOString(),
      capture: null,
      result: null
    };
    setSessions([freshSession]);
    setActiveSessionId(newId);
  };

  const handleDeleteSession = (sessionId, e) => {
    if (e) e.stopPropagation();
    setSessions((prev) => {
      const filtered = prev.filter((s) => s.id !== sessionId);
      if (filtered.length === 0) {
        const newId = `sess_${Math.random().toString(36).substring(2, 9)}`;
        const freshSession = {
          id: newId,
          label: 'Core ALM Session #1',
          createdAt: new Date().toISOString(),
          capture: null,
          result: null
        };
        setActiveSessionId(newId);
        return [freshSession];
      }
      if (activeSessionId === sessionId) {
        setActiveSessionId(filtered[0].id);
      }
      return filtered;
    });
  };

  const handleSelectSession = (sessionId) => {
    setActiveSessionId(sessionId);
  };

  const handleLoadScenario = (scenarioType) => {
    let scenarioData;
    if (scenarioType === 'joke') scenarioData = getSeededJokeSession();
    else if (scenarioType === 'emergency') scenarioData = getEmergencySession();
    else if (scenarioType === 'hypothetical') scenarioData = getHypotheticalSession();
    else scenarioData = getSeededJokeSession();

    const newId = `sess_${scenarioType}_${Math.random().toString(36).substring(2, 7)}`;
    const sessionToInsert = {
      ...scenarioData,
      id: newId,
      createdAt: new Date().toISOString()
    };

    setSessions((prev) => [sessionToInsert, ...prev]);
    setActiveSessionId(newId);
  };

  const handleLiveCaptureStart = () => {
    setIsCapturing(true);
    setApiErrorMessage(null);
  };

  const handleLiveCaptureStop = async (spokenTranscript, duration = 3.0, acousticData = {}, audioBlob = null, questionText = '', llmModel = 'gpt-4o-mini', selectedLanguage = 'auto') => {
    setIsCapturing(false);
    setIsLoadingAnalysis(true);
    setApiErrorMessage(null);

    let mlResult = null;
    try {
      mlResult = await analyzeAudio(
        audioBlob,
        questionText,
        selectedLanguage || 'auto',
        activeSessionId,
        spokenTranscript,
        llmModel
      );
    } catch (err) {
      console.warn('[App] Core ALM API call exception:', err);
      setApiErrorMessage(err.message || 'Core ALM ML Service call failed');
    }

    const answerStr = mlResult?.answer || "Model inference complete.";
    const confidenceVal = mlResult?.confidence || 0.85;

    setSessions((prev) =>
      prev.map((s) => {
        if (s.id === activeSessionId) {
          return {
            ...s,
            label: `Voice: "${questionText.slice(0, 24)}"`,
            capture: {
              audioBlob: audioBlob,
              transcript: mlResult?.speech?.transcript || spokenTranscript || '',
              duration: Number(duration) || 3.0,
              question: questionText,
              vocalBiometrics: mlResult?.paralinguistic || { emotion: 'neutral', arousal: 'low', confidence: 0.90 },
              acousticEvents: mlResult?.audio_events || []
            },
            result: {
              answer: answerStr,
              confidence: confidenceVal,
              evidence: mlResult?.evidence || [],
              reasoning_evidence: mlResult?.reasoning_evidence,
              speech_evidence: mlResult?.speech_evidence,
              non_speech_evidence: mlResult?.non_speech_evidence,
              speaker_evidence: mlResult?.speaker_evidence,
              paralinguistic_evidence: mlResult?.paralinguistic_evidence,
              temporal_evidence: mlResult?.temporal_evidence,
              speech: mlResult?.speech || {},
              speakers: mlResult?.speakers || [],
              audio_events: mlResult?.audio_events || [],
              paralinguistic: mlResult?.paralinguistic || {},
              scene: mlResult?.scene || {},
              analyzedAt: new Date().toISOString()
            }
          };
        }
        return s;
      })
    );

    setIsLoadingAnalysis(false);
  };

  const handleAudioUploaded = async (file, questionText = '', llmModel = 'gpt-4o-mini', selectedLanguage = 'auto') => {
    setIsLoadingAnalysis(true);
    setApiErrorMessage(null);
    const fileName = file.name;

    let mlResult = null;
    try {
      mlResult = await analyzeAudio(
        file,
        questionText,
        selectedLanguage || 'auto',
        activeSessionId,
        '',
        llmModel
      );
    } catch (err) {
      console.warn('[App] Core ALM API upload exception:', err);
      setApiErrorMessage(err.message || 'Core ALM ML Service upload failed');
    }

    const answerStr = mlResult?.answer || "Model inference complete.";
    const confidenceVal = mlResult?.confidence || 0.85;

    setSessions((prev) =>
      prev.map((s) => {
        if (s.id === activeSessionId) {
          return {
            ...s,
            label: `File: ${fileName.slice(0, 20)}`,
            capture: {
              fileName,
              audioFile: file,
              question: questionText,
              transcript: mlResult?.speech?.transcript || '',
              vocalBiometrics: mlResult?.paralinguistic || { emotion: 'neutral', arousal: 'low', confidence: 0.90 },
              acousticEvents: mlResult?.audio_events || []
            },
            result: {
              answer: answerStr,
              confidence: confidenceVal,
              evidence: mlResult?.evidence || [],
              reasoning_evidence: mlResult?.reasoning_evidence,
              speech_evidence: mlResult?.speech_evidence,
              non_speech_evidence: mlResult?.non_speech_evidence,
              speaker_evidence: mlResult?.speaker_evidence,
              paralinguistic_evidence: mlResult?.paralinguistic_evidence,
              temporal_evidence: mlResult?.temporal_evidence,
              speech: mlResult?.speech || {},
              speakers: mlResult?.speakers || [],
              audio_events: mlResult?.audio_events || [],
              paralinguistic: mlResult?.paralinguistic || {},
              scene: mlResult?.scene || {},
              analyzedAt: new Date().toISOString()
            }
          };
        }
        return s;
      })
    );

    setIsLoadingAnalysis(false);
  };

  const handleAskQuestion = async (questionText) => {
    if (!activeSession) return;
    setIsLoadingAnalysis(true);
    setApiErrorMessage(null);

    const audioSource = activeSession.capture?.audioBlob || activeSession.capture?.audioFile;

    let mlResult = null;
    try {
      mlResult = await analyzeAudio(
        audioSource,
        questionText,
        'hi',
        activeSessionId
      );
    } catch (err) {
      console.warn('[App] Core ALM API question call exception:', err);
      setApiErrorMessage(err.message || 'Core ALM ML Service question call failed');
    }

    const answerStr = mlResult?.answer || "Model inference complete.";
    const confidenceVal = mlResult?.confidence || 0.85;

    setSessions((prev) =>
      prev.map((s) => {
        if (s.id === activeSessionId) {
          return {
            ...s,
            capture: {
              ...s.capture,
              question: questionText
            },
            result: {
              ...s.result,
              answer: answerStr,
              confidence: confidenceVal,
              evidence: mlResult?.evidence || s.result?.evidence || [],
              reasoning_evidence: mlResult?.reasoning_evidence || s.result?.reasoning_evidence,
              speech_evidence: mlResult?.speech_evidence || s.result?.speech_evidence,
              non_speech_evidence: mlResult?.non_speech_evidence || s.result?.non_speech_evidence,
              speaker_evidence: mlResult?.speaker_evidence || s.result?.speaker_evidence,
              paralinguistic_evidence: mlResult?.paralinguistic_evidence || s.result?.paralinguistic_evidence,
              temporal_evidence: mlResult?.temporal_evidence || s.result?.temporal_evidence,
              speech: mlResult?.speech || s.result?.speech || {},
              speakers: mlResult?.speakers || s.result?.speakers || [],
              audio_events: mlResult?.audio_events || s.result?.audio_events || [],
              paralinguistic: mlResult?.paralinguistic || s.result?.paralinguistic || {},
              scene: mlResult?.scene || s.result?.scene || {},
              analyzedAt: new Date().toISOString()
            }
          };
        }
        return s;
      })
    );

    setIsLoadingAnalysis(false);
  };

  if (currentView === 'landing') {
    return <LandingPage onOpenConsole={() => setCurrentView('console')} />;
  }

  if (currentView === 'auth') {
    return <AuthScreen onAuthSuccess={handleAuthSuccess} />;
  }

  return (
    <div className="h-screen w-screen bg-[#0d0d0d] text-[#ececec] flex flex-col overflow-hidden font-sans select-none antialiased">
      {/* Top Console Return Bar */}
      <div className="h-10 px-6 bg-[#121212] border-b border-[#262626] flex items-center justify-between text-xs font-mono shrink-0 shadow-xs z-20">
        <button
          type="button"
          onClick={() => setCurrentView('landing')}
          className="flex items-center gap-2 text-[#8e8ea0] hover:text-white transition-all duration-150 cursor-pointer active:scale-95 group focus-visible:outline-white/40"
        >
          <ArrowLeft className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform" />
          <span className="font-medium tracking-tight">Return to Landing Page</span>
        </button>

        <div className="flex items-center gap-3.5 text-[#8e8ea0]">
          <span className="hidden sm:inline tracking-wider uppercase text-[10px] font-semibold">
            CORE ALM WORKSTATION
          </span>
          <span className="px-2.5 py-0.5 rounded-md bg-[#1e1e1e] border border-[#262626] text-[10px] text-white font-bold shadow-2xs">
            SH-DST-02
          </span>
        </div>
      </div>

      {/* Main Layout */}
      <div className="flex-1 flex overflow-hidden">
        <Sidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          onSelectSession={handleSelectSession}
          onNewSession={handleNewSession}
          onClearHistory={handleClearHistory}
          onDeleteSession={handleDeleteSession}
          user={user}
          onSignOut={handleSignOut}
          isCollapsed={isSidebarCollapsed}
          onToggleCollapse={() => setIsSidebarCollapsed((prev) => !prev)}
        />

        <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden bg-[#0d0d0d]">
          <ConsoleHeader activeSession={activeSession} />

          {apiErrorMessage && (
            <div className="bg-red-500/10 border-b border-red-500/30 px-6 py-2 text-xs text-red-400 flex items-center justify-between">
              <span>⚠️ API Warning: {apiErrorMessage}</span>
              <button onClick={() => setApiErrorMessage(null)} className="underline cursor-pointer">Dismiss</button>
            </div>
          )}

          <VerdictPanel
            activeSession={activeSession}
            isLoading={isLoadingAnalysis}
            isCapturing={isCapturing}
            realtimeTelemetry={realtimeTelemetry}
          />

          <ComposerBar
            onLiveCaptureStart={handleLiveCaptureStart}
            onLiveCaptureStop={handleLiveCaptureStop}
            onAudioUploaded={handleAudioUploaded}
            onAskQuestion={handleAskQuestion}
            onLoadScenario={handleLoadScenario}
            onAudioTelemetryUpdate={(bars) => setRealtimeTelemetry(bars)}
            isCapturing={isCapturing}
            isLoading={isLoadingAnalysis}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
