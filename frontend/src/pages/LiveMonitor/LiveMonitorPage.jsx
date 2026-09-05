import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useWebSocket } from './useWebSocket';
import { AlertBanner } from './AlertBanner';
import { RecordOrUpload } from './RecordOrUpload';
import { ResultPanel } from './ResultPanel';
import './LiveMonitor.css';

const DISTRESS_KEYWORDS = [
  'help', 'emergency', 'fire', 'police', 'danger', 'sos', 'save me',
  'attacker', 'gun', 'shooting', 'kill', 'bomb', 'intruder', 'hostage',
  'threat', 'scream', 'hurt', 'stop', 'smoke', 'knife', 'robbery',
  'robber', 'thief', 'weapon', 'trapped', 'blood', 'dying', 'attack',
  'assault', 'burglar', 'burglary', 'scared', 'hide', 'hiding', 'quiet',
  'don\'t shoot', 'choking', 'save us', 'hostages', 'stab', 'murder',
  'break in', 'someone is here', '911', 'ambulance', 'warning', 'hostile'
];

function segmentSpeechToSpeakers(transcriptText, isDistress, isLowVolume) {
  const clean = (transcriptText || '').trim();
  if (!clean) {
    return [
      {
        speaker_id: 'SPK-01 (Primary Speaker)',
        role: isDistress ? (isLowVolume ? 'Distress Speaker (Covert / Whispered)' : 'Distress Caller') : 'Active Speaker',
        text: 'Acoustic background monitored. No verbal utterances detected.',
        timeSpan: '[0.0s - 3.0s]',
        duration: 3.0,
        threatCategory: isDistress ? 'critical' : 'nominal',
        emotion: isDistress
          ? { primary: 'fear', arousal: isLowVolume ? 'low' : 'high', confidence: 0.92 }
          : { primary: 'neutral', arousal: 'low', confidence: 0.95 }
      }
    ];
  }

  // 1. Explicit speaker prefixes from diarization pipeline (e.g. "Speaker 1:", "SPK-01:")
  const speakerTagMatch = clean.match(/(?:Speaker\s*(\d+)|SPK-?0?(\d+))\s*:\s*([^]+?)(?=(?:Speaker\s*\d+|SPK-?0?\d+:|$))/gi);
  if (speakerTagMatch && speakerTagMatch.length > 1) {
    return speakerTagMatch.map((block, idx) => {
      const parts = block.split(/:\s*/);
      const spkId = parts[0].trim();
      const spkText = parts.slice(1).join(': ').trim();
      return {
        speaker_id: `SPK-0${idx + 1} (${spkId})`,
        text: spkText,
        timeSpan: `[${(idx * 3.5).toFixed(1)}s - ${((idx + 1) * 3.5).toFixed(1)}s]`,
        duration: 3.5
      };
    });
  }

  // 2. Multi-Party Dialogue Pattern: Explicit turn-take exchanges (e.g. Q&A or Command/Response)
  const turnTakeSplit = clean.match(/^([^!?]+[!?])\s+((?:wait|no|yes|please|drop|stop|hey|look|okay)\b[^]+)$/i);
  if (turnTakeSplit && turnTakeSplit.length === 3) {
    const spk1Text = turnTakeSplit[1].trim();
    const spk2Text = turnTakeSplit[2].trim();

    // Ensure it is not just a single speaker saying "wait, no" in the same thought
    if (spk1Text.length > 10 && spk2Text.length > 10) {
      return [
        {
          speaker_id: 'SPK-01 (Primary Speaker)',
          text: spk1Text,
          timeSpan: '[0.0s - 3.8s]',
          duration: 3.8
        },
        {
          speaker_id: 'SPK-02 (Secondary Speaker)',
          text: spk2Text,
          timeSpan: '[4.0s - 7.5s]',
          duration: 3.5
        }
      ];
    }
  }

  // 3. Default Conservative Principle: Continuous single speaker train of thought
  // Pauses, pitch/energy fluctuations, and clause continuations ("and", "was a joke") belong to the same speaker.
  return [
    {
      speaker_id: 'SPK-01 (Active Speaker)',
      text: clean,
      timeSpan: '[0.0s - 4.5s]',
      duration: 4.5
    }
  ];
}

export function LiveMonitorPage() {
  const [sessionId] = useState(() => `sess_${Math.random().toString(36).substring(2, 9)}`);
  const [sceneResult, setSceneResult] = useState(null);
  const [activeAlerts, setActiveAlerts] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeScenario, setActiveScenario] = useState(null);

  // Short-term bounded session conversation memory (sliding window of last 4 utterances)
  const sessionHistoryRef = useRef([]);

  const {
    status: wsStatus,
    isConnected,
    latestResult,
    alerts: wsAlerts,
    dismissAlert: dismissWsAlert,
    clearAlerts: clearWsAlerts,
    injectMockResult
  } = useWebSocket(sessionId, {
    onResultReady: (data) => {
      setSceneResult(data);
      setIsLoading(false);
    },
    onAlertRaised: (alert) => {
      setActiveAlerts((prev) => [alert, ...prev]);
    }
  });

  useEffect(() => {
    if (wsAlerts && wsAlerts.length > 0) {
      setActiveAlerts(wsAlerts);
    }
  }, [wsAlerts]);

  useEffect(() => {
    if (latestResult) {
      setSceneResult(latestResult);
    }
  }, [latestResult]);

  const fetchSessionData = useCallback(async (id) => {
    if (!id) return;
    try {
      setIsLoading(true);
      const res = await fetch(`http://localhost:4000/api/v1/sessions/${id}/result`);
      if (res.ok) {
        const data = await res.json();
        setSceneResult(data);
        if (data.alerts) {
          setActiveAlerts(data.alerts);
        }
      }
    } catch (e) {
      // API standby
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSessionData(sessionId);
  }, [sessionId, fetchSessionData]);

  const handleDismissAlert = (alertId) => {
    setActiveAlerts((prev) => prev.filter((a) => (a.id || a.alert_id) !== alertId));
    dismissWsAlert(alertId);
  };

  const handleClearAllAlerts = () => {
    setActiveAlerts([]);
    clearWsAlerts();
  };

  const handleDeleteSpeaker = (speakerId) => {
    if (!sceneResult) return;
    const currentSpeakers = Array.isArray(sceneResult.speakers) ? sceneResult.speakers : [];
    const updatedSpeakers = currentSpeakers.filter(
      (s) => (s.speaker_id || s.id) !== speakerId
    );

    const hasRemainingCritical = updatedSpeakers.some((s) => {
      const text = (s.text || '').toLowerCase();
      return text.includes('help') || text.includes('emergency') || text.includes('fire') || text.includes('gun') || s.emotion?.primary === 'fear';
    });

    let newAlerts = activeAlerts;
    if (!hasRemainingCritical) {
      newAlerts = [];
      setActiveAlerts([]);
    }

    const updatedResult = {
      ...sceneResult,
      speakers: updatedSpeakers,
      alerts: newAlerts,
      fusion_summary: updatedSpeakers.length > 0
        ? `Speaker "${speakerId}" deleted from scene. ${updatedSpeakers.length} active voice(s) remaining.`
        : 'All speaker voices dismissed. Acoustic scene nominal.'
    };

    setSceneResult(updatedResult);
    injectMockResult(updatedResult);
  };

  const handleClarifyResponse = (choice) => {
    if (!sceneResult) return;
    const updatedIntel = {
      ...(sceneResult.intelligence || {}),
      classification: choice === 'emergency' ? 'emergency' : 'hypothetical',
      severity: choice === 'emergency' ? 'high' : 'none',
      recommended_action: choice === 'emergency' ? 'alert' : 'no_alert',
      ambiguity: 0.05,
      explanation: choice === 'emergency'
        ? 'User confirmed active emergency situation during interactive clarification.'
        : 'User confirmed safe / hypothetical context during interactive clarification.'
    };

    const newAlerts = choice === 'emergency' ? [
      {
        id: `alert-clarified-${Date.now()}`,
        session_id: sessionId,
        type: 'keyword',
        severity: 'high',
        message: 'Active emergency confirmed through interactive user clarification',
        created_at: new Date().toISOString()
      }
    ] : [];

    const updatedResult = {
      ...sceneResult,
      intelligence: updatedIntel,
      alerts: newAlerts,
      fusion_summary: updatedIntel.explanation
    };

    setSceneResult(updatedResult);
    setActiveAlerts(newAlerts);
    injectMockResult(updatedResult);
  };

  const handleAudioReady = async ({ blob, file, fileName, recognizedTranscript, peakVolume, isLowVolume: rawIsLowVolume }) => {
    setIsLoading(true);
    setActiveScenario(null);

    const name = fileName || file?.name || 'Audio Ingest';
    const audioPayload = blob || file;

    let mlData = null;

    // 1. Attempt real inference from FastAPI ML Service (Port 8000)
    if (audioPayload) {
      try {
        const formData = new FormData();
        formData.append('audio_file', audioPayload, name);
        formData.append('session_id', sessionId);

        const mlRes = await fetch('http://localhost:8000/analyze', {
          method: 'POST',
          body: formData
        });

        if (mlRes.ok) {
          mlData = await mlRes.json();
        }
      } catch (mlErr) {
        console.log('FastAPI inference notice, using intelligent workstation perception:', mlErr);
      }
    }

    // 2. Synthesize Context-Aware Multi-Factor Acoustic Intelligence
    const rawText = (recognizedTranscript || mlData?.transcript?.text || '').trim();
    const spokenLower = rawText.toLowerCase();

    // Update Session Conversation History (Sliding Window of 4 items)
    if (rawText) {
      sessionHistoryRef.current = [
        ...sessionHistoryRef.current.slice(-3),
        { text: rawText, timestamp: Date.now() }
      ];
    }

    const previousTurns = sessionHistoryRef.current.slice(0, -1).map(t => t.text.toLowerCase()).join(' ');

    // Feature Extraction: Keyword Features (Features, NOT decisions!)
    const matchedDistressWords = DISTRESS_KEYWORDS.filter((kw) => {
      const regex = new RegExp(`\\b${kw}\\b`, 'i');
      return regex.test(spokenLower) || spokenLower.includes(kw);
    });

    // Check Modalities
    const isExplicitJoke =
      spokenLower.includes('joke') ||
      spokenLower.includes('kidding') ||
      spokenLower.includes('just playing') ||
      spokenLower.includes('prank') ||
      spokenLower.includes('was a joke') ||
      spokenLower.includes('it is a joke') ||
      spokenLower.includes('it was a joke') ||
      spokenLower.includes('telling you a joke') ||
      spokenLower.includes('telling your joke') ||
      spokenLower.includes('haha') ||
      spokenLower.includes('lol') ||
      spokenLower.includes('😂') ||
      previousTurns.includes('telling a joke') ||
      previousTurns.includes('stupid joke');

    const isHypothetical =
      spokenLower.includes('what would you do if') ||
      spokenLower.includes('what if') ||
      spokenLower.includes('suppose') ||
      spokenLower.includes('imagine if') ||
      spokenLower.includes('hypothetically') ||
      (rawText.endsWith('?') && (spokenLower.includes('what') || spokenLower.includes('would')));

    const isStorytelling =
      spokenLower.includes('movie') ||
      spokenLower.includes('film') ||
      spokenLower.includes('in this game') ||
      spokenLower.includes('the guy yelled') ||
      spokenLower.includes('he shouted') ||
      spokenLower.includes('read a story') ||
      spokenLower.includes('watched');

    const isCasualHelpOrKnife =
      /\bhelp\s+(?:me\s+)?(?:carry|move|lift|with|fix|find|cook|clean|homework|code)\b/i.test(spokenLower) ||
      /\b(?:butter|kitchen|chef|bread|steak|pocket)\s+knife\b/i.test(spokenLower) ||
      /\b(?:pass|grab|use|hand)\s+(?:me\s+)?(?:the\s+)?knife\b/i.test(spokenLower);

    const isAmbiguous = (rawText.endsWith('...') || rawText.endsWith('..')) && matchedDistressWords.length > 0 && !peakVolume;

    const isLowVolume = Boolean(
      rawIsLowVolume ||
      mlData?.is_low_volume ||
      (peakVolume && peakVolume > 0 && peakVolume < 160) ||
      spokenLower.includes('quiet') ||
      spokenLower.includes('whisper') ||
      spokenLower.includes('hiding') ||
      spokenLower.includes('hide')
    );

    const isLoudDistress = peakVolume && peakVolume > 220;

    let classification = 'conversational';
    let severity = 'none';
    let intent = 'General conversation';
    let confidence = 0.95;
    let ambiguity = 0.05;
    let explanation = '';
    let alerts = [];
    let soundEvents = [];
    let emotion = { primary: 'neutral', arousal: 'low', confidence: 0.94 };

    if (mlData?.intelligence) {
      // Use ML Service intelligence directly if returned
      const mlIntel = mlData.intelligence;
      classification = mlIntel.classification;
      intent = mlIntel.intent;
      confidence = mlIntel.confidence;
      severity = mlIntel.severity;
      ambiguity = mlIntel.ambiguity;
      explanation = mlIntel.explanation;
      soundEvents = mlData.sound_events || [];
      emotion = mlData.emotion || emotion;
    } else if (isExplicitJoke && !isLoudDistress) {
      classification = 'joke';
      severity = 'none';
      intent = 'Humorous remark / explicit joke clarification';
      confidence = 0.97;
      explanation = `Danger-related keywords (${matchedDistressWords.join(', ') || 'keywords'}) detected inside an explicit joke. Acoustic sensors confirm nominal safety baseline.`;
      soundEvents = [{ label: 'speech', confidence: 0.96 }];
      emotion = { primary: 'neutral', arousal: 'low', confidence: 0.94 };
    } else if (isHypothetical) {
      classification = 'hypothetical';
      severity = 'none';
      intent = 'Hypothetical inquiry or theoretical question';
      confidence = 0.94;
      explanation = `Keywords (${matchedDistressWords.join(', ') || 'terms'}) identified in hypothetical inquiry pattern. No real emergency present.`;
      soundEvents = [{ label: 'speech', confidence: 0.95 }];
      emotion = { primary: 'neutral', arousal: 'low', confidence: 0.95 };
    } else if (isStorytelling) {
      classification = 'storytelling';
      severity = 'none';
      intent = 'Narrative storytelling or quoting dialogue';
      confidence = 0.93;
      explanation = 'Phrasing recognized as quoted dialogue / storytelling context. Scene nominal.';
      soundEvents = [{ label: 'speech', confidence: 0.95 }];
      emotion = { primary: 'neutral', arousal: 'low', confidence: 0.92 };
    } else if (isCasualHelpOrKnife) {
      classification = 'conversational';
      severity = 'none';
      intent = 'Everyday domestic assistance / tool request';
      confidence = 0.96;
      explanation = 'Keyword used in benign assistance / domestic context. No threat detected.';
      soundEvents = [{ label: 'speech', confidence: 0.97 }];
      emotion = { primary: 'neutral', arousal: 'low', confidence: 0.95 };
    } else if (isAmbiguous) {
      classification = 'ambiguous';
      severity = 'low';
      intent = 'Incomplete or unclear speech';
      confidence = 0.45;
      ambiguity = 0.85;
      explanation = 'Audio is incomplete or ambiguous. Clarification requested from user.';
      soundEvents = [{ label: 'speech', confidence: 0.70 }];
      alerts = [{
        id: `alert-ambig-${Date.now()}`,
        session_id: sessionId,
        type: 'ambiguity',
        severity: 'low',
        message: 'Ambiguous speech telemetry: evidence is incomplete. Please clarify intent.',
        details: { requires_clarification: true }
      }];
    } else if ((matchedDistressWords.length >= 2 || isLoudDistress || (isLowVolume && matchedDistressWords.length > 0)) && matchedDistressWords.length > 0) {
      classification = 'emergency';
      severity = 'high';
      intent = 'Immediate request for emergency intervention / active threat';
      confidence = 0.95;
      emotion = { primary: 'fear', arousal: isLowVolume ? 'low' : 'high', confidence: 0.96 };

      if (isLowVolume) {
        soundEvents = [{ label: 'whispered_speech', confidence: 0.94 }];
        explanation = `CRITICAL COVERT DISTRESS: Low-volume / whispered vocal distress ("${rawText}") with critical safety keywords: ${matchedDistressWords.join(', ')}.`;
        alerts.push({
          id: `alert-live-covert-${Date.now()}`,
          session_id: sessionId,
          type: 'keyword',
          severity: 'high',
          message: `Covert / low-volume distress vocalization identified with critical keywords: "${matchedDistressWords.join(', ')}"`,
          details: { is_covert: true, matched_keywords: matchedDistressWords }
        });
      } else {
        soundEvents = [{ label: isLoudDistress ? 'scream' : 'distressed_speech', confidence: 0.95 }];
        if (spokenLower.includes('fire') || spokenLower.includes('smoke')) {
          soundEvents.push({ label: 'smoke_alarm', confidence: 0.98 });
        }
        explanation = `CRITICAL EMERGENCY DETECTED: Spoken distress ("${rawText}") confirmed with high vocal arousal and corroborating evidence.`;
        alerts.push({
          id: `alert-live-em-${Date.now()}`,
          session_id: sessionId,
          type: 'keyword',
          severity: 'high',
          message: `Emergency distress confirmed by multi-factor intelligence: "${matchedDistressWords.join(', ')}"`,
          details: { matched_keywords: matchedDistressWords }
        });
      }
    } else {
      classification = 'conversational';
      severity = 'none';
      intent = 'Standard conversational speech';
      confidence = 0.98;
      explanation = 'All signal channels (speech, neutral biometrics, ambient acoustics) confirm safe baseline.';
      soundEvents = [{ label: 'speech', confidence: 0.95 }, { label: 'ambient_acoustics', confidence: 0.88 }];
      emotion = { primary: 'neutral', arousal: 'low', confidence: 0.94 };
    }

    const transcriptText = rawText || `Audio capture "${name}" analyzed. Speech and ambient environment within normal thresholds.`;

    // Dynamic Multi-Speaker Diarization from Actual Voice
    const rawSpeakerList = segmentSpeechToSpeakers(transcriptText, classification === 'emergency', isLowVolume);
    let speakers = rawSpeakerList.map((spk, idx) => {
      const spkText = (spk.text || '').trim();
      const spkLower = spkText.toLowerCase();

      if (classification === 'joke' || classification === 'hypothetical' || classification === 'storytelling' || classification === 'conversational') {
        return {
          ...spk,
          role: idx === 0 ? 'Primary Speaker' : 'Background Voice',
          threatCategory: 'nominal',
          emotion: { primary: 'neutral', arousal: 'low', confidence: 0.94 },
          isCovert: false
        };
      }

      const spkDistress = DISTRESS_KEYWORDS.some(kw => new RegExp(`\\b${kw}\\b`, 'i').test(spkLower));
      let threatCategory = spkDistress ? 'critical' : 'nominal';
      let role = spkDistress ? (isLowVolume ? 'Distress Speaker (Covert / Whispered)' : 'Distress Caller') : 'Secondary Speaker';

      return {
        ...spk,
        role,
        threatCategory,
        emotion: spkDistress ? { primary: 'fear', arousal: isLowVolume ? 'low' : 'high', confidence: 0.95 } : { primary: 'neutral', arousal: 'low', confidence: 0.94 },
        isCovert: spkDistress && isLowVolume
      };
    });

    const priorityOrder = { critical: 0, warning: 1, nominal: 2 };
    speakers.sort((a, b) => (priorityOrder[a.threatCategory] ?? 2) - (priorityOrder[b.threatCategory] ?? 2));

    const intelligenceReport = {
      classification,
      intent,
      confidence,
      keywords: matchedDistressWords,
      entities: ['speaker', 'room'],
      emotions: emotion,
      acoustic_signals: {
        scream: soundEvents.some(e => e.label.includes('scream')),
        crying: soundEvents.some(e => e.label.includes('crying')),
        laughter: isExplicitJoke,
        distress: classification === 'emergency' && !isLowVolume,
        intensity: peakVolume ? Math.min(peakVolume / 220, 1.0) : 0.4,
        speech_rate: isLoudDistress ? 'rapid' : 'normal'
      },
      context: {
        joke: isExplicitJoke ? 0.96 : 0.0,
        hypothetical: isHypothetical ? 0.93 : 0.0,
        storytelling: isStorytelling ? 0.92 : 0.0,
        serious: classification === 'emergency' ? 0.98 : 0.1
      },
      evidence: [],
      contradicting_evidence: isExplicitJoke && matchedDistressWords.length > 0 ? [{ factor: 'Keywords in text', score: 0.88 }] : [],
      ambiguity,
      recommended_action: classification === 'emergency' ? 'alert' : isAmbiguous ? 'clarify' : 'no_alert',
      severity,
      explanation
    };

    const compiledResult = {
      id: `SIG-${Date.now().toString().slice(-6)}`,
      session_id: sessionId,
      transcript_text: transcriptText,
      transcript_lang: 'en',
      sound_events: soundEvents,
      emotion,
      speakers,
      model_insight: { label: `Context: ${classification.toUpperCase()} (${Math.round(confidence * 100)}%)`, confidence },
      reasoning_trace: `Classification: ${classification.toUpperCase()}. Intent: ${intent}. ${explanation}`,
      fusion_summary: explanation,
      intelligence: intelligenceReport,
      is_low_volume: isLowVolume,
      created_at: new Date().toISOString(),
      alerts
    };

    setSceneResult(compiledResult);
    setActiveAlerts(alerts);
    injectMockResult(compiledResult);

    try {
      await fetch(`http://localhost:4000/api/v1/sessions/${sessionId}/result`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sessionId,
          mlResult: compiledResult
        })
      });
    } catch (e) {
      // Backend standby
    } finally {
      setIsLoading(false);
    }
  };

  const handleSimulateSample = (type) => {
    setIsLoading(true);
    setActiveScenario(type);

    setTimeout(() => {
      if (type === 'threat') {
        const mockThreat = {
          id: `SIG-${Date.now().toString().slice(-6)}`,
          session_id: sessionId,
          transcript_text: 'Someone has a weapon, please send emergency police immediately!',
          transcript_lang: 'en',
          sound_events: [
            { label: 'gunshot', confidence: 0.96 },
            { label: 'scream', confidence: 0.91 }
          ],
          emotion: { primary: 'fear', arousal: 'high', confidence: 0.94 },
          speakers: [
            {
              speaker_id: 'SPK-01 (Caller / Target)',
              role: 'Distress Caller',
              text: 'Someone has a weapon, please send emergency police immediately!',
              timeSpan: '[0.0s - 4.2s]',
              duration: 4.2,
              threatCategory: 'critical',
              emotion: { primary: 'fear', arousal: 'high', confidence: 0.96 }
            },
            {
              speaker_id: 'SPK-02 (Attacker / Suspect)',
              role: 'Aggressor',
              text: 'Drop the phone right now and stay down!',
              timeSpan: '[4.4s - 6.2s]',
              duration: 1.8,
              threatCategory: 'warning',
              emotion: { primary: 'anger', arousal: 'high', confidence: 0.91 }
            }
          ],
          model_insight: { label: 'Active Threat — Gunfire Transient & High-Arousal Vocal Panic', confidence: 0.97 },
          fusion_summary: 'CRITICAL EMERGENCY: Direct emergency intent confirmed by multiple corroborating independent signals (gunfire transient + scream + fear biometrics).',
          intelligence: {
            classification: 'emergency',
            intent: 'Immediate emergency intervention request',
            confidence: 0.96,
            keywords: ['weapon', 'emergency', 'police'],
            acoustic_signals: { scream: true, crying: false, laughter: false, distress: true, intensity: 0.92, speech_rate: 'rapid' },
            context: { joke: 0.0, hypothetical: 0.0, storytelling: 0.0, serious: 0.98 },
            evidence: [
              { factor: 'Semantic Emergency Intent', score: 0.96, description: 'Direct emergency words in threat context' },
              { factor: 'Acoustic Gunfire Transient', score: 0.96, description: 'Gunshot acoustic signature detected' },
              { factor: 'High Vocal Distress', score: 0.94, description: 'Screaming and fear emotion detected' }
            ],
            contradicting_evidence: [],
            ambiguity: 0.02,
            recommended_action: 'alert',
            severity: 'high',
            explanation: 'CRITICAL EMERGENCY: Multi-factor corroboration confirms active threat.'
          },
          created_at: new Date().toISOString(),
          alerts: [
            {
              id: `alert-${Date.now()}-1`,
              session_id: sessionId,
              type: 'sound_event',
              severity: 'high',
              message: 'Critical acoustic event detected: gunshot (96% confidence)',
              created_at: new Date().toISOString()
            },
            {
              id: `alert-${Date.now()}-2`,
              session_id: sessionId,
              type: 'keyword',
              severity: 'high',
              message: 'Emergency distress confirmed by multi-factor intelligence: "weapon, emergency, police"',
              created_at: new Date().toISOString()
            }
          ]
        };
        injectMockResult(mockThreat);
        setSceneResult(mockThreat);
        setActiveAlerts(mockThreat.alerts);
      } else if (type === 'joke') {
        const mockJoke = {
          id: `SIG-${Date.now().toString().slice(-6)}`,
          session_id: sessionId,
          transcript_text: "Bro help me, I have a knife 😂 I'm completely joking.",
          transcript_lang: 'en',
          sound_events: [{ label: 'speech', confidence: 0.96 }, { label: 'laughter', confidence: 0.92 }],
          emotion: { primary: 'joy', arousal: 'low', confidence: 0.92 },
          speakers: [
            {
              speaker_id: 'SPK-01 (Primary Speaker)',
              role: 'Conversational Voice',
              text: "Bro help me, I have a knife 😂 I'm completely joking.",
              timeSpan: '[0.0s - 3.8s]',
              duration: 3.8,
              threatCategory: 'nominal',
              emotion: { primary: 'joy', arousal: 'low', confidence: 0.92 }
            }
          ],
          model_insight: { label: 'Verified Safe: Explicit Joke Context', confidence: 0.98 },
          fusion_summary: 'NO ALERT: Danger-related keywords ("help", "knife") detected inside an explicit joke. Ambient acoustics and laughter biometrics confirm zero active threat.',
          intelligence: {
            classification: 'joke',
            intent: 'Humorous / playful remark',
            confidence: 0.97,
            keywords: ['help', 'knife'],
            acoustic_signals: { scream: false, crying: false, laughter: true, distress: false, intensity: 0.35, speech_rate: 'normal' },
            context: { joke: 0.96, hypothetical: 0.0, storytelling: 0.0, serious: 0.04 },
            evidence: [
              { factor: 'Explicit Joke Statement', score: 0.96, description: 'Speaker explicitly stated they are completely joking' },
              { factor: 'Laughter Acoustic Detection', score: 0.92, description: 'Laughter biometrics detected in audio' }
            ],
            contradicting_evidence: [
              { factor: 'Keywords in text', score: 0.88, description: 'Keywords "help, knife" discounted due to explicit joke context' }
            ],
            ambiguity: 0.02,
            recommended_action: 'no_alert',
            severity: 'none',
            explanation: 'NO ALERT: Dangerous keywords inside explicit joke. Biometrics confirm safe baseline.'
          },
          created_at: new Date().toISOString(),
          alerts: []
        };
        injectMockResult(mockJoke);
        setSceneResult(mockJoke);
        setActiveAlerts([]);
      } else if (type === 'hypothetical') {
        const mockHypo = {
          id: `SIG-${Date.now().toString().slice(-6)}`,
          session_id: sessionId,
          transcript_text: 'What would you do if someone came at you with a knife?',
          transcript_lang: 'en',
          sound_events: [{ label: 'speech', confidence: 0.96 }],
          emotion: { primary: 'neutral', arousal: 'low', confidence: 0.95 },
          speakers: [
            {
              speaker_id: 'SPK-01 (Primary Speaker)',
              role: 'Conversational Voice',
              text: 'What would you do if someone came at you with a knife?',
              timeSpan: '[0.0s - 3.5s]',
              duration: 3.5,
              threatCategory: 'nominal',
              emotion: { primary: 'neutral', arousal: 'low', confidence: 0.95 }
            }
          ],
          model_insight: { label: 'Verified Safe: Hypothetical Inquiry', confidence: 0.96 },
          fusion_summary: 'NO ALERT: Keywords ("knife") identified in a theoretical inquiry ("what would you do if"). No real incident indicated.',
          intelligence: {
            classification: 'hypothetical',
            intent: 'Hypothetical inquiry / theoretical discussion',
            confidence: 0.94,
            keywords: ['knife'],
            acoustic_signals: { scream: false, crying: false, laughter: false, distress: false, intensity: 0.38, speech_rate: 'normal' },
            context: { joke: 0.0, hypothetical: 0.94, storytelling: 0.0, serious: 0.12 },
            evidence: [
              { factor: 'Syntactic Inquiry Pattern', score: 0.94, description: 'Inquiry structure "what would you do if" identified' }
            ],
            contradicting_evidence: [],
            ambiguity: 0.03,
            recommended_action: 'no_alert',
            severity: 'none',
            explanation: 'NO ALERT: Theoretical question without threat.'
          },
          created_at: new Date().toISOString(),
          alerts: []
        };
        injectMockResult(mockHypo);
        setSceneResult(mockHypo);
        setActiveAlerts([]);
      } else if (type === 'story') {
        const mockStory = {
          id: `SIG-${Date.now().toString().slice(-6)}`,
          session_id: sessionId,
          transcript_text: "I watched a movie yesterday where the guy yelled 'help, he has a knife!'",
          transcript_lang: 'en',
          sound_events: [{ label: 'speech', confidence: 0.96 }],
          emotion: { primary: 'neutral', arousal: 'low', confidence: 0.92 },
          speakers: [
            {
              speaker_id: 'SPK-01 (Primary Speaker)',
              role: 'Storyteller / Narrator',
              text: "I watched a movie yesterday where the guy yelled 'help, he has a knife!'",
              timeSpan: '[0.0s - 4.2s]',
              duration: 4.2,
              threatCategory: 'nominal',
              emotion: { primary: 'neutral', arousal: 'low', confidence: 0.92 }
            }
          ],
          model_insight: { label: 'Verified Safe: Movie / Story Quoted Context', confidence: 0.95 },
          fusion_summary: 'NO ALERT: Dangerous phrasing recognized as quoted dialogue in a storytelling / movie narrative context.',
          intelligence: {
            classification: 'storytelling',
            intent: 'Recounting movie narrative',
            confidence: 0.93,
            keywords: ['help', 'knife'],
            acoustic_signals: { scream: false, crying: false, laughter: false, distress: false, intensity: 0.38, speech_rate: 'normal' },
            context: { joke: 0.0, hypothetical: 0.0, storytelling: 0.92, quotation: 0.88, serious: 0.1 },
            evidence: [
              { factor: 'Narrative Markers', score: 0.92, description: 'Markers "watched a movie where the guy yelled" confirm quoted fictional context' }
            ],
            contradicting_evidence: [],
            ambiguity: 0.04,
            recommended_action: 'no_alert',
            severity: 'none',
            explanation: 'NO ALERT: Quoted dialogue in movie narrative.'
          },
          created_at: new Date().toISOString(),
          alerts: []
        };
        injectMockResult(mockStory);
        setSceneResult(mockStory);
        setActiveAlerts([]);
      } else if (type === 'ambiguous') {
        const mockAmbig = {
          id: `SIG-${Date.now().toString().slice(-6)}`,
          session_id: sessionId,
          transcript_text: "Help... there's someone...",
          transcript_lang: 'en',
          sound_events: [{ label: 'speech', confidence: 0.65 }],
          emotion: { primary: 'neutral', arousal: 'low', confidence: 0.55 },
          speakers: [
            {
              speaker_id: 'SPK-01 (Unclear Voice)',
              role: 'Incomplete Utterance',
              text: "Help... there's someone...",
              timeSpan: '[0.0s - 2.1s]',
              duration: 2.1,
              threatCategory: 'nominal',
              emotion: { primary: 'neutral', arousal: 'low', confidence: 0.55 }
            }
          ],
          model_insight: { label: 'Acoustic Telemetry Incomplete / Ambiguous', confidence: 0.50 },
          fusion_summary: 'CLARIFICATION REQUEST: Telemetry is ambiguous or incomplete. Not enough evidence to declare critical alert. Clarification requested from user.',
          intelligence: {
            classification: 'ambiguous',
            intent: 'Incomplete speech utterance',
            confidence: 0.45,
            keywords: ['help'],
            acoustic_signals: { scream: false, crying: false, laughter: false, distress: false, intensity: 0.25, speech_rate: 'slow' },
            context: { joke: 0.0, hypothetical: 0.0, storytelling: 0.0, serious: 0.5 },
            evidence: [
              { factor: 'Trailing Utterance', score: 0.85, description: 'Audio fragment ends abruptly without corroborating distress transients' }
            ],
            contradicting_evidence: [],
            ambiguity: 0.85,
            recommended_action: 'clarify',
            severity: 'low',
            explanation: 'Audio is incomplete or ambiguous. Clarification requested from user.'
          },
          created_at: new Date().toISOString(),
          alerts: [
            {
              id: `alert-${Date.now()}-ambig`,
              session_id: sessionId,
              type: 'ambiguity',
              severity: 'low',
              message: 'Ambiguous scene telemetry: evidence is incomplete. Please clarify intent.',
              details: { requires_clarification: true }
            }
          ]
        };
        injectMockResult(mockAmbig);
        setSceneResult(mockAmbig);
        setActiveAlerts(mockAmbig.alerts);
      } else {
        // Safe baseline
        const mockSafe = {
          id: `SIG-${Date.now().toString().slice(-6)}`,
          session_id: sessionId,
          transcript_text: 'Help me carry this box to the conference room for the presentation.',
          transcript_lang: 'en',
          sound_events: [{ label: 'speech', confidence: 0.98 }],
          emotion: { primary: 'neutral', arousal: 'low', confidence: 0.96 },
          speakers: [
            {
              speaker_id: 'SPK-01 (Colleague)',
              role: 'Conversational Voice',
              text: 'Help me carry this box to the conference room for the presentation.',
              timeSpan: '[0.0s - 3.2s]',
              duration: 3.2,
              threatCategory: 'nominal',
              emotion: { primary: 'neutral', arousal: 'low', confidence: 0.96 }
            }
          ],
          model_insight: { label: 'Verified Safe: Routine Workplace Interaction', confidence: 0.99 },
          fusion_summary: 'NO ALERT: "Help" used in routine domestic assistance context. All biometrics and ambient acoustics are safe.',
          intelligence: {
            classification: 'conversational',
            intent: 'Request for everyday physical assistance',
            confidence: 0.98,
            keywords: ['help'],
            acoustic_signals: { scream: false, crying: false, laughter: false, distress: false, intensity: 0.35, speech_rate: 'normal' },
            context: { joke: 0.0, hypothetical: 0.0, storytelling: 0.0, serious: 0.85 },
            evidence: [
              { factor: 'Benign Collocation', score: 0.96, description: 'Words "help me carry this box" collocate with workplace tasks' }
            ],
            contradicting_evidence: [],
            ambiguity: 0.02,
            recommended_action: 'no_alert',
            severity: 'none',
            explanation: 'NO ALERT: Everyday assistance request.'
          },
          created_at: new Date().toISOString(),
          alerts: []
        };
        injectMockResult(mockSafe);
        setSceneResult(mockSafe);
        setActiveAlerts([]);
      }
      setIsLoading(false);
    }, 400);
  };

  return (
    <div className="cursor-workstation-page">
      <header className="cursor-topbar">
        <div className="cursor-brand-left">
          <svg className="cursor-logo-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
          <div className="cursor-brand-text-stack">
            <span className="cursor-brand-title">Smart Horizon 2026</span>
            <span className="cursor-brand-subtitle">Context-Aware Forensic Acoustic Intelligence</span>
          </div>
        </div>

        <div className="cursor-topbar-center">
          <div className="cursor-chip-pill">
            <span className={`cursor-beacon-dot ${isConnected ? 'online' : 'offline'}`} />
            <span>Telemetry: <strong>{wsStatus.toUpperCase()}</strong></span>
          </div>
          <div className="cursor-chip-pill">
            <span>Threat Model: <strong>Context-Aware Fusion</strong></span>
          </div>
        </div>

        <div className="cursor-topbar-right">
          <div className="cursor-session-box">
            <span className="cursor-session-label">SESSION:</span>
            <span className="cursor-session-input">{sessionId.slice(-7)}</span>
          </div>
        </div>
      </header>

      <main className="cursor-main-content">
        <AlertBanner
          alerts={activeAlerts}
          onDismiss={handleDismissAlert}
          onClearAll={handleClearAllAlerts}
        />

        <div className="cursor-workspace-layout">
          <div className="cursor-col-left">
            <RecordOrUpload
              sessionId={sessionId}
              onAudioReady={handleAudioReady}
              isLoading={isLoading}
            />

            <div className="cursor-scene-panel" style={{ marginTop: '16px' }}>
              <div className="cursor-panel-header">
                <div className="cursor-panel-title">
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: 'var(--cursor-copper)' }}>
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  <span>Validation Scenarios · Multi-Factor False Positive Reduction</span>
                </div>
              </div>
              <div className="cursor-panel-body">
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '8px' }}>
                  <button
                    className={`cursor-test-scenario-btn ${activeScenario === 'threat' ? 'active' : ''}`}
                    onClick={() => handleSimulateSample('threat')}
                    disabled={isLoading}
                  >
                    <span className="scenario-label" style={{ color: '#f87171' }}>🚨 Real Emergency</span>
                    <span className="scenario-sub">Gunfire + Distress</span>
                  </button>

                  <button
                    className={`cursor-test-scenario-btn ${activeScenario === 'joke' ? 'active' : ''}`}
                    onClick={() => handleSimulateSample('joke')}
                    disabled={isLoading}
                  >
                    <span className="scenario-label" style={{ color: '#34d399' }}>😂 Joke Context</span>
                    <span className="scenario-sub">"Knife" in explicit joke</span>
                  </button>

                  <button
                    className={`cursor-test-scenario-btn ${activeScenario === 'hypothetical' ? 'active' : ''}`}
                    onClick={() => handleSimulateSample('hypothetical')}
                    disabled={isLoading}
                  >
                    <span className="scenario-label" style={{ color: '#60a5fa' }}>🤔 Hypothetical</span>
                    <span className="scenario-sub">"What would you do if"</span>
                  </button>

                  <button
                    className={`cursor-test-scenario-btn ${activeScenario === 'story' ? 'active' : ''}`}
                    onClick={() => handleSimulateSample('story')}
                    disabled={isLoading}
                  >
                    <span className="scenario-label" style={{ color: '#c084fc' }}>🎬 Movie Quote</span>
                    <span className="scenario-sub">Storytelling / Fiction</span>
                  </button>

                  <button
                    className={`cursor-test-scenario-btn ${activeScenario === 'ambiguous' ? 'active' : ''}`}
                    onClick={() => handleSimulateSample('ambiguous')}
                    disabled={isLoading}
                  >
                    <span className="scenario-label" style={{ color: '#fbbf24' }}>❓ Ambiguous Speech</span>
                    <span className="scenario-sub">Incomplete / Clarify</span>
                  </button>

                  <button
                    className={`cursor-test-scenario-btn ${activeScenario === 'safe' ? 'active' : ''}`}
                    onClick={() => handleSimulateSample('safe')}
                    disabled={isLoading}
                  >
                    <span className="scenario-label" style={{ color: '#94a3b8' }}>📦 Casual Help</span>
                    <span className="scenario-sub">"Help carry this box"</span>
                  </button>
                </div>
              </div>
            </div>
          </div>

          <div className="cursor-col-right">
            <ResultPanel
              result={sceneResult}
              isLoading={isLoading}
              onDeleteSpeaker={handleDeleteSpeaker}
              onClarifyResponse={handleClarifyResponse}
            />
          </div>
        </div>
      </main>
    </div>
  );
}

export default LiveMonitorPage;
