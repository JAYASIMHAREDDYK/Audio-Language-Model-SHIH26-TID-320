import { evaluateThreat } from './fusionSynthesis.js';
import { validateSpeakerSegments } from './diarizationValidator.js';
import { extractAcousticFeatures } from './audioAnalyzer.js';

/**
 * Session Storage & Scenario Library Service
 * 
 * In-memory store per sign-in with persistence seams (saveSessions / loadSessions)
 * ready for real API / database drop-in.
 */

const STORAGE_KEY = 'smart_horizon_sessions_v1';

/**
 * Seeded Regression Test Case
 * Validates:
 * 1. "help" keyword is discounted due to explicit joke intent & neutral biometrics.
 * 2. Pause in middle of sentence does not falsely split into 2 speakers.
 */
export function getSeededJokeSession() {
  const transcript = "hi I am just playing with you, I have this weird thought where my friend is drowning and asking for help but I'm looking at her smiling — am I cruel";
  
  // Raw candidate speech segments across mid-sentence pause
  const rawSegments = [
    {
      start: 0.0,
      end: 3.8,
      text: "hi I am just playing with you, I have this weird thought",
      pitch: 165.0,
      energy: 0.45,
      speaker_id: 'Speaker 1'
    },
    {
      start: 4.1, // 0.3s pause gap
      end: 8.4,
      text: "where my friend is drowning and asking for help but I'm looking at her smiling — am I cruel",
      pitch: 172.0,
      energy: 0.48,
      speaker_id: 'Speaker 1'
    }
  ];

  const vocalBiometrics = {
    emotion: 'neutral',
    arousal: 'low',
    confidence: 0.94
  };

  const acousticEvents = []; // zero acoustic scream/gunfire

  const acousticFeatures = extractAcousticFeatures({
    duration: 8.4,
    wordCount: 26,
    measuredPitch: 142,
    measuredRms: 0.38
  });

  const fusedVerdict = evaluateThreat({
    transcript,
    vocal_biometrics: vocalBiometrics,
    acoustic_events: acousticEvents,
    acoustic_features: acousticFeatures
  });

  const diarization = validateSpeakerSegments(rawSegments);

  return {
    id: 'sess_seeded_01',
    label: 'Forensic Audio Session · Regression Benchmark',
    createdAt: new Date(Date.now() - 1000 * 60 * 12).toISOString(),
    capture: {
      transcript,
      duration: 8.4,
      vocalBiometrics,
      acousticEvents,
      rawSegments
    },
    result: {
      answer: "The recording was taken in a quiet indoor room ambient environment.",
      confidence: 0.94,
      fusion: fusedVerdict,
      diarization: diarization,
      acoustic_features: acousticFeatures,
      threat_level: fusedVerdict.threat_level,
      speech_evidence: `Transcript: "${transcript}" | Language: en`,
      non_speech_evidence: "Detected Events: None",
      speaker_evidence: "1 speaker turn(s) active from 0.0s to 8.4s",
      paralinguistic_evidence: "Emotion: neutral | Arousal: low",
      analyzedAt: new Date().toISOString()
    }
  };
}

/**
 * Real Emergency Sample Session
 */
export function getEmergencySession() {
  const transcript = "Someone has a weapon, please send emergency police immediately!";
  const rawSegments = [
    {
      start: 0.0,
      end: 3.5,
      text: "Someone has a weapon, please send emergency police immediately!",
      speaker_id: 'Speaker 1',
      embedding_similarity_to_prev: 1.0
    },
    {
      start: 4.0,
      end: 6.2,
      text: "Drop the phone right now and step back!",
      speaker_id: 'Speaker 2',
      embedding_similarity_to_prev: 0.32 // Low voiceprint similarity + command turn-take
    }
  ];

  const vocalBiometrics = {
    emotion: 'fear',
    arousal: 'high',
    confidence: 0.96
  };

  const acousticEvents = [
    { type: 'gunshot', confidence: 0.96 },
    { type: 'scream', confidence: 0.91 }
  ];

  const acousticFeatures = extractAcousticFeatures({
    duration: 6.2,
    wordCount: 17,
    measuredPitch: 248,
    measuredRms: 0.88
  });

  const fusedVerdict = evaluateThreat({
    transcript,
    vocal_biometrics: vocalBiometrics,
    acoustic_events: acousticEvents,
    acoustic_features: acousticFeatures
  });

  const diarization = validateSpeakerSegments(rawSegments);

  return {
    id: 'sess_emergency_02',
    label: 'Incident Log · Active Threat Gunfire Dispatch',
    createdAt: new Date(Date.now() - 1000 * 60 * 45).toISOString(),
    capture: {
      transcript,
      duration: 6.2,
      vocalBiometrics,
      acousticEvents,
      rawSegments
    },
    result: {
      answer: "The combination of gunshot and scream sound events with urgent speech indicates an active threat emergency environment.",
      confidence: 0.96,
      fusion: fusedVerdict,
      diarization: diarization,
      acoustic_features: acousticFeatures,
      threat_level: fusedVerdict.threat_level,
      speech_evidence: `Transcript: "${transcript}" | Language: en`,
      non_speech_evidence: "Detected Events: gunshot, scream",
      speaker_evidence: "2 speaker turn(s) active from 0.0s to 6.2s",
      paralinguistic_evidence: "Emotion: fear | Arousal: high",
      analyzedAt: new Date().toISOString()
    }
  };
}

/**
 * Hypothetical Inquiry Session
 */
export function getHypotheticalSession() {
  const transcript = "What would you do if someone came into the building with a weapon?";
  const rawSegments = [
    {
      start: 0.0,
      end: 4.2,
      text: "What would you do if someone came into the building with a weapon?",
      speaker_id: 'Speaker 1'
    }
  ];

  const vocalBiometrics = {
    emotion: 'neutral',
    arousal: 'low',
    confidence: 0.92
  };

  const acousticEvents = [];

  const acousticFeatures = extractAcousticFeatures({
    duration: 4.2,
    wordCount: 13,
    measuredPitch: 128,
    measuredRms: 0.34
  });

  const fusedVerdict = evaluateThreat({
    transcript,
    vocal_biometrics: vocalBiometrics,
    acoustic_events: acousticEvents,
    acoustic_features: acousticFeatures
  });

  const diarization = validateSpeakerSegments(rawSegments);

  return {
    id: 'sess_hypothetical_03',
    label: 'Inquiry Log · Facility Security Discussion',
    createdAt: new Date(Date.now() - 1000 * 60 * 180).toISOString(),
    capture: {
      transcript,
      duration: 4.2,
      vocalBiometrics,
      acousticEvents,
      rawSegments
    },
    result: {
      answer: "The speaker is inquiring about security procedures in an indoor office environment.",
      confidence: 0.92,
      fusion: fusedVerdict,
      diarization: diarization,
      acoustic_features: acousticFeatures,
      threat_level: fusedVerdict.threat_level,
      speech_evidence: `Transcript: "${transcript}" | Language: en`,
      non_speech_evidence: "Detected Events: None",
      speaker_evidence: "1 speaker turn(s) active from 0.0s to 4.2s",
      paralinguistic_evidence: "Emotion: neutral | Arousal: low",
      analyzedAt: new Date().toISOString()
    }
  };
}

/**
 * Creates initial in-memory sessions list
 */
export function createInitialSessions() {
  return [
    getSeededJokeSession(),
    getEmergencySession(),
    getHypotheticalSession()
  ];
}

/**
 * LocalStorage persistence seam
 */
export function saveSessions(sessions) {
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    }
  } catch (err) {
    console.warn('[SessionStore] Could not persist sessions to localStorage', err);
  }
}

/**
 * Load persisted sessions if available
 */
export function loadSessions() {
  try {
    if (typeof window !== 'undefined' && window.localStorage) {
      const data = localStorage.getItem(STORAGE_KEY);
      if (data) {
        return JSON.parse(data);
      }
    }
  } catch (err) {
    console.warn('[SessionStore] Could not load persisted sessions', err);
  }
  return createInitialSessions();
}
