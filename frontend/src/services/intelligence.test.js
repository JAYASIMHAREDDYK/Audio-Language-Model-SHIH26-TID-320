import test from 'node:test';
import assert from 'node:assert';
import { evaluateThreat } from './fusionSynthesis.js';
import { validateSpeakerSegments, segmentSpokenUtterance } from './diarizationValidator.js';
import { extractAcousticFeatures, getPitchCategory } from './audioAnalyzer.js';
import { getSeededJokeSession, getEmergencySession, getHypotheticalSession } from './sessionStore.js';

test('1. Multimodal Fusion Synthesis - Seeded Joke Regression Test Case', () => {
  const transcript = "hi I am just playing with you, I have this weird thought where my friend is drowning and asking for help but I'm looking at her smiling — am I cruel";
  const vocal_biometrics = { emotion: 'neutral', arousal: 'low', confidence: 0.94 };
  const acoustic_events = [];

  const verdict = evaluateThreat({ transcript, vocal_biometrics, acoustic_events });

  assert.strictEqual(verdict.threat_level, 'NOMINAL', 'Should evaluate explicit joke with neutral tone as NOMINAL');
  assert.strictEqual(verdict.confidence, 97, 'Expected 97% confidence on explicit joke regression case');
  assert.ok(verdict.fusion_rationale.includes('help'), 'Rationale must explicitly note that "help" keyword was discounted');
  assert.ok(verdict.fusion_rationale.includes('NEUTRAL'), 'Rationale must cite neutral vocal biometrics');
  assert.ok(verdict.channels_discounted.includes('transcript_keywords'), 'Should mark transcript keywords as discounted');
  assert.strictEqual(verdict.semantic_analysis.joke_detected, true);
});

test('2. Multimodal Fusion Synthesis - Negation Context Reasoning', () => {
  const transcript = "I don't have a knife and there is no weapon here, everything is safe.";
  const vocal_biometrics = { emotion: 'neutral', arousal: 'low', confidence: 0.95 };
  const acoustic_events = [];

  const verdict = evaluateThreat({ transcript, vocal_biometrics, acoustic_events });

  assert.strictEqual(verdict.threat_level, 'NOMINAL', 'Negation of weapon keyword must be NOMINAL');
  assert.strictEqual(verdict.semantic_analysis.negation_detected, true, 'Must detect semantic negation');
  assert.ok(verdict.confidence_metrics.classification_confidence >= 95, 'High classification confidence on negation');
});

test('3. Multimodal Fusion Synthesis - Hypothetical Inquiry Context', () => {
  const transcript = "What would happen if someone came into the building with a weapon?";
  const vocal_biometrics = { emotion: 'neutral', arousal: 'low', confidence: 0.92 };
  const acoustic_events = [];

  const verdict = evaluateThreat({ transcript, vocal_biometrics, acoustic_events });

  assert.strictEqual(verdict.threat_level, 'NOMINAL', 'Hypothetical inquiry must not trigger critical');
  assert.strictEqual(verdict.semantic_analysis.modality, 'Hypothetical Inquiry');
});

test('4. Multimodal Fusion Synthesis - De-escalation Overridden by Vocal Stress & Acoustic Event', () => {
  const transcript = "Haha it is just a joke, don't worry bro";
  const vocal_biometrics = { emotion: 'fear', arousal: 'high', confidence: 0.95 };
  const acoustic_events = [{ type: 'gunshot', confidence: 0.96 }];

  const verdict = evaluateThreat({ transcript, vocal_biometrics, acoustic_events });

  assert.strictEqual(verdict.threat_level, 'CRITICAL', 'Should override joke text when contradicted by fear and gunfire');
  assert.ok(verdict.channels_discounted.some(c => c.includes('transcript')), 'Transcript should be discounted');
  assert.ok(verdict.fusion_rationale.includes('overridden') || verdict.fusion_rationale.includes('contradicted'), 'Rationale must state text was overridden');
});

test('5. Speaker Diarization Validator - Mid-Sentence Pause Merged into Single Speaker', () => {
  const rawSegments = [
    {
      start: 0.0,
      end: 3.8,
      text: "hi I am just playing with you, I have this weird thought",
      pitch: 165.0,
      speaker_id: 'Speaker 1'
    },
    {
      start: 4.1, // 0.3s pause gap
      end: 8.4,
      text: "where my friend is drowning and asking for help but I'm looking at her smiling — am I cruel",
      pitch: 172.0,
      speaker_id: 'Speaker 1'
    }
  ];

  const diarization = validateSpeakerSegments(rawSegments);

  assert.strictEqual(diarization.speaker_segments.length, 1, 'Must NOT falsely split single speaker continuous sentence');
  assert.strictEqual(diarization.speaker_segments[0].start, 0.0);
  assert.strictEqual(diarization.speaker_segments[0].end, 8.4);
  assert.ok(diarization.merge_decisions.some(d => d.decision === 'merged'), 'Must document merge decision');
});

test('6. Speaker Diarization - Multi-Speaker Turn Dialog Segmentation', () => {
  const dialogText = "Did you hear that? Yeah, someone is outside. Don't worry, it was probably nothing.";
  const rawSegments = segmentSpokenUtterance(dialogText, 6.0);
  const diarization = validateSpeakerSegments(rawSegments);

  assert.ok(diarization.speaker_segments.length >= 2, 'Must separate multiple speaker turns in conversational dialog');
  assert.strictEqual(diarization.speaker_segments[0].speaker_id, 'Speaker 1');
});

test('7. Acoustic Analysis - Pitch F0 & Low Tone Independence from Threat Severity', () => {
  const pitchInfoLow = getPitchCategory(92);
  assert.strictEqual(pitchInfoLow.label, 'Low Pitch');

  const acoustics = extractAcousticFeatures({ duration: 4.0, wordCount: 12, measuredPitch: 92, measuredRms: 0.32 });
  assert.strictEqual(acoustics.pitch_f0, 92);
  assert.strictEqual(acoustics.pitch_label, 'Low Pitch');
  assert.ok(acoustics.speech_rate_wpm > 0);
  assert.ok(acoustics.snr_db > 0);
});

test('8. Session Library - Seeded Benchmark Session Validity', () => {
  const seeded = getSeededJokeSession();
  assert.strictEqual(seeded.result.threat_level, 'NOMINAL');
  assert.strictEqual(seeded.result.diarization.speaker_segments.length, 1);
  assert.strictEqual(seeded.result.fusion.confidence, 97);
  assert.ok(seeded.result.acoustic_features.pitch_f0 > 0);
});
