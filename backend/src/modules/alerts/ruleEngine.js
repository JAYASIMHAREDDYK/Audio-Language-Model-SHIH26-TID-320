const CRITICAL_SOUND_EVENTS = [
  'gunshot',
  'explosion',
  'smoke_alarm',
  'glass_breaking',
  'scream'
];

const DISTRESS_KEYWORDS = [
  'help',
  'emergency',
  'fire',
  'police',
  'danger',
  'sos',
  'save me',
  'attacker',
  'gun',
  'shooting',
  'kill',
  'bomb',
  'intruder',
  'hostage',
  'threat'
];

const HIGH_RISK_EMOTIONS = ['fear', 'anger'];

function normalizeSoundEvents(soundEvents) {
  if (!soundEvents) return [];
  if (typeof soundEvents === 'string') {
    return [{ label: soundEvents.toLowerCase().trim(), confidence: 1.0 }];
  }
  if (!Array.isArray(soundEvents)) return [];

  return soundEvents
    .map((event) => {
      if (typeof event === 'string') {
        return { label: event.toLowerCase().trim(), confidence: 1.0 };
      }
      if (event && typeof event === 'object') {
        const label = (event.label || event.name || event.event || event.tag || '').toLowerCase().trim();
        const confidence = typeof event.confidence === 'number' ? event.confidence :
                           typeof event.score === 'number' ? event.score : 1.0;
        return { label, confidence, raw: event };
      }
      return { label: '', confidence: 0 };
    })
    .filter((e) => e.label.length > 0);
}

function normalizeEmotion(emotion) {
  if (!emotion || typeof emotion !== 'object') {
    return { primary: '', arousal: '', confidence: 0 };
  }
  const primary = (emotion.primary || emotion.label || emotion.emotion || emotion.dominant || '').toLowerCase().trim();
  const arousal = (emotion.arousal || emotion.arousal_level || '').toLowerCase().trim();
  const confidence = typeof emotion.confidence === 'number' ? emotion.confidence :
                     typeof emotion.score === 'number' ? emotion.score :
                     typeof emotion.probability === 'number' ? emotion.probability : 0;
  return { primary, arousal, confidence };
}

function extractTranscriptText(sceneResult) {
  if (!sceneResult) return '';
  if (typeof sceneResult.transcript === 'string') return sceneResult.transcript;
  if (sceneResult.transcript && typeof sceneResult.transcript.text === 'string') return sceneResult.transcript.text;
  if (typeof sceneResult.transcript_text === 'string') return sceneResult.transcript_text;
  if (typeof sceneResult.text === 'string') return sceneResult.text;
  return '';
}

function evaluate(sceneResult = {}) {
  const alerts = [];
  if (!sceneResult || typeof sceneResult !== 'object') {
    return alerts;
  }

  const soundEvents = normalizeSoundEvents(sceneResult.sound_events || sceneResult.soundEvents);
  const emotion = normalizeEmotion(sceneResult.emotion);
  const transcriptText = extractTranscriptText(sceneResult);
  const modelInsight = sceneResult.model_insight || sceneResult.modelInsight || null;

  const matchedSounds = soundEvents.filter((ev) =>
    CRITICAL_SOUND_EVENTS.some((critical) => ev.label.includes(critical) || critical.includes(ev.label))
  );

  if (matchedSounds.length > 0) {
    matchedSounds.forEach((sound) => {
      alerts.push({
        type: 'sound_event',
        severity: 'high',
        message: `Critical acoustic event detected: ${sound.label.replace(/_/g, ' ')} (confidence: ${Math.round(sound.confidence * 100)}%)`,
        details: {
          matched_sound: sound.label,
          confidence: sound.confidence
        }
      });
    });
  }

  const isHighRiskEmotion = HIGH_RISK_EMOTIONS.includes(emotion.primary);
  const isHighArousal = emotion.arousal === 'high' || emotion.arousal === 'extreme' || emotion.arousal === 'elevated';
  const isHighConfidence = emotion.confidence > 0.7;

  if (isHighRiskEmotion && isHighArousal && isHighConfidence) {
    alerts.push({
      type: 'emotion',
      severity: 'medium',
      message: `Distress emotion detected: ${emotion.primary.toUpperCase()} with ${emotion.arousal} arousal (confidence: ${Math.round(emotion.confidence * 100)}%)`,
      details: {
        primary: emotion.primary,
        arousal: emotion.arousal,
        confidence: emotion.confidence
      }
    });
  }

  if (transcriptText) {
    const lowerText = transcriptText.toLowerCase();
    const matchedKeywords = DISTRESS_KEYWORDS.filter((kw) => {
      const regex = new RegExp(`\\b${kw}\\b`, 'i');
      return regex.test(lowerText);
    });

    if (matchedKeywords.length > 0) {
      alerts.push({
        type: 'keyword',
        severity: 'medium',
        message: `Distress keywords identified in transcript: "${matchedKeywords.join(', ')}"`,
        details: {
          matched_keywords: matchedKeywords,
          snippet: transcriptText.slice(0, 150)
        }
      });
    }
  }

  if (modelInsight && typeof modelInsight === 'object') {
    const label = (modelInsight.label || '').toLowerCase();
    const confidence = typeof modelInsight.confidence === 'number' ? modelInsight.confidence : 0;

    if (
      (label.includes('threat') || label.includes('danger') || label.includes('emergency') || label.includes('critical')) &&
      confidence >= 0.8
    ) {
      if (alerts.length === 0) {
        alerts.push({
          type: 'sound_event',
          severity: 'high',
          message: `Model insight alert: ${modelInsight.label} (confidence: ${Math.round(confidence * 100)}%)`,
          details: modelInsight
        });
      }
    }
  }

  return alerts;
}

module.exports = {
  evaluate,
  CRITICAL_SOUND_EVENTS,
  DISTRESS_KEYWORDS,
  HIGH_RISK_EMOTIONS,
  normalizeSoundEvents,
  normalizeEmotion
};

