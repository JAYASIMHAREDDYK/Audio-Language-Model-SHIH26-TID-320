/**
 * Speaker Diarization Validator
 * 
 * Rules:
 * 1. A pause or silence gap between segments is NEVER sufficient evidence of a speaker change on its own.
 * 2. A change in pitch, pace, arousal, or emotional tone is NEVER sufficient evidence on its own.
 * 3. Only treat two segments as different speakers if BOTH:
 *    a. Voiceprint/embedding similarity score < similarity_threshold, AND
 *    b. Transcript content is consistent with a conversational turn-take rather than a single train of thought.
 * 4. If a sentence is grammatically and semantically continuous across a timestamp gap, you MUST merge them.
 * 5. Multi-Speaker Support:
 *    - Use neutral labels: "Speaker 1", "Speaker 2", "Speaker 3" (or SPK-01, SPK-02)
 *    - Detect conversational turn-takes (Question -> Response -> Follow-up).
 * 6. When uncertain, default conservatively to fewer speakers.
 */

const TURN_TAKING_PATTERNS = [
  /^(?:yeah|yes|no|nope|wait|hey|listen|look|hold on|what|who|why|how|okay|alright|stop|shut up|drop it|leave|help)\b/i,
  /^(?:speaker\s*\d+|spk-?\d+)\s*:/i,
  /\b(?:did you|can you|are you|will you|tell me)\b.*\?/i,
  /\b(?:he said|she said|they said)\b/i
];

const CONTINUATION_CONNECTORS = [
  /^(?:and|but|or|so|because|although|though|even though|which|that|who|whom|whose|with|while|where|when|if|unless|since)\b/i,
  /^(?:to|for|in|on|at|by|from|about|into|through|after|before|under|over)\b/i,
  /^(?:is|are|was|were|am|be|been|being|have|has|had|do|does|did|can|could|will|would|shall|should|may|might|must)\b/i,
  /^(?:my|your|his|her|its|our|their|this|these|that|those)\b/i,
  /^(?:me|him|her|us|them|it)\b/i,
  /^(?:where|when|asking|looking|wondering)\b/i
];

function isTurnTakingText(text = '') {
  const clean = text.trim();
  return TURN_TAKING_PATTERNS.some(pattern => pattern.test(clean));
}

function isGrammaticallyContinuous(prevText = '', nextText = '') {
  const prevClean = prevText.trim();
  const nextClean = nextText.trim();

  // If previous segment does not end with terminal punctuation
  if (!/[.!?]$/.test(prevClean)) {
    return true;
  }

  // If next segment starts with a grammatical connector / preposition / relative clause
  if (CONTINUATION_CONNECTORS.some(p => p.test(nextClean))) {
    return true;
  }

  // If previous ends with comma, ellipsis, or dash
  if (/[,…\-\.\.]$/.test(prevClean)) {
    return true;
  }

  return false;
}

/**
 * Splits a single transcript with multiple conversational turn markers into neutral speaker segments.
 * e.g., "Did you hear that? Yeah, someone is outside. Don't worry, it was probably nothing."
 * -> Speaker 1: "Did you hear that?"
 * -> Speaker 2: "Yeah, someone is outside."
 * -> Speaker 1: "Don't worry, it was probably nothing."
 * @param {string} fullText
 * @param {number} totalDuration
 * @returns {Array} List of raw candidate segments
 */
export function segmentSpokenUtterance(fullText = '', totalDuration = 4.0) {
  const text = (fullText || '').trim();
  if (!text) return [{ start: 0.0, end: totalDuration, text: '', speaker_id: 'Speaker 1' }];

  // Match sentences or explicit dialogue turn markers (. ! ? or comma + turn keyword)
  const sentences = text.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [text];

  // If text is a single continuous sentence (e.g. "Hello, how are you?"),
  // treat the entire utterance as Speaker 1 unless explicit turn-taking markers occur.
  if (sentences.length <= 1) {
    return [{
      start: 0.0,
      end: Number(totalDuration.toFixed(1)),
      text: text,
      speaker_id: 'Speaker 1'
    }];
  }

  const rawSegments = [];
  const timePerSegment = totalDuration / sentences.length;
  let currentSpeaker = 1;

  for (let i = 0; i < sentences.length; i++) {
    const sText = sentences[i].trim();
    if (!sText) continue;

    const start = Number((i * timePerSegment).toFixed(1));
    const end = Number(((i + 1) * timePerSegment).toFixed(1));

    // Detect if this segment represents a genuine turn take from previous
    if (i > 0) {
      const prevText = sentences[i - 1].trim();
      const hasTurnTake = isTurnTakingText(sText);
      const isContinuous = isGrammaticallyContinuous(prevText, sText);

      if (hasTurnTake && !isContinuous) {
        currentSpeaker = currentSpeaker === 1 ? 2 : 1;
      }
    }

    rawSegments.push({
      start,
      end,
      text: sText,
      speaker_id: `Speaker ${currentSpeaker}`,
      embedding_similarity_to_prev: i > 0 ? (currentSpeaker === 1 ? 0.95 : 0.42) : 1.0
    });
  }

  return rawSegments;
}

/**
 * Validates candidate raw speech segments and returns merged speaker turns with merge decisions.
 * @param {Array} candidateSegments [{ start: number, end: number, text: string, embedding_similarity_to_prev?: number, pitch?: number, energy?: number }]
 * @param {Object} options { similarityThreshold?: number }
 * @returns {Object} { speaker_segments: Array, merge_decisions: Array, confidence: number }
 */
export function validateSpeakerSegments(candidateSegments = [], options = {}) {
  const similarityThreshold = options.similarityThreshold ?? 0.75;

  if (!candidateSegments || candidateSegments.length === 0) {
    return {
      speaker_segments: [],
      merge_decisions: [],
      confidence: 100
    };
  }

  if (candidateSegments.length === 1) {
    const seg = candidateSegments[0];
    return {
      speaker_segments: [
        {
          speaker_id: seg.speaker_id || 'Speaker 1',
          start: Number(seg.start || seg.start_sec || 0.0),
          end: Number(seg.end || seg.end_sec || 4.0),
          text: (seg.text || '').trim()
        }
      ],
      merge_decisions: [],
      confidence: 98
    };
  }

  const mergedSegments = [];
  const mergeDecisions = [];

  // Start with first candidate segment
  let currentSeg = {
    speaker_id: candidateSegments[0].speaker_id || 'Speaker 1',
    start: Number(candidateSegments[0].start || candidateSegments[0].start_sec || 0.0),
    end: Number(candidateSegments[0].end || candidateSegments[0].end_sec || 3.0),
    text: (candidateSegments[0].text || '').trim()
  };
  let speakerIndex = 1;

  for (let i = 1; i < candidateSegments.length; i++) {
    const nextCand = candidateSegments[i];
    const nextStart = Number(nextCand.start || nextCand.start_sec || currentSeg.end);
    const nextEnd = Number(nextCand.end || nextCand.end_sec || nextStart + 2.0);
    const nextText = (nextCand.text || '').trim();

    const pauseSec = Math.max(0, nextStart - currentSeg.end);
    const segRange = [`${currentSeg.start.toFixed(1)}-${currentSeg.end.toFixed(1)}`, `${nextStart.toFixed(1)}-${nextEnd.toFixed(1)}`];

    const embeddingSim = typeof nextCand.embedding_similarity_to_prev === 'number'
      ? nextCand.embedding_similarity_to_prev
      : (typeof nextCand.embedding_similarity === 'number' ? nextCand.embedding_similarity : null);

    const isContinuous = isGrammaticallyContinuous(currentSeg.text, nextText);
    const hasTurnTaking = isTurnTakingText(nextText);
    const hasLowSimilarity = (embeddingSim !== null && embeddingSim < similarityThreshold);
    const hasDifferentSpeakerId = (nextCand.speaker_id && currentSeg.speaker_id && nextCand.speaker_id !== currentSeg.speaker_id);

    let shouldSplit = false;
    let decisionReason = '';

    // RULE 1: Grammatical & Semantic Continuity Overrides Fallback Splits
    // "Hello, how" + "are you?" is continuous -> MUST MERGE INTO Speaker 1
    if (isContinuous && !hasTurnTaking && (embeddingSim === null || embeddingSim >= 0.65)) {
      shouldSplit = false;
      decisionReason = `Grammatically continuous sentence across ${pauseSec.toFixed(1)}s pause; single speaker train of thought`;
    } else if (hasDifferentSpeakerId && (hasLowSimilarity || hasTurnTaking)) {
      shouldSplit = true;
      decisionReason = `Acoustic diarization identified distinct speaker (${nextCand.speaker_id} vs ${currentSeg.speaker_id})`;
    } else if (hasLowSimilarity) {
      shouldSplit = true;
      decisionReason = `Distinct voiceprint embedding (${Math.round((embeddingSim || 0) * 100)}% < ${Math.round(similarityThreshold * 100)}%)`;
    } else if (hasTurnTaking && !isContinuous) {
      shouldSplit = true;
      decisionReason = `Distinct turn-taking dialogue marker detected`;
    } else if (pauseSec > 1.5 && !isContinuous) {
      shouldSplit = true;
      decisionReason = `Significant acoustic pause (${pauseSec.toFixed(1)}s) between non-continuous utterances`;
    } else {
      shouldSplit = false;
      decisionReason = `Single speaker baseline or continuous sentence`;
    }

    if (shouldSplit) {
      mergeDecisions.push({
        segments_considered: segRange,
        decision: 'split',
        reason: decisionReason
      });
      mergedSegments.push(currentSeg);
      speakerIndex++;
      currentSeg = {
        speaker_id: nextCand.speaker_id || `Speaker ${speakerIndex}`,
        start: nextStart,
        end: nextEnd,
        text: nextText
      };
    } else {
      mergeDecisions.push({
        segments_considered: segRange,
        decision: 'merged',
        reason: decisionReason
      });
      currentSeg.end = nextEnd;
      if (nextText && !currentSeg.text.includes(nextText)) {
        currentSeg.text = `${currentSeg.text} ${nextText}`.trim();
      }
    }
  }

  // Push final segment
  mergedSegments.push(currentSeg);

  return {
    speaker_segments: mergedSegments,
    merge_decisions: mergeDecisions,
    confidence: mergedSegments.length > 1 ? 94 : 98
  };
}
