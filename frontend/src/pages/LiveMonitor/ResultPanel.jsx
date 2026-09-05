import React, { useState } from 'react';

export function ResultPanel({ result, isLoading = false, onDeleteSpeaker, onClarifyResponse }) {
  const [showRawJson, setShowRawJson] = useState(false);
  const [speakerFilter, setSpeakerFilter] = useState('all');
  const [clarifiedChoice, setClarifiedChoice] = useState(null);

  if (isLoading) {
    return (
      <section className="cursor-scene-panel" id="panel-scene-loading">
        <div className="cursor-panel-header">
          <div className="cursor-panel-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: 'var(--cursor-copper)' }}>
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
            <span>Composer · Scene Intelligence</span>
          </div>
        </div>
        <div className="cursor-standby-box" style={{ minHeight: '340px' }}>
          <div style={{ width: '24px', height: '24px', border: '2px solid rgba(255,255,255,0.1)', borderTopColor: 'var(--cursor-copper)', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
          <span className="cursor-standby-headline">Fusing Multimodal Acoustic Evidence</span>
          <span className="cursor-standby-caption">
            Evaluating semantic context, speaker intent, vocal biometrics, and acoustic corroboration...
          </span>
        </div>
      </section>
    );
  }

  if (!result) {
    return (
      <section className="cursor-scene-panel" id="panel-scene-empty">
        <div className="cursor-panel-header">
          <div className="cursor-panel-title">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: 'var(--cursor-copper)' }}>
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
            <span>Composer · Scene Intelligence</span>
          </div>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-cursor-muted)' }}>
            Standby
          </span>
        </div>
        <div className="cursor-standby-box" style={{ minHeight: '340px' }}>
          <svg className="cursor-standby-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
            <circle cx="12" cy="12" r="10" />
            <polygon points="12 8 8 12 12 16 16 12 12 8" />
          </svg>
          <span className="cursor-standby-headline">Composer Workspace Ready</span>
          <span className="cursor-standby-caption">
            Awaiting acoustic stream. Start live capture or record audio to evaluate multi-factor evidence, semantic context, and acoustic signatures.
          </span>
        </div>
      </section>
    );
  }

  const transcriptText =
    result.transcript_text ||
    (typeof result.transcript === 'string' ? result.transcript : result.transcript?.text) ||
    'No speech detected in acoustic stream.';
  const transcriptLang = (result.transcript_lang || result.transcript?.lang || 'en').toUpperCase();

  const soundEvents = Array.isArray(result.sound_events)
    ? result.sound_events
    : Array.isArray(result.soundEvents)
    ? result.soundEvents
    : [];

  const rawSpeakers = Array.isArray(result.speakers) ? result.speakers : [];
  const emotion = result.emotion || { primary: 'neutral', arousal: 'low', confidence: 0 };
  const modelInsight = result.model_insight || result.modelInsight || null;
  const reasoningTrace = result.reasoning_trace || result.reasoningTrace || '';
  const fusionSummary = result.fusion_summary || result.fusionSummary || '';
  
  // Extract Multi-Factor Intelligence Report
  const intel = result.intelligence || (result.reasoning && result.reasoning.intelligence) || {
    classification: 'conversational',
    intent: 'General conversation',
    confidence: 0.95,
    keywords: [],
    acoustic_signals: { scream: false, crying: false, laughter: false, distress: false, intensity: 0.3 },
    context: { joke: 0.0, hypothetical: 0.0, storytelling: 0.0, serious: 0.9 },
    evidence: [],
    contradicting_evidence: [],
    ambiguity: 0.0,
    recommended_action: 'no_alert',
    severity: 'none',
    explanation: fusionSummary || 'Scene acoustic parameters within normal baseline.'
  };

  const classification = (intel.classification || 'conversational').toLowerCase();
  const severity = (intel.severity || 'none').toLowerCase();
  const confidencePercent = Math.round((intel.confidence || 0.9) * 100);
  const ambiguityPercent = Math.round((intel.ambiguity || 0) * 100);

  const getClassificationBadge = (cls) => {
    switch (cls) {
      case 'emergency':
        return { label: 'EMERGENCY / ACTIVE THREAT', bg: 'rgba(239, 68, 68, 0.2)', border: 'rgba(239, 68, 68, 0.4)', color: '#f87171' };
      case 'joke':
        return { label: 'JOKE / NON-THREATENING', bg: 'rgba(16, 185, 129, 0.2)', border: 'rgba(16, 185, 129, 0.4)', color: '#34d399' };
      case 'hypothetical':
        return { label: 'HYPOTHETICAL INQUIRY', bg: 'rgba(59, 130, 246, 0.2)', border: 'rgba(59, 130, 246, 0.4)', color: '#60a5fa' };
      case 'storytelling':
      case 'quotation':
        return { label: 'STORYTELLING / QUOTED CONTEXT', bg: 'rgba(168, 85, 247, 0.2)', border: 'rgba(168, 85, 247, 0.4)', color: '#c084fc' };
      case 'ambiguous':
        return { label: 'AMBIGUOUS / UNCERTAIN', bg: 'rgba(245, 158, 11, 0.2)', border: 'rgba(245, 158, 11, 0.4)', color: '#fbbf24' };
      default:
        return { label: 'NOMINAL CONVERSATION', bg: 'rgba(148, 163, 184, 0.15)', border: 'rgba(148, 163, 184, 0.3)', color: '#cbd5e1' };
    }
  };

  const badgeStyle = getClassificationBadge(classification);

  // Classify each speaker voice into Threat Categories (Critical -> Warning -> Nominal)
  const categorizedSpeakers = rawSpeakers.map((spk, idx) => {
    const speakerId = spk.speaker_id || spk.id || `SPK-0${idx + 1}`;
    const text = spk.text || (idx === 0 ? transcriptText : 'Background acoustic vocalization.');
    const isCovert = Boolean(spk.isCovert || (spk.threatCategory === 'critical' && spk.role?.toLowerCase().includes('covert')));

    let threatCategory = spk.threatCategory;
    if (!threatCategory) {
      if (classification === 'emergency') {
        threatCategory = 'critical';
      } else if (classification === 'joke' || classification === 'hypothetical' || classification === 'storytelling' || classification === 'conversational') {
        threatCategory = 'nominal';
      } else {
        threatCategory = 'nominal';
      }
    }

    let role = spk.role || (threatCategory === 'critical' ? 'Distress Caller' : 'Conversational Voice');

    return {
      ...spk,
      speaker_id: speakerId,
      text,
      role,
      threatCategory,
      isCovert,
      emotion: spk.emotion || emotion,
      timeSpan: spk.timeSpan || (spk.duration ? `[0.0s - ${spk.duration}s]` : `[SPK-0${idx + 1}]`)
    };
  });

  const sortedSpeakers = [...categorizedSpeakers].sort((a, b) => {
    const rank = { critical: 0, warning: 1, nominal: 2 };
    return (rank[a.threatCategory] ?? 3) - (rank[b.threatCategory] ?? 3);
  });

  const filteredSpeakers = sortedSpeakers.filter((s) => {
    if (speakerFilter === 'all') return true;
    return s.threatCategory === speakerFilter;
  });

  const criticalCount = categorizedSpeakers.filter(s => s.threatCategory === 'critical').length;
  const warningCount = categorizedSpeakers.filter(s => s.threatCategory === 'warning').length;
  const nominalCount = categorizedSpeakers.filter(s => s.threatCategory === 'nominal').length;

  return (
    <section className="cursor-scene-panel" id="panel-scene-intelligence">
      <div className="cursor-panel-header">
        <div className="cursor-panel-title">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: 'var(--cursor-copper)' }}>
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
          </svg>
          <span>Composer · Multi-Factor Context Intelligence</span>
        </div>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', color: 'var(--cursor-copper)' }}>
          {result.id || 'Live Scene Telemetry'}
        </span>
      </div>

      <div className="cursor-panel-body">
        {/* Multi-Factor Forensic Assessment Banner */}
        <div style={{
          background: 'var(--bg-cursor-elevated)',
          border: `1px solid ${badgeStyle.border}`,
          borderRadius: 'var(--radius-sm)',
          padding: '14px 16px',
          marginBottom: '16px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px', flexWrap: 'wrap', gap: '8px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{
                background: badgeStyle.bg,
                color: badgeStyle.color,
                border: `1px solid ${badgeStyle.border}`,
                padding: '4px 10px',
                borderRadius: '4px',
                fontSize: '0.75rem',
                fontWeight: 700,
                letterSpacing: '0.05em'
              }}>
                {badgeStyle.label}
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-cursor-muted)', fontFamily: 'var(--font-mono)' }}>
                Intent: <strong style={{ color: 'var(--text-cursor)' }}>{intel.intent || 'Conversational'}</strong>
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '0.75rem' }}>
              <span style={{ color: 'var(--text-cursor-muted)' }}>
                Confidence: <strong style={{ color: 'var(--cursor-copper)', fontFamily: 'var(--font-mono)' }}>{confidencePercent}%</strong>
              </span>
              {ambiguityPercent > 20 && (
                <span style={{ color: '#fbbf24' }}>
                  Ambiguity: <strong style={{ fontFamily: 'var(--font-mono)' }}>{ambiguityPercent}%</strong>
                </span>
              )}
            </div>
          </div>

          <p style={{ fontSize: '0.82rem', color: 'var(--text-cursor)', lineHeight: '1.45', margin: '0 0 10px 0' }}>
            {intel.explanation || fusionSummary}
          </p>

          {/* Acoustic & Context Signal Badges */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', paddingTop: '6px', borderTop: '1px solid rgba(255,255,255,0.06)' }}>
            <span style={{ fontSize: '0.68rem', padding: '2px 8px', borderRadius: '3px', background: intel.acoustic_signals?.scream ? 'rgba(239, 68, 68, 0.2)' : 'rgba(255,255,255,0.04)', color: intel.acoustic_signals?.scream ? '#f87171' : 'var(--text-cursor-muted)' }}>
              Scream: {intel.acoustic_signals?.scream ? 'DETECTED' : 'None'}
            </span>
            <span style={{ fontSize: '0.68rem', padding: '2px 8px', borderRadius: '3px', background: intel.acoustic_signals?.laughter ? 'rgba(16, 185, 129, 0.2)' : 'rgba(255,255,255,0.04)', color: intel.acoustic_signals?.laughter ? '#34d399' : 'var(--text-cursor-muted)' }}>
              Laughter: {intel.acoustic_signals?.laughter ? 'DETECTED' : 'None'}
            </span>
            <span style={{ fontSize: '0.68rem', padding: '2px 8px', borderRadius: '3px', background: intel.acoustic_signals?.distress ? 'rgba(239, 68, 68, 0.2)' : 'rgba(255,255,255,0.04)', color: intel.acoustic_signals?.distress ? '#f87171' : 'var(--text-cursor-muted)' }}>
              Vocal Distress: {intel.acoustic_signals?.distress ? 'HIGH' : 'Safe'}
            </span>
            <span style={{ fontSize: '0.68rem', padding: '2px 8px', borderRadius: '3px', background: 'rgba(255,255,255,0.04)', color: 'var(--text-cursor-muted)' }}>
              Speech Rate: {(intel.acoustic_signals?.speech_rate || 'normal').toUpperCase()}
            </span>
            {intel.keywords && intel.keywords.length > 0 && (
              <span style={{ fontSize: '0.68rem', padding: '2px 8px', borderRadius: '3px', background: 'rgba(217, 119, 6, 0.15)', color: '#fbbf24' }}>
                Keyword Features: {intel.keywords.join(', ')}
              </span>
            )}
          </div>
        </div>

        {/* Ambiguity Clarification Prompt when Needed */}
        {(classification === 'ambiguous' || intel.recommended_action === 'clarify' || ambiguityPercent > 60) && (
          <div style={{
            background: 'rgba(245, 158, 11, 0.12)',
            border: '1px solid rgba(245, 158, 11, 0.35)',
            borderRadius: 'var(--radius-sm)',
            padding: '12px 16px',
            marginBottom: '16px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              <span style={{ fontSize: '0.78rem', fontWeight: 600, color: '#fbbf24' }}>
                Clarification Requested · Ambiguous Acoustic Scene
              </span>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-cursor)', margin: '0 0 10px 0' }}>
              The speech or audio telemetry is incomplete or ambiguous. Are you describing an active emergency or speaking hypothetically?
            </p>
            {clarifiedChoice ? (
              <span style={{ fontSize: '0.75rem', color: '#34d399', fontWeight: 600 }}>
                ✓ Clarification Recorded: {clarifiedChoice}
              </span>
            ) : (
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  style={{
                    padding: '5px 12px',
                    fontSize: '0.72rem',
                    background: 'rgba(239, 68, 68, 0.2)',
                    border: '1px solid rgba(239, 68, 68, 0.4)',
                    color: '#f87171',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                  onClick={() => {
                    setClarifiedChoice('Active Emergency Confirmed');
                    if (onClarifyResponse) onClarifyResponse('emergency');
                  }}
                >
                  Active Emergency
                </button>
                <button
                  style={{
                    padding: '5px 12px',
                    fontSize: '0.72rem',
                    background: 'rgba(16, 185, 129, 0.2)',
                    border: '1px solid rgba(16, 185, 129, 0.4)',
                    color: '#34d399',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                  onClick={() => {
                    setClarifiedChoice('Hypothetical / Safe Remark Confirmed');
                    if (onClarifyResponse) onClarifyResponse('safe');
                  }}
                >
                  Hypothetical / Safe Discussion
                </button>
              </div>
            )}
          </div>
        )}

        {/* Multi-Speaker Diarization & Threat Categorization Section */}
        <div className="multi-speaker-section">
          <div className="speaker-section-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className="cursor-section-label">
                Diarized Speaker Streams ({categorizedSpeakers.length})
              </span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', color: 'var(--text-cursor-muted)' }}>
                Dialect: {transcriptLang}-US
              </span>
            </div>

            {/* Filter Tabs */}
            <div className="speaker-filter-tabs">
              <button
                className={`speaker-filter-btn ${speakerFilter === 'all' ? 'active' : ''}`}
                onClick={() => setSpeakerFilter('all')}
              >
                All ({categorizedSpeakers.length})
              </button>
              <button
                className={`speaker-filter-btn ${speakerFilter === 'critical' ? 'active' : ''}`}
                onClick={() => setSpeakerFilter('critical')}
              >
                Critical ({criticalCount})
              </button>
              <button
                className={`speaker-filter-btn ${speakerFilter === 'warning' ? 'active' : ''}`}
                onClick={() => setSpeakerFilter('warning')}
              >
                Warning ({warningCount})
              </button>
              <button
                className={`speaker-filter-btn ${speakerFilter === 'nominal' ? 'active' : ''}`}
                onClick={() => setSpeakerFilter('nominal')}
              >
                Nominal ({nominalCount})
              </button>
            </div>
          </div>

          {/* Speakers Cards Stack */}
          {filteredSpeakers.length === 0 ? (
            <div style={{ padding: '12px 16px', background: 'var(--bg-cursor-elevated)', border: '1px dashed var(--border-cursor)', borderRadius: 'var(--radius-sm)', fontSize: '0.8rem', color: 'var(--text-cursor-muted)', textAlign: 'center' }}>
              No speakers match the "{speakerFilter}" threat filter.
            </div>
          ) : (
            <div className="speakers-cards-stack">
              {filteredSpeakers.map((spk, idx) => (
                <div
                  key={spk.speaker_id || idx}
                  className={`speaker-voice-card threat-${spk.threatCategory}`}
                >
                  <div className="speaker-card-top-row">
                    <div className="speaker-id-badge">
                      <span className="speaker-name-title">{spk.speaker_id}</span>
                      <span className="cursor-mode-tag">· {spk.role}</span>
                      <span className="speaker-time-interval">{spk.timeSpan}</span>
                    </div>

                    <div className="speaker-actions-group">
                      {spk.isCovert ? (
                        <span className="speaker-threat-badge covert" title="Low-volume / covert speech with critical keywords">
                          COVERT DISTRESS · WHISPER
                        </span>
                      ) : (
                        <span className={`speaker-threat-badge ${spk.threatCategory}`}>
                          {spk.threatCategory.toUpperCase()} THREAT
                        </span>
                      )}
                      {onDeleteSpeaker && (
                        <button
                          className="btn-delete-voice"
                          onClick={() => onDeleteSpeaker(spk.speaker_id)}
                          title="Delete/Dismiss this speaker voice"
                        >
                          ✕ Delete
                        </button>
                      )}
                    </div>
                  </div>

                  <div className="speaker-transcript-line">
                    "{spk.text}"
                  </div>

                  <div className="speaker-dynamics-row">
                    <span>
                      Vocal Emotion: <strong>{(spk.emotion?.primary || 'neutral').toUpperCase()}</strong>
                    </span>
                    <span>
                      Arousal: <strong>{(spk.emotion?.arousal || 'low').toUpperCase()}</strong>
                    </span>
                    <span>
                      Confidence: <strong>{Math.round((spk.emotion?.confidence || 0.95) * 100)}%</strong>
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Acoustic Event Classification Table */}
        <div style={{ marginTop: '14px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span className="cursor-section-label">Acoustic Event Classification ({soundEvents.length})</span>
          </div>

          {soundEvents.length === 0 ? (
            <div style={{ padding: '10px 14px', background: 'var(--bg-cursor-elevated)', border: '1px solid var(--border-cursor)', borderRadius: 'var(--radius-sm)', fontSize: '0.8rem', color: 'var(--text-cursor-muted)' }}>
              No acoustic anomalies identified above detection floor.
            </div>
          ) : (
            <div className="cursor-classification-wrapper">
              <table className="cursor-classification-table">
                <thead>
                  <tr>
                    <th>Acoustic Label</th>
                    <th>Classification Confidence</th>
                    <th>Severity Flag</th>
                  </tr>
                </thead>
                <tbody>
                  {soundEvents.map((evt, idx) => {
                    const label = typeof evt === 'string' ? evt : evt.label || evt.name || 'sound';
                    const confidence =
                      typeof evt.confidence === 'number'
                        ? evt.confidence
                        : typeof evt.score === 'number'
                        ? evt.score
                        : 1.0;
                    const critical = ['gunshot', 'explosion', 'smoke_alarm', 'glass_breaking', 'scream'].some(c => label.toLowerCase().includes(c));

                    return (
                      <tr key={idx}>
                        <td style={{ fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                          {label.replace(/_/g, ' ')}
                        </td>
                        <td>
                          <div className="cursor-confidence-bar-unit">
                            <div className="cursor-confidence-track-bg">
                              <div
                                className={`cursor-confidence-track-fill ${critical ? 'critical' : ''}`}
                                style={{ width: `${Math.min(100, Math.round(confidence * 100))}%` }}
                              />
                            </div>
                            <span className="cursor-confidence-digits">{Math.round(confidence * 100)}%</span>
                          </div>
                        </td>
                        <td>
                          <span className={`cursor-diff-tag ${critical ? 'critical' : 'nominal'}`}>
                            {critical ? 'CRITICAL' : 'NOMINAL'}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Custom Situation Insight Card */}
        {modelInsight && (
          <div className="cursor-classifier-card" style={{ marginTop: '14px' }}>
            <div className="cursor-classifier-header">
              <span className="cursor-section-label">Domain Classifier Insight</span>
              <span className="cursor-classifier-tag">Scene Intelligence</span>
            </div>
            <div className="cursor-classifier-body">
              <div className="cursor-classifier-main">
                <span className="cursor-classifier-label-name">
                  {modelInsight.label || 'Situation Classification'}
                </span>
                <span className="cursor-classifier-desc">
                  Confidence: {Math.round((modelInsight.confidence || 0.9) * 100)}%
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Raw JSON Debugging Toggle */}
        <div style={{ marginTop: '14px' }}>
          <button
            className="cursor-raw-toggle"
            onClick={() => setShowRawJson(!showRawJson)}
            style={{ fontSize: '0.74rem' }}
          >
            {showRawJson ? 'Hide Evidence Telemetry' : 'View Evidence Telemetry'}
          </button>
          {showRawJson && (
            <pre className="cursor-raw-json" style={{ marginTop: '8px', maxHeight: '200px', overflowY: 'auto' }}>
              {JSON.stringify(result, null, 2)}
            </pre>
          )}
        </div>
      </div>
    </section>
  );
}

export default ResultPanel;

