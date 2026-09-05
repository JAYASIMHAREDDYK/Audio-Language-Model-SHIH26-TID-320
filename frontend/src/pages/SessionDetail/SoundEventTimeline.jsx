import React, { useState } from 'react';

/**
 * SoundEventTimeline.jsx
 * Interactive visual acoustic timeline rendering detected sound events
 * mapped horizontally across the audio recording duration.
 */
export default function SoundEventTimeline({ soundEvents = [], durationSec = 10 }) {
  const [hoveredEvent, setHoveredEvent] = useState(null);
  const [selectedTag, setSelectedTag] = useState(null);

  // Calculate effective timeline duration
  const maxEnd = soundEvents.reduce((max, evt) => Math.max(max, evt.end_sec || 0), 0);
  const effectiveDuration = Math.max(durationSec || 0, maxEnd, 6.0);

  // Color palette for acoustic classes
  const getColorForLabel = (label) => {
    const l = (label || '').toLowerCase();
    if (l.includes('smoke') || l.includes('alarm') || l.includes('siren')) {
      return { bg: '#ef4444', text: '#ffffff', glow: 'rgba(239, 68, 68, 0.4)' };
    }
    if (l.includes('scream') || l.includes('shout') || l.includes('yell')) {
      return { bg: '#f97316', text: '#ffffff', glow: 'rgba(249, 115, 22, 0.4)' };
    }
    if (l.includes('explosion') || l.includes('gunshot') || l.includes('blast')) {
      return { bg: '#b91c1c', text: '#ffffff', glow: 'rgba(185, 28, 28, 0.5)' };
    }
    if (l.includes('glass')) {
      return { bg: '#06b6d4', text: '#ffffff', glow: 'rgba(6, 182, 212, 0.4)' };
    }
    if (l.includes('footstep') || l.includes('walk')) {
      return { bg: '#8b5cf6', text: '#ffffff', glow: 'rgba(139, 92, 246, 0.4)' };
    }
    if (l.includes('engine') || l.includes('motor')) {
      return { bg: '#f59e0b', text: '#ffffff', glow: 'rgba(245, 158, 11, 0.4)' };
    }
    return { bg: '#3b82f6', text: '#ffffff', glow: 'rgba(59, 130, 246, 0.4)' };
  };

  // Generate tick markers
  const tickCount = Math.min(10, Math.ceil(effectiveDuration));
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => {
    const val = (i * (effectiveDuration / tickCount)).toFixed(1);
    const pct = (i / tickCount) * 100;
    return { val, pct };
  });

  return (
    <div
      id="sound-event-timeline-container"
      style={{
        background: '#111827',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '12px',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px'
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '1.2rem' }}>📊</span>
          <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700, color: '#f8fafc' }}>
            Acoustic Sound Event Timeline
          </h3>
        </div>
        <span style={{ fontSize: '0.82rem', color: '#94a3b8' }}>
          Total Duration: <strong style={{ color: '#38bdf8' }}>{effectiveDuration.toFixed(1)}s</strong>
        </span>
      </div>

      {soundEvents.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '24px 0', color: '#64748b', fontSize: '0.88rem' }}>
          No prominent acoustic sound events isolated in this session.
        </div>
      ) : (
        <>
          {/* Main Visual Timeline Track Area */}
          <div
            style={{
              position: 'relative',
              background: '#090d16',
              border: '1px solid rgba(255, 255, 255, 0.12)',
              borderRadius: '8px',
              padding: '36px 16px 28px 16px',
              minHeight: '80px',
              overflow: 'hidden'
            }}
          >
            {/* Horizontal Timeline Baseline */}
            <div
              style={{
                position: 'absolute',
                top: '50%',
                left: '16px',
                right: '16px',
                height: '2px',
                background: 'rgba(255, 255, 255, 0.15)',
                transform: 'translateY(-50%)'
              }}
            />

            {/* Event Segments */}
            {soundEvents.map((evt, idx) => {
              const start = Math.max(0, evt.start_sec || 0);
              const end = Math.min(effectiveDuration, evt.end_sec || start + 1.0);
              const leftPct = (start / effectiveDuration) * 100;
              const widthPct = Math.max(4, ((end - start) / effectiveDuration) * 100);
              const colorInfo = getColorForLabel(evt.label);
              const isDimmed = selectedTag && selectedTag !== evt.label;
              const isHovered = hoveredEvent === idx;

              return (
                <div
                  key={`${evt.label}-${idx}`}
                  id={`sound-segment-${idx}`}
                  onMouseEnter={() => setHoveredEvent(idx)}
                  onMouseLeave={() => setHoveredEvent(null)}
                  onClick={() => setSelectedTag(selectedTag === evt.label ? null : evt.label)}
                  style={{
                    position: 'absolute',
                    left: `calc(16px + ${leftPct}% * (100% - 32px) / 100)`,
                    width: `calc(${widthPct}% * (100% - 32px) / 100)`,
                    top: '50%',
                    transform: 'translateY(-50%)',
                    height: isHovered ? '42px' : '34px',
                    background: colorInfo.bg,
                    borderRadius: '6px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    cursor: 'pointer',
                    boxShadow: isHovered
                      ? `0 0 16px ${colorInfo.glow}`
                      : `0 2px 8px rgba(0,0,0,0.5)`,
                    opacity: isDimmed ? 0.35 : 1.0,
                    transition: 'all 0.15s ease',
                    zIndex: isHovered ? 10 : 2
                  }}
                >
                  <span
                    style={{
                      fontSize: '0.72rem',
                      fontWeight: 700,
                      color: colorInfo.text,
                      whiteSpace: 'nowrap',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      padding: '0 6px'
                    }}
                  >
                    {evt.label}
                  </span>
                </div>
              );
            })}

            {/* Ticks and Time Labels */}
            <div
              style={{
                position: 'absolute',
                bottom: '4px',
                left: '16px',
                right: '16px',
                height: '14px',
                pointerEvents: 'none'
              }}
            >
              {ticks.map((t, i) => (
                <div
                  key={i}
                  style={{
                    position: 'absolute',
                    left: `${t.pct}%`,
                    transform: 'translateX(-50%)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center'
                  }}
                >
                  <span style={{ width: '1px', height: '4px', background: 'rgba(255, 255, 255, 0.3)' }} />
                  <span style={{ fontSize: '0.68rem', color: '#64748b', marginTop: '2px' }}>
                    {t.val}s
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Hover or Selected Detail Card */}
          {hoveredEvent !== null && soundEvents[hoveredEvent] && (
            <div
              id="event-detail-inspector"
              style={{
                background: 'rgba(15, 23, 42, 0.95)',
                border: '1px solid rgba(59, 130, 246, 0.3)',
                borderRadius: '8px',
                padding: '10px 14px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                fontSize: '0.82rem',
                color: '#e2e8f0'
              }}
            >
              <div>
                <strong>Sound Class:</strong>{' '}
                <span style={{ color: '#38bdf8', fontWeight: 600 }}>
                  {soundEvents[hoveredEvent].label}
                </span>
              </div>
              <div>
                <strong>Time:</strong>{' '}
                <span>
                  {soundEvents[hoveredEvent].start_sec.toFixed(1)}s — {soundEvents[hoveredEvent].end_sec.toFixed(1)}s
                </span>
              </div>
              <div>
                <strong>Confidence:</strong>{' '}
                <span style={{ color: '#34d399', fontWeight: 600 }}>
                  {Math.round((soundEvents[hoveredEvent].confidence || 0) * 100)}%
                </span>
              </div>
            </div>
          )}

          {/* Filter / Legend Tag Chips */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', alignItems: 'center' }}>
            <span style={{ fontSize: '0.75rem', color: '#94a3b8', textTransform: 'uppercase' }}>Legend:</span>
            {Array.from(new Set(soundEvents.map((e) => e.label))).map((label) => {
              const colorInfo = getColorForLabel(label);
              const isSelected = selectedTag === label;
              return (
                <button
                  key={label}
                  type="button"
                  onClick={() => setSelectedTag(isSelected ? null : label)}
                  style={{
                    background: isSelected ? colorInfo.bg : 'rgba(255, 255, 255, 0.06)',
                    color: isSelected ? '#ffffff' : '#cbd5e1',
                    border: isSelected ? `1px solid ${colorInfo.bg}` : '1px solid rgba(255, 255, 255, 0.12)',
                    padding: '4px 10px',
                    borderRadius: '9999px',
                    fontSize: '0.75rem',
                    cursor: 'pointer',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '5px'
                  }}
                >
                  <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: colorInfo.bg }} />
                  {label}
                </button>
              );
            })}
            {selectedTag && (
              <button
                type="button"
                onClick={() => setSelectedTag(null)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  color: '#60a5fa',
                  fontSize: '0.75rem',
                  cursor: 'pointer',
                  textDecoration: 'underline'
                }}
              >
                Reset Filter
              </button>
            )}
          </div>
        </>
      )}
    </div>
  );
}
