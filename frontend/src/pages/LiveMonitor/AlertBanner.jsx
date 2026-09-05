import React, { useEffect, useRef } from 'react';

export function AlertBanner({ alerts = [], onDismiss, onClearAll }) {
  const audioCtxRef = useRef(null);

  useEffect(() => {
    const hasHighSeverity = alerts.some((a) => (a.severity || '').toLowerCase() === 'high');
    if (hasHighSeverity && typeof window !== 'undefined') {
      try {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (AudioContext) {
          if (!audioCtxRef.current) {
            audioCtxRef.current = new AudioContext();
          }
          const ctx = audioCtxRef.current;
          if (ctx.state === 'suspended') {
            ctx.resume();
          }
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();

          osc.type = 'sine';
          osc.frequency.setValueAtTime(880, ctx.currentTime);
          osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.22);

          gain.gain.setValueAtTime(0.08, ctx.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.22);

          osc.connect(gain);
          gain.connect(ctx.destination);

          osc.start();
          osc.stop(ctx.currentTime + 0.22);
        }
      } catch (e) {
        // audio policy fallback
      }
    }
  }, [alerts]);

  const highCount = alerts.filter((a) => (a.severity || '').toLowerCase() === 'high').length;

  return (
    <section className="cursor-alerts-panel" id="panel-live-alerts">
      <div className="cursor-alerts-header">
        <div className="cursor-alerts-title">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: 'var(--cursor-copper)' }}>
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <span>Incident Stream</span>
          <span className={`cursor-counter-chip ${alerts.length > 0 ? (highCount > 0 ? 'critical' : 'nominal') : 'nominal'}`}>
            {alerts.length > 0 ? `${alerts.length} Active` : '0 Active'}
          </span>
        </div>

        {alerts.length > 0 && onClearAll && (
          <button className="btn-cursor-clear-incidents" onClick={onClearAll}>
            Acknowledge All
          </button>
        )}
      </div>

      <div className="cursor-events-list">
        {alerts.length === 0 ? (
          <div className="cursor-standby-box">
            <svg className="cursor-standby-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 2a10 10 0 0 1 10 10" />
              <circle cx="12" cy="12" r="4" />
            </svg>
            <span className="cursor-standby-headline">Acoustic Field Nominal</span>
            <span className="cursor-standby-caption">
              Zero threat incidents detected. Continuous DSP acoustic stream active across 4 channels.
            </span>
          </div>
        ) : (
          alerts.map((alert, idx) => {
            const alertId = alert.id || alert.alert_id || idx;
            const severity = (alert.severity || 'medium').toLowerCase();
            const source = alert.details?.source || `MIC-0${(idx % 4) + 1}`;
            const confidence = alert.details?.confidence ? Math.round(alert.details.confidence * 100) : 95;

            return (
              <div
                key={alertId}
                className={`cursor-event-card ${severity === 'high' ? 'severity-high' : 'severity-medium'}`}
                id={`incident-card-${alertId}`}
              >
                <div className="cursor-event-top-row">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className="cursor-event-time">
                      {alert.created_at ? new Date(alert.created_at).toLocaleTimeString() : '15:12:32'}
                    </span>
                    <span className="cursor-event-source">· {source}</span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className={`cursor-diff-tag ${severity === 'high' ? 'critical' : 'nominal'}`}>
                      {severity.toUpperCase()}
                    </span>
                    {onDismiss && (
                      <button
                        className="cursor-event-dismiss-btn"
                        onClick={() => onDismiss(alertId)}
                        aria-label="Dismiss incident"
                      >
                        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <line x1="18" y1="6" x2="6" y2="18" />
                          <line x1="6" y1="6" x2="18" y2="18" />
                        </svg>
                      </button>
                    )}
                  </div>
                </div>

                <div className="cursor-event-title">{alert.message}</div>

                <div className="cursor-event-progress-row">
                  <div className="cursor-confidence-track-bg">
                    <div
                      className={`cursor-confidence-track-fill ${severity === 'high' ? 'critical' : ''}`}
                      style={{ width: `${confidence}%` }}
                    />
                  </div>
                  <span className="cursor-confidence-digits">{confidence}%</span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </section>
  );
}

export default AlertBanner;
