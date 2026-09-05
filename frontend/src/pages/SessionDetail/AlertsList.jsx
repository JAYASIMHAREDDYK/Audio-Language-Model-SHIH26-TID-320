import React from 'react';

/**
 * AlertsList.jsx
 * List of alerts triggered for a session with severity indicators and trigger details.
 */
export default function AlertsList({ alerts = [] }) {
  const getSeverityStyle = (severity) => {
    switch (severity?.toLowerCase()) {
      case 'high':
        return {
          bg: 'rgba(239, 68, 68, 0.18)',
          border: 'rgba(239, 68, 68, 0.4)',
          text: '#f87171',
          badgeBg: '#ef4444',
          dot: '#ef4444'
        };
      case 'medium':
        return {
          bg: 'rgba(245, 158, 11, 0.18)',
          border: 'rgba(245, 158, 11, 0.4)',
          text: '#fbbf24',
          badgeBg: '#f59e0b',
          dot: '#f59e0b'
        };
      default:
        return {
          bg: 'rgba(59, 130, 246, 0.18)',
          border: 'rgba(59, 130, 246, 0.4)',
          text: '#60a5fa',
          badgeBg: '#3b82f6',
          dot: '#3b82f6'
        };
    }
  };

  const getTypeIcon = (type) => {
    switch (type) {
      case 'sound_event':
        return '📢';
      case 'emotion':
        return '😰';
      case 'keyword':
        return '💬';
      default:
        return '⚠️';
    }
  };

  const highCount = alerts.filter((a) => a.severity === 'high').length;
  const mediumCount = alerts.filter((a) => a.severity === 'medium').length;

  return (
    <div
      id="alerts-list-container"
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
          <span style={{ fontSize: '1.2rem' }}>🚨</span>
          <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700, color: '#f8fafc' }}>
            Alerts & Safety Triggers
          </h3>
        </div>

        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          {highCount > 0 && (
            <span
              style={{
                background: 'rgba(239, 68, 68, 0.2)',
                color: '#f87171',
                padding: '2px 8px',
                borderRadius: '6px',
                fontSize: '0.75rem',
                fontWeight: 600
              }}
            >
              {highCount} High
            </span>
          )}
          {mediumCount > 0 && (
            <span
              style={{
                background: 'rgba(245, 158, 11, 0.2)',
                color: '#fbbf24',
                padding: '2px 8px',
                borderRadius: '6px',
                fontSize: '0.75rem',
                fontWeight: 600
              }}
            >
              {mediumCount} Medium
            </span>
          )}
          <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
            Total: <strong>{alerts.length}</strong>
          </span>
        </div>
      </div>

      {alerts.length === 0 ? (
        <div
          id="alerts-empty-state"
          style={{
            background: 'rgba(16, 185, 129, 0.08)',
            border: '1px solid rgba(16, 185, 129, 0.2)',
            borderRadius: '8px',
            padding: '18px',
            textAlign: 'center',
            color: '#34d399',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <span style={{ fontSize: '1.6rem' }}>🛡️</span>
          <strong style={{ fontSize: '0.92rem' }}>No Alerts Triggered</strong>
          <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
            Acoustic, emotion, and keyword safety parameters are within normal baseline.
          </span>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {alerts.map((alert, idx) => {
            const style = getSeverityStyle(alert.severity);
            return (
              <div
                key={alert.id || idx}
                id={`alert-item-${alert.id || idx}`}
                style={{
                  background: style.bg,
                  border: `1px solid ${style.border}`,
                  borderRadius: '8px',
                  padding: '12px 16px',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '6px'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '6px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span>{getTypeIcon(alert.type)}</span>
                    <span
                      style={{
                        fontSize: '0.75rem',
                        fontWeight: 700,
                        color: style.text,
                        textTransform: 'uppercase',
                        letterSpacing: '0.5px'
                      }}
                    >
                      {alert.severity} SEVERITY • {alert.type?.replace('_', ' ')}
                    </span>
                  </div>

                  {alert.created_at && (
                    <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>
                      {new Date(alert.created_at).toLocaleTimeString()}
                    </span>
                  )}
                </div>

                <div style={{ fontSize: '0.88rem', color: '#f1f5f9', fontWeight: 500, lineHeight: '1.4' }}>
                  {alert.message}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
