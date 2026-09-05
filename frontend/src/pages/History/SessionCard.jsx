import React, { useState } from 'react';

/**
 * SessionCard.jsx
 * Individual audio session card displaying status, alert indicators,
 * timestamps, AI summary, and navigation to Session Detail.
 */
export default function SessionCard({ session, onSelectSession }) {
  const [copied, setCopied] = useState(false);

  if (!session) return null;

  const {
    session_id,
    created_at,
    status = 'pending',
    has_alert = false,
    summary = 'No summary available for this session.'
  } = session;

  const handleCopyId = (e) => {
    e.stopPropagation();
    if (navigator?.clipboard?.writeText) {
      navigator.clipboard.writeText(session_id);
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    }
  };

  const formatDate = (isoString) => {
    if (!isoString) return 'Unknown Date';
    try {
      const d = new Date(isoString);
      return d.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
    } catch {
      return isoString;
    }
  };

  const getStatusColor = (st) => {
    switch (st?.toLowerCase()) {
      case 'done':
        return { bg: 'rgba(16, 185, 129, 0.15)', text: '#34d399', border: 'rgba(16, 185, 129, 0.3)' };
      case 'processing':
        return { bg: 'rgba(99, 102, 241, 0.15)', text: '#818cf8', border: 'rgba(99, 102, 241, 0.3)' };
      case 'failed':
        return { bg: 'rgba(239, 68, 68, 0.15)', text: '#f87171', border: 'rgba(239, 68, 68, 0.3)' };
      default:
        return { bg: 'rgba(148, 163, 184, 0.15)', text: '#94a3b8', border: 'rgba(148, 163, 184, 0.3)' };
    }
  };

  const statusStyle = getStatusColor(status);

  return (
    <div
      id={`session-card-${session_id}`}
      onClick={() => onSelectSession && onSelectSession(session_id)}
      style={{
        background: 'linear-gradient(145deg, #111827, #1e293b)',
        border: has_alert ? '1px solid rgba(239, 68, 68, 0.45)' : '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '12px',
        padding: '18px 20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '14px',
        boxShadow: has_alert
          ? '0 4px 20px -2px rgba(239, 68, 68, 0.25)'
          : '0 4px 14px -2px rgba(0, 0, 0, 0.35)',
        cursor: 'pointer',
        transition: 'transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease',
        position: 'relative',
        overflow: 'hidden'
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = 'translateY(-3px)';
        e.currentTarget.style.boxShadow = has_alert
          ? '0 8px 25px -2px rgba(239, 68, 68, 0.4)'
          : '0 8px 25px -2px rgba(59, 130, 246, 0.25)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = 'translateY(0px)';
        e.currentTarget.style.boxShadow = has_alert
          ? '0 4px 20px -2px rgba(239, 68, 68, 0.25)'
          : '0 4px 14px -2px rgba(0, 0, 0, 0.35)';
      }}
    >
      {/* Top Header: ID, Copier, Status Badge, Alert Tag */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span
            style={{
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
              fontSize: '0.85rem',
              color: '#93c5fd',
              background: 'rgba(59, 130, 246, 0.12)',
              padding: '3px 8px',
              borderRadius: '6px',
              letterSpacing: '0.5px'
            }}
            title={session_id}
          >
            {session_id.length > 18 ? `${session_id.substring(0, 18)}...` : session_id}
          </span>

          <button
            id={`copy-btn-${session_id}`}
            type="button"
            onClick={handleCopyId}
            title="Copy Session ID"
            style={{
              background: 'transparent',
              border: 'none',
              color: copied ? '#10b981' : '#64748b',
              cursor: 'pointer',
              fontSize: '0.75rem',
              padding: '2px 6px',
              borderRadius: '4px',
              transition: 'color 0.2s ease'
            }}
          >
            {copied ? '✓ Copied' : '📋'}
          </button>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {has_alert && (
            <span
              id={`alert-badge-${session_id}`}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '5px',
                background: 'rgba(239, 68, 68, 0.2)',
                border: '1px solid rgba(239, 68, 68, 0.4)',
                color: '#f87171',
                padding: '3px 9px',
                borderRadius: '9999px',
                fontSize: '0.75rem',
                fontWeight: 600,
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
                animation: 'pulse 2s infinite'
              }}
            >
              <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#ef4444' }}></span>
              Alert Raised
            </span>
          )}

          <span
            id={`status-badge-${session_id}`}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              background: statusStyle.bg,
              border: `1px solid ${statusStyle.border}`,
              color: statusStyle.text,
              padding: '3px 10px',
              borderRadius: '9999px',
              fontSize: '0.78rem',
              fontWeight: 500,
              textTransform: 'capitalize'
            }}
          >
            <span
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                background: statusStyle.text
              }}
            ></span>
            {status}
          </span>
        </div>
      </div>

      {/* Timestamp */}
      <div style={{ fontSize: '0.78rem', color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '6px' }}>
        <span>🕒</span>
        <span>{formatDate(created_at)}</span>
      </div>

      {/* Summary Snippet */}
      <div
        style={{
          background: 'rgba(15, 23, 42, 0.65)',
          borderLeft: has_alert ? '3px solid #ef4444' : '3px solid #3b82f6',
          borderRadius: '4px 8px 8px 4px',
          padding: '10px 12px',
          fontSize: '0.84rem',
          lineHeight: '1.45',
          color: '#e2e8f0'
        }}
      >
        <span style={{ color: '#94a3b8', fontSize: '0.75rem', display: 'block', marginBottom: '2px', fontWeight: 600 }}>
          INTELLIGENCE SUMMARY
        </span>
        {summary}
      </div>

      {/* Footer Action */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '2px' }}>
        <button
          id={`inspect-btn-${session_id}`}
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            if (onSelectSession) onSelectSession(session_id);
          }}
          style={{
            background: 'linear-gradient(90deg, #3b82f6, #6366f1)',
            border: 'none',
            color: '#ffffff',
            padding: '7px 14px',
            borderRadius: '6px',
            fontSize: '0.8rem',
            fontWeight: 600,
            cursor: 'pointer',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            transition: 'opacity 0.2s ease, transform 0.1s ease'
          }}
          onMouseEnter={(e) => { e.currentTarget.style.opacity = '0.9'; }}
          onMouseLeave={(e) => { e.currentTarget.style.opacity = '1.0'; }}
        >
          <span>Inspect Session</span>
          <span>→</span>
        </button>
      </div>
    </div>
  );
}
