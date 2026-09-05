import React, { useState, useEffect, useCallback } from 'react';
import SessionCard from './SessionCard';

/**
 * HistoryPage.jsx
 * History dashboard page fetching paginated audio sessions from:
 * GET /api/v1/sessions?page=1&limit=10
 * Offers status filters, search, table/card toggle, and click-through to Session Detail.
 */
export default function HistoryPage({ onSelectSession }) {
  const [sessions, setSessions] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [limit, setLimit] = useState(10);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filterStatus, setFilterStatus] = useState('all'); // all, alerts_only, done, processing, failed
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState('grid'); // 'grid' | 'table'

  const fetchSessions = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/v1/sessions?page=${page}&limit=${limit}`);
      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }
      const data = await res.json();
      setSessions(data.sessions || []);
      setTotal(data.total || 0);
    } catch (err) {
      console.warn('[HistoryPage] Live fetch failed, using realistic fallback data:', err.message);
      // Fallback data for standalone development
      const mockSessions = [
        {
          session_id: 'a1b2c3d4-0001-4000-8000-000000000001',
          created_at: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
          status: 'done',
          has_alert: true,
          summary: 'A person is shouting for help in a high-stress, fearful tone while a smoke alarm is audible — likely a fire emergency.'
        },
        {
          session_id: 'b2c3d4e5-0002-4000-8000-000000000002',
          created_at: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
          status: 'done',
          has_alert: true,
          summary: 'Acoustic glass shatter detected followed by intense hushed whispering. Potential security breach.'
        },
        {
          session_id: 'c3d4e5f6-0003-4000-8000-000000000003',
          created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
          status: 'done',
          has_alert: false,
          summary: 'Routine conversation recorded in conference room. Normal vocal acoustic metrics, no acoustic threats.'
        },
        {
          session_id: 'd4e5f6a7-0004-4000-8000-000000000004',
          created_at: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
          status: 'done',
          has_alert: true,
          summary: 'Loud explosive bang acoustic event followed by vocal screaming in parking perimeter.'
        },
        {
          session_id: 'e5f6a7b8-0005-4000-8000-000000000005',
          created_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
          status: 'done',
          has_alert: false,
          summary: 'Ambient acoustic monitoring in warehouse corridor. Forklift and ventilation sounds, zero threats.'
        },
        {
          session_id: 'f6a7b8c9-0006-4000-8000-000000000006',
          created_at: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
          status: 'processing',
          has_alert: false,
          summary: 'Audio buffer ingested, awaiting ML pipeline analysis...'
        }
      ];
      setSessions(mockSessions);
      setTotal(mockSessions.length);
    } finally {
      setLoading(false);
    }
  }, [page, limit]);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  // Client-side filtering on the active page
  const filteredSessions = sessions.filter((s) => {
    if (filterStatus === 'alerts_only' && !s.has_alert) return false;
    if (filterStatus !== 'all' && filterStatus !== 'alerts_only' && s.status !== filterStatus) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchId = s.session_id.toLowerCase().includes(q);
      const matchSum = (s.summary || '').toLowerCase().includes(q);
      return matchId || matchSum;
    }
    return true;
  });

  const totalPages = Math.max(1, Math.ceil(total / limit));

  const totalAlertsCount = sessions.filter((s) => s.has_alert).length;
  const doneCount = sessions.filter((s) => s.status === 'done').length;

  return (
    <div
      id="history-page-container"
      style={{
        minHeight: '100vh',
        background: '#090d16',
        color: '#f8fafc',
        fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
        padding: '32px 24px'
      }}
    >
      <div style={{ maxWidth: '1280px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '28px' }}>
        
        {/* Header Title & Live Refresh */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span style={{ fontSize: '1.8rem' }}>🎙️</span>
              <h1 style={{ fontSize: '1.9rem', fontWeight: 800, margin: 0, letterSpacing: '-0.5px' }}>
                Session History & Archive
              </h1>
            </div>
            <p style={{ color: '#94a3b8', margin: '6px 0 0 0', fontSize: '0.95rem' }}>
              Chronological log of ingested audio sessions, threat detections, and ML perception analyses.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
            <button
              id="history-refresh-btn"
              onClick={fetchSessions}
              disabled={loading}
              style={{
                background: 'rgba(255, 255, 255, 0.08)',
                border: '1px solid rgba(255, 255, 255, 0.15)',
                color: '#e2e8f0',
                padding: '9px 16px',
                borderRadius: '8px',
                fontSize: '0.85rem',
                fontWeight: 600,
                cursor: loading ? 'not-allowed' : 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                transition: 'background 0.2s ease'
              }}
            >
              <span>{loading ? '⟳ Syncing...' : '⟳ Refresh'}</span>
            </button>
          </div>
        </div>

        {/* Quick KPI Cards Bar */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '14px'
          }}
        >
          <div style={kpiBoxStyle}>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Total Sessions
            </span>
            <span style={{ fontSize: '1.6rem', fontWeight: 700, color: '#60a5fa' }}>{total}</span>
          </div>

          <div style={kpiBoxStyle}>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Sessions with Alerts
            </span>
            <span style={{ fontSize: '1.6rem', fontWeight: 700, color: '#f87171' }}>{totalAlertsCount}</span>
          </div>

          <div style={kpiBoxStyle}>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Successfully Processed
            </span>
            <span style={{ fontSize: '1.6rem', fontWeight: 700, color: '#34d399' }}>{doneCount}</span>
          </div>

          <div style={kpiBoxStyle}>
            <span style={{ fontSize: '0.8rem', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
              Current Page Volume
            </span>
            <span style={{ fontSize: '1.6rem', fontWeight: 700, color: '#a78bfa' }}>{sessions.length}</span>
          </div>
        </div>

        {/* Filter Controls Toolbar */}
        <div
          style={{
            background: '#111827',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            borderRadius: '12px',
            padding: '16px 20px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '14px'
          }}
        >
          {/* Search Box */}
          <div style={{ flex: '1 1 280px', position: 'relative' }}>
            <input
              id="history-search-input"
              type="text"
              placeholder="Search by Session ID or intelligence keywords..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                width: '100%',
                boxSizing: 'border-box',
                background: '#0b0f19',
                border: '1px solid rgba(255, 255, 255, 0.12)',
                borderRadius: '8px',
                padding: '10px 14px',
                color: '#f8fafc',
                fontSize: '0.88rem',
                outline: 'none'
              }}
            />
          </div>

          {/* Filter Pills */}
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
            {[
              { id: 'all', label: 'All' },
              { id: 'alerts_only', label: '⚠️ Alerts Only' },
              { id: 'done', label: 'Done' },
              { id: 'processing', label: 'Processing' },
              { id: 'failed', label: 'Failed' }
            ].map((f) => (
              <button
                key={f.id}
                id={`filter-btn-${f.id}`}
                type="button"
                onClick={() => setFilterStatus(f.id)}
                style={{
                  background: filterStatus === f.id ? '#3b82f6' : 'rgba(255, 255, 255, 0.06)',
                  color: filterStatus === f.id ? '#ffffff' : '#94a3b8',
                  border: filterStatus === f.id ? '1px solid #60a5fa' : '1px solid transparent',
                  padding: '6px 14px',
                  borderRadius: '9999px',
                  fontSize: '0.8rem',
                  fontWeight: 600,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                {f.label}
              </button>
            ))}
          </div>

          {/* View Mode Toggle */}
          <div style={{ display: 'flex', border: '1px solid rgba(255, 255, 255, 0.1)', borderRadius: '8px', overflow: 'hidden' }}>
            <button
              id="view-toggle-grid"
              type="button"
              onClick={() => setViewMode('grid')}
              style={{
                background: viewMode === 'grid' ? 'rgba(59, 130, 246, 0.25)' : 'transparent',
                color: viewMode === 'grid' ? '#93c5fd' : '#64748b',
                border: 'none',
                padding: '8px 12px',
                cursor: 'pointer',
                fontSize: '0.85rem'
              }}
              title="Grid View"
            >
              ⊞ Grid
            </button>
            <button
              id="view-toggle-table"
              type="button"
              onClick={() => setViewMode('table')}
              style={{
                background: viewMode === 'table' ? 'rgba(59, 130, 246, 0.25)' : 'transparent',
                color: viewMode === 'table' ? '#93c5fd' : '#64748b',
                border: 'none',
                padding: '8px 12px',
                cursor: 'pointer',
                fontSize: '0.85rem'
              }}
              title="Table View"
            >
              ☰ Table
            </button>
          </div>
        </div>

        {/* Content Area */}
        {loading ? (
          <div style={{ textAlign: 'center', padding: '60px 0', color: '#94a3b8' }}>
            <div style={{ fontSize: '2rem', animation: 'spin 1s infinite linear', display: 'inline-block' }}>⟳</div>
            <p style={{ marginTop: '12px', fontSize: '0.95rem' }}>Loading session archive...</p>
          </div>
        ) : filteredSessions.length === 0 ? (
          <div
            id="empty-sessions-notice"
            style={{
              background: '#111827',
              border: '1px dashed rgba(255, 255, 255, 0.15)',
              borderRadius: '12px',
              padding: '48px 24px',
              textAlign: 'center',
              color: '#94a3b8'
            }}
          >
            <span style={{ fontSize: '2.5rem', display: 'block', marginBottom: '10px' }}>🔍</span>
            <h3 style={{ margin: '0 0 6px 0', color: '#f1f5f9' }}>No Audio Sessions Found</h3>
            <p style={{ margin: 0, fontSize: '0.88rem' }}>
              No sessions matched your filter or search query. Try clearing the filters.
            </p>
          </div>
        ) : viewMode === 'grid' ? (
          <div
            id="history-grid-view"
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))',
              gap: '18px'
            }}
          >
            {filteredSessions.map((session) => (
              <SessionCard
                key={session.session_id}
                session={session}
                onSelectSession={onSelectSession}
              />
            ))}
          </div>
        ) : (
          /* Table View */
          <div
            id="history-table-view"
            style={{
              background: '#111827',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              borderRadius: '12px',
              overflowX: 'auto'
            }}
          >
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.88rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)', color: '#94a3b8', background: 'rgba(255, 255, 255, 0.02)' }}>
                  <th style={{ padding: '14px 18px' }}>Session ID</th>
                  <th style={{ padding: '14px 18px' }}>Created At</th>
                  <th style={{ padding: '14px 18px' }}>Status</th>
                  <th style={{ padding: '14px 18px' }}>Alert</th>
                  <th style={{ padding: '14px 18px' }}>Intelligence Summary</th>
                  <th style={{ padding: '14px 18px', textAlign: 'right' }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {filteredSessions.map((session) => (
                  <tr
                    key={session.session_id}
                    style={{
                      borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
                      transition: 'background 0.15s ease'
                    }}
                    onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(255, 255, 255, 0.03)'; }}
                    onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; }}
                  >
                    <td style={{ padding: '14px 18px', fontFamily: 'monospace', color: '#93c5fd' }}>
                      {session.session_id.substring(0, 18)}...
                    </td>
                    <td style={{ padding: '14px 18px', color: '#94a3b8', whiteSpace: 'nowrap' }}>
                      {new Date(session.created_at).toLocaleString()}
                    </td>
                    <td style={{ padding: '14px 18px' }}>
                      <span
                        style={{
                          padding: '3px 8px',
                          borderRadius: '9999px',
                          fontSize: '0.75rem',
                          textTransform: 'capitalize',
                          background: session.status === 'done' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(99, 102, 241, 0.15)',
                          color: session.status === 'done' ? '#34d399' : '#818cf8'
                        }}
                      >
                        {session.status}
                      </span>
                    </td>
                    <td style={{ padding: '14px 18px' }}>
                      {session.has_alert ? (
                        <span style={{ color: '#ef4444', fontWeight: 600, fontSize: '0.78rem' }}>⚠️ Raised</span>
                      ) : (
                        <span style={{ color: '#64748b', fontSize: '0.78rem' }}>— Clear</span>
                      )}
                    </td>
                    <td style={{ padding: '14px 18px', color: '#cbd5e1', maxWidth: '380px' }}>
                      <div style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {session.summary}
                      </div>
                    </td>
                    <td style={{ padding: '14px 18px', textAlign: 'right' }}>
                      <button
                        type="button"
                        onClick={() => onSelectSession && onSelectSession(session.session_id)}
                        style={{
                          background: '#3b82f6',
                          border: 'none',
                          color: '#ffffff',
                          padding: '5px 12px',
                          borderRadius: '6px',
                          fontSize: '0.8rem',
                          cursor: 'pointer',
                          fontWeight: 600
                        }}
                      >
                        Inspect →
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination Controls */}
        <div
          id="history-pagination-bar"
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '12px',
            paddingTop: '8px'
          }}
        >
          <div style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
            Showing page <strong style={{ color: '#f8fafc' }}>{page}</strong> of{' '}
            <strong style={{ color: '#f8fafc' }}>{totalPages}</strong> ({total} total sessions)
          </div>

          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <button
              id="prev-page-btn"
              type="button"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              style={{
                background: page <= 1 ? 'rgba(255, 255, 255, 0.04)' : 'rgba(255, 255, 255, 0.09)',
                border: '1px solid rgba(255, 255, 255, 0.12)',
                color: page <= 1 ? '#475569' : '#f8fafc',
                padding: '7px 14px',
                borderRadius: '6px',
                cursor: page <= 1 ? 'not-allowed' : 'pointer',
                fontSize: '0.85rem'
              }}
            >
              ← Previous
            </button>

            <button
              id="next-page-btn"
              type="button"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => p + 1)}
              style={{
                background: page >= totalPages ? 'rgba(255, 255, 255, 0.04)' : 'rgba(255, 255, 255, 0.09)',
                border: '1px solid rgba(255, 255, 255, 0.12)',
                color: page >= totalPages ? '#475569' : '#f8fafc',
                padding: '7px 14px',
                borderRadius: '6px',
                cursor: page >= totalPages ? 'not-allowed' : 'pointer',
                fontSize: '0.85rem'
              }}
            >
              Next →
            </button>

            <select
              id="page-limit-select"
              value={limit}
              onChange={(e) => {
                setLimit(Number(e.target.value));
                setPage(1);
              }}
              style={{
                background: '#111827',
                border: '1px solid rgba(255, 255, 255, 0.12)',
                color: '#f8fafc',
                padding: '7px 10px',
                borderRadius: '6px',
                fontSize: '0.85rem',
                outline: 'none',
                marginLeft: '6px'
              }}
            >
              <option value="5">5 / page</option>
              <option value="10">10 / page</option>
              <option value="25">25 / page</option>
              <option value="50">50 / page</option>
            </select>
          </div>
        </div>
      </div>
    </div>
  );
}

const kpiBoxStyle = {
  background: '#111827',
  border: '1px solid rgba(255, 255, 255, 0.07)',
  borderRadius: '10px',
  padding: '14px 18px',
  display: 'flex',
  flexDirection: 'column',
  gap: '6px'
};
