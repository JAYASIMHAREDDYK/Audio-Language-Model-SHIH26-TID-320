import React, { useState, useEffect } from 'react';

/**
 * SessionsOverTimeChart.jsx
 * Visual line/area trend chart showing audio session volume vs alert frequency over time.
 * Dynamically uses Recharts if installed, with an interactive SVG area chart fallback.
 */
export default function SessionsOverTimeChart({ sessionsOverTime = [] }) {
  const [Recharts, setRecharts] = useState(null);
  const [hoveredPoint, setHoveredPoint] = useState(null);

  useEffect(() => {
    let mounted = true;
    import('recharts')
      .then((mod) => {
        if (mounted) setRecharts(mod);
      })
      .catch(() => {
        if (mounted) setRecharts(null);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const data = sessionsOverTime.length > 0 ? sessionsOverTime : [
    { date: '2026-08-28', sessions: 8, alerts: 3 },
    { date: '2026-08-29', sessions: 12, alerts: 5 },
    { date: '2026-08-30', sessions: 9, alerts: 2 },
    { date: '2026-08-31', sessions: 15, alerts: 6 },
    { date: '2026-09-01', sessions: 18, alerts: 8 },
    { date: '2026-09-02', sessions: 22, alerts: 7 },
    { date: '2026-09-03', sessions: 25, alerts: 9 }
  ];

  const maxVal = Math.max(1, ...data.map((d) => Math.max(d.sessions || 0, d.alerts || 0)));

  return (
    <div
      id="sessions-over-time-chart-card"
      style={{
        background: '#111827',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '12px',
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px',
        minHeight: '340px'
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700, color: '#f8fafc' }}>
            Ingestion Volume & Threat Trends
          </h3>
          <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
            Daily session traffic mapped alongside safety alerts
          </span>
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', gap: '14px', alignItems: 'center', fontSize: '0.8rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#3b82f6' }} />
            <span style={{ color: '#93c5fd' }}>Audio Sessions</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: '#ef4444' }} />
            <span style={{ color: '#fca5a5' }}>Alerts Raised</span>
          </div>
        </div>
      </div>

      {Recharts && Recharts.ResponsiveContainer && (Recharts.AreaChart || Recharts.LineChart) ? (
        /* Recharts Implementation */
        <div style={{ width: '100%', height: '220px' }}>
          <Recharts.ResponsiveContainer width="100%" height="100%">
            <Recharts.AreaChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <defs>
                <linearGradient id="sessionsGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="alertsGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <Recharts.XAxis dataKey="date" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Recharts.YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Recharts.Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: '#f8fafc' }}
              />
              <Recharts.Area type="monotone" dataKey="sessions" stroke="#3b82f6" fillOpacity={1} fill="url(#sessionsGrad)" strokeWidth={2} />
              <Recharts.Area type="monotone" dataKey="alerts" stroke="#ef4444" fillOpacity={1} fill="url(#alertsGrad)" strokeWidth={2} />
            </Recharts.AreaChart>
          </Recharts.ResponsiveContainer>
        </div>
      ) : (
        /* SVG Area Fallback Chart */
        <div style={{ position: 'relative', width: '100%', height: '200px', display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', paddingTop: '20px' }}>
          {/* Columns representation */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', height: '160px', padding: '0 8px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
            {data.map((item, idx) => {
              const sessionHeight = Math.round(((item.sessions || 0) / maxVal) * 140);
              const alertHeight = Math.round(((item.alerts || 0) / maxVal) * 140);
              const isHovered = hoveredPoint === idx;

              return (
                <div
                  key={item.date}
                  onMouseEnter={() => setHoveredPoint(idx)}
                  onMouseLeave={() => setHoveredPoint(null)}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    gap: '4px',
                    flex: 1,
                    cursor: 'pointer',
                    position: 'relative'
                  }}
                >
                  {/* Tooltip on hover */}
                  {isHovered && (
                    <div
                      style={{
                        position: 'absolute',
                        bottom: '100%',
                        marginBottom: '8px',
                        background: '#0f172a',
                        border: '1px solid rgba(59,130,246,0.4)',
                        padding: '6px 10px',
                        borderRadius: '6px',
                        fontSize: '0.72rem',
                        color: '#f8fafc',
                        whiteSpace: 'nowrap',
                        zIndex: 20,
                        boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
                      }}
                    >
                      <div>{item.date}</div>
                      <div style={{ color: '#60a5fa' }}>Sessions: {item.sessions}</div>
                      <div style={{ color: '#f87171' }}>Alerts: {item.alerts}</div>
                    </div>
                  )}

                  {/* Dual Bar Cluster */}
                  <div style={{ display: 'flex', alignItems: 'flex-end', gap: '3px', height: '140px' }}>
                    <div
                      style={{
                        width: '12px',
                        height: `${Math.max(6, sessionHeight)}px`,
                        background: 'linear-gradient(180deg, #60a5fa, #2563eb)',
                        borderRadius: '3px 3px 0 0',
                        transition: 'height 0.3s ease'
                      }}
                    />
                    <div
                      style={{
                        width: '12px',
                        height: `${Math.max(4, alertHeight)}px`,
                        background: 'linear-gradient(180deg, #f87171, #dc2626)',
                        borderRadius: '3px 3px 0 0',
                        transition: 'height 0.3s ease'
                      }}
                    />
                  </div>

                  {/* Date label */}
                  <span style={{ fontSize: '0.68rem', color: '#64748b', marginTop: '6px' }}>
                    {item.date.substring(5)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
