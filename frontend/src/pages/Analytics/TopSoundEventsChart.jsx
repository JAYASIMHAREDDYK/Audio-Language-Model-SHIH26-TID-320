import React, { useState, useEffect } from 'react';

/**
 * TopSoundEventsChart.jsx
 * Visual frequency chart showing top detected acoustic sound events across all sessions.
 * Dynamically uses Recharts if installed, with a polished SVG fallback.
 */
export default function TopSoundEventsChart({ topSoundEvents = [] }) {
  const [Recharts, setRecharts] = useState(null);
  const [hoveredIdx, setHoveredIdx] = useState(null);

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

  const getBarColor = (label, idx) => {
    const l = (label || '').toLowerCase();
    if (l.includes('smoke') || l.includes('alarm') || l.includes('explosion')) return '#ef4444';
    if (l.includes('scream') || l.includes('shout')) return '#f97316';
    if (l.includes('glass')) return '#06b6d4';
    if (l.includes('footstep') || l.includes('applause')) return '#10b981';
    const colors = ['#3b82f6', '#6366f1', '#8b5cf6', '#ec4899', '#14b8a6'];
    return colors[idx % colors.length];
  };

  const data = (topSoundEvents.length > 0 ? topSoundEvents : [
    { label: 'smoke_alarm', count: 12 },
    { label: 'shouting', count: 9 },
    { label: 'scream', count: 5 },
    { label: 'glass_breaking', count: 3 },
    { label: 'footsteps', count: 2 }
  ]).slice(0, 8);

  const maxCount = Math.max(1, ...data.map((d) => d.count));

  return (
    <div
      id="top-sound-events-chart-card"
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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700, color: '#f8fafc' }}>
            Top Sound Events Frequency
          </h3>
          <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
            Most prevalent acoustic tags identified by ML perception
          </span>
        </div>
        <span style={{ fontSize: '0.75rem', background: 'rgba(59, 130, 246, 0.15)', color: '#60a5fa', padding: '3px 8px', borderRadius: '4px' }}>
          Acoustic Fingerprints
        </span>
      </div>

      {Recharts && Recharts.ResponsiveContainer && Recharts.BarChart ? (
        /* Recharts Horizontal Bar Chart */
        <div style={{ width: '100%', height: '230px' }}>
          <Recharts.ResponsiveContainer width="100%" height="100%">
            <Recharts.BarChart layout="vertical" data={data} margin={{ top: 10, right: 30, left: 40, bottom: 5 }}>
              <Recharts.XAxis type="number" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Recharts.YAxis dataKey="label" type="category" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <Recharts.Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: '#f8fafc' }}
              />
              <Recharts.Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {data.map((entry, index) => (
                  <Recharts.Cell key={`cell-${index}`} fill={getBarColor(entry.label, index)} />
                ))}
              </Recharts.Bar>
            </Recharts.BarChart>
          </Recharts.ResponsiveContainer>
        </div>
      ) : (
        /* Fallback Visual Ranked Bar Component */
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', paddingTop: '6px' }}>
          {data.map((item, idx) => {
            const pct = Math.round((item.count / maxCount) * 100);
            const color = getBarColor(item.label, idx);
            const isHovered = hoveredIdx === idx;

            return (
              <div
                key={item.label}
                onMouseEnter={() => setHoveredIdx(idx)}
                onMouseLeave={() => setHoveredIdx(null)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '4px 8px',
                  borderRadius: '6px',
                  background: isHovered ? 'rgba(255, 255, 255, 0.04)' : 'transparent',
                  transition: 'background 0.15s ease'
                }}
              >
                <span
                  style={{
                    fontSize: '0.72rem',
                    fontWeight: 700,
                    color: '#64748b',
                    width: '18px',
                    textAlign: 'right'
                  }}
                >
                  #{idx + 1}
                </span>

                <div style={{ width: '110px', fontSize: '0.82rem', fontWeight: 600, color: '#f1f5f9', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {item.label}
                </div>

                <div style={{ flex: 1, height: '10px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '9999px', overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${Math.max(5, pct)}%`,
                      background: `linear-gradient(90deg, ${color}99, ${color})`,
                      borderRadius: '9999px',
                      transition: 'width 0.4s ease'
                    }}
                  />
                </div>

                <span style={{ fontSize: '0.82rem', fontWeight: 700, color: color, width: '36px', textAlign: 'right' }}>
                  {item.count}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
