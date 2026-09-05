import React, { useState, useEffect } from 'react';

/**
 * AlertsByTypeChart.jsx
 * Visual bar chart displaying alerts grouped by trigger category:
 * sound_event, emotion, keyword.
 * Dynamically uses Recharts if installed, with a polished SVG fallback.
 */
export default function AlertsByTypeChart({ alertsByType = {} }) {
  const [Recharts, setRecharts] = useState(null);
  const [hoveredIdx, setHoveredIdx] = useState(null);

  useEffect(() => {
    let mounted = true;
    import('recharts')
      .then((mod) => {
        if (mounted) setRecharts(mod);
      })
      .catch(() => {
        // Fallback to high-performance SVG chart
        if (mounted) setRecharts(null);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const data = [
    {
      type: 'Sound Events',
      key: 'sound_event',
      count: alertsByType.sound_event || 0,
      color: '#ef4444',
      description: 'Gunshots, smoke alarms, screams, glass break'
    },
    {
      type: 'Emotion / Stress',
      key: 'emotion',
      count: alertsByType.emotion || 0,
      color: '#f59e0b',
      description: 'High arousal fear, panic, and distress speech'
    },
    {
      type: 'Distress Keywords',
      key: 'keyword',
      count: alertsByType.keyword || 0,
      color: '#8b5cf6',
      description: 'Phrases: "help", "emergency", "police", "fire"'
    }
  ];

  const maxCount = Math.max(1, ...data.map((d) => d.count));
  const totalAlerts = data.reduce((sum, d) => sum + d.count, 0);

  return (
    <div
      id="alerts-by-type-chart-card"
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
            Alerts by Trigger Category
          </h3>
          <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
            Threat rule breakdown across {totalAlerts} total alerts
          </span>
        </div>
        <span style={{ fontSize: '0.75rem', background: 'rgba(239, 68, 68, 0.15)', color: '#f87171', padding: '3px 8px', borderRadius: '4px' }}>
          Threat Distribution
        </span>
      </div>

      {Recharts && Recharts.ResponsiveContainer && Recharts.BarChart ? (
        /* Recharts Implementation */
        <div style={{ width: '100%', height: '220px' }}>
          <Recharts.ResponsiveContainer width="100%" height="100%">
            <Recharts.BarChart data={data} margin={{ top: 20, right: 20, left: 0, bottom: 20 }}>
              <Recharts.XAxis dataKey="type" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <Recharts.YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} allowDecimals={false} />
              <Recharts.Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', color: '#f8fafc' }}
              />
              <Recharts.Bar dataKey="count" radius={[6, 6, 0, 0]}>
                {data.map((entry, index) => (
                  <Recharts.Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Recharts.Bar>
            </Recharts.BarChart>
          </Recharts.ResponsiveContainer>
        </div>
      ) : (
        /* High-Fidelity SVG Bar Chart Engine */
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', paddingTop: '10px' }}>
          {data.map((item, idx) => {
            const pct = Math.round((item.count / maxCount) * 100);
            const share = totalAlerts > 0 ? Math.round((item.count / totalAlerts) * 100) : 0;
            const isHovered = hoveredIdx === idx;

            return (
              <div
                key={item.key}
                onMouseEnter={() => setHoveredIdx(idx)}
                onMouseLeave={() => setHoveredIdx(null)}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '6px',
                  padding: '8px 10px',
                  borderRadius: '8px',
                  background: isHovered ? 'rgba(255, 255, 255, 0.03)' : 'transparent',
                  transition: 'background 0.15s ease'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.85rem' }}>
                  <span style={{ fontWeight: 600, color: '#f1f5f9', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '10px', height: '10px', borderRadius: '3px', background: item.color }} />
                    {item.type}
                  </span>
                  <span style={{ color: '#cbd5e1' }}>
                    <strong style={{ color: item.color }}>{item.count}</strong> alerts ({share}%)
                  </span>
                </div>

                {/* Bar */}
                <div style={{ width: '100%', height: '14px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '9999px', overflow: 'hidden' }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${Math.max(4, pct)}%`,
                      background: `linear-gradient(90deg, ${item.color}88, ${item.color})`,
                      borderRadius: '9999px',
                      transition: 'width 0.4s ease'
                    }}
                  />
                </div>

                <span style={{ fontSize: '0.73rem', color: '#64748b' }}>
                  {item.description}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
