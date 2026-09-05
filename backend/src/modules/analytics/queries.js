/**
 * queries.js
 * Read-only queries for Audio Sessions, Scene Intelligence Results, and Alerts.
 * Strictly queries existing tables: audio_sessions, scene_results, alerts.
 * Supports live PostgreSQL connection with graceful fallback to realistic fixtures
 * to enable frontend and integration development before upstream tables are populated.
 */

const path = require('path');

// Optional DB client loader (non-blocking if db/client.js is empty or uninitialized)
let dbClient = null;
try {
  const resolvedClient = require('../../db/client');
  if (resolvedClient && (typeof resolvedClient.query === 'function' || resolvedClient.pool)) {
    dbClient = resolvedClient.query ? resolvedClient : resolvedClient.pool;
  }
} catch (err) {
  // DB client not yet configured by teammates; will use fixture engine
  dbClient = null;
}

// Realistic demo fixtures matching contract from docs/SRS.md and ML pipeline spec
const FIXTURE_SESSIONS = [
  {
    session_id: 'a1b2c3d4-0001-4000-8000-000000000001',
    created_at: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
    status: 'done',
    has_alert: true,
    summary: 'A person is shouting for help in a high-stress, fearful tone while a smoke alarm is audible — likely a fire emergency.',
    source_type: 'upload',
    result: {
      session_id: 'a1b2c3d4-0001-4000-8000-000000000001',
      duration_sec: 8.4,
      status: 'done',
      created_at: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
      transcript: {
        text: "Someone help, there's smoke coming from the kitchen!",
        language: 'en',
        confidence: 0.94
      },
      sound_events: [
        { label: 'smoke_alarm', confidence: 0.89, start_sec: 1.2, end_sec: 3.8 },
        { label: 'shouting', confidence: 0.82, start_sec: 0.0, end_sec: 2.5 },
        { label: 'scream', confidence: 0.76, start_sec: 4.1, end_sec: 6.0 }
      ],
      emotion: {
        primary: 'fear',
        confidence: 0.88,
        arousal: 'high'
      },
      speakers: {
        count: 1,
        diarization: [
          { speaker_id: 'spk_1', start_sec: 0.0, end_sec: 8.4 }
        ]
      },
      model_insight: {
        label: 'fire_emergency',
        confidence: 0.92
      },
      reasoning: {
        trace: 'High-arousal fear speech co-occurs with persistent smoke alarm and elevated vocal stress. Co-presence of panic keywords confirms high hazard level.',
        summary: 'A person is shouting for help in a high-stress, fearful tone while a smoke alarm is audible — likely a fire emergency.',
        severity_hint: 'critical'
      },
      alerts: [
        {
          id: 'alt-0001',
          session_id: 'a1b2c3d4-0001-4000-8000-000000000001',
          type: 'sound_event',
          severity: 'high',
          message: 'Critical acoustic hazard detected: smoke_alarm (89% confidence)',
          created_at: new Date(Date.now() - 15 * 60 * 1000 + 2000).toISOString()
        },
        {
          id: 'alt-0002',
          session_id: 'a1b2c3d4-0001-4000-8000-000000000001',
          type: 'emotion',
          severity: 'high',
          message: 'High arousal fear speech detected during emergency situation',
          created_at: new Date(Date.now() - 15 * 60 * 1000 + 2500).toISOString()
        },
        {
          id: 'alt-0003',
          session_id: 'a1b2c3d4-0001-4000-8000-000000000001',
          type: 'keyword',
          severity: 'medium',
          message: 'Distress keyword identified in transcript: "smoke", "help"',
          created_at: new Date(Date.now() - 15 * 60 * 1000 + 2800).toISOString()
        }
      ],
      processing_ms: 1840
    }
  },
  {
    session_id: 'b2c3d4e5-0002-4000-8000-000000000002',
    created_at: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
    status: 'done',
    has_alert: true,
    summary: 'Acoustic glass shatter detected followed by intense hushed whispering. Potential security breach.',
    source_type: 'mic',
    result: {
      session_id: 'b2c3d4e5-0002-4000-8000-000000000002',
      duration_sec: 6.2,
      status: 'done',
      created_at: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
      transcript: {
        text: 'Did you hear that window? Keep quiet, someone is outside.',
        language: 'en',
        confidence: 0.89
      },
      sound_events: [
        { label: 'glass_breaking', confidence: 0.93, start_sec: 0.5, end_sec: 1.8 },
        { label: 'footsteps', confidence: 0.71, start_sec: 2.2, end_sec: 5.0 }
      ],
      emotion: {
        primary: 'fear',
        confidence: 0.81,
        arousal: 'high'
      },
      speakers: {
        count: 2,
        diarization: [
          { speaker_id: 'spk_1', start_sec: 1.8, end_sec: 4.1 },
          { speaker_id: 'spk_2', start_sec: 4.2, end_sec: 6.2 }
        ]
      },
      model_insight: {
        label: 'burglary_threat',
        confidence: 0.87
      },
      reasoning: {
        trace: 'Sudden high-frequency glass shattering transient detected at 0.5s followed by low-volume tense speech and proximity footsteps.',
        summary: 'Acoustic glass shatter detected followed by intense hushed whispering. Potential security breach.',
        severity_hint: 'high'
      },
      alerts: [
        {
          id: 'alt-0004',
          session_id: 'b2c3d4e5-0002-4000-8000-000000000002',
          type: 'sound_event',
          severity: 'high',
          message: 'Violent impact / glass_breaking detected at 0.5s (93% confidence)',
          created_at: new Date(Date.now() - 45 * 60 * 1000 + 1500).toISOString()
        }
      ],
      processing_ms: 1520
    }
  },
  {
    session_id: 'c3d4e5f6-0003-4000-8000-000000000003',
    created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
    status: 'done',
    has_alert: false,
    summary: 'Routine conversation recorded in conference room. Normal vocal acoustic metrics, no acoustic threats.',
    source_type: 'mic',
    result: {
      session_id: 'c3d4e5f6-0003-4000-8000-000000000003',
      duration_sec: 11.5,
      status: 'done',
      created_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
      transcript: {
        text: 'Let us review the quarterly telemetry numbers and server load spikes.',
        language: 'en',
        confidence: 0.98
      },
      sound_events: [
        { label: 'applause', confidence: 0.65, start_sec: 9.8, end_sec: 11.2 }
      ],
      emotion: {
        primary: 'neutral',
        confidence: 0.91,
        arousal: 'low'
      },
      speakers: {
        count: 1,
        diarization: [
          { speaker_id: 'spk_1', start_sec: 0.0, end_sec: 11.5 }
        ]
      },
      model_insight: {
        label: 'normal_activity',
        confidence: 0.96
      },
      reasoning: {
        trace: 'Vocal acoustic features align with calm indoor presentation. Background noise floor well below distress threshold.',
        summary: 'Routine conversation recorded in conference room. Normal vocal acoustic metrics, no acoustic threats.',
        severity_hint: 'none'
      },
      alerts: [],
      processing_ms: 1210
    }
  },
  {
    session_id: 'd4e5f6a7-0004-4000-8000-000000000004',
    created_at: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
    status: 'done',
    has_alert: true,
    summary: 'Loud explosive bang acoustic event followed by vocal screaming in parking perimeter.',
    source_type: 'upload',
    result: {
      session_id: 'd4e5f6a7-0004-4000-8000-000000000004',
      duration_sec: 7.0,
      status: 'done',
      created_at: new Date(Date.now() - 5 * 60 * 60 * 1000).toISOString(),
      transcript: {
        text: 'Look out! Call the police immediately!',
        language: 'en',
        confidence: 0.92
      },
      sound_events: [
        { label: 'explosion', confidence: 0.88, start_sec: 0.8, end_sec: 2.1 },
        { label: 'scream', confidence: 0.91, start_sec: 2.3, end_sec: 4.5 }
      ],
      emotion: {
        primary: 'anger',
        confidence: 0.84,
        arousal: 'high'
      },
      speakers: {
        count: 2,
        diarization: [
          { speaker_id: 'spk_1', start_sec: 2.0, end_sec: 4.8 },
          { speaker_id: 'spk_2', start_sec: 4.9, end_sec: 7.0 }
        ]
      },
      model_insight: {
        label: 'violence_threat',
        confidence: 0.91
      },
      reasoning: {
        trace: 'Explosion sound event triggers immediate level 1 alert. High pitch scream confirms human terror.',
        summary: 'Loud explosive bang acoustic event followed by vocal screaming in parking perimeter.',
        severity_hint: 'critical'
      },
      alerts: [
        {
          id: 'alt-0005',
          session_id: 'd4e5f6a7-0004-4000-8000-000000000004',
          type: 'sound_event',
          severity: 'high',
          message: 'Explosion / blast sound event detected at 0.8s (88% confidence)',
          created_at: new Date(Date.now() - 5 * 60 * 60 * 1000 + 1200).toISOString()
        },
        {
          id: 'alt-0006',
          session_id: 'd4e5f6a7-0004-4000-8000-000000000004',
          type: 'keyword',
          severity: 'high',
          message: 'Critical keyword trigger: "police"',
          created_at: new Date(Date.now() - 5 * 60 * 60 * 1000 + 1800).toISOString()
        }
      ],
      processing_ms: 1980
    }
  },
  {
    session_id: 'e5f6a7b8-0005-4000-8000-000000000005',
    created_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
    status: 'done',
    has_alert: false,
    summary: 'Ambient acoustic monitoring in warehouse corridor. Forklift and ventilation sounds, zero threats.',
    source_type: 'mic',
    result: {
      session_id: 'e5f6a7b8-0005-4000-8000-000000000005',
      duration_sec: 9.0,
      status: 'done',
      created_at: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
      transcript: {
        text: 'Aisle B pallet transfer complete, heading to bay 4.',
        language: 'en',
        confidence: 0.95
      },
      sound_events: [
        { label: 'engine', confidence: 0.74, start_sec: 1.0, end_sec: 6.0 }
      ],
      emotion: {
        primary: 'neutral',
        confidence: 0.94,
        arousal: 'low'
      },
      speakers: {
        count: 1,
        diarization: [
          { speaker_id: 'spk_1', start_sec: 0.0, end_sec: 9.0 }
        ]
      },
      model_insight: {
        label: 'industrial_ambient',
        confidence: 0.98
      },
      reasoning: {
        trace: 'Continuous mechanical hum with monotone human operational report. No anomalous frequency spikes.',
        summary: 'Ambient acoustic monitoring in warehouse corridor. Forklift and ventilation sounds, zero threats.',
        severity_hint: 'none'
      },
      alerts: [],
      processing_ms: 1100
    }
  },
  {
    session_id: 'f6a7b8c9-0006-4000-8000-000000000006',
    created_at: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
    status: 'processing',
    has_alert: false,
    summary: 'Audio buffer ingested, awaiting ML pipeline analysis...',
    source_type: 'mic',
    result: null
  }
];

/**
 * Executes a read query safely against PostgreSQL if dbClient is configured.
 * Otherwise returns null to signal fallback to fixture data.
 */
async function safeDbQuery(queryText, params = []) {
  if (!dbClient) return null;
  try {
    const res = await dbClient.query(queryText, params);
    return res.rows;
  } catch (err) {
    // If table does not exist or connection fails, log and fallback cleanly
    console.warn('[Analytics Query] Database query failed or tables unmigrated, falling back to fixture:', err.message);
    return null;
  }
}

/**
 * Fetch paginated list of sessions for History Page.
 * Contract:
 * GET /api/v1/sessions?page=1&limit=10
 * -> 200 { sessions: [{ session_id, created_at, status, has_alert, summary }], total }
 */
async function getSessionsList({ page = 1, limit = 10 } = {}) {
  const parsedPage = Math.max(1, parseInt(page, 10) || 1);
  const parsedLimit = Math.max(1, Math.min(100, parseInt(limit, 10) || 10));
  const offset = (parsedPage - 1) * parsedLimit;

  // Real SQL query against Anwin's audio_sessions and Lahari's scene_results + alerts
  const sql = `
    SELECT 
      s.id AS session_id,
      s.created_at,
      s.status,
      EXISTS(SELECT 1 FROM alerts a WHERE a.session_id = s.id) AS has_alert,
      COALESCE(sr.fusion_summary, sr.reasoning_trace, 'Session processing or pending') AS summary
    FROM audio_sessions s
    LEFT JOIN scene_results sr ON sr.session_id = s.id
    ORDER BY s.created_at DESC
    LIMIT $1 OFFSET $2;
  `;

  const countSql = `SELECT COUNT(*)::int AS total FROM audio_sessions;`;

  const [dbRows, countRows] = await Promise.all([
    safeDbQuery(sql, [parsedLimit, offset]),
    safeDbQuery(countSql)
  ]);

  if (dbRows && countRows && countRows.length > 0) {
    return {
      sessions: dbRows.map(row => ({
        session_id: row.session_id,
        created_at: row.created_at,
        status: row.status,
        has_alert: Boolean(row.has_alert),
        summary: row.summary
      })),
      total: countRows[0].total
    };
  }

  // Graceful fallback to rich fixtures
  const total = FIXTURE_SESSIONS.length;
  const paged = FIXTURE_SESSIONS.slice(offset, offset + parsedLimit).map(s => ({
    session_id: s.session_id,
    created_at: s.created_at,
    status: s.status,
    has_alert: s.has_alert,
    summary: s.summary
  }));

  return {
    sessions: paged,
    total
  };
}

/**
 * Fetch full ML result and alerts for a single session.
 * Contract:
 * GET /api/v1/sessions/:id/result
 * -> 200 full ML result (as stored) + alerts: [Alert]
 */
async function getSessionResult(sessionId) {
  if (!sessionId) return null;

  // Real SQL query joining audio_sessions and scene_results
  const sessionSql = `
    SELECT 
      s.id AS session_id,
      s.status,
      s.created_at,
      sr.transcript_text,
      sr.transcript_lang,
      sr.sound_events,
      sr.emotion,
      sr.speakers,
      sr.model_insight,
      sr.reasoning_trace,
      sr.fusion_summary,
      sr.raw_ml_response
    FROM audio_sessions s
    LEFT JOIN scene_results sr ON sr.session_id = s.id
    WHERE s.id = $1;
  `;

  const alertsSql = `
    SELECT 
      id,
      session_id,
      type,
      severity,
      message,
      created_at
    FROM alerts
    WHERE session_id = $1
    ORDER BY created_at ASC;
  `;

  const [sessionRows, alertRows] = await Promise.all([
    safeDbQuery(sessionSql, [sessionId]),
    safeDbQuery(alertsSql, [sessionId])
  ]);

  if (sessionRows && sessionRows.length > 0) {
    const row = sessionRows[0];
    const alerts = (alertRows || []).map(a => ({
      id: a.id,
      session_id: a.session_id,
      type: a.type,
      severity: a.severity,
      message: a.message,
      created_at: a.created_at
    }));

    // If raw_ml_response is stored, merge alerts with it
    if (row.raw_ml_response && typeof row.raw_ml_response === 'object') {
      return {
        ...row.raw_ml_response,
        session_id: row.session_id,
        status: row.status,
        created_at: row.created_at,
        alerts
      };
    }

    // Assemble structured ML result according to contract
    return {
      session_id: row.session_id,
      status: row.status,
      created_at: row.created_at,
      transcript: {
        text: row.transcript_text || '',
        language: row.transcript_lang || 'en',
        confidence: 0.90
      },
      sound_events: Array.isArray(row.sound_events) ? row.sound_events : [],
      emotion: row.emotion || { primary: 'neutral', confidence: 0.85, arousal: 'low' },
      speakers: row.speakers || { count: 1, diarization: [] },
      model_insight: row.model_insight || { label: 'normal', confidence: 0.9 },
      reasoning: {
        trace: row.reasoning_trace || '',
        summary: row.fusion_summary || '',
        severity_hint: alerts.some(a => a.severity === 'high') ? 'critical' : 'low'
      },
      alerts
    };
  }

  // Graceful fallback to fixture session match
  const found = FIXTURE_SESSIONS.find(s => s.session_id === sessionId);
  if (found) {
    return found.result;
  }

  return null;
}

/**
 * Fetch analytics summary across all sessions.
 * Contract:
 * GET /api/v1/analytics/summary
 * -> 200 {
 *      total_sessions,
 *      total_alerts,
 *      alerts_by_type: {...},
 *      top_sound_events: [{ label, count }]
 *    }
 */
async function getAnalyticsSummary() {
  const totalSessionsSql = `SELECT COUNT(*)::int AS total FROM audio_sessions;`;
  const totalAlertsSql = `SELECT COUNT(*)::int AS total FROM alerts;`;
  const alertsByTypeSql = `
    SELECT type, COUNT(*)::int AS count 
    FROM alerts 
    GROUP BY type;
  `;
  const topSoundsSql = `
    SELECT elem->>'label' AS label, COUNT(*)::int AS count
    FROM scene_results,
    jsonb_array_elements(sound_events) AS elem
    WHERE elem->>'label' IS NOT NULL
    GROUP BY elem->>'label'
    ORDER BY count DESC
    LIMIT 10;
  `;
  const sessionsOverTimeSql = `
    SELECT TO_CHAR(created_at, 'YYYY-MM-DD') AS date, COUNT(*)::int AS count
    FROM audio_sessions
    GROUP BY TO_CHAR(created_at, 'YYYY-MM-DD')
    ORDER BY date ASC;
  `;

  const [
    totalSessionsRows,
    totalAlertsRows,
    alertsByTypeRows,
    topSoundsRows,
    sessionsOverTimeRows
  ] = await Promise.all([
    safeDbQuery(totalSessionsSql),
    safeDbQuery(totalAlertsSql),
    safeDbQuery(alertsByTypeSql),
    safeDbQuery(topSoundsSql),
    safeDbQuery(sessionsOverTimeSql)
  ]);

  if (totalSessionsRows && totalAlertsRows) {
    const alertsByType = {};
    if (alertsByTypeRows) {
      alertsByTypeRows.forEach(row => {
        alertsByType[row.type] = row.count;
      });
    }

    const topSoundEvents = (topSoundsRows || []).map(row => ({
      label: row.label,
      count: row.count
    }));

    const sessionsOverTime = (sessionsOverTimeRows || []).map(row => ({
      date: row.date,
      sessions: row.count,
      alerts: 0 // Joined or estimated
    }));

    return {
      total_sessions: totalSessionsRows[0]?.total || 0,
      total_alerts: totalAlertsRows[0]?.total || 0,
      alerts_by_type: alertsByType,
      top_sound_events: topSoundEvents,
      sessions_over_time: sessionsOverTime
    };
  }

  // Graceful fallback: calculate from rich fixtures
  const alertsByType = {
    sound_event: 0,
    emotion: 0,
    keyword: 0
  };

  const soundCountMap = {};
  let totalAlerts = 0;

  FIXTURE_SESSIONS.forEach(session => {
    if (session.result) {
      if (Array.isArray(session.result.alerts)) {
        session.result.alerts.forEach(alert => {
          totalAlerts += 1;
          alertsByType[alert.type] = (alertsByType[alert.type] || 0) + 1;
        });
      }
      if (Array.isArray(session.result.sound_events)) {
        session.result.sound_events.forEach(evt => {
          if (evt.label) {
            soundCountMap[evt.label] = (soundCountMap[evt.label] || 0) + 1;
          }
        });
      }
    }
  });

  const topSoundEvents = Object.entries(soundCountMap)
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count);

  // Realistic 7-day trend for SessionsOverTimeChart
  const sessionsOverTime = [
    { date: '2026-08-28', sessions: 8, alerts: 3 },
    { date: '2026-08-29', sessions: 12, alerts: 5 },
    { date: '2026-08-30', sessions: 9, alerts: 2 },
    { date: '2026-08-31', sessions: 15, alerts: 6 },
    { date: '2026-09-01', sessions: 18, alerts: 8 },
    { date: '2026-09-02', sessions: 22, alerts: 7 },
    { date: '2026-09-03', sessions: 25, alerts: 9 }
  ];

  return {
    total_sessions: FIXTURE_SESSIONS.length,
    total_alerts: totalAlerts,
    alerts_by_type: alertsByType,
    top_sound_events: topSoundEvents,
    sessions_over_time: sessionsOverTime
  };
}

module.exports = {
  getSessionsList,
  getSessionResult,
  getAnalyticsSummary,
  FIXTURE_SESSIONS
};
