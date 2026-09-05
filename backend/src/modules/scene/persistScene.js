const crypto = require('crypto');
const { evaluate } = require('../alerts/ruleEngine');
const { emitResultReady, emitAlertRaised } = require('../../websocket/broadcast');

const memorySceneStore = new Map();
const memoryAlertStore = [];

function normalizePayload(sessionId, rawResult = {}) {
  const id = rawResult.id || crypto.randomUUID();
  const createdAt = rawResult.created_at || new Date().toISOString();

  let transcriptText = '';
  let transcriptLang = 'en';
  if (typeof rawResult.transcript === 'string') {
    transcriptText = rawResult.transcript;
  } else if (rawResult.transcript && typeof rawResult.transcript === 'object') {
    transcriptText = rawResult.transcript.text || rawResult.transcript.transcript_text || '';
    transcriptLang = rawResult.transcript.lang || rawResult.transcript.language || 'en';
  } else if (typeof rawResult.transcript_text === 'string') {
    transcriptText = rawResult.transcript_text;
    transcriptLang = rawResult.transcript_lang || 'en';
  }

  const soundEvents = rawResult.sound_events || rawResult.soundEvents || [];
  const emotion = rawResult.emotion || { primary: 'neutral', arousal: 'low', confidence: 1.0 };
  const speakers = rawResult.speakers || rawResult.speaker_diarization || [];
  const modelInsight = rawResult.model_insight || rawResult.modelInsight || null;

  const reasoningTrace =
    rawResult.reasoning_trace ||
    rawResult.reasoningTrace ||
    (rawResult.fusion ? rawResult.fusion.reasoning_trace : '') ||
    '';
  const fusionSummary =
    rawResult.fusion_summary ||
    rawResult.fusionSummary ||
    (rawResult.fusion ? rawResult.fusion.summary : '') ||
    '';

  return {
    id,
    session_id: sessionId,
    transcript_text: transcriptText,
    transcript_lang: transcriptLang,
    sound_events: soundEvents,
    emotion,
    speakers,
    model_insight: modelInsight,
    reasoning_trace: reasoningTrace,
    fusion_summary: fusionSummary,
    raw_ml_response: rawResult,
    created_at: createdAt
  };
}

async function persistSceneResult({ sessionId, mlResult, dbClient = null }) {
  if (!sessionId) {
    throw new Error('sessionId is required to persist scene result');
  }

  const normalizedScene = normalizePayload(sessionId, mlResult);
  const rawAlerts = evaluate(mlResult || normalizedScene);
  const alerts = rawAlerts.map((alert) => ({
    id: crypto.randomUUID(),
    session_id: sessionId,
    type: alert.type,
    severity: alert.severity,
    message: alert.message,
    details: alert.details || null,
    created_at: new Date().toISOString()
  }));

  if (dbClient && typeof dbClient.query === 'function') {
    try {
      const sceneSql = `
        INSERT INTO scene_results (
          id, session_id, transcript_text, transcript_lang,
          sound_events, emotion, speakers, model_insight,
          reasoning_trace, fusion_summary, raw_ml_response, created_at
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (id) DO UPDATE SET
          transcript_text = EXCLUDED.transcript_text,
          fusion_summary = EXCLUDED.fusion_summary
        RETURNING *;
      `;
      const sceneValues = [
        normalizedScene.id,
        normalizedScene.session_id,
        normalizedScene.transcript_text,
        normalizedScene.transcript_lang,
        JSON.stringify(normalizedScene.sound_events),
        JSON.stringify(normalizedScene.emotion),
        JSON.stringify(normalizedScene.speakers),
        normalizedScene.model_insight ? JSON.stringify(normalizedScene.model_insight) : null,
        normalizedScene.reasoning_trace,
        normalizedScene.fusion_summary,
        JSON.stringify(normalizedScene.raw_ml_response),
        normalizedScene.created_at
      ];
      await dbClient.query(sceneSql, sceneValues);

      for (const alert of alerts) {
        const alertSql = `
          INSERT INTO alerts (id, session_id, type, severity, message, created_at)
          VALUES ($1, $2, $3, $4, $5, $6);
        `;
        await dbClient.query(alertSql, [
          alert.id,
          alert.session_id,
          alert.type,
          alert.severity,
          alert.message,
          alert.created_at
        ]);
      }
    } catch (dbErr) {
      console.warn('DB query failed, fallback to memory store:', dbErr.message);
    }
  }

  memorySceneStore.set(sessionId, normalizedScene);
  alerts.forEach((a) => memoryAlertStore.push(a));

  emitResultReady(sessionId, normalizedScene);
  alerts.forEach((alert) => {
    emitAlertRaised(sessionId, alert);
  });

  return {
    sceneResult: normalizedScene,
    alerts
  };
}

async function getSceneResultBySession(sessionId, dbClient = null) {
  if (!sessionId) return null;

  let sceneResult = null;
  let alerts = [];

  if (dbClient && typeof dbClient.query === 'function') {
    try {
      const sceneRes = await dbClient.query(
        'SELECT * FROM scene_results WHERE session_id = $1 ORDER BY created_at DESC LIMIT 1',
        [sessionId]
      );
      if (sceneRes.rows && sceneRes.rows.length > 0) {
        sceneResult = sceneRes.rows[0];
      }

      const alertRes = await dbClient.query(
        'SELECT * FROM alerts WHERE session_id = $1 ORDER BY created_at DESC',
        [sessionId]
      );
      if (alertRes.rows) {
        alerts = alertRes.rows;
      }
    } catch (err) {
      console.warn('DB query error, fallback to memory store:', err.message);
    }
  }

  if (!sceneResult && memorySceneStore.has(sessionId)) {
    sceneResult = memorySceneStore.get(sessionId);
    alerts = memoryAlertStore.filter((a) => a.session_id === sessionId);
  }

  if (!sceneResult) {
    return null;
  }

  return {
    ...sceneResult,
    alerts
  };
}

async function getAlerts({ sessionId = null, dbClient = null } = {}) {
  if (dbClient && typeof dbClient.query === 'function') {
    try {
      let query = 'SELECT * FROM alerts';
      const params = [];
      if (sessionId) {
        query += ' WHERE session_id = $1';
        params.push(sessionId);
      }
      query += ' ORDER BY created_at DESC';
      const res = await dbClient.query(query, params);
      return res.rows || [];
    } catch (err) {
      console.warn('DB query error, fallback to memory store:', err.message);
    }
  }

  if (sessionId) {
    return memoryAlertStore.filter((a) => a.session_id === sessionId);
  }
  return [...memoryAlertStore].reverse();
}

function clearStores() {
  memorySceneStore.clear();
  memoryAlertStore.length = 0;
}

module.exports = {
  persistSceneResult,
  getSceneResultBySession,
  getAlerts,
  normalizePayload,
  clearStores
};

