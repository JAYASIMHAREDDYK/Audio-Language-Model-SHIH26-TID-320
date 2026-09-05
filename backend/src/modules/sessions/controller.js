const crypto = require('crypto');
const EventEmitter = require('events');
const mlClient = require('./mlClient');

// Event bus for session lifecycle events (e.g., handing off to Lahari)
const sessionEvents = new EventEmitter();

// In-memory sessions store fallback for testing or when DB is not yet connected
const memorySessions = new Map();
const memoryResults = new Map();

/**
 * DB query helper
 */
async function queryDb(text, params = []) {
  try {
    const clientModule = require('../../db/client');
    if (clientModule && typeof clientModule.query === 'function') {
      return await clientModule.query(text, params);
    }
    if (clientModule && clientModule.pool && typeof clientModule.pool.query === 'function') {
      return await clientModule.pool.query(text, params);
    }
  } catch {
    // client.js not configured
  }

  if (process.env.DB_URL || process.env.DATABASE_URL) {
    try {
      const { Pool } = require('pg');
      if (!queryDb._pool) {
        queryDb._pool = new Pool({
          connectionString: process.env.DB_URL || process.env.DATABASE_URL,
        });
      }
      return await queryDb._pool.query(text, params);
    } catch {
      // pg connection fallback
    }
  }

  // In-memory fallback
  const lowerText = text.toLowerCase();

  if (lowerText.includes('insert into audio_sessions')) {
    const [id, user_id, source_type, status] = params;
    const session = {
      id: id || crypto.randomUUID(),
      user_id: user_id || null,
      source_type,
      status: status || 'pending',
      created_at: new Date().toISOString()
    };
    memorySessions.set(session.id, session);
    return { rows: [session], rowCount: 1 };
  }

  if (lowerText.includes('select') && lowerText.includes('audio_sessions') && lowerText.includes('id = $1')) {
    const id = params[0];
    const session = memorySessions.get(id);
    return { rows: session ? [session] : [], rowCount: session ? 1 : 0 };
  }

  if (lowerText.includes('update audio_sessions') && lowerText.includes('set status = $1')) {
    const [status, id] = params;
    const session = memorySessions.get(id);
    if (session) {
      session.status = status;
      memorySessions.set(id, session);
      return { rows: [session], rowCount: 1 };
    }
    return { rows: [], rowCount: 0 };
  }

  if (lowerText.includes('select') && lowerText.includes('scene_results') && lowerText.includes('session_id = $1')) {
    const id = params[0];
    const result = memoryResults.get(id);
    return { rows: result ? [result] : [], rowCount: result ? 1 : 0 };
  }

  return { rows: [], rowCount: 0 };
}

/**
 * Controller: POST /api/v1/sessions
 * Contract: { source_type: "upload"|"mic" } -> 201 { session_id, status: "pending", created_at }
 */
async function createSession(req, res, next) {
  try {
    const { source_type } = req.body || {};

    if (!source_type || !['upload', 'mic'].includes(source_type)) {
      return res.status(400).json({
        error: 'Invalid source_type. Must be either "upload" or "mic"'
      });
    }

    const sessionId = crypto.randomUUID();
    const userId = req.user?.user_id || null;
    const status = 'pending';

    const insertResult = await queryDb(
      'INSERT INTO audio_sessions (id, user_id, source_type, status, created_at) VALUES ($1, $2, $3, $4, now()) RETURNING id, source_type, status, created_at',
      [sessionId, userId, source_type, status]
    );

    const created = (insertResult.rows && insertResult.rows[0]) || {
      id: sessionId,
      status: 'pending',
      created_at: new Date().toISOString()
    };

    return res.status(201).json({
      session_id: created.id,
      status: created.status,
      created_at: created.created_at
    });
  } catch (error) {
    if (next) return next(error);
    return res.status(error.statusCode || 500).json({ error: error.message });
  }
}

/**
 * Controller: POST /api/v1/sessions/:id/audio
 * Contract:
 *  - multipart/form-data { audio_file }
 *  - respond 202 { session_id, status: "processing" } IMMEDIATELY
 *  - async forward to ML POST /analyze
 *  - handoff result to Lahari's persistence layer
 */
async function uploadAudio(req, res, next) {
  try {
    const sessionId = req.params.id;
    if (!sessionId) {
      return res.status(400).json({ error: 'Session ID is required in URL parameters' });
    }

    // Verify session exists
    const sessionRes = await queryDb('SELECT id, status FROM audio_sessions WHERE id = $1', [sessionId]);
    if (!sessionRes.rows || sessionRes.rows.length === 0) {
      return res.status(404).json({ error: `Session not found: ${sessionId}` });
    }

    // Check for uploaded file
    const file = req.file;
    if (!file && !req.body?.audio_file) {
      return res.status(400).json({ error: 'audio_file is required as multipart/form-data' });
    }

    const audioBuffer = file ? (file.buffer || file) : Buffer.from(req.body.audio_file);
    const filename = file?.originalname || 'audio.wav';
    const mimeType = file?.mimetype || 'audio/wav';

    // Update status to processing
    await queryDb('UPDATE audio_sessions SET status = $1 WHERE id = $2', ['processing', sessionId]);

    // Critical NFR: Respond 202 immediately to not block HTTP thread
    res.status(202).json({
      session_id: sessionId,
      status: 'processing'
    });

    // Execute ML call asynchronously in the background
    setImmediate(async () => {
      console.log(`[Sessions] Starting ML analysis for session ${sessionId}...`);
      try {
        const mlResult = await mlClient.analyzeAudio({
          sessionId,
          audioBuffer,
          filename,
          mimeType,
          question: req.body?.question,
          languageHint: req.body?.language_hint,
          spokenTranscript: req.body?.spoken_transcript
        });

        console.log(`[Sessions] ML analysis complete for session ${sessionId}`);

        // Update session status to done
        await queryDb('UPDATE audio_sessions SET status = $1 WHERE id = $2', ['done', sessionId]);

        // Save into memory store for fast retrieval if available
        memoryResults.set(sessionId, mlResult);

        // Handoff to Lahari via EventEmitter hook
        sessionEvents.emit('ml_result_ready', {
          session_id: sessionId,
          result: mlResult
        });

        // Persist and broadcast via WebSocket
        try {
          const sceneModule = require('../scene/persistScene');
          if (sceneModule && typeof sceneModule.persistSceneResult === 'function') {
            const persisted = await sceneModule.persistSceneResult({ sessionId, mlResult });
            console.log(`[Sessions] Result persisted & broadcast for session ${sessionId}`);
          } else {
            console.warn(`[Sessions] persistSceneResult not found, result saved to memory only`);
          }
        } catch (err) {
          console.error(`[Sessions] Error in persistSceneResult for ${sessionId}:`, err.message);
        }
      } catch (mlError) {
        console.error(`[Sessions] ML analysis FAILED for session ${sessionId}:`, mlError.message);
        // Mark session as failed
        await queryDb('UPDATE audio_sessions SET status = $1 WHERE id = $2', ['failed', sessionId]);

        sessionEvents.emit('ml_result_failed', {
          session_id: sessionId,
          error: mlError.message
        });
      }
    });
  } catch (error) {
    if (next) return next(error);
    return res.status(error.statusCode || 500).json({ error: error.message });
  }
}

/**
 * Controller: GET /api/v1/sessions/:id
 * Contract: -> 200 { session_id, status, created_at, result | null }
 */
async function getSession(req, res, next) {
  try {
    const sessionId = req.params.id;
    if (!sessionId) {
      return res.status(400).json({ error: 'Session ID is required' });
    }

    const sessionRes = await queryDb('SELECT id, status, created_at FROM audio_sessions WHERE id = $1', [sessionId]);
    if (!sessionRes.rows || sessionRes.rows.length === 0) {
      return res.status(404).json({ error: 'Session not found' });
    }

    const session = sessionRes.rows[0];

    // Attempt to fetch result from scene_results or memory store
    let result = memoryResults.get(sessionId) || null;

    if (!result && session.status === 'done') {
      try {
        const sceneRes = await queryDb('SELECT * FROM scene_results WHERE session_id = $1 LIMIT 1', [sessionId]);
        if (sceneRes.rows && sceneRes.rows.length > 0) {
          result = sceneRes.rows[0];
        }
      } catch {
        // scene_results table might not be queried yet
      }
    }

    return res.status(200).json({
      session_id: session.id,
      status: session.status,
      created_at: session.created_at,
      result
    });
  } catch (error) {
    if (next) return next(error);
    return res.status(error.statusCode || 500).json({ error: error.message });
  }
}

/**
 * Helper to clear test cache
 */
function _clearMemorySessions() {
  memorySessions.clear();
  memoryResults.clear();
  sessionEvents.removeAllListeners();
}

module.exports = {
  createSession,
  uploadAudio,
  getSession,
  sessionEvents,
  _clearMemorySessions
};
