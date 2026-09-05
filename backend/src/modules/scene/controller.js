const { getSceneResultBySession, persistSceneResult } = require('./persistScene');

async function getSessionResult(req, res) {
  try {
    const sessionId = req.params.id || req.params.sessionId;
    if (!sessionId) {
      return res.status(400).json({ error: 'Session ID is required' });
    }

    const result = await getSceneResultBySession(sessionId);
    if (!result) {
      return res.status(404).json({
        error: 'Scene result not found for this session',
        session_id: sessionId
      });
    }

    return res.status(200).json({
      success: true,
      ...result
    });
  } catch (err) {
    return res.status(500).json({
      error: 'Internal server error retrieving scene result',
      details: err.message
    });
  }
}

async function ingestResult(req, res) {
  try {
    const sessionId = req.params.id || req.body.session_id || req.body.sessionId;
    const mlResult = req.body.mlResult || req.body.result || req.body;

    if (!sessionId) {
      return res.status(400).json({ error: 'Session ID is required for ingestion' });
    }

    const persisted = await persistSceneResult({ sessionId, mlResult });
    return res.status(201).json({
      success: true,
      message: 'Scene result persisted and alerts evaluated successfully',
      ...persisted
    });
  } catch (err) {
    return res.status(500).json({
      error: 'Failed to ingest scene result',
      details: err.message
    });
  }
}

module.exports = {
  getSessionResult,
  ingestResult
};

