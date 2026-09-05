/**
 * controller.js
 * Request handlers for History, Session Detail, and Analytics endpoints.
 * Read-only aggregation controller.
 */

const queries = require('./queries');

/**
 * GET /api/v1/sessions?page=1&limit=10
 * Returns paginated sessions list with alert flag and summary.
 */
async function listSessions(req, res) {
  try {
    const page = parseInt(req.query.page, 10) || 1;
    const limit = parseInt(req.query.limit, 10) || 10;

    if (page < 1 || limit < 1) {
      return res.status(400).json({
        error: 'Invalid pagination parameters. Page and limit must be positive integers.'
      });
    }

    const { sessions, total } = await queries.getSessionsList({ page, limit });

    return res.status(200).json({
      sessions,
      total,
      page,
      limit
    });
  } catch (error) {
    console.error('[Analytics Controller] Error in listSessions:', error);
    return res.status(500).json({
      error: 'Failed to retrieve sessions history',
      message: error.message
    });
  }
}

/**
 * GET /api/v1/sessions/:id/result
 * Returns full ML result and alerts array for a given session.
 */
async function getSessionResult(req, res) {
  try {
    const { id } = req.params;

    if (!id) {
      return res.status(400).json({ error: 'Session ID parameter is required.' });
    }

    const result = await queries.getSessionResult(id);

    if (!result) {
      return res.status(404).json({
        error: `Session result for ID '${id}' was not found.`,
        session_id: id
      });
    }

    return res.status(200).json(result);
  } catch (error) {
    console.error(`[Analytics Controller] Error in getSessionResult for ${req.params.id}:`, error);
    return res.status(500).json({
      error: 'Failed to retrieve session detail result',
      message: error.message
    });
  }
}

/**
 * GET /api/v1/analytics/summary
 * Returns aggregated statistics across all audio sessions and alerts.
 */
async function getAnalyticsSummary(req, res) {
  try {
    const summary = await queries.getAnalyticsSummary();
    return res.status(200).json(summary);
  } catch (error) {
    console.error('[Analytics Controller] Error in getAnalyticsSummary:', error);
    return res.status(500).json({
      error: 'Failed to generate analytics summary',
      message: error.message
    });
  }
}

module.exports = {
  listSessions,
  getSessionResult,
  getAnalyticsSummary
};
