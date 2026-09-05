/**
 * routes.js
 * Express routing definitions for Sessions and Analytics read endpoints.
 *
 * Contract:
 * - GET /api/v1/sessions?page=1&limit=10
 * - GET /api/v1/sessions/:id/result
 * - GET /api/v1/analytics/summary
 */

const express = require('express');
const router = express.Router();
const controller = require('./controller');

// Mount handlers with flexible routing to support mounting at /api/v1, /api/v1/sessions, or /api/v1/analytics

// 1. Explicit full paths (when mounted at /api/v1 or app root)
router.get('/sessions', controller.listSessions);
router.get('/sessions/:id/result', controller.getSessionResult);
router.get('/analytics/summary', controller.getAnalyticsSummary);

// 2. Relative aliases (when mounted at /api/v1/analytics or /api/v1/sessions)
router.get('/summary', controller.getAnalyticsSummary);

// Fallback relative route for root and :id/result when mounted at /api/v1/sessions
router.get('/', (req, res, next) => {
  if (req.baseUrl && req.baseUrl.includes('analytics')) {
    return controller.getAnalyticsSummary(req, res);
  }
  return controller.listSessions(req, res);
});

router.get('/:id/result', controller.getSessionResult);

module.exports = router;
