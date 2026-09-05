const express = require('express');
const router = express.Router();
const { getAlerts } = require('../scene/persistScene');

router.get('/', async (req, res) => {
  try {
    const { session_id } = req.query;
    const alerts = await getAlerts({ sessionId: session_id });

    return res.status(200).json({
      success: true,
      alerts: alerts.map((a) => ({
        alert_id: a.id || a.alert_id,
        session_id: a.session_id,
        type: a.type,
        severity: a.severity,
        message: a.message,
        details: a.details || null,
        created_at: a.created_at
      }))
    });
  } catch (error) {
    return res.status(500).json({
      success: false,
      error: 'Failed to retrieve alerts',
      details: error.message
    });
  }
});

module.exports = router;

