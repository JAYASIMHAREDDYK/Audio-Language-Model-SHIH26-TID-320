const express = require('express');
const router = express.Router();
const { getSessionResult, ingestResult } = require('./controller');

router.get('/:id/result', getSessionResult);
router.post('/:id/result', ingestResult);

module.exports = router;

