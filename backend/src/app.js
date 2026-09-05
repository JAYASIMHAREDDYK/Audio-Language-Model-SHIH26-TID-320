const http = require('http');
const express = require('express');
const cors = require('cors');

const sceneRoutes = require('./modules/scene/routes');
const sessionRoutes = require('./modules/sessions/routes');
const alertRoutes = require('./modules/alerts/routes');
const { initWebSocketServer } = require('./websocket/server');

const app = express();
const server = http.createServer(app);

app.use(cors());
app.use(express.json());

const mlClient = require('./modules/sessions/mlClient');

let upload;
try {
  const multer = require('multer');
  upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });
} catch {
  upload = { single: () => (req, res, next) => next() };
}

app.use('/api/v1/sessions', sessionRoutes);
app.use('/api/v1/sessions', sceneRoutes);
app.use('/api/v1/alerts', alertRoutes);

async function handleAnalyzeRoute(req, res) {
  try {
    const file = req.file;
    const question = req.body?.question || req.query?.question || '';
    const audioBuffer = file ? file.buffer : (req.body?.audio_file ? Buffer.from(req.body.audio_file) : Buffer.from('mock audio'));
    const sessionId = req.body?.session_id || `sess_analyze_${Date.now()}`;

    const llmModel = req.body?.llm_model || req.query?.llm_model || 'gpt-4o-mini';

    try {
      const mlResponse = await mlClient.analyzeAudio({
        sessionId,
        audioBuffer,
        filename: file?.originalname || 'audio.wav',
        mimeType: file?.mimetype || 'audio/wav',
        question,
        llmModel
      });

      return res.status(200).json({
        answer: mlResponse.answer,
        confidence: mlResponse.confidence,
        speech: mlResponse.speech || {},
        speakers: mlResponse.speakers || [],
        audio_events: mlResponse.audio_events || [],
        paralinguistic: mlResponse.paralinguistic || {},
        scene: mlResponse.scene || {},
        evidence: mlResponse.evidence || []
      });
    } catch (mlErr) {
      console.error('[Backend] ML service call error:', mlErr.message);
      return res.status(502).json({ error: `ML Service Error: ${mlErr.message}` });
    }
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
}

app.post('/analyze', upload.single('audio_file'), handleAnalyzeRoute);
app.post('/api/v1/analyze', upload.single('audio_file'), handleAnalyzeRoute);

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'ALM-NHCE Scene & Alerts API' });
});

initWebSocketServer(server);

const PORT = process.env.PORT || 4000;

if (process.env.NODE_ENV !== 'test') {
  server.listen(PORT, () => {
    console.log(`[Backend] Server listening on http://localhost:${PORT}`);
    console.log(`[Backend] WebSocket available on ws://localhost:${PORT}/ws`);
  });
}

module.exports = { app, server };
