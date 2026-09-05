const { test, describe, beforeEach, afterEach } = require('node:test');
const assert = require('node:assert');
const sessionsController = require('./controller');
const mlClient = require('./mlClient');

// Helper to simulate express req/res
function createMockRes() {
  const res = {
    statusCode: 200,
    data: null,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(obj) {
      this.data = obj;
      return this;
    }
  };
  return res;
}

describe('Sessions Module Tests', () => {
  const originalMockMl = process.env.MOCK_ML;

  beforeEach(() => {
    process.env.MOCK_ML = 'true';
    sessionsController._clearMemorySessions();
  });

  afterEach(() => {
    process.env.MOCK_ML = originalMockMl;
  });

  describe('ML Client Contract Unit Tests', () => {
    test('getMockMlResponse returns exact contract JSON structure', () => {
      const mock = mlClient.getMockMlResponse('sess_test_123');
      assert.strictEqual(mock.session_id, 'sess_test_123');
      assert.strictEqual(typeof mock.duration_sec, 'number');
      assert.ok(mock.transcript);
      assert.ok(mock.transcript.text);
      assert.ok(Array.isArray(mock.sound_events));
      assert.ok(mock.emotion);
      assert.ok(mock.emotion.primary);
      assert.ok(mock.speakers);
      assert.ok(mock.model_insight);
      assert.strictEqual(typeof mock.model_insight.label, 'string');
      assert.ok(mock.reasoning);
      assert.strictEqual(typeof mock.processing_ms, 'number');
    });

    test('analyzeAudio returns result in mock mode', async () => {
      const result = await mlClient.analyzeAudio({
        sessionId: 'sess_abc999',
        audioBuffer: Buffer.from('RIFF....WAVEfmt....data....'),
        filename: 'alarm.wav'
      });

      assert.strictEqual(result.session_id, 'sess_abc999');
      assert.ok(result.transcript.text.length > 0);
      assert.ok(result.sound_events.length > 0);
    });
  });

  describe('Sessions Controller Contract Endpoint Tests', () => {
    test('POST /api/v1/sessions creates session with status pending (upload)', async () => {
      const req = {
        body: { source_type: 'upload' }
      };
      const res = createMockRes();

      await sessionsController.createSession(req, res);

      assert.strictEqual(res.statusCode, 201);
      assert.ok(res.data.session_id);
      assert.strictEqual(res.data.status, 'pending');
      assert.ok(res.data.created_at);
    });

    test('POST /api/v1/sessions creates session with status pending (mic)', async () => {
      const req = {
        body: { source_type: 'mic' }
      };
      const res = createMockRes();

      await sessionsController.createSession(req, res);

      assert.strictEqual(res.statusCode, 201);
      assert.ok(res.data.session_id);
      assert.strictEqual(res.data.status, 'pending');
    });

    test('POST /api/v1/sessions rejects invalid source_type with 400', async () => {
      const req = {
        body: { source_type: 'stream' }
      };
      const res = createMockRes();

      await sessionsController.createSession(req, res);

      assert.strictEqual(res.statusCode, 400);
      assert.match(res.data.error, /source_type/);
    });

    test('POST /api/v1/sessions/:id/audio responds 202 immediately and processes async ML in background', async () => {
      // 1. Create session first
      const createReq = { body: { source_type: 'upload' } };
      const createRes = createMockRes();
      await sessionsController.createSession(createReq, createRes);
      const sessionId = createRes.data.session_id;

      // 2. Setup listener for Lahari handoff event
      let eventFired = false;
      let handoffPayload = null;

      sessionsController.sessionEvents.on('ml_result_ready', (data) => {
        eventFired = true;
        handoffPayload = data;
      });

      // 3. Upload audio
      const uploadReq = {
        params: { id: sessionId },
        file: {
          buffer: Buffer.from('fake-audio-content'),
          originalname: 'sample.wav',
          mimetype: 'audio/wav'
        }
      };
      const uploadRes = createMockRes();

      const startTime = Date.now();
      await sessionsController.uploadAudio(uploadReq, uploadRes);
      const durationMs = Date.now() - startTime;

      // Critical NFR: MUST respond 202 processing right away, not blocking on ML
      assert.strictEqual(uploadRes.statusCode, 202);
      assert.strictEqual(uploadRes.data.session_id, sessionId);
      assert.strictEqual(uploadRes.data.status, 'processing');
      assert.ok(durationMs < 50, `Immediate response took ${durationMs}ms, expected < 50ms`);

      // 4. Wait for background ML task to complete
      await new Promise((resolve) => setTimeout(resolve, 450));

      assert.strictEqual(eventFired, true, 'ml_result_ready event should have fired for Lahari');
      assert.strictEqual(handoffPayload.session_id, sessionId);
      assert.ok(handoffPayload.result);
      assert.strictEqual(handoffPayload.result.session_id, sessionId);

      // 5. Query session via GET /api/v1/sessions/:id
      const getReq = { params: { id: sessionId } };
      const getRes = createMockRes();
      await sessionsController.getSession(getReq, getRes);

      assert.strictEqual(getRes.statusCode, 200);
      assert.strictEqual(getRes.data.session_id, sessionId);
      assert.strictEqual(getRes.data.status, 'done');
      assert.ok(getRes.data.result);
      assert.strictEqual(getRes.data.result.session_id, sessionId);
    });

    test('POST /api/v1/sessions/:id/audio returns 404 for unknown session', async () => {
      const uploadReq = {
        params: { id: 'non_existent_uuid' },
        file: { buffer: Buffer.from('abc') }
      };
      const uploadRes = createMockRes();

      await sessionsController.uploadAudio(uploadReq, uploadRes);
      assert.strictEqual(uploadRes.statusCode, 404);
    });

    test('POST /api/v1/sessions/:id/audio returns 400 when audio_file is missing', async () => {
      const createReq = { body: { source_type: 'upload' } };
      const createRes = createMockRes();
      await sessionsController.createSession(createReq, createRes);
      const sessionId = createRes.data.session_id;

      const uploadReq = {
        params: { id: sessionId }
      };
      const uploadRes = createMockRes();

      await sessionsController.uploadAudio(uploadReq, uploadRes);
      assert.strictEqual(uploadRes.statusCode, 400);
      assert.match(uploadRes.data.error, /audio_file/);
    });
  });
});
