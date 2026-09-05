const test = require('node:test');
const assert = require('node:assert/strict');
const { persistSceneResult, getSceneResultBySession, getAlerts, clearStores, normalizePayload } = require('./persistScene');
const { registerClient, clearAllSubscribers } = require('../../websocket/broadcast');

test('Scene Intelligence & Persistence Module', async (t) => {
  t.beforeEach(() => {
    clearStores();
    clearAllSubscribers();
  });

  await t.test('normalizes complex ML result payload correctly', () => {
    const sessionId = 'test-session-uuid-123';
    const mlInput = {
      transcript: { text: 'Testing audio pipeline', lang: 'en' },
      sound_events: [{ label: 'glass_breaking', confidence: 0.91 }],
      emotion: { primary: 'fear', arousal: 'high', confidence: 0.88 },
      speakers: [{ speaker_id: 'speaker_0', start: 0, end: 2.5 }],
      model_insight: { label: 'high_distress', confidence: 0.89 },
      reasoning_trace: 'Step 1: Detected breaking glass. Step 2: Correlated with vocal fear.',
      fusion_summary: 'Critical distress situation detected with audio and vocal cues.'
    };

    const normalized = normalizePayload(sessionId, mlInput);
    assert.equal(normalized.session_id, sessionId);
    assert.equal(normalized.transcript_text, 'Testing audio pipeline');
    assert.equal(normalized.transcript_lang, 'en');
    assert.equal(normalized.sound_events.length, 1);
    assert.equal(normalized.model_insight.label, 'high_distress');
    assert.equal(normalized.fusion_summary, mlInput.fusion_summary);
  });

  await t.test('persists scene result and automatically generates alerts', async () => {
    const sessionId = 'session-alert-test-01';
    const mlInput = {
      transcript: { text: 'Help me, someone is breaking the door!' },
      sound_events: [{ label: 'screaming', confidence: 0.89 }, { label: 'glass_breaking', confidence: 0.94 }],
      emotion: { primary: 'fear', arousal: 'high', confidence: 0.85 }
    };

    const result = await persistSceneResult({ sessionId, mlResult: mlInput });
    assert.ok(result.sceneResult);
    assert.equal(result.sceneResult.session_id, sessionId);
    assert.ok(result.alerts.length >= 2);

    const retrieved = await getSceneResultBySession(sessionId);
    assert.ok(retrieved);
    assert.equal(retrieved.session_id, sessionId);
    assert.equal(retrieved.alerts.length, result.alerts.length);
  });

  await t.test('filters alerts by session_id', async () => {
    await persistSceneResult({
      sessionId: 'session-A',
      mlResult: { sound_events: [{ label: 'gunshot' }] }
    });
    await persistSceneResult({
      sessionId: 'session-B',
      mlResult: { transcript: { text: 'fire emergency help' } }
    });

    const alertsA = await getAlerts({ sessionId: 'session-A' });
    const alertsB = await getAlerts({ sessionId: 'session-B' });
    const allAlerts = await getAlerts();

    assert.equal(alertsA.length, 1);
    assert.equal(alertsA[0].type, 'sound_event');
    assert.equal(alertsB.length, 1);
    assert.equal(alertsB[0].type, 'keyword');
    assert.equal(allAlerts.length, 2);
  });

  await t.test('broadcasts result_ready and alert_raised to registered WebSocket clients', async () => {
    const sessionId = 'ws-test-session';
    const receivedMessages = [];

    const mockWs = {
      readyState: 1,
      send: (data) => receivedMessages.push(JSON.parse(data))
    };

    registerClient(sessionId, mockWs);

    await persistSceneResult({
      sessionId,
      mlResult: {
        transcript: { text: 'Everything is fine' },
        sound_events: [{ label: 'smoke_alarm', confidence: 0.99 }]
      }
    });

    assert.ok(receivedMessages.length >= 2);
    const resultReadyMsg = receivedMessages.find((m) => m.event === 'result_ready');
    const alertRaisedMsg = receivedMessages.find((m) => m.event === 'alert_raised');

    assert.ok(resultReadyMsg);
    assert.equal(resultReadyMsg.sessionId, sessionId);
    assert.ok(alertRaisedMsg);
    assert.equal(alertRaisedMsg.payload.severity, 'high');
  });
});

