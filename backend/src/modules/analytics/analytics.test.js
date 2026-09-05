/**
 * analytics.test.js
 * Unit and integration tests for History, Session Detail, and Analytics queries & controllers.
 * Uses Node's built-in assert module for zero-dependency portability.
 */

const assert = require('assert');
const queries = require('./queries');
const controller = require('./controller');

// Mock response creator for controller tests
function createMockRes() {
  return {
    statusCode: 200,
    body: null,
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(data) {
      this.body = data;
      return this;
    }
  };
}

async function runTests() {
  console.log('--- Starting Analytics & History Test Suite ---');
  let passed = 0;
  let failed = 0;

  async function test(name, fn) {
    try {
      await fn();
      console.log(`  PASS: ${name}`);
      passed++;
    } catch (err) {
      console.error(`  FAIL: ${name}`);
      console.error(`    -> ${err.message}`);
      failed++;
    }
  }

  // 1. queries.getSessionsList tests
  await test('queries.getSessionsList returns paginated structure with contract fields', async () => {
    const res = await queries.getSessionsList({ page: 1, limit: 3 });
    assert(res, 'Result should be defined');
    assert(Array.isArray(res.sessions), 'sessions should be an array');
    assert.strictEqual(res.sessions.length, 3, 'Should respect limit of 3');
    assert(typeof res.total === 'number', 'total should be a number');

    const first = res.sessions[0];
    assert(first.session_id, 'session_id is required');
    assert(first.created_at, 'created_at is required');
    assert(first.status, 'status is required');
    assert(typeof first.has_alert === 'boolean', 'has_alert must be a boolean');
    assert(typeof first.summary === 'string', 'summary must be a string');
  });

  await test('queries.getSessionsList handles pagination offset correctly', async () => {
    const page1 = await queries.getSessionsList({ page: 1, limit: 2 });
    const page2 = await queries.getSessionsList({ page: 2, limit: 2 });
    assert.notStrictEqual(
      page1.sessions[0].session_id,
      page2.sessions[0].session_id,
      'Page 2 should return different items than Page 1'
    );
  });

  // 2. queries.getSessionResult tests
  await test('queries.getSessionResult returns full ML result + alerts', async () => {
    const sessionId = 'a1b2c3d4-0001-4000-8000-000000000001';
    const result = await queries.getSessionResult(sessionId);
    assert(result, 'Session result should be found');
    assert.strictEqual(result.session_id, sessionId);
    assert(result.transcript && typeof result.transcript.text === 'string', 'transcript.text must exist');
    assert(Array.isArray(result.sound_events), 'sound_events must be an array');
    assert(result.emotion && result.emotion.primary, 'emotion.primary must exist');
    assert(result.speakers && typeof result.speakers.count === 'number', 'speakers.count must exist');
    assert(result.model_insight && result.model_insight.label, 'model_insight.label must exist');
    assert(result.reasoning && result.reasoning.trace, 'reasoning.trace must exist');
    assert(Array.isArray(result.alerts), 'alerts must be an array');
    assert(result.alerts.length > 0, 'Should contain alerts for high hazard session');

    const alert = result.alerts[0];
    assert(alert.id, 'alert id required');
    assert(alert.type, 'alert type required');
    assert(['low', 'medium', 'high'].includes(alert.severity), 'severity must be low/medium/high');
    assert(alert.message, 'alert message required');
  });

  await test('queries.getSessionResult returns null for unknown session ID', async () => {
    const result = await queries.getSessionResult('non-existent-uuid-9999');
    assert.strictEqual(result, null, 'Unknown session should return null');
  });

  // 3. queries.getAnalyticsSummary tests
  await test('queries.getAnalyticsSummary returns aggregates conforming to contract', async () => {
    const summary = await queries.getAnalyticsSummary();
    assert(summary, 'Summary should be defined');
    assert(typeof summary.total_sessions === 'number', 'total_sessions must be number');
    assert(typeof summary.total_alerts === 'number', 'total_alerts must be number');
    assert(typeof summary.alerts_by_type === 'object', 'alerts_by_type must be object');
    assert(Array.isArray(summary.top_sound_events), 'top_sound_events must be array');

    if (summary.top_sound_events.length > 0) {
      assert(summary.top_sound_events[0].label, 'Event label required');
      assert(typeof summary.top_sound_events[0].count === 'number', 'Event count required');
    }
  });

  // 4. controller handler tests
  await test('controller.listSessions responds with 200 and paginated json', async () => {
    const req = { query: { page: '1', limit: '5' } };
    const res = createMockRes();
    await controller.listSessions(req, res);
    assert.strictEqual(res.statusCode, 200);
    assert(res.body.sessions && Array.isArray(res.body.sessions));
    assert.strictEqual(res.body.page, 1);
    assert.strictEqual(res.body.limit, 5);
  });

  await test('controller.getSessionResult responds with 200 for existing session', async () => {
    const req = { params: { id: 'a1b2c3d4-0001-4000-8000-000000000001' } };
    const res = createMockRes();
    await controller.getSessionResult(req, res);
    assert.strictEqual(res.statusCode, 200);
    assert.strictEqual(res.body.session_id, 'a1b2c3d4-0001-4000-8000-000000000001');
    assert(Array.isArray(res.body.alerts));
  });

  await test('controller.getSessionResult responds with 404 for unknown session', async () => {
    const req = { params: { id: 'unknown-id' } };
    const res = createMockRes();
    await controller.getSessionResult(req, res);
    assert.strictEqual(res.statusCode, 404);
    assert(res.body.error);
  });

  await test('controller.getAnalyticsSummary responds with 200 and summary object', async () => {
    const req = {};
    const res = createMockRes();
    await controller.getAnalyticsSummary(req, res);
    assert.strictEqual(res.statusCode, 200);
    assert(res.body.total_sessions >= 0);
    assert(res.body.alerts_by_type);
  });

  console.log(`\nTest Summary: ${passed} passed, ${failed} failed`);
  if (failed > 0) {
    process.exit(1);
  }
}

if (require.main === module) {
  runTests();
}

module.exports = { runTests };
