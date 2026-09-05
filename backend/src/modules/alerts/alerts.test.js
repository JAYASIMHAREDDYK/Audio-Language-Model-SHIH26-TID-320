const test = require('node:test');
const assert = require('node:assert/strict');
const { evaluate, CRITICAL_SOUND_EVENTS } = require('./ruleEngine');

test('Alert Rule Engine - Critical Sound Events', async (t) => {
  await t.test('triggers high severity alert for gunshot', () => {
    const input = {
      sound_events: [{ label: 'gunshot', confidence: 0.95 }],
      emotion: { primary: 'neutral', arousal: 'low', confidence: 0.8 },
      transcript: { text: 'routine conversation' }
    };
    const alerts = evaluate(input);
    assert.equal(alerts.length, 1);
    assert.equal(alerts[0].type, 'sound_event');
    assert.equal(alerts[0].severity, 'high');
    assert.match(alerts[0].message, /gunshot/i);
  });

  await t.test('triggers high severity for all critical sound labels', () => {
    CRITICAL_SOUND_EVENTS.forEach((label) => {
      const alerts = evaluate({ sound_events: [label] });
      assert.ok(alerts.length >= 1, `Failed to trigger for sound: ${label}`);
      assert.equal(alerts[0].severity, 'high');
      assert.equal(alerts[0].type, 'sound_event');
    });
  });

  await t.test('ignores non-critical ambient sounds', () => {
    const input = {
      sound_events: [
        { label: 'bird_chirping', confidence: 0.99 },
        { label: 'keyboard_typing', confidence: 0.88 }
      ]
    };
    const alerts = evaluate(input);
    assert.equal(alerts.length, 0);
  });
});

test('Alert Rule Engine - Emotion Distress', async (t) => {
  await t.test('triggers medium severity for fear with high arousal & confidence > 0.7', () => {
    const input = {
      emotion: { primary: 'fear', arousal: 'high', confidence: 0.85 }
    };
    const alerts = evaluate(input);
    assert.equal(alerts.length, 1);
    assert.equal(alerts[0].type, 'emotion');
    assert.equal(alerts[0].severity, 'medium');
    assert.match(alerts[0].message, /FEAR/);
  });

  await t.test('triggers medium severity for anger with high arousal & confidence > 0.7', () => {
    const input = {
      emotion: { primary: 'anger', arousal: 'high', confidence: 0.75 }
    };
    const alerts = evaluate(input);
    assert.equal(alerts.length, 1);
    assert.equal(alerts[0].type, 'emotion');
    assert.equal(alerts[0].severity, 'medium');
    assert.match(alerts[0].message, /ANGER/);
  });

  await t.test('does not trigger if confidence <= 0.7', () => {
    const input = {
      emotion: { primary: 'fear', arousal: 'high', confidence: 0.69 }
    };
    const alerts = evaluate(input);
    assert.equal(alerts.length, 0);
  });

  await t.test('does not trigger if arousal is low', () => {
    const input = {
      emotion: { primary: 'fear', arousal: 'low', confidence: 0.95 }
    };
    const alerts = evaluate(input);
    assert.equal(alerts.length, 0);
  });
});

test('Alert Rule Engine - Transcript Keywords', async (t) => {
  await t.test('triggers medium severity for distress keywords', () => {
    const input = {
      transcript: { text: 'Someone call 911 we need police and emergency support now!' }
    };
    const alerts = evaluate(input);
    assert.equal(alerts.length, 1);
    assert.equal(alerts[0].type, 'keyword');
    assert.equal(alerts[0].severity, 'medium');
    assert.ok(alerts[0].details.matched_keywords.includes('police'));
    assert.ok(alerts[0].details.matched_keywords.includes('emergency'));
  });

  await t.test('does not trigger false positive for word substrings', () => {
    const input = {
      transcript: { text: 'The tutorial was very helpful and pleasant.' }
    };
    const alerts = evaluate(input);
    assert.equal(alerts.length, 0);
  });
});

test('Alert Rule Engine - Multi-Trigger & Edge Cases', async (t) => {
  await t.test('handles simultaneous sound, emotion, and keyword triggers', () => {
    const input = {
      sound_events: [{ label: 'explosion', confidence: 0.98 }],
      emotion: { primary: 'fear', arousal: 'high', confidence: 0.92 },
      transcript: { text: 'help! there is a huge fire!' }
    };
    const alerts = evaluate(input);
    assert.equal(alerts.length, 3);
    const types = alerts.map((a) => a.type);
    assert.ok(types.includes('sound_event'));
    assert.ok(types.includes('emotion'));
    assert.ok(types.includes('keyword'));
  });

  await t.test('handles empty or malformed inputs without throwing', () => {
    assert.deepEqual(evaluate(null), []);
    assert.deepEqual(evaluate(undefined), []);
    assert.deepEqual(evaluate({}), []);
    assert.deepEqual(evaluate({ sound_events: null, emotion: null, transcript: null }), []);
  });
});

