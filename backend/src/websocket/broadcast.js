const sessionSubscribers = new Map();
const globalSubscribers = new Set();

function registerClient(sessionId, ws) {
  if (!ws) return;

  if (sessionId && sessionId !== 'all' && sessionId !== '*') {
    if (!sessionSubscribers.has(sessionId)) {
      sessionSubscribers.set(sessionId, new Set());
    }
    sessionSubscribers.get(sessionId).add(ws);
  } else {
    globalSubscribers.add(ws);
  }

  const cleanup = () => unregisterClient(sessionId, ws);
  if (typeof ws.once === 'function') {
    ws.once('close', cleanup);
    ws.once('error', cleanup);
  } else if (typeof ws.addEventListener === 'function') {
    ws.addEventListener('close', cleanup, { once: true });
    ws.addEventListener('error', cleanup, { once: true });
  }
}

function unregisterClient(sessionId, ws) {
  if (sessionId && sessionSubscribers.has(sessionId)) {
    const set = sessionSubscribers.get(sessionId);
    set.delete(ws);
    if (set.size === 0) {
      sessionSubscribers.delete(sessionId);
    }
  }
  globalSubscribers.delete(ws);
}

function sendToSocket(ws, data) {
  try {
    const raw = typeof data === 'string' ? data : JSON.stringify(data);
    if (ws && (ws.readyState === 1 || ws.readyState === undefined) && typeof ws.send === 'function') {
      ws.send(raw);
      return true;
    }
  } catch (err) {
    return false;
  }
  return false;
}

function broadcast(sessionId, eventName, payload) {
  const message = {
    event: eventName,
    sessionId: sessionId || null,
    payload,
    timestamp: new Date().toISOString()
  };

  let count = 0;

  if (sessionId && sessionSubscribers.has(sessionId)) {
    for (const ws of sessionSubscribers.get(sessionId)) {
      if (sendToSocket(ws, message)) count++;
    }
  }

  for (const ws of globalSubscribers) {
    if (sendToSocket(ws, message)) count++;
  }

  return count;
}

function emitResultReady(sessionId, sceneResult) {
  return broadcast(sessionId, 'result_ready', sceneResult);
}

function emitAlertRaised(sessionId, alert) {
  return broadcast(sessionId, 'alert_raised', alert);
}

function getStats() {
  const sessions = {};
  for (const [id, set] of sessionSubscribers.entries()) {
    sessions[id] = set.size;
  }
  return {
    totalSessions: sessionSubscribers.size,
    globalSubscribers: globalSubscribers.size,
    sessions
  };
}

function clearAllSubscribers() {
  sessionSubscribers.clear();
  globalSubscribers.clear();
}

module.exports = {
  registerClient,
  unregisterClient,
  emitResultReady,
  emitAlertRaised,
  broadcast,
  getStats,
  clearAllSubscribers
};

