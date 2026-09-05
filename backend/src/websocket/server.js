const { registerClient, unregisterClient } = require('./broadcast');

let wssInstance = null;

function initWebSocketServer(httpServer) {
  let WebSocketServer;

  try {
    const wsModule = require('ws');
    WebSocketServer = wsModule.WebSocketServer || wsModule.Server;
  } catch (err) {
    return null;
  }

  if (!httpServer) {
    return null;
  }

  const wss = new WebSocketServer({ noServer: true });
  wssInstance = wss;

  httpServer.on('upgrade', (request, socket, head) => {
    const parsedUrl = new URL(request.url, `http://${request.headers.host || 'localhost'}`);
    const pathname = parsedUrl.pathname || '';

    if (pathname.startsWith('/ws')) {
      wss.handleUpgrade(request, socket, head, (ws) => {
        wss.emit('connection', ws, request);
      });
    }
  });

  wss.on('connection', (ws, req) => {
    const parsedUrl = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
    const pathname = parsedUrl.pathname || '';

    let sessionId = null;
    const sessionMatch = pathname.match(/^\/ws\/sessions\/([^/]+)/);
    if (sessionMatch) {
      sessionId = sessionMatch[1];
    } else if (parsedUrl.searchParams && parsedUrl.searchParams.get('session_id')) {
      sessionId = parsedUrl.searchParams.get('session_id');
    }

    registerClient(sessionId, ws);

    try {
      ws.send(
        JSON.stringify({
          event: 'connected',
          sessionId: sessionId || 'all',
          timestamp: new Date().toISOString()
        })
      );
    } catch (e) {
      // ignore
    }

    ws.on('message', (data) => {
      try {
        const msg = JSON.parse(data.toString());
        if (msg.action === 'subscribe' && msg.sessionId) {
          registerClient(msg.sessionId, ws);
          ws.send(
            JSON.stringify({
              event: 'subscribed',
              sessionId: msg.sessionId,
              timestamp: new Date().toISOString()
            })
          );
        } else if (msg.action === 'ping') {
          ws.send(JSON.stringify({ event: 'pong', timestamp: new Date().toISOString() }));
        }
      } catch (err) {
        // ignore malformed message
      }
    });

    ws.on('close', () => {
      unregisterClient(sessionId, ws);
    });

    ws.on('error', () => {
      unregisterClient(sessionId, ws);
    });
  });

  return wss;
}

function getWSS() {
  return wssInstance;
}

module.exports = {
  initWebSocketServer,
  getWSS
};

