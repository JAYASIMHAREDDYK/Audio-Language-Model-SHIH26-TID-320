const controller = require('./controller');

let router;

try {
  const express = require('express');
  router = express.Router();

  router.post('/register', controller.register);
  router.post('/login', controller.login);
  router.get('/me', controller.authenticateToken, controller.me);
} catch {
  // Lightweight router fallback when Express is not installed
  router = {
    post: () => {},
    get: () => {},
    use: () => {},
    routes: [
      { method: 'POST', path: '/register', handler: controller.register },
      { method: 'POST', path: '/login', handler: controller.login },
      { method: 'GET', path: '/me', handler: [controller.authenticateToken, controller.me] }
    ]
  };
}

module.exports = router;
module.exports.controller = controller;
