const controller = require('./controller');

let uploadMiddleware;

try {
  const multer = require('multer');
  const storage = multer.memoryStorage();
  const upload = multer({
    storage,
    limits: { fileSize: 50 * 1024 * 1024 } // 50MB max
  });
  uploadMiddleware = upload.single('audio_file');
} catch {
  // Fallback middleware if multer is not installed
  uploadMiddleware = (req, res, next) => {
    // If request already has req.file or req.body, continue
    if (next) next();
  };
}

let router;

try {
  const express = require('express');
  router = express.Router();

  router.post('/', controller.createSession);
  router.post('/:id/audio', uploadMiddleware, controller.uploadAudio);
  router.get('/:id', controller.getSession);
} catch {
  // Lightweight router fallback when Express is not installed
  router = {
    post: () => {},
    get: () => {},
    use: () => {},
    routes: [
      { method: 'POST', path: '/', handler: controller.createSession },
      { method: 'POST', path: '/:id/audio', handler: [uploadMiddleware, controller.uploadAudio] },
      { method: 'GET', path: '/:id', handler: controller.getSession }
    ]
  };
}

module.exports = router;
module.exports.controller = controller;
module.exports.uploadMiddleware = uploadMiddleware;
