const authService = require('./service');

/**
 * Controller for POST /api/v1/auth/register
 * Contract: { email, password, name } -> 201 { user_id, token }
 */
async function register(req, res, next) {
  try {
    const { email, password, name } = req.body || {};
    const result = await authService.registerUser({ email, password, name });
    return res.status(201).json({
      user_id: result.user_id,
      token: result.token
    });
  } catch (error) {
    if (next) return next(error);
    const status = error.statusCode || 500;
    return res.status(status).json({ error: error.message });
  }
}

/**
 * Controller for POST /api/v1/auth/login
 * Contract: { email, password } -> 200 { user_id, token, name }
 */
async function login(req, res, next) {
  try {
    const { email, password } = req.body || {};
    const result = await authService.loginUser({ email, password });
    return res.status(200).json({
      user_id: result.user_id,
      token: result.token,
      name: result.name
    });
  } catch (error) {
    if (next) return next(error);
    const status = error.statusCode || 500;
    return res.status(status).json({ error: error.message });
  }
}

/**
 * Controller for GET /api/v1/auth/me
 * Contract: header Authorization: Bearer <token> -> 200 { user_id, email, name }
 */
async function me(req, res, next) {
  try {
    const userId = req.user?.user_id;
    if (!userId) {
      return res.status(401).json({ error: 'Unauthorized: missing user context' });
    }

    const user = await authService.getUserById(userId);
    return res.status(200).json({
      user_id: user.user_id,
      email: user.email,
      name: user.name
    });
  } catch (error) {
    if (next) return next(error);
    const status = error.statusCode || 500;
    return res.status(status).json({ error: error.message });
  }
}

/**
 * Middleware: Verify Bearer JWT token in Authorization header
 */
function authenticateToken(req, res, next) {
  const authHeader = req.headers && (req.headers.authorization || req.headers.Authorization);
  if (!authHeader) {
    return res.status(401).json({ error: 'Unauthorized: No token provided' });
  }

  const parts = authHeader.split(' ');
  if (parts.length !== 2 || parts[0].toLowerCase() !== 'bearer') {
    return res.status(401).json({ error: 'Unauthorized: Format must be Bearer <token>' });
  }

  const token = parts[1];
  try {
    const decoded = authService.verifyToken(token);
    req.user = decoded;
    if (next) next();
  } catch (err) {
    return res.status(401).json({ error: `Unauthorized: ${err.message}` });
  }
}

module.exports = {
  register,
  login,
  me,
  authenticateToken
};
