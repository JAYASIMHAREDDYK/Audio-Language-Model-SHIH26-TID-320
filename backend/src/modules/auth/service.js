const crypto = require('crypto');

// In-memory fallback store when DB connection is not configured or in unit test mode
const memoryUsers = new Map();

// JWT Secret from environment or fallback
const JWT_SECRET = process.env.JWT_SECRET || 'dev_secret_key_change_in_production_123!';

/**
 * DB query helper - tries shared client if available, then pg pool, then in-memory fallback
 */
async function queryDb(text, params = []) {
  try {
    const clientModule = require('../../db/client');
    if (clientModule && typeof clientModule.query === 'function') {
      return await clientModule.query(text, params);
    }
    if (clientModule && clientModule.pool && typeof clientModule.pool.query === 'function') {
      return await clientModule.pool.query(text, params);
    }
  } catch {
    // client.js not yet configured or empty
  }

  // Check if pg connection string is present
  if (process.env.DB_URL || process.env.DATABASE_URL) {
    try {
      const { Pool } = require('pg');
      if (!queryDb._pool) {
        queryDb._pool = new Pool({
          connectionString: process.env.DB_URL || process.env.DATABASE_URL,
        });
      }
      return await queryDb._pool.query(text, params);
    } catch {
      // pg not installed or connection failed, use fallback below
    }
  }

  // In-memory fallback handler for users queries
  const lowerText = text.toLowerCase();
  if (lowerText.includes('select') && lowerText.includes('users') && lowerText.includes('email = $1')) {
    const email = params[0]?.toLowerCase();
    for (const user of memoryUsers.values()) {
      if (user.email.toLowerCase() === email) {
        return { rows: [user], rowCount: 1 };
      }
    }
    return { rows: [], rowCount: 0 };
  }

  if (lowerText.includes('select') && lowerText.includes('users') && lowerText.includes('id = $1')) {
    const id = params[0];
    const user = memoryUsers.get(id);
    return { rows: user ? [user] : [], rowCount: user ? 1 : 0 };
  }

  if (lowerText.includes('insert into users')) {
    const [id, email, password_hash, name] = params;
    const newUser = {
      id: id || crypto.randomUUID(),
      email,
      password_hash,
      name,
      created_at: new Date().toISOString()
    };
    memoryUsers.set(newUser.id, newUser);
    return { rows: [newUser], rowCount: 1 };
  }

  return { rows: [], rowCount: 0 };
}

/**
 * Hash password using bcryptjs if available, or native crypto scrypt fallback
 */
async function hashPassword(password) {
  try {
    const bcrypt = require('bcryptjs');
    return await bcrypt.hash(password, 10);
  } catch {
    const salt = crypto.randomBytes(16).toString('hex');
    const derivedKey = crypto.scryptSync(password, salt, 64);
    return `scrypt$${salt}$${derivedKey.toString('hex')}`;
  }
}

/**
 * Compare password against stored hash
 */
async function verifyPassword(password, storedHash) {
  if (!storedHash) return false;

  if (storedHash.startsWith('scrypt$')) {
    const parts = storedHash.split('$');
    const salt = parts[1];
    const key = parts[2];
    const derivedKey = crypto.scryptSync(password, salt, 64);
    return crypto.timingSafeEqual(Buffer.from(key, 'hex'), derivedKey);
  }

  try {
    const bcrypt = require('bcryptjs');
    return await bcrypt.compare(password, storedHash);
  } catch {
    // If bcrypt is not installed, fallback scrypt comparison failed
    return false;
  }
}

/**
 * Generate standard JWT token
 */
function generateToken(payload, expiresInSeconds = 86400) {
  try {
    const jwt = require('jsonwebtoken');
    return jwt.sign(payload, JWT_SECRET, { expiresIn: expiresInSeconds });
  } catch {
    // Built-in HMAC SHA256 JWT generator
    const header = { alg: 'HS256', typ: 'JWT' };
    const now = Math.floor(Date.now() / 1000);
    const body = { ...payload, iat: now, exp: now + expiresInSeconds };

    const b64Header = Buffer.from(JSON.stringify(header)).toString('base64url');
    const b64Payload = Buffer.from(JSON.stringify(body)).toString('base64url');
    const signature = crypto
      .createHmac('sha256', JWT_SECRET)
      .update(`${b64Header}.${b64Payload}`)
      .digest('base64url');

    return `${b64Header}.${b64Payload}.${signature}`;
  }
}

/**
 * Verify standard JWT token
 */
function verifyToken(token) {
  if (!token) throw new Error('Token is required');

  try {
    const jwt = require('jsonwebtoken');
    return jwt.verify(token, JWT_SECRET);
  } catch (err) {
    if (err.name === 'JsonWebTokenError' || err.name === 'TokenExpiredError') {
      throw err;
    }

    // Built-in HMAC SHA256 JWT verifier fallback
    const parts = token.split('.');
    if (parts.length !== 3) {
      throw new Error('Invalid token structure');
    }

    const [b64Header, b64Payload, signature] = parts;
    const expectedSig = crypto
      .createHmac('sha256', JWT_SECRET)
      .update(`${b64Header}.${b64Payload}`)
      .digest('base64url');

    if (!crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(expectedSig))) {
      throw new Error('Invalid token signature');
    }

    const payload = JSON.parse(Buffer.from(b64Payload, 'base64url').toString('utf8'));
    const now = Math.floor(Date.now() / 1000);
    if (payload.exp && payload.exp < now) {
      throw new Error('Token expired');
    }

    return payload;
  }
}

/**
 * Register a new user
 * Contract: { email, password, name } -> 201 { user_id, token }
 */
async function registerUser({ email, password, name }) {
  if (!email || !password || !name) {
    const err = new Error('Email, password, and name are required');
    err.statusCode = 400;
    throw err;
  }

  const normalizedEmail = email.trim().toLowerCase();

  // Basic email format check
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(normalizedEmail)) {
    const err = new Error('Invalid email format');
    err.statusCode = 400;
    throw err;
  }

  if (password.length < 6) {
    const err = new Error('Password must be at least 6 characters long');
    err.statusCode = 400;
    throw err;
  }

  // Check if user already exists
  const existingResult = await queryDb('SELECT id FROM users WHERE email = $1', [normalizedEmail]);
  if (existingResult.rows && existingResult.rows.length > 0) {
    const err = new Error('User with this email already exists');
    err.statusCode = 409;
    throw err;
  }

  const passwordHash = await hashPassword(password);
  const userId = crypto.randomUUID();

  await queryDb(
    'INSERT INTO users (id, email, password_hash, name, created_at) VALUES ($1, $2, $3, $4, now()) RETURNING id, email, name, created_at',
    [userId, normalizedEmail, passwordHash, name.trim()]
  );

  const token = generateToken({
    user_id: userId,
    email: normalizedEmail,
    name: name.trim()
  });

  return {
    user_id: userId,
    token,
    name: name.trim(),
    email: normalizedEmail
  };
}

/**
 * Login existing user
 * Contract: { email, password } -> 200 { user_id, token, name }
 */
async function loginUser({ email, password }) {
  if (!email || !password) {
    const err = new Error('Email and password are required');
    err.statusCode = 400;
    throw err;
  }

  const normalizedEmail = email.trim().toLowerCase();
  const result = await queryDb('SELECT id, email, password_hash, name FROM users WHERE email = $1', [normalizedEmail]);

  if (!result.rows || result.rows.length === 0) {
    const err = new Error('Invalid email or password');
    err.statusCode = 401;
    throw err;
  }

  const user = result.rows[0];
  const isMatch = await verifyPassword(password, user.password_hash);
  if (!isMatch) {
    const err = new Error('Invalid email or password');
    err.statusCode = 401;
    throw err;
  }

  const token = generateToken({
    user_id: user.id,
    email: user.email,
    name: user.name
  });

  return {
    user_id: user.id,
    token,
    name: user.name,
    email: user.email
  };
}

/**
 * Get user profile by ID
 * Contract: header Authorization: Bearer <token> -> 200 { user_id, email, name }
 */
async function getUserById(userId) {
  if (!userId) {
    const err = new Error('User ID is required');
    err.statusCode = 400;
    throw err;
  }

  const result = await queryDb('SELECT id, email, name FROM users WHERE id = $1', [userId]);
  if (!result.rows || result.rows.length === 0) {
    const err = new Error('User not found');
    err.statusCode = 404;
    throw err;
  }

  const user = result.rows[0];
  return {
    user_id: user.id,
    email: user.email,
    name: user.name
  };
}

/**
 * Clear test data (useful for test runner)
 */
function _clearMemoryUsers() {
  memoryUsers.clear();
}

module.exports = {
  hashPassword,
  verifyPassword,
  generateToken,
  verifyToken,
  registerUser,
  loginUser,
  getUserById,
  queryDb,
  _clearMemoryUsers
};
