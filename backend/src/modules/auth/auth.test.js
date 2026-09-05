const { test, describe, beforeEach } = require('node:test');
const assert = require('node:assert');
const authService = require('./service');
const authController = require('./controller');

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

describe('Auth Module Tests', () => {
  beforeEach(() => {
    authService._clearMemoryUsers();
  });

  describe('Service Layer Unit Tests', () => {
    test('Password hashing and verification works', async () => {
      const password = 'securePassword123!';
      const hash = await authService.hashPassword(password);
      assert.ok(hash);
      assert.notStrictEqual(hash, password);

      const isValid = await authService.verifyPassword(password, hash);
      assert.strictEqual(isValid, true);

      const isInvalid = await authService.verifyPassword('wrongPassword', hash);
      assert.strictEqual(isInvalid, false);
    });

    test('JWT token generation and verification works', () => {
      const payload = { user_id: 'usr_123', email: 'test@example.com', name: 'Test User' };
      const token = authService.generateToken(payload);
      assert.ok(token);

      const decoded = authService.verifyToken(token);
      assert.strictEqual(decoded.user_id, payload.user_id);
      assert.strictEqual(decoded.email, payload.email);
      assert.strictEqual(decoded.name, payload.name);
    });

    test('registerUser creates user and issues valid token', async () => {
      const user = await authService.registerUser({
        email: 'anwin@smarthorizon.io',
        password: 'password123',
        name: 'Anwin Reji'
      });

      assert.ok(user.user_id);
      assert.ok(user.token);
      assert.strictEqual(user.name, 'Anwin Reji');
      assert.strictEqual(user.email, 'anwin@smarthorizon.io');

      const verified = authService.verifyToken(user.token);
      assert.strictEqual(verified.user_id, user.user_id);
    });

    test('registerUser rejects duplicate email', async () => {
      await authService.registerUser({
        email: 'duplicate@test.com',
        password: 'pass123456',
        name: 'First User'
      });

      await assert.rejects(
        async () => {
          await authService.registerUser({
            email: 'duplicate@test.com',
            password: 'pass123456',
            name: 'Second User'
          });
        },
        (err) => {
          assert.strictEqual(err.statusCode, 409);
          assert.match(err.message, /already exists/i);
          return true;
        }
      );
    });

    test('loginUser authenticates with valid credentials', async () => {
      await authService.registerUser({
        email: 'login@test.com',
        password: 'correctPassword123',
        name: 'Login User'
      });

      const loggedIn = await authService.loginUser({
        email: 'login@test.com',
        password: 'correctPassword123'
      });

      assert.ok(loggedIn.user_id);
      assert.ok(loggedIn.token);
      assert.strictEqual(loggedIn.name, 'Login User');
    });

    test('loginUser rejects wrong password', async () => {
      await authService.registerUser({
        email: 'wrongpass@test.com',
        password: 'correctPassword123',
        name: 'Test'
      });

      await assert.rejects(
        async () => {
          await authService.loginUser({
            email: 'wrongpass@test.com',
            password: 'incorrectPassword'
          });
        },
        (err) => {
          assert.strictEqual(err.statusCode, 401);
          return true;
        }
      );
    });
  });

  describe('Controller Contract Endpoint Tests', () => {
    test('POST /api/v1/auth/register responds 201 with { user_id, token }', async () => {
      const req = {
        body: {
          email: 'endpoint@test.com',
          password: 'pass123456',
          name: 'Endpoint User'
        }
      };
      const res = createMockRes();

      await authController.register(req, res);

      assert.strictEqual(res.statusCode, 201);
      assert.ok(res.data.user_id);
      assert.ok(res.data.token);
      // Password hash should not be leaked in response
      assert.strictEqual(res.data.password, undefined);
      assert.strictEqual(res.data.password_hash, undefined);
    });

    test('POST /api/v1/auth/login responds 200 with { user_id, token, name }', async () => {
      // First register
      await authService.registerUser({
        email: 'login_endpoint@test.com',
        password: 'pass123456',
        name: 'Login Endpoint'
      });

      const req = {
        body: {
          email: 'login_endpoint@test.com',
          password: 'pass123456'
        }
      };
      const res = createMockRes();

      await authController.login(req, res);

      assert.strictEqual(res.statusCode, 200);
      assert.ok(res.data.user_id);
      assert.ok(res.data.token);
      assert.strictEqual(res.data.name, 'Login Endpoint');
    });

    test('GET /api/v1/auth/me responds 200 with { user_id, email, name } when authenticated', async () => {
      const registered = await authService.registerUser({
        email: 'me_endpoint@test.com',
        password: 'pass123456',
        name: 'Me Endpoint User'
      });

      const req = {
        headers: {
          authorization: `Bearer ${registered.token}`
        }
      };
      const res = createMockRes();

      let nextCalled = false;
      authController.authenticateToken(req, res, () => {
        nextCalled = true;
      });
      assert.strictEqual(nextCalled, true);
      assert.ok(req.user);

      await authController.me(req, res);
      assert.strictEqual(res.statusCode, 200);
      assert.strictEqual(res.data.user_id, registered.user_id);
      assert.strictEqual(res.data.email, 'me_endpoint@test.com');
      assert.strictEqual(res.data.name, 'Me Endpoint User');
    });

    test('GET /api/v1/auth/me rejects unauthorized request with missing/invalid token', async () => {
      const req = { headers: {} };
      const res = createMockRes();

      authController.authenticateToken(req, res);
      assert.strictEqual(res.statusCode, 401);
      assert.match(res.data.error, /Unauthorized/);
    });
  });
});
