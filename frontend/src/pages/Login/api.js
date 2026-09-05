/**
 * Authentication API Client
 * Connects frontend Login/Register forms to backend auth endpoints
 */

const API_BASE_URL = (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_URL) 
  ? import.meta.env.VITE_API_URL 
  : 'http://localhost:4000/api/v1';

/**
 * Register a new user
 * Contract: POST /api/v1/auth/register -> { email, password, name } -> 201 { user_id, token }
 */
export async function register({ email, password, name }) {
  const response = await fetch(`${API_BASE_URL}/auth/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ email, password, name })
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `Registration failed with status ${response.status}`);
  }

  return data;
}

/**
 * Login an existing user
 * Contract: POST /api/v1/auth/login -> { email, password } -> 200 { user_id, token, name }
 */
export async function login({ email, password }) {
  const response = await fetch(`${API_BASE_URL}/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ email, password })
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `Login failed with status ${response.status}`);
  }

  return data;
}

/**
 * Get current authenticated user profile
 * Contract: GET /api/v1/auth/me -> header Authorization: Bearer <token> -> 200 { user_id, email, name }
 */
export async function getMe(token) {
  const authToken = token || getToken();
  if (!authToken) {
    throw new Error('No authentication token found');
  }

  const response = await fetch(`${API_BASE_URL}/auth/me`, {
    method: 'GET',
    headers: {
      'Authorization': `Bearer ${authToken}`
    }
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `Failed to fetch user profile`);
  }

  return data;
}

// Local storage management helpers
const TOKEN_KEY = 'alm_auth_token';
const USER_KEY = 'alm_auth_user';

export function saveAuth({ token, user }) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  if (user) localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || null;
}

export function getUser() {
  const raw = localStorage.getItem(USER_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export function clearAuth() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export default {
  register,
  login,
  getMe,
  saveAuth,
  getToken,
  getUser,
  clearAuth,
  API_BASE_URL
};
