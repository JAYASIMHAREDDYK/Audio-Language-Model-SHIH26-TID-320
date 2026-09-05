import React, { useState, useEffect } from 'react';
import { login, saveAuth, getUser, clearAuth } from './api';
import RegisterForm from './RegisterForm';

export default function LoginPage({ onLoginSuccess }) {
  const [activeTab, setActiveTab] = useState('login'); // 'login' | 'register'
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentUser, setCurrentUser] = useState(null);

  useEffect(() => {
    const existing = getUser();
    if (existing) {
      setCurrentUser(existing);
    }
  }, []);

  const handleLoginSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    if (!email.trim() || !password) {
      setError('Please provide both email and password.');
      return;
    }

    try {
      setLoading(true);
      const res = await login({ email: email.trim(), password });

      const user = { user_id: res.user_id, name: res.name, email: email.trim() };
      saveAuth({ token: res.token, user });
      setCurrentUser(user);

      if (onLoginSuccess) {
        onLoginSuccess({ ...res, ...user });
      }
    } catch (err) {
      setError(err.message || 'Invalid credentials. Please verify your email and password.');
    } finally {
      setLoading(false);
    }
  };

  const handleQuickDemoFill = () => {
    setEmail('demo@smarthorizon.io');
    setPassword('demoPass123!');
    setError(null);
  };

  const handleLogout = () => {
    clearAuth();
    setCurrentUser(null);
  };

  return (
    <div className="alm-auth-wrapper">
      {/* Background Decorative Lighting */}
      <div className="ambient-glow glow-1"></div>
      <div className="ambient-glow glow-2"></div>

      <div className="alm-auth-card">
        {/* Brand Header */}
        <div className="alm-brand-header">
          <div className="alm-brand-badge">
            <span className="pulse-indicator"></span>
            <span>VOX · ALM-NHCE</span>
          </div>
          <h1 className="alm-title">Acoustic Scene Intelligence</h1>
          <p className="alm-subtitle">Real-time emergency perception, scene fusion & multi-modal audio monitoring</p>
        </div>

        {currentUser ? (
          <div className="alm-logged-in-box">
            <div className="user-avatar">
              {currentUser.name ? currentUser.name.charAt(0).toUpperCase() : 'U'}
            </div>
            <div className="user-details">
              <h3>Welcome back, {currentUser.name}!</h3>
              <p className="user-email">{currentUser.email}</p>
              <span className="session-tag">Active Session Token Active</span>
            </div>
            <div className="logged-in-actions">
              <button
                type="button"
                className="auth-submit-btn primary"
                onClick={() => onLoginSuccess && onLoginSuccess(currentUser)}
              >
                Enter Audio Dashboard →
              </button>
              <button
                type="button"
                className="auth-btn secondary"
                onClick={handleLogout}
              >
                Sign Out
              </button>
            </div>
          </div>
        ) : (
          <>
            {/* Mode Switcher Tabs */}
            <div className="auth-tabs" role="tablist">
              <button
                type="button"
                role="tab"
                aria-selected={activeTab === 'login'}
                className={`tab-btn ${activeTab === 'login' ? 'active' : ''}`}
                onClick={() => { setActiveTab('login'); setError(null); }}
              >
                Sign In
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={activeTab === 'register'}
                className={`tab-btn ${activeTab === 'register' ? 'active' : ''}`}
                onClick={() => { setActiveTab('register'); setError(null); }}
              >
                Create Account
              </button>
            </div>

            {activeTab === 'login' ? (
              <form onSubmit={handleLoginSubmit} className="auth-form" noValidate>
                {error && (
                  <div className="auth-alert error">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <line x1="12" y1="8" x2="12" y2="12" />
                      <line x1="12" y1="16" x2="12.01" y2="16" />
                    </svg>
                    <span>{error}</span>
                  </div>
                )}

                <div className="auth-field">
                  <label htmlFor="login-email">Email Address</label>
                  <div className="input-wrapper">
                    <svg className="field-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
                      <polyline points="22,6 12,13 2,6" />
                    </svg>
                    <input
                      id="login-email"
                      type="email"
                      placeholder="user@smarthorizon.io"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      disabled={loading}
                      required
                    />
                  </div>
                </div>

                <div className="auth-field">
                  <div className="field-header">
                    <label htmlFor="login-password">Password</label>
                    <button
                      type="button"
                      className="demo-fill-btn"
                      onClick={handleQuickDemoFill}
                    >
                      ⚡ Demo Fill
                    </button>
                  </div>
                  <div className="input-wrapper">
                    <svg className="field-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                    </svg>
                    <input
                      id="login-password"
                      type={showPassword ? 'text' : 'password'}
                      placeholder="••••••••"
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      disabled={loading}
                      required
                    />
                    <button
                      type="button"
                      className="toggle-password"
                      onClick={() => setShowPassword(!showPassword)}
                      tabIndex={-1}
                    >
                      {showPassword ? 'Hide' : 'Show'}
                    </button>
                  </div>
                </div>

                <button
                  type="submit"
                  className="auth-submit-btn"
                  disabled={loading}
                >
                  {loading ? (
                    <span className="btn-spinner-content">
                      <span className="spinner"></span> Signing In...
                    </span>
                  ) : (
                    'Authenticate Session →'
                  )}
                </button>

                <div className="auth-footer-switch">
                  <span>Need an account?</span>{' '}
                  <button
                    type="button"
                    className="link-btn"
                    onClick={() => setActiveTab('register')}
                  >
                    Register now
                  </button>
                </div>
              </form>
            ) : (
              <RegisterForm
                onSwitchToLogin={() => setActiveTab('login')}
                onSuccess={(user) => {
                  setCurrentUser(user);
                  if (onLoginSuccess) onLoginSuccess(user);
                }}
              />
            )}
          </>
        )}

        {/* Footer Meta */}
        <div className="alm-card-footer">
          <span>SH-DST-02 · Team ID: SHIH26-TID-320</span>
          <span>End-to-End Encrypted Session</span>
        </div>
      </div>

      {/* Scoped Styles for Rich Aesthetics */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

        .alm-auth-wrapper {
          min-height: 100vh;
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 1.5rem;
          background: #090d16;
          font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
          color: #f1f5f9;
          position: relative;
          overflow: hidden;
          box-sizing: border-box;
        }

        .alm-auth-wrapper * {
          box-sizing: border-box;
        }

        .ambient-glow {
          position: absolute;
          border-radius: 50%;
          filter: blur(120px);
          pointer-events: none;
          opacity: 0.25;
        }

        .glow-1 {
          width: 480px;
          height: 480px;
          background: #38bdf8;
          top: -100px;
          left: -80px;
        }

        .glow-2 {
          width: 520px;
          height: 520px;
          background: #6366f1;
          bottom: -120px;
          right: -90px;
        }

        .alm-auth-card {
          width: 100%;
          max-width: 460px;
          background: rgba(15, 23, 42, 0.78);
          border: 1px solid rgba(255, 255, 255, 0.1);
          backdrop-filter: blur(24px);
          -webkit-backdrop-filter: blur(24px);
          border-radius: 20px;
          padding: 2.25rem;
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.05);
          position: relative;
          z-index: 10;
        }

        .alm-brand-header {
          text-align: center;
          margin-bottom: 1.75rem;
        }

        .alm-brand-badge {
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.35rem 0.85rem;
          background: rgba(56, 189, 248, 0.1);
          border: 1px solid rgba(56, 189, 248, 0.25);
          border-radius: 9999px;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          color: #38bdf8;
          margin-bottom: 0.85rem;
          text-transform: uppercase;
        }

        .pulse-indicator {
          width: 7px;
          height: 7px;
          background: #38bdf8;
          border-radius: 50%;
          box-shadow: 0 0 10px #38bdf8;
          animation: pulse-glow 2s infinite ease-in-out;
        }

        @keyframes pulse-glow {
          0%, 100% { transform: scale(1); opacity: 1; }
          50% { transform: scale(1.35); opacity: 0.6; }
        }

        .alm-title {
          font-size: 1.45rem;
          font-weight: 700;
          letter-spacing: -0.02em;
          margin: 0 0 0.4rem 0;
          background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .alm-subtitle {
          font-size: 0.85rem;
          color: #94a3b8;
          margin: 0;
          line-height: 1.4;
        }

        .auth-tabs {
          display: flex;
          background: rgba(2, 6, 23, 0.6);
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 12px;
          padding: 0.3rem;
          margin-bottom: 1.5rem;
        }

        .tab-btn {
          flex: 1;
          padding: 0.6rem 0.8rem;
          border: none;
          background: transparent;
          color: #94a3b8;
          font-size: 0.88rem;
          font-weight: 600;
          border-radius: 8px;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .tab-btn:hover {
          color: #f1f5f9;
        }

        .tab-btn.active {
          background: rgba(56, 189, 248, 0.15);
          color: #38bdf8;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }

        .auth-form {
          display: flex;
          flex-direction: column;
          gap: 1.15rem;
        }

        .auth-alert {
          display: flex;
          align-items: center;
          gap: 0.65rem;
          padding: 0.75rem 1rem;
          border-radius: 10px;
          font-size: 0.85rem;
          line-height: 1.4;
        }

        .auth-alert.error {
          background: rgba(239, 68, 68, 0.12);
          border: 1px solid rgba(239, 68, 68, 0.3);
          color: #fca5a5;
        }

        .auth-alert.success {
          background: rgba(34, 197, 94, 0.12);
          border: 1px solid rgba(34, 197, 94, 0.3);
          color: #86efac;
        }

        .auth-field {
          display: flex;
          flex-direction: column;
          gap: 0.4rem;
        }

        .field-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .auth-field label {
          font-size: 0.82rem;
          font-weight: 600;
          color: #cbd5e1;
        }

        .demo-fill-btn {
          background: transparent;
          border: none;
          color: #38bdf8;
          font-size: 0.75rem;
          font-weight: 600;
          cursor: pointer;
          padding: 0;
          transition: opacity 0.2s;
        }

        .demo-fill-btn:hover {
          opacity: 0.8;
          text-decoration: underline;
        }

        .input-wrapper {
          position: relative;
          display: flex;
          align-items: center;
        }

        .field-icon {
          position: absolute;
          left: 1rem;
          color: #64748b;
          pointer-events: none;
          transition: color 0.2s;
        }

        .input-wrapper input {
          width: 100%;
          padding: 0.75rem 1rem 0.75rem 2.65rem;
          background: rgba(2, 6, 23, 0.7);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 10px;
          color: #f8fafc;
          font-size: 0.92rem;
          outline: none;
          transition: all 0.2s ease;
        }

        .input-wrapper input:focus {
          border-color: #38bdf8;
          box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2);
        }

        .input-wrapper input:focus + .field-icon,
        .input-wrapper:focus-within .field-icon {
          color: #38bdf8;
        }

        .toggle-password {
          position: absolute;
          right: 0.85rem;
          background: none;
          border: none;
          color: #64748b;
          font-size: 0.75rem;
          font-weight: 600;
          cursor: pointer;
          padding: 0.25rem 0.4rem;
        }

        .toggle-password:hover {
          color: #cbd5e1;
        }

        .auth-submit-btn {
          width: 100%;
          padding: 0.85rem;
          margin-top: 0.4rem;
          background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
          border: none;
          border-radius: 10px;
          color: #ffffff;
          font-size: 0.95rem;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
          box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
        }

        .auth-submit-btn:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 6px 18px rgba(2, 132, 199, 0.5);
        }

        .auth-submit-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .btn-spinner-content {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.5rem;
        }

        .spinner {
          width: 16px;
          height: 16px;
          border: 2px solid rgba(255, 255, 255, 0.3);
          border-top-color: #ffffff;
          border-radius: 50%;
          animation: spin 0.7s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        .auth-footer-switch {
          text-align: center;
          font-size: 0.84rem;
          color: #94a3b8;
          margin-top: 0.25rem;
        }

        .link-btn {
          background: none;
          border: none;
          color: #38bdf8;
          font-weight: 600;
          cursor: pointer;
          padding: 0;
          margin-left: 0.2rem;
        }

        .link-btn:hover {
          text-decoration: underline;
        }

        /* Logged in state box */
        .alm-logged-in-box {
          text-align: center;
          padding: 1.5rem 0.5rem;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 1rem;
        }

        .user-avatar {
          width: 64px;
          height: 64px;
          border-radius: 50%;
          background: linear-gradient(135deg, #0284c7, #6366f1);
          color: #ffffff;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 1.6rem;
          font-weight: 700;
          box-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
        }

        .user-details h3 {
          margin: 0;
          font-size: 1.2rem;
        }

        .user-email {
          color: #94a3b8;
          font-size: 0.9rem;
          margin: 0.25rem 0 0.5rem 0;
        }

        .session-tag {
          display: inline-block;
          font-size: 0.75rem;
          padding: 0.2rem 0.6rem;
          background: rgba(34, 197, 94, 0.15);
          color: #4ade80;
          border-radius: 9999px;
          font-family: 'JetBrains Mono', monospace;
        }

        .logged-in-actions {
          width: 100%;
          display: flex;
          flex-direction: column;
          gap: 0.65rem;
          margin-top: 0.5rem;
        }

        .auth-btn.secondary {
          background: rgba(255, 255, 255, 0.06);
          border: 1px solid rgba(255, 255, 255, 0.1);
          color: #cbd5e1;
          padding: 0.75rem;
          border-radius: 10px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
        }

        .auth-btn.secondary:hover {
          background: rgba(255, 255, 255, 0.12);
          color: #ffffff;
        }

        .alm-card-footer {
          margin-top: 1.75rem;
          padding-top: 1.1rem;
          border-top: 1px solid rgba(255, 255, 255, 0.08);
          display: flex;
          justify-content: space-between;
          font-size: 0.72rem;
          color: #64748b;
          font-family: 'JetBrains Mono', monospace;
        }
      `}</style>
    </div>
  );
}
