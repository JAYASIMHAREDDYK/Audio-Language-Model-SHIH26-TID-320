import React, { useState } from 'react';
import { Shield, Sparkles, ArrowRight, Lock } from 'lucide-react';
import { BrandLogo } from './BrandLogo';

/**
 * Stage 1: Auth Screen (OAuth-Style Mock)
 * 
 * Stubbed handleAuth function clearly marked for drop-in OAuth2 / OIDC providers.
 */
export function AuthScreen({ onAuthSuccess }) {
  const [isLoading, setIsLoading] = useState(false);
  const [loadingProvider, setLoadingProvider] = useState(null);

  /**
   * OAuth Handler Stub
   * [INTEGRATION POINT]: Replace with real Google/GitHub OAuth2 or Supabase/Auth0 SDK calls.
   */
  const handleAuth = async (provider) => {
    setIsLoading(true);
    setLoadingProvider(provider);

    // Simulate network authentication handshake
    setTimeout(() => {
      const mockUser = {
        name: provider === 'google' ? 'Alex Rivera' : 'arivera-sec',
        email: provider === 'google' ? 'alex.rivera@horizon.sec' : 'alex@github.com',
        avatar: provider === 'google' ? 'AR' : 'GH',
        provider: provider,
        role: 'Forensic Intelligence Analyst'
      };

      setIsLoading(false);
      setLoadingProvider(null);
      onAuthSuccess(mockUser);
    }, 700);
  };

  return (
    <div className="min-h-screen bg-background text-[#E6E6E8] flex flex-col justify-center items-center px-4 relative overflow-hidden">
      {/* Subtle background ambient gradient */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-amber/5 rounded-full blur-3xl pointer-events-none" />

      <div className="w-full max-w-[420px] bg-panel border border-border-hairline rounded-lg p-8 shadow-2xl relative z-10 flex flex-col gap-6">
        {/* Brand Header */}
        <div className="flex flex-col items-center text-center gap-3">
          <BrandLogo size="w-14 h-14" showText={true} textClassName="text-xl" />
          <p className="text-xs text-[#9E9EA4] font-mono">
            Context-Aware Audio Language Model Console
          </p>
        </div>

        {/* Info Banner */}
        <div className="bg-panel-raised border border-border-hairline rounded-md p-3 text-xs text-[#B8B8BE] flex items-start gap-2.5">
          <Lock className="w-4 h-4 text-amber shrink-0 mt-0.5" />
          <span>
            Restricted intelligence console. Authenticate with verified analyst credentials to continue.
          </span>
        </div>

        {/* OAuth Action Buttons */}
        <div className="flex flex-col gap-3">
          {/* Google OAuth Mock */}
          <button
            type="button"
            onClick={() => handleAuth('google')}
            disabled={isLoading}
            className="w-full h-11 bg-panel-raised hover:bg-panel-hover active:bg-panel-active border border-border-hairline rounded-md px-4 flex items-center justify-between text-sm font-medium text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed group cursor-pointer"
          >
            <div className="flex items-center gap-3">
              <svg className="w-4 h-4" viewBox="0 0 24 24">
                <path fill="#EA4335" d="M12 5c1.6 0 3 .6 4.1 1.7l3.1-3.1C17.3 1.8 14.8 1 12 1 7.5 1 3.7 3.6 1.9 7.3l3.7 2.9C6.5 7.4 9 5 12 5z" />
                <path fill="#4285F4" d="M23.5 12.3c0-.8-.1-1.6-.2-2.3H12v4.6h6.5c-.3 1.5-1.1 2.8-2.4 3.7l3.7 2.9c2.2-2 3.7-5 3.7-8.9z" />
                <path fill="#FBBC05" d="M5.6 14.8c-.2-.7-.4-1.5-.4-2.3s.2-1.6.4-2.3L1.9 7.3C.7 9.7 0 12.3 0 15.1s.7 5.4 1.9 7.8l3.7-3.1z" />
                <path fill="#34A853" d="M12 23.5c3.2 0 6-1.1 8-3l-3.7-2.9c-1.1.7-2.5 1.2-4.3 1.2-3 0-5.5-2-6.4-4.8L1.9 17c1.8 3.7 5.6 6.5 10.1 6.5z" />
              </svg>
              <span>{loadingProvider === 'google' ? 'Authorizing identity...' : 'Continue with Google'}</span>
            </div>
            {loadingProvider === 'google' ? (
              <div className="w-4 h-4 border-2 border-amber border-t-transparent rounded-full animate-spin" />
            ) : (
              <ArrowRight className="w-4 h-4 text-[#6A6A72] group-hover:text-amber transition-colors" />
            )}
          </button>

          {/* GitHub OAuth Mock */}
          <button
            type="button"
            onClick={() => handleAuth('github')}
            disabled={isLoading}
            className="w-full h-11 bg-panel-raised hover:bg-panel-hover active:bg-panel-active border border-border-hairline rounded-md px-4 flex items-center justify-between text-sm font-medium text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed group cursor-pointer"
          >
            <div className="flex items-center gap-3">
              <svg className="w-4 h-4 fill-white" viewBox="0 0 24 24">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
              </svg>
              <span>{loadingProvider === 'github' ? 'Authenticating token...' : 'Continue with GitHub'}</span>
            </div>
            {loadingProvider === 'github' ? (
              <div className="w-4 h-4 border-2 border-amber border-t-transparent rounded-full animate-spin" />
            ) : (
              <ArrowRight className="w-4 h-4 text-[#6A6A72] group-hover:text-amber transition-colors" />
            )}
          </button>
        </div>

        {/* Footer Note */}
        <div className="pt-2 border-t border-border-hairline text-center text-[11px] text-[#6A6A72] font-mono">
          Security Protocol 2026.4 · Zero-Trust Evidence Gateway
        </div>
      </div>
    </div>
  );
}

