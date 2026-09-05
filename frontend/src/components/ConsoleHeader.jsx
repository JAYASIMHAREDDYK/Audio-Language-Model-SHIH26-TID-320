import React, { useState } from 'react';
import { MoreHorizontal, Info } from 'lucide-react';
import { BrandLogo } from './BrandLogo';

/**
 * ChatGPT-style Slim Top Header
 * Consolidates session info into a slim bar with session title and a minimal "..." menu for metadata.
 */
export function ConsoleHeader({ activeSession }) {
  const [showMetadataMenu, setShowMetadataMenu] = useState(false);

  const formatDateTime = (isoString) => {
    try {
      return new Date(isoString).toLocaleString([], {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      });
    } catch (e) {
      return 'Active';
    }
  };

  return (
    <header className="h-12 px-5 bg-[#121212] border-b border-[#262626] flex items-center justify-between shrink-0 select-none relative">
      {/* Session Title & Logo */}
      <div className="flex items-center gap-3 min-w-0">
        <BrandLogo size="w-6 h-6" showText={false} />
        <h2 className="text-sm font-medium text-[#ececec] truncate">
          {activeSession?.label || 'New Forensic Session'}
        </h2>
      </div>

      {/* Right: Minimal "..." Menu */}
      <div className="relative">
        <button
          type="button"
          onClick={() => setShowMetadataMenu((prev) => !prev)}
          className="p-1.5 rounded-lg text-[#8e8ea0] hover:text-[#ececec] hover:bg-[#181818] transition-colors cursor-pointer"
          title="Session Metadata"
        >
          <MoreHorizontal className="w-4 h-4" />
        </button>

        {showMetadataMenu && (
          <div className="absolute right-0 mt-2 w-64 bg-[#181818] border border-[#262626] rounded-xl p-3 shadow-xl z-50 text-xs text-[#ececec] flex flex-col gap-2">
            <div className="flex items-center gap-2 pb-2 border-b border-[#262626] text-[#8e8ea0] font-medium">
              <Info className="w-3.5 h-3.5" />
              <span>Session Metadata</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#8e8ea0]">Session ID:</span>
              <span className="font-mono text-[11px] text-[#ececec]">{activeSession?.id || 'sess_active'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#8e8ea0]">Captured:</span>
              <span>{formatDateTime(activeSession?.createdAt)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#8e8ea0]">Telemetry:</span>
              <span className="text-emerald-400 font-medium">Connected</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#8e8ea0]">Engine Spec:</span>
              <span>Multimodal Fusion 2026.4</span>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}

