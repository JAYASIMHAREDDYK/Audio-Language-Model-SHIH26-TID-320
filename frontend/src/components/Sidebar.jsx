import React from 'react';
import { Plus, Radio, Volume2, Clock, Inbox, LogOut, MessageSquare, Trash2 } from 'lucide-react';
import { BrandLogo } from './BrandLogo';

/**
 * ChatGPT-style Left Sidebar
 * Clean dark palette (#171717 background, #212121 hover), Inter font, non-ai-sloppy minimalism.
 */
export function Sidebar({
  sessions = [],
  activeSessionId,
  onSelectSession,
  onNewSession,
  onClearHistory,
  onDeleteSession,
  user,
  onSignOut,
  isCollapsed,
  onToggleCollapse
}) {
  const getBadgeStyle = (threatLevel = 'NOMINAL') => {
    switch (threatLevel.toUpperCase()) {
      case 'CRITICAL':
      case 'HIGH':
        return 'bg-[#3f2d2d] text-[#f87171] border border-red-500/20';
      case 'NOMINAL':
      default:
        return 'bg-[#2f2f2f] text-white border border-[#424242]';
    }
  };

  const formatTime = (isoString) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (e) {
      return 'Just now';
    }
  };

  return (
    <aside
      className={`h-screen bg-[#121212] border-r border-[#262626] flex flex-col shrink-0 transition-all duration-200 ease-out z-30 select-none ${
        isCollapsed ? 'w-[64px]' : 'w-[260px]'
      }`}
    >
      {/* Sidebar Header */}
      <div className="h-12 px-3.5 border-b border-[#262626] flex items-center justify-between shrink-0">
        {!isCollapsed ? (
          <BrandLogo size="w-7 h-7" showText={true} />
        ) : (
          <BrandLogo size="w-7 h-7" showText={false} />
        )}

        <button
          type="button"
          onClick={onToggleCollapse}
          title={isCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="p-1.5 rounded-lg text-[#8e8ea0] hover:text-[#ececec] hover:bg-[#181818] transition-all duration-150 cursor-pointer"
        >
          <Radio className="w-4 h-4" />
        </button>
      </div>

      {/* New Session Button */}
      <div className="p-3 border-b border-[#262626] shrink-0">
        <button
          type="button"
          onClick={onNewSession}
          className="w-full h-8 bg-transparent hover:bg-[#181818] border border-[#262626] rounded-lg px-3 flex items-center justify-start gap-2 text-xs font-medium text-[#ececec] transition-all duration-150 cursor-pointer"
        >
          <Plus className="w-3.5 h-3.5 text-[#ececec] shrink-0" />
          {!isCollapsed && <span>New Session</span>}
        </button>
      </div>

      {/* Scrollable Sessions List */}
      <div className="flex-1 overflow-y-auto px-2 py-2 flex flex-col gap-0.5">
        {!isCollapsed && (
          <div className="px-2.5 py-1.5 flex items-center justify-between text-[11px] font-medium text-[#8e8ea0]">
            <span>History</span>
            {sessions.length > 0 && onClearHistory && (
              <button
                type="button"
                onClick={onClearHistory}
                title="Clear all session history"
                className="px-1.5 py-0.5 rounded hover:bg-[#262626] text-[#8e8ea0] hover:text-red-400 transition-colors flex items-center gap-1 cursor-pointer text-[10px]"
              >
                <Trash2 className="w-3 h-3" />
                <span>Clear All</span>
              </button>
            )}
          </div>
        )}

        {sessions.length === 0 ? (
          <div className="p-6 text-center text-xs text-[#8e8ea0] flex flex-col items-center gap-2 my-auto">
            <Inbox className="w-5 h-5 text-[#676767]" />
            {!isCollapsed && <span>No previous sessions</span>}
          </div>
        ) : (
          sessions.map((sess) => {
            const isActive = sess.id === activeSessionId;
            const threatLevel = sess.result?.threat_level || sess.result?.fusion?.threat_level || 'NOMINAL';
            const isCritical = threatLevel === 'CRITICAL' || threatLevel === 'HIGH';

            return (
              <div
                key={sess.id}
                onClick={() => onSelectSession(sess.id)}
                className={`w-full rounded-lg px-2.5 py-2 text-left flex items-center justify-between gap-2 transition-colors cursor-pointer text-xs group ${
                  isActive
                    ? 'bg-[#181818] text-[#ececec]'
                    : 'bg-transparent hover:bg-[#181818]/70 text-[#8e8ea0] hover:text-[#ececec]'
                }`}
                title={sess.label}
              >
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <MessageSquare className={`w-3.5 h-3.5 shrink-0 ${isActive ? 'text-[#ececec]' : 'text-[#8e8ea0]'}`} />
                  {!isCollapsed && (
                    <span className="truncate text-xs font-normal">
                      {sess.label}
                    </span>
                  )}
                </div>

                {!isCollapsed && (
                  <div className="flex items-center gap-1.5 shrink-0">
                    <span className="text-[11px] text-[#8e8ea0] group-hover:hidden">
                      {formatTime(sess.createdAt)}
                    </span>
                    {/* Trash Delete Icon Button on Hover */}
                    {onDeleteSession && (
                      <button
                        type="button"
                        onClick={(e) => onDeleteSession(sess.id, e)}
                        title="Delete session"
                        className="hidden group-hover:flex p-1 rounded hover:bg-[#262626] text-[#8e8ea0] hover:text-red-400 transition-colors cursor-pointer"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    )}
                    {/* Subtle Dot Status Indicator */}
                    <span
                      className={`w-1.5 h-1.5 rounded-full ${
                        isCritical ? 'bg-red-400/80' : 'bg-emerald-400/60'
                      }`}
                      title={`Status: ${threatLevel}`}
                    />
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* User Footer */}
      <div className="p-3 border-t border-[#262626] shrink-0 bg-[#121212]">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-7 h-7 rounded-full bg-[#181818] text-white font-semibold text-xs flex items-center justify-center shrink-0 border border-[#262626]">
              {user?.avatar || 'AN'}
            </div>
            {!isCollapsed && (
              <div className="flex flex-col min-w-0">
                <span className="text-xs font-medium text-[#ececec] truncate">{user?.name || 'Analyst'}</span>
                <span className="text-[11px] text-[#8e8ea0] truncate">{user?.role || 'Forensic Lead'}</span>
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={onSignOut}
            title="Sign out of console"
            className="p-1.5 rounded-lg text-[#8e8ea0] hover:text-red-400 hover:bg-[#181818] transition-all duration-150 cursor-pointer"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </div>
    </aside>
  );
}

