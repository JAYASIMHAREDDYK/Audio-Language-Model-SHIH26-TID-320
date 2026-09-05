import React from 'react';

/**
 * Official Vox ALM Soundwave Microphone Brand Logo Component
 */
export function BrandLogo({ size = 'w-8 h-8', className = '', showText = false, textClassName = '' }) {
  return (
    <div className={`flex items-center gap-2.5 shrink-0 select-none ${className}`}>
      <div className={`${size} rounded-xl bg-gradient-to-b from-[#262626] to-[#141414] border border-[#3f3f46] p-1.5 flex items-center justify-center shrink-0 shadow-lg group hover:border-amber-500/50 transition-all`}>
        <img
          src="/logo.png"
          alt="Vox ALM Soundwave Logo"
          className="w-full h-full object-contain group-hover:scale-110 transition-transform filter drop-shadow-[0_0_8px_rgba(245,158,11,0.3)]"
        />
      </div>
      {showText && (
        <div className="flex flex-col min-w-0">
          <div className="flex items-center gap-1">
            <span className={`text-sm font-bold tracking-tight text-white truncate ${textClassName}`}>
              Vox
            </span>
            <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30">
              ALM
            </span>
          </div>
          <span className="text-[10px] text-[#8e8ea0] tracking-wide truncate">
            Audio Language Model
          </span>
        </div>
      )}
    </div>
  );
}

export default BrandLogo;
