import React from 'react';
import { ArrowRight, ShieldCheck, Terminal, ArrowUpRight } from 'lucide-react';

export function ClosingCtaSection({ onOpenConsole }) {
  return (
    <section className="py-24 px-6 bg-white border-b border-[#E4E4E4] relative overflow-hidden select-none">
      <div className="max-w-7xl mx-auto">
        <div className="bg-[#0A0A0A] text-white rounded-2xl p-10 md:p-16 flex flex-col lg:flex-row items-center justify-between gap-10 shadow-2xl relative overflow-hidden">
          {/* Ambient Wireframe Texture */}
          <div className="absolute inset-0 bg-[radial-gradient(#2E2E2E_1px,transparent_1px)] [background-size:16px_16px] opacity-35 pointer-events-none" />

          {/* Left Callout Text */}
          <div className="flex flex-col gap-4 relative z-10 max-w-xl">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-1 rounded-sm bg-white text-[#0A0A0A] text-[10px] font-mono font-bold tracking-widest uppercase shadow-xs">
                [05/06] DEPLOYMENT
              </span>
              <span className="text-[11px] font-mono font-semibold text-[#A8A8A8] uppercase">
                Zero-Trust Incident Gateway
              </span>
            </div>

            <h2 className="font-['Space_Grotesk'] text-3xl md:text-5xl font-bold tracking-tight text-white uppercase leading-tight">
              Ready to deploy forensic acoustic intelligence?
            </h2>

            <p className="text-xs md:text-sm font-sans text-[#A8A8A8] leading-relaxed">
              Integrate the real-time multimodal fusion synthesis pipeline directly into your monitoring infrastructure or explore our interactive forensic workstation.
            </p>
          </div>

          {/* Right Action Stack */}
          <div className="flex flex-col sm:flex-row lg:flex-col gap-3.5 relative z-10 shrink-0 w-full sm:w-auto">
            <button
              type="button"
              onClick={onOpenConsole}
              className="h-12 px-8 rounded-full bg-white hover:bg-[#E4E4E4] active:scale-[0.98] text-[#0A0A0A] font-mono text-xs font-bold tracking-tight flex items-center justify-center gap-2 transition-all duration-150 cursor-pointer shadow-lg transform hover:-translate-y-0.5 focus-visible:outline-white"
            >
              <span>Launch Live Console</span>
              <ArrowRight className="w-4 h-4" />
            </button>

            <a
              href="#detection"
              className="h-12 px-6 rounded-full bg-transparent hover:bg-white/10 active:scale-[0.98] border border-white text-white font-mono text-xs font-semibold tracking-tight flex items-center justify-center gap-1.5 transition-all duration-150 cursor-pointer focus-visible:outline-white"
            >
              <span>View Technical Specs</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
