import React, { useState, useEffect, useRef } from 'react';
import { ArrowRight, Activity, Terminal } from 'lucide-react';
import { BrandLogo } from './BrandLogo';

const ASCII_RAMP = " .:-=+*#%@";

/**
 * Stage 2: Landing Page
 * Features real-time sinusoidal ASCII acoustic waveform rendered inside a <pre> block.
 * Respects prefers-reduced-motion.
 */
export function LandingPage({ user, onProceed }) {
  const [asciiFrame, setAsciiFrame] = useState('');
  const animFrameRef = useRef(null);
  const phaseRef = useRef(0);

  useEffect(() => {
    // Check user accessibility preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const width = 64;
    const height = 14;

    const renderAscii = () => {
      phaseRef.current += prefersReducedMotion ? 0 : 0.045;
      const phase = phaseRef.current;
      const lines = [];

      for (let y = 0; y < height; y++) {
        let line = '';
        const normalizedY = (y - height / 2) / (height / 2);

        for (let x = 0; x < width; x++) {
          const normalizedX = (x / width) * Math.PI * 4;
          
          // Compound sinusoidal waves simulating acoustic resonance
          const wave1 = Math.sin(normalizedX * 1.2 + phase) * 0.45;
          const wave2 = Math.sin(normalizedX * 2.5 - phase * 1.5) * 0.25;
          const wave3 = Math.cos(normalizedX * 0.8 + phase * 0.7) * 0.15;
          const waveVal = wave1 + wave2 + wave3;

          const dist = Math.abs(normalizedY - waveVal);

          if (dist < 0.14) {
            line += ASCII_RAMP[9]; // Core wave center
          } else if (dist < 0.28) {
            line += ASCII_RAMP[7];
          } else if (dist < 0.45) {
            line += ASCII_RAMP[5];
          } else if (dist < 0.65) {
            line += ASCII_RAMP[3];
          } else if (dist < 0.85) {
            line += ASCII_RAMP[1];
          } else {
            line += ' ';
          }
        }
        lines.push(line);
      }

      setAsciiFrame(lines.join('\n'));

      if (!prefersReducedMotion) {
        animFrameRef.current = requestAnimationFrame(renderAscii);
      }
    };

    renderAscii();

    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, []);

  return (
    <div className="min-h-screen bg-background text-[#E6E6E8] flex flex-col justify-between items-center px-6 py-12 relative overflow-hidden select-none">
      {/* Top Header Tag */}
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-panel border border-border-hairline text-xs font-mono text-[#9E9EA4]">
        <BrandLogo size="w-4 h-4" showText={false} />
        <span>SYSTEM CALIBRATED · 48 KHZ ACOUSTIC SPECTRAL BUS</span>
      </div>

      {/* Center Hero with Logo and ASCII Waveform */}
      <div className="w-full max-w-3xl flex flex-col items-center text-center my-auto gap-8">
        <div className="flex flex-col items-center gap-3">
          <BrandLogo size="w-16 h-16" showText={true} textClassName="text-2xl" />
          <p className="text-sm font-mono text-[#A3A3AC] max-w-md">
            Context-Aware Forensic Acoustic Intelligence & Multimodal Decision Engine
          </p>
        </div>

        {/* ASCII Waveform Display */}
        <div className="w-full bg-panel border border-border-hairline rounded-lg p-5 shadow-2xl relative">
          <div className="flex items-center justify-between pb-3 mb-2 border-b border-border-hairline text-[11px] font-mono text-[#6A6A72]">
            <div className="flex items-center gap-2">
              <Terminal className="w-3.5 h-3.5 text-amber" />
              <span>Acoustic Waveform Telemetry</span>
            </div>
            <span>64-band FFT Synthesis</span>
          </div>

          <pre className="text-amber text-[10px] md:text-xs leading-[1.1] font-mono overflow-x-hidden font-bold tracking-widest text-center py-2">
            {asciiFrame}
          </pre>
        </div>

        {/* Welcome message referencing authenticated user */}
        <div className="flex flex-col items-center gap-1.5">
          <div className="text-xs font-mono text-[#9E9EA4]">
            Authenticated Analyst:{' '}
            <span className="text-white font-semibold">{user?.name || 'Authorized Analyst'}</span>
            {user?.email && <span className="text-[#6A6A72]"> ({user.email})</span>}
          </div>
          <p className="text-xs text-[#6A6A72]">
            Evidence fusion synthesized. All forensic pipelines primed and standing by.
          </p>
        </div>

        {/* Single Proceed Action Button */}
        <button
          type="button"
          onClick={onProceed}
          className="h-11 px-6 bg-amber hover:bg-amber-hover active:bg-[#C98F46] text-[#0A0A0C] font-semibold text-sm rounded-md flex items-center gap-2.5 transition-colors shadow-lg cursor-pointer"
        >
          <span>Proceed to console</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Footer System Specs */}
      <div className="text-[11px] font-mono text-[#6A6A72]">
        Version 2026.4 · Multi-Factor False Positive Reduction Standard
      </div>
    </div>
  );
}
