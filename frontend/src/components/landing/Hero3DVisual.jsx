import React from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { Activity, Radio, Disc, Mic, Volume2 } from 'lucide-react';

/**
 * 3D-Style Monochrome Hero Composition
 * Built from precision acoustic receiver, concentric wave rings, forensic reels, and telemetry tags.
 * Materials: Matte white, charcoal, brushed steel, deep black.
 */
export function Hero3DVisual() {
  const shouldReduceMotion = useReducedMotion();

  const floatAnimation = shouldReduceMotion
    ? {}
    : {
        y: [0, -14, 0],
        rotateX: [0, 2.5, 0],
        rotateY: [0, -3.5, 0],
        transition: {
          duration: 7,
          repeat: Infinity,
          ease: 'easeInOut'
        }
      };

  const ringAnimation1 = shouldReduceMotion
    ? {}
    : {
        scale: [1, 1.08, 1],
        opacity: [0.35, 0.65, 0.35],
        transition: {
          duration: 4.5,
          repeat: Infinity,
          ease: 'easeInOut'
        }
      };

  const ringAnimation2 = shouldReduceMotion
    ? {}
    : {
        scale: [1, 1.15, 1],
        opacity: [0.2, 0.45, 0.2],
        transition: {
          duration: 6,
          repeat: Infinity,
          ease: 'easeInOut',
          delay: 0.8
        }
      };

  const reelRotate = shouldReduceMotion
    ? {}
    : {
        rotate: [0, 360],
        transition: {
          duration: 18,
          repeat: Infinity,
          ease: 'linear'
        }
      };

  return (
    <div className="relative w-full max-w-[460px] h-[400px] md:h-[460px] flex items-center justify-center perspective-[1200px]">
      {/* Studio Shadow Base */}
      <div className="absolute bottom-6 w-3/4 h-12 bg-[#0A0A0A]/10 rounded-[100%] blur-2xl pointer-events-none transform -rotate-2" />

      {/* Floating Concentric Acoustic Shock Rings */}
      <motion.div
        animate={ringAnimation1}
        className="absolute w-[360px] h-[360px] md:w-[420px] md:h-[420px] rounded-full border border-dashed border-[#6B6B6B]/40 pointer-events-none"
      />
      <motion.div
        animate={ringAnimation2}
        className="absolute w-[440px] h-[440px] md:w-[500px] md:h-[500px] rounded-full border border-[#A8A8A8]/25 pointer-events-none"
      />

      {/* Main 3D Floating Forensic Sculpture */}
      <motion.div
        animate={floatAnimation}
        className="relative z-10 w-[300px] md:w-[340px] bg-white border border-[#0A0A0A] rounded-2xl p-6 shadow-[0_24px_60px_-15px_rgba(10,10,10,0.18)] flex flex-col gap-5 transform-gpu"
        style={{
          transformStyle: 'preserve-3d'
        }}
      >
        {/* Top Metallic Forensic Recorder Bar */}
        <div className="flex items-center justify-between pb-3 border-b border-[#E4E4E4]">
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-[#0A0A0A]" />
            <span className="text-[10px] font-mono font-bold tracking-wider text-[#0A0A0A] uppercase">
              REEL-TO-REEL · 48 KHZ LINEAR
            </span>
          </div>
          <span className="text-[9px] font-mono text-[#6B6B6B]">REC_CH-01</span>
        </div>

        {/* Dual Rotating Forensic Tape Reels */}
        <div className="flex items-center justify-around py-1 bg-[#F5F5F5] border border-[#E4E4E4] rounded-lg p-3">
          {/* Left Reel */}
          <motion.div
            animate={reelRotate}
            className="w-14 h-14 rounded-full bg-white border-2 border-[#0A0A0A] flex items-center justify-center relative shadow-sm"
          >
            <div className="w-4 h-4 rounded-full bg-[#0A0A0A]" />
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-full h-[1px] bg-[#6B6B6B]" />
              <div className="h-full w-[1px] bg-[#6B6B6B] absolute" />
            </div>
          </motion.div>

          {/* Center Soundwave Level Indicator */}
          <div className="flex flex-col items-center gap-1">
            <div className="flex items-end gap-1 h-8">
              {[40, 75, 100, 60, 85, 30, 90, 50].map((h, idx) => (
                <div
                  key={idx}
                  style={{ height: `${h}%` }}
                  className="w-1 bg-[#0A0A0A] rounded-t-sm"
                />
              ))}
            </div>
            <span className="text-[8px] font-mono font-semibold text-[#6B6B6B] tracking-widest">
              FFT BIOMETRIC
            </span>
          </div>

          {/* Right Reel */}
          <motion.div
            animate={reelRotate}
            className="w-14 h-14 rounded-full bg-white border-2 border-[#0A0A0A] flex items-center justify-center relative shadow-sm"
          >
            <div className="w-4 h-4 rounded-full bg-[#0A0A0A]" />
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="w-full h-[1px] bg-[#6B6B6B]" />
              <div className="h-full w-[1px] bg-[#6B6B6B] absolute" />
            </div>
          </motion.div>
        </div>

        {/* Precision Acoustic Sensor Grill */}
        <div className="bg-[#0A0A0A] text-white rounded-xl p-4 flex items-center justify-between shadow-inner">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-full bg-[#1F1F1F] border border-[#2E2E2E] flex items-center justify-center text-white shrink-0">
              <Mic className="w-4 h-4" />
            </div>
            <div className="flex flex-col">
              <span className="text-[11px] font-mono font-bold tracking-tight text-white uppercase">
                Multimodal Sensor
              </span>
              <span className="text-[9px] font-mono text-[#A8A8A8]">
                Arousal: LOW · Emotion: NEUTRAL
              </span>
            </div>
          </div>
          <div className="text-right">
            <span className="px-2 py-0.5 rounded bg-white text-[#0A0A0A] font-mono font-bold text-[9px] uppercase">
              NOMINAL
            </span>
          </div>
        </div>

        {/* Floating Telemetry Tag 1: Upper Left */}
        <div className="absolute -top-4 -left-6 bg-white border border-[#0A0A0A] rounded-md px-2.5 py-1 text-[9px] font-mono font-bold text-[#0A0A0A] shadow-md flex items-center gap-1.5 transform -rotate-3">
          <span className="w-1.5 h-1.5 rounded-full bg-[#0A0A0A]" />
          <span>CONFIDENCE: 98.4%</span>
        </div>

        {/* Floating Telemetry Tag 2: Lower Right */}
        <div className="absolute -bottom-4 -right-6 bg-[#0A0A0A] text-white rounded-md px-2.5 py-1 text-[9px] font-mono font-bold shadow-md flex items-center gap-1.5 transform rotate-3">
          <Radio className="w-2.5 h-2.5 text-white" />
          <span>ANTI-FRAGMENTATION DIARIZER</span>
        </div>
      </motion.div>
    </div>
  );
}
