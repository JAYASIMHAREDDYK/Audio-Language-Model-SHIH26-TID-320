import React from 'react';
import { motion, useReducedMotion } from 'framer-motion';
import { ArrowRight, ChevronDown, Terminal, ArrowUpRight } from 'lucide-react';
import { Hero3DVisual } from './Hero3DVisual';

export function HeroSection({ onOpenConsole }) {
  const shouldReduceMotion = useReducedMotion();

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: shouldReduceMotion ? 0 : 0.12,
        delayChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: shouldReduceMotion ? 0 : 16 },
    visible: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: [0.16, 1, 0.3, 1] }
    }
  };

  return (
    <section
      id="hero"
      className="relative min-h-screen pt-24 pb-16 px-6 flex flex-col justify-between overflow-hidden bg-[#FAFAFA]"
    >
      {/* 1. Giant Oversized Background Parallax Wordmark */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none select-none z-0 overflow-hidden">
        <div className="font-['Space_Grotesk'] font-black text-[13vw] leading-none tracking-tighter text-[#0A0A0A]/[0.035] uppercase whitespace-nowrap transform -translate-y-6">
          VOX
        </div>
      </div>

      {/* 2. Top Header Meta Row */}
      <div className="max-w-7xl mx-auto w-full flex items-center justify-between z-10 pt-4">
        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 rounded-sm bg-[#0A0A0A] text-white text-[10px] font-mono font-bold tracking-widest uppercase shadow-xs">
            [01/06]
          </span>
          <span className="text-[11px] font-mono font-semibold tracking-wider text-[#6B6B6B] uppercase">
            Voice made easy
          </span>
        </div>

        {/* Upper-Right Stat Callout */}
        <div className="hidden lg:flex flex-col items-end max-w-xs text-right">
          <div className="flex items-baseline gap-1 font-['Space_Grotesk'] font-bold text-3xl text-[#0A0A0A] tracking-tight">
            <span>98.4%</span>
            <span className="text-xs font-mono font-medium text-[#6B6B6B]">ACCURACY</span>
          </div>
          <span className="text-[11px] font-mono font-bold text-[#0A0A0A] tracking-tight uppercase mt-0.5">
            False Positives Eliminated
          </span>
          <p className="text-[11px] text-[#6B6B6B] font-sans leading-tight mt-1">
            Keywords alone never trigger incidents without multimodal biometric and acoustic corroboration.
          </p>
        </div>
      </div>

      {/* 3. Main Center Hero Composition */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="max-w-7xl mx-auto w-full my-auto py-8 grid grid-cols-1 lg:grid-cols-12 gap-10 items-center z-10 relative"
      >
        {/* Left Column: Mixed-Weight Headline & Subtext */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          <motion.div variants={itemVariants} className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-xl bg-[#0A0A0A] p-2 flex items-center justify-center shrink-0 shadow-lg border border-black/20">
              <img src="/logo_amber.png" alt="Vox Logo" className="w-full h-full object-contain" />
            </div>
            <div className="flex flex-col">
              <span className="font-['Space_Grotesk'] font-bold text-lg text-[#0A0A0A] uppercase tracking-tight">
                Vox ALM Engine
              </span>
              <span className="text-xs font-mono text-[#6B6B6B]">
                Audio Language Model Platform
              </span>
            </div>
          </motion.div>

          <motion.h1
            variants={itemVariants}
            className="font-['Space_Grotesk'] text-4xl sm:text-6xl md:text-7xl font-bold tracking-[-0.035em] text-[#0A0A0A] leading-[0.95] flex flex-col uppercase"
          >
            <span>
              LISTENING <span className="font-light text-[#6B6B6B]">FOR</span>
            </span>
            <span>
              THE MOMENT <span className="font-light text-[#6B6B6B]">THAT</span>
            </span>
            <span className="font-black text-[#0A0A0A]">
              MATTERS.
            </span>
          </motion.h1>

          <motion.p
            variants={itemVariants}
            className="text-sm md:text-base font-sans text-[#2E2E2E] leading-relaxed max-w-lg"
          >
            Real-time distress detection, anti-fragmentation speaker diarization, and multimodal threat fusion from raw acoustic scenes — engineered to discount jokes and recognize true emergencies.
          </motion.p>

          {/* Action CTAs */}
          <motion.div variants={itemVariants} className="flex flex-wrap items-center gap-4 pt-2">
            {/* Primary Solid Black Pill */}
            <button
              type="button"
              onClick={onOpenConsole}
              className="h-12 px-7 rounded-full bg-[#0A0A0A] hover:bg-white hover:text-[#0A0A0A] active:scale-[0.98] border border-[#0A0A0A] text-white font-mono text-xs font-bold tracking-tight transition-all duration-150 flex items-center gap-2 cursor-pointer shadow-md hover:shadow-xl focus-visible:outline-[#0A0A0A]"
            >
              <span>Try live demo</span>
              <ArrowRight className="w-4 h-4" />
            </button>

            {/* Ghost Outline Button */}
            <a
              href="#how-it-works"
              className="h-12 px-6 rounded-full bg-white hover:bg-[#0A0A0A] hover:text-white active:scale-[0.98] border border-[#0A0A0A] text-[#0A0A0A] font-mono text-xs font-semibold tracking-tight transition-all duration-150 flex items-center gap-1.5 cursor-pointer focus-visible:outline-[#0A0A0A]"
            >
              <span>See how it works</span>
            </a>
          </motion.div>
        </div>

        {/* Right Column: Floating 3D Sculpture */}
        <motion.div variants={itemVariants} className="lg:col-span-5 flex justify-center items-center relative">
          <Hero3DVisual />
        </motion.div>
      </motion.div>

      {/* 4. Floating Utility Icons Stack (Right Edge) */}
      <div className="hidden md:flex fixed right-6 top-1/2 -translate-y-1/2 z-40 flex-col gap-2.5">
        <a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          title="Source Repository"
          className="w-9 h-9 rounded-full bg-white border border-[#E4E4E4] hover:border-[#0A0A0A] active:scale-95 text-[#0A0A0A] flex items-center justify-center shadow-xs hover:shadow-md transition-all duration-150 cursor-pointer focus-visible:outline-[#0A0A0A]"
        >
          <svg className="w-4 h-4 fill-current" viewBox="0 0 24 24">
            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
          </svg>
        </a>
        <button
          type="button"
          onClick={onOpenConsole}
          title="Launch Telemetry"
          className="w-9 h-9 rounded-full bg-white border border-[#E4E4E4] hover:border-[#0A0A0A] active:scale-95 text-[#0A0A0A] flex items-center justify-center shadow-xs hover:shadow-md transition-all duration-150 cursor-pointer focus-visible:outline-[#0A0A0A]"
        >
          <Terminal className="w-4 h-4" />
        </button>
      </div>

      {/* 5. Bottom Center Scroll Indicator */}
      <div className="max-w-7xl mx-auto w-full flex justify-center z-10 pt-2">
        <a
          href="#how-it-works"
          className="flex flex-col items-center gap-1 text-[10px] font-mono font-semibold tracking-widest text-[#6B6B6B] hover:text-[#0A0A0A] transition-colors uppercase cursor-pointer"
        >
          <span>Scroll to explore</span>
          <ChevronDown className="w-3.5 h-3.5 animate-bounce" />
        </a>
      </div>
    </section>
  );
}
