import React from 'react';
import { Navbar } from './Navbar';
import { HeroSection } from './HeroSection';
import { InteractiveDemoSection } from './InteractiveDemoSection';
import { HowItWorksSection } from './HowItWorksSection';
import { DetectionArchitectureSection } from './DetectionArchitectureSection';
import { ClosingCtaSection } from './ClosingCtaSection';
import { Footer } from './Footer';

/**
 * Smart Horizon Main Landing Page
 * Restored to original landing page composition
 */
export function LandingPage({ onOpenConsole }) {
  return (
    <div className="min-h-screen bg-[#070709] text-white selection:bg-white selection:text-black overflow-x-hidden font-sans">
      {/* 1. Fixed Top Navigation */}
      <Navbar onOpenConsole={onOpenConsole} />

      <main>
        {/* 2. Hero Section with 3D Waveform */}
        <HeroSection onOpenConsole={onOpenConsole} />

        {/* 3. Interactive Live Demo Benchmarks */}
        <InteractiveDemoSection onOpenConsole={onOpenConsole} />

        {/* 4. 3-Step Forensic Pipeline */}
        <HowItWorksSection />

        {/* 5. Detection Architecture & Context Reasoning */}
        <DetectionArchitectureSection />

        {/* 6. High-Contrast Closing Call To Action */}
        <ClosingCtaSection onOpenConsole={onOpenConsole} />
      </main>

      {/* 7. Footer with Real-time Clock */}
      <Footer />
    </div>
  );
}
