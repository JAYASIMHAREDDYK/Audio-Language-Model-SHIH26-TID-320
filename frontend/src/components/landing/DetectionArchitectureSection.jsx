import React from 'react';
import { Shield, Sparkles, Check, X, FileText, Activity, Zap } from 'lucide-react';

export function DetectionArchitectureSection() {
  return (
    <section
      id="detection"
      className="py-24 px-6 bg-[#FAFAFA] border-b border-[#E4E4E4] relative overflow-hidden"
    >
      <div className="max-w-7xl mx-auto flex flex-col gap-16">
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-6 border-b border-[#E4E4E4]">
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-widest text-[#6B6B6B] uppercase">
              <span>[04/06] DETECTION SPECIFICATIONS</span>
            </div>
            <h2 className="font-['Space_Grotesk'] text-3xl md:text-5xl font-bold tracking-tight text-[#0A0A0A] uppercase">
              Context-Aware Architecture
            </h2>
          </div>
          <p className="text-sm font-sans text-[#6B6B6B] max-w-md leading-relaxed">
            Standard keyword systems create unbearable false alarm fatigue. Vox correlates intent, biometrics, and acoustic physics.
          </p>
        </div>

        {/* Comparison Architecture Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left: Traditional Naive Systems */}
          <div className="bg-white border border-[#E4E4E4] rounded-xl p-8 flex flex-col gap-6 shadow-sm">
            <div className="flex items-center justify-between pb-4 border-b border-[#E4E4E4]">
              <span className="text-xs font-mono font-bold text-[#6B6B6B] uppercase">
                Legacy Acoustic Systems
              </span>
              <span className="px-2 py-0.5 rounded bg-[#F5F5F5] text-[#6B6B6B] font-mono text-[10px] font-bold">
                HIGH FALSE POSITIVE RATE
              </span>
            </div>

            <div className="flex flex-col gap-4 text-xs font-sans text-[#6B6B6B]">
              <div className="flex items-start gap-3 p-3 bg-[#F5F5F5] rounded-lg border border-[#E4E4E4]">
                <X className="w-4 h-4 text-[#0A0A0A] shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-[#0A0A0A] block">Naive Keyword Matching:</span>
                  Triggers critical dispatch whenever "help", "knife", or "fire" appears, even in movies or jokes.
                </div>
              </div>

              <div className="flex items-start gap-3 p-3 bg-[#F5F5F5] rounded-lg border border-[#E4E4E4]">
                <X className="w-4 h-4 text-[#0A0A0A] shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-[#0A0A0A] block">Pause-Based Speaker Splitting:</span>
                  Splits a single speaker's sentence across normal breathing pauses, tearing keywords out of context.
                </div>
              </div>

              <div className="flex items-start gap-3 p-3 bg-[#F5F5F5] rounded-lg border border-[#E4E4E4]">
                <X className="w-4 h-4 text-[#0A0A0A] shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-[#0A0A0A] block">Single-Channel Reliance:</span>
                  Fails when speech is garbled, whispered, sarcastic, or muffled by background ambient noise.
                </div>
              </div>
            </div>
          </div>

          {/* Right: Vox Multimodal Standard */}
          <div className="bg-[#0A0A0A] text-white border border-[#0A0A0A] rounded-xl p-8 flex flex-col gap-6 shadow-xl">
            <div className="flex items-center justify-between pb-4 border-b border-[#2E2E2E]">
              <span className="text-xs font-mono font-bold text-white uppercase">
                Vox Standard
              </span>
              <span className="px-2 py-0.5 rounded bg-white text-[#0A0A0A] font-mono text-[10px] font-bold">
                MULTIMODAL FUSION
              </span>
            </div>

            <div className="flex flex-col gap-4 text-xs font-sans text-[#E4E4E4]">
              <div className="flex items-start gap-3 p-3 bg-[#141414] rounded-lg border border-[#2E2E2E]">
                <Check className="w-4 h-4 text-white shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-white block">Multi-Factor Intent Classification:</span>
                  Cross-references transcript text against vocal arousal biometrics. Sarcasm and jokes produce zero alarms.
                </div>
              </div>

              <div className="flex items-start gap-3 p-3 bg-[#141414] rounded-lg border border-[#2E2E2E]">
                <Check className="w-4 h-4 text-white shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-white block">6-Rule Anti-Fragmentation Diarizer:</span>
                  Merges grammatically continuous clauses across pauses and checks reassembly coherence.
                </div>
              </div>

              <div className="flex items-start gap-3 p-3 bg-[#141414] rounded-lg border border-[#2E2E2E]">
                <Check className="w-4 h-4 text-white shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-white block">Independent Acoustic Transients:</span>
                  Gunfire, screaming, and glass break transients corroborate true emergencies even on broken audio.
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
