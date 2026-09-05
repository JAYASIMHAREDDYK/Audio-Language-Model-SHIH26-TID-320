import React from 'react';
import { Mic, Cpu, ShieldCheck, ArrowRight, Activity, Radio, Layers } from 'lucide-react';

export function HowItWorksSection() {
  const steps = [
    {
      index: '01',
      tag: 'INGESTION & PERCEPTION',
      title: 'Multichannel Acoustic Sensing',
      icon: <Mic className="w-5 h-5 text-[#0A0A0A]" />,
      desc: 'Ingests raw 48 kHz audio streams. Simultaneously extracts semantic transcription, acoustic non-speech transients (screams, gunshots), and vocal biometrics (pitch, arousal, jitter).',
      bullets: [
        'Multilingual IndicVoices ASR with low-coherence safety checks',
        'Independent PANNs acoustic event tagging',
        'Pitch & energy vocal stress biometrics'
      ]
    },
    {
      index: '02',
      tag: 'CONTEXTUAL REASONING',
      title: 'Multimodal Threat Fusion',
      icon: <Cpu className="w-5 h-5 text-[#0A0A0A]" />,
      desc: 'Separates literal words from true intent. If a speaker uses words like "help" or "knife" in a joke or quote, the engine discounts the text if vocal tone is calm and acoustics are nominal.',
      bullets: [
        'Joke & hypothetical intent classification',
        'De-escalation text discount on high arousal',
        'Non-silent fusion rationale output'
      ]
    },
    {
      index: '03',
      tag: 'DECISION & DIARIZATION',
      title: 'Anti-Fragmentation Verification',
      icon: <ShieldCheck className="w-5 h-5 text-[#0A0A0A]" />,
      desc: 'Executes a 6-rule diarization validator to prevent false speaker splits across breathing pauses. Emits high-certainty alerts only when multi-factor evidence is corroborated.',
      bullets: [
        'Grammatical continuity merge across pauses',
        'Dual-condition speaker split validation',
        'Zero-trust incident dispatch logging'
      ]
    }
  ];

  return (
    <section
      id="how-it-works"
      className="py-24 px-6 bg-white border-t border-b border-[#E4E4E4] relative overflow-hidden"
    >
      <div className="max-w-7xl mx-auto flex flex-col gap-16">
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-6 border-b border-[#E4E4E4]">
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2 text-xs font-mono font-bold tracking-widest text-[#6B6B6B] uppercase">
              <span>[02/06] ARCHITECTURE</span>
            </div>
            <h2 className="font-['Space_Grotesk'] text-3xl md:text-5xl font-bold tracking-tight text-[#0A0A0A] uppercase">
              How Vox Works
            </h2>
          </div>
          <p className="text-sm font-sans text-[#6B6B6B] max-w-md leading-relaxed">
            A three-stage forensic pipeline designed to eliminate false alarms and preserve context continuity on real distress signals.
          </p>
        </div>

        {/* 3 Steps Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {steps.map((st, i) => (
            <div
              key={st.index}
              className="bg-[#FAFAFA] border border-[#0A0A0A] rounded-xl p-8 flex flex-col justify-between gap-6 shadow-[0_4px_20px_rgba(0,0,0,0.03)] hover:shadow-[0_10px_30px_rgba(0,0,0,0.08)] transition-all transform hover:-translate-y-1 group"
            >
              <div className="flex flex-col gap-4">
                {/* Step Top Bar */}
                <div className="flex items-center justify-between pb-3 border-b border-[#E4E4E4]">
                  <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-[#0A0A0A] text-white">
                    {st.index}
                  </span>
                  <span className="text-[10px] font-mono font-bold tracking-wider text-[#6B6B6B] uppercase">
                    {st.tag}
                  </span>
                </div>

                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-white border border-[#E4E4E4] flex items-center justify-center group-hover:border-[#0A0A0A] transition-colors">
                    {st.icon}
                  </div>
                  <h3 className="font-['Space_Grotesk'] text-lg font-bold text-[#0A0A0A] tracking-tight">
                    {st.title}
                  </h3>
                </div>

                <p className="text-xs font-sans text-[#2E2E2E] leading-relaxed">
                  {st.desc}
                </p>
              </div>

              {/* Bullets */}
              <div className="pt-4 border-t border-[#E4E4E4] flex flex-col gap-2 text-[11px] font-mono text-[#6B6B6B]">
                {st.bullets.map((b, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <span className="w-1 h-1 rounded-full bg-[#0A0A0A]" />
                    <span className="text-[#2E2E2E]">{b}</span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
