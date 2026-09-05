import React, { useState, useEffect } from 'react';
import { Shield, ChevronDown, ArrowUpRight, Terminal, Globe } from 'lucide-react';

export function Navbar({ onOpenConsole }) {
  const [isLangOpen, setIsLangOpen] = useState(false);
  const [currentLocale, setCurrentLocale] = useState('GLOBAL [UTC] · EN');
  const [isVisible, setIsVisible] = useState(true);

  // Dynamic Scroll Listener: Hides Navbar when scrolling down, shows when scrolling up
  useEffect(() => {
    let lastScrollY = window.scrollY;

    const handleScroll = () => {
      const currentScrollY = window.scrollY;
      if (currentScrollY > lastScrollY && currentScrollY > 60) {
        setIsVisible(false); // Hide on scroll down
      } else {
        setIsVisible(true);  // Show on scroll up
      }
      lastScrollY = currentScrollY;
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header
      className={`w-full fixed top-0 left-0 z-50 bg-[#FAFAFA]/90 backdrop-blur-md border-b border-[#E4E4E4] transition-transform duration-300 ease-in-out ${
        isVisible ? 'translate-y-0' : '-translate-y-full'
      }`}
    >
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Left: Brand Wordmark with Official Soundwave Logo */}
        <a href="#hero" className="flex items-center gap-3 group">
          <div className="w-8 h-8 rounded-lg bg-[#0A0A0A] p-1 flex items-center justify-center shrink-0 border border-black/10 shadow-sm group-hover:scale-105 transition-transform">
            <img src="/logo_amber.png" alt="Vox Logo" className="w-full h-full object-contain" />
          </div>
          <div className="flex flex-col">
            <span className="font-['Space_Grotesk'] font-bold text-sm tracking-tight text-[#0A0A0A] uppercase group-hover:text-[#2E2E2E] transition-colors">
              Vox <span className="text-amber-600 font-mono text-xs">ALM</span>
            </span>
            <span className="text-[9px] font-mono font-semibold tracking-widest text-[#6B6B6B] uppercase">
              Audio Language Model
            </span>
          </div>
        </a>

        {/* Center: Horizontal Navigation Links */}
        <nav className="hidden md:flex items-center gap-8 text-xs font-mono font-medium text-[#2E2E2E]">
          <a href="#hero" className="nav-link hover:text-[#0A0A0A] py-1">
            Home
          </a>
          <a href="#how-it-works" className="nav-link hover:text-[#0A0A0A] py-1">
            How it works
          </a>
          <a href="#detection" className="nav-link hover:text-[#0A0A0A] py-1">
            Detection
          </a>
          <a href="#live-demo" className="nav-link hover:text-[#0A0A0A] py-1">
            Live Engine
          </a>
        </nav>

        {/* Right: Locale Dropdown + Launch Console Pill */}
        <div className="flex items-center gap-3">
          {/* Locale / Mode Pill */}
          <div className="relative hidden sm:block">
            <button
              type="button"
              onClick={() => setIsLangOpen(!isLangOpen)}
              className="h-8 px-2.5 rounded-full border border-[#E4E4E4] bg-white hover:border-[#0A0A0A] active:scale-[0.98] text-[10px] font-mono font-medium text-[#2E2E2E] flex items-center gap-1.5 transition-all duration-150 cursor-pointer shadow-2xs"
            >
              <Globe className="w-3 h-3 text-[#6B6B6B]" />
              <span>{currentLocale}</span>
              <ChevronDown className="w-2.5 h-2.5 text-[#6B6B6B]" />
            </button>

            {isLangOpen && (
              <div className="absolute right-0 mt-1.5 w-44 bg-white border border-[#E4E4E4] rounded-md shadow-xl py-1 text-[11px] font-mono z-50 animate-in fade-in slide-in-from-top-1 duration-150">
                <button
                  type="button"
                  onClick={() => {
                    setCurrentLocale('GLOBAL [UTC] · EN');
                    setIsLangOpen(false);
                  }}
                  className="w-full text-left px-3 py-2 hover:bg-[#F5F5F5] active:bg-[#E4E4E4] text-[#0A0A0A] transition-colors cursor-pointer"
                >
                  GLOBAL [UTC] · EN
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setCurrentLocale('EU-WEST [UTC+1] · EN');
                    setIsLangOpen(false);
                  }}
                  className="w-full text-left px-3 py-2 hover:bg-[#F5F5F5] active:bg-[#E4E4E4] text-[#0A0A0A] transition-colors cursor-pointer"
                >
                  EU-WEST [UTC+1] · EN
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setCurrentLocale('US-EAST [EST] · EN');
                    setIsLangOpen(false);
                  }}
                  className="w-full text-left px-3 py-2 hover:bg-[#F5F5F5] active:bg-[#E4E4E4] text-[#0A0A0A] transition-colors cursor-pointer"
                >
                  US-EAST [EST] · EN
                </button>
              </div>
            )}
          </div>

          {/* Action Button: Sign In / Launch Console */}
          <button
            type="button"
            onClick={onOpenConsole}
            className="h-9 px-4 rounded-full bg-[#0A0A0A] hover:bg-white hover:text-[#0A0A0A] hover:border hover:border-[#0A0A0A] active:scale-[0.98] text-white font-mono text-xs font-semibold tracking-tight transition-all duration-150 flex items-center gap-1.5 cursor-pointer shadow-sm hover:shadow-md focus-visible:outline-[#0A0A0A]"
          >
            <span>Launch Console</span>
            <ArrowUpRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </header>
  );
}
