import React, { useEffect, useRef } from 'react';

/**
 * Voice AI Fluid Orb Visualizer (Dribbble Shot 10164464 Inspired)
 * High-performance 60 FPS HTML5 Canvas rendering an organic, fluid, glowing voice sphere
 * that scales, morphs, and pulsates dynamically in response to real-time audio FFT telemetry.
 */
export function VoiceAIOrbVisualizer({ telemetry = [], isCapturing = true }) {
  const canvasRef = useRef(null);
  const animFrameRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let phase = 0;

    const render = () => {
      const width = canvas.width;
      const height = canvas.height;
      const centerX = width / 2;
      const centerY = height / 2;

      ctx.clearRect(0, 0, width, height);

      // Compute average volume & peak energy from FFT telemetry
      const validTelemetry = telemetry.length > 0 ? telemetry : Array.from({ length: 48 }).map(() => 8);
      const avgVol = validTelemetry.reduce((acc, val) => acc + val, 0) / validTelemetry.length;
      const normVol = Math.min(100, Math.max(5, avgVol));

      // Dynamic scale factor based on audio volume
      const volumeScale = 1 + (normVol / 100) * 0.8;
      const baseRadius = 55 * volumeScale;

      phase += 0.02 + (normVol / 100) * 0.03;

      // 1. Draw Outer Concentric Fluid Waves (4 Layers)
      const layers = [
        { points: 8, radiusOffset: 35, speedMult: 1.0, alpha: 0.15, strokeWidth: 1.5, dash: [] },
        { points: 10, radiusOffset: 25, speedMult: 1.3, alpha: 0.25, strokeWidth: 2, dash: [4, 4] },
        { points: 7, radiusOffset: 15, speedMult: 0.8, alpha: 0.4, strokeWidth: 2.5, dash: [] },
        { points: 12, radiusOffset: 5, speedMult: 1.5, alpha: 0.6, strokeWidth: 3, dash: [] }
      ];

      layers.forEach((layer, layerIdx) => {
        ctx.save();
        ctx.beginPath();

        const numPoints = layer.points;
        const currentRadius = baseRadius + layer.radiusOffset;

        ctx.setLineDash(layer.dash);
        ctx.lineWidth = layer.strokeWidth;
        ctx.strokeStyle = `rgba(255, 255, 255, ${layer.alpha})`;
        ctx.shadowColor = 'rgba(255, 255, 255, 0.4)';
        ctx.shadowBlur = 15;

        for (let i = 0; i <= numPoints; i++) {
          const angle = (i / numPoints) * Math.PI * 2;
          const fftVal = validTelemetry[i % validTelemetry.length] || 10;
          const distortion = Math.sin(phase * layer.speedMult + angle * 3) * (10 + (fftVal / 100) * 20);

          const r = currentRadius + distortion;
          const x = centerX + Math.cos(angle) * r;
          const y = centerY + Math.sin(angle) * r;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            // Smooth quadratic curves between points
            const prevAngle = ((i - 1) / numPoints) * Math.PI * 2;
            const prevFft = validTelemetry[(i - 1) % validTelemetry.length] || 10;
            const prevDistortion = Math.sin(phase * layer.speedMult + prevAngle * 3) * (10 + (prevFft / 100) * 20);
            const prevR = currentRadius + prevDistortion;
            const prevX = centerX + Math.cos(prevAngle) * prevR;
            const prevY = centerY + Math.sin(prevAngle) * prevR;

            const cpX = (prevX + x) / 2;
            const cpY = (prevY + y) / 2;
            ctx.quadraticCurveTo(prevX, prevY, cpX, cpY);
          }
        }

        ctx.closePath();
        ctx.stroke();
        ctx.restore();
      });

      // 2. Draw Central Glowing Core Sphere
      ctx.save();
      const coreGradient = ctx.createRadialGradient(
        centerX - baseRadius * 0.2,
        centerY - baseRadius * 0.2,
        baseRadius * 0.1,
        centerX,
        centerY,
        baseRadius
      );
      coreGradient.addColorStop(0, '#ffffff');
      coreGradient.addColorStop(0.4, '#e2e8f0');
      coreGradient.addColorStop(0.8, '#475569');
      coreGradient.addColorStop(1, 'rgba(15, 23, 42, 0.9)');

      ctx.beginPath();
      ctx.arc(centerX, centerY, baseRadius, 0, Math.PI * 2);
      ctx.fillStyle = coreGradient;
      ctx.shadowColor = 'rgba(255, 255, 255, 0.7)';
      ctx.shadowBlur = 30 + normVol * 0.3;
      ctx.fill();
      ctx.restore();

      // 3. Inner Pulsating Highlight Ring
      ctx.save();
      ctx.beginPath();
      ctx.arc(centerX, centerY, baseRadius * 0.65 + Math.sin(phase * 2) * 3, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.restore();

      animFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, [telemetry]);

  return (
    <div className="w-full flex flex-col items-center justify-center relative select-none">
      <canvas
        ref={canvasRef}
        width={340}
        height={280}
        className="w-[340px] h-[280px] drop-shadow-2xl"
      />
    </div>
  );
}
