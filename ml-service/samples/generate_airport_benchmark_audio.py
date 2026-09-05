import os
import tempfile
import numpy as np
import scipy.io.wavfile as wavfile
from gtts import gTTS
import subprocess

def create_airport_benchmark_audio():
    print("[Airport Generator] Synthesizing PS Airport Concourse Benchmark Audio...")
    sr = 16000
    duration = 10.0  # 10 seconds audio benchmark
    
    # 1. Synthesize Low-Frequency Jet Engine Roar & Highway Traffic Ambient Noise
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Pink noise simulation for jet turbine engine
    noise = np.random.normal(0, 1, int(sr * duration))
    # Low pass filter for jet engine rumble (around 150Hz - 300Hz)
    b = [0.05, 0.1, 0.05]
    a = [1.0, -0.8, 0.2]
    jet_rumble = np.convolve(noise, b, mode='same') * 0.15
    jet_whine = np.sin(2 * np.pi * 1800 * t) * 0.02 * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))
    
    # Highway vehicle traffic hum
    traffic_hum = np.sin(2 * np.pi * 85 * t) * 0.08 + np.sin(2 * np.pi * 170 * t) * 0.04
    
    # 2. Synthesize Airport Terminal Chime (Ding-Dong PA Chime)
    chime_sound = np.zeros(int(sr * duration))
    
    def add_chime(start_sec):
        idx = int(start_sec * sr)
        chime_t = np.linspace(0, 0.8, int(sr * 0.8), endpoint=False)
        env = np.exp(-chime_t * 4.0)
        c1 = np.sin(2 * np.pi * 523.25 * chime_t) * env  # C5
        c2 = np.sin(2 * np.pi * 659.25 * chime_t) * env  # E5
        c3 = np.sin(2 * np.pi * 783.99 * chime_t) * env  # G5
        tone = (c1 + c2 + c3) * 0.25
        end_idx = min(len(chime_sound), idx + len(tone))
        chime_sound[idx:end_idx] += tone[:end_idx - idx]

    add_chime(0.5)  # Initial PA Chime at 0.5s
    add_chime(6.5)  # Second PA Chime at 6.5s

    # 3. Generate Spoken Hindi Dialogue & English PA Announcement via gTTS
    tmp_hindi = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tmp_hindi.close()
    tts_hi = gTTS(text="अरे भैया जल्दी करो, फ्लाइट का बोर्डिंग शुरू हो गया है!", lang='hi')
    tts_hi.save(tmp_hindi.name)

    tmp_pa = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    tmp_pa.close()
    tts_pa = gTTS(text="Attention passengers, flight four zero two is now boarding at Gate seven.", lang='en')
    tts_pa.save(tmp_pa.name)

    # Convert TTS mp3 files to WAV using FFmpeg
    def mp3_to_wav(mp3_path):
        wav_out = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wav_out.close()
        cmd = ['ffmpeg', '-y', '-i', mp3_path, '-ar', str(sr), '-ac', '1', wav_out.name]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sample_rate, data = wavfile.read(wav_out.name)
        os.remove(wav_out.name)
        return data.astype(np.float32) / 32768.0

    hindi_wav = mp3_to_wav(tmp_hindi.name)
    pa_wav = mp3_to_wav(tmp_pa.name)

    os.remove(tmp_hindi.name)
    os.remove(tmp_pa.name)

    # 4. Mix All Multimodal Audio Components into Single Track
    speech_layer = np.zeros(int(sr * duration))
    
    # Place spoken Hindi dialogue at 1.5s
    h_idx = int(1.5 * sr)
    h_end = min(len(speech_layer), h_idx + len(hindi_wav))
    speech_layer[h_idx:h_end] += hindi_wav[:h_end - h_idx] * 0.85

    # Place Airport PA Announcement at 7.2s
    pa_idx = int(7.2 * sr)
    pa_end = min(len(speech_layer), pa_idx + len(pa_wav))
    speech_layer[pa_idx:pa_end] += pa_wav[:pa_end - pa_idx] * 0.65

    # Composite audio mix
    final_mix = jet_rumble + jet_whine + traffic_hum + chime_sound + speech_layer
    # Normalize to -1.0 to 1.0
    max_val = np.max(np.abs(final_mix))
    if max_val > 0:
        final_mix = final_mix / max_val * 0.90

    # Save to WAV
    int_data = (final_mix * 32767).astype(np.int16)
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    wav_path = os.path.join(out_dir, "airport_concourse_benchmark.wav")
    wavfile.write(wav_path, sr, int_data)
    print(f"[Airport Generator] Benchmark WAV saved to: {wav_path}")

    # Also copy to frontend public directory so web UI can play it natively
    frontend_pub = os.path.abspath(os.path.join(out_dir, "..", "..", "frontend", "public", "samples"))
    os.makedirs(frontend_pub, exist_ok=True)
    pub_wav = os.path.join(frontend_pub, "airport_concourse_benchmark.wav")
    wavfile.write(pub_wav, sr, int_data)
    print(f"[Airport Generator] Frontend Public WAV saved to: {pub_wav}")

    # Convert to MP3 as well for web compatibility
    mp3_path = os.path.join(out_dir, "airport_concourse_benchmark.mp3")
    cmd_mp3 = ['ffmpeg', '-y', '-i', wav_path, '-b:a', '192k', mp3_path]
    subprocess.run(cmd_mp3, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    pub_mp3 = os.path.join(frontend_pub, "airport_concourse_benchmark.mp3")
    cmd_pub_mp3 = ['ffmpeg', '-y', '-i', wav_path, '-b:a', '192k', pub_mp3]
    subprocess.run(cmd_pub_mp3, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    print("[Airport Generator] All airport benchmark audio files generated successfully!")

if __name__ == "__main__":
    create_airport_benchmark_audio()
