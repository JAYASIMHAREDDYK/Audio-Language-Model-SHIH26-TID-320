try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np
    HAS_RESEMBLYZER = True
except ImportError:
    HAS_RESEMBLYZER = False

def diarize_simple(wav_path, window_s=1.5, hop_s=0.75, n_speakers=None):
    if not HAS_RESEMBLYZER:
        return [{"time": 0.0, "speaker": "SPEAKER_0"}]
        
    encoder = VoiceEncoder()
    wav = preprocess_wav(wav_path)
    sr = 16000
    win, hop = int(window_s * sr), int(hop_s * sr)
    embeddings, timestamps = [], []
    for start in range(0, len(wav) - win, hop):
        chunk = wav[start:start + win]
        emb = encoder.embed_utterance(chunk)
        embeddings.append(emb)
        timestamps.append(start / sr)
    embeddings = np.array(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=n_speakers,
        distance_threshold=None if n_speakers else 0.9,
        metric="cosine", linkage="average"
    )
    labels = clustering.fit_predict(embeddings)
    return [{"time": round(t, 2), "speaker": f"SPEAKER_{spk}"} for t, spk in zip(timestamps, labels)]
