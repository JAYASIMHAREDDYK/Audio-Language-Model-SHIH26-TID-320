CREATE TABLE IF NOT EXISTS scene_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES audio_sessions(id) ON DELETE CASCADE,
  transcript_text TEXT,
  transcript_lang TEXT,
  sound_events JSONB,
  emotion JSONB,
  speakers JSONB,
  model_insight JSONB,
  reasoning_trace TEXT,
  fusion_summary TEXT,
  raw_ml_response JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_scene_results_session_id ON scene_results(session_id);
CREATE INDEX IF NOT EXISTS idx_scene_results_created_at ON scene_results(created_at DESC);

CREATE TABLE IF NOT EXISTS alerts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID REFERENCES audio_sessions(id) ON DELETE CASCADE,
  type TEXT NOT NULL,
  severity TEXT CHECK (severity IN ('low', 'medium', 'high')),
  message TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_alerts_session_id ON alerts(session_id);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at DESC);

