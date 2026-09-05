/**
 * Canonical Core ALM Frontend API Client
 * Orchestrates communication between React Frontend and the Core ALM ML Service API.
 */

const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || 'http://localhost:8000';

/**
 * Check backend health status
 * @returns {Promise<{status: string, model_loaded: boolean, service: string}>}
 */
export async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
    if (!response.ok) {
      throw new Error(`Health check failed with status ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('[API Client] Health check error:', error);
    return { status: 'error', model_loaded: false, service: 'Core ALM Service' };
  }
}

/**
 * Send audio file/blob and natural language question to POST /analyze
 *
 * @param {Blob|File} audioSource - Uploaded audio File or recorded Blob
 * @param {string} question - User's natural language question
 * @param {string} [languageHint='auto'] - Optional language hint
 * @param {string} [sessionId='sess_default'] - Session identifier
 * @returns {Promise<Object>} Core ALM structured JSON response
 */
export async function analyzeAudio(audioSource, question = '', languageHint = 'auto', sessionId = 'sess_01', spokenTranscript = '', llmModel = 'gpt-4o-mini') {
  const formData = new FormData();

  if (audioSource) {
    let fileName = audioSource.name;
    if (!fileName) {
      const mimeType = audioSource.type || 'audio/webm';
      if (mimeType.includes('webm')) fileName = 'audio_capture.webm';
      else if (mimeType.includes('ogg')) fileName = 'audio_capture.ogg';
      else if (mimeType.includes('mp3')) fileName = 'audio_capture.mp3';
      else if (mimeType.includes('wav')) fileName = 'audio_capture.wav';
      else fileName = 'audio_capture.webm';
    }
    formData.append('audio_file', audioSource, fileName);
  }

  formData.append('question', question || '');
  formData.append('language_hint', languageHint || 'auto');
  formData.append('session_id', sessionId);
  formData.append('llm_model', llmModel);
  if (spokenTranscript) {
    formData.append('spoken_transcript', spokenTranscript);
  }

  console.log(`[API Client] Posting audio (${audioSource?.size || 0} bytes) to ${API_BASE_URL}/analyze with question: "${question}"`);

  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      let errorDetail = '';
      try {
        const errJson = await response.json();
        errorDetail = errJson.detail || errJson.message || '';
      } catch (e) {
        errorDetail = await response.text().catch(() => '');
      }

      if (response.status === 413) {
        throw new Error('The audio file is too large (max 50 MB allowed).');
      } else if (response.status === 400) {
        throw new Error(`Invalid request parameters: ${errorDetail}`);
      } else if (response.status === 503) {
        throw new Error('The Core ALM reasoning model service is initializing or unavailable.');
      } else {
        throw new Error(`Core ALM service returned error (${response.status}): ${errorDetail || 'Internal server error'}`);
      }
    }

    const data = await response.json();
    console.log('[API Client] Core ALM analysis response received:', data);
    return data;

  } catch (error) {
    console.error('[API Client] Analyze request failed:', error);
    throw error;
  }
}
