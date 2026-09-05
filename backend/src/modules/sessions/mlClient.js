/**
 * ML Service HTTP Client
 * Forwards audio to Sanket's /analyze endpoint
 */

const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://127.0.0.1:8000';
const ML_TIMEOUT_MS = parseInt(process.env.ML_TIMEOUT_MS || '120000', 10);

/**
 * Call POST /analyze on ML Service
 * Implements retry-once-then-fail on 500 or transient errors
 *
 * @param {Object} options
 * @param {string} options.sessionId
 * @param {Buffer|Blob|Stream} options.audioBuffer
 * @param {string} [options.filename='audio.wav']
 * @param {string} [options.mimeType='audio/wav']
 * @param {string} [options.languageHint]
 * @returns {Promise<Object>} ML response JSON
 */
async function analyzeAudio({ sessionId, audioBuffer, filename = 'audio.wav', mimeType = 'audio/wav', languageHint, question, spokenTranscript, llmModel = 'gpt-4o-mini' }) {
  if (!sessionId) {
    throw new Error('sessionId is required for ML analysis');
  }

  let attempt = 0;
  const maxAttempts = 2; // initial + 1 retry on 500

  while (attempt < maxAttempts) {
    attempt++;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), ML_TIMEOUT_MS);

    try {
      // Build multipart/form-data using standard Node FormData/Blob or FormData polyfill
      const formData = new FormData();
      const audioBlob = new Blob([audioBuffer || Buffer.alloc(0)], { type: mimeType });
      formData.append('audio_file', audioBlob, filename);
      formData.append('session_id', sessionId);
      formData.append('question', question || '');
      formData.append('llm_model', llmModel);
      if (languageHint) {
        formData.append('language_hint', languageHint);
      }
      if (spokenTranscript) {
        formData.append('spoken_transcript', spokenTranscript);
      }

      const targetUrl = `${ML_SERVICE_URL.replace(/\/+$/, '')}/analyze`;
      const response = await fetch(targetUrl, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (response.ok) {
        return await response.json();
      }

      // If status is 500 and we have attempts remaining, retry once
      if (response.status === 500 && attempt < maxAttempts) {
        await new Promise((res) => setTimeout(res, 500));
        continue;
      }

      const errorText = await response.text().catch(() => '');
      const err = new Error(`ML Service error (${response.status}): ${errorText}`);
      err.status = response.status;
      throw err;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error.name === 'AbortError') {
        const timeoutErr = new Error(`ML Service timed out after ${ML_TIMEOUT_MS}ms`);
        timeoutErr.code = 'ETIMEDOUT';
        throw timeoutErr;
      }

      // Retry once on network connection refusal or 500
      if (attempt < maxAttempts && (error.code === 'ECONNREFUSED' || error.status === 500)) {
        await new Promise((res) => setTimeout(res, 500));
        continue;
      }

      throw error;
    }
  }
}

module.exports = {
  analyzeAudio,
  ML_SERVICE_URL,
  ML_TIMEOUT_MS
};
