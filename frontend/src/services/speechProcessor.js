/**
 * speechProcessor.js
 * 
 * Safely bounds transcript strings. 
 * Deliberately excludes clinical functionality, emotion detection, and arbitrary confidence scores.
 */

const MAX_TRANSCRIPT_LENGTH = 2000;

export function processTranscript(rawTranscript) {
  if (!rawTranscript || typeof rawTranscript !== 'string') {
    return {
      transcript_available: false,
      transcript: "",
      transcript_length: 0,
      language: "en",
      confidence_available: false
    };
  }

  // Safe truncation
  let safeTranscript = rawTranscript.trim();
  if (safeTranscript.length > MAX_TRANSCRIPT_LENGTH) {
    safeTranscript = safeTranscript.substring(0, MAX_TRANSCRIPT_LENGTH) + "... [truncated]";
  }

  return {
    transcript_available: true,
    transcript: safeTranscript,
    transcript_length: safeTranscript.length,
    language: "en",
    confidence_available: false
  };
}
