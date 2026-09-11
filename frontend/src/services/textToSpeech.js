/**
 * Browser-local Text-to-Speech service.
 * STRICT PRIVACY INVARIANT: This completely encapsulates speech synthesis
 * and NEVER persists, logs, or transmits audio or text data.
 */

class TextToSpeechService {
  constructor() {
    this.supported = 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;
    this.currentUtterance = null;
    this.voicesLoaded = false;
    
    // Attempt to load voices, they might be loaded asynchronously in some browsers
    if (this.supported) {
      this._loadVoices();
      if (window.speechSynthesis.onvoiceschanged !== undefined) {
        window.speechSynthesis.onvoiceschanged = () => this._loadVoices();
      }
    }
  }

  _loadVoices() {
    if (this.supported) {
      const voices = window.speechSynthesis.getVoices();
      if (voices.length > 0) {
        this.voicesLoaded = true;
      }
    }
  }

  isSupported() {
    return this.supported;
  }

  /**
   * Speak the provided text.
   * @param {string} text - The text to speak.
   * @param {object} options - Optional configuration (language, rate, pitch, volume).
   * @param {function} onStart - Callback when speech starts.
   * @param {function} onEnd - Callback when speech ends.
   * @param {function} onError - Callback when an error occurs.
   */
  speak(text, options = {}, onStart = null, onEnd = null, onError = null) {
    if (!this.supported) {
      if (onError) onError(new Error("Text-to-speech not supported"));
      return;
    }

    if (typeof text !== 'string') {
      if (onError) onError(new Error("Text must be a string"));
      return;
    }

    const trimmedText = text.trim();
    if (!trimmedText) {
      if (onEnd) onEnd(); // Treat empty as immediately finished
      return;
    }

    // Conservative maximum length
    const MAX_LENGTH = 2000;
    const finalString = trimmedText.slice(0, MAX_LENGTH);

    // Cancel any ongoing speech
    this.stop();

    const utterance = new window.SpeechSynthesisUtterance(finalString);
    
    // Set conservative defaults
    utterance.rate = 0.9;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    // Apply safe overrides if provided
    if (options.rate && typeof options.rate === 'number' && options.rate >= 0.1 && options.rate <= 2.0) {
      utterance.rate = options.rate;
    }
    if (options.pitch && typeof options.pitch === 'number' && options.pitch >= 0.1 && options.pitch <= 2.0) {
      utterance.pitch = options.pitch;
    }
    if (options.volume && typeof options.volume === 'number' && options.volume >= 0.0 && options.volume <= 1.0) {
      utterance.volume = options.volume;
    }

    // Handle language / voice selection
    const targetLang = options.language || 'en-US';
    
    if (this.voicesLoaded) {
      const voices = window.speechSynthesis.getVoices();
      let selectedVoice = null;

      // 1. Exact match
      selectedVoice = voices.find(v => v.lang === targetLang);
      
      // 2. Prefix match (e.g., target 'en' matches 'en-US' or 'en-GB')
      if (!selectedVoice) {
        const prefix = targetLang.split('-')[0].toLowerCase();
        selectedVoice = voices.find(v => v.lang.toLowerCase().startsWith(prefix));
      }

      if (selectedVoice) {
        utterance.voice = selectedVoice;
      }
    }

    // Lifecycle callbacks
    utterance.onstart = () => {
      if (onStart) onStart();
    };

    utterance.onend = () => {
      this.currentUtterance = null;
      if (onEnd) onEnd();
    };

    utterance.onerror = (event) => {
      this.currentUtterance = null;
      if (onError) onError(event);
    };

    this.currentUtterance = utterance;

    try {
      window.speechSynthesis.speak(utterance);
    } catch (e) {
      this.currentUtterance = null;
      if (onError) onError(e);
    }
  }

  stop() {
    if (!this.supported) return;
    try {
      window.speechSynthesis.cancel();
      this.currentUtterance = null;
    } catch (e) {
      // Safely ignore cancel errors
    }
  }
}

export const textToSpeech = new TextToSpeechService();
