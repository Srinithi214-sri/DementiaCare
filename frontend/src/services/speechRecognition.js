/**
 * Browser-local Speech Recognition service.
 * STRICT PRIVACY INVARIANT: This completely encapsulates microphone access 
 * without ever generating raw audio streams (MediaStream, MediaRecorder) in userland JS.
 */

export function createSpeechRecognizer(options = {}) {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  
  if (!SpeechRecognition) {
    return {
      supported: false,
      start: () => { throw new Error("Speech Recognition not supported in this browser"); },
      stop: () => {},
      abort: () => {}
    };
  }

  const recognition = new SpeechRecognition();
  
  // We prefer single-shot, non-continuous listening for explicit safety
  recognition.continuous = options.continuous || false;
  recognition.interimResults = options.interimResults || false;
  recognition.lang = options.lang || 'en-US';

  const wrapper = {
    supported: true,
    start: () => {
      try {
        recognition.start();
      } catch (err) {
        if (options.onError) options.onError(err);
      }
    },
    stop: () => recognition.stop(),
    abort: () => recognition.abort(),
    setOnResult: (cb) => {
      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        cb(transcript);
      };
    },
    setOnError: (cb) => { recognition.onerror = cb; },
    setOnStart: (cb) => { recognition.onstart = cb; },
    setOnEnd: (cb) => { recognition.onend = cb; }
  };

  return wrapper;
}
