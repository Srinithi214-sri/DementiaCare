# Phase 11: Real-Time Microphone & Speech-to-Text Foundation

## Architecture
Phase 11 introduces voice interaction via browser-native Speech Recognition, adhering strictly to the privacy invariants of DementiaCare OS.

```text
Browser Microphone
      ↓
Web Speech API (SpeechRecognition)
      ↓
Transient Transcript Text
      ↓
Local Speech Processor (Safe Truncation)
      ↓
Normalized Event (POST /api/v1/events)
      ↓
Fusion Agent (POST /api/v1/agent/process)
      ↓
FusionDecision
```

## Privacy Guarantees
- **No Raw Audio Processing:** `getUserMedia` with `audio: true` is completely avoided. The application relies entirely on the browser's native `SpeechRecognition` API, which encapsulates microphone access and returns only transcribed text strings.
- **No Audio Persistence:** No `MediaRecorder`, `Blob`, `AudioContext`, or encoded audio data structures are ever instantiated.
- **No Audio Transmission:** No audio data is ever sent to the backend, to Gemini, or to MongoDB.
- **Strict Bounding:** Transcripts are forcefully truncated to a maximum of 2000 characters to prevent overflow injection attacks into the Gemini prompt context.

## Microphone Lifecycle
- Caregivers must explicitly press the "Start Speech" button to initiate listening.
- A single-shot (`continuous: false`) recognition listener is activated.
- When speech ends and a transcript is finalized, the microphone immediately disengages.
- The UI automatically cleans up Speech Recognition if the caregiver switches patients, logs out, unmounts the component, or explicitly presses "Stop Speech".

## Security & Authorization
Every speech event flows through the Phase 8 security perimeter:
1. `api.js` submits the transcript alongside the caregiver's JWT.
2. The backend validates the JWT and verifies the caregiver's authorization for the targeted `patient_id`.
3. The Fusion Agent retrieves only `approved=True` memories before appending the safe transcript to the LLM context block.

## Unsupported Browsers
Because the `SpeechRecognition` API is not universally supported (e.g., Firefox), the frontend gracefully detects the lack of support. If unsupported, the UI sets a clean error state (`"Speech Recognition not supported in this browser"`) and disables the feature, ensuring the core Companion Screen functionality never crashes.

## Excluded Functionality
To preserve clinical isolation and deterministic safety, Phase 11 intentionally omits:
- Medical classification or dementia scoring based on speech patterns.
- Emotion recognition or sentiment analysis based on vocal tone.
- Audio file persistence or playback features.
- Gemini Voice/Audio multimodal processing (all inference relies strictly on the transient text transcript).
