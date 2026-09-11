# Phase 12: Text-to-Speech (TTS) & Verbal Companion Output

This document outlines the architecture, constraints, and privacy guarantees for the Text-to-Speech (TTS) implementation in the DementiaCare Companion Screen.

## TTS Architecture

The Text-to-Speech integration is entirely localized to the frontend browser using the native Web Speech API (`window.speechSynthesis`).

The architecture works by intercepting the `FusionDecision` already returned by the backend `POST /api/v1/agent/process` endpoint. The existing `response_text` field is extracted and passed to the frontend TTS service.

No backend changes were required because the `FusionDecision` object is already authorized and correctly formatted by the Policy Engine. By doing this on the frontend, we avoid needing any TTS credentials on the backend and avoid transmitting or persisting audio data.

## Privacy Guarantees

TTS audio is generated locally by the browser and is never persisted, uploaded, or sent to Gemini.
There is strictly:
- No raw audio recording for synthesis output.
- No `MediaRecorder`, `Blob`, `ArrayBuffer`, or `AudioContext` usage in the synthesis pipeline.
- No audio file creation or serialization.
- No generated audio stored in `localStorage`, `sessionStorage`, or MongoDB.
- No Gemini audio processing or cloud TTS APIs.

## Supported Lifecycle and Voice Selection

The `textToSpeech.js` service handles the lifecycle for `SpeechSynthesisUtterance`:
- **Voice Selection:** It attempts to find a voice that exactly matches `patient.language`, falling back to prefix matching (e.g., matching `en-GB` when `en` is requested). If no match is found, the browser's default voice is used.
- **Constraints:** Max 2000 characters per utterance. Empty text is skipped.
- **Settings:** Conservative settings are applied (rate=0.9, pitch=1.0) to maintain a calm tone.

## Interaction Safeguards

- **Stale-Response Protection:** The CompanionScreen maintains a `currentRequestId` ref. If a response arrives for an old request (e.g., network delay, patient switch), the UI and TTS ignore it.
- **Patient Switch & Logout:** Switching the patient or logging out immediately invokes `textToSpeech.stop()`, canceling any active speech.
- **Speech-Recognition Interaction:** To prevent a microphone feedback loop (TTS → Microphone → Event → TTS), the active `SpeechRecognition` is explicitly stopped when `onStart` is triggered from the TTS utterance. Speech recognition remains opt-in and does not automatically restart after TTS completes.

## Browser Compatibility

The application degrades gracefully if `window.speechSynthesis` is unavailable. The TTS controls will either be hidden or safely ignored, and the core visual interaction remains fully functional. No crashes will occur if TTS is unsupported.
