# Phase 10: Real-Time Camera & Visual Event Processing

## Architecture
Phase 10 integrates real-time visual observations into the DementiaCare OS.

```text
Browser Camera
      ↓
Ephemeral Frame
      ↓
Local Visual Processor
      ↓
Structured Observation
      ↓
Normalized Event (POST /api/v1/events)
      ↓
Fusion Agent (POST /api/v1/agent/process)
      ↓
FusionDecision
```

## Privacy Model
Privacy is our primary non-negotiable invariant.

- **Raw frames remain in browser memory.** The `<video>` and offscreen `<canvas>` operate entirely in local memory.
- **Raw frames are never persisted.** No browser storage API (`localStorage`, `sessionStorage`, `IndexedDB`) is used for media.
- **Raw frames are never sent to the backend.** The payloads are strictly JSON metadata.
- **Raw frames are never sent to Gemini.** The LLM receives only the deterministic, structured output of the `ContextBuilder`.

## Camera Permissions & Lifecycle
- Caregivers explicitly trigger the camera using the "Start Camera" button in the Companion Screen.
- The browser prompts for `video` access. We **explicitly prohibit** `audio` access in this phase.
- An observation interval captures frame metadata every 3 seconds.
- The stream completely stops, freeing the hardware light, when:
  - "Stop Camera" is pressed.
  - The caregiver logs out.
  - The Companion Screen unmounts.
  - The active patient is switched.

## Security
Every camera event passes through the Phase 8 perimeter:

```text
JWT
 ↓
get_current_caregiver()
 ↓
authorize_patient_access()
 ↓
event validation
 ↓
Fusion Agent
```
A caregiver can never submit or process a camera event for an unauthorized patient.

## Limitations & Deferrals
Phase 10 is explicitly conservative. The visual processor proves the architecture but deliberately avoids medical inference.

This phase **DOES NOT** perform:
- Dementia detection or cognitive assessment
- Emotion recognition
- Medical inference or diagnosis
- Persistent image storage
- Gemini Vision analysis (all processing is local)
