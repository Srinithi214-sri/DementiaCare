# Phase 6: Multimodal Fusion Agent Core

*The Phase 6 Fusion Agent is deterministic and does not yet use Gemini or any external LLM.*

## 1. Fusion Agent Architecture

The Fusion Agent orchestrates context collection, intent evaluation, policy enforcement, and response generation in a safe, deterministic pipeline:

```
                ┌──────────────┐
                │   Event      │
                └──────┬───────┘
                       ↓
              ┌─────────────────┐
              │ Context Builder │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │  Fusion Agent   │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Decision Engine │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │  Policy Engine  │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Safe Response   │
              └─────────────────┘
```

## 2. Context Builder
The `ContextBuilder` (`backend/agent/context.py`) resolves bounded context required to evaluate the current event safely:
- Retrieves Patient Profile.
- Retrieves up to 20 Approved Memories. Excludes inactive/unapproved.
- Retrieves up to 20 Recent Events. Enforces 10KB payload limit.
- Returns a structured `FusionInput` schema.

## 3. Decision Engine
The `DecisionEngine` (`backend/agent/decision.py`) evaluates the `FusionInput` to determine the user's intent deterministicly. For example, a `speech_transcript` containing "hello" yields `intent = greeting`.

## 4. Policy Engine
The `PolicyEngine` (`backend/agent/policy.py`) acts as a hard boundary verifying the requested tuple: `intent`, `response_type`, `action`. Attempting arbitrary or unknown actions (e.g. `launch_drone`) will violently reject the pipeline, falling back to a safe clarification response.

## 5. LLM Abstraction
The `LLMProvider` abstraction (`backend/agent/llm.py`) prepares the system for Gemini. Currently, the `MockLLMProvider` leverages the deterministic DecisionEngine, ensuring the pipeline operates exactly as it will with Gemini, outputting structured `FusionDecision` schemas. Gemini is explicitly NOT being called.

## 6. Approved Memory Rules
If the decision engine recognizes a reminiscence request, it may ONLY select a memory provided by the `ContextBuilder` (which has already filtered out unapproved memories).

## 7. Patient Isolation
A hard security boundary in `FusionAgent.process_event` mandates that the requested Event (`event_id`) possesses a `patient_id` matching the targeted processing pipeline. Attempting to process Patient B's event through Patient A's context returns an immediate HTTP 400.

## 8. Action Restrictions
Available restricted actions: `none`, `speak`, `display_memory`, `request_clarification`, `notify_caregiver`.
LLMs cannot invent arbitrary new actions.

## 9. Confidence Semantics
Confidence is clamped between `0.0` and `1.0`. It is used for internal thresholding, not for making diagnostic/medical risk claims.

## 10. Safety Escalation Concept
An intent of `safety_escalation` maps strictly to `response_type = escalation` and `action = none` (for now). Caregiver alerts are conceptually mapped but not physically transmitted. No medical instructions are provided.

## 11. API Endpoint
`POST /api/v1/agent/process`
Accepts `{"patient_id": "...", "event_id": "..."}`. Returns the resolved `FusionDecision`.

## 12. Test Coverage
Rigorous Pytest coverage in `test_agent.py`, `test_agent_context.py`, and `test_agent_policy.py`. Validates memory safety, bounds constraints, explicit action denials, and patient isolation boundaries.

## 13. Future Gemini Integration
The framework is fully initialized. Swapping `MockLLMProvider` with a real `GeminiProvider` that outputs `FusionDecision` JSON is all that's required.

## 14. Future Camera/Audio Integration
Camera/Microphone models will simply emit normalized events to the V1 Event Ingestion layer, which are subsequently routed to the Fusion Agent. No structural changes required.
