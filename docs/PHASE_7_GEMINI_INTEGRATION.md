# Phase 7: Gemini LLM Integration

This phase integrates Google's Gemini API with the existing Fusion Agent architecture securely, guaranteeing that the deterministic components from Phase 6 remain completely authoritative over the LLM.

## 1. Architecture Flow

```
                Event
                  ↓
            ContextBuilder
                  ↓
             FusionInput
                  ↓
         GeminiLLMProvider
                  ↓
         Schema Validation
                  ↓
      Application Validation
          ↓             ↓
   memory IDs        actions
          ↓             ↓
      validate       PolicyEngine
          └──────┬──────┘
                 ↓
            Safe Response
```

## 2. Deterministic Fallback Strategy

Gemini is strictly treated as an intelligence provider. The `MockLLMProvider` is preserved for isolated tests. In production, if Gemini fails (e.g., API key missing, network timeout, malformed output), the `FusionAgent` automatically triggers the **Phase 6 Deterministic DecisionEngine Fallback**.

```
                Event
                  ↓
            ContextBuilder
                  ↓
             FusionInput
                  ↓
        Deterministic Phase 6
            DecisionEngine
                  ↓
             PolicyEngine
                  ↓
            Safe Response
```

## 3. Configuration & Security

- **Environment Variables**: Managed via `.env` (using `.env.example` as a template). `GEMINI_API_KEY`, `GEMINI_MODEL`, and `GEMINI_TIMEOUT` are loaded gracefully.
- **API Key Protection**: The key is *backend-only*. It is never logged, exposed to the frontend, nor leaked in API responses.
- **Untracked Keys**: The `.env` file is heavily `.gitignore`'d and will not be pushed to the repository.

## 4. Prompt Isolation & Injection Defense

- **Separation**: The system instruction is isolated via `SYSTEM_PROMPT`. The runtime variables are built via `build_runtime_context` which wraps event payloads inside explicit XML-like delimiters (e.g., `<current_event_untrusted>`). This guarantees Gemini treats patient transcripts strictly as data, not as directives.
- **Patient Isolation**: The context builder bounds `FusionInput` exclusively to the authenticated `patient_id`. Gemini never receives other patients' data.

## 5. Structured Outputs & Validation

- The provider requests a JSON response explicitly matching the Pydantic `FusionDecision` schema.
- Extraneous fields are stripped.

## 6. Action & Memory Restrictions

- **Memory Validation**: If Gemini attempts to reference a memory via `memory_ids`, the ID is strictly validated against the `approved_memories` passed in the context. If Gemini hallucinated an ID, it is forcefully stripped.
- **Policy Enforcement**: Gemini is permitted to propose an action. The `PolicyEngine` exclusively governs if that action may proceed, rejecting hazardous or unauthorized operations outright.
- **Data Integrity**: Gemini has absolutely NO direct database access.

## 7. Operational Testing

Tests run completely in isolation without requiring `GEMINI_API_KEY` or network egress. Explicit integration mock tests are defined in `test_gemini.py`. All tests pass perfectly locally.

## 8. Frontend & External Services

- The frontend remains entirely decoupled from LLM API calls, interacting strictly with the existing API endpoints.
- Modules for camera processing, raw audio manipulation, and other external APIs have intentionally NOT been implemented yet, ensuring scope integrity.
