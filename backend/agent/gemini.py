import json
import logging
from agent.types import FusionInput, FusionDecision
from agent.llm import LLMProvider
from agent.prompts import SYSTEM_PROMPT, build_runtime_context
from config.settings import settings

try:
    from google import genai
    from google.genai import types
    has_genai = True
except ImportError:
    has_genai = False

class GeminiLLMProvider(LLMProvider):
    def __init__(self):
        if not has_genai:
            logging.error("google-genai SDK is not installed.")
            raise RuntimeError("google-genai SDK is not installed.")
            
        if not settings.GEMINI_API_KEY:
            logging.warning("GEMINI_API_KEY is not set.")
            raise ValueError("GEMINI_API_KEY is required for GeminiLLMProvider.")
            
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GEMINI_MODEL
        self.timeout = settings.GEMINI_TIMEOUT
        
    def generate_decision(self, context: FusionInput) -> FusionDecision:
        try:
            runtime_context = build_runtime_context(context)
            
            # Call Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=runtime_context,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=FusionDecision,
                    temperature=0.0
                )
            )
            
            # The SDK parses the JSON response back into a string or dictionary based on output
            response_text = response.text
            if not response_text:
                raise ValueError("Empty response from Gemini")
                
            decision_data = json.loads(response_text)
            
            # Validate output using Pydantic model
            decision = FusionDecision(**decision_data)
            
            # Validate Memory IDs - Gemini MUST NOT hallucinate memory IDs
            approved_ids = {m["id"] for m in context.approved_memories}
            for mem_id in decision.memory_ids:
                if mem_id not in approved_ids:
                    logging.warning(f"Gemini proposed unapproved memory ID: {mem_id}. Rejecting memory.")
                    # Clear invalid memory ids
                    decision.memory_ids = []
                    break
                    
            return decision
            
        except Exception as e:
            logging.error(f"Gemini LLM Provider failed: {str(e)}. Triggering deterministic fallback.")
            # Trigger deterministic fallback manually if Gemini fails
            intent = "unclear"
            response_type = "clarification"
            response_text = "I'm having a little trouble understanding. Could you tell me more?"
            action = "request_clarification"
            
            return FusionDecision(
                intent=intent,
                response_type=response_type,
                response_text=response_text,
                action=action,
                confidence=0.0,
                memory_ids=[],
                reason=f"Fallback due to Gemini failure: {str(e)}"
            )
