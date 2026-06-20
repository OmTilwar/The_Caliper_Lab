"""
LLM Client — Multi-Backend (Gemini / OpenRouter)
Automatically selects OpenRouter if OPENROUTER_API_KEY is set, 
otherwise falls back to Gemini. Includes rate limiting and retry logic.
"""

import os
import time
import json
from typing import Type, TypeVar

import instructor
from openai import OpenAI
from pydantic import BaseModel

import config

T = TypeVar("T", bound=BaseModel)


class GeminiClient:
    """
    LLM client with support for OpenRouter (via OpenAI SDK + instructor)
    and Google Gemini. Includes built-in rate limiting and retry.
    
    Automatically uses OpenRouter if OPENROUTER_API_KEY is found in env,
    otherwise falls back to Gemini.
    """
    
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self._total_requests = 0
        self._request_times: list[float] = []
        
        # Check for OpenRouter first
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        
        if openrouter_key:
            self._backend = "openrouter"
            self.model = model or config.OPENROUTER_MODEL
            self._rpm_limit = config.OPENROUTER_RPM
            
            # Create OpenAI client pointed at OpenRouter
            raw_client = OpenAI(
                base_url=config.OPENROUTER_BASE_URL,
                api_key=openrouter_key,
            )
            # Wrap with instructor for structured Pydantic output
            self.client = instructor.from_openai(raw_client)
            
            print(f"  Using OpenRouter backend ({self.model})")
        else:
            # Fall back to Gemini
            self._backend = "gemini"
            self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "No API key found. Set OPENROUTER_API_KEY or GOOGLE_API_KEY "
                    "in your .env file.\n"
                    "Get a free OpenRouter key at: https://openrouter.ai/\n"
                    "Get a free Gemini key at: https://aistudio.google.com/apikey"
                )
            
            from google import genai
            self.model = model or config.GEMINI_MODEL
            self._rpm_limit = config.GEMINI_RPM
            self.client = genai.Client(api_key=self.api_key)
            
            print(f"  Using Gemini backend ({self.model})")
        
    def _wait_for_rate_limit(self):
        """Enforce RPM rate limit with sliding window."""
        now = time.time()
        # Remove timestamps older than 60 seconds
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        if len(self._request_times) >= self._rpm_limit:
            # Wait until the oldest request in the window expires
            sleep_time = 60 - (now - self._request_times[0]) + 1
            if sleep_time > 0:
                print(f"    ... Rate limit: waiting {sleep_time:.0f}s...")
                time.sleep(sleep_time)
        
        self._request_times.append(time.time())
        self._total_requests += 1
    
    def generate(
        self,
        prompt: str,
        response_schema: Type[T],
        temperature: float = 0.3,
    ) -> T:
        """
        Generate structured output from the LLM.
        
        Args:
            prompt: The full prompt text
            response_schema: Pydantic model class for structured output
            temperature: Generation temperature
            
        Returns:
            Parsed Pydantic model instance
        """
        if self._backend == "openrouter":
            return self._generate_openrouter(prompt, response_schema, temperature)
        else:
            return self._generate_gemini(prompt, response_schema, temperature)
    
    def _generate_openrouter(
        self,
        prompt: str,
        response_schema: Type[T],
        temperature: float,
    ) -> T:
        """Generate using OpenRouter via OpenAI SDK + instructor."""
        for attempt in range(config.MAX_RETRIES + 1):
            try:
                self._wait_for_rate_limit()
                
                result = self.client.chat.completions.create(
                    model=self.model,
                    response_model=response_schema,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise financial document analysis assistant. "
                                       "Always respond with valid structured data exactly matching "
                                       "the requested schema."
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=temperature,
                    max_retries=0,  # We handle retries ourselves
                )
                
                return result
                    
            except Exception as e:
                error_str = str(e)
                is_retryable = any(code in error_str for code in [
                    "429", "500", "503", "rate_limit", "overloaded", "timeout"
                ])
                
                if is_retryable and attempt < config.MAX_RETRIES:
                    delay = config.RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"    ! API error (attempt {attempt + 1}/{config.MAX_RETRIES}): {error_str[:100]}")
                    print(f"    ... Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
    
    def _generate_gemini(
        self,
        prompt: str,
        response_schema: Type[T],
        temperature: float,
    ) -> T:
        """Generate using Google Gemini (legacy fallback)."""
        from google.genai import types
        
        for attempt in range(config.MAX_RETRIES + 1):
            try:
                self._wait_for_rate_limit()
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=temperature,
                    ),
                )
                
                if response.text:
                    data = json.loads(response.text)
                    return response_schema.model_validate(data)
                else:
                    raise ValueError("Empty response from Gemini")
                    
            except Exception as e:
                error_str = str(e)
                is_retryable = any(code in error_str for code in ["429", "500", "503", "RESOURCE_EXHAUSTED"])
                
                if is_retryable and attempt < config.MAX_RETRIES:
                    delay = config.RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"    ! API error (attempt {attempt + 1}/{config.MAX_RETRIES}): {error_str[:100]}")
                    print(f"    ... Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
    
    @property
    def total_requests(self) -> int:
        """Total API requests made so far."""
        return self._total_requests
