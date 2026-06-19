"""
Gemini LLM Client
Thin wrapper around Google's genai SDK with rate limiting and retry logic.
"""

import os
import time
import json
from typing import Type, TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

import config

T = TypeVar("T", bound=BaseModel)


class GeminiClient:
    """
    Gemini API client with built-in rate limiting and retry.
    Designed to stay within free-tier limits.
    """
    
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Set it in your .env file or environment.\n"
                "Get a free key at: https://aistudio.google.com/apikey"
            )
        
        self.model = model or config.GEMINI_MODEL
        self.client = genai.Client(api_key=self.api_key)
        
        # Rate limiting state
        self._request_times: list[float] = []
        self._total_requests = 0
        self._rpm_limit = config.GEMINI_RPM
        
    def _wait_for_rate_limit(self):
        """Enforce RPM rate limit with sliding window."""
        now = time.time()
        # Remove timestamps older than 60 seconds
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        if len(self._request_times) >= self._rpm_limit:
            # Wait until the oldest request in the window expires
            sleep_time = 60 - (now - self._request_times[0]) + 1
            if sleep_time > 0:
                print(f"    ⏳ Rate limit: waiting {sleep_time:.0f}s...")
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
        Generate structured output from Gemini.
        
        Args:
            prompt: The full prompt text
            response_schema: Pydantic model class for structured output
            temperature: Generation temperature
            
        Returns:
            Parsed Pydantic model instance
        """
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
                
                # Parse the structured response
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
                    print(f"    ⚠ API error (attempt {attempt + 1}/{config.MAX_RETRIES}): {error_str[:100]}")
                    print(f"    ⏳ Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
    
    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate plain text from Gemini (no schema).
        Used as a fallback when structured output isn't needed.
        """
        for attempt in range(config.MAX_RETRIES + 1):
            try:
                self._wait_for_rate_limit()
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                    ),
                )
                
                if response.text:
                    return response.text
                raise ValueError("Empty response from Gemini")
                
            except Exception as e:
                error_str = str(e)
                is_retryable = any(code in error_str for code in ["429", "500", "503", "RESOURCE_EXHAUSTED"])
                
                if is_retryable and attempt < config.MAX_RETRIES:
                    delay = config.RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"    ⚠ API error (attempt {attempt + 1}/{config.MAX_RETRIES}): {error_str[:100]}")
                    print(f"    ⏳ Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise
    
    @property
    def total_requests(self) -> int:
        """Total API requests made so far."""
        return self._total_requests
