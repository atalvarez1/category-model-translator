"""
Claude LLM Translation Integration

Uses Claude API for context-aware translation of call center QA phrases.
Understands meaning/intent rather than literal word translation.
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import requests


@dataclass
class TranslationResult:
    """Result of a translation operation."""
    success: bool
    translations: Dict[str, str]  # original -> translated
    error_message: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0


@dataclass
class TokenUsage:
    """Track token usage and costs."""
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def estimate_cost(self, model: str) -> float:
        """Estimate cost based on model pricing."""
        pricing = {
            'claude-3-5-haiku-20241022': {'input': 0.25, 'output': 1.25},  # per 1M tokens
            'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00},
            'claude-sonnet-4-20250514': {'input': 3.00, 'output': 15.00},
        }

        if model not in pricing:
            model = 'claude-3-5-haiku-20241022'  # default

        rates = pricing[model]
        input_cost = (self.input_tokens / 1_000_000) * rates['input']
        output_cost = (self.output_tokens / 1_000_000) * rates['output']
        return input_cost + output_cost


class ClaudeTranslator:
    """Claude API integration for intelligent translation."""

    API_URL = "https://api.anthropic.com/v1/messages"

    # Batch size - number of phrases per API call to avoid token limits
    BATCH_SIZE = 50  # Conservative batch size to avoid output truncation

    # Supported languages with display names
    SUPPORTED_LANGUAGES = {
        'ES': 'Spanish',
        'FR': 'French',
        'FR-CA': 'French (Canadian)',
        'DE': 'German',
        'IT': 'Italian',
        'PT': 'Portuguese',
        'PT-BR': 'Portuguese (Brazilian)',
        'NL': 'Dutch',
        'PL': 'Polish',
        'RU': 'Russian',
        'JA': 'Japanese',
        'KO': 'Korean',
        'ZH': 'Chinese (Simplified)',
        'ZH-TW': 'Chinese (Traditional)',
        'AR': 'Arabic',
        'HI': 'Hindi',
        'TR': 'Turkish',
        'VI': 'Vietnamese',
        'TH': 'Thai',
        'ID': 'Indonesian',
        'MS': 'Malay',
        'FIL': 'Filipino/Tagalog',
    }

    # Available models
    MODELS = {
        'claude-3-5-haiku-20241022': 'Claude 3.5 Haiku (Fast & Cheap)',
        'claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet (Balanced)',
        'claude-sonnet-4-20250514': 'Claude Sonnet 4 (Latest)',
    }

    def __init__(self, api_key: str, model: str = 'claude-3-5-haiku-20241022'):
        """
        Initialize Claude translator.

        Args:
            api_key: Anthropic API key
            model: Model to use for translation
        """
        self.api_key = api_key
        self.model = model
        self._cache: Dict[str, Dict[str, str]] = {}  # lang -> {phrase -> translation}
        self.usage = TokenUsage()

    def _build_system_prompt(self, target_lang: str, category_context: Optional[str] = None) -> str:
        """Build the system prompt for translation."""
        lang_name = self.SUPPORTED_LANGUAGES.get(target_lang, target_lang)

        prompt = f"""You are an expert translator specializing in call center quality assurance (QA) terminology.

Your task is to translate English phrases to {lang_name} for a call center analytics platform.

CONTEXT:
These phrases are used to automatically detect and categorize:
1. What AGENTS say to customers (greetings, empathy statements, closing remarks, probing questions)
2. What CUSTOMERS say (complaints, requests, escalation demands)
3. Quality indicators for call handling

CRITICAL TRANSLATION RULES:

1. TRANSLATE MEANING, NOT LITERAL WORDS
   - "May I" = polite request phrase, NOT "Mayo I" (the month)
   - "How may I help you" = polite service offer
   - "bear with me" = please wait patiently, NOT about bears
   - Understand idioms and translate to equivalent expressions in {lang_name}

2. PRESERVE TECHNICAL SYNTAX EXACTLY
   - Wildcards: "call*" should become "llam*" (Spanish) - keep the asterisk
   - Proximity operators: "~2" stays as "~2"
   - Never translate operators or special characters

3. KEEP PHRASES NATURAL FOR SPOKEN CONVERSATION
   - These phrases will be matched against actual speech transcripts
   - Use natural, conversational {lang_name} that people actually speak
   - Consider regional variations appropriate for {lang_name}

4. OUTPUT FORMAT
   - Return ONLY a valid JSON object mapping original phrases to translations
   - No explanations, no markdown code blocks, just the raw JSON object
   - Example: {{"May I": "Puedo", "thank you": "gracias"}}
   - IMPORTANT: Ensure the JSON is complete and valid
"""

        if category_context:
            prompt += f"""
CURRENT CATEGORY CONTEXT:
{category_context}
This context tells you what type of phrases these are (e.g., empathy, greetings, complaints).
Use this to better understand the intent of ambiguous phrases.
"""

        return prompt

    def _build_user_prompt(self, phrases: List[str]) -> str:
        """Build the user prompt with phrases to translate."""
        phrases_json = json.dumps(phrases, ensure_ascii=False)
        return f"""Translate these {len(phrases)} phrases to the target language. Return ONLY a valid JSON object.

Phrases to translate:
{phrases_json}

Remember:
- Translate meaning/intent, not literal words
- Keep wildcards (*) at the end of word stems
- Return ONLY the JSON object with translations - no markdown, no explanation"""

    def translate_batch(self, phrases: List[str], target_lang: str,
                        category_context: Optional[str] = None,
                        progress_callback: Optional[callable] = None) -> TranslationResult:
        """
        Translate a batch of phrases using Claude.
        Automatically splits into smaller batches to avoid token limits.

        Args:
            phrases: List of phrases to translate
            target_lang: Target language code
            category_context: Optional context about the category
            progress_callback: Optional callback(message, percent) for progress updates

        Returns:
            TranslationResult with translations dict and token usage
        """
        if not phrases:
            return TranslationResult(success=True, translations={})

        # Check cache first
        cache_key = target_lang
        if cache_key not in self._cache:
            self._cache[cache_key] = {}

        uncached_phrases = []
        cached_translations = {}

        for phrase in phrases:
            if phrase in self._cache[cache_key]:
                cached_translations[phrase] = self._cache[cache_key][phrase]
            else:
                uncached_phrases.append(phrase)

        # If all cached, return early
        if not uncached_phrases:
            return TranslationResult(
                success=True,
                translations=cached_translations,
                input_tokens=0,
                output_tokens=0
            )

        # Split into batches to avoid token limits
        all_translations = dict(cached_translations)
        total_input_tokens = 0
        total_output_tokens = 0

        batches = [uncached_phrases[i:i + self.BATCH_SIZE]
                   for i in range(0, len(uncached_phrases), self.BATCH_SIZE)]

        num_batches = len(batches)

        for batch_num, batch in enumerate(batches, 1):
            # Report progress
            if progress_callback:
                pct = 0.2 + (0.5 * (batch_num - 1) / num_batches)  # Progress from 20% to 70%
                progress_callback(f"Translating batch {batch_num} of {num_batches} ({len(batch)} phrases)...", pct)

            result = self._translate_single_batch(batch, target_lang, category_context)

            if not result.success:
                # Return partial results with error
                result.translations = all_translations
                return result

            # Merge translations
            all_translations.update(result.translations)
            total_input_tokens += result.input_tokens
            total_output_tokens += result.output_tokens

            # Cache new translations
            for phrase, translation in result.translations.items():
                self._cache[cache_key][phrase] = translation

        # Update usage tracking
        self.usage.add(total_input_tokens, total_output_tokens)
        estimated_cost = TokenUsage(total_input_tokens, total_output_tokens).estimate_cost(self.model)

        return TranslationResult(
            success=True,
            translations=all_translations,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            estimated_cost=estimated_cost
        )

    def _translate_single_batch(self, phrases: List[str], target_lang: str,
                                 category_context: Optional[str] = None) -> TranslationResult:
        """Translate a single batch of phrases (internal method)."""
        try:
            response = requests.post(
                self.API_URL,
                headers={
                    'x-api-key': self.api_key,
                    'Content-Type': 'application/json',
                    'anthropic-version': '2023-06-01'
                },
                json={
                    'model': self.model,
                    'max_tokens': 8192,  # Increased for larger responses
                    'system': self._build_system_prompt(target_lang, category_context),
                    'messages': [
                        {'role': 'user', 'content': self._build_user_prompt(phrases)}
                    ]
                },
                timeout=180  # Increased timeout for larger batches
            )

            if response.status_code == 401:
                return TranslationResult(
                    success=False,
                    translations={},
                    error_message="Invalid API key. Please check your Anthropic API key."
                )

            if response.status_code == 429:
                return TranslationResult(
                    success=False,
                    translations={},
                    error_message="Rate limit exceeded. Please wait and try again."
                )

            if response.status_code != 200:
                return TranslationResult(
                    success=False,
                    translations={},
                    error_message=f"API error: {response.status_code} - {response.text}"
                )

            data = response.json()

            # Extract token usage
            usage = data.get('usage', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)

            # Check if response was truncated (stop_reason == 'max_tokens')
            stop_reason = data.get('stop_reason', '')
            if stop_reason == 'max_tokens':
                return TranslationResult(
                    success=False,
                    translations={},
                    error_message="Response truncated - batch too large. This shouldn't happen with batching."
                )

            # Parse the response
            content = data.get('content', [{}])[0].get('text', '{}')

            # Clean up the response (remove markdown code blocks if present)
            content = content.strip()
            if content.startswith('```'):
                content = re.sub(r'^```\w*\n?', '', content)
                content = re.sub(r'\n?```$', '', content)
            content = content.strip()

            try:
                translations = json.loads(content)
            except json.JSONDecodeError as e:
                return TranslationResult(
                    success=False,
                    translations={},
                    error_message=f"Failed to parse JSON response. Error: {str(e)}. Response preview: {content[:500]}..."
                )

            return TranslationResult(
                success=True,
                translations=translations,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

        except requests.exceptions.Timeout:
            return TranslationResult(
                success=False,
                translations={},
                error_message="Request timed out. Please try again."
            )
        except requests.exceptions.RequestException as e:
            return TranslationResult(
                success=False,
                translations={},
                error_message=f"Network error: {str(e)}"
            )
        except Exception as e:
            return TranslationResult(
                success=False,
                translations={},
                error_message=f"Unexpected error: {str(e)}"
            )

    def validate_api_key(self) -> bool:
        """Test if the API key is valid."""
        try:
            response = requests.post(
                self.API_URL,
                headers={
                    'x-api-key': self.api_key,
                    'Content-Type': 'application/json',
                    'anthropic-version': '2023-06-01'
                },
                json={
                    'model': self.model,
                    'max_tokens': 10,
                    'messages': [{'role': 'user', 'content': 'Hi'}]
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False

    def estimate_tokens(self, phrases: List[str]) -> int:
        """Rough estimate of tokens for a list of phrases."""
        total_chars = sum(len(p) for p in phrases)
        estimated_tokens = int(total_chars / 3.5)
        prompt_overhead = 500 * ((len(phrases) // self.BATCH_SIZE) + 1)  # Per batch overhead
        return estimated_tokens + prompt_overhead

    def estimate_cost(self, phrases: List[str]) -> float:
        """Estimate cost for translating phrases."""
        input_tokens = self.estimate_tokens(phrases)
        output_tokens = input_tokens  # Assume similar output size
        usage = TokenUsage(input_tokens, output_tokens)
        return usage.estimate_cost(self.model)

    def get_total_usage(self) -> Tuple[int, int, float]:
        """Get total usage for this session."""
        return (
            self.usage.input_tokens,
            self.usage.output_tokens,
            self.usage.estimate_cost(self.model)
        )

    @classmethod
    def get_language_options(cls) -> List[tuple]:
        """Get list of (code, name) tuples for UI dropdowns."""
        return sorted(cls.SUPPORTED_LANGUAGES.items(), key=lambda x: x[1])

    @classmethod
    def get_model_options(cls) -> List[tuple]:
        """Get list of (model_id, display_name) tuples for UI dropdowns."""
        return list(cls.MODELS.items())
