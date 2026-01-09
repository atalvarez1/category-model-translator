"""
Google Gemini Translation Integration

Uses Google Gemini API for context-aware translation of call center QA phrases.
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
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
            'gemini-1.5-flash': {'input': 0.075, 'output': 0.30},  # per 1M tokens
            'gemini-1.5-pro': {'input': 1.25, 'output': 5.00},
            'gemini-2.0-flash': {'input': 0.10, 'output': 0.40},
        }

        if model not in pricing:
            model = 'gemini-1.5-flash'  # default

        rates = pricing[model]
        input_cost = (self.input_tokens / 1_000_000) * rates['input']
        output_cost = (self.output_tokens / 1_000_000) * rates['output']
        return input_cost + output_cost


class GeminiTranslator:
    """Google Gemini API integration for intelligent translation."""

    API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    # Batch size - number of phrases per API call to avoid token limits
    BATCH_SIZE = 50

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
        'gemini-1.5-flash': 'Gemini 1.5 Flash (Fast & Cheap)',
        'gemini-2.0-flash': 'Gemini 2.0 Flash (Latest)',
        'gemini-1.5-pro': 'Gemini 1.5 Pro (Most Capable)',
    }

    def __init__(self, api_key: str, model: str = 'gemini-1.5-flash'):
        """
        Initialize Gemini translator.

        Args:
            api_key: Google AI API key
            model: Model to use for translation
        """
        self.api_key = api_key
        self.model = model
        self._cache: Dict[str, Dict[str, str]] = {}  # lang -> {phrase -> translation}
        self.usage = TokenUsage()

    def _get_api_url(self) -> str:
        """Get the API URL for the current model."""
        return f"{self.API_BASE}/{self.model}:generateContent?key={self.api_key}"

    def _build_prompt(self, phrases: List[str], target_lang: str, category_context: Optional[str] = None) -> str:
        """Build the full prompt for translation."""
        lang_name = self.SUPPORTED_LANGUAGES.get(target_lang, target_lang)
        phrases_json = json.dumps(phrases, ensure_ascii=False)

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

        prompt += f"""
Translate these {len(phrases)} phrases to {lang_name}. Return ONLY a valid JSON object.

Phrases to translate:
{phrases_json}

Remember:
- Translate meaning/intent, not literal words
- Keep wildcards (*) at the end of word stems
- Return ONLY the JSON object with translations - no markdown, no explanation"""

        return prompt

    def translate_batch(self, phrases: List[str], target_lang: str,
                        category_context: Optional[str] = None,
                        progress_callback: Optional[callable] = None) -> TranslationResult:
        """
        Translate a batch of phrases using Gemini.
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

        if not uncached_phrases:
            return TranslationResult(
                success=True,
                translations=cached_translations,
                input_tokens=0,
                output_tokens=0
            )

        # Split into batches
        all_translations = dict(cached_translations)
        total_input_tokens = 0
        total_output_tokens = 0

        batches = [uncached_phrases[i:i + self.BATCH_SIZE]
                   for i in range(0, len(uncached_phrases), self.BATCH_SIZE)]

        num_batches = len(batches)

        for batch_num, batch in enumerate(batches, 1):
            if progress_callback:
                pct = 0.2 + (0.5 * (batch_num - 1) / num_batches)
                progress_callback(f"Translating batch {batch_num} of {num_batches} ({len(batch)} phrases)...", pct)

            result = self._translate_single_batch(batch, target_lang, category_context)

            if not result.success:
                result.translations = all_translations
                return result

            all_translations.update(result.translations)
            total_input_tokens += result.input_tokens
            total_output_tokens += result.output_tokens

            for phrase, translation in result.translations.items():
                self._cache[cache_key][phrase] = translation

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
        """Translate a single batch of phrases."""
        try:
            response = requests.post(
                self._get_api_url(),
                headers={
                    'Content-Type': 'application/json',
                },
                json={
                    'contents': [{
                        'parts': [{
                            'text': self._build_prompt(phrases, target_lang, category_context)
                        }]
                    }],
                    'generationConfig': {
                        'maxOutputTokens': 8192,
                    }
                },
                timeout=180
            )

            if response.status_code == 400:
                error_data = response.json()
                error_msg = error_data.get('error', {}).get('message', 'Bad request')
                if 'API_KEY' in error_msg.upper() or 'API key' in error_msg:
                    return TranslationResult(
                        success=False,
                        translations={},
                        error_message="Invalid API key. Please check your Google AI API key."
                    )
                return TranslationResult(
                    success=False,
                    translations={},
                    error_message=f"API error: {error_msg}"
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
            usage_metadata = data.get('usageMetadata', {})
            input_tokens = usage_metadata.get('promptTokenCount', 0)
            output_tokens = usage_metadata.get('candidatesTokenCount', 0)

            # Get the response content
            candidates = data.get('candidates', [])
            if not candidates:
                return TranslationResult(
                    success=False,
                    translations={},
                    error_message="No response generated"
                )

            content = candidates[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')

            # Check finish reason
            finish_reason = candidates[0].get('finishReason', '')
            if finish_reason == 'MAX_TOKENS':
                return TranslationResult(
                    success=False,
                    translations={},
                    error_message="Response truncated - batch too large."
                )

            # Clean up the response
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
                self._get_api_url(),
                headers={
                    'Content-Type': 'application/json',
                },
                json={
                    'contents': [{
                        'parts': [{
                            'text': 'Hi'
                        }]
                    }],
                    'generationConfig': {
                        'maxOutputTokens': 10,
                    }
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
        prompt_overhead = 500 * ((len(phrases) // self.BATCH_SIZE) + 1)
        return estimated_tokens + prompt_overhead

    def estimate_cost(self, phrases: List[str]) -> float:
        """Estimate cost for translating phrases."""
        input_tokens = self.estimate_tokens(phrases)
        output_tokens = input_tokens
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
