"""
DeepL Translation Integration

Handles API communication with DeepL for translating phrases.
"""

import requests
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TranslationResult:
    """Result of a translation operation."""
    success: bool
    translations: Dict[str, str]  # original -> translated
    error_message: Optional[str] = None
    characters_used: int = 0


class DeepLTranslator:
    """DeepL API integration for translation."""

    # DeepL API endpoints
    FREE_API_URL = "https://api-free.deepl.com/v2/translate"
    PRO_API_URL = "https://api.deepl.com/v2/translate"

    # Supported languages (code -> display name)
    SUPPORTED_LANGUAGES = {
        'BG': 'Bulgarian',
        'CS': 'Czech',
        'DA': 'Danish',
        'DE': 'German',
        'EL': 'Greek',
        'EN-GB': 'English (British)',
        'EN-US': 'English (American)',
        'ES': 'Spanish',
        'ET': 'Estonian',
        'FI': 'Finnish',
        'FR': 'French',
        'HU': 'Hungarian',
        'ID': 'Indonesian',
        'IT': 'Italian',
        'JA': 'Japanese',
        'KO': 'Korean',
        'LT': 'Lithuanian',
        'LV': 'Latvian',
        'NB': 'Norwegian',
        'NL': 'Dutch',
        'PL': 'Polish',
        'PT-BR': 'Portuguese (Brazilian)',
        'PT-PT': 'Portuguese (European)',
        'RO': 'Romanian',
        'RU': 'Russian',
        'SK': 'Slovak',
        'SL': 'Slovenian',
        'SV': 'Swedish',
        'TR': 'Turkish',
        'UK': 'Ukrainian',
        'ZH': 'Chinese (Simplified)',
    }

    def __init__(self, api_key: str, use_free_api: bool = True):
        """
        Initialize DeepL translator.

        Args:
            api_key: DeepL API key
            use_free_api: Use free API endpoint (default True)
        """
        self.api_key = api_key
        self.api_url = self.FREE_API_URL if use_free_api else self.PRO_API_URL
        self._cache: Dict[str, Dict[str, str]] = {}  # lang -> {phrase -> translation}

    def translate_batch(self, phrases: List[str], target_lang: str,
                        source_lang: str = 'EN') -> TranslationResult:
        """
        Translate a batch of phrases.

        Args:
            phrases: List of phrases to translate
            target_lang: Target language code (e.g., 'ES', 'FR')
            source_lang: Source language code (default 'EN')

        Returns:
            TranslationResult with translations dict
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
                characters_used=0
            )

        # Call DeepL API
        try:
            response = requests.post(
                self.api_url,
                headers={
                    'Authorization': f'DeepL-Auth-Key {self.api_key}',
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                data={
                    'text': uncached_phrases,
                    'target_lang': target_lang,
                    'source_lang': source_lang,
                },
                timeout=60
            )

            if response.status_code == 403:
                return TranslationResult(
                    success=False,
                    translations={},
                    error_message="Invalid API key. Please check your DeepL API key."
                )

            if response.status_code == 456:
                return TranslationResult(
                    success=False,
                    translations={},
                    error_message="Quota exceeded. Please check your DeepL account."
                )

            if response.status_code != 200:
                return TranslationResult(
                    success=False,
                    translations={},
                    error_message=f"API error: {response.status_code} - {response.text}"
                )

            data = response.json()
            translations_list = data.get('translations', [])

            # Map translations back to original phrases
            new_translations = {}
            chars_used = 0

            for i, phrase in enumerate(uncached_phrases):
                if i < len(translations_list):
                    translated = translations_list[i].get('text', phrase)
                    new_translations[phrase] = translated
                    self._cache[cache_key][phrase] = translated
                    chars_used += len(phrase)

            # Combine cached and new translations
            all_translations = {**cached_translations, **new_translations}

            return TranslationResult(
                success=True,
                translations=all_translations,
                characters_used=chars_used
            )

        except requests.exceptions.Timeout:
            return TranslationResult(
                success=False,
                translations=cached_translations,
                error_message="Request timed out. Please try again."
            )
        except requests.exceptions.RequestException as e:
            return TranslationResult(
                success=False,
                translations=cached_translations,
                error_message=f"Network error: {str(e)}"
            )
        except Exception as e:
            return TranslationResult(
                success=False,
                translations=cached_translations,
                error_message=f"Unexpected error: {str(e)}"
            )

    def validate_api_key(self) -> bool:
        """Test if the API key is valid."""
        try:
            response = requests.post(
                self.api_url,
                headers={
                    'Authorization': f'DeepL-Auth-Key {self.api_key}',
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                data={
                    'text': ['test'],
                    'target_lang': 'ES',
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_usage(self) -> Optional[Dict]:
        """Get API usage statistics."""
        try:
            usage_url = self.api_url.replace('/translate', '/usage')
            response = requests.get(
                usage_url,
                headers={'Authorization': f'DeepL-Auth-Key {self.api_key}'},
                timeout=10
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    @classmethod
    def get_language_options(cls) -> List[tuple]:
        """Get list of (code, name) tuples for UI dropdowns."""
        return sorted(cls.SUPPORTED_LANGUAGES.items(), key=lambda x: x[1])
