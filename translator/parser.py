"""
Keyword Parser for Category Models

Extracts translatable phrases from keyword syntax while preserving:
- Boolean operators (AND, OR)
- Proximity operators (~1, ~2, etc.)
- Wildcards (*)
- Attribute patterns (attribute:value)
- Parentheses and commas
"""

import re
from typing import List, Tuple, Dict, Set


class KeywordParser:
    """Parses keyword syntax and extracts translatable content."""

    # Patterns to preserve (never translate)
    BOOLEAN_OPERATORS = {'AND', 'OR', 'TO'}  # TO is used in ranges like [0 TO 0.96]

    def __init__(self):
        self._translation_cache: Dict[str, str] = {}

    def extract_translatable_phrases(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract all translatable phrases from keyword text.

        Returns list of tuples: (original_match, phrase_to_translate, match_type)
        - original_match: The exact string found (including quotes)
        - phrase_to_translate: The text to send to translation API
        - match_type: 'double_quoted', 'single_quoted', or 'unquoted'
        """
        if not text or not isinstance(text, str):
            return []

        results = []
        matched_spans: List[Tuple[int, int]] = []

        # 1. Find double-quoted phrases: ""phrase""
        # These use escaped double quotes in CSV format
        double_quote_pattern = re.compile(r'""([^"]+)""')
        for match in double_quote_pattern.finditer(text):
            phrase = match.group(1).strip()
            if phrase and self._is_translatable_phrase(phrase):
                results.append((match.group(0), phrase, 'double_quoted'))
                matched_spans.append((match.start(), match.end()))

        # 2. Find single-quoted phrases: "phrase"
        # Only if not overlapping with double-quoted
        single_quote_pattern = re.compile(r'"([^"]+)"')
        for match in single_quote_pattern.finditer(text):
            # Skip if overlaps with already matched span
            if self._overlaps(match.start(), match.end(), matched_spans):
                continue

            phrase = match.group(1).strip()
            if phrase and self._is_translatable_phrase(phrase):
                results.append((match.group(0), phrase, 'single_quoted'))
                matched_spans.append((match.start(), match.end()))

        # 3. Find unquoted words
        # Split by common delimiters and find translatable words
        unquoted = self._extract_unquoted_words(text, matched_spans)
        results.extend(unquoted)

        return results

    def _overlaps(self, start: int, end: int, spans: List[Tuple[int, int]]) -> bool:
        """Check if a span overlaps with any existing spans."""
        for s_start, s_end in spans:
            # Overlaps if: start < s_end AND end > s_start
            if start < s_end and end > s_start:
                return True
        return False

    def _is_translatable_phrase(self, phrase: str) -> bool:
        """Check if a phrase should be translated."""
        if not phrase or not phrase.strip():
            return False

        # Skip if it looks like an attribute pattern
        if ':' in phrase:
            return False

        # Skip if it's just punctuation or operators
        if phrase.upper() in self.BOOLEAN_OPERATORS:
            return False

        # Skip if it's just parentheses/commas
        if re.match(r'^[\(\)\[\],\s]+$', phrase):
            return False

        return True

    def _extract_unquoted_words(self, text: str,
                                 quoted_spans: List[Tuple[int, int]]) -> List[Tuple[str, str, str]]:
        """Extract unquoted words that should be translated."""
        results = []

        # First, mask out the quoted spans and attribute patterns
        masked = list(text)
        for start, end in quoted_spans:
            for i in range(start, min(end, len(masked))):
                masked[i] = ' '

        masked_text = ''.join(masked)

        # Mask out attribute patterns (word:value, word:[range], etc.)
        attr_pattern = re.compile(r'\b[\w_]+:\S+')
        for match in attr_pattern.finditer(masked_text):
            for i in range(match.start(), match.end()):
                if i < len(masked):
                    masked[i] = ' '

        masked_text = ''.join(masked)

        # Now find words in the masked text
        # A word is: letters (and possibly ending with *)
        word_pattern = re.compile(r'\b([a-zA-Z][a-zA-Z\']*\*?)\b')

        for match in word_pattern.finditer(masked_text):
            word = match.group(1)

            # Skip boolean operators
            if word.upper().rstrip('*') in self.BOOLEAN_OPERATORS:
                continue

            # Skip very short words (likely syntax artifacts)
            base_word = word.rstrip('*')
            if len(base_word) < 3:
                continue

            # This is a translatable word
            results.append((word, word, 'unquoted'))

        return results

    def replace_phrases(self, text: str, translations: Dict[str, str]) -> str:
        """
        Replace original phrases with translations in the text.

        Args:
            text: Original keyword text
            translations: Dict mapping original phrases to translations

        Returns:
            Text with phrases replaced by translations
        """
        if not text or not isinstance(text, str):
            return text

        result = text

        # Sort by length (longest first) to avoid partial replacements
        sorted_translations = sorted(translations.items(),
                                     key=lambda x: len(x[0]),
                                     reverse=True)

        for original, translation in sorted_translations:
            if not original or not translation:
                continue

            # Preserve wildcards: if original ends with *, keep it in translation
            has_wildcard = original.endswith('*')
            clean_original = original.rstrip('*')
            clean_translation = translation.rstrip('*')  # Remove if API added one

            if has_wildcard:
                clean_translation = clean_translation + '*'

            # Handle double-quoted phrases: ""phrase""
            dq_original = f'""{clean_original}""'
            dq_translation = f'""{clean_translation}""'
            result = result.replace(dq_original, dq_translation)

            # Handle single-quoted phrases: "phrase"
            sq_original = f'"{clean_original}"'
            sq_translation = f'"{clean_translation}"'
            result = result.replace(sq_original, sq_translation)

            # Handle unquoted words (with word boundaries)
            # Be careful not to replace parts of attribute patterns
            if ' ' not in clean_original:
                # For single words, use word boundary matching
                # But exclude if followed by colon (attribute pattern)
                pattern = re.compile(
                    r'(?<!["\w])' + re.escape(clean_original) + r'(?!["\w:])',
                    re.IGNORECASE
                )
                # Preserve case of first letter
                def replace_preserve_case(m):
                    matched = m.group(0)
                    if matched[0].isupper():
                        return clean_translation.capitalize()
                    return clean_translation.lower()

                result = pattern.sub(replace_preserve_case, result)

        return result

    def get_unique_phrases(self, extractions: List[Tuple[str, str, str]]) -> Set[str]:
        """Get unique phrases to translate (for batching API calls)."""
        return {phrase for _, phrase, _ in extractions}


# Utility function for testing
def test_parser():
    """Test the parser with sample data."""
    parser = KeywordParser()

    test_cases = [
        '"hi my name is" "thank you for calling"',
        'cb_sentence_quartile:1,_verbatimtype:agentverbatimcall',
        '((""can you"", ""could you"")AND(""help me understand"", ""tell me""))',
        'sorry, apologize',
        'confidence:[0 TO 0.96]',
        '""thanks calling""~2',
        'recorded,"call*, line"',
        'hurry',
        'dude, bruh, brah, homie',
    ]

    print('Testing KeywordParser...')
    print('=' * 60)

    for test in test_cases:
        print(f'\nInput: {test[:60]}...' if len(test) > 60 else f'\nInput: {test}')
        results = parser.extract_translatable_phrases(test)
        phrases = [phrase for _, phrase, _ in results]
        print(f'Extracted phrases: {phrases}')

    print('\n' + '=' * 60)
    print('Parser test complete!')


if __name__ == '__main__':
    test_parser()
