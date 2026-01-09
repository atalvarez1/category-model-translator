"""
CSV Processor for Category Models

Handles reading, processing, and writing CSV files with proper
row insertion logic for translated rule pages.
"""

import pandas as pd
import io
from typing import List, Dict, Optional, Callable, Union, Any
from dataclasses import dataclass, field


@dataclass
class ProcessingResult:
    """Result of CSV processing."""
    success: bool
    output_csv: Optional[str] = None
    rows_processed: int = 0
    rows_added: int = 0
    phrases_translated: int = 0
    characters_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    error_message: Optional[str] = None


class CSVProcessor:
    """Process category model CSVs."""

    # Columns that contain translatable keywords
    KEYWORD_COLUMNS = [
        'Keywords',
        'And Words',
        'And(2) Words',
        'Not Words',
        'Verbatim Keywords',
        'Verbatim And Words',
        'Verbatim And(2) Words',
        'Verbatim Not Words',
        'Parent Doc Keywords',
        'Parent Doc And Words',
        'Parent Doc And(2) Words',
        'Parent Doc Not Words',
        'Other Verbatim Keywords',
        'Other Verbatim And Words',
        'Other Verbatim And(2) Words',
        'Other Verbatim Not Words',
    ]

    # Category columns (not translated, cleared on new rows)
    # Support up to 10 levels to handle any model structure
    CATEGORY_COLUMNS = [
        'Category 1', 'Category 2', 'Category 3', 'Category 4', 'Category 5',
        'Category 6', 'Category 7', 'Category 8', 'Category 9', 'Category 10'
    ]

    # Node-level columns that should NOT be copied to translated rule pages
    NODE_LEVEL_COLUMNS = ['Smart Other', 'Description']

    # AND columns for language attribute placement (in order of preference)
    AND_COLUMNS = ['And Words', 'And(2) Words']

    # Language attribute values for supported languages
    LANGUAGE_ATTRIBUTES = {
        'EN': 'en-us',      # English (for original rows)
        'ES': 'es-us',      # Spanish
        'FR-CA': 'fr-ca',   # French Canadian
    }

    def __init__(self, parser: Any, translator: Any):
        """
        Initialize CSV processor.

        Args:
            parser: KeywordParser instance
            translator: DeepLTranslator or ClaudeTranslator instance
        """
        self.parser = parser
        self.translator = translator

    def process_csv(self, csv_content: str, target_lang: str,
                    progress_callback: Optional[Callable[[str, float], None]] = None) -> ProcessingResult:
        """
        Process a CSV file and add translated rule pages.

        Args:
            csv_content: Raw CSV content as string
            target_lang: Target language code (e.g., 'ES', 'FR-CA')
            progress_callback: Optional callback(status_message, progress_percent)

        Returns:
            ProcessingResult with output CSV and stats
        """
        def update_progress(msg: str, pct: float):
            if progress_callback:
                progress_callback(msg, pct)

        try:
            update_progress("Reading CSV...", 0.05)

            # Parse CSV (skip the header row and empty rows)
            df = self._read_csv(csv_content)
            if df is None or df.empty:
                return ProcessingResult(
                    success=False,
                    error_message="Could not parse CSV file. Please check the format."
                )

            update_progress("Analyzing structure...", 0.1)

            # Identify row types and group rule pages
            row_groups = self._identify_row_groups(df)

            update_progress("Extracting phrases...", 0.15)

            # Collect all phrases to translate (for batching)
            all_phrases = self._collect_all_phrases(df, row_groups)

            if not all_phrases:
                return ProcessingResult(
                    success=False,
                    error_message="No translatable phrases found in the CSV."
                )

            update_progress(f"Translating {len(all_phrases)} unique phrases...", 0.2)

            # Build category context for LLM translator
            category_context = self._build_category_context(df, row_groups)

            # Translate all phrases in batch
            # Check if translator supports category_context (Claude does, DeepL doesn't)
            if hasattr(self.translator, 'translate_batch'):
                try:
                    # Try with category_context and progress_callback (Claude translator)
                    translation_result = self.translator.translate_batch(
                        list(all_phrases),
                        target_lang,
                        category_context=category_context,
                        progress_callback=update_progress
                    )
                except TypeError:
                    # Fall back without category_context/progress (DeepL translator)
                    translation_result = self.translator.translate_batch(
                        list(all_phrases),
                        target_lang
                    )
            else:
                return ProcessingResult(
                    success=False,
                    error_message="Invalid translator instance."
                )

            if not translation_result.success:
                return ProcessingResult(
                    success=False,
                    error_message=translation_result.error_message
                )

            update_progress("Inserting translated rows...", 0.7)

            # Create new dataframe with inserted translated rows
            new_df = self._insert_translated_rows(
                df, row_groups, translation_result.translations, target_lang
            )

            update_progress("Generating output CSV...", 0.9)

            # Generate output CSV
            output_csv = self._write_csv(new_df, csv_content)

            update_progress("Complete!", 1.0)

            rows_added = len(new_df) - len(df)

            # Build result with all available stats
            result = ProcessingResult(
                success=True,
                output_csv=output_csv,
                rows_processed=len(df),
                rows_added=rows_added,
                phrases_translated=len(translation_result.translations),
            )

            # Add DeepL-specific stats
            if hasattr(translation_result, 'characters_used'):
                result.characters_used = translation_result.characters_used

            # Add Claude-specific stats
            if hasattr(translation_result, 'input_tokens'):
                result.input_tokens = translation_result.input_tokens
            if hasattr(translation_result, 'output_tokens'):
                result.output_tokens = translation_result.output_tokens
            if hasattr(translation_result, 'estimated_cost'):
                result.estimated_cost = translation_result.estimated_cost

            return result

        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"Processing error: {str(e)}"
            )

    def _read_csv(self, csv_content: str) -> Optional[pd.DataFrame]:
        """Read and parse the category model CSV format."""
        try:
            lines = csv_content.strip().split('\n')

            # Find the header row (contains "Category 1")
            header_idx = None
            for i, line in enumerate(lines):
                if 'Category 1' in line:
                    header_idx = i
                    break

            if header_idx is None:
                return None

            # Read from header row onwards
            csv_from_header = '\n'.join(lines[header_idx:])
            df = pd.read_csv(io.StringIO(csv_from_header))

            # Remove any completely empty rows
            df = df.dropna(how='all')

            return df

        except Exception:
            return None

    def _identify_row_groups(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identify category rows and their associated rule pages.

        Returns list of dicts:
        {
            'category_row': int or None,
            'category_name': str,  # For context
            'rule_rows': [int, ...],
            'has_rules': bool
        }
        """
        groups = []
        current_group = None

        for idx, row in df.iterrows():
            is_category_row = self._is_category_row(row)
            has_keywords = self._has_keywords(row)

            if is_category_row:
                # Save previous group if exists
                if current_group is not None:
                    groups.append(current_group)

                # Get category name for context
                category_name = self._get_category_name(row)

                # Start new group
                current_group = {
                    'category_row': idx,
                    'category_name': category_name,
                    'rule_rows': [],
                    'has_rules': False
                }

                # Category row might also have keywords
                if has_keywords:
                    current_group['rule_rows'].append(idx)
                    current_group['has_rules'] = True

            elif has_keywords and current_group is not None:
                # This is a rule page row
                current_group['rule_rows'].append(idx)
                current_group['has_rules'] = True

        # Don't forget the last group
        if current_group is not None:
            groups.append(current_group)

        return groups

    def _get_category_name(self, row: pd.Series) -> str:
        """Get the category name from a row (most specific level)."""
        for col in reversed(self.CATEGORY_COLUMNS):  # Check Cat 3 first, then 2, then 1
            if col in row.index:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    return str(val).strip()
        return "Unknown"

    def _is_category_row(self, row: pd.Series) -> bool:
        """Check if row defines a category (has value in Category 1, 2, or 3)."""
        for col in self.CATEGORY_COLUMNS:
            if col in row.index:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    return True
        return False

    def _has_keywords(self, row: pd.Series) -> bool:
        """Check if row has any keyword content."""
        for col in self.KEYWORD_COLUMNS:
            if col in row.index:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    return True
        return False

    def _has_translatable_content(self, row: pd.Series) -> bool:
        """
        Check if row has actual translatable content (not just attributes).

        Returns True only if there are phrases that can be translated,
        not just structured attributes like _verbatimtype:clientverbatimcall.
        """
        for col in self.KEYWORD_COLUMNS:
            if col in row.index:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    # Use the parser to extract translatable phrases
                    extractions = self.parser.extract_translatable_phrases(str(val))
                    if extractions:
                        # Found at least one translatable phrase
                        return True
        return False

    def _build_category_context(self, df: pd.DataFrame, row_groups: List[Dict]) -> str:
        """Build context string about categories for LLM translation."""
        categories = []
        for group in row_groups:
            if group.get('category_name') and group.get('has_rules'):
                categories.append(group['category_name'])

        if categories:
            return f"Categories being translated: {', '.join(categories[:10])}"
        return ""

    def _collect_all_phrases(self, df: pd.DataFrame,
                             row_groups: List[Dict]) -> set:
        """Collect all unique phrases to translate."""
        all_phrases = set()

        for group in row_groups:
            for row_idx in group['rule_rows']:
                row = df.loc[row_idx]
                for col in self.KEYWORD_COLUMNS:
                    if col in row.index:
                        val = row[col]
                        if pd.notna(val) and str(val).strip():
                            extractions = self.parser.extract_translatable_phrases(str(val))
                            for _, phrase, _ in extractions:
                                all_phrases.add(phrase)

        return all_phrases

    def _insert_translated_rows(self, df: pd.DataFrame,
                                 row_groups: List[Dict],
                                 translations: Dict[str, str],
                                 target_lang: str) -> pd.DataFrame:
        """Create new dataframe with translated rows inserted and language attributes added."""
        new_rows = []
        processed_indices = set()

        for group in row_groups:
            # Add the category row if it exists and has no keywords
            cat_row_idx = group['category_row']
            if cat_row_idx is not None and cat_row_idx not in group['rule_rows']:
                new_rows.append(df.loc[cat_row_idx].copy())
                processed_indices.add(cat_row_idx)

            # Track which rows have translatable content for this group
            rows_with_translatable_content = []

            # Add all original rule rows
            for row_idx in group['rule_rows']:
                original_row = df.loc[row_idx].copy()
                row_has_translatable = self._has_translatable_content(df.loc[row_idx])

                # Only add language attribute if row has translatable content
                if row_has_translatable:
                    original_row = self._add_language_attribute(original_row, 'EN')
                    rows_with_translatable_content.append(row_idx)

                new_rows.append(original_row)
                processed_indices.add(row_idx)

            # Add translated versions ONLY for rows that have translatable content
            for row_idx in rows_with_translatable_content:
                translated_row = self._translate_row(
                    df.loc[row_idx], translations
                )
                translated_row = self._add_language_attribute(translated_row, target_lang)
                new_rows.append(translated_row)

        # Add any rows that weren't part of groups
        for idx in df.index:
            if idx not in processed_indices:
                new_rows.append(df.loc[idx].copy())

        return pd.DataFrame(new_rows).reset_index(drop=True)

    def _translate_row(self, row: pd.Series,
                       translations: Dict[str, str]) -> pd.Series:
        """Create a translated copy of a rule row."""
        new_row = row.copy()

        # Clear category columns (translated rows are just rule pages)
        for col in self.CATEGORY_COLUMNS:
            if col in new_row.index:
                new_row[col] = ''

        # Clear node-level columns (Smart Other, Description)
        # These exist at the NODE level, not RULE PAGE level
        for col in self.NODE_LEVEL_COLUMNS:
            if col in new_row.index:
                new_row[col] = ''

        # Translate keyword columns
        for col in self.KEYWORD_COLUMNS:
            if col in new_row.index:
                val = new_row[col]
                if pd.notna(val) and str(val).strip():
                    new_row[col] = self.parser.replace_phrases(str(val), translations)

        return new_row

    def _add_language_attribute(self, row: pd.Series, lang_code: str) -> pd.Series:
        """
        Add language attribute to a row in the appropriate AND column.

        Logic:
        1. If 'And Words' is empty, put language attribute there
        2. Else if 'And(2) Words' is empty, put language attribute there
        3. Else modify 'And(2) Words': ((existing)AND(language:"xx-xx"))

        Args:
            row: The row to modify
            lang_code: Language code (e.g., 'EN', 'ES', 'FR-CA')

        Returns:
            Modified row with language attribute added
        """
        # Get the language attribute value
        lang_attr_value = self.LANGUAGE_ATTRIBUTES.get(lang_code)
        if not lang_attr_value:
            # Unsupported language, return row unchanged
            return row

        lang_attribute = f'language:"{lang_attr_value}"'

        # Check AND columns in order
        for i, col in enumerate(self.AND_COLUMNS):
            if col not in row.index:
                continue

            val = row[col]
            is_empty = pd.isna(val) or not str(val).strip()

            if is_empty:
                # Found an empty AND column - put the attribute here
                row[col] = lang_attribute
                return row

        # All AND columns have content - modify the last one
        last_col = self.AND_COLUMNS[-1]  # 'And(2) Words'
        if last_col in row.index:
            existing = str(row[last_col]).strip()
            # Wrap: ((existing)AND(language:"xx-xx"))
            row[last_col] = f'(({existing})AND({lang_attribute}))'

        return row

    def _write_csv(self, df: pd.DataFrame, original_content: str) -> str:
        """Generate output CSV, preserving original header and blank row after column headers."""
        lines = original_content.strip().split('\n')
        header_lines = []
        for line in lines:
            if 'Category 1' in line:
                break
            header_lines.append(line)

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        data_csv = csv_buffer.getvalue()

        # Split the CSV data into header and rows
        data_lines = data_csv.split('\n')
        column_header = data_lines[0] if data_lines else ''
        data_rows = '\n'.join(data_lines[1:]) if len(data_lines) > 1 else ''

        # Reconstruct with blank row between column headers and data
        # Format: [timestamp header] [empty] [column headers] [empty] [data rows]
        if header_lines:
            return '\n'.join(header_lines) + '\n' + column_header + '\n""\n' + data_rows
        return column_header + '\n""\n' + data_rows
