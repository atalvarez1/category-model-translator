# Category Model Translator

## Project Overview

A web-based tool to translate category model exports (CSV) into other languages. The tool preserves the model structure and special syntax while translating keyword phrases.

**Key Feature:** Uses Claude LLM for context-aware translation that understands call center terminology and translates *meaning* rather than literal words.

## Tech Stack

- **Backend/UI**: Python + Streamlit
- **Translation APIs**:
  - **Claude (Recommended)**: Context-aware, understands call center phrases
  - **DeepL**: Fast, literal translation (fallback option)
- **CSV Processing**: pandas
- **Parsing**: regex for keyword syntax extraction

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## User Workflow

1. Upload a CSV export
2. Select translation service (Claude recommended)
3. Enter API key
4. Select target language (e.g., Spanish, French Canadian)
5. Click "Translate" (see cost estimate first for Claude)
6. Download the translated CSV

---

## How It Works - Summary

When you translate a model, the tool performs these steps:

### 1. Parse the CSV Structure
- Identifies category nodes (rows with Category 1/2/3 values)
- Groups rule pages under their parent categories
- Extracts all translatable phrases from keyword columns

### 2. Translate Phrases
- Sends phrases to Claude/DeepL in batches
- Claude uses specialized prompts to understand call center context
- Translations preserve wildcards (`*`) and special syntax

### 3. Insert Translated Rule Pages
For each category with rule pages:
- **Original rule pages are kept** (with `language:"en-us"` attribute added)
- **New translated rule pages are inserted** after the originals
- New rows have empty Category/Smart Other/Description columns
- New rows include the target language attribute (e.g., `language:"es-us"`)

### 4. Add Language Attributes
Every rule page gets a language attribute in an AND column:
- Allows filtering by language at runtime
- Original English rows: `language:"en-us"`
- Translated rows: `language:"es-us"` or `language:"fr-ca"`

### End Result
The translated model contains **both** English and translated rule pages. Since rule pages are OR'd together, the model will match phrases in **either** language. The language attribute allows you to filter results by language if needed.

---

## Translation Services

### Claude (Recommended)

Uses Anthropic's Claude API with a specialized prompt that:
- Understands these are **call center QA phrases**
- Knows context (agent speech, customer speech, empathy, etc.)
- Translates **meaning/intent**, not literal words
- Example: "May I" → "¿Puedo" (NOT "Mayo I")

**Models available:**
- Claude 3.5 Haiku (fastest, cheapest): $0.25/1M input, $1.25/1M output
- Claude 3.5 Sonnet (more capable): $3/1M input, $15/1M output
- Claude Sonnet 4 (latest): $3/1M input, $15/1M output

**Get API key:** https://console.anthropic.com/

### OpenAI / ChatGPT

Uses OpenAI's GPT models with the same specialized prompt approach.

**Models available:**
- GPT-4o Mini (fastest, cheapest): $0.15/1M input, $0.60/1M output
- GPT-4o (balanced): $2.50/1M input, $10/1M output
- GPT-4 Turbo (most capable): $10/1M input, $30/1M output

**Get API key:** https://platform.openai.com/api-keys

### Google Gemini

Uses Google's Gemini models with the same specialized prompt approach.

**Models available:**
- Gemini 1.5 Flash (fastest, cheapest): $0.075/1M input, $0.30/1M output
- Gemini 2.0 Flash (latest): $0.10/1M input, $0.40/1M output
- Gemini 1.5 Pro (most capable): $1.25/1M input, $5/1M output

**Get API key:** https://aistudio.google.com/apikey

### DeepL (Fallback)

Traditional machine translation. Fast but literal - does not understand context.
- Free tier: 500K chars/month
- Pro: $5.49 per 1M chars

**Get API key:** https://www.deepl.com/pro-api

---

## CSV Structure

### File Format

- **Row 1**: Header with timestamp (`"Categorization Tree at: [date]"`)
- **Row 2**: Empty
- **Row 3**: Column headers
- **Row 4**: Empty
- **Row 5+**: Data rows

### Column Structure

| Column | Name | Purpose | Translated? |
|--------|------|---------|-------------|
| A | Category 1 | Top-level category (L1) | No |
| B | Category 2 | Mid-level category (L2) | No |
| C | Category 3 | Leaf-level category (L3) | No |
| D | Description | Category description | No |
| E | Smart Other | Boolean flag (NODE-level only) | No - cleared on new rows |
| F | Keywords | Primary keyword rules | **Yes** |
| G | And Words | Required co-occurring words | **Yes** |
| H | And(2) Words | Secondary AND condition | **Yes** |
| I | Not Words | Exclusion words | **Yes** |
| J+ | Verbatim columns | Additional rule columns | **Yes** |

### Important: Node-Level vs Rule-Page-Level Columns

- **Smart Other** and **Description** exist at the NODE level only
- They appear in the same row as the first rule page
- When creating translated rule pages, these columns are **cleared** (not copied)

### Hierarchy Logic

- **Category rows**: Have a value in Category 1, 2, or 3
- **Rule page rows**: Empty category columns, but have Keywords/And Words/etc.
- A category (node) can have multiple rule pages (rows)
- Rule pages are **OR'd together** within a category
- Parent categories may or may not have their own rules

### Example Structure

```
Row 5:  QA Model [Category 1]     <- Root node
Row 6:  "" | Call Handling [Cat2] <- L2 parent node (no rules)
Row 7:  "" | "" | Proper Opening  <- L3 behavior + Rule Page 1 (has Smart Other)
Row 8:  "" | "" | ""              <- Rule Page 2
Row 9:  "" | "" | ""              <- Rule Page 3
Row 10: "" | "" | ""              <- Rule Page 4
Row 11: "" | "" | Proper Closing  <- Next L3 behavior
```

---

## Translation Rules

### What to Translate

Only the **keyword/rule columns** (F onwards):
- Keywords
- And Words
- And(2) Words
- Not Words
- Verbatim Keywords
- Verbatim And Words
- Verbatim And(2) Words
- Verbatim Not Words
- Parent Doc Keywords (and And/Not variants)
- Other Verbatim Keywords (and And/Not variants)

### What to Preserve (NEVER Translate)

1. **Category names** (Category 1, 2, 3 columns)
2. **Description column**
3. **Smart Other column** (cleared on new rows, not copied)
4. **Boolean operators**: `AND`, `OR`, `TO`
5. **Proximity operators**: `~1`, `~2`, `~3`, etc.
6. **Wildcards**: `*`
7. **Attribute structures**: Any `attribute_name:attribute_value` pattern
   - Examples: `cb_sentence_quartile:1`, `_verbatimtype:agentverbatimcall`, `confidence:[0 TO 0.96]`, `cb_bc_person:*`, `cb_bc_profanity:*`
8. **Parentheses and commas** used for grouping

### Keyword Syntax Examples

**Simple phrases:**
```
"hi my name is" "thank you for calling"
```
Translate to (Spanish):
```
"hola mi nombre es" "gracias por llamar"
```

**Complex boolean with attributes:**
```
((""can you"", ""could you"")AND(""help me understand"", ""tell me"")),cb_sentence_quartile:1,_verbatimtype:agentverbatimcall
```
Translate only the quoted phrases, preserve everything else:
```
((""puedes"", ""podrías"")AND(""ayúdame a entender"", ""dime"")),cb_sentence_quartile:1,_verbatimtype:agentverbatimcall
```

**Mixed with unquoted words:**
```
sorry, apologize
```
These single unquoted words should also be translated:
```
perdón, disculpa
```

---

## Row Insertion Logic (Same Model Mode)

For each category node that has rule pages:

1. Identify all rule page rows (consecutive rows with keywords after a category row)
2. After the **last** rule page, insert N new rows (one translated version per original)
3. New rows have:
   - **Empty** Category 1/2/3 columns (they inherit from the behavior)
   - **Empty** Smart Other column (node-level property, not copied)
   - **Empty** Description column (node-level property, not copied)
   - **Translated** keyword columns

### Before Translation (4 rule pages)

```
Row 7:  [Proper Opening] | Smart Other: false | Rule Page 1 (EN)
Row 8:  [empty]          |                    | Rule Page 2 (EN)
Row 9:  [empty]          |                    | Rule Page 3 (EN)
Row 10: [empty]          |                    | Rule Page 4 (EN)
Row 11: [Proper Closing] | ...
```

### After Translation (Spanish added)

```
Row 7:  [Proper Opening] | Smart Other: false | Rule Page 1 (EN)
Row 8:  [empty]          |                    | Rule Page 2 (EN)
Row 9:  [empty]          |                    | Rule Page 3 (EN)
Row 10: [empty]          |                    | Rule Page 4 (EN)
Row 11: [empty]          | [empty]            | Rule Page 1 (ES) <- INSERTED
Row 12: [empty]          | [empty]            | Rule Page 2 (ES) <- INSERTED
Row 13: [empty]          | [empty]            | Rule Page 3 (ES) <- INSERTED
Row 14: [empty]          | [empty]            | Rule Page 4 (ES) <- INSERTED
Row 15: [Proper Closing] | ...
```

This makes the behavior match **either** English **or** Spanish phrases (rule pages are OR'd).

---

## Language Attribute Feature

Each rule page is tagged with a language attribute to differentiate between languages. This allows the model to filter by language at runtime.

### Supported Languages

| Language Code | Attribute Value |
|--------------|-----------------|
| EN (English) | `language:"en-us"` |
| ES (Spanish) | `language:"es-us"` |
| FR-CA (French Canadian) | `language:"fr-ca"` |

### Placement Logic

The language attribute is added to **one** AND column per rule page:

1. **If `And Words` is empty** → Put `language:"xx-xx"` there
2. **Else if `And(2) Words` is empty** → Put `language:"xx-xx"` there
3. **Else (both have content)** → Modify `And(2) Words` by wrapping:
   ```
   ((existing_content)AND(language:"xx-xx"))
   ```

### Example

**Before (original row with existing AND content):**
| And Words | And(2) Words |
|-----------|--------------|
| `"hello"` | `verbatimtype:"clientverbatim"` |

**After processing:**

Original row (English):
| And Words | And(2) Words |
|-----------|--------------|
| `"hello"` | `((verbatimtype:"clientverbatim")AND(language:"en-us"))` |

Translated row (Spanish):
| And Words | And(2) Words |
|-----------|--------------|
| `"hola"` | `((verbatimtype:"clientverbatim")AND(language:"es-us"))` |

---

## Parsing Strategy

### Step 1: Extract Translatable Content

Use regex to identify:
1. **Double-quoted phrases**: `""phrase""` (CSV-escaped quotes)
2. **Single-quoted phrases**: `"phrase"`
3. **Unquoted single words**: Words not inside quotes (min 3 chars)
4. **Skip attribute patterns**: `word:value` or `word:[range]`
5. **Skip operators**: AND, OR, TO

### Step 2: Translate

- Batch all unique phrases for API efficiency
- For Claude: Include category context for better understanding
- Cache translations to avoid re-translating duplicates

### Step 3: Reassemble

- Replace original phrases with translations
- Preserve all syntax, operators, and attributes in original positions
- Handle wildcards: keep `*` at end of translated word stems

---

## Claude Translation Prompt

The Claude translator uses a specialized system prompt:

```
You are translating call center quality assurance phrases from English to {language}.

These phrases are used to:
1. Detect what AGENTS say to customers (greetings, empathy, closing)
2. Detect what CUSTOMERS say (complaints, requests, escalations)

CRITICAL RULES:
- Translate the MEANING/INTENT, not literal words
- "May I" = polite request phrase (NOT the month of May)
- Preserve wildcards: "call*" stays as "{translated}*"
- Preserve proximity operators: "~2" stays as "~2"
- Keep phrases natural for spoken conversation
```

---

## Supported Languages

### Claude (22+ languages)
Spanish, French, French (Canadian), German, Italian, Portuguese, Portuguese (Brazilian), Dutch, Polish, Russian, Japanese, Korean, Chinese (Simplified), Chinese (Traditional), Arabic, Hindi, Turkish, Vietnamese, Thai, Indonesian, Malay, Filipino/Tagalog

### DeepL (29+ languages)
Bulgarian, Czech, Danish, German, Greek, English (British/American), Spanish, Estonian, Finnish, French, Hungarian, Indonesian, Italian, Japanese, Korean, Lithuanian, Latvian, Norwegian, Dutch, Polish, Portuguese (Brazilian/European), Romanian, Russian, Slovak, Slovenian, Swedish, Turkish, Ukrainian, Chinese

---

## File Structure

```
model-translator/
├── CLAUDE.md               # This file - project documentation
├── app.py                  # Streamlit application
├── requirements.txt        # Python dependencies
├── .gitignore
├── translator/
│   ├── __init__.py
│   ├── parser.py           # Keyword syntax parser
│   ├── translator.py       # DeepL API integration
│   ├── llm_translator.py   # Claude API integration
│   ├── openai_translator.py # OpenAI/ChatGPT API integration
│   ├── gemini_translator.py # Google Gemini API integration
│   └── csv_processor.py    # CSV manipulation and row insertion
└── test/
    └── QA Model (5).csv    # Sample test file
```

---

## Cost Tracking

The app displays:
- **Before translation**: Estimated cost (for Claude)
- **After translation**:
  - Input/output tokens used
  - Actual cost
  - Rows processed/added
  - Phrases translated

---

## Future Enhancements (Out of Scope for MVP)

- [ ] Multi-language translation in one pass
- [ ] Translation memory/glossary for consistent terminology
- [ ] Preview changes before download
- [ ] "New Model" mode (separate translated model)
- [ ] Batch processing multiple files
- [ ] Export translation log for review
