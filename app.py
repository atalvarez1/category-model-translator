"""
Category Model Translator

A Streamlit app for translating category models to other languages
while preserving the model structure and syntax.

Supports two translation backends:
- DeepL: Fast, literal translation (cheaper)
- Claude: Context-aware, intelligent translation (better quality)
"""

import streamlit as st
from translator import KeywordParser, DeepLTranslator, ClaudeTranslator, OpenAITranslator, GeminiTranslator, CSVProcessor


# Page config
st.set_page_config(
    page_title="Category Model Translator",
    page_icon="🌐",
    layout="centered"
)

# Title and description
st.title("🌐 Category Model Translator")
st.markdown("""
Translate your category models to other languages.
Upload a CSV export, select your target language, and download the translated model.
""")

# Expandable "How it works" section
with st.expander("ℹ️ How it works"):
    st.markdown("""
    **This tool translates keyword phrases while preserving model structure.**

    When you translate a model:
    1. **Original rule pages are kept** with a `language:"en-us"` attribute added
    2. **Translated rule pages are inserted** after the originals with the target language attribute (e.g., `language:"es-us"`)
    3. **Both languages work together** - rule pages are OR'd, so the model matches phrases in either language
    4. **Language attributes** allow filtering results by language at runtime

    *Only keyword columns are translated. Category names, descriptions, and special syntax (AND, OR, wildcards, attributes) are preserved.*
    """)

st.divider()

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")

    # Translation service selection
    translation_service = st.radio(
        "Translation Service",
        options=["Claude (Recommended)", "OpenAI / ChatGPT", "Google Gemini", "DeepL"],
        help="LLM services (Claude, OpenAI, Gemini) provide context-aware translation. DeepL is faster but more literal."
    )

    st.divider()

    if translation_service == "Claude (Recommended)":
        st.subheader("🤖 Claude API")

        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Get your API key at https://console.anthropic.com/"
        )

        # Model selection
        model_options = ClaudeTranslator.get_model_options()
        selected_model = st.selectbox(
            "Model",
            options=[m[0] for m in model_options],
            format_func=lambda x: dict(model_options)[x],
            index=0,
            help="Haiku is fastest and cheapest. Sonnet is more capable."
        )

        if api_key:
            st.success("✓ API key entered")
        else:
            st.warning("Please enter your Anthropic API key")

        st.markdown("""
        **Pricing (per 1M tokens):**
        - Haiku: $0.25 in / $1.25 out
        - Sonnet: $3 in / $15 out

        [Get API Key →](https://console.anthropic.com/)
        """)

    elif translation_service == "OpenAI / ChatGPT":
        st.subheader("🧠 OpenAI API")

        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Get your API key at https://platform.openai.com/api-keys"
        )

        # Model selection
        model_options = OpenAITranslator.get_model_options()
        selected_model = st.selectbox(
            "Model",
            options=[m[0] for m in model_options],
            format_func=lambda x: dict(model_options)[x],
            index=0,
            help="GPT-4o Mini is fastest and cheapest. GPT-4 Turbo is most capable."
        )

        if api_key:
            st.success("✓ API key entered")
        else:
            st.warning("Please enter your OpenAI API key")

        st.markdown("""
        **Pricing (per 1M tokens):**
        - GPT-4o Mini: $0.15 in / $0.60 out
        - GPT-4o: $2.50 in / $10 out

        [Get API Key →](https://platform.openai.com/api-keys)
        """)

    elif translation_service == "Google Gemini":
        st.subheader("✨ Google Gemini API")

        api_key = st.text_input(
            "Google AI API Key",
            type="password",
            help="Get your API key at https://aistudio.google.com/apikey"
        )

        # Model selection
        model_options = GeminiTranslator.get_model_options()
        selected_model = st.selectbox(
            "Model",
            options=[m[0] for m in model_options],
            format_func=lambda x: dict(model_options)[x],
            index=0,
            help="Flash models are fastest and cheapest. Pro is most capable."
        )

        if api_key:
            st.success("✓ API key entered")
        else:
            st.warning("Please enter your Google AI API key")

        st.markdown("""
        **Pricing (per 1M tokens):**
        - Gemini 1.5 Flash: $0.075 in / $0.30 out
        - Gemini 1.5 Pro: $1.25 in / $5 out

        [Get API Key →](https://aistudio.google.com/apikey)
        """)

    else:  # DeepL
        st.subheader("📝 DeepL API")

        api_key = st.text_input(
            "DeepL API Key",
            type="password",
            help="Get your free API key at https://www.deepl.com/pro-api"
        )

        selected_model = None  # Not applicable for DeepL

        if api_key:
            st.success("✓ API key entered")
        else:
            st.warning("Please enter your DeepL API key")

        st.markdown("""
        **Pricing:**
        - Free tier: 500K chars/month
        - Pro: $5.49 per 1M chars

        [Get API Key →](https://www.deepl.com/pro-api)
        """)

# Main content
col1, col2 = st.columns(2)

with col1:
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Upload CSV Export",
        type=['csv'],
        help="Upload your category model CSV export"
    )

with col2:
    # Language selection - use appropriate language list
    if translation_service == "Claude (Recommended)":
        language_options = ClaudeTranslator.get_language_options()
    elif translation_service == "OpenAI / ChatGPT":
        language_options = OpenAITranslator.get_language_options()
    elif translation_service == "Google Gemini":
        language_options = GeminiTranslator.get_language_options()
    else:
        language_options = DeepLTranslator.get_language_options()

    language_dict = dict(language_options)

    # Find French Canadian or default to Spanish
    default_lang = 'ES'
    if 'FR-CA' in language_dict:
        default_idx = list(language_dict.keys()).index('FR-CA')
    elif 'ES' in language_dict:
        default_idx = list(language_dict.keys()).index('ES')
    else:
        default_idx = 0

    selected_lang = st.selectbox(
        "🌍 Target Language",
        options=[code for code, name in language_options],
        format_func=lambda x: f"{language_dict[x]} ({x})",
        index=default_idx
    )

# Cost estimation section
st.divider()

if uploaded_file and api_key:
    # Show cost estimate for LLM services
    if translation_service in ["Claude (Recommended)", "OpenAI / ChatGPT", "Google Gemini"]:
        csv_content = uploaded_file.getvalue().decode('utf-8-sig')

        # Quick phrase count estimate
        parser = KeywordParser()
        if translation_service == "Claude (Recommended)":
            translator = ClaudeTranslator(api_key, selected_model)
        elif translation_service == "OpenAI / ChatGPT":
            translator = OpenAITranslator(api_key, selected_model)
        else:
            translator = GeminiTranslator(api_key, selected_model)

        # Rough estimate of phrases
        phrase_count = csv_content.count('""') // 2 + csv_content.count('"') // 4
        estimated_cost = translator.estimate_cost(['phrase'] * max(phrase_count, 100))

        st.info(f"💰 **Estimated cost:** ${estimated_cost:.4f} (rough estimate based on file size)")

        # Reset file position
        uploaded_file.seek(0)

    # Process button
    if st.button("🚀 Translate Model", type="primary", use_container_width=True):

        # Read file content
        csv_content = uploaded_file.getvalue().decode('utf-8-sig')

        # Initialize components
        parser = KeywordParser()

        if translation_service == "Claude (Recommended)":
            translator = ClaudeTranslator(api_key, selected_model)
        elif translation_service == "OpenAI / ChatGPT":
            translator = OpenAITranslator(api_key, selected_model)
        elif translation_service == "Google Gemini":
            translator = GeminiTranslator(api_key, selected_model)
        else:
            translator = DeepLTranslator(api_key, use_free_api=True)

        processor = CSVProcessor(parser, translator)

        # Progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(msg: str, pct: float):
            progress_bar.progress(pct)
            status_text.text(msg)

        # Process the CSV
        with st.spinner("Processing..."):
            result = processor.process_csv(
                csv_content,
                selected_lang,
                progress_callback=update_progress
            )

        # Show results
        if result.success:
            st.success("✅ Translation complete!")

            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows Processed", result.rows_processed)
            with col2:
                st.metric("Rows Added", result.rows_added)
            with col3:
                st.metric("Phrases Translated", result.phrases_translated)

            # Cost/usage stats (service-specific)
            if translation_service in ["Claude (Recommended)", "OpenAI / ChatGPT", "Google Gemini"]:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Input Tokens", f"{result.input_tokens:,}")
                with col2:
                    st.metric("Output Tokens", f"{result.output_tokens:,}")
                with col3:
                    st.metric("Actual Cost", f"${result.estimated_cost:.4f}")
            else:
                if result.characters_used > 0:
                    st.metric("Characters Used", f"{result.characters_used:,}")

            # Download button
            original_name = uploaded_file.name.replace('.csv', '')
            output_filename = f"{original_name}_{selected_lang}.csv"

            st.download_button(
                label="📥 Download Translated CSV",
                data=result.output_csv,
                file_name=output_filename,
                mime="text/csv",
                type="primary",
                use_container_width=True
            )

        else:
            st.error(f"❌ Error: {result.error_message}")

elif not api_key:
    st.info("👈 Please enter your API key in the sidebar to continue.")
elif not uploaded_file:
    st.info("👆 Please upload a CSV file to continue.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    Built with Streamlit | Translation by Claude, OpenAI, Gemini, or DeepL<br>
    <strong>v1.5</strong> - Added OpenAI and Gemini translation options
</div>
""", unsafe_allow_html=True)
