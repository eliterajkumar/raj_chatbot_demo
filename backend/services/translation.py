from googletrans import Translator, LANGUAGES

def translate_text(text: str, dest_lang: str) -> str:
    """
    Translates text to a destination language using Googletrans.
    Note: This library scrapes Google Translate and is not for production use.
    For production, use the official Google Cloud Translate API.
    """
    if not text:
        return ""
    if dest_lang not in LANGUAGES:
        raise ValueError(f"Invalid destination language: {dest_lang}")

    try:
        translator = Translator()
        translated = translator.translate(text, dest=dest_lang)
        return translated.text
    except Exception as e:
        raise ConnectionError(f"Translation failed: {e}") 