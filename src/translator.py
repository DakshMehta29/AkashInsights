"""
Translation Module - Make-in-India Multilingual Support
Supports Hindi, Tamil, Bengali, and other Indian languages.
"""

from __future__ import annotations

from typing import Dict, Optional

# Try to import IndicTrans (preferred for Indian languages)
try:
    from indicTrans.inference.engine import Model
    INDICTRANS_AVAILABLE = True
except ImportError:
    INDICTRANS_AVAILABLE = False

# Try to import IndicBERT
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Try deep-translator (works with Python 3.13)
try:
    from deep_translator import GoogleTranslator as DeepGoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False

# Fallback: Use googletrans (may not work with Python 3.13+)
try:
    from googletrans import Translator as GoogleTranslator
    GOOGLETRANS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    GOOGLETRANS_AVAILABLE = False


# Language codes
LANGUAGE_CODES = {
    "english": "en",
    "hindi": "hi",
    "tamil": "ta",
    "bengali": "bn",
    "telugu": "te",
    "marathi": "mr",
    "gujarati": "gu",
    "kannada": "kn",
    "malayalam": "ml",
    "punjabi": "pa",
    "urdu": "ur"
}


class Translator:
    """Multilingual translator for Indian languages."""
    
    def __init__(self):
        """Initialize translator with best available backend."""
        self.model = None
        self.backend = None
        self._load_model()
    
    def _load_model(self):
        """Load translation model (prefer deep-translator, then googletrans, then IndicTrans, then IndicBERT)."""
        # Try deep-translator first (works with Python 3.13+)
        if DEEP_TRANSLATOR_AVAILABLE:
            try:
                print("📂 Loading Google Translate (via deep-translator)...")
                self.model = DeepGoogleTranslator
                self.backend = "deep_translator"
                print("✅ Google Translate ready")
                return
            except Exception as e:
                print(f"⚠️  Deep Translator load error: {e}")
        
        # Try googletrans (may not work with Python 3.13+)
        if GOOGLETRANS_AVAILABLE:
            try:
                print("📂 Loading Google Translate...")
                self.model = GoogleTranslator()
                self.backend = "googletrans"
                print("✅ Google Translate ready")
                return
            except Exception as e:
                print(f"⚠️  Google Translate load error: {e}")
        
        # Try IndicTrans (best for Indian languages, but requires setup)
        if INDICTRANS_AVAILABLE:
            try:
                print("📂 Loading IndicTrans model...")
                self.model = Model(expdir="indic-en")
                self.backend = "indictrans"
                print("✅ IndicTrans loaded")
                return
            except Exception as e:
                print(f"⚠️  IndicTrans load error: {e}")
        
        # Try IndicBERT (requires HuggingFace auth)
        if TRANSFORMERS_AVAILABLE:
            try:
                print("📂 Loading IndicBERT model...")
                model_name = "ai4bharat/indictrans2-en-indic-1B"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.backend = "indicbert"
                print("✅ IndicBERT loaded")
                return
            except Exception as e:
                print(f"⚠️  IndicBERT load error: {e}")
        
        print("⚠️  No translation backend available. Install: pip install deep-translator")
    
    def translate_text(
        self,
        text: str,
        target_lang: str,
        source_lang: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Translate text to target language.
        
        Args:
            text: Text to translate
            target_lang: Target language (name or code, e.g., "hindi" or "hi")
            source_lang: Source language (auto-detect if None)
            
        Returns:
            Dictionary with translation and metadata
        """
        if self.model is None:
            return {
                "original": text,
                "translated": text,
                "error": "No translation model available"
            }
        
        # Normalize language codes
        target_code = LANGUAGE_CODES.get(target_lang.lower(), target_lang.lower())
        if source_lang:
            source_code = LANGUAGE_CODES.get(source_lang.lower(), source_lang.lower())
        else:
            source_code = None
        
        try:
            if self.backend == "indictrans":
                # IndicTrans format
                if source_code is None:
                    source_code = "en"  # Default to English
                translated = self.model.translate_paragraph(text, source_code, target_code)
            
            elif self.backend == "indicbert":
                # IndicBERT format
                if source_code is None:
                    source_code = "eng_Latn"
                target_code_formatted = f"{target_code}_Deva" if target_code != "en" else "eng_Latn"
                inputs = self.tokenizer(text, return_tensors="pt", padding=True)
                outputs = self.model.generate(**inputs, max_length=512)
                translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            elif self.backend == "deep_translator":
                # Deep Translator (works with Python 3.13+)
                try:
                    translator = self.model(source='auto', target=target_code)
                    translated = translator.translate(text)
                    # Try to detect source language
                    if source_code is None:
                        try:
                            from deep_translator import single_detection
                            source_code = single_detection(text, api_key=None) or 'auto'
                        except:
                            source_code = 'auto'
                except Exception as e:
                    raise Exception(f"Deep Translator error: {e}")
            
            elif self.backend == "googletrans":
                # Google Translate (googletrans 4.0.0rc1 API - may not work with Python 3.13+)
                try:
                    # Auto-detect source language if not provided
                    if source_code is None:
                        detected = self.model.detect(text)
                        source_code = detected.lang if hasattr(detected, 'lang') else detected
                    
                    # Translate
                    result = self.model.translate(text, src=source_code, dest=target_code)
                    translated = result.text
                    source_code = result.src if hasattr(result, 'src') else source_code
                except Exception as googletrans_error:
                    # Try alternative approach for googletrans
                    try:
                        if source_code is None:
                            source_code = 'auto'
                        result = self.model.translate(text, src=source_code, dest=target_code)
                        translated = result.text
                        source_code = result.src if hasattr(result, 'src') else source_code
                    except Exception as e2:
                        raise Exception(f"Google Translate error: {e2}")
            
            else:
                translated = text
            
            return {
                "original": text,
                "translated": translated,
                "source_lang": source_code or "auto",
                "target_lang": target_code,
                "backend": self.backend
            }
        
        except Exception as e:
            return {
                "original": text,
                "translated": text,
                "error": str(e)
            }
    
    def detect_language(self, text: str) -> Dict[str, str]:
        """
        Detect language of input text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with detected language
        """
        if self.backend == "googletrans" and self.model:
            try:
                detected = self.model.detect(text)
                return {
                    "language": detected.lang,
                    "confidence": detected.confidence
                }
            except:
                pass
        
        # Simple heuristic: check for Devanagari, Tamil, etc.
        # This is a basic fallback
        if any(ord(char) >= 0x0900 and ord(char) <= 0x097F for char in text):
            return {"language": "hi", "confidence": 0.7}  # Hindi
        elif any(ord(char) >= 0x0B80 and ord(char) <= 0x0BFF for char in text):
            return {"language": "ta", "confidence": 0.7}  # Tamil
        elif any(ord(char) >= 0x0980 and ord(char) <= 0x09FF for char in text):
            return {"language": "bn", "confidence": 0.7}  # Bengali
        
        return {"language": "en", "confidence": 0.5}  # Default to English


def translate_text(text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
    """
    Convenience function for translation.
    
    Args:
        text: Text to translate
        target_lang: Target language
        source_lang: Source language (optional)
        
    Returns:
        Translated text
    """
    translator = Translator()
    result = translator.translate_text(text, target_lang, source_lang)
    return result.get("translated", text)


if __name__ == "__main__":
    print("Translation Module")
    print("=" * 50)
    
    translator = Translator()
    
    if translator.model:
        # Test translation
        test_text = "Engine status is normal. All systems operational."
        result = translator.translate_text(test_text, "hindi")
        print(f"\nOriginal: {result['original']}")
        print(f"Translated: {result['translated']}")
        print(f"Backend: {result['backend']}")
    else:
        print("\n⚠️  No translation model available.")
        print("   Install: pip install googletrans==4.0.0rc1")
        print("   Or: pip install indicTrans (for better Indian language support)")

