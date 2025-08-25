"""
Internationalization (i18n) and Localization for Quantum Error Mitigation

Provides comprehensive i18n support including:
- Multi-language support for user interfaces and documentation
- Cultural adaptation of quantum terminology and concepts
- Localized quantum metrics and units
- Region-specific quantum hardware considerations
- Collaborative research across language barriers
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from abc import ABC, abstractmethod

class Language(Enum):
    """Supported languages for internationalization"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ITALIAN = "it"
    DUTCH = "nl"
    SWEDISH = "sv"
    ARABIC = "ar"
    HINDI = "hi"

class CulturalContext(Enum):
    """Cultural contexts that affect quantum terminology and presentation"""
    WESTERN_ACADEMIC = "western_academic"
    EASTERN_ACADEMIC = "eastern_academic"
    INDUSTRY_FOCUSED = "industry_focused"
    EDUCATIONAL = "educational"
    RESEARCH_COLLABORATIVE = "research_collaborative"
    GENERAL_PUBLIC = "general_public"

@dataclass
class LocalizationConfig:
    """Configuration for localization settings"""
    primary_language: Language
    fallback_languages: List[Language]
    cultural_context: CulturalContext
    number_format: str  # e.g., "1,234.56" vs "1.234,56"
    decimal_separator: str
    thousand_separator: str
    date_format: str
    time_format: str
    currency_symbol: str
    measurement_system: str  # "metric" or "imperial"
    scientific_notation: str  # "western" or "eastern"
    
@dataclass
class TranslationEntry:
    """Single translation entry with metadata"""
    key: str
    translations: Dict[str, str]
    context: str
    category: str
    technical_level: str  # "beginner", "intermediate", "advanced", "expert"
    last_updated: float = field(default_factory=time.time)
    verified: bool = False
    cultural_notes: Dict[str, str] = field(default_factory=dict)

class TranslationService:
    """Service for managing translations and localization"""
    
    def __init__(self):
        self.translations: Dict[str, TranslationEntry] = {}
        self.translation_cache: Dict[Tuple[str, str], str] = {}
        self.missing_translations: Dict[str, List[str]] = {}
        
        # Initialize quantum-specific terminology
        self._initialize_quantum_terminology()
        
        # Cultural adaptation rules
        self.cultural_adaptations: Dict[CulturalContext, Dict[str, Any]] = {
            CulturalContext.WESTERN_ACADEMIC: {
                "formality_level": "formal",
                "technical_depth": "high",
                "notation_preference": "standard",
                "unit_preference": "si_units"
            },
            CulturalContext.EASTERN_ACADEMIC: {
                "formality_level": "very_formal",
                "technical_depth": "high",
                "notation_preference": "traditional",
                "unit_preference": "si_units"
            },
            CulturalContext.INDUSTRY_FOCUSED: {
                "formality_level": "professional",
                "technical_depth": "moderate",
                "notation_preference": "practical",
                "unit_preference": "practical_units"
            },
            CulturalContext.EDUCATIONAL: {
                "formality_level": "accessible",
                "technical_depth": "progressive",
                "notation_preference": "pedagogical",
                "unit_preference": "educational_units"
            }
        }
    
    def _initialize_quantum_terminology(self) -> None:
        """Initialize quantum computing terminology translations"""
        
        quantum_terms = [
            {
                "key": "quantum_error_mitigation",
                "translations": {
                    "en": "Quantum Error Mitigation",
                    "es": "Mitigación de Errores Cuánticos",
                    "fr": "Atténuation d'Erreurs Quantiques",
                    "de": "Quantenfehler-Milderung",
                    "ja": "量子誤り緩和",
                    "zh-CN": "量子误差缓解",
                    "ko": "양자 오류 완화",
                    "pt": "Mitigação de Erros Quânticos",
                    "ru": "Смягчение Квантовых Ошибок"
                },
                "context": "Main field of study",
                "category": "core_concepts"
            },
            {
                "key": "zero_noise_extrapolation", 
                "translations": {
                    "en": "Zero-Noise Extrapolation",
                    "es": "Extrapolación de Ruido Cero",
                    "fr": "Extrapolation à Bruit Zéro",
                    "de": "Null-Rauschen-Extrapolation",
                    "ja": "ゼロノイズ外挿法",
                    "zh-CN": "零噪声外推法",
                    "ko": "제로 노이즈 외삽법",
                    "pt": "Extrapolação de Ruído Zero",
                    "ru": "Экстраполяция Нулевого Шума"
                },
                "context": "Error mitigation technique",
                "category": "techniques"
            },
            {
                "key": "quantum_fidelity",
                "translations": {
                    "en": "Quantum Fidelity", 
                    "es": "Fidelidad Cuántica",
                    "fr": "Fidélité Quantique",
                    "de": "Quantenfidelität",
                    "ja": "量子忠実度",
                    "zh-CN": "量子保真度", 
                    "ko": "양자 충실도",
                    "pt": "Fidelidade Quântica",
                    "ru": "Квантовая Точность"
                },
                "context": "Measure of quantum state accuracy",
                "category": "metrics"
            },
            {
                "key": "coherence_time",
                "translations": {
                    "en": "Coherence Time",
                    "es": "Tiempo de Coherencia", 
                    "fr": "Temps de Cohérence",
                    "de": "Kohärenzzeit",
                    "ja": "コヒーレンス時間",
                    "zh-CN": "相干时间",
                    "ko": "결맞음 시간",
                    "pt": "Tempo de Coerência",
                    "ru": "Время Когерентности"
                },
                "context": "Duration of quantum coherence",
                "category": "physics"
            },
            {
                "key": "quantum_circuit",
                "translations": {
                    "en": "Quantum Circuit",
                    "es": "Circuito Cuántico",
                    "fr": "Circuit Quantique", 
                    "de": "Quantenschaltung",
                    "ja": "量子回路",
                    "zh-CN": "量子电路",
                    "ko": "양자 회로",
                    "pt": "Circuito Quântico",
                    "ru": "Квантовая Схема"
                },
                "context": "Quantum algorithm representation",
                "category": "hardware"
            },
            {
                "key": "qubit",
                "translations": {
                    "en": "Qubit",
                    "es": "Cubit",
                    "fr": "Qubit",
                    "de": "Qubit",
                    "ja": "量子ビット",
                    "zh-CN": "量子比特", 
                    "ko": "큐비트",
                    "pt": "Qubit",
                    "ru": "Кубит"
                },
                "context": "Quantum bit",
                "category": "hardware"
            },
            {
                "key": "superposition",
                "translations": {
                    "en": "Superposition",
                    "es": "Superposición",
                    "fr": "Superposition",
                    "de": "Überlagerung",
                    "ja": "重ね合わせ",
                    "zh-CN": "叠加态",
                    "ko": "중첩", 
                    "pt": "Superposição",
                    "ru": "Суперпозиция"
                },
                "context": "Quantum mechanical principle",
                "category": "physics"
            },
            {
                "key": "entanglement",
                "translations": {
                    "en": "Entanglement",
                    "es": "Entrelazamiento",
                    "fr": "Intrication",
                    "de": "Verschränkung", 
                    "ja": "もつれ",
                    "zh-CN": "纠缠",
                    "ko": "얽힘",
                    "pt": "Emaranhamento",
                    "ru": "Запутанность"
                },
                "context": "Quantum correlation phenomenon",
                "category": "physics"
            }
        ]
        
        # Add all quantum terms to translations
        for term in quantum_terms:
            entry = TranslationEntry(
                key=term["key"],
                translations=term["translations"],
                context=term["context"],
                category=term["category"],
                technical_level="intermediate",
                verified=True
            )
            self.translations[term["key"]] = entry
    
    def translate(self, key: str, language: Language, 
                 fallback_languages: Optional[List[Language]] = None,
                 context: Optional[str] = None) -> str:
        """Translate a key to the specified language"""
        
        # Check cache first
        cache_key = (key, language.value)
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # Get translation entry
        if key not in self.translations:
            self._record_missing_translation(key, language.value)
            return f"[{key}]"  # Return key in brackets if missing
        
        entry = self.translations[key]
        
        # Try primary language
        if language.value in entry.translations:
            translation = entry.translations[language.value]
            self.translation_cache[cache_key] = translation
            return translation
        
        # Try fallback languages
        if fallback_languages:
            for fallback_lang in fallback_languages:
                if fallback_lang.value in entry.translations:
                    translation = entry.translations[fallback_lang.value]
                    self.translation_cache[cache_key] = translation
                    return translation
        
        # Default to English if available
        if Language.ENGLISH.value in entry.translations:
            translation = entry.translations[Language.ENGLISH.value]
            self.translation_cache[cache_key] = translation
            return translation
        
        # Record missing translation and return key
        self._record_missing_translation(key, language.value)
        return f"[{key}]"
    
    def add_translation(self, key: str, language: Language, 
                       translation: str, context: str = "",
                       category: str = "general") -> None:
        """Add or update a translation"""
        
        if key not in self.translations:
            self.translations[key] = TranslationEntry(
                key=key,
                translations={},
                context=context,
                category=category,
                technical_level="intermediate"
            )
        
        entry = self.translations[key]
        entry.translations[language.value] = translation
        entry.last_updated = time.time()
        
        # Clear relevant cache entries
        cache_keys_to_remove = [
            (k, l) for k, l in self.translation_cache.keys()
            if k == key
        ]
        for cache_key in cache_keys_to_remove:
            del self.translation_cache[cache_key]
    
    def _record_missing_translation(self, key: str, language: str) -> None:
        """Record missing translation for future attention"""
        if key not in self.missing_translations:
            self.missing_translations[key] = []
        
        if language not in self.missing_translations[key]:
            self.missing_translations[key].append(language)
    
    def get_translation_coverage(self, language: Language) -> Dict[str, Any]:
        """Get translation coverage statistics for a language"""
        
        total_keys = len(self.translations)
        translated_keys = sum(
            1 for entry in self.translations.values()
            if language.value in entry.translations
        )
        
        coverage_by_category = {}
        for entry in self.translations.values():
            category = entry.category
            if category not in coverage_by_category:
                coverage_by_category[category] = {"total": 0, "translated": 0}
            
            coverage_by_category[category]["total"] += 1
            if language.value in entry.translations:
                coverage_by_category[category]["translated"] += 1
        
        return {
            "language": language.value,
            "overall_coverage": translated_keys / max(1, total_keys),
            "total_keys": total_keys,
            "translated_keys": translated_keys,
            "missing_keys": total_keys - translated_keys,
            "coverage_by_category": coverage_by_category,
            "last_updated": time.time()
        }

class LocalizedQuantumMetrics:
    """Handles localization of quantum metrics and units"""
    
    def __init__(self, config: LocalizationConfig):
        self.config = config
        
        # Unit conversion factors
        self.unit_conversions = {
            "time": {
                "ns": {"factor": 1e-9, "symbol": "ns", "name": "nanoseconds"},
                "μs": {"factor": 1e-6, "symbol": "μs", "name": "microseconds"},
                "ms": {"factor": 1e-3, "symbol": "ms", "name": "milliseconds"},
                "s": {"factor": 1.0, "symbol": "s", "name": "seconds"}
            },
            "frequency": {
                "Hz": {"factor": 1.0, "symbol": "Hz", "name": "hertz"},
                "kHz": {"factor": 1e3, "symbol": "kHz", "name": "kilohertz"},
                "MHz": {"factor": 1e6, "symbol": "MHz", "name": "megahertz"},
                "GHz": {"factor": 1e9, "symbol": "GHz", "name": "gigahertz"}
            },
            "fidelity": {
                "percentage": {"factor": 100.0, "symbol": "%", "name": "percentage"},
                "decimal": {"factor": 1.0, "symbol": "", "name": "decimal"}
            }
        }
        
        # Regional preferences for units
        self.regional_unit_preferences = {
            Language.ENGLISH: {"time": "ms", "frequency": "GHz", "fidelity": "percentage"},
            Language.GERMAN: {"time": "ms", "frequency": "GHz", "fidelity": "decimal"}, 
            Language.JAPANESE: {"time": "ns", "frequency": "MHz", "fidelity": "decimal"},
            Language.CHINESE_SIMPLIFIED: {"time": "ns", "frequency": "GHz", "fidelity": "percentage"}
        }
    
    def format_number(self, value: float, precision: int = 3) -> str:
        """Format number according to locale preferences"""
        
        # Apply decimal and thousand separators
        formatted = f"{value:.{precision}f}"
        
        # Replace decimal separator if needed
        if self.config.decimal_separator != ".":
            formatted = formatted.replace(".", self.config.decimal_separator)
        
        # Add thousand separators for large numbers
        if abs(value) >= 1000:
            parts = formatted.split(self.config.decimal_separator)
            integer_part = parts[0]
            
            # Add thousand separators
            if len(integer_part) > 3:
                separated_integer = ""
                for i, digit in enumerate(reversed(integer_part)):
                    if i > 0 and i % 3 == 0:
                        separated_integer = self.config.thousand_separator + separated_integer
                    separated_integer = digit + separated_integer
                
                if len(parts) > 1:
                    formatted = separated_integer + self.config.decimal_separator + parts[1]
                else:
                    formatted = separated_integer
        
        return formatted
    
    def format_quantum_metric(self, metric_type: str, value: float,
                            unit: Optional[str] = None) -> str:
        """Format quantum metric with appropriate units and localization"""
        
        # Get preferred unit for this metric and language
        if unit is None:
            preferences = self.regional_unit_preferences.get(
                self.config.primary_language, 
                self.regional_unit_preferences[Language.ENGLISH]
            )
            unit = preferences.get(metric_type, "")
        
        # Format the value
        formatted_value = self.format_number(value)
        
        # Add unit if specified
        if unit and metric_type in self.unit_conversions:
            unit_info = self.unit_conversions[metric_type].get(unit, {})
            unit_symbol = unit_info.get("symbol", unit)
            
            if unit_symbol:
                return f"{formatted_value} {unit_symbol}"
        
        return formatted_value
    
    def convert_units(self, value: float, metric_type: str, 
                     from_unit: str, to_unit: str) -> float:
        """Convert between different units for quantum metrics"""
        
        if metric_type not in self.unit_conversions:
            return value
        
        conversions = self.unit_conversions[metric_type]
        
        if from_unit not in conversions or to_unit not in conversions:
            return value
        
        from_factor = conversions[from_unit]["factor"]
        to_factor = conversions[to_unit]["factor"]
        
        # Convert to base unit, then to target unit
        base_value = value * from_factor
        converted_value = base_value / to_factor
        
        return converted_value
    
    def get_localized_unit_name(self, metric_type: str, unit: str) -> str:
        """Get localized name for a unit"""
        
        if metric_type not in self.unit_conversions:
            return unit
        
        unit_info = self.unit_conversions[metric_type].get(unit, {})
        base_name = unit_info.get("name", unit)
        
        # In a full implementation, this would use the translation service
        # For now, return the base name
        return base_name

class CulturalQuantumAdaptation:
    """Adapts quantum concepts and presentations to different cultural contexts"""
    
    def __init__(self, cultural_context: CulturalContext):
        self.cultural_context = cultural_context
        
        # Cultural preferences for explanations
        self.explanation_styles = {
            CulturalContext.WESTERN_ACADEMIC: {
                "approach": "analytical",
                "detail_level": "comprehensive",
                "examples": "mathematical",
                "tone": "objective"
            },
            CulturalContext.EASTERN_ACADEMIC: {
                "approach": "systematic",
                "detail_level": "thorough",
                "examples": "conceptual",
                "tone": "respectful"
            },
            CulturalContext.INDUSTRY_FOCUSED: {
                "approach": "practical",
                "detail_level": "focused",
                "examples": "application_based",
                "tone": "direct"
            },
            CulturalContext.EDUCATIONAL: {
                "approach": "progressive",
                "detail_level": "scaffolded",
                "examples": "intuitive",
                "tone": "encouraging"
            }
        }
    
    def adapt_explanation(self, concept: str, base_explanation: str) -> str:
        """Adapt explanation to cultural context"""
        
        style = self.explanation_styles[self.cultural_context]
        
        # Apply cultural adaptations (simplified implementation)
        if style["approach"] == "practical":
            # Add practical applications
            adapted = f"{base_explanation}\n\nPractical application: This concept is directly used in quantum error correction protocols to improve computational reliability."
        elif style["approach"] == "systematic":
            # Add systematic breakdown
            adapted = f"Overview: {concept}\n\n{base_explanation}\n\nSystematic breakdown:\n1. Theoretical foundation\n2. Mathematical formulation\n3. Experimental validation\n4. Practical implementation"
        elif style["approach"] == "progressive":
            # Add learning progression
            adapted = f"Introduction to {concept}:\n\n{base_explanation}\n\nLearning path:\n• Begin with basic principles\n• Explore mathematical foundations\n• Practice with examples\n• Apply to real problems"
        else:
            # Default analytical approach
            adapted = base_explanation
        
        return adapted
    
    def get_cultural_preferences(self) -> Dict[str, str]:
        """Get cultural preferences for the current context"""
        return self.explanation_styles[self.cultural_context].copy()

class I18nManager:
    """Main internationalization manager"""
    
    def __init__(self, config: LocalizationConfig):
        self.config = config
        self.translation_service = TranslationService()
        self.localized_metrics = LocalizedQuantumMetrics(config)
        self.cultural_adaptation = CulturalQuantumAdaptation(config.cultural_context)
        
        # Current locale information
        self.current_locale = {
            "language": config.primary_language,
            "fallback_languages": config.fallback_languages,
            "cultural_context": config.cultural_context,
            "number_format": config.number_format,
            "measurement_system": config.measurement_system
        }
    
    def t(self, key: str, context: Optional[str] = None, 
         **kwargs) -> str:
        """Translate key (shorthand method)"""
        
        translation = self.translation_service.translate(
            key, 
            self.config.primary_language,
            self.config.fallback_languages,
            context
        )
        
        # Apply string formatting if kwargs provided
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError):
                # If formatting fails, return unformatted translation
                pass
        
        return translation
    
    def translate_quantum_concept(self, concept: str, 
                                explanation: str) -> Dict[str, str]:
        """Translate and culturally adapt quantum concept"""
        
        translated_concept = self.t(concept)
        translated_explanation = self.t(f"{concept}_explanation", 
                                      context=explanation)
        
        # If no specific explanation translation, use provided explanation
        if translated_explanation.startswith("["):
            translated_explanation = explanation
        
        # Apply cultural adaptation
        adapted_explanation = self.cultural_adaptation.adapt_explanation(
            translated_concept, translated_explanation
        )
        
        return {
            "concept": translated_concept,
            "explanation": adapted_explanation,
            "cultural_context": self.config.cultural_context.value,
            "language": self.config.primary_language.value
        }
    
    def format_quantum_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format quantum computation result for current locale"""
        
        localized_result = result.copy()
        
        # Localize numeric values
        for key, value in result.items():
            if isinstance(value, float):
                if "fidelity" in key.lower():
                    localized_result[key] = self.localized_metrics.format_quantum_metric(
                        "fidelity", value
                    )
                elif "time" in key.lower():
                    localized_result[key] = self.localized_metrics.format_quantum_metric(
                        "time", value
                    )
                elif "frequency" in key.lower():
                    localized_result[key] = self.localized_metrics.format_quantum_metric(
                        "frequency", value
                    )
                else:
                    localized_result[key] = self.localized_metrics.format_number(value)
        
        # Translate string keys and values
        translated_result = {}
        for key, value in localized_result.items():
            translated_key = self.t(key)
            if isinstance(value, str) and not any(c.isdigit() for c in value):
                translated_value = self.t(value)
                translated_result[translated_key] = translated_value
            else:
                translated_result[translated_key] = value
        
        return translated_result
    
    def get_localization_info(self) -> Dict[str, Any]:
        """Get current localization information"""
        
        coverage = self.translation_service.get_translation_coverage(
            self.config.primary_language
        )
        
        return {
            "locale": self.current_locale,
            "translation_coverage": coverage,
            "cultural_preferences": self.cultural_adaptation.get_cultural_preferences(),
            "supported_languages": [lang.value for lang in Language],
            "missing_translations": len(self.translation_service.missing_translations)
        }
    
    def switch_language(self, new_language: Language,
                       fallback_languages: Optional[List[Language]] = None) -> bool:
        """Switch to a different language"""
        
        try:
            self.config.primary_language = new_language
            if fallback_languages:
                self.config.fallback_languages = fallback_languages
            
            # Update current locale
            self.current_locale["language"] = new_language
            self.current_locale["fallback_languages"] = fallback_languages or []
            
            # Clear translation cache
            self.translation_service.translation_cache.clear()
            
            return True
            
        except Exception as e:
            print(f"Failed to switch language: {e}")
            return False

# Factory functions

def create_i18n_manager(language: Language = Language.ENGLISH,
                       cultural_context: CulturalContext = CulturalContext.WESTERN_ACADEMIC) -> I18nManager:
    """Create an internationalization manager"""
    
    config = LocalizationConfig(
        primary_language=language,
        fallback_languages=[Language.ENGLISH],
        cultural_context=cultural_context,
        number_format="1,234.56",
        decimal_separator=".",
        thousand_separator=",",
        date_format="YYYY-MM-DD",
        time_format="24h",
        currency_symbol="$",
        measurement_system="metric",
        scientific_notation="western"
    )
    
    return I18nManager(config)

def create_translation_service() -> TranslationService:
    """Create a translation service"""
    return TranslationService()

def create_localized_metrics(config: LocalizationConfig) -> LocalizedQuantumMetrics:
    """Create localized quantum metrics handler"""
    return LocalizedQuantumMetrics(config)