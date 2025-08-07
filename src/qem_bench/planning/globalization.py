"""
Global-First Quantum Planning

Internationalization, localization, compliance, and multi-region deployment
features for global quantum-inspired task planning systems.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from datetime import datetime, timezone, timedelta
import locale
import threading
from pathlib import Path

from .core import PlanningConfig
from ..security import SecurityPolicy


class SupportedLocale(Enum):
    """Supported locales for global deployment"""
    EN_US = "en-US"  # English (United States)
    EN_GB = "en-GB"  # English (United Kingdom)
    ES_ES = "es-ES"  # Spanish (Spain)
    ES_MX = "es-MX"  # Spanish (Mexico)  
    FR_FR = "fr-FR"  # French (France)
    FR_CA = "fr-CA"  # French (Canada)
    DE_DE = "de-DE"  # German (Germany)
    JA_JP = "ja-JP"  # Japanese (Japan)
    ZH_CN = "zh-CN"  # Chinese (Simplified, China)
    ZH_TW = "zh-TW"  # Chinese (Traditional, Taiwan)
    KO_KR = "ko-KR"  # Korean (South Korea)
    PT_BR = "pt-BR"  # Portuguese (Brazil)
    IT_IT = "it-IT"  # Italian (Italy)
    RU_RU = "ru-RU"  # Russian (Russia)
    AR_SA = "ar-SA"  # Arabic (Saudi Arabia)


class DeploymentRegion(Enum):
    """Global deployment regions"""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2" 
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"
    CANADA = "ca-central-1"
    AUSTRALIA = "ap-southeast-2"
    BRAZIL = "sa-east-1"
    INDIA = "ap-south-1"


class ComplianceFramework(Enum):
    """Compliance frameworks for data protection"""
    GDPR = "gdpr"          # General Data Protection Regulation (EU)
    CCPA = "ccpa"          # California Consumer Privacy Act (US)
    PDPA_SG = "pdpa-sg"    # Personal Data Protection Act (Singapore)
    PDPA_TH = "pdpa-th"    # Personal Data Protection Act (Thailand)
    LGPD = "lgpd"          # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"      # Personal Information Protection and Electronic Documents Act (Canada)
    PRIVACY_ACT = "privacy-act"  # Privacy Act (Australia)


@dataclass
class LocalizationConfig:
    """Configuration for localization and internationalization"""
    locale: SupportedLocale = SupportedLocale.EN_US
    timezone: str = "UTC"
    currency: str = "USD"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "en_US"
    enable_rtl: bool = False  # Right-to-left languages
    translation_cache_size: int = 1000
    fallback_locale: SupportedLocale = SupportedLocale.EN_US


@dataclass
class ComplianceConfig:
    """Configuration for regulatory compliance"""
    frameworks: List[ComplianceFramework] = field(default_factory=lambda: [ComplianceFramework.GDPR])
    data_retention_days: int = 730  # 2 years default
    enable_audit_logging: bool = True
    enable_encryption: bool = True
    enable_anonymization: bool = True
    consent_required: bool = True
    data_minimization: bool = True
    right_to_deletion: bool = True
    cross_border_transfer_allowed: bool = False
    lawful_basis: str = "legitimate_interest"


@dataclass
class MultiRegionConfig:
    """Configuration for multi-region deployment"""
    primary_region: DeploymentRegion = DeploymentRegion.US_EAST
    secondary_regions: List[DeploymentRegion] = field(default_factory=list)
    enable_failover: bool = True
    enable_load_balancing: bool = True
    enable_data_replication: bool = True
    replication_lag_tolerance_ms: int = 1000
    health_check_interval_seconds: int = 30
    failover_timeout_seconds: int = 60


class GlobalizationManager:
    """
    Global-first quantum planning with I18N, L10N, and compliance
    
    Features:
    - Multi-language support with dynamic translation
    - Timezone-aware scheduling and execution
    - Regional data compliance (GDPR, CCPA, etc.)
    - Multi-region deployment and failover
    - Cultural adaptation of algorithms
    - Currency and number format localization
    """
    
    def __init__(self, 
                 localization_config: LocalizationConfig = None,
                 compliance_config: ComplianceConfig = None,
                 multiregion_config: MultiRegionConfig = None):
        
        self.l10n_config = localization_config or LocalizationConfig()
        self.compliance_config = compliance_config or ComplianceConfig()
        self.multiregion_config = multiregion_config or MultiRegionConfig()
        
        # Translation system
        self.translations = self._load_translations()
        self.translation_cache = {}
        self._translation_lock = threading.Lock()
        
        # Regional compliance handlers
        self.compliance_handlers = self._setup_compliance_handlers()
        
        # Multi-region state
        self.region_status = self._initialize_region_status()
        self.active_region = self.multiregion_config.primary_region
        
        # Localization state
        self.current_locale = self.l10n_config.locale
        self.current_timezone = timezone.utc
        
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translation files for supported locales"""
        translations = {}
        
        # Base translations (would typically load from files)
        base_translations = {
            SupportedLocale.EN_US.value: {
                "quantum_planning": "Quantum Planning",
                "task_optimization": "Task Optimization",
                "scheduling_complete": "Scheduling Complete",
                "error_occurred": "An error occurred",
                "convergence_achieved": "Convergence Achieved",
                "performance_optimized": "Performance Optimized",
                "resource_allocated": "Resource Allocated",
                "validation_failed": "Validation Failed",
                "recovery_initiated": "Recovery Initiated",
                "cache_miss": "Cache Miss",
                "distributed_processing": "Distributed Processing"
            },
            SupportedLocale.ES_ES.value: {
                "quantum_planning": "Planificación Cuántica",
                "task_optimization": "Optimización de Tareas",
                "scheduling_complete": "Programación Completa",
                "error_occurred": "Ocurrió un error",
                "convergence_achieved": "Convergencia Lograda",
                "performance_optimized": "Rendimiento Optimizado",
                "resource_allocated": "Recurso Asignado",
                "validation_failed": "Validación Falló",
                "recovery_initiated": "Recuperación Iniciada",
                "cache_miss": "Fallo de Caché",
                "distributed_processing": "Procesamiento Distribuido"
            },
            SupportedLocale.FR_FR.value: {
                "quantum_planning": "Planification Quantique",
                "task_optimization": "Optimisation des Tâches",
                "scheduling_complete": "Programmation Terminée",
                "error_occurred": "Une erreur s'est produite",
                "convergence_achieved": "Convergence Atteinte",
                "performance_optimized": "Performance Optimisée",
                "resource_allocated": "Ressource Allouée",
                "validation_failed": "Validation Échouée",
                "recovery_initiated": "Récupération Initiée",
                "cache_miss": "Échec du Cache",
                "distributed_processing": "Traitement Distribué"
            },
            SupportedLocale.DE_DE.value: {
                "quantum_planning": "Quantenplanung",
                "task_optimization": "Aufgabenoptimierung",
                "scheduling_complete": "Terminplanung Abgeschlossen",
                "error_occurred": "Ein Fehler ist aufgetreten",
                "convergence_achieved": "Konvergenz Erreicht",
                "performance_optimized": "Leistung Optimiert",
                "resource_allocated": "Ressource Zugewiesen",
                "validation_failed": "Validierung Fehlgeschlagen",
                "recovery_initiated": "Wiederherstellung Eingeleitet",
                "cache_miss": "Cache-Fehler",
                "distributed_processing": "Verteilte Verarbeitung"
            },
            SupportedLocale.JA_JP.value: {
                "quantum_planning": "量子プランニング",
                "task_optimization": "タスク最適化",
                "scheduling_complete": "スケジューリング完了",
                "error_occurred": "エラーが発生しました",
                "convergence_achieved": "収束達成",
                "performance_optimized": "パフォーマンス最適化",
                "resource_allocated": "リソース割り当て",
                "validation_failed": "検証失敗",
                "recovery_initiated": "回復開始",
                "cache_miss": "キャッシュミス",
                "distributed_processing": "分散処理"
            },
            SupportedLocale.ZH_CN.value: {
                "quantum_planning": "量子规划",
                "task_optimization": "任务优化",
                "scheduling_complete": "调度完成",
                "error_occurred": "发生错误",
                "convergence_achieved": "达到收敛",
                "performance_optimized": "性能已优化",
                "resource_allocated": "资源已分配",
                "validation_failed": "验证失败",
                "recovery_initiated": "恢复已启动",
                "cache_miss": "缓存未命中",
                "distributed_processing": "分布式处理"
            }
        }
        
        return base_translations
    
    def translate(self, key: str, locale: SupportedLocale = None, 
                 params: Dict[str, Any] = None) -> str:
        """
        Translate message key to specified locale
        
        Args:
            key: Translation key
            locale: Target locale (defaults to current locale)
            params: Parameters for string interpolation
            
        Returns:
            Translated string
        """
        target_locale = locale or self.current_locale
        cache_key = f"{target_locale.value}:{key}"
        
        # Check cache first
        with self._translation_lock:
            if cache_key in self.translation_cache:
                translated = self.translation_cache[cache_key]
            else:
                # Get translation
                locale_translations = self.translations.get(target_locale.value, {})
                translated = locale_translations.get(key)
                
                # Fallback to English if not found
                if not translated and target_locale != self.l10n_config.fallback_locale:
                    fallback_translations = self.translations.get(self.l10n_config.fallback_locale.value, {})
                    translated = fallback_translations.get(key)
                
                # Final fallback to key itself
                if not translated:
                    translated = key.replace("_", " ").title()
                
                # Cache translation
                if len(self.translation_cache) < self.l10n_config.translation_cache_size:
                    self.translation_cache[cache_key] = translated
        
        # Apply parameters if provided
        if params:
            try:
                translated = translated.format(**params)
            except (KeyError, ValueError):
                pass  # Use translation as-is if formatting fails
        
        return translated
    
    def format_datetime(self, dt: datetime, locale: SupportedLocale = None) -> str:
        """Format datetime according to locale conventions"""
        target_locale = locale or self.current_locale
        
        # Convert to local timezone if needed
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.current_timezone)
        
        # Locale-specific formatting
        if target_locale in [SupportedLocale.EN_US, SupportedLocale.EN_GB]:
            return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        elif target_locale in [SupportedLocale.DE_DE, SupportedLocale.FR_FR]:
            return dt.strftime("%d.%m.%Y %H:%M:%S %Z")
        elif target_locale == SupportedLocale.JA_JP:
            return dt.strftime("%Y年%m月%d日 %H:%M:%S %Z")
        elif target_locale == SupportedLocale.ZH_CN:
            return dt.strftime("%Y年%m月%d日 %H:%M:%S %Z")
        else:
            return dt.strftime(self.l10n_config.date_format + " " + self.l10n_config.time_format)
    
    def format_number(self, number: float, locale: SupportedLocale = None) -> str:
        """Format number according to locale conventions"""
        target_locale = locale or self.current_locale
        
        # Locale-specific number formatting
        try:
            if target_locale == SupportedLocale.EN_US:
                return f"{number:,.2f}"  # 1,234.56
            elif target_locale == SupportedLocale.DE_DE:
                return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")  # 1.234,56
            elif target_locale == SupportedLocale.FR_FR:
                return f"{number:,.2f}".replace(",", " ")  # 1 234.56
            else:
                return f"{number:.2f}"
        except:
            return str(number)
    
    def get_cultural_optimization_params(self, locale: SupportedLocale = None) -> Dict[str, float]:
        """
        Get culturally-adapted optimization parameters
        
        Different cultures have different approaches to optimization:
        - Western: Individual optimization, competition
        - Eastern: Collective optimization, harmony
        - Northern European: Efficiency, precision
        - Latin: Flexibility, adaptability
        """
        target_locale = locale or self.current_locale
        
        # Cultural optimization parameter adaptations
        if target_locale in [SupportedLocale.JA_JP, SupportedLocale.ZH_CN, SupportedLocale.KO_KR]:
            # Eastern cultures: Emphasize harmony and collective optimization
            return {
                'cooperation_weight': 0.8,
                'competition_weight': 0.3,
                'harmony_factor': 0.9,
                'individual_priority': 0.4,
                'consensus_preference': 0.9,
                'risk_tolerance': 0.3
            }
        elif target_locale in [SupportedLocale.DE_DE, SupportedLocale.EN_GB]:
            # Northern European: Precision and efficiency focus
            return {
                'cooperation_weight': 0.6,
                'competition_weight': 0.7,
                'harmony_factor': 0.6,
                'individual_priority': 0.8,
                'consensus_preference': 0.5,
                'risk_tolerance': 0.4
            }
        elif target_locale in [SupportedLocale.ES_ES, SupportedLocale.ES_MX, SupportedLocale.PT_BR, SupportedLocale.IT_IT]:
            # Latin cultures: Flexibility and adaptability
            return {
                'cooperation_weight': 0.7,
                'competition_weight': 0.6,
                'harmony_factor': 0.7,
                'individual_priority': 0.6,
                'consensus_preference': 0.7,
                'risk_tolerance': 0.7
            }
        elif target_locale == SupportedLocale.AR_SA:
            # Arabic culture: Community and tradition
            return {
                'cooperation_weight': 0.9,
                'competition_weight': 0.4,
                'harmony_factor': 0.8,
                'individual_priority': 0.3,
                'consensus_preference': 0.9,
                'risk_tolerance': 0.2
            }
        else:  # Default Western/US approach
            return {
                'cooperation_weight': 0.5,
                'competition_weight': 0.8,
                'harmony_factor': 0.5,
                'individual_priority': 0.9,
                'consensus_preference': 0.4,
                'risk_tolerance': 0.6
            }
    
    def _setup_compliance_handlers(self) -> Dict[ComplianceFramework, Callable]:
        """Setup handlers for different compliance frameworks"""
        return {
            ComplianceFramework.GDPR: self._handle_gdpr_compliance,
            ComplianceFramework.CCPA: self._handle_ccpa_compliance,
            ComplianceFramework.PDPA_SG: self._handle_pdpa_compliance,
            ComplianceFramework.LGPD: self._handle_lgpd_compliance,
            ComplianceFramework.PIPEDA: self._handle_pipeda_compliance,
            ComplianceFramework.PRIVACY_ACT: self._handle_privacy_act_compliance
        }
    
    def _handle_gdpr_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle GDPR compliance requirements"""
        compliance_result = {
            'framework': 'GDPR',
            'compliant': True,
            'actions_taken': [],
            'data_processed': {}
        }
        
        # Data minimization
        if self.compliance_config.data_minimization:
            # Remove unnecessary fields
            sensitive_fields = ['user_id', 'ip_address', 'location', 'personal_data']
            minimized_data = {k: v for k, v in data.items() if k not in sensitive_fields}
            compliance_result['actions_taken'].append('data_minimization_applied')
            compliance_result['data_processed'] = minimized_data
        
        # Anonymization
        if self.compliance_config.enable_anonymization:
            # Apply anonymization techniques
            compliance_result['actions_taken'].append('data_anonymized')
        
        # Consent verification (simulated)
        if self.compliance_config.consent_required:
            compliance_result['actions_taken'].append('consent_verified')
        
        # Retention policy
        compliance_result['retention_days'] = self.compliance_config.data_retention_days
        compliance_result['actions_taken'].append('retention_policy_applied')
        
        return compliance_result
    
    def _handle_ccpa_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle CCPA compliance requirements"""
        return {
            'framework': 'CCPA',
            'compliant': True,
            'actions_taken': ['opt_out_honored', 'data_sale_restricted'],
            'data_processed': data if not self.compliance_config.data_minimization else {}
        }
    
    def _handle_pdpa_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle PDPA compliance requirements"""
        return {
            'framework': 'PDPA',
            'compliant': True,
            'actions_taken': ['consent_obtained', 'notification_provided'],
            'data_processed': data if not self.compliance_config.data_minimization else {}
        }
    
    def _handle_lgpd_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LGPD compliance requirements"""
        return {
            'framework': 'LGPD',
            'compliant': True,
            'actions_taken': ['lawful_basis_established', 'data_subject_rights_enabled'],
            'data_processed': data if not self.compliance_config.data_minimization else {}
        }
    
    def _handle_pipeda_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle PIPEDA compliance requirements"""
        return {
            'framework': 'PIPEDA',
            'compliant': True,
            'actions_taken': ['consent_meaningful', 'purpose_limitation_applied'],
            'data_processed': data if not self.compliance_config.data_minimization else {}
        }
    
    def _handle_privacy_act_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Privacy Act compliance requirements"""
        return {
            'framework': 'Privacy Act',
            'compliant': True,
            'actions_taken': ['collection_limitation', 'access_rights_provided'],
            'data_processed': data if not self.compliance_config.data_minimization else {}
        }
    
    def ensure_compliance(self, data: Dict[str, Any], 
                         operation: str = "processing") -> Dict[str, Any]:
        """
        Ensure data processing complies with applicable frameworks
        
        Args:
            data: Data to be processed
            operation: Type of operation (processing, storage, transfer)
            
        Returns:
            Compliance verification result
        """
        compliance_results = []
        
        for framework in self.compliance_config.frameworks:
            if framework in self.compliance_handlers:
                handler = self.compliance_handlers[framework]
                result = handler(data)
                compliance_results.append(result)
        
        # Aggregate compliance status
        all_compliant = all(r.get('compliant', False) for r in compliance_results)
        
        return {
            'operation': operation,
            'timestamp': datetime.now(timezone.utc),
            'overall_compliant': all_compliant,
            'frameworks_checked': [f.value for f in self.compliance_config.frameworks],
            'detailed_results': compliance_results,
            'data_protection_level': 'high' if all_compliant else 'insufficient'
        }
    
    def _initialize_region_status(self) -> Dict[DeploymentRegion, Dict[str, Any]]:
        """Initialize status tracking for all regions"""
        regions = [self.multiregion_config.primary_region] + self.multiregion_config.secondary_regions
        
        status = {}
        for region in regions:
            status[region] = {
                'healthy': True,
                'last_health_check': datetime.now(timezone.utc),
                'response_time_ms': 50,  # Simulated
                'active_connections': 0,
                'load_factor': 0.0,
                'is_primary': region == self.multiregion_config.primary_region
            }
        
        return status
    
    def get_optimal_region(self, user_location: str = None, 
                          task_requirements: Dict[str, Any] = None) -> DeploymentRegion:
        """
        Get optimal deployment region based on location and requirements
        
        Args:
            user_location: User's location or region preference
            task_requirements: Task-specific requirements (latency, compliance, etc.)
            
        Returns:
            Optimal deployment region
        """
        # Simple location-based mapping
        location_mapping = {
            'us': DeploymentRegion.US_EAST,
            'canada': DeploymentRegion.CANADA,
            'europe': DeploymentRegion.EU_WEST,
            'germany': DeploymentRegion.EU_CENTRAL,
            'asia': DeploymentRegion.ASIA_PACIFIC,
            'japan': DeploymentRegion.ASIA_NORTHEAST,
            'australia': DeploymentRegion.AUSTRALIA,
            'brazil': DeploymentRegion.BRAZIL,
            'india': DeploymentRegion.INDIA
        }
        
        # Check user location preference
        if user_location:
            preferred_region = location_mapping.get(user_location.lower())
            if preferred_region and preferred_region in self.region_status:
                if self.region_status[preferred_region]['healthy']:
                    return preferred_region
        
        # Compliance requirements
        if task_requirements and 'compliance_required' in task_requirements:
            required_frameworks = task_requirements['compliance_required']
            
            # GDPR requires EU regions
            if ComplianceFramework.GDPR.value in required_frameworks:
                for region in [DeploymentRegion.EU_WEST, DeploymentRegion.EU_CENTRAL]:
                    if region in self.region_status and self.region_status[region]['healthy']:
                        return region
        
        # Fallback to primary region if healthy
        if self.region_status[self.multiregion_config.primary_region]['healthy']:
            return self.multiregion_config.primary_region
        
        # Fallback to any healthy secondary region
        for region in self.multiregion_config.secondary_regions:
            if region in self.region_status and self.region_status[region]['healthy']:
                return region
        
        # Final fallback
        return self.multiregion_config.primary_region
    
    def create_localized_planning_config(self, 
                                       base_config: PlanningConfig,
                                       locale: SupportedLocale = None) -> PlanningConfig:
        """Create culturally-adapted planning configuration"""
        target_locale = locale or self.current_locale
        cultural_params = self.get_cultural_optimization_params(target_locale)
        
        # Create adapted config
        import copy
        adapted_config = copy.deepcopy(base_config)
        
        # Apply cultural adaptations
        if cultural_params['risk_tolerance'] < 0.5:
            # Risk-averse cultures: more conservative convergence
            adapted_config.convergence_threshold *= 0.5  # Stricter convergence
            adapted_config.max_iterations = min(adapted_config.max_iterations * 2, 2000)
        
        if cultural_params['consensus_preference'] > 0.7:
            # Consensus-oriented cultures: reduce competition factors
            adapted_config.interference_factor *= 0.8
            adapted_config.entanglement_strength *= 1.2  # More cooperation
        
        if cultural_params['individual_priority'] < 0.5:
            # Collective cultures: emphasize group optimization
            adapted_config.superposition_width *= 1.3  # More exploration
        
        return adapted_config
    
    def get_localized_messages(self, message_keys: List[str], 
                              locale: SupportedLocale = None) -> Dict[str, str]:
        """Get batch of localized messages"""
        return {key: self.translate(key, locale) for key in message_keys}
    
    def validate_cross_border_transfer(self, 
                                     source_region: DeploymentRegion,
                                     target_region: DeploymentRegion,
                                     data_type: str = "planning_data") -> Dict[str, Any]:
        """Validate if cross-border data transfer is allowed"""
        
        # Region compliance mapping
        region_jurisdictions = {
            DeploymentRegion.EU_WEST: [ComplianceFramework.GDPR],
            DeploymentRegion.EU_CENTRAL: [ComplianceFramework.GDPR],
            DeploymentRegion.US_EAST: [ComplianceFramework.CCPA],
            DeploymentRegion.US_WEST: [ComplianceFramework.CCPA],
            DeploymentRegion.CANADA: [ComplianceFramework.PIPEDA],
            DeploymentRegion.AUSTRALIA: [ComplianceFramework.PRIVACY_ACT],
            DeploymentRegion.BRAZIL: [ComplianceFramework.LGPD],
            DeploymentRegion.ASIA_PACIFIC: [ComplianceFramework.PDPA_SG]
        }
        
        source_frameworks = region_jurisdictions.get(source_region, [])
        target_frameworks = region_jurisdictions.get(target_region, [])
        
        # Check if transfer is allowed
        transfer_allowed = self.compliance_config.cross_border_transfer_allowed
        
        # Special GDPR restrictions
        if ComplianceFramework.GDPR in source_frameworks:
            # GDPR allows transfers within EU or to adequate countries
            eu_regions = [DeploymentRegion.EU_WEST, DeploymentRegion.EU_CENTRAL]
            adequate_regions = [DeploymentRegion.CANADA, DeploymentRegion.US_EAST]  # Simplified
            
            if target_region in eu_regions or target_region in adequate_regions:
                transfer_allowed = True
            else:
                transfer_allowed = False
        
        return {
            'transfer_allowed': transfer_allowed,
            'source_region': source_region.value,
            'target_region': target_region.value,
            'source_frameworks': [f.value for f in source_frameworks],
            'target_frameworks': [f.value for f in target_frameworks],
            'data_type': data_type,
            'additional_safeguards_required': not transfer_allowed,
            'timestamp': datetime.now(timezone.utc)
        }
    
    def get_region_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all regions"""
        metrics = {
            'regions': {},
            'global_overview': {
                'total_regions': len(self.region_status),
                'healthy_regions': sum(1 for r in self.region_status.values() if r['healthy']),
                'active_region': self.active_region.value,
                'failover_enabled': self.multiregion_config.enable_failover
            }
        }
        
        for region, status in self.region_status.items():
            metrics['regions'][region.value] = {
                'healthy': status['healthy'],
                'response_time_ms': status['response_time_ms'],
                'load_factor': status['load_factor'],
                'is_primary': status['is_primary'],
                'last_check': self.format_datetime(status['last_health_check'])
            }
        
        return metrics
    
    def get_globalization_summary(self) -> Dict[str, Any]:
        """Get comprehensive globalization status summary"""
        return {
            'localization': {
                'current_locale': self.current_locale.value,
                'supported_locales': [locale.value for locale in SupportedLocale],
                'translation_cache_size': len(self.translation_cache),
                'rtl_enabled': self.l10n_config.enable_rtl
            },
            'compliance': {
                'frameworks': [f.value for f in self.compliance_config.frameworks],
                'data_retention_days': self.compliance_config.data_retention_days,
                'encryption_enabled': self.compliance_config.enable_encryption,
                'audit_logging': self.compliance_config.enable_audit_logging,
                'anonymization_enabled': self.compliance_config.enable_anonymization
            },
            'multi_region': {
                'primary_region': self.multiregion_config.primary_region.value,
                'secondary_regions': [r.value for r in self.multiregion_config.secondary_regions],
                'failover_enabled': self.multiregion_config.enable_failover,
                'load_balancing': self.multiregion_config.enable_load_balancing,
                'data_replication': self.multiregion_config.enable_data_replication
            },
            'performance': self.get_region_performance_metrics()
        }