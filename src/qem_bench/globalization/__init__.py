"""
Global-First Quantum Error Mitigation Framework

Provides multi-region deployment, internationalization, and compliance features:
- Multi-region cloud deployment and scaling
- Internationalization (i18n) for global accessibility
- Data privacy and regulatory compliance (GDPR, CCPA, PDPA)
- Cross-cultural quantum research collaboration
- Global quantum hardware resource management
"""

# Multi-region deployment
from .deployment import (
    MultiRegionDeployment, RegionConfig, CloudProvider,
    GlobalLoadBalancer, RegionalDataReplication,
    create_global_deployment
)

# Internationalization and localization
from .i18n import (
    I18nManager, LocalizationConfig, TranslationService,
    LocalizedQuantumMetrics, CulturalQuantumAdaptation,
    create_i18n_manager
)

# Compliance and privacy
from .compliance import (
    ComplianceFramework, PrivacyManager, DataGovernance,
    RegulationCompliance, AuditTrail,
    create_compliance_framework
)

# Global collaboration
from .collaboration import (
    GlobalCollaborationPlatform, CrossCulturalResearch,
    InternationalQuantumConsortium, ResearchExchange,
    create_global_collaboration
)

__all__ = [
    # Multi-region deployment
    "MultiRegionDeployment",
    "RegionConfig",
    "CloudProvider", 
    "GlobalLoadBalancer",
    "RegionalDataReplication",
    "create_global_deployment",
    
    # Internationalization
    "I18nManager",
    "LocalizationConfig",
    "TranslationService",
    "LocalizedQuantumMetrics",
    "CulturalQuantumAdaptation",
    "create_i18n_manager",
    
    # Compliance and privacy
    "ComplianceFramework",
    "PrivacyManager",
    "DataGovernance", 
    "RegulationCompliance",
    "AuditTrail",
    "create_compliance_framework",
    
    # Global collaboration
    "GlobalCollaborationPlatform",
    "CrossCulturalResearch",
    "InternationalQuantumConsortium",
    "ResearchExchange",
    "create_global_collaboration"
]