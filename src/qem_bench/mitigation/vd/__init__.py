"""Virtual Distillation (VD) implementation."""

from .core import VirtualDistillation, VDConfig, virtual_distillation
from .verification import (
    VerificationStrategy,
    BellStateVerification,
    GHZStateVerification,
    ProductStateVerification,
    RandomStateVerification,
    create_verification_strategy,
    estimate_verification_overhead
)
from .result import VDResult, VDBatchResult

__all__ = [
    "VirtualDistillation",
    "VDConfig",
    "virtual_distillation",
    "VerificationStrategy",
    "BellStateVerification",
    "GHZStateVerification", 
    "ProductStateVerification",
    "RandomStateVerification",
    "create_verification_strategy",
    "estimate_verification_overhead",
    "VDResult",
    "VDBatchResult"
]