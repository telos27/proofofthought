"""Backend implementations for Z3 DSL execution."""

from z3adapter.backends.abstract import Backend, VerificationResult
from z3adapter.backends.json_backend import JSONBackend
from z3adapter.backends.smt2_backend import SMT2Backend

# Import IKR and Souffle backends with graceful degradation
# (they may not be available if dependencies are missing)
try:
    from z3adapter.backends.ikr_backend import IKRBackend
except ImportError:
    IKRBackend = None  # type: ignore

try:
    from z3adapter.backends.souffle_backend import SouffleBackend
except ImportError:
    SouffleBackend = None  # type: ignore

__all__ = [
    "Backend",
    "VerificationResult",
    "JSONBackend",
    "SMT2Backend",
    "IKRBackend",
    "SouffleBackend",
]
