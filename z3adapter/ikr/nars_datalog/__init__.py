"""NARS-Datalog: Python Datalog engine with NARS truth value propagation.

This module provides a native Python Datalog engine that integrates
NARS-style truth values throughout inference. Unlike approaches that
bolt uncertainty onto symbolic solvers, this engine treats truth values
as first-class citizens that propagate through rule applications.

Key features:
- Semi-naive evaluation for efficient fixpoint computation
- NARS truth values on all facts (frequency, confidence)
- Evidence combination via NARS revision
- Stratified negation support

Example usage:
    from z3adapter.ikr.nars_datalog import NARSDatalogEngine, from_ikr
    from z3adapter.ikr.schema import IKR

    # Load IKR
    ikr = IKR.model_validate(ikr_data)

    # Create and run engine
    engine = from_ikr(ikr)
    result = engine.query(ikr.query)

    # Check results
    if result.found:
        print(f"Answer: True")
        print(f"Truth value: f={result.truth_value.frequency:.3f}, "
              f"c={result.truth_value.confidence:.3f}")
    else:
        print("Answer: False (not derivable)")
"""

from .truth_functions import (
    conjunction,
    deduction,
    negation,
    revise,
    revise_multiple,
    DEFAULT_RULE_TRUTH,
)
from .fact_store import (
    GroundAtom,
    StoredFact,
    FactStore,
)
from .unification import (
    Bindings,
    RuleAtom,
    is_variable,
    unify_atom_with_fact,
    find_all_bindings,
)
from .rule import (
    InternalRule,
    compile_rules,
)
from .engine import (
    NARSDatalogEngine,
    InferenceResult,
    from_ikr,
)
from .kb_loader import (
    KBLoader,
    KB_DIR,
)

__all__ = [
    # Truth functions
    "conjunction",
    "deduction",
    "negation",
    "revise",
    "revise_multiple",
    "DEFAULT_RULE_TRUTH",
    # Fact storage
    "GroundAtom",
    "StoredFact",
    "FactStore",
    # Unification
    "Bindings",
    "RuleAtom",
    "is_variable",
    "unify_atom_with_fact",
    "find_all_bindings",
    # Rules
    "InternalRule",
    "compile_rules",
    # Engine
    "NARSDatalogEngine",
    "InferenceResult",
    "from_ikr",
    # Knowledge Base
    "KBLoader",
    "KB_DIR",
]
