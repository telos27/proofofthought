"""Intermediate Knowledge Representation (IKR) for improved SMT generation.

This module provides a structured intermediate representation between
natural language questions and formal logic code. The IKR layer:

1. Separates semantic understanding from target language syntax
2. Makes implicit world knowledge explicit
3. Enables deterministic compilation to SMT2 or Datalog
4. Supports two-stage prompting for background knowledge

Flow (SMT2):
    NL Question -> LLM -> IKR (Stage 1: explicit facts)
                      -> IKR (Stage 2: background knowledge)
                      -> IKRCompiler -> SMT2 -> Z3

Flow (Souffle/Datalog):
    NL Question -> LLM -> IKR (Stage 1: explicit facts)
                      -> IKR (Stage 2: background knowledge)
                      -> IKRSouffleCompiler -> .dl + .facts -> Souffle
"""

from z3adapter.ikr.schema import (
    IKR,
    Entity,
    Fact,
    Query,
    Relation,
    Rule,
    Type,
    TruthValue,
)
from z3adapter.ikr.compiler import IKRCompiler
from z3adapter.ikr.souffle_compiler import IKRSouffleCompiler, SouffleProgram
from z3adapter.ikr.fuzzy_nars import (
    VerificationTriple,
    UnificationResult,
    VerificationVerdict,
    VerificationResult,
    PREDICATE_OPPOSITES,
    get_predicate_polarity,
    lexical_similarity,
    jaccard_word_similarity,
    combined_lexical_similarity,
    make_embedding_similarity,
    make_hybrid_similarity,
    fuzzy_nars_unify,
    revise,
    revise_multiple,
    verify_triple,
    verify_answer,
)

__all__ = [
    # Schema
    "IKR",
    "Type",
    "Entity",
    "Relation",
    "Fact",
    "Rule",
    "Query",
    "TruthValue",
    # Compilers
    "IKRCompiler",
    "IKRSouffleCompiler",
    "SouffleProgram",
    # Fuzzy-NARS verification
    "VerificationTriple",
    "UnificationResult",
    "VerificationVerdict",
    "VerificationResult",
    "PREDICATE_OPPOSITES",
    "get_predicate_polarity",
    "lexical_similarity",
    "jaccard_word_similarity",
    "combined_lexical_similarity",
    "make_embedding_similarity",
    "make_hybrid_similarity",
    "fuzzy_nars_unify",
    "revise",
    "revise_multiple",
    "verify_triple",
    "verify_answer",
]
