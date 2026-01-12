"""Intermediate Knowledge Representation (IKR) for improved SMT generation.

This module provides a structured intermediate representation between
natural language questions and SMT2 code. The IKR layer:

1. Separates semantic understanding from SMT2 syntax
2. Makes implicit world knowledge explicit
3. Enables deterministic compilation to SMT2
4. Supports two-stage prompting for background knowledge

Flow:
    NL Question -> LLM -> IKR (Stage 1: explicit facts)
                      -> IKR (Stage 2: background knowledge)
                      -> Compiler -> SMT2 -> Z3
"""

from z3adapter.ikr.schema import (
    IKR,
    Entity,
    Fact,
    Query,
    Relation,
    Rule,
    Type,
)
from z3adapter.ikr.compiler import IKRCompiler

__all__ = [
    "IKR",
    "IKRCompiler",
    "Type",
    "Entity",
    "Relation",
    "Fact",
    "Rule",
    "Query",
]
