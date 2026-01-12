"""Compiler for IKR to SMT2 translation.

This module provides deterministic compilation from the Intermediate
Knowledge Representation (IKR) to SMT-LIB 2.0 format. The compiler
ensures syntactically correct SMT2 output, eliminating the syntax
errors that commonly occur with direct LLM generation.

The compilation process:
1. Declare sorts from types
2. Declare functions from relations
3. Declare constants from entities
4. Assert facts as ground truths
5. Assert rules as quantified formulas
6. Assert query for satisfiability check
7. Add (check-sat) command
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from z3adapter.ikr.schema import (
        Entity,
        Fact,
        IKR,
        Query,
        Relation,
        Rule,
        RuleCondition,
        Type,
    )

logger = logging.getLogger(__name__)


class IKRCompiler:
    """Compiles IKR to SMT2 format."""

    # Built-in sorts that don't need declaration
    BUILTIN_SORTS = {"Bool", "Int", "Real"}

    def __init__(self) -> None:
        """Initialize the compiler."""
        self._type_map: dict[str, str] = {}  # IKR type name -> SMT2 sort name
        self._entity_map: dict[str, str] = {}  # IKR entity name -> SMT2 constant name

    def compile(self, ikr: IKR) -> str:
        """Compile IKR to SMT2 code.

        Args:
            ikr: The IKR to compile

        Returns:
            SMT2 code as a string

        Raises:
            ValueError: If IKR has validation errors
        """
        # Validate references first
        errors = ikr.validate_references()
        if errors:
            raise ValueError(f"IKR validation errors: {errors}")

        # Build mappings
        self._build_mappings(ikr)

        # Compile sections
        sections = []

        # Add header comment
        sections.append(self._compile_header(ikr))

        # Declare sorts
        if ikr.types:
            sections.append(self._compile_types(ikr.types))

        # Declare functions
        if ikr.relations:
            sections.append(self._compile_relations(ikr.relations))

        # Declare constants
        if ikr.entities:
            sections.append(self._compile_entities(ikr.entities))

        # Assert facts
        if ikr.facts:
            sections.append(self._compile_facts(ikr.facts))

        # Assert rules
        if ikr.rules:
            sections.append(self._compile_rules(ikr.rules))

        # Assert query
        sections.append(self._compile_query(ikr.query))

        # Check satisfiability
        sections.append("(check-sat)")

        return "\n\n".join(sections)

    def _build_mappings(self, ikr: IKR) -> None:
        """Build internal mappings from IKR names to SMT2 identifiers."""
        self._type_map = {t.name: t.name for t in ikr.types}
        self._type_map.update({"Bool": "Bool", "Int": "Int", "Real": "Real"})
        self._entity_map = {e.name: e.name for e in ikr.entities}

    def _compile_header(self, ikr: IKR) -> str:
        """Generate header comment."""
        return f"""; IKR-generated SMT2 program
; Question: {ikr.meta.question}
; Type: {ikr.meta.question_type.value}"""

    def _compile_types(self, types: list[Type]) -> str:
        """Compile type declarations to SMT2 sorts."""
        lines = ["; === Types ==="]
        for t in types:
            if t.name not in self.BUILTIN_SORTS:
                lines.append(f"(declare-sort {t.name} 0)")
                if t.description:
                    lines[-1] = f"{lines[-1]}  ; {t.description}"
        return "\n".join(lines)

    def _compile_relations(self, relations: list[Relation]) -> str:
        """Compile relation declarations to SMT2 functions."""
        lines = ["; === Relations ==="]
        for rel in relations:
            domain = " ".join(rel.signature)
            range_type = rel.range
            lines.append(f"(declare-fun {rel.name} ({domain}) {range_type})")

            # Add symmetry axiom if needed
            if rel.symmetric and len(rel.signature) == 2:
                sort1, sort2 = rel.signature
                if sort1 == sort2:
                    lines.append(
                        f"(assert (forall ((x {sort1}) (y {sort2})) "
                        f"(= ({rel.name} x y) ({rel.name} y x))))"
                    )

            # Add transitivity axiom if needed
            if rel.transitive and len(rel.signature) == 2:
                sort1, sort2 = rel.signature
                if sort1 == sort2:
                    lines.append(
                        f"(assert (forall ((x {sort1}) (y {sort1}) (z {sort1})) "
                        f"(=> (and ({rel.name} x y) ({rel.name} y z)) "
                        f"({rel.name} x z))))"
                    )

        return "\n".join(lines)

    def _compile_entities(self, entities: list[Entity]) -> str:
        """Compile entity declarations to SMT2 constants."""
        lines = ["; === Entities ==="]
        for ent in entities:
            lines.append(f"(declare-const {ent.name} {ent.type})")
        return "\n".join(lines)

    def _compile_facts(self, facts: list[Fact]) -> str:
        """Compile facts to SMT2 assertions."""
        explicit = [f for f in facts if f.source == "explicit"]
        background = [f for f in facts if f.source == "background"]

        lines = []

        if explicit:
            lines.append("; === Explicit Facts ===")
            for fact in explicit:
                lines.append(self._compile_fact(fact))

        if background:
            lines.append("; === Background Knowledge ===")
            for fact in background:
                assertion = self._compile_fact(fact)
                if fact.justification:
                    assertion = f"{assertion}  ; {fact.justification}"
                lines.append(assertion)

        return "\n".join(lines)

    def _compile_fact(self, fact: Fact) -> str:
        """Compile a single fact to SMT2 assertion."""
        args = " ".join(fact.arguments)

        if fact.value is not None:
            # Function equality: f(a) = value
            expr = f"(= ({fact.predicate} {args}) {self._compile_value(fact.value)})"
        else:
            # Predicate: p(a) or (not (p(a)))
            expr = f"({fact.predicate} {args})"

        if fact.negated:
            expr = f"(not {expr})"

        return f"(assert {expr})"

    def _compile_value(self, value: int | float | str) -> str:
        """Compile a value to SMT2 literal."""
        if isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, int):
            return str(value)
        elif isinstance(value, float):
            return str(value)
        else:
            # Assume it's an entity reference
            return str(value)

    def _compile_rules(self, rules: list[Rule]) -> str:
        """Compile rules to SMT2 quantified assertions."""
        lines = ["; === Rules ==="]
        for rule in rules:
            lines.append(self._compile_rule(rule))
        return "\n".join(lines)

    def _compile_rule(self, rule: Rule) -> str:
        """Compile a single rule to SMT2."""
        if rule.name:
            comment = f"; {rule.name}"
            if rule.justification:
                comment = f"{comment}: {rule.justification}"
        else:
            comment = ""

        if rule.is_constraint():
            # Simple constraint (possibly quantified)
            constraint_expr = self._compile_condition(rule.constraint)
            if rule.quantified_vars:
                var_decls = " ".join(
                    f"({v.name} {v.type})" for v in rule.quantified_vars
                )
                expr = f"(forall ({var_decls}) {constraint_expr})"
            else:
                expr = constraint_expr
        elif rule.is_implication():
            # Implication rule
            antecedent = self._compile_condition(rule.antecedent)
            consequent = self._compile_condition(rule.consequent)
            implication = f"(=> {antecedent} {consequent})"

            if rule.quantified_vars:
                var_decls = " ".join(
                    f"({v.name} {v.type})" for v in rule.quantified_vars
                )
                expr = f"(forall ({var_decls}) {implication})"
            else:
                expr = implication
        else:
            raise ValueError(f"Rule must be either constraint or implication: {rule}")

        assertion = f"(assert {expr})"
        if comment:
            assertion = f"{comment}\n{assertion}"
        return assertion

    def _compile_condition(self, cond: RuleCondition) -> str:
        """Compile a rule condition to SMT2 expression."""
        if cond.and_:
            # Conjunction
            sub_exprs = [self._compile_condition(c) for c in cond.and_]
            expr = f"(and {' '.join(sub_exprs)})"
        elif cond.or_:
            # Disjunction
            sub_exprs = [self._compile_condition(c) for c in cond.or_]
            expr = f"(or {' '.join(sub_exprs)})"
        elif cond.predicate:
            # Simple predicate
            args = " ".join(cond.arguments)
            if cond.value is not None:
                expr = f"(= ({cond.predicate} {args}) {self._compile_value(cond.value)})"
            else:
                expr = f"({cond.predicate} {args})"
        else:
            raise ValueError(f"Invalid condition: {cond}")

        if cond.negated:
            expr = f"(not {expr})"

        return expr

    def _compile_query(self, query: Query) -> str:
        """Compile query to SMT2 assertion."""
        args = " ".join(query.arguments)
        expr = f"({query.predicate} {args})"

        if query.negated:
            expr = f"(not {expr})"

        return f"""; === Query ===
(assert {expr})"""


def compile_ikr_to_smt2(ikr: IKR) -> str:
    """Convenience function to compile IKR to SMT2.

    Args:
        ikr: The IKR to compile

    Returns:
        SMT2 code as a string
    """
    compiler = IKRCompiler()
    return compiler.compile(ikr)
