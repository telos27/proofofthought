"""Compiler for IKR to SMT2 translation.

This module provides deterministic compilation from the Intermediate
Knowledge Representation (IKR) to SMT-LIB 2.0 format. The compiler
ensures syntactically correct SMT2 output, eliminating the syntax
errors that commonly occur with direct LLM generation.

The compilation process:
1. Declare sorts from types (including World for epistemic logic)
2. Declare functions from relations (with accessibility relations for epistemic)
3. Declare constants from entities
4. Assert facts as ground truths (or soft constraints for uncertain facts)
5. Assert rules as quantified formulas
6. Assert epistemic axioms (if epistemic_config present)
7. Assert query for satisfiability check
8. Add (check-sat) or (check-sat-using opt) command

Extended features:
- Epistemic logic: possible worlds with accessibility relations
- Uncertainty: soft constraints with weights for MaxSMT optimization
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from z3adapter.ikr.schema import (
        Entity,
        EpistemicConfig,
        EpistemicContext,
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
    """Compiles IKR to SMT2 format.

    Supports extended features:
    - Epistemic logic via possible worlds semantics
    - Uncertainty via soft constraints (MaxSMT)
    """

    # Built-in sorts that don't need declaration
    BUILTIN_SORTS = {"Bool", "Int", "Real"}

    def __init__(self, use_optimization: bool | None = None) -> None:
        """Initialize the compiler.

        Args:
            use_optimization: Whether to use Z3 optimization for soft constraints.
                             If None, automatically enabled when uncertain facts present.
        """
        self._type_map: dict[str, str] = {}  # IKR type name -> SMT2 sort name
        self._entity_map: dict[str, str] = {}  # IKR entity name -> SMT2 constant name
        self._use_optimization = use_optimization
        self._has_epistemic = False
        self._has_uncertainty = False

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

        # Check for extended features
        self._has_epistemic = ikr.has_epistemic_content()
        self._has_uncertainty = ikr.has_uncertainty()

        # Determine if we need optimization
        use_opt = self._use_optimization
        if use_opt is None:
            use_opt = self._has_uncertainty

        # Compile sections
        sections = []

        # Add header comment
        sections.append(self._compile_header(ikr))

        # Declare sorts (including World for epistemic)
        types_section = self._compile_types(ikr.types)
        if self._has_epistemic and ikr.epistemic_config:
            types_section += "\n" + self._compile_epistemic_sorts(ikr.epistemic_config)
        if types_section.strip():
            sections.append(types_section)

        # Declare functions (including accessibility relations)
        if ikr.relations:
            relations_section = self._compile_relations(ikr.relations)
            if self._has_epistemic and ikr.epistemic_config:
                relations_section += "\n" + self._compile_accessibility_relations(
                    ikr.epistemic_config
                )
            sections.append(relations_section)

        # Declare constants
        if ikr.entities:
            sections.append(self._compile_entities(ikr.entities))

        # Assert epistemic axioms
        if self._has_epistemic and ikr.epistemic_config:
            axioms = self._compile_epistemic_axioms(ikr.epistemic_config)
            if axioms:
                sections.append(axioms)

        # Assert facts (with soft constraints for uncertain facts)
        if ikr.facts:
            sections.append(self._compile_facts(ikr.facts, use_soft=use_opt))

        # Assert rules
        if ikr.rules:
            sections.append(self._compile_rules(ikr.rules))

        # Assert query
        sections.append(self._compile_query(ikr.query))

        # Check satisfiability (with optimization if needed)
        if use_opt:
            sections.append("(check-sat)")
            sections.append("(get-objectives)")
        else:
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

    def _compile_facts(self, facts: list[Fact], use_soft: bool = False) -> str:
        """Compile facts to SMT2 assertions.

        Args:
            facts: List of facts to compile
            use_soft: If True, compile uncertain facts as soft constraints
        """
        explicit = [f for f in facts if f.source == "explicit"]
        background = [f for f in facts if f.source == "background"]
        kb_facts = [f for f in facts if f.source == "kb"]
        revised = [f for f in facts if f.source == "revised"]

        lines = []

        if explicit:
            lines.append("; === Explicit Facts ===")
            for fact in explicit:
                lines.append(self._compile_fact(fact, use_soft=use_soft))

        if background:
            lines.append("; === Background Knowledge ===")
            for fact in background:
                assertion = self._compile_fact(fact, use_soft=use_soft)
                if fact.justification:
                    assertion = f"{assertion}  ; {fact.justification}"
                lines.append(assertion)

        if kb_facts:
            lines.append("; === Knowledge Base Facts ===")
            for fact in kb_facts:
                assertion = self._compile_fact(fact, use_soft=use_soft)
                if fact.justification:
                    assertion = f"{assertion}  ; {fact.justification}"
                lines.append(assertion)

        if revised:
            lines.append("; === Revised Facts (from NARS) ===")
            for fact in revised:
                assertion = self._compile_fact(fact, use_soft=use_soft)
                if fact.truth_value:
                    assertion = f"{assertion}  ; f={fact.truth_value.frequency:.3f}, c={fact.truth_value.confidence:.3f}"
                lines.append(assertion)

        return "\n".join(lines)

    def _compile_fact(self, fact: Fact, use_soft: bool = False) -> str:
        """Compile a single fact to SMT2 assertion.

        Args:
            fact: Fact to compile
            use_soft: If True and fact has truth value, use soft constraint

        Returns:
            SMT2 assertion string
        """
        args = " ".join(fact.arguments)

        if fact.value is not None:
            # Function equality: f(a) = value
            expr = f"(= ({fact.predicate} {args}) {self._compile_value(fact.value)})"
        else:
            # Predicate: p(a) or (not (p(a)))
            expr = f"({fact.predicate} {args})"

        if fact.negated:
            expr = f"(not {expr})"

        # Handle epistemic context
        if fact.epistemic_context:
            expr = self._wrap_epistemic(expr, fact.epistemic_context)

        # Handle uncertain facts as soft constraints
        if use_soft and fact.truth_value is not None:
            weight = self._compute_weight(fact)
            return f"(assert-soft {expr} :weight {weight})"

        return f"(assert {expr})"

    def _wrap_epistemic(self, expr: str, ctx: EpistemicContext) -> str:
        """Wrap expression in epistemic context.

        Uses possible worlds encoding:
        Ka(φ) → ∀w'(Ra(actual, w') → φ[w'])

        For simplicity, we use a named world variable approach.
        """
        agent = ctx.agent

        # Handle nested contexts (recursively wrap)
        if ctx.nested_in:
            expr = self._wrap_epistemic(expr, ctx.nested_in)

        # Wrap in forall with accessibility relation
        # Note: This is a simplified encoding. For full epistemic logic,
        # predicates would need world parameters.
        return f"(forall ((w World)) (=> (R_{agent} actual_world w) {expr}))"

    def _compute_weight(self, fact: Fact) -> int:
        """Compute weight for soft constraint from truth value."""
        if fact.truth_value is None:
            return 1000  # Max weight for classical facts

        tv = fact.truth_value
        # Weight = frequency * confidence * 1000
        weight = int(tv.frequency * tv.confidence * 1000)
        return max(1, min(1000, weight))

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

    # === Epistemic Logic Support ===

    def _compile_epistemic_sorts(self, config: EpistemicConfig) -> str:
        """Compile epistemic sorts (World type and actual_world constant)."""
        lines = ["; === Epistemic Framework ==="]
        lines.append("(declare-sort World 0)")
        lines.append("(declare-const actual_world World)")
        return "\n".join(lines)

    def _compile_accessibility_relations(self, config: EpistemicConfig) -> str:
        """Compile accessibility relations for each agent."""
        lines = ["; === Accessibility Relations ==="]
        for agent in config.agents:
            # R_agent: World -> World -> Bool
            lines.append(f"(declare-fun R_{agent} (World World) Bool)")
        return "\n".join(lines)

    def _compile_epistemic_axioms(self, config: EpistemicConfig) -> str:
        """Compile epistemic axioms based on the axiom system.

        - K: Basic modal logic (distribution axiom only - implicit)
        - KD45: Belief logic (serial, transitive, euclidean)
        - S5: Knowledge logic (reflexive, symmetric, transitive)
        """
        lines = ["; === Epistemic Axioms ==="]

        for agent in config.agents:
            R = f"R_{agent}"

            if config.axiom_system == "S5":
                # S5: Equivalence relation (reflexive, symmetric, transitive)
                # Reflexivity: R(w, w) for all w
                lines.append(f"; {agent} knowledge axioms (S5)")
                lines.append(
                    f"(assert (forall ((w World)) ({R} w w)))  ; reflexivity"
                )
                # Symmetry: R(w1, w2) => R(w2, w1)
                lines.append(
                    f"(assert (forall ((w1 World) (w2 World)) "
                    f"(=> ({R} w1 w2) ({R} w2 w1))))  ; symmetry"
                )
                # Transitivity: R(w1, w2) & R(w2, w3) => R(w1, w3)
                lines.append(
                    f"(assert (forall ((w1 World) (w2 World) (w3 World)) "
                    f"(=> (and ({R} w1 w2) ({R} w2 w3)) ({R} w1 w3))))  ; transitivity"
                )

            elif config.axiom_system == "KD45":
                # KD45: Belief logic (serial, transitive, euclidean)
                lines.append(f"; {agent} belief axioms (KD45)")
                # Seriality: For all w, exists w' such that R(w, w')
                lines.append(
                    f"(assert (forall ((w World)) "
                    f"(exists ((w2 World)) ({R} w w2))))  ; seriality"
                )
                # Transitivity
                lines.append(
                    f"(assert (forall ((w1 World) (w2 World) (w3 World)) "
                    f"(=> (and ({R} w1 w2) ({R} w2 w3)) ({R} w1 w3))))  ; transitivity"
                )
                # Euclidean: R(w1, w2) & R(w1, w3) => R(w2, w3)
                lines.append(
                    f"(assert (forall ((w1 World) (w2 World) (w3 World)) "
                    f"(=> (and ({R} w1 w2) ({R} w1 w3)) ({R} w2 w3))))  ; euclidean"
                )

            elif config.axiom_system == "K":
                # K: Basic modal logic - no special axioms needed
                # The distribution axiom K: K(p → q) → (Kp → Kq) is built into the semantics
                lines.append(f"; {agent} basic belief axioms (K) - no special constraints")

        return "\n".join(lines)


def compile_ikr_to_smt2(ikr: IKR) -> str:
    """Convenience function to compile IKR to SMT2.

    Args:
        ikr: The IKR to compile

    Returns:
        SMT2 code as a string
    """
    compiler = IKRCompiler()
    return compiler.compile(ikr)
