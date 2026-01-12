"""Compiler for IKR to Souffle/Datalog translation.

This module provides compilation from the Intermediate Knowledge
Representation (IKR) to Souffle Datalog format. The compiler generates:

1. A .dl program file with:
   - Type declarations
   - Relation declarations (.decl)
   - Input/output directives
   - Rules (Horn clauses)

2. .facts files for input relations (tab-separated)

The compilation process differs from SMT2 in key ways:
- SMT2: SAT = query consistent, UNSAT = query contradicts
- Datalog: derivable = True, not derivable = False/Unknown

For yes/no questions, we check if the query tuple is derivable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass
class SouffleProgram:
    """Compiled Souffle program ready for execution.

    Attributes:
        program: The .dl program source code
        facts: Mapping of relation name to list of fact tuples
        query_relation: Name of the output relation to check for results
    """

    program: str
    facts: dict[str, list[tuple[str, ...]]] = field(default_factory=dict)
    query_relation: str = "query_result"


class IKRSouffleCompiler:
    """Compiles IKR to Souffle Datalog format.

    Type mapping:
    - Custom IKR types → symbol (string interning)
    - Bool → number (0/1)
    - Int → number
    - Real → float
    """

    # Souffle type mapping
    TYPE_MAP = {
        "Bool": "number",  # 0 or 1
        "Int": "number",
        "Real": "float",
    }

    def __init__(self) -> None:
        """Initialize the compiler."""
        self._type_map: dict[str, str] = {}
        self._entity_map: dict[str, str] = {}
        self._relation_arities: dict[str, int] = {}

    def compile(self, ikr: IKR) -> SouffleProgram:
        """Compile IKR to Souffle program.

        Args:
            ikr: The IKR to compile

        Returns:
            SouffleProgram with .dl source and facts

        Raises:
            ValueError: If IKR has validation errors
        """
        # Validate references first
        errors = ikr.validate_references()
        if errors:
            raise ValueError(f"IKR validation errors: {errors}")

        # Build mappings
        self._build_mappings(ikr)

        # Compile program sections
        sections = []

        # Header comment
        sections.append(self._compile_header(ikr))

        # Type aliases (for documentation, Souffle uses symbol/number/float)
        sections.append(self._compile_type_aliases(ikr.types))

        # Relation declarations
        sections.append(self._compile_relations(ikr.relations))

        # Input/output directives
        sections.append(self._compile_io_directives(ikr))

        # Rules (including implicit rules for symmetric/transitive relations)
        if ikr.rules or self._has_special_relations(ikr.relations):
            sections.append(self._compile_rules(ikr))

        # Query derivation rule
        sections.append(self._compile_query_rule(ikr.query))

        program = "\n\n".join(filter(None, sections))

        # Compile facts to separate structure
        facts = self._compile_facts(ikr)

        return SouffleProgram(
            program=program,
            facts=facts,
            query_relation="query_result",
        )

    def write_program(
        self, souffle_program: SouffleProgram, output_dir: Path
    ) -> tuple[Path, Path]:
        """Write compiled program to files.

        Args:
            souffle_program: Compiled Souffle program
            output_dir: Directory to write files to

        Returns:
            Tuple of (program_path, facts_dir)
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write .dl program
        program_path = output_dir / "program.dl"
        program_path.write_text(souffle_program.program)

        # Write facts to facts directory
        facts_dir = output_dir / "facts"
        facts_dir.mkdir(exist_ok=True)

        for relation_name, tuples in souffle_program.facts.items():
            facts_file = facts_dir / f"{relation_name}.facts"
            with open(facts_file, "w") as f:
                for tup in tuples:
                    f.write("\t".join(str(v) for v in tup) + "\n")

        return program_path, facts_dir

    def _build_mappings(self, ikr: IKR) -> None:
        """Build internal mappings from IKR names."""
        # Type mapping: custom types → symbol
        self._type_map = {t.name: "symbol" for t in ikr.types}
        self._type_map.update(self.TYPE_MAP)

        # Entity mapping (keep names as-is for now)
        self._entity_map = {e.name: e.name for e in ikr.entities}

        # Track relation arities
        self._relation_arities = {r.name: len(r.signature) for r in ikr.relations}

    def _compile_header(self, ikr: IKR) -> str:
        """Generate header comment."""
        return f"""// IKR-generated Souffle program
// Question: {ikr.meta.question}
// Type: {ikr.meta.question_type.value}"""

    def _compile_type_aliases(self, types: list[Type]) -> str:
        """Generate type alias comments (Souffle doesn't have true type aliases)."""
        if not types:
            return ""

        lines = ["// === Types (mapped to symbol) ==="]
        for t in types:
            desc = f" - {t.description}" if t.description else ""
            lines.append(f"// {t.name}: symbol{desc}")
        return "\n".join(lines)

    def _compile_relations(self, relations: list[Relation]) -> str:
        """Compile relation declarations."""
        if not relations:
            return ""

        lines = ["// === Relations ==="]
        for rel in relations:
            decl = self._compile_relation_decl(rel)
            lines.append(decl)

        # Add query result relation
        lines.append("")
        lines.append("// Query result")
        lines.append(".decl query_result()")

        return "\n".join(lines)

    def _compile_relation_decl(self, rel: Relation) -> str:
        """Compile a single relation declaration."""
        # Build argument list with generated names
        args = []
        for i, sig_type in enumerate(rel.signature):
            souffle_type = self._type_map.get(sig_type, "symbol")
            args.append(f"x{i}: {souffle_type}")

        # Handle range type for functions (non-Bool relations)
        if rel.range != "Bool":
            range_type = self._type_map.get(rel.range, "symbol")
            args.append(f"val: {range_type}")

        return f".decl {rel.name}({', '.join(args)})"

    def _compile_io_directives(self, ikr: IKR) -> str:
        """Compile input/output directives."""
        lines = ["// === I/O Directives ==="]

        # All base relations are input
        for rel in ikr.relations:
            lines.append(f".input {rel.name}")

        # Query result is output
        lines.append(".output query_result")

        return "\n".join(lines)

    def _has_special_relations(self, relations: list[Relation]) -> bool:
        """Check if any relations need special axioms."""
        return any(r.symmetric or r.transitive for r in relations)

    def _compile_rules(self, ikr: IKR) -> str:
        """Compile rules to Datalog Horn clauses."""
        lines = ["// === Rules ==="]

        # Generate axioms for symmetric/transitive relations
        for rel in ikr.relations:
            if rel.symmetric and len(rel.signature) == 2:
                # r(X, Y) :- r(Y, X).
                lines.append(f"// Symmetry axiom for {rel.name}")
                lines.append(f"{rel.name}(X, Y) :- {rel.name}(Y, X).")

            if rel.transitive and len(rel.signature) == 2:
                # r(X, Z) :- r(X, Y), r(Y, Z).
                lines.append(f"// Transitivity axiom for {rel.name}")
                lines.append(f"{rel.name}(X, Z) :- {rel.name}(X, Y), {rel.name}(Y, Z).")

        # Compile user-defined rules
        for rule in ikr.rules:
            compiled = self._compile_rule(rule)
            if compiled:
                if rule.name:
                    lines.append(f"// {rule.name}")
                lines.append(compiled)

        return "\n".join(lines)

    def _compile_rule(self, rule: Rule) -> str | None:
        """Compile a single rule to Datalog.

        Datalog rules are Horn clauses: head :- body.
        IKR rules are implications: antecedent => consequent

        Translation: consequent :- antecedent
        """
        if rule.is_implication():
            # antecedent => consequent becomes consequent :- antecedent
            head = self._compile_condition_as_head(rule.consequent)
            body = self._compile_condition_as_body(rule.antecedent)

            if head and body:
                return f"{head} :- {body}."
            elif head:
                # Unconditional fact (no antecedent)
                return f"{head}."

        elif rule.is_constraint():
            # Constraints in Datalog require special handling
            # For now, we skip pure constraints as they don't have direct Datalog equivalents
            logger.warning(f"Skipping constraint rule (not directly expressible in Datalog): {rule}")
            return None

        return None

    def _compile_condition_as_head(self, cond: RuleCondition | None) -> str | None:
        """Compile a condition as a rule head (must be single atom)."""
        if cond is None:
            return None

        if cond.is_simple() and not cond.negated:
            args = ", ".join(cond.arguments)
            return f"{cond.predicate}({args})"

        # Compound or negated conditions can't be rule heads in Datalog
        logger.warning(f"Cannot use compound/negated condition as rule head: {cond}")
        return None

    def _compile_condition_as_body(self, cond: RuleCondition | None) -> str | None:
        """Compile a condition as a rule body."""
        if cond is None:
            return None

        if cond.and_:
            # Conjunction: flatten to comma-separated atoms
            parts = []
            for sub in cond.and_:
                compiled = self._compile_condition_as_body(sub)
                if compiled:
                    parts.append(compiled)
            return ", ".join(parts) if parts else None

        elif cond.or_:
            # Disjunction: Datalog doesn't directly support OR in bodies
            # Would need multiple rules, but for now we warn
            logger.warning(f"Disjunction in rule body not directly supported: {cond}")
            # Try to use first disjunct as approximation
            if cond.or_:
                return self._compile_condition_as_body(cond.or_[0])
            return None

        elif cond.is_simple():
            args = ", ".join(cond.arguments)
            atom = f"{cond.predicate}({args})"
            if cond.negated:
                return f"!{atom}"
            return atom

        return None

    def _compile_query_rule(self, query: Query) -> str:
        """Compile query as a derivation rule.

        The query becomes a rule that derives query_result() if
        the query predicate holds for the given arguments.
        """
        lines = ["// === Query ==="]

        args = ", ".join(query.arguments)
        query_atom = f"{query.predicate}({args})"

        if query.negated:
            # query_result() :- !pred(args).
            lines.append(f"query_result() :- !{query_atom}.")
        else:
            # query_result() :- pred(args).
            lines.append(f"query_result() :- {query_atom}.")

        return "\n".join(lines)

    def _compile_facts(self, ikr: IKR) -> dict[str, list[tuple[str, ...]]]:
        """Compile facts to dictionary of relation -> tuples."""
        facts: dict[str, list[tuple[str, ...]]] = {}

        for fact in ikr.facts:
            if fact.predicate not in facts:
                facts[fact.predicate] = []

            # Build tuple from arguments
            if fact.value is not None:
                # Function assignment: include value
                tup = tuple(fact.arguments) + (str(fact.value),)
            else:
                tup = tuple(fact.arguments)

            # Handle negation: in Datalog, we can't have negative base facts
            # We'd need a separate "not_predicate" relation
            if fact.negated:
                logger.warning(
                    f"Negated fact not directly supported in Datalog base facts: {fact}"
                )
                # Create negated version of relation if needed
                neg_rel = f"not_{fact.predicate}"
                if neg_rel not in facts:
                    facts[neg_rel] = []
                facts[neg_rel].append(tup)
            else:
                facts[fact.predicate].append(tup)

        return facts


def compile_ikr_to_souffle(ikr: IKR) -> SouffleProgram:
    """Convenience function to compile IKR to Souffle.

    Args:
        ikr: The IKR to compile

    Returns:
        SouffleProgram with .dl source and facts
    """
    compiler = IKRSouffleCompiler()
    return compiler.compile(ikr)
