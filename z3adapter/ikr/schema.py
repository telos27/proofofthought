"""Pydantic models for Intermediate Knowledge Representation (IKR).

This module defines the schema for the minimal IKR used to represent
logical reasoning problems. The schema supports:

- Types: Domain declarations (Person, Food, etc.)
- Entities: Named individuals of specific types
- Relations: Predicates and functions with signatures
- Facts: Ground assertions (explicit and background knowledge)
- Rules: Universally quantified implications
- Query: The property to check for satisfiability

Example IKR (JSON):
    {
        "meta": {"question": "Would a vegetarian eat a plant burger?"},
        "types": [{"name": "Person"}, {"name": "Food"}],
        "entities": [
            {"name": "vegetarian_person", "type": "Person"},
            {"name": "plant_burger", "type": "Food"}
        ],
        "relations": [
            {"name": "is_vegetarian", "signature": ["Person"], "range": "Bool"},
            {"name": "would_eat", "signature": ["Person", "Food"], "range": "Bool"}
        ],
        "facts": [
            {"predicate": "is_vegetarian", "arguments": ["vegetarian_person"]}
        ],
        "rules": [...],
        "query": {"predicate": "would_eat", "arguments": ["vegetarian_person", "plant_burger"]}
    }
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class QuestionType(str, Enum):
    """Type of question being asked."""

    YES_NO = "yes_no"
    COMPARISON = "comparison"
    POSSIBILITY = "possibility"
    FACTUAL = "factual"


class RangeType(str, Enum):
    """Built-in range types for relations."""

    BOOL = "Bool"
    INT = "Int"
    REAL = "Real"


class TruthValue(BaseModel):
    """NARS-style truth value with frequency and confidence.

    This enables handling uncertain information and contradictions:
    - frequency (f): [0.0, 1.0] - proportion of positive evidence
    - confidence (c): (0.0, 1.0) - strength of evidence

    Example:
        - TruthValue(frequency=0.8, confidence=0.9) means "likely true, high confidence"
        - TruthValue(frequency=0.5, confidence=0.1) means "uncertain, low evidence"
    """

    frequency: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Proportion of positive evidence [0.0, 1.0]",
    )
    confidence: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        description="Strength of evidence (0.0, 1.0)",
    )

    @classmethod
    def from_evidence(cls, positive: int, total: int, k: float = 1.0) -> "TruthValue":
        """Create truth value from evidence counts.

        Args:
            positive: Number of positive observations
            total: Total number of observations
            k: Evidence horizon constant (higher = more evidence needed for high confidence)

        Returns:
            TruthValue computed from evidence
        """
        f = positive / total if total > 0 else 0.5
        c = total / (total + k) if (total + k) > 0 else 0.0
        # Clamp confidence to valid range
        c = max(0.001, min(0.999, c))
        return cls(frequency=f, confidence=c)

    def revise(self, other: "TruthValue") -> "TruthValue":
        """NARS revision function for combining conflicting evidence.

        Combines this truth value with another using weighted average
        based on confidence levels.

        Args:
            other: Another truth value to combine with

        Returns:
            New truth value representing combined evidence
        """
        c1, c2 = self.confidence, other.confidence
        f1, f2 = self.frequency, other.frequency

        # Weighted combination based on confidence
        # w1 = c1 * (1 - c2), w2 = c2 * (1 - c1)
        w1 = c1 * (1 - c2)
        w2 = c2 * (1 - c1)
        total_weight = w1 + w2

        if total_weight > 0:
            f_new = (f1 * w1 + f2 * w2) / total_weight
        else:
            f_new = (f1 + f2) / 2  # Equal weight if both have max confidence

        # Combined confidence (approaches 1 as evidence accumulates)
        c_new = (c1 + c2 - c1 * c2)
        c_new = max(0.001, min(0.999, c_new))

        return TruthValue(frequency=f_new, confidence=c_new)

    def negate(self) -> "TruthValue":
        """Return negated truth value (frequency inverted, confidence preserved)."""
        return TruthValue(frequency=1.0 - self.frequency, confidence=self.confidence)

    def to_evidence(self, k: float = 1.0) -> tuple[float, float]:
        """Convert truth value back to evidence counts.

        Inverse of from_evidence(). Useful for NARS revision operations
        that need to pool evidence from multiple sources.

        Args:
            k: Evidential horizon constant (must match from_evidence)

        Returns:
            (positive_evidence, total_evidence) tuple
        """
        if self.confidence >= 0.999:
            # Near-maximum confidence = very large evidence
            return (float("inf"), float("inf"))

        # c = w/(w+k) → w = ck/(1-c)
        total = self.confidence * k / (1 - self.confidence)
        positive = self.frequency * total
        return (positive, total)

    def expectation(self) -> float:
        """Return expected truth value (frequency * confidence + 0.5 * (1 - confidence))."""
        return self.frequency * self.confidence + 0.5 * (1 - self.confidence)

    def is_positive(self) -> bool:
        """Return True if frequency >= 0.5 (more positive than negative evidence)."""
        return self.frequency >= 0.5


class EpistemicOperator(str, Enum):
    """Epistemic modality operators for belief/knowledge attribution."""

    BELIEVES = "believes"  # Doxastic: agent holds belief (may be false)
    KNOWS = "knows"  # Epistemic: justified true belief (S5 axioms)


class EpistemicContext(BaseModel):
    """Epistemic context for belief/knowledge attribution.

    Enables representing nested beliefs like "A believes B believes C".
    Uses possible worlds semantics where each agent has an accessibility
    relation defining which worlds they consider possible.

    Example:
        {"agent": "alice", "modality": "believes"}
        means this fact is within Alice's belief state

        {"agent": "bob", "modality": "believes",
         "nested_in": {"agent": "alice", "modality": "believes"}}
        means Alice believes that Bob believes this fact
    """

    agent: str = Field(..., description="Agent holding the belief/knowledge")
    modality: EpistemicOperator = Field(
        default=EpistemicOperator.BELIEVES,
        description="Type of epistemic attitude",
    )
    nested_in: "EpistemicContext | None" = Field(
        default=None,
        description="Outer epistemic context (for nested beliefs)",
    )

    def depth(self) -> int:
        """Return nesting depth of this context (1 = not nested)."""
        if self.nested_in is None:
            return 1
        return 1 + self.nested_in.depth()

    def agents_chain(self) -> list[str]:
        """Return list of agents from outermost to innermost."""
        if self.nested_in is None:
            return [self.agent]
        return self.nested_in.agents_chain() + [self.agent]


class EpistemicConfig(BaseModel):
    """Configuration for epistemic reasoning in the IKR.

    Defines agents and the modal logic axiom system to use:
    - K: Basic modal logic (distribution axiom only)
    - KD45: Belief logic (serial, transitive, euclidean)
    - S5: Knowledge logic (reflexive, symmetric, transitive - equivalence relation)
    """

    agents: list[str] = Field(
        default_factory=list,
        description="List of epistemic agents in the problem",
    )
    axiom_system: Literal["K", "KD45", "S5"] = Field(
        default="KD45",
        description="Modal logic axiom system to use",
    )
    common_knowledge: bool = Field(
        default=False,
        description="Enable common knowledge operator (fixed-point computation)",
    )


class Meta(BaseModel):
    """Metadata about the reasoning problem."""

    question: str = Field(..., description="Original natural language question")
    question_type: QuestionType = Field(
        default=QuestionType.YES_NO, description="Type of question"
    )


class Type(BaseModel):
    """A type/sort declaration.

    Types represent domains like Person, Food, Location, etc.
    They correspond to Z3's uninterpreted sorts.
    """

    name: str = Field(..., description="Type name (e.g., 'Person', 'Food')")
    description: str | None = Field(
        default=None, description="Optional description of what this type represents"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure type name is a valid identifier."""
        if not v.isidentifier():
            raise ValueError(f"Type name must be a valid identifier, got: {v}")
        return v


class Entity(BaseModel):
    """A named entity/constant of a specific type.

    Entities are ground terms like 'vegetarian_person' or 'plant_burger'.
    They correspond to Z3 constants.
    """

    name: str = Field(..., description="Entity name (snake_case preferred)")
    type: str = Field(..., description="Type this entity belongs to")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names from the question text",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure entity name is a valid identifier."""
        if not v.replace("_", "").isalnum():
            raise ValueError(f"Entity name must be alphanumeric with underscores, got: {v}")
        return v


class Relation(BaseModel):
    """A relation (predicate or function) declaration.

    Relations define the vocabulary of the logical theory.
    - Predicates return Bool (e.g., is_vegetarian(Person) -> Bool)
    - Functions return other types (e.g., age(Person) -> Int)
    """

    name: str = Field(..., description="Relation name")
    signature: list[str] = Field(
        ..., description="Argument types in order", min_length=1
    )
    range: str = Field(
        default="Bool",
        description="Return type: 'Bool', 'Int', 'Real', or a custom type name",
    )
    symmetric: bool = Field(default=False, description="Is this relation symmetric?")
    transitive: bool = Field(default=False, description="Is this relation transitive?")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure relation name is a valid identifier."""
        if not v.replace("_", "").isalnum():
            raise ValueError(f"Relation name must be alphanumeric with underscores, got: {v}")
        return v


class Fact(BaseModel):
    """A ground fact assertion.

    Facts are assertions about specific entities, e.g.:
    - is_vegetarian(vegetarian_person)
    - age(alice) = 30
    - NOT contains(plant_burger, meat)

    Extended with:
    - truth_value: NARS-style uncertainty (frequency, confidence)
    - epistemic_context: Whose belief state this fact belongs to
    """

    predicate: str = Field(..., description="Relation name being asserted")
    arguments: list[str] = Field(..., description="Arguments (entity names)")
    negated: bool = Field(default=False, description="Whether this is a negative fact")
    value: int | float | str | None = Field(
        default=None,
        description="For functions: the value being asserted (e.g., age = 30)",
    )
    source: Literal["explicit", "background", "kb", "revised"] = Field(
        default="explicit",
        description="Source: explicit (from question), background (world knowledge), kb (knowledge base), revised (from NARS revision)",
    )
    justification: str | None = Field(
        default=None,
        description="For background facts: why this is common knowledge",
    )

    # Uncertainty support (NARS-style)
    truth_value: TruthValue | None = Field(
        default=None,
        description="NARS-style truth value for uncertain facts (None = classical binary)",
    )

    # Epistemic context support
    epistemic_context: EpistemicContext | None = Field(
        default=None,
        description="Epistemic context (whose belief this is; None = objective fact)",
    )

    def is_uncertain(self) -> bool:
        """Check if this fact has explicit uncertainty."""
        return self.truth_value is not None

    def is_epistemic(self) -> bool:
        """Check if this fact is in an epistemic context (belief/knowledge)."""
        return self.epistemic_context is not None

    def effective_truth(self) -> bool:
        """Return effective boolean truth value.

        For uncertain facts, returns True if frequency >= 0.5.
        For classical facts, returns the negated flag inverted.
        """
        if self.truth_value is not None:
            return self.truth_value.is_positive() != self.negated
        return not self.negated


class QuantifiedVariable(BaseModel):
    """A variable in a quantified formula."""

    name: str = Field(..., description="Variable name (typically single letter)")
    type: str = Field(..., description="Type/sort of this variable")


class RuleCondition(BaseModel):
    """A condition in a rule (antecedent or consequent).

    Can be:
    - Simple predicate: {"predicate": "is_vegetarian", "arguments": ["x"]}
    - Negated: {"predicate": "contains", "arguments": ["f", "meat"], "negated": true}
    - Conjunction: {"and": [condition1, condition2, ...]}
    - Disjunction: {"or": [condition1, condition2, ...]}
    """

    model_config = {"populate_by_name": True}

    predicate: str | None = Field(default=None, description="Relation name")
    arguments: list[str] = Field(default_factory=list, description="Arguments")
    negated: bool = Field(default=False, description="Negate this condition")
    value: int | float | str | None = Field(
        default=None, description="For equality: the value"
    )
    and_: list[RuleCondition] = Field(
        default_factory=list, alias="and", description="Conjunction of conditions"
    )
    or_: list[RuleCondition] = Field(
        default_factory=list, alias="or", description="Disjunction of conditions"
    )

    def is_compound(self) -> bool:
        """Check if this is a compound (and/or) condition."""
        return bool(self.and_) or bool(self.or_)

    def is_simple(self) -> bool:
        """Check if this is a simple predicate condition."""
        return self.predicate is not None and not self.is_compound()


class Rule(BaseModel):
    """A universally quantified rule/implication.

    Rules encode general knowledge like:
    - forall p: is_vegetarian(p) => avoids_eating(p, Meat)
    - forall x,y: greater(x, y) => NOT greater(y, x)

    Can also be a simple constraint without quantification.
    """

    name: str | None = Field(default=None, description="Optional rule name")
    quantified_vars: list[QuantifiedVariable] = Field(
        default_factory=list, description="Universally quantified variables"
    )
    antecedent: RuleCondition | None = Field(
        default=None, description="If this is true..."
    )
    consequent: RuleCondition | None = Field(
        default=None, description="...then this must be true"
    )
    constraint: RuleCondition | None = Field(
        default=None,
        description="For non-implication rules: just a constraint to assert",
    )
    source: Literal["explicit", "implicit"] = Field(
        default="implicit", description="Whether rule is stated or derived"
    )
    justification: str | None = Field(
        default=None, description="Why this rule holds"
    )

    def is_implication(self) -> bool:
        """Check if this is an implication rule (antecedent => consequent)."""
        return self.antecedent is not None and self.consequent is not None

    def is_constraint(self) -> bool:
        """Check if this is a simple constraint."""
        return self.constraint is not None


class Query(BaseModel):
    """The query to check for satisfiability.

    The query represents what we're asking about:
    - would_eat(vegetarian_person, plant_burger)

    SAT result means the query is consistent with the knowledge base (True).
    UNSAT result means the query contradicts the knowledge base (False).
    """

    predicate: str = Field(..., description="Relation to query")
    arguments: list[str] = Field(..., description="Arguments to the predicate")
    negated: bool = Field(
        default=False, description="Query the negation of the predicate"
    )


class IKR(BaseModel):
    """Complete Intermediate Knowledge Representation.

    This is the top-level schema for representing a logical reasoning problem.
    The IKR is designed to be:
    1. Easy for LLMs to generate (structured, explicit)
    2. Deterministically compilable to SMT2
    3. Debuggable (each component is inspectable)

    Extended with:
    - epistemic_config: Configuration for epistemic reasoning (agents, axiom system)
    - kb_modules: Knowledge base modules to load
    """

    meta: Meta = Field(..., description="Question metadata")
    types: list[Type] = Field(default_factory=list, description="Type declarations")
    entities: list[Entity] = Field(
        default_factory=list, description="Named entities"
    )
    relations: list[Relation] = Field(
        default_factory=list, description="Relation declarations"
    )
    facts: list[Fact] = Field(
        default_factory=list, description="Ground fact assertions"
    )
    rules: list[Rule] = Field(
        default_factory=list, description="Universally quantified rules"
    )
    query: Query = Field(..., description="The property to check")

    # Epistemic reasoning configuration
    epistemic_config: EpistemicConfig | None = Field(
        default=None,
        description="Configuration for epistemic reasoning (agents, axiom system)",
    )

    # Knowledge base modules to load
    kb_modules: list[str] = Field(
        default_factory=list,
        description="Knowledge base modules to load (e.g., ['commonsense'])",
    )

    def get_explicit_facts(self) -> list[Fact]:
        """Get facts that are explicitly stated in the question."""
        return [f for f in self.facts if f.source == "explicit"]

    def get_background_facts(self) -> list[Fact]:
        """Get facts that are background/world knowledge."""
        return [f for f in self.facts if f.source == "background"]

    def get_kb_facts(self) -> list[Fact]:
        """Get facts that are from knowledge base modules."""
        return [f for f in self.facts if f.source == "kb"]

    def get_uncertain_facts(self) -> list[Fact]:
        """Get facts that have explicit uncertainty (truth values)."""
        return [f for f in self.facts if f.is_uncertain()]

    def get_epistemic_facts(self) -> list[Fact]:
        """Get facts that are in epistemic contexts (beliefs/knowledge)."""
        return [f for f in self.facts if f.is_epistemic()]

    def has_uncertainty(self) -> bool:
        """Check if any facts have uncertainty."""
        return any(f.is_uncertain() for f in self.facts)

    def has_epistemic_content(self) -> bool:
        """Check if IKR has epistemic content (agents or epistemic facts)."""
        return (
            self.epistemic_config is not None
            and len(self.epistemic_config.agents) > 0
        ) or any(f.is_epistemic() for f in self.facts)

    def get_type_names(self) -> set[str]:
        """Get all declared type names."""
        return {t.name for t in self.types}

    def get_entity_names(self) -> set[str]:
        """Get all declared entity names."""
        return {e.name for e in self.entities}

    def get_relation_names(self) -> set[str]:
        """Get all declared relation names."""
        return {r.name for r in self.relations}

    def validate_references(self) -> list[str]:
        """Validate that all references are to declared types/entities/relations.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        type_names = self.get_type_names() | {"Bool", "Int", "Real"}
        entity_names = self.get_entity_names()
        relation_names = self.get_relation_names()

        # Check entity types
        for entity in self.entities:
            if entity.type not in type_names:
                errors.append(
                    f"Entity '{entity.name}' references undefined type '{entity.type}'"
                )

        # Check relation signatures
        for relation in self.relations:
            for arg_type in relation.signature:
                if arg_type not in type_names:
                    errors.append(
                        f"Relation '{relation.name}' references undefined type '{arg_type}'"
                    )
            if relation.range not in type_names:
                errors.append(
                    f"Relation '{relation.name}' has undefined range type '{relation.range}'"
                )

        # Check fact predicates and arguments
        for fact in self.facts:
            if fact.predicate not in relation_names:
                errors.append(
                    f"Fact references undefined relation '{fact.predicate}'"
                )
            for arg in fact.arguments:
                if arg not in entity_names:
                    errors.append(
                        f"Fact '{fact.predicate}' references undefined entity '{arg}'"
                    )

        # Check query
        if self.query.predicate not in relation_names:
            errors.append(
                f"Query references undefined relation '{self.query.predicate}'"
            )
        for arg in self.query.arguments:
            if arg not in entity_names:
                errors.append(
                    f"Query references undefined entity '{arg}'"
                )

        return errors

    class Config:
        populate_by_name = True  # Allow using 'and' instead of 'and_' in JSON
