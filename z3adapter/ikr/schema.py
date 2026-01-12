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
    """

    predicate: str = Field(..., description="Relation name being asserted")
    arguments: list[str] = Field(..., description="Arguments (entity names)")
    negated: bool = Field(default=False, description="Whether this is a negative fact")
    value: int | float | str | None = Field(
        default=None,
        description="For functions: the value being asserted (e.g., age = 30)",
    )
    source: Literal["explicit", "background"] = Field(
        default="explicit",
        description="Whether fact is from question or world knowledge",
    )
    justification: str | None = Field(
        default=None,
        description="For background facts: why this is common knowledge",
    )


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

    def get_explicit_facts(self) -> list[Fact]:
        """Get facts that are explicitly stated in the question."""
        return [f for f in self.facts if f.source == "explicit"]

    def get_background_facts(self) -> list[Fact]:
        """Get facts that are background/world knowledge."""
        return [f for f in self.facts if f.source == "background"]

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
