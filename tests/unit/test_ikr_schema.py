"""Unit tests for IKR schema validation.

Note: These tests require z3-solver to be installed because the z3adapter
package imports z3 at the top level.
"""

import json
import sys
import unittest
from pathlib import Path

from pydantic import ValidationError

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Skip if z3 is not available
try:
    from z3adapter.ikr.schema import (
        Entity,
        Fact,
        IKR,
        Meta,
        QuantifiedVariable,
        Query,
        QuestionType,
        Relation,
        Rule,
        RuleCondition,
        Type,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestType(unittest.TestCase):
    """Tests for Type model."""

    def test_valid_type(self):
        t = Type(name="Person", description="A human individual")
        self.assertEqual(t.name, "Person")
        self.assertEqual(t.description, "A human individual")

    def test_type_without_description(self):
        t = Type(name="Food")
        self.assertEqual(t.name, "Food")
        self.assertIsNone(t.description)

    def test_invalid_type_name(self):
        with self.assertRaises(ValidationError):
            Type(name="123invalid")

    def test_type_name_with_spaces(self):
        with self.assertRaises(ValidationError):
            Type(name="My Type")


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestEntity(unittest.TestCase):
    """Tests for Entity model."""

    def test_valid_entity(self):
        e = Entity(name="alice", type="Person", aliases=["Alice", "a person"])
        self.assertEqual(e.name, "alice")
        self.assertEqual(e.type, "Person")
        self.assertEqual(e.aliases, ["Alice", "a person"])

    def test_entity_snake_case(self):
        e = Entity(name="vegetarian_person", type="Person")
        self.assertEqual(e.name, "vegetarian_person")

    def test_entity_without_aliases(self):
        e = Entity(name="bob", type="Person")
        self.assertEqual(e.aliases, [])


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestRelation(unittest.TestCase):
    """Tests for Relation model."""

    def test_predicate_relation(self):
        r = Relation(name="is_vegetarian", signature=["Person"], range="Bool")
        self.assertEqual(r.name, "is_vegetarian")
        self.assertEqual(r.signature, ["Person"])
        self.assertEqual(r.range, "Bool")
        self.assertFalse(r.symmetric)
        self.assertFalse(r.transitive)

    def test_binary_predicate(self):
        r = Relation(name="would_eat", signature=["Person", "Food"], range="Bool")
        self.assertEqual(len(r.signature), 2)

    def test_function_relation(self):
        r = Relation(name="age", signature=["Person"], range="Int")
        self.assertEqual(r.range, "Int")

    def test_symmetric_relation(self):
        r = Relation(name="knows", signature=["Person", "Person"], range="Bool", symmetric=True)
        self.assertTrue(r.symmetric)

    def test_transitive_relation(self):
        r = Relation(
            name="ancestor_of", signature=["Person", "Person"], range="Bool", transitive=True
        )
        self.assertTrue(r.transitive)


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestFact(unittest.TestCase):
    """Tests for Fact model."""

    def test_simple_fact(self):
        f = Fact(predicate="is_vegetarian", arguments=["alice"])
        self.assertEqual(f.predicate, "is_vegetarian")
        self.assertEqual(f.arguments, ["alice"])
        self.assertFalse(f.negated)
        self.assertEqual(f.source, "explicit")

    def test_negated_fact(self):
        f = Fact(predicate="contains_meat", arguments=["salad"], negated=True)
        self.assertTrue(f.negated)

    def test_background_fact(self):
        f = Fact(
            predicate="is_healthy",
            arguments=["vegetables"],
            source="background",
            justification="Common knowledge about nutrition",
        )
        self.assertEqual(f.source, "background")
        self.assertIsNotNone(f.justification)

    def test_fact_with_value(self):
        f = Fact(predicate="age", arguments=["alice"], value=30)
        self.assertEqual(f.value, 30)


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestRuleCondition(unittest.TestCase):
    """Tests for RuleCondition model."""

    def test_simple_condition(self):
        c = RuleCondition(predicate="is_vegetarian", arguments=["p"])
        self.assertTrue(c.is_simple())
        self.assertFalse(c.is_compound())

    def test_negated_condition(self):
        c = RuleCondition(predicate="contains_meat", arguments=["f"], negated=True)
        self.assertTrue(c.negated)

    def test_conjunction(self):
        c = RuleCondition(
            and_=[
                RuleCondition(predicate="is_vegetarian", arguments=["p"]),
                RuleCondition(predicate="is_hungry", arguments=["p"]),
            ]
        )
        self.assertTrue(c.is_compound())
        self.assertEqual(len(c.and_), 2)

    def test_disjunction(self):
        c = RuleCondition(
            or_=[
                RuleCondition(predicate="likes_pizza", arguments=["p"]),
                RuleCondition(predicate="likes_pasta", arguments=["p"]),
            ]
        )
        self.assertTrue(c.is_compound())
        self.assertEqual(len(c.or_), 2)


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestRule(unittest.TestCase):
    """Tests for Rule model."""

    def test_implication_rule(self):
        r = Rule(
            name="vegetarians avoid meat",
            quantified_vars=[QuantifiedVariable(name="p", type="Person")],
            antecedent=RuleCondition(predicate="is_vegetarian", arguments=["p"]),
            consequent=RuleCondition(predicate="avoids_meat", arguments=["p"]),
            justification="Definition of vegetarian",
        )
        self.assertTrue(r.is_implication())
        self.assertFalse(r.is_constraint())
        self.assertEqual(len(r.quantified_vars), 1)

    def test_constraint_rule(self):
        r = Rule(
            constraint=RuleCondition(predicate="positive", arguments=["x"]),
            quantified_vars=[QuantifiedVariable(name="x", type="Int")],
        )
        self.assertTrue(r.is_constraint())
        self.assertFalse(r.is_implication())


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestQuery(unittest.TestCase):
    """Tests for Query model."""

    def test_simple_query(self):
        q = Query(predicate="would_eat", arguments=["alice", "salad"])
        self.assertEqual(q.predicate, "would_eat")
        self.assertEqual(q.arguments, ["alice", "salad"])
        self.assertFalse(q.negated)

    def test_negated_query(self):
        q = Query(predicate="is_healthy", arguments=["soda"], negated=True)
        self.assertTrue(q.negated)


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestIKR(unittest.TestCase):
    """Tests for complete IKR model."""

    def test_minimal_ikr(self):
        ikr = IKR(
            meta=Meta(question="Is the sky blue?"),
            types=[Type(name="Object")],
            entities=[Entity(name="sky", type="Object")],
            relations=[Relation(name="is_blue", signature=["Object"], range="Bool")],
            facts=[Fact(predicate="is_blue", arguments=["sky"])],
            rules=[],
            query=Query(predicate="is_blue", arguments=["sky"]),
        )
        self.assertEqual(ikr.meta.question, "Is the sky blue?")
        self.assertEqual(len(ikr.types), 1)
        self.assertEqual(len(ikr.entities), 1)

    def test_validate_references_valid(self):
        ikr = IKR(
            meta=Meta(question="Test"),
            types=[Type(name="Person")],
            entities=[Entity(name="alice", type="Person")],
            relations=[Relation(name="is_happy", signature=["Person"], range="Bool")],
            facts=[Fact(predicate="is_happy", arguments=["alice"])],
            rules=[],
            query=Query(predicate="is_happy", arguments=["alice"]),
        )
        errors = ikr.validate_references()
        self.assertEqual(errors, [])

    def test_validate_references_undefined_type(self):
        ikr = IKR(
            meta=Meta(question="Test"),
            types=[],  # No types defined
            entities=[Entity(name="alice", type="Person")],  # References undefined type
            relations=[],
            facts=[],
            rules=[],
            query=Query(predicate="is_happy", arguments=["alice"]),
        )
        errors = ikr.validate_references()
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("undefined type" in e.lower() for e in errors))

    def test_validate_references_undefined_entity(self):
        ikr = IKR(
            meta=Meta(question="Test"),
            types=[Type(name="Person")],
            entities=[],  # No entities defined
            relations=[Relation(name="is_happy", signature=["Person"], range="Bool")],
            facts=[Fact(predicate="is_happy", arguments=["alice"])],  # Undefined entity
            rules=[],
            query=Query(predicate="is_happy", arguments=["bob"]),  # Undefined entity
        )
        errors = ikr.validate_references()
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("undefined entity" in e.lower() for e in errors))

    def test_get_explicit_facts(self):
        ikr = IKR(
            meta=Meta(question="Test"),
            types=[Type(name="Thing")],
            entities=[Entity(name="x", type="Thing")],
            relations=[Relation(name="p", signature=["Thing"], range="Bool")],
            facts=[
                Fact(predicate="p", arguments=["x"], source="explicit"),
                Fact(predicate="p", arguments=["x"], source="background"),
            ],
            rules=[],
            query=Query(predicate="p", arguments=["x"]),
        )
        explicit = ikr.get_explicit_facts()
        background = ikr.get_background_facts()
        self.assertEqual(len(explicit), 1)
        self.assertEqual(len(background), 1)


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestIKRFromFixtures(unittest.TestCase):
    """Test IKR parsing from fixture files."""

    @property
    def fixtures_dir(self):
        return Path(__file__).parent.parent / "fixtures" / "ikr_examples"

    def test_parse_simple_test(self):
        with open(self.fixtures_dir / "simple_test.json") as f:
            data = json.load(f)
        ikr = IKR.model_validate(data)
        self.assertEqual(ikr.meta.question, "Is the sky blue?")
        self.assertEqual(len(ikr.entities), 1)
        errors = ikr.validate_references()
        self.assertEqual(errors, [])

    def test_parse_vegetarian_burger(self):
        with open(self.fixtures_dir / "vegetarian_burger.json") as f:
            data = json.load(f)
        ikr = IKR.model_validate(data)
        self.assertIn("vegetarian", ikr.meta.question.lower())
        self.assertEqual(len(ikr.rules), 3)
        errors = ikr.validate_references()
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
