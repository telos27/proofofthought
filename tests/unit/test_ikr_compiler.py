"""Unit tests for IKR to SMT2 compiler.

Note: These tests require z3-solver to be installed because the z3adapter
package imports z3 at the top level.
"""

import json
import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Skip if z3 is not available
try:
    from z3adapter.ikr.compiler import IKRCompiler, compile_ikr_to_smt2
    from z3adapter.ikr.schema import (
        Entity,
        Fact,
        IKR,
        Meta,
        QuantifiedVariable,
        Query,
        Relation,
        Rule,
        RuleCondition,
        Type,
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestIKRCompiler(unittest.TestCase):
    """Tests for the IKR to SMT2 compiler."""

    def setUp(self):
        self.compiler = IKRCompiler()
        self.simple_ikr = IKR(
            meta=Meta(question="Is the sky blue?"),
            types=[Type(name="Object")],
            entities=[Entity(name="sky", type="Object")],
            relations=[Relation(name="is_blue", signature=["Object"], range="Bool")],
            facts=[Fact(predicate="is_blue", arguments=["sky"])],
            rules=[],
            query=Query(predicate="is_blue", arguments=["sky"]),
        )

    def test_compile_simple_ikr(self):
        smt2 = self.compiler.compile(self.simple_ikr)

        # Check header
        self.assertIn("; IKR-generated SMT2 program", smt2)
        self.assertIn("Is the sky blue?", smt2)

        # Check sort declaration
        self.assertIn("(declare-sort Object 0)", smt2)

        # Check function declaration
        self.assertIn("(declare-fun is_blue (Object) Bool)", smt2)

        # Check constant declaration
        self.assertIn("(declare-const sky Object)", smt2)

        # Check fact assertion
        self.assertIn("(assert (is_blue sky))", smt2)

        # Check query assertion
        self.assertIn("; === Query ===", smt2)
        self.assertIn("(check-sat)", smt2)

    def test_compile_with_rules(self):
        """Test compilation with quantified rules."""
        ikr = IKR(
            meta=Meta(question="Would a vegetarian eat a plant burger?"),
            types=[Type(name="Person"), Type(name="Food")],
            entities=[
                Entity(name="vegetarian_person", type="Person"),
                Entity(name="plant_burger", type="Food"),
            ],
            relations=[
                Relation(name="is_vegetarian", signature=["Person"], range="Bool"),
                Relation(name="avoids_meat", signature=["Person"], range="Bool"),
            ],
            facts=[Fact(predicate="is_vegetarian", arguments=["vegetarian_person"])],
            rules=[
                Rule(
                    name="vegetarians avoid meat",
                    quantified_vars=[QuantifiedVariable(name="p", type="Person")],
                    antecedent=RuleCondition(predicate="is_vegetarian", arguments=["p"]),
                    consequent=RuleCondition(predicate="avoids_meat", arguments=["p"]),
                )
            ],
            query=Query(predicate="avoids_meat", arguments=["vegetarian_person"]),
        )

        smt2 = self.compiler.compile(ikr)

        # Check rule with forall quantifier
        self.assertIn("forall", smt2)
        self.assertIn("((p Person))", smt2)
        self.assertIn("(=> (is_vegetarian p) (avoids_meat p))", smt2)

    def test_compile_negated_condition(self):
        """Test compilation of negated conditions."""
        ikr = IKR(
            meta=Meta(question="Does food not contain meat?"),
            types=[Type(name="Food")],
            entities=[Entity(name="salad", type="Food")],
            relations=[Relation(name="contains_meat", signature=["Food"], range="Bool")],
            facts=[Fact(predicate="contains_meat", arguments=["salad"], negated=True)],
            rules=[],
            query=Query(predicate="contains_meat", arguments=["salad"], negated=True),
        )

        smt2 = self.compiler.compile(ikr)

        # Check negated fact
        self.assertIn("(assert (not (contains_meat salad)))", smt2)

    def test_compile_compound_antecedent(self):
        """Test compilation of rules with compound antecedents."""
        ikr = IKR(
            meta=Meta(question="Test compound condition"),
            types=[Type(name="Person"), Type(name="Food")],
            entities=[
                Entity(name="alice", type="Person"),
                Entity(name="pizza", type="Food"),
            ],
            relations=[
                Relation(name="is_hungry", signature=["Person"], range="Bool"),
                Relation(name="likes", signature=["Person", "Food"], range="Bool"),
                Relation(name="would_eat", signature=["Person", "Food"], range="Bool"),
            ],
            facts=[],
            rules=[
                Rule(
                    name="eating rule",
                    quantified_vars=[
                        QuantifiedVariable(name="p", type="Person"),
                        QuantifiedVariable(name="f", type="Food"),
                    ],
                    antecedent=RuleCondition(
                        and_=[
                            RuleCondition(predicate="is_hungry", arguments=["p"]),
                            RuleCondition(predicate="likes", arguments=["p", "f"]),
                        ]
                    ),
                    consequent=RuleCondition(predicate="would_eat", arguments=["p", "f"]),
                )
            ],
            query=Query(predicate="would_eat", arguments=["alice", "pizza"]),
        )

        smt2 = self.compiler.compile(ikr)

        # Check compound antecedent with AND
        self.assertIn("(and (is_hungry p) (likes p f))", smt2)
        self.assertIn("forall ((p Person) (f Food))", smt2)

    def test_compile_symmetric_relation(self):
        """Test compilation of symmetric relations."""
        ikr = IKR(
            meta=Meta(question="Does Alice know Bob?"),
            types=[Type(name="Person")],
            entities=[
                Entity(name="alice", type="Person"),
                Entity(name="bob", type="Person"),
            ],
            relations=[
                Relation(
                    name="knows", signature=["Person", "Person"], range="Bool", symmetric=True
                )
            ],
            facts=[Fact(predicate="knows", arguments=["alice", "bob"])],
            rules=[],
            query=Query(predicate="knows", arguments=["bob", "alice"]),
        )

        smt2 = self.compiler.compile(ikr)

        # Check symmetry axiom
        self.assertIn("(= (knows x y) (knows y x))", smt2)

    def test_compile_transitive_relation(self):
        """Test compilation of transitive relations."""
        ikr = IKR(
            meta=Meta(question="Is A greater than C?"),
            types=[Type(name="Number")],
            entities=[
                Entity(name="a", type="Number"),
                Entity(name="b", type="Number"),
                Entity(name="c", type="Number"),
            ],
            relations=[
                Relation(
                    name="greater_than",
                    signature=["Number", "Number"],
                    range="Bool",
                    transitive=True,
                )
            ],
            facts=[
                Fact(predicate="greater_than", arguments=["a", "b"]),
                Fact(predicate="greater_than", arguments=["b", "c"]),
            ],
            rules=[],
            query=Query(predicate="greater_than", arguments=["a", "c"]),
        )

        smt2 = self.compiler.compile(ikr)

        # Check transitivity axiom
        self.assertIn(
            "(=> (and (greater_than x y) (greater_than y z)) (greater_than x z))", smt2
        )

    def test_compile_function_fact(self):
        """Test compilation of function facts with values."""
        ikr = IKR(
            meta=Meta(question="Is Alice older than 18?"),
            types=[Type(name="Person")],
            entities=[Entity(name="alice", type="Person")],
            relations=[Relation(name="age", signature=["Person"], range="Int")],
            facts=[Fact(predicate="age", arguments=["alice"], value=25)],
            rules=[],
            query=Query(predicate="age", arguments=["alice"]),
        )

        smt2 = self.compiler.compile(ikr)

        # Check function equality assertion
        self.assertIn("(assert (= (age alice) 25))", smt2)

    def test_compile_validates_references(self):
        """Test that compilation fails with validation errors."""
        ikr = IKR(
            meta=Meta(question="Invalid IKR"),
            types=[],  # No types
            entities=[Entity(name="x", type="UndefinedType")],  # Undefined type
            relations=[],
            facts=[],
            rules=[],
            query=Query(predicate="undefined", arguments=["x"]),
        )

        with self.assertRaises(ValueError) as context:
            self.compiler.compile(ikr)
        self.assertIn("validation errors", str(context.exception).lower())

    def test_convenience_function(self):
        """Test the convenience compile function."""
        smt2 = compile_ikr_to_smt2(self.simple_ikr)
        self.assertIn("(declare-sort Object 0)", smt2)
        self.assertIn("(check-sat)", smt2)


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestIKRCompilerFromFixtures(unittest.TestCase):
    """Test IKR compilation from fixture files."""

    @property
    def fixtures_dir(self):
        return Path(__file__).parent.parent / "fixtures" / "ikr_examples"

    def setUp(self):
        self.compiler = IKRCompiler()

    def test_compile_simple_test(self):
        with open(self.fixtures_dir / "simple_test.json") as f:
            data = json.load(f)
        ikr = IKR.model_validate(data)
        smt2 = self.compiler.compile(ikr)

        # Verify valid SMT2 structure
        self.assertIn("(declare-sort Object 0)", smt2)
        self.assertIn("(declare-fun is_blue (Object) Bool)", smt2)
        self.assertIn("(declare-const sky Object)", smt2)
        self.assertIn("(assert (is_blue sky))", smt2)
        self.assertIn("(check-sat)", smt2)

    def test_compile_vegetarian_burger(self):
        with open(self.fixtures_dir / "vegetarian_burger.json") as f:
            data = json.load(f)
        ikr = IKR.model_validate(data)
        smt2 = self.compiler.compile(ikr)

        # Verify all types
        self.assertIn("(declare-sort Person 0)", smt2)
        self.assertIn("(declare-sort Food 0)", smt2)

        # Verify relations
        self.assertIn("(declare-fun is_vegetarian (Person) Bool)", smt2)
        self.assertIn("(declare-fun would_eat (Person Food) Bool)", smt2)

        # Verify entities
        self.assertIn("(declare-const vegetarian_person Person)", smt2)
        self.assertIn("(declare-const plant_burger Food)", smt2)

        # Verify rules (should have forall quantifiers)
        self.assertIn("forall", smt2)

        # Verify query
        self.assertIn("(would_eat vegetarian_person plant_burger)", smt2)
        self.assertIn("(check-sat)", smt2)


if __name__ == "__main__":
    unittest.main()
