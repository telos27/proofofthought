"""Unit tests for IKR to Souffle/Datalog compiler.

Note: These tests require z3-solver to be installed because the z3adapter
package imports z3 at the top level.
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Skip if z3 is not available
try:
    from z3adapter.ikr.souffle_compiler import (
        IKRSouffleCompiler,
        SouffleProgram,
        compile_ikr_to_souffle,
    )
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
class TestIKRSouffleCompiler(unittest.TestCase):
    """Tests for the IKR to Souffle compiler."""

    def setUp(self):
        self.compiler = IKRSouffleCompiler()
        self.simple_ikr = IKR(
            meta=Meta(question="Is the sky blue?"),
            types=[Type(name="Object")],
            entities=[Entity(name="sky", type="Object")],
            relations=[Relation(name="is_blue", signature=["Object"], range="Bool")],
            facts=[Fact(predicate="is_blue", arguments=["sky"])],
            rules=[],
            query=Query(predicate="is_blue", arguments=["sky"]),
        )

    def test_compile_returns_souffle_program(self):
        """Test that compile returns a SouffleProgram."""
        result = self.compiler.compile(self.simple_ikr)

        self.assertIsInstance(result, SouffleProgram)
        self.assertIsInstance(result.program, str)
        self.assertIsInstance(result.facts, dict)
        self.assertEqual(result.query_relation, "query_result")

    def test_compile_header(self):
        """Test that compiled program has proper header."""
        result = self.compiler.compile(self.simple_ikr)

        self.assertIn("// IKR-generated Souffle program", result.program)
        self.assertIn("Is the sky blue?", result.program)

    def test_compile_relation_declarations(self):
        """Test that relations are declared with .decl."""
        result = self.compiler.compile(self.simple_ikr)

        # Check relation declaration
        self.assertIn(".decl is_blue(x0: symbol)", result.program)

        # Check query result declaration
        self.assertIn(".decl query_result()", result.program)

    def test_compile_io_directives(self):
        """Test that input/output directives are generated."""
        result = self.compiler.compile(self.simple_ikr)

        self.assertIn(".input is_blue", result.program)
        self.assertIn(".output query_result", result.program)

    def test_compile_query_rule(self):
        """Test that query is compiled to derivation rule."""
        result = self.compiler.compile(self.simple_ikr)

        # Query: is_blue(sky) → query_result() :- is_blue(sky).
        self.assertIn("query_result() :- is_blue(sky).", result.program)

    def test_compile_facts_to_dict(self):
        """Test that facts are compiled to dictionary."""
        result = self.compiler.compile(self.simple_ikr)

        self.assertIn("is_blue", result.facts)
        self.assertEqual(result.facts["is_blue"], [("sky",)])

    def test_compile_multi_arg_relation(self):
        """Test compilation of multi-argument relations."""
        ikr = IKR(
            meta=Meta(question="Does Alice know Bob?"),
            types=[Type(name="Person")],
            entities=[
                Entity(name="alice", type="Person"),
                Entity(name="bob", type="Person"),
            ],
            relations=[
                Relation(name="knows", signature=["Person", "Person"], range="Bool")
            ],
            facts=[Fact(predicate="knows", arguments=["alice", "bob"])],
            rules=[],
            query=Query(predicate="knows", arguments=["alice", "bob"]),
        )

        result = self.compiler.compile(ikr)

        # Check declaration with two arguments
        self.assertIn(".decl knows(x0: symbol, x1: symbol)", result.program)

        # Check facts
        self.assertEqual(result.facts["knows"], [("alice", "bob")])

    def test_compile_with_rules(self):
        """Test compilation of implication rules to Horn clauses."""
        ikr = IKR(
            meta=Meta(question="Would a vegetarian eat a plant burger?"),
            types=[Type(name="Person"), Type(name="Food")],
            entities=[
                Entity(name="vegetarian_person", type="Person"),
                Entity(name="plant_burger", type="Food"),
            ],
            relations=[
                Relation(name="is_vegetarian", signature=["Person"], range="Bool"),
                Relation(name="is_plant_based", signature=["Food"], range="Bool"),
                Relation(name="would_eat", signature=["Person", "Food"], range="Bool"),
            ],
            facts=[
                Fact(predicate="is_vegetarian", arguments=["vegetarian_person"]),
                Fact(predicate="is_plant_based", arguments=["plant_burger"]),
            ],
            rules=[
                Rule(
                    name="vegetarians eat plants",
                    quantified_vars=[
                        QuantifiedVariable(name="p", type="Person"),
                        QuantifiedVariable(name="f", type="Food"),
                    ],
                    antecedent=RuleCondition(
                        and_=[
                            RuleCondition(predicate="is_vegetarian", arguments=["p"]),
                            RuleCondition(predicate="is_plant_based", arguments=["f"]),
                        ]
                    ),
                    consequent=RuleCondition(
                        predicate="would_eat", arguments=["p", "f"]
                    ),
                )
            ],
            query=Query(
                predicate="would_eat", arguments=["vegetarian_person", "plant_burger"]
            ),
        )

        result = self.compiler.compile(ikr)

        # Check rule compilation (consequent :- antecedent)
        self.assertIn("would_eat(p, f) :- is_vegetarian(p), is_plant_based(f).", result.program)

        # Check query
        self.assertIn(
            "query_result() :- would_eat(vegetarian_person, plant_burger).",
            result.program,
        )

    def test_compile_symmetric_relation(self):
        """Test compilation of symmetric relations."""
        ikr = IKR(
            meta=Meta(question="Does Bob know Alice?"),
            types=[Type(name="Person")],
            entities=[
                Entity(name="alice", type="Person"),
                Entity(name="bob", type="Person"),
            ],
            relations=[
                Relation(
                    name="knows",
                    signature=["Person", "Person"],
                    range="Bool",
                    symmetric=True,
                )
            ],
            facts=[Fact(predicate="knows", arguments=["alice", "bob"])],
            rules=[],
            query=Query(predicate="knows", arguments=["bob", "alice"]),
        )

        result = self.compiler.compile(ikr)

        # Check symmetry axiom: knows(X, Y) :- knows(Y, X).
        self.assertIn("knows(X, Y) :- knows(Y, X).", result.program)

    def test_compile_transitive_relation(self):
        """Test compilation of transitive relations."""
        ikr = IKR(
            meta=Meta(question="Is A ancestor of C?"),
            types=[Type(name="Person")],
            entities=[
                Entity(name="a", type="Person"),
                Entity(name="b", type="Person"),
                Entity(name="c", type="Person"),
            ],
            relations=[
                Relation(
                    name="ancestor_of",
                    signature=["Person", "Person"],
                    range="Bool",
                    transitive=True,
                )
            ],
            facts=[
                Fact(predicate="ancestor_of", arguments=["a", "b"]),
                Fact(predicate="ancestor_of", arguments=["b", "c"]),
            ],
            rules=[],
            query=Query(predicate="ancestor_of", arguments=["a", "c"]),
        )

        result = self.compiler.compile(ikr)

        # Check transitivity axiom
        self.assertIn(
            "ancestor_of(X, Z) :- ancestor_of(X, Y), ancestor_of(Y, Z).", result.program
        )

    def test_compile_negated_query(self):
        """Test compilation of negated queries."""
        ikr = IKR(
            meta=Meta(question="Is the sky not blue?"),
            types=[Type(name="Object")],
            entities=[Entity(name="sky", type="Object")],
            relations=[Relation(name="is_blue", signature=["Object"], range="Bool")],
            facts=[],
            rules=[],
            query=Query(predicate="is_blue", arguments=["sky"], negated=True),
        )

        result = self.compiler.compile(ikr)

        # Negated query uses stratified negation
        self.assertIn("query_result() :- !is_blue(sky).", result.program)

    def test_compile_function_relation(self):
        """Test compilation of function-style relations (with return value)."""
        ikr = IKR(
            meta=Meta(question="What is Alice's age?"),
            types=[Type(name="Person")],
            entities=[Entity(name="alice", type="Person")],
            relations=[Relation(name="age", signature=["Person"], range="Int")],
            facts=[Fact(predicate="age", arguments=["alice"], value=25)],
            rules=[],
            query=Query(predicate="age", arguments=["alice"]),
        )

        result = self.compiler.compile(ikr)

        # Function relation has extra argument for value
        self.assertIn(".decl age(x0: symbol, val: number)", result.program)

        # Fact includes value
        self.assertEqual(result.facts["age"], [("alice", "25")])

    def test_compile_validates_references(self):
        """Test that compilation fails with validation errors."""
        ikr = IKR(
            meta=Meta(question="Invalid IKR"),
            types=[],
            entities=[Entity(name="x", type="UndefinedType")],
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
        result = compile_ikr_to_souffle(self.simple_ikr)

        self.assertIsInstance(result, SouffleProgram)
        self.assertIn(".decl is_blue", result.program)


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestSouffleProgramWriting(unittest.TestCase):
    """Tests for writing Souffle programs to files."""

    def setUp(self):
        self.compiler = IKRSouffleCompiler()
        self.ikr = IKR(
            meta=Meta(question="Test question"),
            types=[Type(name="Thing")],
            entities=[
                Entity(name="a", type="Thing"),
                Entity(name="b", type="Thing"),
            ],
            relations=[
                Relation(name="related", signature=["Thing", "Thing"], range="Bool")
            ],
            facts=[
                Fact(predicate="related", arguments=["a", "b"]),
            ],
            rules=[],
            query=Query(predicate="related", arguments=["a", "b"]),
        )

    def test_write_program_creates_files(self):
        """Test that write_program creates .dl and .facts files."""
        import tempfile

        result = self.compiler.compile(self.ikr)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            program_path, facts_dir = self.compiler.write_program(result, output_dir)

            # Check program file exists
            self.assertTrue(program_path.exists())
            self.assertEqual(program_path.suffix, ".dl")

            # Check facts directory exists
            self.assertTrue(facts_dir.exists())
            self.assertTrue(facts_dir.is_dir())

            # Check facts file exists
            facts_file = facts_dir / "related.facts"
            self.assertTrue(facts_file.exists())

            # Check facts content
            content = facts_file.read_text()
            self.assertEqual(content.strip(), "a\tb")

    def test_write_program_content(self):
        """Test that written program file has correct content."""
        import tempfile

        result = self.compiler.compile(self.ikr)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            program_path, _ = self.compiler.write_program(result, output_dir)

            content = program_path.read_text()
            self.assertIn(".decl related", content)
            self.assertIn(".input related", content)
            self.assertIn(".output query_result", content)


if __name__ == "__main__":
    unittest.main()
