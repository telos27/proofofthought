"""Unit tests for NARS-Datalog engine.

Tests cover:
- Truth functions (conjunction, deduction)
- Fact storage with revision
- Variable unification
- Rule compilation
- Inference engine
- Backend integration
"""

import json
import tempfile
from pathlib import Path

import pytest

from z3adapter.ikr.schema import (
    IKR,
    Meta,
    QuestionType,
    Type,
    Entity,
    Relation,
    Fact,
    Rule,
    RuleCondition,
    QuantifiedVariable,
    Query,
    TruthValue,
)
from z3adapter.ikr.nars_datalog import (
    # Truth functions
    conjunction,
    deduction,
    DEFAULT_RULE_TRUTH,
    # Fact storage
    GroundAtom,
    StoredFact,
    FactStore,
    # Unification
    RuleAtom,
    is_variable,
    unify_atom_with_fact,
    # Rules
    InternalRule,
    compile_rules,
    # Engine
    NARSDatalogEngine,
    InferenceResult,
    from_ikr,
    # Knowledge Base
    KBLoader,
    KB_DIR,
)


# ==============================================================================
# Truth Functions Tests
# ==============================================================================


class TestConjunction:
    """Test NARS conjunction (intersection) for rule bodies."""

    def test_empty_conjunction(self):
        """Empty conjunction is vacuously true."""
        result = conjunction([])
        assert result.frequency == 1.0
        assert result.confidence == 0.9

    def test_single_element(self):
        """Single element conjunction preserves truth value."""
        tv = TruthValue(frequency=0.8, confidence=0.7)
        result = conjunction([tv])
        assert result.frequency == 0.8
        assert result.confidence == 0.7

    def test_two_elements(self):
        """Two element conjunction multiplies frequencies and confidences."""
        tv1 = TruthValue(frequency=0.8, confidence=0.9)
        tv2 = TruthValue(frequency=0.5, confidence=0.8)
        result = conjunction([tv1, tv2])
        assert result.frequency == pytest.approx(0.4, rel=0.01)
        assert result.confidence == pytest.approx(0.72, rel=0.01)

    def test_confidence_clamping(self):
        """Very low confidences are clamped to minimum."""
        tv1 = TruthValue(frequency=0.5, confidence=0.1)
        tv2 = TruthValue(frequency=0.5, confidence=0.05)
        result = conjunction([tv1, tv2])
        assert result.confidence >= 0.001
        assert result.confidence < 1.0


class TestDeduction:
    """Test NARS deduction for rule application."""

    def test_certain_premise_and_rule(self):
        """Certain premise with certain rule gives certain conclusion."""
        premise = TruthValue(frequency=1.0, confidence=0.9)
        rule = TruthValue(frequency=1.0, confidence=0.9)
        result = deduction(premise, rule)
        assert result.frequency == 1.0
        assert result.confidence == pytest.approx(0.81, rel=0.01)

    def test_uncertain_premise(self):
        """Uncertain premise reduces conclusion frequency."""
        premise = TruthValue(frequency=0.5, confidence=0.9)
        result = deduction(premise)  # Default rule truth
        assert result.frequency == pytest.approx(0.5, rel=0.01)

    def test_confidence_degradation(self):
        """Confidence degrades through inference chain."""
        premise = TruthValue(frequency=0.8, confidence=0.9)
        result = deduction(premise)
        # Confidence should be lower than premise
        assert result.confidence < premise.confidence

    def test_default_rule_truth(self):
        """Default rule truth is used when not specified."""
        premise = TruthValue(frequency=0.9, confidence=0.8)
        result = deduction(premise)
        # Should use DEFAULT_RULE_TRUTH
        assert result.frequency == pytest.approx(0.9, rel=0.01)


# ==============================================================================
# Fact Storage Tests
# ==============================================================================


class TestGroundAtom:
    """Test GroundAtom hashable representation."""

    def test_equality(self):
        """Equal atoms are equal."""
        a1 = GroundAtom("parent", ("tom", "mary"), False)
        a2 = GroundAtom("parent", ("tom", "mary"), False)
        assert a1 == a2
        assert hash(a1) == hash(a2)

    def test_inequality_predicate(self):
        """Different predicates are not equal."""
        a1 = GroundAtom("parent", ("tom", "mary"), False)
        a2 = GroundAtom("child", ("tom", "mary"), False)
        assert a1 != a2

    def test_inequality_arguments(self):
        """Different arguments are not equal."""
        a1 = GroundAtom("parent", ("tom", "mary"), False)
        a2 = GroundAtom("parent", ("tom", "bob"), False)
        assert a1 != a2

    def test_positive_method(self):
        """positive() returns non-negated version."""
        neg = GroundAtom("likes", ("a", "b"), True)
        pos = neg.positive()
        assert pos.negated is False
        assert pos.predicate == "likes"
        assert pos.arguments == ("a", "b")


class TestFactStore:
    """Test indexed fact storage with revision."""

    def test_add_and_get(self):
        """Add a fact and retrieve it."""
        store = FactStore()
        atom = GroundAtom("parent", ("tom", "mary"), False)
        tv = TruthValue(frequency=1.0, confidence=0.9)

        is_new = store.add(atom, tv)

        assert is_new is True
        stored = store.get(atom)
        assert stored is not None
        assert stored.truth_value.frequency == 1.0

    def test_revision_on_duplicate(self):
        """Adding duplicate fact revises truth value."""
        store = FactStore()
        atom = GroundAtom("likes", ("a", "b"), False)

        tv1 = TruthValue(frequency=0.8, confidence=0.5)
        tv2 = TruthValue(frequency=0.8, confidence=0.5)

        is_new1 = store.add(atom, tv1)
        is_new2 = store.add(atom, tv2)

        assert is_new1 is True
        assert is_new2 is False

        stored = store.get(atom)
        # Revised confidence should be higher
        assert stored.truth_value.confidence > 0.5
        assert stored.derivation_count == 2

    def test_get_by_predicate(self):
        """Get all facts with given predicate."""
        store = FactStore()
        store.add(GroundAtom("parent", ("tom", "mary"), False), TruthValue())
        store.add(GroundAtom("parent", ("tom", "bob"), False), TruthValue())
        store.add(GroundAtom("child", ("mary", "tom"), False), TruthValue())

        parents = list(store.get_by_predicate("parent"))
        assert len(parents) == 2

    def test_contains(self):
        """Check fact existence."""
        store = FactStore()
        atom = GroundAtom("test", ("x",), False)
        store.add(atom, TruthValue())

        assert store.contains(atom)
        assert not store.contains(GroundAtom("test", ("y",), False))


# ==============================================================================
# Unification Tests
# ==============================================================================


class TestIsVariable:
    """Test variable detection."""

    def test_uppercase_is_variable(self):
        """Uppercase first letter is a variable."""
        assert is_variable("X")
        assert is_variable("Person")
        assert is_variable("Var")

    def test_lowercase_is_constant(self):
        """Lowercase first letter is a constant."""
        assert not is_variable("tom")
        assert not is_variable("mary")
        assert not is_variable("plant_burger")

    def test_empty_string(self):
        """Empty string is not a variable."""
        assert not is_variable("")


class TestUnification:
    """Test variable unification."""

    def test_ground_match(self):
        """Ground atoms match exactly."""
        rule_atom = RuleAtom("parent", ("tom", "mary"), False)
        fact_atom = GroundAtom("parent", ("tom", "mary"), False)

        bindings = unify_atom_with_fact(rule_atom, fact_atom, {})

        assert bindings is not None
        assert bindings == {}  # No variables bound

    def test_variable_binding(self):
        """Variables are bound to constants."""
        rule_atom = RuleAtom("parent", ("X", "Y"), False)
        fact_atom = GroundAtom("parent", ("tom", "mary"), False)

        bindings = unify_atom_with_fact(rule_atom, fact_atom, {})

        assert bindings == {"X": "tom", "Y": "mary"}

    def test_existing_binding_match(self):
        """Existing bindings must match."""
        rule_atom = RuleAtom("parent", ("X", "Y"), False)
        fact_atom = GroundAtom("parent", ("tom", "mary"), False)
        existing = {"X": "tom"}

        bindings = unify_atom_with_fact(rule_atom, fact_atom, existing)

        assert bindings == {"X": "tom", "Y": "mary"}

    def test_binding_conflict(self):
        """Conflicting bindings fail."""
        rule_atom = RuleAtom("parent", ("X", "Y"), False)
        fact_atom = GroundAtom("parent", ("tom", "mary"), False)
        existing = {"X": "bob"}  # Conflicts with "tom"

        bindings = unify_atom_with_fact(rule_atom, fact_atom, existing)

        assert bindings is None

    def test_predicate_mismatch(self):
        """Different predicates don't unify."""
        rule_atom = RuleAtom("parent", ("X",), False)
        fact_atom = GroundAtom("child", ("tom",), False)

        bindings = unify_atom_with_fact(rule_atom, fact_atom, {})

        assert bindings is None


# ==============================================================================
# Rule Compilation Tests
# ==============================================================================


class TestInternalRule:
    """Test rule compilation from IKR."""

    def test_simple_rule(self):
        """Simple implication compiles correctly."""
        ikr_rule = Rule(
            name="test_rule",
            quantified_vars=[QuantifiedVariable(name="X", type="Person")],
            antecedent=RuleCondition(predicate="human", arguments=["X"]),
            consequent=RuleCondition(predicate="mortal", arguments=["X"]),
        )

        internal = InternalRule.from_ikr_rule(ikr_rule)

        assert internal is not None
        assert internal.name == "test_rule"
        assert internal.head.predicate == "mortal"
        assert internal.head.arguments == ("X",)
        assert len(internal.body) == 1
        assert internal.body[0].predicate == "human"

    def test_conjunction_body(self):
        """Conjunction in antecedent creates multiple body atoms."""
        ikr_rule = Rule(
            quantified_vars=[
                QuantifiedVariable(name="X", type="Person"),
                QuantifiedVariable(name="F", type="Food"),
            ],
            antecedent=RuleCondition(
                and_=[
                    RuleCondition(predicate="vegetarian", arguments=["X"]),
                    RuleCondition(predicate="meatfree", arguments=["F"]),
                ]
            ),
            consequent=RuleCondition(predicate="would_eat", arguments=["X", "F"]),
        )

        internal = InternalRule.from_ikr_rule(ikr_rule)

        assert internal is not None
        assert len(internal.body) == 2

    def test_negated_body_atom(self):
        """Negated atoms in body are detected."""
        ikr_rule = Rule(
            quantified_vars=[QuantifiedVariable(name="X", type="Food")],
            antecedent=RuleCondition(
                predicate="contains_meat", arguments=["X"], negated=True
            ),
            consequent=RuleCondition(predicate="vegetarian_food", arguments=["X"]),
        )

        internal = InternalRule.from_ikr_rule(ikr_rule)

        assert internal is not None
        assert internal.has_negation is True


# ==============================================================================
# Engine Tests
# ==============================================================================


class TestNARSDatalogEngine:
    """Test the main inference engine."""

    def create_simple_ikr(self) -> IKR:
        """Create a simple IKR for testing."""
        return IKR(
            meta=Meta(question="Is Socrates mortal?", question_type=QuestionType.YES_NO),
            types=[Type(name="Person")],
            entities=[Entity(name="socrates", type="Person")],
            relations=[
                Relation(name="human", signature=["Person"]),
                Relation(name="mortal", signature=["Person"]),
            ],
            facts=[
                Fact(predicate="human", arguments=["socrates"]),
            ],
            rules=[
                Rule(
                    name="mortality",
                    quantified_vars=[QuantifiedVariable(name="X", type="Person")],
                    antecedent=RuleCondition(predicate="human", arguments=["X"]),
                    consequent=RuleCondition(predicate="mortal", arguments=["X"]),
                )
            ],
            query=Query(predicate="mortal", arguments=["socrates"]),
        )

    def test_load_facts(self):
        """Facts are loaded from IKR."""
        ikr = self.create_simple_ikr()
        engine = NARSDatalogEngine()
        engine.load_ikr(ikr)

        assert engine.fact_store.size() == 1
        atom = GroundAtom("human", ("socrates",), False)
        assert engine.fact_store.contains(atom)

    def test_load_rules(self):
        """Rules are compiled from IKR."""
        ikr = self.create_simple_ikr()
        engine = NARSDatalogEngine()
        engine.load_ikr(ikr)

        assert len(engine.rules) == 1

    def test_simple_derivation(self):
        """Simple one-step derivation works."""
        ikr = self.create_simple_ikr()
        engine = from_ikr(ikr)
        result = engine.query(ikr.query)

        assert result.found is True
        assert result.truth_value is not None
        assert result.truth_value.frequency > 0.5

    def test_chain_derivation(self):
        """Multi-step derivation chain works."""
        ikr = IKR(
            meta=Meta(question="Is A related to C?", question_type=QuestionType.YES_NO),
            types=[Type(name="Thing")],
            entities=[
                Entity(name="a", type="Thing"),
                Entity(name="b", type="Thing"),
                Entity(name="c", type="Thing"),
            ],
            relations=[
                Relation(name="connected", signature=["Thing", "Thing"], transitive=True),
            ],
            facts=[
                Fact(predicate="connected", arguments=["a", "b"]),
                Fact(predicate="connected", arguments=["b", "c"]),
            ],
            rules=[
                # Transitivity rule
                Rule(
                    name="transitivity",
                    quantified_vars=[
                        QuantifiedVariable(name="X", type="Thing"),
                        QuantifiedVariable(name="Y", type="Thing"),
                        QuantifiedVariable(name="Z", type="Thing"),
                    ],
                    antecedent=RuleCondition(
                        and_=[
                            RuleCondition(predicate="connected", arguments=["X", "Y"]),
                            RuleCondition(predicate="connected", arguments=["Y", "Z"]),
                        ]
                    ),
                    consequent=RuleCondition(predicate="connected", arguments=["X", "Z"]),
                )
            ],
            query=Query(predicate="connected", arguments=["a", "c"]),
        )

        engine = from_ikr(ikr)
        result = engine.query(ikr.query)

        assert result.found is True
        # Truth value degrades through chain
        assert result.truth_value.confidence < 0.9

    def test_truth_propagation(self):
        """Truth values propagate through inference."""
        ikr = IKR(
            meta=Meta(question="Test", question_type=QuestionType.YES_NO),
            types=[Type(name="T")],
            entities=[Entity(name="x", type="T")],
            relations=[
                Relation(name="p", signature=["T"]),
                Relation(name="q", signature=["T"]),
            ],
            facts=[
                Fact(
                    predicate="p",
                    arguments=["x"],
                    truth_value=TruthValue(frequency=0.8, confidence=0.9),
                ),
            ],
            rules=[
                Rule(
                    quantified_vars=[QuantifiedVariable(name="X", type="T")],
                    antecedent=RuleCondition(predicate="p", arguments=["X"]),
                    consequent=RuleCondition(predicate="q", arguments=["X"]),
                )
            ],
            query=Query(predicate="q", arguments=["x"]),
        )

        engine = from_ikr(ikr)
        result = engine.query(ikr.query)

        assert result.found is True
        # Derived truth should reflect uncertainty
        assert result.truth_value.frequency == pytest.approx(0.8, rel=0.01)

    def test_negation_as_failure(self):
        """Negation-as-failure works for underivable atoms."""
        ikr = IKR(
            meta=Meta(question="Test NAF", question_type=QuestionType.YES_NO),
            types=[Type(name="T")],
            entities=[Entity(name="x", type="T")],
            relations=[Relation(name="p", signature=["T"])],
            facts=[],  # No facts about p
            rules=[],
            query=Query(predicate="p", arguments=["x"], negated=True),
        )

        engine = from_ikr(ikr)
        result = engine.query(ikr.query)

        # NOT p(x) should be true since p(x) is not derivable
        assert result.found is True

    def test_fixpoint_termination(self):
        """Engine terminates at fixpoint."""
        ikr = self.create_simple_ikr()
        engine = from_ikr(ikr)

        iterations = engine.run()

        assert iterations < engine.max_iterations
        assert iterations >= 1


class TestVegetarianBurger:
    """Integration test with vegetarian burger example."""

    def test_vegetarian_would_eat_plant_burger(self):
        """Classic example: vegetarian would eat plant burger."""
        ikr = IKR(
            meta=Meta(
                question="Would a vegetarian eat a plant burger?",
                question_type=QuestionType.YES_NO,
            ),
            types=[
                Type(name="Person"),
                Type(name="Food"),
            ],
            entities=[
                Entity(name="vegetarian_person", type="Person"),
                Entity(name="plant_burger", type="Food"),
            ],
            relations=[
                Relation(name="is_vegetarian", signature=["Person"]),
                Relation(name="contains_meat", signature=["Food"]),
                Relation(name="would_eat", signature=["Person", "Food"]),
            ],
            facts=[
                Fact(predicate="is_vegetarian", arguments=["vegetarian_person"]),
                Fact(predicate="contains_meat", arguments=["plant_burger"], negated=True),
            ],
            rules=[
                Rule(
                    name="vegetarians_eat_meatfree",
                    quantified_vars=[
                        QuantifiedVariable(name="P", type="Person"),
                        QuantifiedVariable(name="F", type="Food"),
                    ],
                    antecedent=RuleCondition(
                        and_=[
                            RuleCondition(predicate="is_vegetarian", arguments=["P"]),
                            RuleCondition(
                                predicate="contains_meat", arguments=["F"], negated=True
                            ),
                        ]
                    ),
                    consequent=RuleCondition(predicate="would_eat", arguments=["P", "F"]),
                )
            ],
            query=Query(
                predicate="would_eat", arguments=["vegetarian_person", "plant_burger"]
            ),
        )

        engine = from_ikr(ikr)
        result = engine.query(ikr.query)

        assert result.found is True
        assert result.truth_value.frequency > 0.5
        assert result.explanation  # Should have explanation


# ==============================================================================
# Backend Tests
# ==============================================================================


class TestNARSDatalogBackend:
    """Test backend integration."""

    def test_execute_simple_ikr(self):
        """Backend executes simple IKR correctly."""
        from z3adapter.backends import NARSDatalogBackend

        ikr_data = {
            "meta": {"question": "Is socrates mortal?", "question_type": "yes_no"},
            "types": [{"name": "Person"}],
            "entities": [{"name": "socrates", "type": "Person"}],
            "relations": [
                {"name": "human", "signature": ["Person"]},
                {"name": "mortal", "signature": ["Person"]},
            ],
            "facts": [{"predicate": "human", "arguments": ["socrates"]}],
            "rules": [
                {
                    "name": "mortality",
                    "quantified_vars": [{"name": "X", "type": "Person"}],
                    "antecedent": {"predicate": "human", "arguments": ["X"]},
                    "consequent": {"predicate": "mortal", "arguments": ["X"]},
                }
            ],
            "query": {"predicate": "mortal", "arguments": ["socrates"]},
        }

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(ikr_data, f)
            temp_path = f.name

        try:
            backend = NARSDatalogBackend()
            result = backend.execute(temp_path)

            assert result.success is True
            assert result.answer is True
            assert "NARS-Datalog" in result.output
        finally:
            Path(temp_path).unlink()

    def test_invalid_json(self):
        """Backend handles invalid JSON gracefully."""
        from z3adapter.backends import NARSDatalogBackend

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("not valid json")
            temp_path = f.name

        try:
            backend = NARSDatalogBackend()
            result = backend.execute(temp_path)

            assert result.success is False
            assert result.error is not None
            assert "JSON" in result.error
        finally:
            Path(temp_path).unlink()


# ==============================================================================
# Knowledge Base Tests
# ==============================================================================


class TestKBLoader:
    """Test knowledge base loading."""

    def test_available_modules(self):
        """Can list available KB modules."""
        modules = KBLoader.available_modules()
        assert isinstance(modules, list)
        # Should have our test modules
        assert "biology" in modules
        assert "psychology" in modules
        assert "commonsense" in modules

    def test_load_biology_module(self):
        """Can load biology KB module."""
        engine = NARSDatalogEngine()
        stats = KBLoader.load_modules(engine, ["biology"])

        assert "biology" in stats
        assert stats["biology"] > 0
        # Should have loaded rules
        assert len(engine.rules) > 0

    def test_load_multiple_modules(self):
        """Can load multiple KB modules."""
        engine = NARSDatalogEngine()
        stats = KBLoader.load_modules(engine, ["biology", "psychology"])

        assert stats["biology"] > 0
        assert stats["psychology"] > 0

    def test_rule_truth_values(self):
        """KB rules preserve truth values."""
        engine = NARSDatalogEngine()
        KBLoader.load_modules(engine, ["biology"])

        # Find the "birds_typically_fly" rule
        bird_rule = None
        for rule in engine.rules:
            if rule.name == "birds_typically_fly":
                bird_rule = rule
                break

        assert bird_rule is not None
        # Should have defeasible truth value
        assert bird_rule.rule_truth.frequency < 1.0
        assert bird_rule.rule_truth.frequency > 0.8

    def test_get_module_info(self):
        """Can get module metadata."""
        info = KBLoader.get_module_info("psychology")

        assert info["name"] == "psychology"
        assert "rules_count" in info
        assert info["rules_count"] > 0

    def test_missing_module(self):
        """Handles missing module gracefully."""
        engine = NARSDatalogEngine()
        stats = KBLoader.load_modules(engine, ["nonexistent_module"])

        assert stats["nonexistent_module"] == 0


# =============================================================================
# Truth Strategy Tests
# =============================================================================


class TestTruthStrategies:
    """Test pluggable truth function strategies."""

    def test_get_strategy_current(self):
        """Can get the current (default) strategy."""
        from z3adapter.ikr.nars_datalog.truth_strategies import get_strategy

        strategy = get_strategy("current")
        assert strategy is not None

    def test_get_strategy_opennars(self):
        """Can get the OpenNARS strategy."""
        from z3adapter.ikr.nars_datalog.truth_strategies import get_strategy

        strategy = get_strategy("opennars")
        assert strategy is not None

    def test_opennars_confidence_independent_of_frequency(self):
        """OpenNARS confidence should not depend on frequency."""
        from z3adapter.ikr.nars_datalog.truth_strategies import get_strategy

        strategy = get_strategy("opennars")
        premise = TruthValue(frequency=0.5, confidence=0.9)
        rule = TruthValue(frequency=1.0, confidence=0.9)

        result = strategy.deduction(premise, rule)
        # Confidence should be 0.9 * 0.9 = 0.81, not affected by f=0.5
        assert abs(result.confidence - 0.81) < 0.01

    def test_floor_strategy_prevents_low_confidence(self):
        """Floor strategy should maintain minimum confidence."""
        from z3adapter.ikr.nars_datalog.truth_strategies import FloorStrategy

        strategy = FloorStrategy(floor=0.2)
        premise = TruthValue(frequency=0.1, confidence=0.1)
        rule = TruthValue(frequency=0.1, confidence=0.1)

        result = strategy.deduction(premise, rule)
        assert result.confidence >= 0.2

    def test_engine_uses_selected_strategy(self):
        """Engine should use the configured truth strategy."""
        from z3adapter.ikr.nars_datalog import from_ikr
        from z3adapter.ikr.schema import (
            IKR, Meta, QuestionType, Type, Entity, Relation,
            Fact, Rule, RuleCondition, Query, QuantifiedVariable
        )

        # Create a chain with UNCERTAIN premise: A -> B -> C with 2-hop inference
        # Use frequency < 1.0 to show the difference between strategies
        ikr = IKR(
            meta=Meta(question="Chain test", question_type=QuestionType.YES_NO),
            types=[Type(name="Thing")],
            entities=[Entity(name="x", type="Thing")],
            relations=[
                Relation(name="a", signature=["Thing"]),
                Relation(name="b", signature=["Thing"]),
                Relation(name="c", signature=["Thing"]),
            ],
            # Use uncertain fact with f=0.8 to see the difference
            facts=[Fact(
                predicate="a",
                arguments=["x"],
                truth_value=TruthValue(frequency=0.8, confidence=0.9),
            )],
            rules=[
                Rule(
                    antecedent=RuleCondition(predicate="a", arguments=["X"]),
                    consequent=RuleCondition(predicate="b", arguments=["X"]),
                    quantified_vars=[QuantifiedVariable(name="X", type="Thing")],
                ),
                Rule(
                    antecedent=RuleCondition(predicate="b", arguments=["X"]),
                    consequent=RuleCondition(predicate="c", arguments=["X"]),
                    quantified_vars=[QuantifiedVariable(name="X", type="Thing")],
                ),
            ],
            query=Query(predicate="c", arguments=["x"]),
        )

        # With current strategy (aggressive degradation: c = f1*f2*c1*c2)
        engine_current = from_ikr(ikr, truth_formula="current")
        result_current = engine_current.query(ikr.query)

        # With opennars strategy (less aggressive: c = c1*c2)
        engine_opennars = from_ikr(ikr, truth_formula="opennars")
        result_opennars = engine_opennars.query(ikr.query)

        # Both should find the result
        assert result_current.found
        assert result_opennars.found

        # OpenNARS should preserve confidence better when premise has f < 1.0
        # Current: uses f in confidence calc, OpenNARS doesn't
        assert result_opennars.truth_value.confidence > result_current.truth_value.confidence


# =============================================================================
# Epistemic Logic Tests
# =============================================================================


class TestEpistemicLogic:
    """Test epistemic logic execution (MVP)."""

    def test_objective_facts_visible_to_all(self):
        """Objective facts should be visible to all agents."""
        from z3adapter.ikr.nars_datalog import from_ikr
        from z3adapter.ikr.schema import (
            IKR, Meta, QuestionType, Type, Entity, Relation,
            Fact, Query, EpistemicConfig
        )

        ikr = IKR(
            meta=Meta(question="Test", question_type=QuestionType.YES_NO),
            epistemic_config=EpistemicConfig(agents=["alice", "bob"]),
            types=[Type(name="Thing")],
            entities=[Entity(name="x", type="Thing")],
            relations=[Relation(name="p", signature=["Thing"])],
            facts=[Fact(predicate="p", arguments=["x"])],  # Objective fact
            rules=[],
            query=Query(predicate="p", arguments=["x"]),
        )

        engine = from_ikr(ikr)

        # Visible from any perspective
        result_objective = engine.query(ikr.query, agent=None)
        result_alice = engine.query(ikr.query, agent="alice")
        result_bob = engine.query(ikr.query, agent="bob")

        assert result_objective.found
        assert result_alice.found  # Objective facts visible to alice
        assert result_bob.found  # Objective facts visible to bob

    def test_belief_visible_only_to_agent(self):
        """Agent beliefs should only be visible to that agent."""
        from z3adapter.ikr.nars_datalog import from_ikr
        from z3adapter.ikr.schema import (
            IKR, Meta, QuestionType, Type, Entity, Relation,
            Fact, Query, EpistemicConfig, EpistemicContext, EpistemicOperator
        )

        ikr = IKR(
            meta=Meta(question="Test", question_type=QuestionType.YES_NO),
            epistemic_config=EpistemicConfig(agents=["alice", "bob"]),
            types=[Type(name="Thing")],
            entities=[Entity(name="x", type="Thing")],
            relations=[Relation(name="p", signature=["Thing"])],
            facts=[
                Fact(
                    predicate="p",
                    arguments=["x"],
                    epistemic_context=EpistemicContext(
                        agent="alice",
                        modality=EpistemicOperator.BELIEVES
                    ),
                ),
            ],
            rules=[],
            query=Query(predicate="p", arguments=["x"]),
        )

        engine = from_ikr(ikr)

        # Visible to alice (her belief)
        result_alice = engine.query(ikr.query, agent="alice")
        assert result_alice.found

        # Not visible to bob (he doesn't share Alice's belief)
        result_bob = engine.query(ikr.query, agent="bob")
        assert not result_bob.found

        # Not visible from objective perspective
        result_objective = engine.query(ikr.query, agent=None)
        assert not result_objective.found

    def test_agent_index_in_fact_store(self):
        """FactStore should properly index facts by agent."""
        from z3adapter.ikr.nars_datalog import FactStore, GroundAtom

        store = FactStore()

        # Add objective fact
        obj_atom = GroundAtom(predicate="p", arguments=("x",), agent=None)
        store.add(obj_atom, TruthValue(frequency=1.0, confidence=0.9), source="base")

        # Add Alice's belief
        alice_atom = GroundAtom(predicate="p", arguments=("y",), agent="alice")
        store.add(alice_atom, TruthValue(frequency=1.0, confidence=0.9), source="base")

        # Check agents
        agents = store.get_agents()
        assert "alice" in agents
        assert None not in agents  # None is not returned as an "agent"

        # Check visibility
        visible_to_alice = list(store.get_by_predicate_for_agent("p", agent="alice"))
        assert len(visible_to_alice) == 2  # objective + alice's belief

        visible_to_bob = list(store.get_by_predicate_for_agent("p", agent="bob"))
        assert len(visible_to_bob) == 1  # only objective


# =============================================================================
# Predicate Opposites Tests
# =============================================================================


class TestExpandedPredicateOpposites:
    """Test expanded predicate opposites dictionary."""

    def test_symmetric_opposites(self):
        """Opposites should be symmetric."""
        from z3adapter.ikr.fuzzy_nars import PREDICATE_OPPOSITES, _OPPOSITE_PAIRS

        for a, b in _OPPOSITE_PAIRS:
            assert PREDICATE_OPPOSITES.get(a) == b, f"{a} should map to {b}"
            assert PREDICATE_OPPOSITES.get(b) == a, f"{b} should map to {a}"

    def test_temporal_opposites(self):
        """Test temporal opposites."""
        from z3adapter.ikr.fuzzy_nars import get_predicate_polarity

        sim, pol = get_predicate_polarity("before", "after", 0.3)
        assert pol == -1.0  # Opposites

    def test_spatial_opposites(self):
        """Test spatial opposites."""
        from z3adapter.ikr.fuzzy_nars import get_predicate_polarity

        sim, pol = get_predicate_polarity("above", "below", 0.3)
        assert pol == -1.0  # Opposites

    def test_quantity_opposites(self):
        """Test quantity opposites."""
        from z3adapter.ikr.fuzzy_nars import get_predicate_polarity

        sim, pol = get_predicate_polarity("more", "less", 0.3)
        assert pol == -1.0  # Opposites

    def test_quality_opposites(self):
        """Test quality opposites."""
        from z3adapter.ikr.fuzzy_nars import get_predicate_polarity

        sim, pol = get_predicate_polarity("good", "bad", 0.3)
        assert pol == -1.0  # Opposites

    def test_no_false_opposites(self):
        """Unrelated predicates should not be opposites."""
        from z3adapter.ikr.fuzzy_nars import get_predicate_polarity

        sim, pol = get_predicate_polarity("causes", "contains", 0.3)
        assert pol == 1.0  # Not opposites

    def test_expanded_coverage(self):
        """Should have more pairs than before."""
        from z3adapter.ikr.fuzzy_nars import _OPPOSITE_PAIRS, PREDICATE_OPPOSITES

        # Original had 14 base pairs (28 entries with bidirectional)
        # New should have many more
        assert len(_OPPOSITE_PAIRS) > 30
        # Bidirectional dict should have ~2x entries
        assert len(PREDICATE_OPPOSITES) >= len(_OPPOSITE_PAIRS) * 2 - 10  # Some overlap
