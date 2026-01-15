"""Knowledge Base loader for NARS-Datalog engine.

This module provides functionality to load and merge knowledge base modules
directly into the NARS-Datalog engine. Unlike the generic IKR KB loader,
this handles NARS-specific features:

- Truth values on facts and rules
- Direct loading into FactStore
- Rule compilation with truth values

Example usage:
    from z3adapter.ikr.nars_datalog import NARSDatalogEngine
    from z3adapter.ikr.nars_datalog.kb_loader import KBLoader

    engine = NARSDatalogEngine()
    KBLoader.load_modules(engine, ["food", "social"])
    # Now engine has KB facts and rules loaded
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from z3adapter.ikr.schema import TruthValue

from .fact_store import GroundAtom
from .rule import InternalRule, RuleAtom

if TYPE_CHECKING:
    from .engine import NARSDatalogEngine

__all__ = ["KBLoader", "KB_DIR"]

logger = logging.getLogger(__name__)

# Knowledge base directory
KB_DIR = Path(__file__).parent / "kb"


def _parse_truth_value(data: dict | None) -> TruthValue:
    """Parse truth value from KB data, defaulting to high confidence."""
    if data is None:
        return TruthValue(frequency=1.0, confidence=0.9)
    return TruthValue(
        frequency=data.get("frequency", 1.0),
        confidence=data.get("confidence", 0.9),
    )


class KBLoader:
    """Loader for NARS-Datalog knowledge base modules.

    KB modules are JSON files containing:
    - facts: Ground facts with optional truth values
    - rules: Inference rules with optional truth values

    Truth value format in JSON:
        "truth_value": {"frequency": 0.9, "confidence": 0.85}

    If omitted, defaults to (f=1.0, c=0.9) for certain knowledge.
    """

    # Cache for loaded raw modules
    _cache: dict[str, dict] = {}

    @classmethod
    def available_modules(cls) -> list[str]:
        """List available NARS KB modules.

        Returns:
            List of module names
        """
        if not KB_DIR.exists():
            return []
        return sorted(f.stem for f in KB_DIR.glob("*.json"))

    @classmethod
    def load_modules(
        cls,
        engine: "NARSDatalogEngine",
        module_names: list[str],
    ) -> dict[str, int]:
        """Load KB modules directly into a NARS-Datalog engine.

        Args:
            engine: The engine to load into
            module_names: List of module names to load

        Returns:
            Dict mapping module name to number of facts/rules loaded
        """
        stats = {}

        for name in module_names:
            try:
                kb_data = cls._load_module_data(name)
                count = cls._load_into_engine(engine, kb_data)
                stats[name] = count
                logger.debug(f"Loaded KB module '{name}': {count} facts/rules")
            except FileNotFoundError as e:
                logger.warning(f"KB module not found: {e}")
                stats[name] = 0

        return stats

    @classmethod
    def _load_module_data(cls, name: str) -> dict:
        """Load raw module data from JSON file."""
        if name in cls._cache:
            return cls._cache[name]

        path = KB_DIR / f"{name}.json"
        if not path.exists():
            available = cls.available_modules()
            raise FileNotFoundError(
                f"NARS KB module '{name}' not found. Available: {available}"
            )

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        cls._cache[name] = data
        return data

    @classmethod
    def _load_into_engine(cls, engine: "NARSDatalogEngine", kb_data: dict) -> int:
        """Load KB data into engine's fact store and rules.

        Returns count of items loaded.
        """
        count = 0

        # Load facts
        for fact_data in kb_data.get("facts", []):
            atom = GroundAtom(
                predicate=fact_data["predicate"],
                arguments=tuple(fact_data.get("arguments", [])),
                negated=fact_data.get("negated", False),
            )
            tv = _parse_truth_value(fact_data.get("truth_value"))
            engine.fact_store.add(atom, tv, source="kb")
            count += 1

        # Load rules
        for rule_data in kb_data.get("rules", []):
            rule = cls._parse_rule(rule_data)
            if rule:
                engine._rules.append(rule)
                count += 1

        return count

    @classmethod
    def _parse_rule(cls, rule_data: dict) -> InternalRule | None:
        """Parse a rule from KB JSON format."""
        # Parse head (consequent)
        consequent = rule_data.get("consequent")
        if not consequent:
            return None

        head = RuleAtom(
            predicate=consequent["predicate"],
            arguments=tuple(consequent.get("arguments", [])),
            negated=consequent.get("negated", False),
        )

        # Parse body (antecedent)
        antecedent = rule_data.get("antecedent")
        body = cls._parse_condition(antecedent) if antecedent else []

        # Collect variables
        variables = set()
        for var in rule_data.get("quantified_vars", []):
            variables.add(var["name"])

        # Check for negation
        has_negation = any(atom.negated for atom in body)

        # Parse rule truth value
        rule_tv = _parse_truth_value(rule_data.get("truth_value"))

        return InternalRule(
            name=rule_data.get("name"),
            head=head,
            body=body,
            variables=variables,
            has_negation=has_negation,
            rule_truth=rule_tv,
        )

    @classmethod
    def _parse_condition(cls, cond: dict) -> list[RuleAtom]:
        """Parse a condition into list of body atoms."""
        if "and" in cond:
            result = []
            for sub in cond["and"]:
                result.extend(cls._parse_condition(sub))
            return result
        elif "predicate" in cond:
            return [
                RuleAtom(
                    predicate=cond["predicate"],
                    arguments=tuple(cond.get("arguments", [])),
                    negated=cond.get("negated", False),
                )
            ]
        return []

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the module cache."""
        cls._cache.clear()

    @classmethod
    def get_module_info(cls, name: str) -> dict:
        """Get metadata about a KB module."""
        data = cls._load_module_data(name)
        return {
            "name": data.get("name", name),
            "version": data.get("version", "unknown"),
            "description": data.get("description", ""),
            "facts_count": len(data.get("facts", [])),
            "rules_count": len(data.get("rules", [])),
        }
