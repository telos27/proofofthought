"""Commonsense Knowledge Base loader and manager.

This module provides functionality to load and merge pre-built commonsense
knowledge modules into IKR representations. Knowledge base modules contain
reusable types, relations, facts, and rules that can be combined with
question-specific IKR content.

Example usage:
    from z3adapter.ikr.knowledge_base import KnowledgeBase

    # List available modules
    modules = KnowledgeBase.available_modules()

    # Load a specific module
    commonsense_kb = KnowledgeBase.load_module("commonsense")

    # Merge KB modules into existing IKR data
    ikr_data = {...}  # Your IKR dictionary
    merged = KnowledgeBase.merge_into_ikr(ikr_data, ["commonsense"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Knowledge base directory (relative to this module)
KB_DIR = Path(__file__).parent / "kb"


class KnowledgeBase:
    """Manager for commonsense knowledge base modules.

    Provides loading and merging of pre-built knowledge modules.
    Modules are JSON files in the kb/ directory containing:
    - types: Domain type declarations
    - relations: Predicate/function declarations
    - facts: Ground facts (common knowledge)
    - rules: Universal rules (commonsense implications)
    """

    # Cache for loaded modules
    _cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def available_modules(cls) -> list[str]:
        """List available knowledge base modules.

        Returns:
            List of module names (without .json extension)
        """
        if not KB_DIR.exists():
            return []
        return sorted(f.stem for f in KB_DIR.glob("*.json"))

    @classmethod
    def load_module(cls, name: str) -> dict[str, Any]:
        """Load a knowledge base module by name.

        Args:
            name: Module name (e.g., "commonsense")

        Returns:
            Dictionary containing the module's types, relations, facts, rules

        Raises:
            FileNotFoundError: If module doesn't exist
            json.JSONDecodeError: If module file is invalid JSON
        """
        if name in cls._cache:
            return cls._cache[name]

        path = KB_DIR / f"{name}.json"
        if not path.exists():
            available = cls.available_modules()
            raise FileNotFoundError(
                f"KB module '{name}' not found. "
                f"Available modules: {available}"
            )

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # Mark all facts and rules as from KB
        for fact in data.get("facts", []):
            fact["source"] = "kb"
        for rule in data.get("rules", []):
            rule["source"] = "kb"

        cls._cache[name] = data
        logger.debug(f"Loaded KB module: {name}")
        return data

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the module cache."""
        cls._cache.clear()

    @classmethod
    def merge_into_ikr(
        cls,
        ikr_data: dict[str, Any],
        module_names: list[str],
    ) -> dict[str, Any]:
        """Merge knowledge base modules into IKR data.

        Precedence (highest to lowest):
        1. Explicit facts from the question
        2. Background facts from Stage 2
        3. KB module facts

        For types and relations, KB definitions are added only if
        they don't already exist in the IKR.

        Args:
            ikr_data: The IKR dictionary to merge into
            module_names: List of KB module names to load and merge

        Returns:
            New dictionary with KB content merged in
        """
        if not module_names:
            return ikr_data

        merged = _deep_copy_ikr(ikr_data)

        # Track existing names to avoid duplicates
        existing_types = {t["name"] for t in merged.get("types", [])}
        existing_relations = {r["name"] for r in merged.get("relations", [])}
        existing_entities = {e["name"] for e in merged.get("entities", [])}

        for name in module_names:
            try:
                kb = cls.load_module(name)
            except FileNotFoundError as e:
                logger.warning(f"Skipping unavailable KB module: {e}")
                continue

            # Add types (if not already defined)
            for t in kb.get("types", []):
                if t["name"] not in existing_types:
                    merged.setdefault("types", []).append(t)
                    existing_types.add(t["name"])

            # Add relations (if not already defined)
            for r in kb.get("relations", []):
                if r["name"] not in existing_relations:
                    merged.setdefault("relations", []).append(r)
                    existing_relations.add(r["name"])

            # Add entities (if not already defined)
            for e in kb.get("entities", []):
                if e["name"] not in existing_entities:
                    merged.setdefault("entities", []).append(e)
                    existing_entities.add(e["name"])

            # Add KB facts (lower priority, added at end)
            for fact in kb.get("facts", []):
                fact_copy = dict(fact)
                fact_copy["source"] = "kb"
                merged.setdefault("facts", []).append(fact_copy)

            # Add KB rules (lower priority, added at end)
            for rule in kb.get("rules", []):
                rule_copy = dict(rule)
                rule_copy["source"] = "kb"
                merged.setdefault("rules", []).append(rule_copy)

            logger.debug(f"Merged KB module '{name}' into IKR")

        return merged

    @classmethod
    def get_module_info(cls, name: str) -> dict[str, Any]:
        """Get metadata about a knowledge base module.

        Args:
            name: Module name

        Returns:
            Dictionary with module metadata (name, version, description, counts)
        """
        kb = cls.load_module(name)
        return {
            "name": kb.get("name", name),
            "version": kb.get("version", "unknown"),
            "description": kb.get("description", ""),
            "types_count": len(kb.get("types", [])),
            "relations_count": len(kb.get("relations", [])),
            "entities_count": len(kb.get("entities", [])),
            "facts_count": len(kb.get("facts", [])),
            "rules_count": len(kb.get("rules", [])),
        }


def _deep_copy_ikr(ikr_data: dict[str, Any]) -> dict[str, Any]:
    """Create a deep copy of IKR data to avoid mutating the original."""
    result = {}
    for key, value in ikr_data.items():
        if isinstance(value, list):
            result[key] = [dict(item) if isinstance(item, dict) else item for item in value]
        elif isinstance(value, dict):
            result[key] = dict(value)
        else:
            result[key] = value
    return result
