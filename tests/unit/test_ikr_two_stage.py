"""Unit tests for IKR two-stage generation.

Tests the two-stage prompting mechanism where:
- Stage 1: Extracts explicit facts, types, entities, relations, and query
- Stage 2: Generates background knowledge (facts and rules)
"""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Skip if z3 is not available
try:
    from z3adapter.reasoning.program_generator import (
        Z3ProgramGenerator,
        GenerationResult,
        IKRStageResult,
    )

    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


# Sample Stage 1 response (explicit knowledge)
STAGE1_RESPONSE = """Here is the IKR for the question:

```json
{
  "meta": {
    "question": "Would a vegetarian eat a plant burger?",
    "question_type": "yes_no"
  },
  "types": [
    {"name": "Person", "description": "A human individual"},
    {"name": "Food", "description": "An edible item"}
  ],
  "entities": [
    {"name": "vegetarian_person", "type": "Person", "aliases": ["a vegetarian"]},
    {"name": "plant_burger", "type": "Food", "aliases": ["a plant burger"]}
  ],
  "relations": [
    {"name": "is_vegetarian", "signature": ["Person"], "range": "Bool"},
    {"name": "is_plant_based", "signature": ["Food"], "range": "Bool"},
    {"name": "would_eat", "signature": ["Person", "Food"], "range": "Bool"}
  ],
  "facts": [
    {"predicate": "is_vegetarian", "arguments": ["vegetarian_person"], "source": "explicit"},
    {"predicate": "is_plant_based", "arguments": ["plant_burger"], "source": "explicit"}
  ],
  "query": {
    "predicate": "would_eat",
    "arguments": ["vegetarian_person", "plant_burger"]
  }
}
```
"""

# Sample Stage 2 response (background knowledge)
STAGE2_RESPONSE = """Based on the explicit knowledge, here is the background knowledge needed:

```json
{
  "background_facts": [],
  "rules": [
    {
      "name": "vegetarians avoid meat",
      "quantified_vars": [{"name": "p", "type": "Person"}],
      "antecedent": {"predicate": "is_vegetarian", "arguments": ["p"]},
      "consequent": {"predicate": "avoids_meat", "arguments": ["p"]},
      "justification": "Definition of vegetarian diet"
    },
    {
      "name": "plant-based means no meat",
      "quantified_vars": [{"name": "f", "type": "Food"}],
      "antecedent": {"predicate": "is_plant_based", "arguments": ["f"]},
      "consequent": {"predicate": "contains_meat", "arguments": ["f"], "negated": true},
      "justification": "Plant-based foods by definition contain no meat"
    },
    {
      "name": "dietary compatibility",
      "quantified_vars": [{"name": "p", "type": "Person"}, {"name": "f", "type": "Food"}],
      "antecedent": {
        "and": [
          {"predicate": "avoids_meat", "arguments": ["p"]},
          {"predicate": "contains_meat", "arguments": ["f"], "negated": true}
        ]
      },
      "consequent": {"predicate": "would_eat", "arguments": ["p", "f"]},
      "justification": "People eat foods compatible with their dietary restrictions"
    }
  ]
}
```
"""


def create_mock_llm_client(responses):
    """Create a mock LLM client that returns predefined responses."""
    mock_client = MagicMock()
    mock_responses = []
    for response_text in responses:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = response_text
        mock_responses.append(mock_response)

    mock_client.chat.completions.create.side_effect = mock_responses
    return mock_client


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestIKRTwoStageGeneration(unittest.TestCase):
    """Tests for two-stage IKR generation."""

    def test_successful_two_stage_generation(self):
        """Test successful two-stage IKR generation."""
        mock_client = create_mock_llm_client([STAGE1_RESPONSE, STAGE2_RESPONSE])

        generator = Z3ProgramGenerator(
            llm_client=mock_client, model="gpt-4o", backend="ikr", ikr_two_stage=True
        )

        result = generator.generate("Would a vegetarian eat a plant burger?")

        # Verify success
        self.assertTrue(result.success)
        self.assertTrue(result.two_stage)
        self.assertEqual(result.backend, "ikr")

        # Verify merged IKR structure
        ikr = result.program
        self.assertIsNotNone(ikr)
        self.assertIn("meta", ikr)
        self.assertIn("types", ikr)
        self.assertIn("entities", ikr)
        self.assertIn("relations", ikr)
        self.assertIn("facts", ikr)
        self.assertIn("rules", ikr)
        self.assertIn("query", ikr)

        # Verify Stage 1 content preserved
        self.assertEqual(len(ikr["types"]), 2)
        self.assertEqual(len(ikr["entities"]), 2)

        # Verify Stage 2 rules were merged
        self.assertEqual(len(ikr["rules"]), 3)

        # Verify both stage responses are recorded
        self.assertIsNotNone(result.stage1_response)
        self.assertIsNotNone(result.stage2_response)

        # Verify two LLM calls were made
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)

    def test_stage1_failure(self):
        """Test handling of Stage 1 failure."""
        invalid_response = "This is not valid JSON"
        mock_client = create_mock_llm_client([invalid_response])

        generator = Z3ProgramGenerator(
            llm_client=mock_client, model="gpt-4o", backend="ikr", ikr_two_stage=True
        )

        result = generator.generate("Would a vegetarian eat a plant burger?")

        # Verify failure
        self.assertFalse(result.success)
        self.assertTrue(result.two_stage)
        self.assertIn("Stage 1", result.error)
        self.assertIsNone(result.program)

        # Only one LLM call should be made
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)

    def test_stage1_missing_required_fields(self):
        """Test handling of Stage 1 output missing required fields."""
        incomplete_response = """```json
{
  "meta": {"question": "Test?"},
  "types": []
}
```"""
        mock_client = create_mock_llm_client([incomplete_response])

        generator = Z3ProgramGenerator(
            llm_client=mock_client, model="gpt-4o", backend="ikr", ikr_two_stage=True
        )

        result = generator.generate("Test question")

        self.assertFalse(result.success)
        self.assertIn("missing required fields", result.error)

    def test_stage2_failure_returns_partial(self):
        """Test that Stage 2 failure returns partial IKR from Stage 1."""
        invalid_stage2 = "This is not valid JSON for stage 2"
        mock_client = create_mock_llm_client([STAGE1_RESPONSE, invalid_stage2])

        generator = Z3ProgramGenerator(
            llm_client=mock_client, model="gpt-4o", backend="ikr", ikr_two_stage=True
        )

        result = generator.generate("Would a vegetarian eat a plant burger?")

        # Should succeed with partial result
        self.assertTrue(result.success)
        self.assertTrue(result.two_stage)
        self.assertIn("Stage 2 failed", result.error)

        # Verify we get Stage 1 IKR
        ikr = result.program
        self.assertIsNotNone(ikr)
        self.assertEqual(len(ikr["facts"]), 2)  # Only explicit facts
        self.assertEqual(ikr["rules"], [])  # Empty rules

    def test_single_stage_fallback(self):
        """Test single-stage generation when two_stage=False."""
        single_stage_response = """```json
{
  "meta": {"question": "Is water wet?", "question_type": "yes_no"},
  "types": [{"name": "Substance"}],
  "entities": [{"name": "water", "type": "Substance"}],
  "relations": [{"name": "is_wet", "signature": ["Substance"], "range": "Bool"}],
  "facts": [{"predicate": "is_wet", "arguments": ["water"], "source": "explicit"}],
  "rules": [],
  "query": {"predicate": "is_wet", "arguments": ["water"]}
}
```"""
        mock_client = create_mock_llm_client([single_stage_response])

        generator = Z3ProgramGenerator(
            llm_client=mock_client, model="gpt-4o", backend="ikr", ikr_two_stage=False
        )

        result = generator.generate("Is water wet?")

        self.assertTrue(result.success)
        self.assertFalse(result.two_stage)
        self.assertEqual(result.backend, "ikr")

        # Only one LLM call
        self.assertEqual(mock_client.chat.completions.create.call_count, 1)


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestIKRStageMerging(unittest.TestCase):
    """Tests for IKR stage merging logic."""

    def test_merge_background_facts(self):
        """Test that background facts are properly merged."""
        stage2_with_facts = """```json
{
  "background_facts": [
    {"predicate": "likes_food", "arguments": ["vegetarian_person"], "source": "background", "justification": "People generally like food"}
  ],
  "rules": []
}
```"""
        mock_client = create_mock_llm_client([STAGE1_RESPONSE, stage2_with_facts])

        generator = Z3ProgramGenerator(
            llm_client=mock_client, model="gpt-4o", backend="ikr", ikr_two_stage=True
        )

        result = generator.generate("Would a vegetarian eat a plant burger?")

        self.assertTrue(result.success)
        ikr = result.program

        # Should have 2 explicit + 1 background = 3 facts
        self.assertEqual(len(ikr["facts"]), 3)

        # Verify background fact was added
        background_facts = [f for f in ikr["facts"] if f.get("source") == "background"]
        self.assertEqual(len(background_facts), 1)
        self.assertEqual(background_facts[0]["predicate"], "likes_food")

    def test_merge_empty_stage2(self):
        """Test handling of Stage 2 with empty lists."""
        empty_stage2 = """```json
{
  "background_facts": [],
  "rules": []
}
```"""
        mock_client = create_mock_llm_client([STAGE1_RESPONSE, empty_stage2])

        generator = Z3ProgramGenerator(
            llm_client=mock_client, model="gpt-4o", backend="ikr", ikr_two_stage=True
        )

        result = generator.generate("Would a vegetarian eat a plant burger?")

        self.assertTrue(result.success)
        ikr = result.program

        # Should only have original facts
        self.assertEqual(len(ikr["facts"]), 2)
        self.assertEqual(len(ikr["rules"]), 0)


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestIKRStagePrompts(unittest.TestCase):
    """Tests for IKR stage prompt construction."""

    def test_stage1_prompt_contains_question(self):
        """Test that Stage 1 prompt contains the question."""
        mock_client = create_mock_llm_client([STAGE1_RESPONSE, STAGE2_RESPONSE])

        generator = Z3ProgramGenerator(
            llm_client=mock_client, model="gpt-4o", backend="ikr", ikr_two_stage=True
        )

        generator.generate("Would a vegetarian eat a plant burger?")

        # Get the first call's prompt
        first_call = mock_client.chat.completions.create.call_args_list[0]
        messages = first_call.kwargs["messages"]
        prompt = messages[0]["content"]

        self.assertIn("Would a vegetarian eat a plant burger?", prompt)
        self.assertIn("explicit", prompt.lower())  # Stage 1 focuses on explicit facts

    def test_stage2_prompt_contains_stage1_output(self):
        """Test that Stage 2 prompt contains Stage 1 output."""
        mock_client = create_mock_llm_client([STAGE1_RESPONSE, STAGE2_RESPONSE])

        generator = Z3ProgramGenerator(
            llm_client=mock_client, model="gpt-4o", backend="ikr", ikr_two_stage=True
        )

        generator.generate("Would a vegetarian eat a plant burger?")

        # Get the second call's prompt
        second_call = mock_client.chat.completions.create.call_args_list[1]
        messages = second_call.kwargs["messages"]
        prompt = messages[0]["content"]

        # Stage 2 prompt should contain Stage 1 IKR elements
        self.assertIn("vegetarian_person", prompt)
        self.assertIn("plant_burger", prompt)
        self.assertIn("is_vegetarian", prompt)
        self.assertIn("background", prompt.lower())  # Stage 2 focuses on background


@unittest.skipUnless(Z3_AVAILABLE, "z3-solver not installed")
class TestGenerationResultMetadata(unittest.TestCase):
    """Tests for GenerationResult metadata in two-stage mode."""

    def test_two_stage_flag(self):
        """Test that two_stage flag is properly set."""
        mock_client = create_mock_llm_client([STAGE1_RESPONSE, STAGE2_RESPONSE])

        generator = Z3ProgramGenerator(
            llm_client=mock_client, model="gpt-4o", backend="ikr", ikr_two_stage=True
        )

        result = generator.generate("Test question")

        self.assertTrue(result.two_stage)
        self.assertIsNotNone(result.stage1_response)
        self.assertIsNotNone(result.stage2_response)

    def test_raw_response_contains_both_stages(self):
        """Test that raw_response contains both stage responses."""
        mock_client = create_mock_llm_client([STAGE1_RESPONSE, STAGE2_RESPONSE])

        generator = Z3ProgramGenerator(
            llm_client=mock_client, model="gpt-4o", backend="ikr", ikr_two_stage=True
        )

        result = generator.generate("Test question")

        # Raw response should contain both stages
        self.assertIn("Stage 1:", result.raw_response)
        self.assertIn("Stage 2:", result.raw_response)

    def test_ikr_program_property(self):
        """Test the ikr_program property returns the IKR dict."""
        mock_client = create_mock_llm_client([STAGE1_RESPONSE, STAGE2_RESPONSE])

        generator = Z3ProgramGenerator(
            llm_client=mock_client, model="gpt-4o", backend="ikr", ikr_two_stage=True
        )

        result = generator.generate("Test question")

        self.assertIsNotNone(result.ikr_program)
        self.assertIsInstance(result.ikr_program, dict)
        self.assertIn("meta", result.ikr_program)


if __name__ == "__main__":
    unittest.main()
