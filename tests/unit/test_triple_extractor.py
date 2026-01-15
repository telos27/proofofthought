"""Tests for Triple extraction.

Tests the TripleExtractor with mocked LLM responses.
"""

import json
import pytest
from unittest.mock import MagicMock

from z3adapter.ikr.triples import (
    Predicate,
    Triple,
    TripleExtractor,
    ExtractionResult,
    extract_triples,
)


# =============================================================================
# Mock LLM Client
# =============================================================================


def create_mock_client(response_text: str) -> MagicMock:
    """Create a mock LLM client that returns a predefined response."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = response_text
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def create_mock_client_sequence(responses: list[str]) -> MagicMock:
    """Create a mock LLM client that returns responses in sequence."""
    mock_client = MagicMock()
    mock_responses = []
    for response_text in responses:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = response_text
        mock_responses.append(mock_response)
    mock_client.chat.completions.create.side_effect = mock_responses
    return mock_client


# =============================================================================
# Sample Responses
# =============================================================================


SIMPLE_RESPONSE = '''```json
{
  "triples": [
    {
      "id": "t1",
      "subject": "stress",
      "predicate": "causes",
      "object": "anxiety",
      "negated": false,
      "surface_form": "Stress causes anxiety"
    }
  ]
}
```'''


MULTI_TRIPLE_RESPONSE = '''```json
{
  "triples": [
    {
      "id": "t1",
      "subject": "chronic_stress",
      "predicate": "causes",
      "object": "elevated_cortisol",
      "negated": false,
      "surface_form": "Chronic stress leads to elevated cortisol levels"
    },
    {
      "id": "t2",
      "subject": "elevated_cortisol",
      "predicate": "causes",
      "object": "memory_impairment",
      "negated": false,
      "surface_form": "elevated cortisol levels can impair memory formation"
    }
  ]
}
```'''


NESTED_BELIEF_RESPONSE = '''```json
{
  "triples": [
    {
      "id": "t1",
      "subject": "meditation",
      "predicate": "prevents",
      "object": "anxiety",
      "negated": true,
      "surface_form": "meditation does not reduce anxiety in all cases"
    },
    {
      "id": "t2",
      "subject": "dr_smith",
      "predicate": "believes",
      "object": "t:t1",
      "negated": false,
      "surface_form": "Dr. Smith argues that"
    }
  ]
}
```'''


RESPONSE_WITHOUT_CODE_BLOCK = '''{
  "triples": [
    {
      "id": "t1",
      "subject": "exercise",
      "predicate": "prevents",
      "object": "depression",
      "negated": false,
      "surface_form": "Exercise prevents depression"
    }
  ]
}'''


RESPONSE_WITH_TRUTH_VALUE = '''```json
{
  "triples": [
    {
      "id": "t1",
      "subject": "birds",
      "predicate": "has",
      "object": "flight_ability",
      "negated": false,
      "surface_form": "Birds can fly",
      "truth": {
        "frequency": 0.9,
        "confidence": 0.8
      }
    }
  ]
}
```'''


INVALID_PREDICATE_RESPONSE = '''```json
{
  "triples": [
    {
      "id": "t1",
      "subject": "cat",
      "predicate": "invented_by",
      "object": "nature",
      "negated": false
    }
  ]
}
```'''


MISSING_FIELD_RESPONSE = '''```json
{
  "triples": [
    {
      "id": "t1",
      "subject": "stress"
    }
  ]
}
```'''


EMPTY_RESPONSE = '''```json
{
  "triples": []
}
```'''


MALFORMED_JSON_RESPONSE = "This is not valid JSON at all"


# =============================================================================
# TripleExtractor Tests
# =============================================================================


class TestTripleExtractorBasic:
    """Basic tests for TripleExtractor."""

    def test_creation(self):
        """Test extractor creation."""
        client = create_mock_client(SIMPLE_RESPONSE)
        extractor = TripleExtractor(client, model="gpt-4o")

        assert extractor.client == client
        assert extractor.model == "gpt-4o"
        assert extractor.max_tokens == 2000

    def test_creation_custom_params(self):
        """Test extractor creation with custom parameters."""
        client = create_mock_client(SIMPLE_RESPONSE)
        extractor = TripleExtractor(client, model="gpt-3.5-turbo", max_tokens=1000)

        assert extractor.model == "gpt-3.5-turbo"
        assert extractor.max_tokens == 1000


class TestTripleExtraction:
    """Tests for triple extraction."""

    def test_simple_extraction(self):
        """Test extracting a single triple."""
        client = create_mock_client(SIMPLE_RESPONSE)
        extractor = TripleExtractor(client)

        result = extractor.extract("Stress causes anxiety.")

        assert isinstance(result, ExtractionResult)
        assert len(result.triples) == 1

        triple = result.triples[0]
        assert triple.id == "t1"
        assert triple.subject == "stress"
        assert triple.predicate == Predicate.CAUSES
        assert triple.object == "anxiety"
        assert triple.negated is False
        assert triple.surface_form == "Stress causes anxiety"

    def test_multi_triple_extraction(self):
        """Test extracting multiple triples."""
        client = create_mock_client(MULTI_TRIPLE_RESPONSE)
        extractor = TripleExtractor(client)

        result = extractor.extract("Chronic stress leads to elevated cortisol...")

        assert len(result.triples) == 2
        assert result.triples[0].subject == "chronic_stress"
        assert result.triples[1].subject == "elevated_cortisol"

    def test_nested_belief_extraction(self):
        """Test extracting nested beliefs with triple references."""
        client = create_mock_client(NESTED_BELIEF_RESPONSE)
        extractor = TripleExtractor(client)

        result = extractor.extract("Dr. Smith argues that meditation...")

        assert len(result.triples) == 2

        # First triple: the claim
        t1 = result.triples[0]
        assert t1.predicate == Predicate.PREVENTS
        assert t1.negated is True

        # Second triple: the belief about the claim
        t2 = result.triples[1]
        assert t2.subject == "dr_smith"
        assert t2.predicate == Predicate.BELIEVES
        assert t2.object == "t:t1"
        assert t2.object_is_triple is True

    def test_source_propagation(self):
        """Test that source is propagated to triples."""
        client = create_mock_client(SIMPLE_RESPONSE)
        extractor = TripleExtractor(client)

        result = extractor.extract("Stress causes anxiety.", source="Smith 2020 p.42")

        assert result.source == "Smith 2020 p.42"
        assert result.triples[0].source == "Smith 2020 p.42"

    def test_context_included_in_prompt(self):
        """Test that context is passed to LLM."""
        client = create_mock_client(SIMPLE_RESPONSE)
        extractor = TripleExtractor(client)

        extractor.extract("Some text", context="This is psychology research")

        # Verify context was included in the prompt
        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        user_message = messages[1]["content"]
        assert "Context: This is psychology research" in user_message


class TestTripleExtractionEdgeCases:
    """Tests for edge cases in triple extraction."""

    def test_response_without_code_block(self):
        """Test handling response without markdown code block."""
        client = create_mock_client(RESPONSE_WITHOUT_CODE_BLOCK)
        extractor = TripleExtractor(client)

        result = extractor.extract("Exercise prevents depression.")

        assert len(result.triples) == 1
        assert result.triples[0].predicate == Predicate.PREVENTS

    def test_truth_value_parsing(self):
        """Test parsing truth values from response."""
        client = create_mock_client(RESPONSE_WITH_TRUTH_VALUE)
        extractor = TripleExtractor(client)

        result = extractor.extract("Birds can fly.")

        assert len(result.triples) == 1
        triple = result.triples[0]
        assert triple.truth is not None
        assert triple.truth.frequency == 0.9
        assert triple.truth.confidence == 0.8

    def test_invalid_predicate_fallback(self):
        """Test that invalid predicates fall back to related_to."""
        client = create_mock_client(INVALID_PREDICATE_RESPONSE)
        extractor = TripleExtractor(client)

        result = extractor.extract("Cats were invented by nature.")

        assert len(result.triples) == 1
        assert result.triples[0].predicate == Predicate.RELATED_TO

    def test_missing_field_skipped(self):
        """Test that triples with missing required fields are skipped."""
        client = create_mock_client(MISSING_FIELD_RESPONSE)
        extractor = TripleExtractor(client)

        result = extractor.extract("Some incomplete text.")

        assert len(result.triples) == 0

    def test_empty_response(self):
        """Test handling empty triples array."""
        client = create_mock_client(EMPTY_RESPONSE)
        extractor = TripleExtractor(client)

        result = extractor.extract("No meaningful content.")

        assert len(result.triples) == 0
        assert result.raw_response == EMPTY_RESPONSE

    def test_malformed_json(self):
        """Test handling malformed JSON response."""
        client = create_mock_client(MALFORMED_JSON_RESPONSE)
        extractor = TripleExtractor(client)

        result = extractor.extract("Some text.")

        assert len(result.triples) == 0
        assert result.raw_response == MALFORMED_JSON_RESPONSE


class TestEntityNormalization:
    """Tests for entity name normalization."""

    def test_lowercase_normalization(self):
        """Test that entities are lowercased."""
        response = '''{"triples": [{"id": "t1", "subject": "STRESS", "predicate": "causes", "object": "ANXIETY"}]}'''
        client = create_mock_client(response)
        extractor = TripleExtractor(client)

        result = extractor.extract("STRESS causes ANXIETY.")

        assert result.triples[0].subject == "stress"
        assert result.triples[0].object == "anxiety"

    def test_space_to_underscore(self):
        """Test that spaces become underscores."""
        response = '''{"triples": [{"id": "t1", "subject": "chronic stress", "predicate": "causes", "object": "elevated cortisol"}]}'''
        client = create_mock_client(response)
        extractor = TripleExtractor(client)

        result = extractor.extract("Chronic stress causes elevated cortisol.")

        assert result.triples[0].subject == "chronic_stress"
        assert result.triples[0].object == "elevated_cortisol"

    def test_hyphen_to_underscore(self):
        """Test that hyphens become underscores."""
        response = '''{"triples": [{"id": "t1", "subject": "self-esteem", "predicate": "causes", "object": "well-being"}]}'''
        client = create_mock_client(response)
        extractor = TripleExtractor(client)

        result = extractor.extract("Self-esteem affects well-being.")

        assert result.triples[0].subject == "self_esteem"
        assert result.triples[0].object == "well_being"

    def test_triple_reference_not_normalized(self):
        """Test that triple references are not normalized."""
        response = '''{"triples": [{"id": "t1", "subject": "alice", "predicate": "believes", "object": "t:T1"}]}'''
        client = create_mock_client(response)
        extractor = TripleExtractor(client)

        result = extractor.extract("Alice believes something.")

        # Triple reference should preserve its form
        assert result.triples[0].object == "t:T1"


class TestBatchExtraction:
    """Tests for batch extraction."""

    def test_batch_extraction(self):
        """Test extracting from multiple texts."""
        responses = [SIMPLE_RESPONSE, MULTI_TRIPLE_RESPONSE]
        client = create_mock_client_sequence(responses)
        extractor = TripleExtractor(client)

        results = extractor.extract_batch(
            ["Text 1", "Text 2"],
            source="Book Chapter 1"
        )

        assert len(results) == 2
        assert len(results[0].triples) == 1
        assert len(results[1].triples) == 2

        # Check source propagation
        assert "chunk 1" in results[0].source
        assert "chunk 2" in results[1].source


class TestLLMErrors:
    """Tests for LLM error handling."""

    def test_llm_exception(self):
        """Test handling LLM API exception."""
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")
        extractor = TripleExtractor(client)

        result = extractor.extract("Some text.")

        assert len(result.triples) == 0
        assert "API error" in result.raw_response


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestExtractTriplesFunction:
    """Tests for extract_triples convenience function."""

    def test_extract_triples_function(self):
        """Test the convenience function."""
        client = create_mock_client(SIMPLE_RESPONSE)

        triples = extract_triples("Stress causes anxiety.", client, model="gpt-4o")

        assert len(triples) == 1
        assert triples[0].subject == "stress"

    def test_extract_triples_with_source(self):
        """Test convenience function with source."""
        client = create_mock_client(SIMPLE_RESPONSE)

        triples = extract_triples(
            "Stress causes anxiety.",
            client,
            source="Test source"
        )

        assert triples[0].source == "Test source"


# =============================================================================
# Integration Tests (with mocked LLM)
# =============================================================================


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_psychology_text_extraction(self):
        """Test extracting from psychology text."""
        response = '''```json
{
  "triples": [
    {"id": "t1", "subject": "phobia", "predicate": "is_a", "object": "anxiety_disorder", "negated": false},
    {"id": "t2", "subject": "phobia", "predicate": "causes", "object": "avoidance_behavior", "negated": false},
    {"id": "t3", "subject": "exposure_therapy", "predicate": "prevents", "object": "phobia", "negated": false}
  ]
}
```'''
        client = create_mock_client(response)
        extractor = TripleExtractor(client)

        result = extractor.extract(
            "Phobias are anxiety disorders that cause avoidance behavior. "
            "Exposure therapy is effective in treating phobias.",
            source="DSM-5"
        )

        assert len(result.triples) == 3

        # Verify taxonomy
        taxonomy = [t for t in result.triples if t.predicate == Predicate.IS_A]
        assert len(taxonomy) == 1
        assert taxonomy[0].subject == "phobia"

        # Verify causation
        causal = [t for t in result.triples if t.predicate == Predicate.CAUSES]
        assert len(causal) == 1

        # Verify treatment
        prevention = [t for t in result.triples if t.predicate == Predicate.PREVENTS]
        assert len(prevention) == 1
        assert prevention[0].subject == "exposure_therapy"

    def test_complex_belief_chain(self):
        """Test complex belief attribution chain."""
        response = '''```json
{
  "triples": [
    {"id": "t1", "subject": "vaccines", "predicate": "prevents", "object": "disease", "negated": false},
    {"id": "t2", "subject": "who", "predicate": "believes", "object": "t:t1", "negated": false},
    {"id": "t3", "subject": "article", "predicate": "believes", "object": "t:t2", "negated": false}
  ]
}
```'''
        client = create_mock_client(response)
        extractor = TripleExtractor(client)

        result = extractor.extract(
            "The article reports that WHO believes vaccines prevent disease."
        )

        assert len(result.triples) == 3

        # Verify belief chain
        t1 = next(t for t in result.triples if t.id == "t1")
        t2 = next(t for t in result.triples if t.id == "t2")
        t3 = next(t for t in result.triples if t.id == "t3")

        assert t2.object == "t:t1"
        assert t3.object == "t:t2"

        # Verify triple references
        assert t2.object_is_triple
        assert t3.object_is_triple
