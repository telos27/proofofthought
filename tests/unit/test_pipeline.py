"""Tests for Triple extraction pipeline.

Tests the end-to-end ExtractionPipeline with mocked LLM responses.
"""

import json
import pytest
from unittest.mock import MagicMock

from z3adapter.ikr.fuzzy_nars import VerificationVerdict
from z3adapter.ikr.triples import (
    Predicate,
    Triple,
    TripleStore,
    TripleExtractor,
    EntityResolver,
)
from z3adapter.ikr.triples.pipeline import (
    ExtractionPipeline,
    IngestResult,
    QueryResult,
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
      "surface_form": "Chronic stress leads to elevated cortisol"
    },
    {
      "id": "t2",
      "subject": "elevated_cortisol",
      "predicate": "causes",
      "object": "memory_impairment",
      "negated": false,
      "surface_form": "elevated cortisol can impair memory"
    }
  ]
}
```'''


QUERY_STRESS_RESPONSE = '''```json
{
  "triples": [
    {
      "id": "q1",
      "subject": "stress",
      "predicate": "causes",
      "object": "anxiousness",
      "negated": false
    }
  ]
}
```'''


PREVENTION_RESPONSE = '''```json
{
  "triples": [
    {
      "id": "t1",
      "subject": "exercise",
      "predicate": "prevents",
      "object": "anxiety",
      "negated": false,
      "surface_form": "Exercise prevents anxiety"
    }
  ]
}
```'''


QUERY_EXERCISE_CAUSES = '''```json
{
  "triples": [
    {
      "id": "q1",
      "subject": "exercise",
      "predicate": "causes",
      "object": "anxiety",
      "negated": false
    }
  ]
}
```'''


EMPTY_RESPONSE = '''```json
{
  "triples": []
}
```'''


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_pipeline():
    """Create a pipeline with mock client returning simple response."""
    client = create_mock_client(SIMPLE_RESPONSE)
    extractor = TripleExtractor(client, model="test")
    resolver = EntityResolver()
    return ExtractionPipeline(
        extractor=extractor,
        resolver=resolver,
        resolve_entities=True,
    )


@pytest.fixture
def pipeline_no_resolution():
    """Create a pipeline without entity resolution."""
    client = create_mock_client(SIMPLE_RESPONSE)
    extractor = TripleExtractor(client, model="test")
    resolver = EntityResolver()
    return ExtractionPipeline(
        extractor=extractor,
        resolver=resolver,
        resolve_entities=False,
    )


# =============================================================================
# Pipeline Creation Tests
# =============================================================================


class TestPipelineCreation:
    """Tests for pipeline creation."""

    def test_create_basic(self):
        """Test creating a basic pipeline."""
        client = create_mock_client(SIMPLE_RESPONSE)
        pipeline = ExtractionPipeline.create(client, model="test")
        assert pipeline.extractor is not None
        assert pipeline.resolver is not None
        assert pipeline.store is not None
        assert pipeline.storage is None

    def test_create_with_sqlite(self, tmp_path):
        """Test creating a pipeline with SQLite persistence."""
        client = create_mock_client(SIMPLE_RESPONSE)
        db_path = str(tmp_path / "test.db")
        pipeline = ExtractionPipeline.create(client, model="test", db_path=db_path)
        assert pipeline.storage is not None
        assert pipeline.auto_persist is True

    def test_create_custom_threshold(self):
        """Test creating a pipeline with custom entity threshold."""
        client = create_mock_client(SIMPLE_RESPONSE)
        pipeline = ExtractionPipeline.create(
            client, model="test", entity_threshold=0.9
        )
        assert pipeline.resolver.threshold == 0.9

    def test_create_no_resolution(self):
        """Test creating a pipeline without entity resolution."""
        client = create_mock_client(SIMPLE_RESPONSE)
        pipeline = ExtractionPipeline.create(
            client, model="test", resolve_entities=False
        )
        assert pipeline.resolve_entities is False


# =============================================================================
# Ingest Tests
# =============================================================================


class TestIngest:
    """Tests for ingesting text."""

    def test_ingest_simple(self, simple_pipeline):
        """Test ingesting simple text."""
        result = simple_pipeline.ingest("Stress causes anxiety.")
        assert isinstance(result, IngestResult)
        assert result.triple_count == 1
        assert len(simple_pipeline) == 1

    def test_ingest_with_source(self, simple_pipeline):
        """Test ingesting with source provenance."""
        result = simple_pipeline.ingest(
            "Stress causes anxiety.",
            source="Psychology 101"
        )
        assert result.triples[0].source == "Psychology 101"

    def test_ingest_multiple_triples(self):
        """Test ingesting text with multiple triples."""
        client = create_mock_client(MULTI_TRIPLE_RESPONSE)
        pipeline = ExtractionPipeline.create(client)
        result = pipeline.ingest("Complex text with multiple facts.")
        assert result.triple_count == 2
        assert len(pipeline) == 2

    def test_ingest_generates_unique_ids(self):
        """Test that ingested triples get unique IDs."""
        client = create_mock_client_sequence([SIMPLE_RESPONSE, SIMPLE_RESPONSE])
        pipeline = ExtractionPipeline.create(client)

        result1 = pipeline.ingest("First text")
        result2 = pipeline.ingest("Second text")

        ids = {t.id for t in result1.triples} | {t.id for t in result2.triples}
        assert len(ids) == 2  # All unique

    def test_ingest_resolves_entities(self):
        """Test that ingestion resolves entities."""
        client = create_mock_client(SIMPLE_RESPONSE)
        pipeline = ExtractionPipeline.create(client)

        # Pre-add an entity with surface form
        pipeline.add_entity("anxiety_disorder", ["anxiety"])

        result = pipeline.ingest("Stress causes anxiety.")

        # The object should be resolved to canonical form
        assert result.triples[0].object == "anxiety_disorder"

    def test_ingest_no_resolution(self, pipeline_no_resolution):
        """Test ingestion without entity resolution."""
        result = pipeline_no_resolution.ingest("Stress causes anxiety.")
        # Entity should be normalized but not resolved
        assert result.triples[0].object == "anxiety"

    def test_ingest_batch(self):
        """Test batch ingestion."""
        responses = [SIMPLE_RESPONSE, MULTI_TRIPLE_RESPONSE]
        client = create_mock_client_sequence(responses)
        pipeline = ExtractionPipeline.create(client)

        results = pipeline.ingest_batch(["Text 1", "Text 2"], source="Book")

        assert len(results) == 2
        assert results[0].triple_count == 1
        assert results[1].triple_count == 2
        assert len(pipeline) == 3

    def test_ingest_empty_response(self):
        """Test ingesting when LLM returns no triples."""
        client = create_mock_client(EMPTY_RESPONSE)
        pipeline = ExtractionPipeline.create(client)

        result = pipeline.ingest("Text with nothing extractable.")
        assert result.triple_count == 0
        assert len(pipeline) == 0


# =============================================================================
# Query Tests
# =============================================================================


class TestQuery:
    """Tests for querying the knowledge base."""

    def test_query_supported(self):
        """Test query that is supported by KB."""
        # First ingest, then query
        ingest_response = SIMPLE_RESPONSE
        query_response = QUERY_STRESS_RESPONSE
        client = create_mock_client_sequence([ingest_response, query_response])
        pipeline = ExtractionPipeline.create(client)

        # Ingest: stress causes anxiety
        pipeline.ingest("Stress causes anxiety.")

        # Query: stress causes anxiousness (fuzzy matches anxiety)
        # Use lower thresholds for fuzzy matching
        result = pipeline.query("Does stress cause anxiousness?", match_threshold=0.4)

        assert isinstance(result, QueryResult)
        assert result.query_triple is not None
        assert result.verdict == VerificationVerdict.SUPPORTED
        assert result.is_supported is True

    def test_query_contradicted(self):
        """Test query that contradicts KB."""
        # Ingest: exercise prevents anxiety
        # Query: exercise causes anxiety (opposite predicate)
        ingest_response = PREVENTION_RESPONSE
        query_response = QUERY_EXERCISE_CAUSES
        client = create_mock_client_sequence([ingest_response, query_response])
        pipeline = ExtractionPipeline.create(client)

        pipeline.ingest("Exercise prevents anxiety.")
        result = pipeline.query("Does exercise cause anxiety?")

        assert result.verdict == VerificationVerdict.CONTRADICTED
        assert result.is_contradicted is True

    def test_query_insufficient(self):
        """Test query with no relevant knowledge."""
        # Query about something not in KB
        ingest_response = SIMPLE_RESPONSE
        # Query about depression (not in KB)
        query_depression = '''```json
{
  "triples": [
    {"id": "q1", "subject": "medication", "predicate": "prevents", "object": "depression", "negated": false}
  ]
}
```'''
        client = create_mock_client_sequence([ingest_response, query_depression])
        pipeline = ExtractionPipeline.create(client)

        pipeline.ingest("Stress causes anxiety.")
        result = pipeline.query("Does medication prevent depression?")

        assert result.verdict == VerificationVerdict.INSUFFICIENT

    def test_query_empty_extraction(self):
        """Test query when LLM returns no triple."""
        client = create_mock_client_sequence([SIMPLE_RESPONSE, EMPTY_RESPONSE])
        pipeline = ExtractionPipeline.create(client)

        pipeline.ingest("Stress causes anxiety.")
        result = pipeline.query("What?")

        assert result.query_triple is None
        assert result.verification is None
        assert result.verdict is None

    def test_query_resolves_entities(self):
        """Test that query resolves entities to canonical form."""
        ingest_response = SIMPLE_RESPONSE
        # Query with synonym
        query_response = '''```json
{
  "triples": [
    {"id": "q1", "subject": "tension", "predicate": "causes", "object": "worry", "negated": false}
  ]
}
```'''
        client = create_mock_client_sequence([ingest_response, query_response])
        pipeline = ExtractionPipeline.create(client)

        # Pre-add entities
        pipeline.add_entity("stress", ["tension", "pressure"])
        pipeline.add_entity("anxiety", ["worry", "nervousness"])

        pipeline.ingest("Stress causes anxiety.")
        result = pipeline.query("Does tension cause worry?")

        # Should resolve and match
        assert result.query_triple.subject == "stress"
        assert result.query_triple.object == "anxiety"


# =============================================================================
# Verify Tests
# =============================================================================


class TestVerify:
    """Tests for direct triple verification."""

    def test_verify_exact_match(self, simple_pipeline):
        """Test verifying an exact match triple."""
        simple_pipeline.ingest("Stress causes anxiety.")

        triple = Triple(
            id="q1",
            subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiety",
        )
        result = simple_pipeline.verify(triple)
        assert result.verdict == VerificationVerdict.SUPPORTED

    def test_verify_fuzzy_match(self, simple_pipeline):
        """Test verifying a fuzzy match triple."""
        simple_pipeline.ingest("Stress causes anxiety.")

        triple = Triple(
            id="q1",
            subject="stress",
            predicate=Predicate.CAUSES,
            object="anxiousness",  # Similar to anxiety
        )
        # Lower threshold for fuzzy match (anxiousness ~ anxiety = 0.45)
        result = simple_pipeline.verify(triple, match_threshold=0.4)
        assert result.verdict == VerificationVerdict.SUPPORTED


# =============================================================================
# Entity Management Tests
# =============================================================================


class TestEntityManagement:
    """Tests for entity management."""

    def test_add_entity(self, simple_pipeline):
        """Test adding an entity."""
        simple_pipeline.add_entity("anxiety_disorder", ["anxiety", "nervousness"])
        assert "anxiety_disorder" in simple_pipeline.list_entities()

    def test_get_entity_forms(self, simple_pipeline):
        """Test getting surface forms."""
        simple_pipeline.add_entity("anxiety_disorder", ["anxiety", "nervousness"])
        forms = simple_pipeline.get_entity_forms("anxiety_disorder")
        assert "anxiety" in forms
        assert "nervousness" in forms

    def test_list_entities(self, simple_pipeline):
        """Test listing all entities."""
        simple_pipeline.add_entity("stress")
        simple_pipeline.add_entity("anxiety")
        entities = simple_pipeline.list_entities()
        assert "stress" in entities
        assert "anxiety" in entities


# =============================================================================
# Store Access Tests
# =============================================================================


class TestStoreAccess:
    """Tests for store access methods."""

    def test_get_triple(self, simple_pipeline):
        """Test getting a triple by ID."""
        simple_pipeline.ingest("Stress causes anxiety.")
        triple = simple_pipeline.get_triple("t1")
        assert triple is not None
        assert triple.subject == "stress"

    def test_query_triples(self, simple_pipeline):
        """Test querying triples."""
        simple_pipeline.ingest("Stress causes anxiety.")
        triples = simple_pipeline.query_triples(predicate=Predicate.CAUSES)
        assert len(triples) == 1

    def test_count_triples(self, simple_pipeline):
        """Test counting triples."""
        assert simple_pipeline.count_triples() == 0
        simple_pipeline.ingest("Stress causes anxiety.")
        assert simple_pipeline.count_triples() == 1

    def test_contains(self, simple_pipeline):
        """Test __contains__ operator."""
        simple_pipeline.ingest("Stress causes anxiety.")
        assert "t1" in simple_pipeline
        assert "nonexistent" not in simple_pipeline

    def test_len(self, simple_pipeline):
        """Test __len__ operator."""
        assert len(simple_pipeline) == 0
        simple_pipeline.ingest("Stress causes anxiety.")
        assert len(simple_pipeline) == 1


# =============================================================================
# Persistence Tests
# =============================================================================


class TestPersistence:
    """Tests for SQLite persistence."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading from SQLite."""
        db_path = str(tmp_path / "test.db")
        client = create_mock_client(SIMPLE_RESPONSE)
        pipeline = ExtractionPipeline.create(client, db_path=db_path)

        # Add data
        pipeline.ingest("Stress causes anxiety.")
        pipeline.add_entity("depression", ["sadness"])
        assert len(pipeline) == 1

        # Save explicitly
        pipeline.save()

        # Create new pipeline and load
        client2 = create_mock_client(EMPTY_RESPONSE)
        pipeline2 = ExtractionPipeline.create(client2, db_path=db_path)
        pipeline2.load()

        assert len(pipeline2) == 1
        assert "depression" in pipeline2.list_entities()

    def test_auto_persist(self, tmp_path):
        """Test automatic persistence to SQLite."""
        db_path = str(tmp_path / "test.db")
        client = create_mock_client(SIMPLE_RESPONSE)
        pipeline = ExtractionPipeline.create(client, db_path=db_path)

        pipeline.ingest("Stress causes anxiety.")

        # Check storage directly
        assert len(pipeline.storage) == 1

    def test_clear(self, tmp_path):
        """Test clearing all data."""
        db_path = str(tmp_path / "test.db")
        client = create_mock_client(SIMPLE_RESPONSE)
        pipeline = ExtractionPipeline.create(client, db_path=db_path)

        pipeline.ingest("Stress causes anxiety.")
        pipeline.add_entity("test")
        assert len(pipeline) == 1

        pipeline.clear()

        assert len(pipeline) == 0
        assert len(pipeline.list_entities()) == 0
        assert len(pipeline.storage) == 0

    def test_save_without_storage_raises(self, simple_pipeline):
        """Test that save raises without SQLite storage."""
        with pytest.raises(ValueError, match="No SQLite storage"):
            simple_pipeline.save()

    def test_load_without_storage_raises(self, simple_pipeline):
        """Test that load raises without SQLite storage."""
        with pytest.raises(ValueError, match="No SQLite storage"):
            simple_pipeline.load()


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for end-to-end workflows."""

    def test_full_workflow(self, tmp_path):
        """Test complete ingest -> query workflow."""
        db_path = str(tmp_path / "test.db")

        # Ingest phase
        ingest_responses = [
            '''```json
{"triples": [
  {"id": "t1", "subject": "chronic_stress", "predicate": "causes", "object": "cortisol_elevation", "negated": false},
  {"id": "t2", "subject": "cortisol_elevation", "predicate": "causes", "object": "memory_problems", "negated": false}
]}
```''',
            '''```json
{"triples": [
  {"id": "t1", "subject": "exercise", "predicate": "prevents", "object": "stress", "negated": false},
  {"id": "t2", "subject": "meditation", "predicate": "prevents", "object": "anxiety", "negated": false}
]}
```''',
        ]

        # Query phase - query facts that ARE in the KB directly
        query_responses = [
            '''```json
{"triples": [{"id": "q1", "subject": "cortisol_elevation", "predicate": "causes", "object": "memory_problem", "negated": false}]}
```''',
            '''```json
{"triples": [{"id": "q1", "subject": "exercise", "predicate": "causes", "object": "stress", "negated": false}]}
```''',
        ]

        client = create_mock_client_sequence(ingest_responses + query_responses)
        pipeline = ExtractionPipeline.create(client, db_path=db_path)

        # Ingest knowledge
        pipeline.ingest("Chronic stress causes cortisol elevation and memory problems.")
        pipeline.ingest("Exercise prevents stress. Meditation prevents anxiety.")

        assert len(pipeline) == 4

        # Query - should be supported (fuzzy match: memory_problem vs memory_problems)
        # KB has cortisol_elevation causes memory_problems
        result1 = pipeline.query("Does cortisol elevation cause memory problems?", match_threshold=0.4)
        assert result1.verdict == VerificationVerdict.SUPPORTED

        # Query - should be contradicted (opposite predicate)
        result2 = pipeline.query("Does exercise cause stress?")
        assert result2.verdict == VerificationVerdict.CONTRADICTED

    def test_entity_resolution_workflow(self):
        """Test entity resolution in full workflow."""
        # Setup entities first
        client = create_mock_client_sequence([
            '''```json
{"triples": [{"id": "t1", "subject": "work_stress", "predicate": "causes", "object": "anxiety_disorder", "negated": false}]}
```''',
            '''```json
{"triples": [{"id": "q1", "subject": "job_stress", "predicate": "causes", "object": "worry", "negated": false}]}
```''',
        ])
        pipeline = ExtractionPipeline.create(client)

        # Pre-add entities with surface forms
        pipeline.add_entity("work_stress", ["job_stress", "occupational_stress"])
        pipeline.add_entity("anxiety_disorder", ["anxiety", "worry", "nervousness"])

        # Ingest
        pipeline.ingest("Work stress causes anxiety disorder.")

        # Query with synonyms
        result = pipeline.query("Does job stress cause worry?")

        # Should resolve and match
        assert result.query_triple.subject == "work_stress"
        assert result.query_triple.object == "anxiety_disorder"
        assert result.verdict == VerificationVerdict.SUPPORTED
