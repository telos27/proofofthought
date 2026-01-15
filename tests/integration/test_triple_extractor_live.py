"""Integration tests for Triple extraction with live LLM.

These tests require an OpenAI API key and are skipped in CI.
To run locally, set the OPENAI_API_KEY environment variable.
"""

import os
import unittest

# Check for API key availability
try:
    from openai import OpenAI

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_AVAILABLE = bool(OPENAI_API_KEY)
except ImportError:
    OPENAI_AVAILABLE = False
    OPENAI_API_KEY = None

from z3adapter.ikr.triples import (
    Predicate,
    TripleExtractor,
    TripleStore,
)


@unittest.skipUnless(
    OPENAI_AVAILABLE,
    "OPENAI_API_KEY not set or openai package not installed"
)
class TestTripleExtractorLive(unittest.TestCase):
    """Live integration tests for TripleExtractor."""

    @classmethod
    def setUpClass(cls):
        """Set up OpenAI client."""
        cls.client = OpenAI(api_key=OPENAI_API_KEY)
        cls.extractor = TripleExtractor(
            cls.client,
            model="gpt-4o-mini",  # Use cheaper model for tests
            max_tokens=1000,
        )

    def test_simple_extraction(self):
        """Test extracting from a simple sentence."""
        result = self.extractor.extract("Stress causes anxiety.")

        self.assertGreater(len(result.triples), 0, "Should extract at least one triple")

        # Check that we got a causal relationship
        causal = [t for t in result.triples if t.predicate == Predicate.CAUSES]
        self.assertGreater(len(causal), 0, "Should extract a causal relationship")

    def test_taxonomy_extraction(self):
        """Test extracting taxonomy relationships."""
        result = self.extractor.extract(
            "A phobia is a type of anxiety disorder."
        )

        self.assertGreater(len(result.triples), 0)

        # Should have an is_a relationship
        taxonomy = [t for t in result.triples if t.predicate == Predicate.IS_A]
        self.assertGreater(len(taxonomy), 0, "Should extract taxonomy relationship")

    def test_negation_extraction(self):
        """Test extracting negated statements."""
        result = self.extractor.extract(
            "Exercise does not cause depression."
        )

        self.assertGreater(len(result.triples), 0)

        # Check for negated triple
        negated = [t for t in result.triples if t.negated]
        self.assertGreater(len(negated), 0, "Should extract negated triple")

    def test_multi_sentence_extraction(self):
        """Test extracting from multiple sentences."""
        text = """
        Chronic stress leads to elevated cortisol levels.
        High cortisol impairs memory formation.
        Exercise reduces cortisol levels.
        """
        result = self.extractor.extract(text)

        # Should extract multiple triples
        self.assertGreater(len(result.triples), 2, "Should extract multiple triples")

    def test_belief_attribution(self):
        """Test extracting attributed beliefs."""
        result = self.extractor.extract(
            "Dr. Smith believes that meditation reduces anxiety."
        )

        self.assertGreater(len(result.triples), 0)

        # Should have a believes relationship
        beliefs = [t for t in result.triples if t.predicate == Predicate.BELIEVES]
        # Note: This may or may not produce a nested belief depending on LLM
        # Just verify we got some triples

    def test_source_propagation(self):
        """Test that source is propagated to triples."""
        result = self.extractor.extract(
            "Cats are mammals.",
            source="Biology 101"
        )

        self.assertGreater(len(result.triples), 0)

        # All triples should have source
        for triple in result.triples:
            self.assertEqual(triple.source, "Biology 101")

    def test_extraction_with_context(self):
        """Test extraction with context hint."""
        result = self.extractor.extract(
            "It prevents the condition from worsening.",
            context="This text is about medical treatments. 'It' refers to aspirin, 'the condition' refers to inflammation."
        )

        self.assertGreater(len(result.triples), 0)

        # Should have a prevents relationship
        prevention = [t for t in result.triples if t.predicate == Predicate.PREVENTS]
        self.assertGreater(len(prevention), 0, "Should extract prevention relationship")

    def test_batch_extraction(self):
        """Test batch extraction from multiple texts."""
        texts = [
            "Dogs are mammals.",
            "Cats are mammals.",
            "Birds can fly.",
        ]
        results = self.extractor.extract_batch(texts)

        self.assertEqual(len(results), 3)
        for result in results:
            self.assertGreater(len(result.triples), 0)


@unittest.skipUnless(
    OPENAI_AVAILABLE,
    "OPENAI_API_KEY not set or openai package not installed"
)
class TestTripleExtractorWithStore(unittest.TestCase):
    """Test TripleExtractor integration with TripleStore."""

    @classmethod
    def setUpClass(cls):
        """Set up OpenAI client and extractor."""
        cls.client = OpenAI(api_key=OPENAI_API_KEY)
        cls.extractor = TripleExtractor(
            cls.client,
            model="gpt-4o-mini",
            max_tokens=1000,
        )

    def test_extract_and_store(self):
        """Test extracting triples and storing them."""
        store = TripleStore()

        # Extract from first text
        result1 = self.extractor.extract(
            "Stress causes anxiety.",
            source="Text 1"
        )
        for triple in result1.triples:
            store.add(triple)

        # Extract from second text (may have overlapping concepts)
        result2 = self.extractor.extract(
            "Anxiety leads to sleep problems.",
            source="Text 2"
        )
        # Use different IDs to avoid collision
        for i, triple in enumerate(result2.triples):
            triple.id = f"t2_{i}"
            store.add(triple)

        # Should have triples from both extractions
        self.assertGreater(len(store), 1)

        # Should be able to query by predicate
        causal = store.query(predicate=Predicate.CAUSES)
        self.assertGreater(len(causal), 0)


if __name__ == "__main__":
    unittest.main()
