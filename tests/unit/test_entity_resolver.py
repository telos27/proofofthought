"""Unit tests for EntityResolver."""

import pytest

from z3adapter.ikr.triples.entity_resolver import EntityMatch, EntityResolver


class TestEntityMatch:
    """Tests for EntityMatch dataclass."""

    def test_basic_match(self):
        """Test basic EntityMatch creation."""
        match = EntityMatch(
            canonical="working_memory",
            surface_form="Working Memory",
            similarity=0.95,
        )
        assert match.canonical == "working_memory"
        assert match.surface_form == "Working Memory"
        assert match.similarity == 0.95
        assert match.is_new is False

    def test_new_entity_match(self):
        """Test EntityMatch with is_new flag."""
        match = EntityMatch(
            canonical="depression",
            surface_form="Depression",
            similarity=1.0,
            is_new=True,
        )
        assert match.is_new is True

    def test_repr(self):
        """Test EntityMatch string representation."""
        match = EntityMatch("anxiety", "Anxiety", 0.9)
        assert "anxiety" in repr(match)
        assert "Anxiety" in repr(match)
        assert "0.9" in repr(match)

    def test_repr_new(self):
        """Test EntityMatch repr shows 'new' flag."""
        match = EntityMatch("stress", "Stress", 1.0, is_new=True)
        assert "(new)" in repr(match)


class TestEntityResolverBasics:
    """Tests for EntityResolver basic operations."""

    def test_default_threshold(self):
        """Test default threshold is 0.8."""
        resolver = EntityResolver()
        assert resolver.threshold == 0.8

    def test_custom_threshold(self):
        """Test custom threshold."""
        resolver = EntityResolver(threshold=0.9)
        assert resolver.threshold == 0.9

    def test_add_entity(self):
        """Test adding an entity."""
        resolver = EntityResolver()
        resolver.add_entity("anxiety_disorder")
        assert "anxiety_disorder" in resolver
        assert len(resolver) == 1

    def test_add_entity_with_surface_forms(self):
        """Test adding entity with surface forms."""
        resolver = EntityResolver()
        resolver.add_entity("working_memory", ["WM", "short-term memory"])
        assert "working_memory" in resolver
        forms = resolver.get_surface_forms("working_memory")
        assert "wm" in forms  # normalized
        assert "short_term_memory" in forms  # normalized

    def test_add_surface_form(self):
        """Test adding surface form to existing entity."""
        resolver = EntityResolver()
        resolver.add_entity("stress")
        resolver.add_surface_form("stress", "psychological stress")
        forms = resolver.get_surface_forms("stress")
        assert "psychological_stress" in forms

    def test_add_surface_form_missing_entity(self):
        """Test adding surface form to non-existent entity raises error."""
        resolver = EntityResolver()
        with pytest.raises(KeyError):
            resolver.add_surface_form("nonexistent", "form")

    def test_get_all_entities(self):
        """Test getting all entities."""
        resolver = EntityResolver()
        resolver.add_entity("stress")
        resolver.add_entity("anxiety")
        resolver.add_entity("depression")
        entities = resolver.get_all_entities()
        assert set(entities) == {"stress", "anxiety", "depression"}

    def test_clear(self):
        """Test clearing all entities."""
        resolver = EntityResolver()
        resolver.add_entity("stress")
        resolver.add_entity("anxiety")
        resolver.clear()
        assert len(resolver) == 0

    def test_len(self):
        """Test __len__."""
        resolver = EntityResolver()
        assert len(resolver) == 0
        resolver.add_entity("stress")
        assert len(resolver) == 1
        resolver.add_entity("anxiety")
        assert len(resolver) == 2

    def test_contains(self):
        """Test __contains__."""
        resolver = EntityResolver()
        resolver.add_entity("stress")
        assert "stress" in resolver
        assert "anxiety" not in resolver


class TestNormalization:
    """Tests for entity name normalization."""

    def test_lowercase(self):
        """Test lowercasing."""
        resolver = EntityResolver()
        resolver.add_entity("STRESS")
        assert "stress" in resolver
        assert "STRESS" in resolver  # __contains__ normalizes

    def test_spaces_to_underscores(self):
        """Test spaces converted to underscores."""
        resolver = EntityResolver()
        resolver.add_entity("working memory")
        assert "working_memory" in resolver
        assert "working memory" in resolver

    def test_hyphens_to_underscores(self):
        """Test hyphens converted to underscores."""
        resolver = EntityResolver()
        resolver.add_entity("short-term")
        assert "short_term" in resolver

    def test_strip_whitespace(self):
        """Test whitespace is stripped."""
        resolver = EntityResolver()
        resolver.add_entity("  stress  ")
        assert "stress" in resolver

    def test_collapse_underscores(self):
        """Test multiple underscores are collapsed."""
        resolver = EntityResolver()
        resolver.add_entity("working__memory")
        forms = resolver.get_surface_forms("working_memory")
        assert "working_memory" in forms

    def test_combined_normalization(self):
        """Test combined normalization."""
        resolver = EntityResolver()
        resolver.add_entity("  Working - Memory  ")
        assert "working_memory" in resolver


class TestExactResolution:
    """Tests for exact match resolution."""

    def test_exact_canonical_match(self):
        """Test exact match on canonical name."""
        resolver = EntityResolver()
        resolver.add_entity("anxiety_disorder")
        match = resolver.resolve("anxiety_disorder")
        assert match.canonical == "anxiety_disorder"
        assert match.similarity == 1.0
        assert match.is_new is False

    def test_exact_match_case_insensitive(self):
        """Test exact match is case-insensitive."""
        resolver = EntityResolver()
        resolver.add_entity("anxiety_disorder")
        match = resolver.resolve("Anxiety_Disorder")
        assert match.canonical == "anxiety_disorder"
        assert match.similarity == 1.0

    def test_exact_match_normalized(self):
        """Test exact match with normalization."""
        resolver = EntityResolver()
        resolver.add_entity("working_memory")
        match = resolver.resolve("Working Memory")
        assert match.canonical == "working_memory"
        assert match.similarity == 1.0

    def test_exact_match_surface_form(self):
        """Test exact match on registered surface form."""
        resolver = EntityResolver()
        resolver.add_entity("working_memory", ["WM"])
        match = resolver.resolve("WM")
        assert match.canonical == "working_memory"
        assert match.similarity == 1.0


class TestFuzzyResolution:
    """Tests for fuzzy match resolution."""

    def test_fuzzy_match_similar(self):
        """Test fuzzy match on similar entity."""
        resolver = EntityResolver(threshold=0.7)
        resolver.add_entity("anxiety_disorder")
        # "anxiety disorders" is very similar
        match = resolver.resolve("anxiety_disorders")
        assert match.canonical == "anxiety_disorder"
        assert match.similarity > 0.7
        assert match.is_new is False

    def test_fuzzy_match_learns_surface_form(self):
        """Test fuzzy match learns new surface form."""
        resolver = EntityResolver(threshold=0.6)  # Lower threshold for "anxieties" -> "anxiety" (0.667 similarity)
        resolver.add_entity("anxiety")
        resolver.resolve("anxieties")  # fuzzy match, learns this form
        # Second resolution should be exact
        match = resolver.resolve("anxieties")
        assert match.canonical == "anxiety"
        assert match.similarity == 1.0  # exact after learning

    def test_new_entity_below_threshold(self):
        """Test new entity created when below threshold."""
        resolver = EntityResolver(threshold=0.9)
        resolver.add_entity("stress")
        match = resolver.resolve("depression")
        assert match.canonical == "depression"
        assert match.is_new is True
        assert "depression" in resolver

    def test_no_auto_add(self):
        """Test resolve with auto_add=False."""
        resolver = EntityResolver()
        resolver.add_entity("stress")
        match = resolver.resolve("depression", auto_add=False)
        assert match.is_new is True
        assert "depression" not in resolver

    def test_resolve_or_none_found(self):
        """Test resolve_or_none when match found."""
        resolver = EntityResolver()
        resolver.add_entity("stress")
        match = resolver.resolve_or_none("stress")
        assert match is not None
        assert match.canonical == "stress"

    def test_resolve_or_none_not_found(self):
        """Test resolve_or_none when no match found."""
        resolver = EntityResolver(threshold=0.9)
        resolver.add_entity("stress")
        match = resolver.resolve_or_none("depression")
        assert match is None


class TestMergeEntities:
    """Tests for entity merging."""

    def test_merge_entities(self):
        """Test merging two entities."""
        resolver = EntityResolver()
        resolver.add_entity("anxiety", ["anxious"])
        resolver.add_entity("anxiety_disorder", ["disorder"])
        forms_added = resolver.merge_entities("anxiety_disorder", "anxiety")
        assert "anxiety_disorder" in resolver
        assert "anxiety" not in resolver
        assert forms_added > 0

    def test_merge_entities_combines_forms(self):
        """Test merge combines surface forms."""
        resolver = EntityResolver()
        resolver.add_entity("stress", ["chronic stress"])
        resolver.add_entity("psychological_stress", ["mental stress"])
        resolver.merge_entities("stress", "psychological_stress")
        forms = resolver.get_surface_forms("stress")
        assert "chronic_stress" in forms
        assert "mental_stress" in forms
        assert "psychological_stress" in forms

    def test_merge_missing_keep(self):
        """Test merge with missing 'keep' entity."""
        resolver = EntityResolver()
        resolver.add_entity("anxiety")
        with pytest.raises(KeyError):
            resolver.merge_entities("nonexistent", "anxiety")

    def test_merge_missing_merge(self):
        """Test merge with missing 'merge' entity."""
        resolver = EntityResolver()
        resolver.add_entity("anxiety")
        with pytest.raises(KeyError):
            resolver.merge_entities("anxiety", "nonexistent")

    def test_merge_same_entity(self):
        """Test merge same entity returns 0."""
        resolver = EntityResolver()
        resolver.add_entity("stress")
        forms_added = resolver.merge_entities("stress", "stress")
        assert forms_added == 0


class TestCustomSimilarity:
    """Tests for custom similarity functions."""

    def test_custom_similarity_function(self):
        """Test using a custom similarity function."""

        def always_one(a: str, b: str) -> float:
            """Similarity that always returns 1.0."""
            return 1.0

        resolver = EntityResolver(threshold=0.5)
        resolver.similarity_fn = always_one
        resolver.add_entity("stress")
        # Even very different strings should match with always_one
        match = resolver.resolve("xyz")
        assert match.canonical == "stress"  # matched due to always_one
        assert match.is_new is False

    def test_custom_similarity_function_init(self):
        """Test passing similarity function at init."""

        def custom_sim(a: str, b: str) -> float:
            return 0.5 if a != b else 1.0

        resolver = EntityResolver(threshold=0.4, _similarity_fn=custom_sim)
        resolver.add_entity("stress")
        match = resolver.resolve("anxiety")
        assert match.canonical == "stress"  # 0.5 >= 0.4 threshold


class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_psychology_terms(self):
        """Test resolution of psychology terms."""
        resolver = EntityResolver(threshold=0.8)
        resolver.add_entity("working_memory", ["WM", "short-term memory"])
        resolver.add_entity("long_term_memory", ["LTM"])
        resolver.add_entity("classical_conditioning", ["Pavlovian conditioning"])

        # Exact matches
        assert resolver.resolve("WM").canonical == "working_memory"
        assert resolver.resolve("LTM").canonical == "long_term_memory"

        # Fuzzy matches
        m1 = resolver.resolve("short term memories")
        assert m1.canonical == "working_memory"

        # Case variations
        assert resolver.resolve("CLASSICAL CONDITIONING").canonical == "classical_conditioning"

    def test_incremental_learning(self):
        """Test resolver learns from multiple resolutions."""
        resolver = EntityResolver(threshold=0.75)
        resolver.add_entity("anxiety")

        # First fuzzy match learns surface form
        resolver.resolve("anxieties")
        # Second resolution is exact
        match = resolver.resolve("anxieties")
        assert match.similarity == 1.0

    def test_build_vocabulary_from_text(self):
        """Test building vocabulary incrementally."""
        resolver = EntityResolver(threshold=0.85)

        # Simulate extracting entities from text
        entities_from_text = [
            "stress",
            "Stress",  # should match existing
            "chronic stress",  # might be new or match
            "anxiety",
            "depression",
            "major depression",  # might match depression
        ]

        for entity in entities_from_text:
            resolver.resolve(entity)

        # Should have collapsed some entities
        assert "stress" in resolver
        assert len(resolver) < len(entities_from_text)

    def test_triple_entity_resolution(self):
        """Test resolving entities from triples."""
        resolver = EntityResolver(threshold=0.8)

        # Pre-populate with known entities
        resolver.add_entity("chronic_stress", ["prolonged stress"])
        resolver.add_entity("cortisol", ["cortisol levels", "high cortisol"])
        resolver.add_entity("memory_impairment", ["memory problems"])

        # Simulate triple extraction results
        extracted_triples = [
            {"subject": "Chronic Stress", "object": "cortisol levels"},  # exact match on surface form
            {"subject": "high cortisol", "object": "memory problems"},  # exact match on surface form
        ]

        resolved_triples = []
        for triple in extracted_triples:
            resolved_triples.append({
                "subject": resolver.resolve(triple["subject"]).canonical,
                "object": resolver.resolve(triple["object"]).canonical,
            })

        # Check resolution
        assert resolved_triples[0]["subject"] == "chronic_stress"
        assert resolved_triples[0]["object"] == "cortisol"
        assert resolved_triples[1]["subject"] == "cortisol"
        assert resolved_triples[1]["object"] == "memory_impairment"
