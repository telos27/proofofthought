# Session: Link-Based Architecture Phase 5 - QueryExpander

**Date**: 2026-01-16

## Summary

Implemented Phase 5 of the link-based knowledge architecture: QueryExpander for expanding queries using entity similarity links.

## Work Done

### 1. Created `QueryExpander` Class

New class in `z3adapter/ikr/entities/query_expander.py`:

- **Query expansion**: Expand query triples using pre-computed entity similarity links
- **Cross-product generation**: Generate all combinations of subject/object expansions
- **Confidence calculation**: Combined confidence = subject_score × object_score × predicate_score
- **Predicate expansion**: Optional expansion to semantically similar predicates
- **Opposite patterns**: Generate patterns with opposite predicates for contradiction detection

Key methods:
```python
class QueryExpander:
    def expand(self, query: QueryTriple) -> list[ExpandedPattern]
    def expand_by_name(self, subject: str, predicate: Predicate, obj: str, ...) -> list[ExpandedPattern]
    def get_opposite_patterns(self, patterns: list[ExpandedPattern]) -> list[ExpandedPattern]
```

### 2. Created `ExpandedPattern` Dataclass

Represents a query pattern with confidence scores:
- `subject_id`, `subject_name`: Subject entity info
- `predicate`: Predicate (possibly expanded)
- `object_id`, `object_name`: Object entity info
- `confidence`: Combined confidence score [0, 1]
- `subject_score`, `object_score`, `predicate_score`: Component scores for explainability
- `is_original`: Flag for the original query pattern

### 3. Created `QueryTriple` Dataclass

Simple dataclass to represent queries to expand:
- `subject_id`, `subject_name`, `predicate`, `object_id`, `object_name`

### 4. Predicate Similarity

Added `PREDICATE_SIMILARITY` constant mapping predicates to similar predicates with scores:
- `CAUSES` → `RELATED_TO` (0.5)
- `PART_OF` ↔ `HAS` (0.6) - inverse relationship
- `RELATED_TO` → no expansion (most generic)

### 5. Comprehensive Test Suite

Created `tests/unit/test_query_expander.py` with 32 tests:
- ExpandedPattern tests
- QueryTriple tests
- Predicate similarity tests
- QueryExpander initialization tests
- Expand tests (single entity, subject links, object links, cross-product)
- Confidence calculation tests
- Sorting and limiting tests
- Predicate expansion tests
- expand_by_name convenience method tests
- Opposite pattern generation tests
- Integration tests (psychology domain, empty store, bidirectional links)

## Files Created/Modified

**Created:**
- `z3adapter/ikr/entities/query_expander.py` - QueryExpander, ExpandedPattern, QueryTriple
- `tests/unit/test_query_expander.py` - 32 unit tests
- `.claude/sessions/2026-01-16-query-expander-phase5.md` - This session note

**Modified:**
- `z3adapter/ikr/entities/__init__.py` - Added QueryExpander exports
- `CLAUDE.md` - Updated implemented components, added Phase 5 usage example

## Test Results

- 32 new QueryExpander tests: All pass
- Total: 219 tests passing

## Usage Example

```python
from z3adapter.ikr.entities import (
    Entity, EntityLink, EntityStore, LinkType,
    QueryExpander, QueryTriple,
)
from z3adapter.ikr.triples import Predicate

# Setup
store = EntityStore("knowledge.db")
store.add(Entity(id="e1", name="anxiety"))
store.add(Entity(id="e2", name="stress"))
store.add(Entity(id="e3", name="memory"))
store.add_link(EntityLink(source_id="e1", target_id="e2", link_type=LinkType.SIMILAR_TO, score=0.85))

expander = QueryExpander(store, min_score=0.5, max_expansions=20)

# Expand query
query = QueryTriple(
    subject_id="e1", subject_name="anxiety",
    predicate=Predicate.CAUSES,
    object_id="e3", object_name="memory",
)
patterns = expander.expand(query)

# Returns:
#   (anxiety, causes, memory) conf=1.0 (original)
#   (stress, causes, memory) conf=0.85 (expanded)

# Or use convenience method
patterns = expander.expand_by_name("chronic_stress", Predicate.CAUSES, "memory_impairment")

# Generate opposite patterns for contradiction detection
opposites = expander.get_opposite_patterns(patterns)
```

## Next Steps (Phase 6: Integration)

**Updated Query Pipeline:**
- Integrate QueryExpander with TripleStore matching
- Add pattern matching against stored triples
- Combine evidence using NARS truth propagation
- End-to-end query flow: Query → Expand → Match → Combine → Result

**Components needed:**
- `PatternMatcher`: Match expanded patterns against TripleStore
- `EvidenceCombiner`: Combine matching evidence with NARS revision
- Updated `ExtractionPipeline.query()` to use expansion

## Design Notes

### Expansion Algorithm

```
1. Get subject expansions
   └─ Original subject (score=1.0)
   └─ Similar entities via get_similar()

2. Get object expansions
   └─ Original object (score=1.0)
   └─ Similar entities via get_similar()

3. Get predicate expansions (if enabled)
   └─ Original predicate (score=1.0)
   └─ Similar predicates from PREDICATE_SIMILARITY

4. Generate cross-product
   └─ For each (subj, obj, pred) combination:
      └─ confidence = subj_score × obj_score × pred_score
      └─ Filter by min_score

5. Sort by confidence, limit to max_expansions
```

### Confidence Decay

Confidence decays multiplicatively through expansions:
- Subject expansion 0.85 × Object expansion 0.80 = 0.68 combined
- This models uncertainty accumulation through inference chains

### Contradiction Detection

The `get_opposite_patterns()` method enables contradiction detection:
- Query: "Does exercise cause stress?"
- Check both: `(exercise, causes, stress)` and `(exercise, prevents, stress)`
- If KB contains `(exercise, prevents, stress)`, the query is contradicted
