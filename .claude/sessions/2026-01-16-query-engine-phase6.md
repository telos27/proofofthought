# Session: Link-Based Architecture Phase 6 - QueryEngine Integration

**Date**: 2026-01-16

## Summary

Implemented Phase 6 of the link-based knowledge architecture: QueryEngine with PatternMatcher and EvidenceCombiner for the full query pipeline.

## Work Done

### 1. Created `PatternMatcher` Class

Matches expanded patterns against a TripleStore:

- **Entity ID matching**: Exact match when IDs available
- **Entity link matching**: Check EntityStore for pre-computed links
- **Fuzzy string fallback**: Fall back to string similarity when IDs unavailable
- **Predicate polarity detection**: Detect opposite predicates (causes ↔ prevents)
- **Effective truth computation**: Adjust truth values for match quality

### 2. Created `EvidenceCombiner` Class

Combines evidence from multiple matches using NARS revision:

- **Evidence pooling**: Pool evidence from all matches
- **Truth value revision**: Use NARS revision to combine
- **Verdict determination**: SUPPORTED/CONTRADICTED/INSUFFICIENT
- **Configurable thresholds**: Support and confidence thresholds

### 3. Created `QueryEngine` Class

Full query pipeline orchestration:

- **Entity resolution**: Resolve query entities via EntityStore
- **Query expansion**: Expand using pre-computed entity links
- **Pattern matching**: Match against TripleStore
- **Evidence combination**: Combine using NARS revision
- **Contradiction detection**: Check opposite predicates

Key methods:
```python
class QueryEngine:
    def query(self, subject: str, predicate: Predicate, obj: str, check_contradiction: bool = True) -> QueryResult
    def query_triple(self, query: QueryTriple, check_contradiction: bool = True) -> QueryResult
    def verify(self, triple: Triple) -> QueryResult
```

### 4. Created Supporting Dataclasses

- `PatternMatch`: A match between a pattern and stored triple
- `QueryResult`: Complete query result with verdict and evidence

### 5. Comprehensive Test Suite

Created `tests/unit/test_query_engine.py` with 28 tests:
- PatternMatch tests
- QueryResult tests
- PatternMatcher initialization tests
- PatternMatcher.match() tests (exact, fuzzy, opposite predicates)
- PatternMatcher.match_all() tests
- EvidenceCombiner tests (supported, contradicted, insufficient)
- QueryEngine initialization tests
- QueryEngine.query() tests (exact match, via expansion, contradiction)
- QueryEngine.query_triple() tests
- QueryEngine.verify() tests
- Integration tests (psychology scenario, empty stores)

## Files Created/Modified

**Created:**
- `z3adapter/ikr/entities/query_engine.py` - PatternMatcher, EvidenceCombiner, QueryEngine
- `tests/unit/test_query_engine.py` - 28 unit tests
- `.claude/sessions/2026-01-16-query-engine-phase6.md` - This session note

**Modified:**
- `z3adapter/ikr/entities/__init__.py` - Added QueryEngine exports
- `CLAUDE.md` - Added Phase 6 usage example

## Test Results

- 28 new QueryEngine tests: All pass
- 32 QueryExpander tests: All pass
- Total with pytest: 773 tests passing

Note: The run_tests.py script uses unittest which doesn't discover pytest-style tests. Use `python3 -m pytest tests/` for the full count.

## Usage Example

```python
from z3adapter.ikr.entities import (
    Entity, EntityLink, EntityStore, LinkType,
    QueryEngine,
)
from z3adapter.ikr.triples import Predicate, Triple, TripleStore

# Setup
entity_store = EntityStore("knowledge.db")
triple_store = TripleStore()

# Add entities and links
entity_store.add(Entity(id="e1", name="stress"))
entity_store.add(Entity(id="e2", name="chronic_stress"))
entity_store.add_link(EntityLink(
    source_id="e1", target_id="e2",
    link_type=LinkType.SIMILAR_TO, score=0.85
))

# Add triples
triple_store.add(Triple(
    id="t1", subject="chronic_stress",
    predicate=Predicate.CAUSES, object="anxiety"
))

# Create engine
engine = QueryEngine(entity_store, triple_store)

# Query via entity expansion
result = engine.query("stress", Predicate.CAUSES, "anxiety")
print(result.verdict)  # SUPPORTED (stress ~ chronic_stress)
```

## Architecture

### Query Flow

```
Query("stress", CAUSES, "anxiety")
        │
        ▼
┌───────────────────┐
│  EntityStore      │ → Resolve "stress" to Entity(e1)
│  get_or_create()  │ → Resolve "anxiety" to Entity(e3)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  QueryExpander    │ → Expand entities via links
│  expand()         │ → [stress, chronic_stress] × [anxiety, fear]
└───────────────────┘
        │
        ▼
┌───────────────────┐
│  PatternMatcher   │ → Match patterns against TripleStore
│  match_all()      │ → Find matching triples
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ EvidenceCombiner  │ → Pool evidence using NARS revision
│  combine()        │ → Compute verdict
└───────────────────┘
        │
        ▼
    QueryResult(
        verdict=SUPPORTED,
        combined_truth=TruthValue(f=0.9, c=0.7),
        matches=[...],
        explanation="Evidence supports claim..."
    )
```

### Matching Strategy

1. **Entity ID Match**: If pattern and triple have matching entity IDs, score = 1.0
2. **Entity Link Match**: If entities are linked in EntityStore, use link score
3. **Fuzzy String Match**: Fall back to lexical similarity

### Contradiction Detection

When `check_contradiction=True`:
1. Generate patterns with original predicate (e.g., CAUSES)
2. Generate patterns with opposite predicate (e.g., PREVENTS)
3. Match both sets against store
4. If opposite predicate matches with high confidence → CONTRADICTED

## Next Steps (Phase 7: Evaluation)

**Benchmark Pipeline:**
- Ingest 1-3 psychology books
- Create evaluation queries with ground truth
- Measure precision/recall/F1
- Compare with baseline (no entity expansion)

**Performance Optimization:**
- Index TripleStore by entity IDs for O(1) lookup
- Parallelize pattern matching
- Cache query expansions

**Potential Improvements:**
- Transitive link expansion (A ~ B, B ~ C → A ~ C)
- Type-based filtering (query "emotion" entities only)
- Confidence decay through link chains
