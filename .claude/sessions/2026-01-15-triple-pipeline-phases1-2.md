# Session: Triple Extraction Pipeline - Phases 1 & 2

## Summary

Implemented Phases 1 and 2 of the Triple Extraction Pipeline:
- **Phase 1**: Core data model with `Triple`, `TripleStore`, and `Predicate` types
- **Phase 2**: LLM-based triple extraction with `TripleExtractor`

## Files Created

| File | Purpose |
|------|---------|
| `z3adapter/ikr/triples/__init__.py` | Package exports |
| `z3adapter/ikr/triples/schema.py` | Triple, TripleStore, Predicate |
| `z3adapter/ikr/triples/extractor.py` | TripleExtractor, ExtractionResult |
| `tests/unit/test_triple_schema.py` | 37 unit tests |
| `tests/unit/test_triple_extractor.py` | 23 unit tests (mocked LLM) |
| `tests/integration/test_triple_extractor_live.py` | 9 integration tests (live LLM) |

## Phase 1: Core Data Model

### Predicate Enum
7 generic predicates following Wikidata philosophy:
- `is_a` - Taxonomy
- `part_of` - Structure
- `has` - Attributes
- `causes` / `prevents` - Causation (with opposite mapping)
- `believes` - Epistemic
- `related_to` - Catch-all

### Triple Dataclass
- Core fields: `id`, `subject`, `predicate`, `object`
- Optional: `negated`, `truth` (NARS TruthValue), `source`, `surface_form`
- Triple reference support: `t:` prefix for reification
- Hashable by ID for set operations

### TripleStore
- In-memory store with O(1) lookup by ID
- Indexed queries by subject, predicate, object
- `resolve()` method for dereferencing triple references
- Replace-on-add semantics (same ID replaces existing)

## Phase 2: Triple Extraction

### TripleExtractor
- OpenAI-compatible LLM client interface
- System prompt with 7 predicates and JSON output format
- Entity normalization (lowercase, underscores)
- Source/provenance propagation
- Batch extraction support

### ExtractionResult
- List of extracted triples
- Raw LLM response for debugging
- Optional source metadata

### Features
- JSON parsing from markdown code blocks
- Invalid predicate fallback to `related_to`
- Missing field handling (skip invalid triples)
- Truth value parsing (NARS frequency/confidence)
- Triple reference preservation (not normalized)

## Test Coverage

**Total: 69 tests passing**

Phase 1 (37 tests):
- Predicate enum validation
- Triple creation and reference detection
- TripleStore CRUD operations
- Index-based queries
- Integration scenarios (nested beliefs, negation, taxonomy chains)

Phase 2 (32 tests):
- Basic extraction with mocked LLM
- Multi-triple and nested belief extraction
- Edge cases (malformed JSON, missing fields, invalid predicates)
- Entity normalization
- Batch extraction
- LLM error handling
- Live LLM integration tests

## Next Steps

**Phase 3: Entity Resolution** - Fuzzy entity matching with:
- `EntityResolver` class
- Reuse existing `combined_lexical_similarity()` from fuzzy_nars.py
- Surface form tracking
- Configurable threshold
