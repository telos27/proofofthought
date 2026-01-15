# Session: Triple Extraction Pipeline - Phases 5 & 6

## Summary

Implemented Phases 5 and 6 of the Triple Extraction Pipeline:
- **Phase 5**: SQLite Persistence (58 tests)
- **Phase 6**: End-to-End Pipeline (34 tests)

Also fixed a bug in Phase 5 where `SQLiteTripleStorage` was evaluating to `False` in boolean context because `__len__` returns 0 for empty storage.

## Files Created

| File | Purpose |
|------|---------|
| `z3adapter/ikr/triples/storage.py` | SQLiteTripleStorage class |
| `z3adapter/ikr/triples/pipeline.py` | ExtractionPipeline class |
| `tests/unit/test_triple_storage.py` | 58 unit tests for storage |
| `tests/unit/test_pipeline.py` | 34 unit tests for pipeline |

## Files Modified

| File | Change |
|------|--------|
| `z3adapter/ikr/triples/__init__.py` | Added all new exports |

## Phase 5: SQLite Persistence

### SQLiteTripleStorage Class

Key features:
- In-memory (`:memory:`) or file-based storage
- Persistent connections for in-memory databases
- Full CRUD for triples and entities with surface forms
- Import/export to TripleStore and EntityResolver

Bug fixed: Added `__bool__` method to return `True` always. Without this, empty storage evaluated to `False` in boolean context (due to `__len__` returning 0), causing `if self.auto_persist and self.storage` to fail.

## Phase 6: End-to-End Pipeline

### ExtractionPipeline Class

Orchestrates the full text → verified triples workflow:
1. **Extract**: Use LLM to extract triples from text
2. **Resolve**: Normalize entities to canonical forms
3. **Store**: Persist to in-memory and/or SQLite storage
4. **Verify**: Check claims against stored knowledge

### Key Methods

- `ingest(text, source)` → IngestResult
  - Extract triples via LLM
  - Resolve entities
  - Store in memory and SQLite

- `query(question)` → QueryResult
  - Extract query triple via LLM
  - Resolve entities
  - Verify against stored knowledge
  - Returns verdict (SUPPORTED/CONTRADICTED/INSUFFICIENT)

- `verify(triple)` → VerificationResult
  - Direct verification without LLM extraction

### Result Types

```python
@dataclass
class IngestResult:
    triples: list[Triple]
    extraction_result: ExtractionResult
    entities_resolved: int
    new_entities: int

@dataclass
class QueryResult:
    question: str
    query_triple: Optional[Triple]
    verification: Optional[VerificationResult]
    extraction_result: ExtractionResult

    @property
    def verdict(self) -> Optional[VerificationVerdict]: ...
    @property
    def is_supported(self) -> bool: ...
    @property
    def is_contradicted(self) -> bool: ...
```

### Usage Example

```python
from openai import OpenAI
from z3adapter.ikr.triples import ExtractionPipeline

# Create pipeline with persistence
client = OpenAI()
pipeline = ExtractionPipeline.create(
    client,
    model="gpt-4o",
    db_path="knowledge.db"
)

# Ingest knowledge
pipeline.ingest("Stress causes anxiety.", source="Psychology 101")
pipeline.ingest("Exercise prevents stress.", source="Health Guide")

# Query the knowledge base
result = pipeline.query("Does stress cause anxiousness?", match_threshold=0.4)
print(result.verdict)  # SUPPORTED (fuzzy match)

result2 = pipeline.query("Does exercise cause stress?")
print(result2.verdict)  # CONTRADICTED (opposite predicate)
```

## Test Coverage

- **Phase 5 (storage)**: 58 tests
- **Phase 6 (pipeline)**: 34 tests
- **Total triple module**: 177 tests passing

## Pipeline Complete

The triple extraction pipeline is now complete:

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Schema (Triple, TripleStore, Predicate) | ✓ |
| 2 | LLM Extractor | ✓ |
| 3 | Entity Resolution | ✓ |
| 4 | Verification Integration | ✓ |
| 5 | SQLite Persistence | ✓ |
| 6 | End-to-End Pipeline | ✓ |

## Next Steps (Optional)

**Phase 7: Cleanup** - Optional tasks:
- Remove domain-specific KB files if no longer needed
- Update CLAUDE.md documentation
