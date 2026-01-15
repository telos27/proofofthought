# Session: Triple Extraction Pipeline - Phases 3 & 4

## Summary

Implemented Phases 3 and 4 of the Triple Extraction Pipeline:
- **Phase 3**: Entity Resolution with fuzzy matching
- **Phase 4**: Verification Integration with Fuzzy-NARS

## Files Created

| File | Purpose |
|------|---------|
| `z3adapter/ikr/triples/entity_resolver.py` | EntityResolver, EntityMatch classes |
| `z3adapter/ikr/triples/verification.py` | Verification bridge to Fuzzy-NARS |
| `tests/unit/test_entity_resolver.py` | 41 unit tests |
| `tests/unit/test_triple_verification.py` | 25 unit tests |

## Files Modified

| File | Change |
|------|--------|
| `z3adapter/ikr/triples/__init__.py` | Added all new exports |

## Phase 3: Entity Resolution

### EntityMatch Dataclass
- `canonical`: The canonical entity name
- `surface_form`: Original surface form that was resolved
- `similarity`: Match confidence score in [0, 1]
- `is_new`: Whether this entity was newly added

### EntityResolver Class
Key features:
- Pluggable similarity function (default: `combined_lexical_similarity` from fuzzy_nars.py)
- Configurable match threshold (default: 0.8)
- Surface form tracking (remembers aliases for future exact matches)
- Normalization (lowercase, underscores, collapse multiples)

Methods:
- `add_entity(canonical, surface_forms)`: Register entity with aliases
- `add_surface_form(canonical, surface_form)`: Add alias to existing entity
- `resolve(surface_form, auto_add)`: Find canonical entity for surface form
- `resolve_or_none(surface_form)`: Resolve without auto-adding new entities
- `get_surface_forms(canonical)`: Get all known aliases
- `get_all_entities()`: List all canonical names
- `merge_entities(keep, merge)`: Merge two entities, keeping one as canonical
- `clear()`: Remove all entities

Resolution strategy:
1. Exact match on canonical name (after normalization)
2. Exact match on registered surface forms
3. Fuzzy match against all canonical names and surface forms
4. If best match >= threshold, use that entity and learn surface form
5. Otherwise, create new entity (if auto_add=True)

### Integration with Fuzzy-NARS
Reuses similarity functions from `z3adapter/ikr/fuzzy_nars.py`:
- `combined_lexical_similarity()` (default)
- Custom functions can be passed at init or via property setter

## Test Coverage

**Total: 41 tests passing**

Test categories:
- EntityMatch basics (4 tests)
- EntityResolver basics (10 tests)
- Normalization (6 tests)
- Exact resolution (4 tests)
- Fuzzy resolution (6 tests)
- Entity merging (5 tests)
- Custom similarity (2 tests)
- Integration scenarios (4 tests)

## Phase 4: Verification Integration

### Conversion Functions
- `triple_to_verification(triple)`: Convert Triple to VerificationTriple
  - Handles predicate enum to string conversion
  - Propagates truth values
  - Inverts frequency for negated triples
- `verification_to_triple(vt, id)`: Convert back to Triple
  - Handles unknown predicates (fallback to RELATED_TO)
  - Interprets low frequency as negation
- `store_to_kb(store)`: Convert TripleStore to list of VerificationTriples
  - Skips meta-level triples (triple references)

### Verification Functions
- `verify_triple_against_store(query, store, ...)`: Main verification entry point
  - Converts query and store to Fuzzy-NARS format
  - Runs fuzzy-NARS verification
  - Returns VerificationResult with verdict
- `verify_triples_against_store(queries, store, ...)`: Batch verification
  - Verifies multiple triples
  - Returns summary with overall verdict
  - Any contradiction → overall CONTRADICTED

### Test Coverage

**Total: 66 new tests (41 + 25)**

Phase 4 tests (25):
- Triple to verification conversion (5 tests)
- Verification to triple conversion (5 tests)
- Store to KB conversion (3 tests)
- Single triple verification (6 tests)
- Batch triple verification (4 tests)
- Integration scenarios (2 tests)

## Next Steps

**Phase 5: Persistence (SQLite)** - Add persistent storage:
- SQLite schema for triples and entities
- CRUD operations
- Surface form tracking
- Indexed queries
