# Session: 2026-01-15 - Fuzzy-NARS Integration

## Summary

Integrated Fuzzy-NARS verification from `pysem/psychology-knowledge` into proofofthought. This enables similarity-based triple verification with NARS evidence combination, providing richer uncertainty handling than Z3 soft constraints alone.

## What was done

### 1. Enhanced TruthValue (`z3adapter/ikr/schema.py`)

Added `to_evidence()` method to convert truth values back to evidence counts:
```python
def to_evidence(self, k: float = 1.0) -> tuple[float, float]:
    """Convert truth value back to evidence counts.
    Inverse of from_evidence(). Useful for NARS revision operations.
    """
```

### 2. Created `z3adapter/ikr/fuzzy_nars.py`

Adapted from `pysem/psychology-knowledge/knowledge_store/fuzzy_nars.py`:

**Data structures:**
- `VerificationTriple` - (subject, predicate, object) with optional truth value
- `UnificationResult` - result of fuzzy matching between triples
- `VerificationVerdict` - SUPPORTED / CONTRADICTED / INSUFFICIENT
- `VerificationResult` - complete verification result with matches

**Predicate polarity:**
- `PREDICATE_OPPOSITES` - mapping of semantic opposites (causes↔prevents, increases↔decreases)
- `get_predicate_polarity()` - detect contradicting predicates

**Similarity functions:**
- `lexical_similarity()` - Levenshtein-based
- `jaccard_word_similarity()` - word overlap
- `combined_lexical_similarity()` - max of both
- `make_embedding_similarity()` - factory for embedding-based
- `make_hybrid_similarity()` - lexical + embedding

**Core algorithms:**
- `fuzzy_nars_unify()` - match query triple against KB triple with fuzzy matching
- `revise()` - NARS evidence pooling for two truth values
- `revise_multiple()` - combine multiple sources

**Verification pipeline:**
- `verify_triple()` - verify one triple against KB
- `verify_answer()` - verify multiple triples with summary

### 3. Updated exports (`z3adapter/ikr/__init__.py`)

Added all fuzzy_nars exports to `__all__`.

### 4. Created tests (`tests/unit/test_fuzzy_nars.py`)

43 tests covering:
- TruthValue (including new to_evidence)
- VerificationTriple
- Similarity functions
- Predicate polarity
- Fuzzy unification
- NARS revision
- Verification pipeline
- Integration tests

## Files Created
- `z3adapter/ikr/fuzzy_nars.py` - Fuzzy-NARS verification module
- `tests/unit/test_fuzzy_nars.py` - 43 tests
- `.claude/sessions/2026-01-15-fuzzy-nars-integration.md` - This session note

## Files Modified
- `z3adapter/ikr/schema.py` - Added `to_evidence()` to TruthValue
- `z3adapter/ikr/__init__.py` - Added fuzzy_nars exports

## Test Results
All 253 tests pass (210 original + 43 new).

## Usage Example

```python
from z3adapter.ikr import (
    TruthValue,
    VerificationTriple,
    verify_triple,
    verify_answer,
    combined_lexical_similarity,
    VerificationVerdict,
)

# Create a knowledge base
kb = [
    VerificationTriple("phobia", "is_a", "anxiety_disorder", TruthValue(1.0, 0.9)),
    VerificationTriple("stress", "causes", "cortisol_release", TruthValue(0.95, 0.9)),
    VerificationTriple("relaxation", "prevents", "stress", TruthValue(0.9, 0.85)),
]

# Verify a claim (supported)
query = VerificationTriple("phobia", "is_a", "disorder")
result = verify_triple(query, kb, combined_lexical_similarity)
print(result.verdict)  # VerificationVerdict.SUPPORTED

# Verify a contradicting claim
query2 = VerificationTriple("relaxation", "causes", "stress")
result2 = verify_triple(query2, kb, combined_lexical_similarity)
print(result2.verdict)  # VerificationVerdict.CONTRADICTED

# Verify multiple answer triples
answer = [
    VerificationTriple("phobias", "is_a", "anxiety_disorder"),  # fuzzy match
    VerificationTriple("gravity", "causes", "falling"),  # no evidence
]
result3 = verify_answer(answer, kb, combined_lexical_similarity)
print(result3["summary"]["overall_verdict"])  # depends on evidence
```

## Key Features

| Feature | Description |
|---------|-------------|
| Fuzzy matching | "phobias" matches "phobia", "anxiety_disorders" matches "anxiety_disorder" |
| Predicate polarity | "causes" vs "prevents" detected as contradiction |
| Evidence pooling | NARS revision combines multiple matching KB triples |
| Confidence threshold | Low-confidence matches marked as INSUFFICIENT |
| Embedding support | `make_embedding_similarity()` for semantic matching |

## Difference from Z3 Soft Constraints

| Aspect | Z3 Soft Constraints | Fuzzy-NARS |
|--------|---------------------|------------|
| Matching | Exact symbolic | Fuzzy (lexical/semantic) |
| Output | Single model | Verdict + confidence |
| Contradiction | Implicit (weight balance) | Explicit (polarity detection) |
| Evidence combination | MaxSAT | NARS revision |

Both approaches are now available in proofofthought.
