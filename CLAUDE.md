# CLAUDE.md

This file provides guidance for AI assistants working with the ProofOfThought codebase.

## Project Overview

ProofOfThought is a neurosymbolic reasoning system that combines LLMs with Z3 theorem proving. It translates natural language questions into formal logic programs, executes them with Z3, and returns verified answers.

## Quick Reference

```bash
# Run tests (no API key required)
python run_tests.py

# Run a single test file
python -m pytest tests/unit/test_sort_manager.py -v

# Run benchmarks (requires LLM API key)
python experiments_pipeline.py

# Format code
black .

# Lint code
ruff check .

# Type check
mypy z3adapter/
```

## Project Structure

```
proofofthought/
├── z3adapter/                 # Main package (import name)
│   ├── reasoning/             # High-level API (ProofOfThought, EvaluationPipeline)
│   ├── backends/              # Execution backends (SMT2, JSON, IKR, NARS-Datalog)
│   ├── ikr/                   # Intermediate Knowledge Representation
│   │   ├── nars_datalog/      # NARS-Datalog engine
│   │   │   ├── kb/            # Knowledge base modules (JSON)
│   │   │   ├── engine.py      # Semi-naive evaluation engine
│   │   │   ├── fact_store.py  # Indexed fact storage
│   │   │   ├── rule.py        # Rule compilation
│   │   │   └── kb_loader.py   # KB module loader
│   │   ├── triples/           # Triple extraction pipeline
│   │   ├── entities/          # Entity storage for link-based architecture
│   │   │   ├── schema.py      # Entity, EntityLink, LinkType
│   │   │   ├── store.py       # EntityStore (SQLite backend)
│   │   │   └── vector_index.py # VectorIndex (FAISS ANN search)
│   │   ├── fuzzy_nars.py      # Fuzzy-NARS verification
│   │   └── schema.py          # IKR Pydantic schema
│   ├── postprocessors/        # Enhancement techniques (self-refine, etc.)
│   ├── dsl/                   # DSL components (sorts, expressions)
│   ├── solvers/               # Z3 solver wrappers
│   ├── verification/          # Result verification
│   ├── security/              # Input validation
│   ├── optimization/          # Z3 optimization
│   ├── interpreter.py         # JSON DSL interpreter
│   └── cli.py                 # Command-line interface
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── fixtures/              # Test data
├── examples/                  # Usage examples
├── benchmark/                 # Benchmark scripts
├── data/                      # Benchmark datasets
└── docs/                      # MkDocs documentation
```

## Key Concepts

### Package vs Import Name
- **PyPI package**: `proofofthought`
- **Import name**: `z3adapter`

```python
# Correct
from z3adapter.reasoning import ProofOfThought

# Wrong
from proofofthought import ...
```

### Four Backends
1. **SMT2** (default): Standard SMT-LIB 2.0 format, executed via Z3 CLI
2. **JSON**: Custom DSL interpreted via Python Z3 API
3. **IKR**: Intermediate Knowledge Representation - structured JSON compiled to SMT2
4. **NARS-Datalog**: Python Datalog engine with NARS truth value propagation

### LLM Client Interface
The project uses OpenAI's API format. Any client with `chat.completions.create()` works:
- OpenAI
- Azure OpenAI
- Ollama (via OpenAI-compatible endpoint)
- Any OpenAI-compatible API

## Common Tasks

### Adding a New Postprocessor
1. Create file in `z3adapter/postprocessors/`
2. Inherit from `Postprocessor` abstract class
3. Register in `z3adapter/postprocessors/registry.py`

### Adding a New Backend
1. Create file in `z3adapter/backends/`
2. Inherit from `Backend` abstract class
3. Implement `execute()` and `get_file_extension()` methods

### Modifying Prompts
- JSON backend prompts: `z3adapter/reasoning/prompt_template.py`
- SMT2 backend prompts: `z3adapter/reasoning/smt2_prompt_template.py`
- IKR backend prompts: `z3adapter/reasoning/ikr_prompt_template.py`

## Testing Guidelines

### Running Tests
```bash
# All tests
python run_tests.py

# Specific test file
python -m pytest tests/unit/test_sort_manager.py -v

# With coverage
python -m pytest tests/ --cov=z3adapter
```

### Test Categories
- **Unit tests** (`tests/unit/`): Test individual components, no LLM required
- **Integration tests** (`tests/integration/`): Test component interactions, no LLM required
- **Benchmark tests** (`benchmark/`): Require LLM API access

### Test Fixtures
Located in `tests/fixtures/`. JSON files for testing the interpreter.

## Dependencies

### Required for Tests
```bash
pip install z3-solver
```

### Required for Full Usage
```bash
pip install z3-solver openai scikit-learn numpy python-dotenv
```

### Development Tools
```bash
pip install black ruff mypy pytest pre-commit
```

## Code Style

- **Formatter**: Black (line length: 100)
- **Linter**: Ruff
- **Type hints**: Required for new code
- **Python version**: 3.12+

## Environment Variables

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_KEY=...
AZURE_DEPLOYMENT_NAME=gpt-4o
AZURE_API_VERSION=2024-12-01-preview

# Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3
```

## Architecture Notes

### Data Flow (ProofOfThought - Direct Generation)
1. User provides natural language question
2. `Z3ProgramGenerator` creates prompt and calls LLM
3. LLM returns Z3 program (SMT2 or JSON format)
4. Backend executes program with Z3 solver
5. Result is parsed and returned to user

### Data Flow (VerifiedQA - Two-Stage Verification)
1. User provides natural language question
2. LLM answers naturally with chain-of-thought reasoning
3. Second LLM call generates SMT2 encoding both question facts AND claimed answer
4. Z3 checks for contradictions between facts and answer
5. If contradiction found, answer is flipped; otherwise LLM answer is trusted

```python
# VerifiedQA usage
from z3adapter.reasoning import VerifiedQA

vqa = VerifiedQA(llm_client=client, model="qwen2.5-coder:32b")
result = vqa.query("If all cats are mammals, is a cat an animal?")

print(result.llm_answer)          # Natural language reasoning
print(result.llm_verdict)         # Extracted True/False
print(result.contradiction_found) # Did Z3 find contradiction?
print(result.final_answer)        # Verified answer
```

### Data Flow (IKR Backend - Structured Generation)
1. User provides natural language question
2. LLM generates structured IKR JSON (types, entities, relations, facts, rules, query)
3. IKR is validated with Pydantic schema
4. `IKRCompiler` deterministically compiles IKR to SMT2 (no syntax errors)
5. Z3 CLI executes the SMT2 program
6. Result is parsed and returned to user

```python
# IKR usage
from z3adapter.reasoning import ProofOfThought

pot = ProofOfThought(llm_client=client, model="gpt-4o", backend="ikr")
result = pot.query("Would a vegetarian eat a plant burger?")

print(result.answer)  # True/False/None
```

IKR benefits:
- Eliminates SMT2 syntax errors (deterministic compilation)
- Explicit background knowledge with justifications
- Debuggable intermediate representation
- Supports symmetric/transitive relation axioms

IKR extended features (for commonsense reasoning):
- **Uncertainty**: NARS-style truth values with frequency/confidence
- **Epistemic contexts**: "A believes B believes C" with possible worlds
- **Knowledge base modules**: Reusable commonsense rules (`food`, `social`)
- **Fuzzy-NARS verification**: Similarity-based triple verification with evidence pooling

```python
# IKR with uncertainty
from z3adapter.ikr.schema import Fact, TruthValue
fact = Fact(
    predicate='can_fly', arguments=['tweety'],
    truth_value=TruthValue(frequency=0.8, confidence=0.9)
)

# IKR with epistemic context
from z3adapter.ikr.schema import EpistemicContext, EpistemicOperator
fact = Fact(
    predicate='raining', arguments=[],
    epistemic_context=EpistemicContext(agent='alice', modality=EpistemicOperator.BELIEVES)
)

# Knowledge base modules
from z3adapter.ikr.knowledge_base import KnowledgeBase
merged = KnowledgeBase.merge_into_ikr(ikr_data, ['food', 'social'])

# Fuzzy-NARS verification (verify answers against KB with fuzzy matching)
from z3adapter.ikr import (
    VerificationTriple, TruthValue,
    verify_triple, combined_lexical_similarity, VerificationVerdict
)

kb = [
    VerificationTriple("phobia", "is_a", "anxiety_disorder", TruthValue(1.0, 0.9)),
    VerificationTriple("stress", "prevents", "relaxation", TruthValue(0.9, 0.85)),
]

# Fuzzy matching: "phobias" matches "phobia"
result = verify_triple(
    VerificationTriple("phobias", "is_a", "disorder"),
    kb, combined_lexical_similarity
)
print(result.verdict)  # VerificationVerdict.SUPPORTED

# Polarity detection: "causes" contradicts "prevents"
# (60+ opposite pairs: temporal, spatial, quantity, quality, etc.)
result2 = verify_triple(
    VerificationTriple("stress", "causes", "relaxation"),
    kb, combined_lexical_similarity
)
print(result2.verdict)  # VerificationVerdict.CONTRADICTED
```

### Data Flow (NARS-Datalog Backend - Native Inference)
1. User provides natural language question
2. LLM generates IKR JSON (same format as IKR backend)
3. IKR facts and rules loaded into Python Datalog engine
4. Semi-naive evaluation runs to fixpoint with NARS truth propagation
5. Query result includes truth value (frequency, confidence)

```python
# NARS-Datalog usage (direct)
from z3adapter.ikr.nars_datalog import NARSDatalogEngine, from_ikr, KBLoader
from z3adapter.ikr.schema import IKR

# Load IKR and run inference
ikr = IKR.model_validate(ikr_data)
engine = from_ikr(ikr)
result = engine.query(ikr.query)

print(f"Found: {result.found}")
print(f"Truth: f={result.truth_value.frequency:.3f}, c={result.truth_value.confidence:.3f}")

# Load knowledge base modules
engine = NARSDatalogEngine()
KBLoader.load_modules(engine, ["commonsense"])
engine.load_ikr(ikr)
result = engine.query(ikr.query)

# Pluggable truth strategies (to address confidence degradation)
engine = NARSDatalogEngine(truth_formula="opennars")  # Options: current, opennars, floor, evidence
# or
engine = from_ikr(ikr, truth_formula="floor")

# Epistemic queries (query from agent's perspective)
result = engine.query(ikr.query, agent="alice")  # Alice's beliefs + objective facts
result = engine.query(ikr.query, agent=None)     # Objective facts only
```

NARS-Datalog benefits:
- **Native truth propagation**: NARS truth values flow through inference (not converted to weights)
- **No external dependencies**: Pure Python, no Z3 or Souffle needed
- **Evidence combination**: Multiple derivations combine via NARS revision
- **Defeasible reasoning**: Rules can have uncertain truth values (e.g., "birds typically fly")
- **Transparent inference**: Explainable derivation chains
- **Pluggable truth strategies**: Choose from `current`, `opennars`, `floor`, or `evidence` to control confidence degradation
- **Epistemic logic (MVP)**: Facts can have `epistemic_context` (agent beliefs), queryable via `agent` parameter

### Triple Extraction Pipeline

A text-to-triples extraction pipeline for knowledge extraction from books and documents. Design follows Wikidata's philosophy: **predicates are fixed schema, entities emerge from content**.

**7 Generic Predicates:**
```python
from z3adapter.ikr.triples import Predicate

# Available predicates
Predicate.IS_A        # Taxonomy: X is a type of Y
Predicate.PART_OF     # Structure: X is part of Y
Predicate.HAS         # Attributes: X has property Y
Predicate.CAUSES      # Causation: X causes Y
Predicate.PREVENTS    # Negative causation: X prevents Y
Predicate.BELIEVES    # Epistemic: X believes Y
Predicate.RELATED_TO  # Catch-all: X relates to Y
```

**Usage Example:**
```python
from openai import OpenAI
from z3adapter.ikr.triples import Triple, TripleStore, Predicate, TripleExtractor

# Extract triples from text
client = OpenAI()
extractor = TripleExtractor(client, model="gpt-4o")
result = extractor.extract(
    "Chronic stress causes elevated cortisol levels, which impairs memory.",
    source="Psychology 101"
)

# Store and query triples
store = TripleStore()
for triple in result.triples:
    store.add(triple)

# Query by predicate
causal = store.query(predicate=Predicate.CAUSES)
for t in causal:
    print(f"{t.subject} causes {t.object}")

# Handle nested beliefs (reification)
belief_triple = Triple(
    id="t1",
    subject="stress",
    predicate=Predicate.CAUSES,
    object="anxiety"
)
meta_triple = Triple(
    id="t2",
    subject="dr_smith",
    predicate=Predicate.BELIEVES,
    object="t:t1"  # Reference to t1
)
store.add(belief_triple)
store.add(meta_triple)

# Resolve triple references
resolved = store.resolve("t:t1")  # Returns the Triple object

# Entity resolution (lexical fuzzy matching)
from z3adapter.ikr.triples import EntityResolver

resolver = EntityResolver(threshold=0.8)
resolver.add_entity("working_memory", ["WM", "short-term memory"])
match = resolver.resolve("short term memory")
print(match.canonical)  # "working_memory"

# Entity resolution (embedding-based semantic matching)
from z3adapter.ikr.triples.embeddings import OpenAIEmbedding, MockEmbedding

# For testing (no API calls)
resolver = EntityResolver.with_embeddings(use_mock=True, threshold=0.5)

# For production (requires OPENAI_API_KEY)
resolver = EntityResolver.with_embeddings(threshold=0.7)

resolver.add_entity("anxiety_disorder")
match = resolver.resolve("worry and fear")  # Semantic match!

# Hybrid similarity (lexical + embedding)
resolver = EntityResolver.with_hybrid_similarity(
    lexical_weight=0.3,  # 30% lexical, 70% embedding
    threshold=0.6
)

# Verification against TripleStore
from z3adapter.ikr.triples import verify_triple_against_store
from z3adapter.ikr.fuzzy_nars import VerificationVerdict

store = TripleStore()
store.add(Triple(id="t1", subject="stress", predicate=Predicate.CAUSES, object="anxiety"))

query = Triple(id="q1", subject="stress", predicate=Predicate.CAUSES, object="worry")
result = verify_triple_against_store(query, store)
print(result.verdict)  # VerificationVerdict.SUPPORTED (fuzzy match)

# Contradiction detection via predicate polarity
store.add(Triple(id="t2", subject="exercise", predicate=Predicate.PREVENTS, object="anxiety"))
query2 = Triple(id="q2", subject="exercise", predicate=Predicate.CAUSES, object="anxiety")
result2 = verify_triple_against_store(query2, store)
print(result2.verdict)  # VerificationVerdict.CONTRADICTED
```

**Key Design Decisions:**
- **No domain-specific KB**: Vocabulary emerges from text + LLM common sense
- **Triple references**: `t:` prefix enables multi-level beliefs (reification)
- **Entity resolution**: Fuzzy matching with surface form learning
- **NARS truth values**: Uncertainty propagation throughout
- **Predicate polarity**: causes ↔ prevents for contradiction detection

**End-to-End Pipeline:**
```python
from openai import OpenAI
from z3adapter.ikr.triples import ExtractionPipeline

# Create pipeline with SQLite persistence
client = OpenAI()
pipeline = ExtractionPipeline.create(
    client,
    model="gpt-4o",
    db_path="knowledge.db"  # Optional: omit for in-memory only
)

# Ingest knowledge from text
pipeline.ingest("Stress causes anxiety.", source="Psychology 101")
pipeline.ingest("Exercise prevents stress.", source="Health Guide")

# Query the knowledge base
result = pipeline.query("Does stress cause anxiousness?", match_threshold=0.4)
print(result.verdict)  # SUPPORTED (fuzzy match)

result2 = pipeline.query("Does exercise cause stress?")
print(result2.verdict)  # CONTRADICTED (opposite predicate)

# Direct verification without LLM
triple = Triple(id="q1", subject="stress", predicate=Predicate.CAUSES, object="anxiety")
result = pipeline.verify(triple)

# Save/load from SQLite
pipeline.save()   # Persist current state
pipeline.load()   # Load from database
```

**File Structure:**
```
z3adapter/ikr/triples/
├── __init__.py         # Package exports
├── schema.py           # Triple, TripleStore, Predicate
├── extractor.py        # LLM-based triple extraction
├── entity_resolver.py  # Fuzzy entity matching
├── embeddings.py       # Embedding backends (OpenAI, Mock)
├── verification.py     # Fuzzy-NARS bridge
├── storage.py          # SQLite persistence
└── pipeline.py         # End-to-end extraction pipeline
```

See `.claude/sessions/2026-01-15-triple-extraction-plan.md` for design decisions.

### Link-Based Architecture

The next evolution of the triple extraction pipeline uses **entity linking** instead of entity merging for large-scale commonsense reasoning.

**Key Concepts:**
- **Link, don't merge**: "tension" and "stress" remain separate entities connected by similarity links
- **Pre-computed similarity**: Entity embeddings and similarity links computed offline
- **Query expansion**: Queries expanded via links, inference stays symbolic (fast)
- **Wikidata-scale**: Architecture designed for 100M+ entities

**Architecture:**
```
Text → LLM Extract → Entity Linker → Stored Triples
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
      Vector Index                Entity Links
      (FAISS ANN)               (pre-computed)
            │                           │
            └─────────────┬─────────────┘
                          ▼
Query → Entity Resolution → Query Expansion → Symbolic Match → Result
```

**Implemented Components:**
- `Entity`: Knowledge graph nodes with canonical names, types, external IDs
- `EntityLink`: Pre-computed similarity links between entities
- `EntityStore`: SQLite-backed storage for entities, links, surface forms, embeddings
- `VectorIndex`: FAISS-based ANN search for fast similarity lookup
- `EntityLinker`: Multi-level resolution (exact → surface form → vector)
- `PrecomputationPipeline`: Batch embedding and link computation

**Planned Components:**
- `QueryExpander`: Expand queries via similarity links

**Usage Example (Phase 1 - Entity Storage):**
```python
from z3adapter.ikr.entities import Entity, EntityLink, EntityStore, LinkType

# Create store (in-memory or persistent)
store = EntityStore("knowledge.db")  # or EntityStore() for in-memory

# Add entities
anxiety = Entity(name="anxiety", entity_type="emotion")
stress = Entity(name="stress", entity_type="state")
store.add(anxiety)
store.add(stress)

# Add similarity link
link = EntityLink(
    source_id=anxiety.id,
    target_id=stress.id,
    link_type=LinkType.SIMILAR_TO,
    score=0.75,
    method="embedding"
)
store.add_link(link)

# Query similar entities
similar = store.get_similar(anxiety.id, min_score=0.5)
for entity, score in similar:
    print(f"{entity.name}: {score}")

# Add surface form for O(1) lookup
store.add_surface_form("nervousness", anxiety.id, 0.9)
result = store.lookup_surface_form("nervousness")  # Returns (entity_id, score)

# Store embeddings
store.save_embedding(anxiety.id, [0.1, 0.2, ...], model="text-embedding-3-small")
```

**Usage Example (Phase 2 - Vector Index):**
```python
from z3adapter.ikr.entities import VectorIndex

# Create index (dimension matches your embedding model)
index = VectorIndex(dimension=1536)  # OpenAI text-embedding-3-small

# Build from embedding dict
embeddings = {
    "anxiety": [...],  # 1536-dim vector
    "stress": [...],
    "fear": [...],
}
index.build(embeddings)  # Uses IVF for large datasets, flat for small

# Search for similar entities
query_embedding = [...]  # Query vector
results = index.search(query_embedding, k=10)
for entity_id, score in results:
    print(f"{entity_id}: {score:.3f}")  # Cosine similarity in [0, 1]

# Incremental add
index.add("new_entity", new_embedding)

# Batch search (more efficient)
results = index.search_batch([query1, query2, query3], k=5)

# Persistence
index.save("my_index")  # Creates .index and .idmap files
index.load("my_index")  # Restore later
```

**Usage Example (Phase 3 - EntityLinker):**
```python
from z3adapter.ikr.entities import EntityStore, EntityLinker, VectorIndex, LinkResult

store = EntityStore("knowledge.db")
index = VectorIndex(dimension=1536)

def embed_fn(text: str) -> list[float]:
    return openai.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding

linker = EntityLinker(
    entity_store=store,
    vector_index=index,
    embed_fn=embed_fn,
    link_threshold=0.5,      # Create links above this score
    identity_threshold=0.9,  # Use existing entity above this score
)

# Link a mention to an entity
result = linker.link("working memory")

if result.is_new:
    # New entity created - persist it
    store.add(result.entity)
    for link in result.links:
        store.add_link(link)
else:
    print(f"Resolved to: {result.entity.name}")
    print(f"Method: {result.resolution_method}")  # "exact", "surface_form", "vector_high"

# Or use convenience method (auto-persists)
result = linker.link_and_store("cognitive load", entity_type="concept")
```

**Usage Example (Phase 4 - PrecomputationPipeline):**
```python
from z3adapter.ikr.entities import EntityStore, VectorIndex, PrecomputationPipeline

store = EntityStore("knowledge.db")
index = VectorIndex(dimension=1536)

def batch_embed(texts: list[str]) -> list[list[float]]:
    response = openai.embeddings.create(input=texts, model="text-embedding-3-small")
    return [d.embedding for d in response.data]

pipeline = PrecomputationPipeline(
    entity_store=store,
    vector_index=index,
    batch_embed_fn=batch_embed,
    model_name="text-embedding-3-small",
)

# Run full pipeline
stats = pipeline.run_full_pipeline(
    embedding_batch_size=100,
    link_k=50,
    link_min_score=0.5,
    index_path="knowledge_index",
)

print(f"Embeddings computed: {stats.embeddings_computed}")
print(f"Links created: {stats.links_created}")

# Or run stages individually
pipeline.compute_embeddings(batch_size=100)
pipeline.rebuild_vector_index(index_path="my_index")
pipeline.compute_links(k=50, min_score=0.5)

# With progress callback
def on_progress(operation: str, current: int, total: int):
    print(f"{operation}: {current}/{total}")

pipeline.progress_callback = on_progress
pipeline.run_full_pipeline()
```

See `.claude/sessions/2026-01-16-link-based-architecture-design.md` for full design.

### Error Handling
- Failed generations trigger retry with error feedback
- Maximum 3 attempts by default
- Postprocessors can enhance results after initial success

## Gotchas

1. **Import name differs from package name**: Use `z3adapter`, not `proofofthought`
2. **Tests don't need LLM**: Unit/integration tests use Z3 directly
3. **Z3 enum sorts are global**: Use unique names to avoid conflicts in tests
4. **SMT2 backend needs Z3 CLI**: Ensure `z3` is in PATH or specify `z3_path`
5. **IKR tests require z3-solver**: The `z3adapter/__init__.py` imports z3, so IKR tests skip if z3 is not installed

---

## Session Notes

Session notes are stored in `.claude/sessions/` with one file per session.

**Naming convention:** `YYYY-MM-DD-brief-description.md`

**At the start of each session:** Read recent session notes to understand context.

**At the end of each session:** Create a new session file documenting:
- Summary of work done
- Files created/modified
- Pending/future work
