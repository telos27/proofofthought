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
│   ├── backends/              # Execution backends (SMT2, JSON, IKR)
│   ├── ikr/                   # Intermediate Knowledge Representation
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

### Three Backends
1. **SMT2** (default): Standard SMT-LIB 2.0 format, executed via Z3 CLI
2. **JSON**: Custom DSL interpreted via Python Z3 API
3. **IKR**: Intermediate Knowledge Representation - structured JSON compiled to SMT2

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
result2 = verify_triple(
    VerificationTriple("stress", "causes", "relaxation"),
    kb, combined_lexical_similarity
)
print(result2.verdict)  # VerificationVerdict.CONTRADICTED
```

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
