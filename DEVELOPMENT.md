# Development Guide

This guide covers setting up a development environment and common development tasks for ProofOfThought.

## Quick Start

```bash
# Clone and setup
git clone https://github.com/debarghaG/proofofthought.git
cd proofofthought

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests to verify setup
python run_tests.py
```

## Environment Setup

### Minimal Setup (Tests Only)

To run tests without LLM access:

```bash
pip install z3-solver
python run_tests.py
```

### Full Development Setup

```bash
pip install -r requirements.txt
pip install -e .  # Editable install
pre-commit install  # Git hooks
```

### LLM Provider Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` with your credentials (see [LLM Providers](#llm-providers) below).

## LLM Providers

### OpenAI

```bash
# .env
OPENAI_API_KEY=sk-your-key-here
```

```python
from openai import OpenAI
client = OpenAI()  # Uses OPENAI_API_KEY from environment
```

### Azure OpenAI

```bash
# .env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-key-here
AZURE_DEPLOYMENT_NAME=gpt-4o
AZURE_API_VERSION=2024-12-01-preview
```

```python
from openai import AzureOpenAI
import os

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_API_VERSION")
)
```

### Ollama (Local)

Run Ollama locally, then:

```bash
# .env
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3
```

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key="ollama"  # Required but not validated
)

# Use with ProofOfThought
from z3adapter.reasoning import ProofOfThought
pot = ProofOfThought(
    llm_client=client,
    model=os.getenv("OLLAMA_MODEL", "llama3")
)
```

### Other OpenAI-Compatible APIs

Any API following OpenAI's format works:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-api-endpoint/v1",
    api_key="your-api-key"
)
```

## Running Tests

### All Tests

```bash
python run_tests.py
```

### Specific Tests

```bash
# Single file
python -m pytest tests/unit/test_sort_manager.py -v

# Single test
python -m pytest tests/unit/test_sort_manager.py::TestSortManager::test_create_enum_sort -v

# By pattern
python -m pytest tests/ -k "sort" -v
```

### With Coverage

```bash
python -m pytest tests/ --cov=z3adapter --cov-report=html
open htmlcov/index.html  # View report
```

### Test Categories

| Directory | Description | LLM Required |
|-----------|-------------|--------------|
| `tests/unit/` | Unit tests for components | No |
| `tests/integration/` | Integration tests | No |
| `benchmark/` | Benchmark evaluation | Yes |

## Code Quality

### Formatting

```bash
# Format all files
black .

# Check without modifying
black --check .
```

### Linting

```bash
# Run linter
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Type Checking

```bash
mypy z3adapter/
```

### Pre-commit Hooks

Runs automatically on commit after `pre-commit install`:

```bash
# Run manually on all files
pre-commit run --all-files
```

## Running Benchmarks

### Prerequisites

- LLM API access configured
- Benchmark datasets in `data/`

### Run All Benchmarks

```bash
python experiments_pipeline.py
```

### Run Specific Benchmark

```bash
# Using the evaluation pipeline
python -c "
from z3adapter.reasoning import ProofOfThought, EvaluationPipeline
from openai import OpenAI

client = OpenAI()
pot = ProofOfThought(llm_client=client, model='gpt-4o')
evaluator = EvaluationPipeline(proof_of_thought=pot)
result = evaluator.evaluate(
    dataset='data/prontoqa_test.json',
    max_samples=10
)
print(f'Accuracy: {result.metrics.accuracy:.2%}')
"
```

## Building Documentation

```bash
# Install doc dependencies
pip install mkdocs mkdocs-material

# Serve locally
mkdocs serve

# Build static site
mkdocs build
```

## Package Building

### Build Distribution

```bash
pip install build
python -m build
```

### Test PyPI Upload

```bash
pip install twine
twine upload --repository testpypi dist/*
```

## Project Structure Deep Dive

### Core Components

```
z3adapter/
├── reasoning/
│   ├── proof_of_thought.py   # Main API class
│   ├── program_generator.py  # LLM → Z3 program
│   ├── evaluation.py         # Batch evaluation
│   ├── prompt_template.py    # JSON backend prompts
│   └── smt2_prompt_template.py  # SMT2 backend prompts
├── backends/
│   ├── abstract.py           # Backend interface
│   ├── json_backend.py       # JSON DSL backend
│   └── smt2_backend.py       # SMT2 backend
├── dsl/
│   ├── sorts.py              # Z3 sort management
│   └── expressions.py        # Expression parsing
└── postprocessors/
    ├── abstract.py           # Postprocessor interface
    ├── registry.py           # Postprocessor registration
    ├── self_refine.py        # Self-refinement
    ├── self_consistency.py   # Majority voting
    ├── decomposed.py         # Question decomposition
    └── least_to_most.py      # Progressive solving
```

### Data Flow

```
Question → ProgramGenerator → LLM → Z3 Program
                                        ↓
                                    Backend
                                        ↓
                                   Z3 Solver
                                        ↓
                                QueryResult
```

## Debugging Tips

### Enable Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Generated Programs

```python
result = pot.query("Your question", save_program=True)
# Check cache_dir for saved program files
```

### Debug Z3 Directly

```python
from z3 import *

# Create solver and test constraints
s = Solver()
x = Int('x')
s.add(x > 0, x < 10)
print(s.check())  # sat
print(s.model())  # [x = 1]
```

## Common Issues

### Import Errors

```python
# Wrong
from proofofthought import ProofOfThought

# Correct
from z3adapter.reasoning import ProofOfThought
```

### Z3 Not Found (SMT2 Backend)

Ensure Z3 CLI is in PATH or specify path:

```python
pot = ProofOfThought(llm_client=client, z3_path="/path/to/z3")
```

### Enum Sort Conflicts in Tests

Use unique names for enum sorts:

```python
import uuid
unique_name = f"Color_{uuid.uuid4().hex[:8]}"
```

### LLM Rate Limits

Add delays between requests or use batch evaluation with appropriate pacing.
