# Session: 2026-01-11 - IKR Implementation

## Summary
Implemented the Intermediate Knowledge Representation (IKR) layer for improved SMT generation. IKR provides a structured intermediate format between natural language and SMT2, focusing on commonsense/world knowledge reasoning.

## Design Decisions
1. **Minimal schema first**: types, entities, predicates, facts, rules, query
2. **Two-stage prompting**: Stage 1 extracts explicit facts, Stage 2 generates background knowledge
3. **Deterministic compilation**: IKR compiles to SMT2 without LLM errors

## Files Created
- `z3adapter/ikr/__init__.py` - Package exports
- `z3adapter/ikr/schema.py` - Pydantic models for IKR validation
- `z3adapter/ikr/compiler.py` - IKR to SMT2 deterministic compilation
- `z3adapter/reasoning/ikr_prompt_template.py` - Two-stage prompts for IKR generation
- `z3adapter/backends/ikr_backend.py` - Backend implementing Backend ABC
- `tests/unit/test_ikr_schema.py` - Schema validation tests
- `tests/unit/test_ikr_compiler.py` - Compiler tests
- `tests/fixtures/ikr_examples/simple_test.json` - Simple test fixture
- `tests/fixtures/ikr_examples/vegetarian_burger.json` - Complex example fixture

## Files Modified
- `z3adapter/reasoning/program_generator.py` - Added `backend="ikr"` support
- `z3adapter/reasoning/proof_of_thought.py` - Wired up IKR backend

## IKR Schema Structure
```yaml
meta:
  question: string
  question_type: yes_no | comparison | possibility

types:
  - name: string
    description: string?

entities:
  - name: string
    type: string
    aliases: [string]

relations:
  - name: string
    signature: [type1, ...]
    range: Bool | Int | Real | type
    symmetric: bool?
    transitive: bool?

facts:
  - predicate: string
    arguments: [string]
    negated: bool
    source: explicit | background
    justification: string?

rules:
  - name: string?
    quantified_vars: [{name, type}]
    antecedent: condition
    consequent: condition
    justification: string?

query:
  predicate: string
  arguments: [string]
  negated: bool
```

## Usage
```python
from z3adapter.reasoning import ProofOfThought

pot = ProofOfThought(llm_client=client, model="gpt-4o", backend="ikr")
result = pot.query("Would a vegetarian eat a plant burger?")
```

## Documentation Updated
- `CHANGELOG.md` - Added v1.1.0 with IKR feature
- `docs/backends.md` - Full IKR backend documentation
- `CLAUDE.md` - Added IKR to project structure, backends, prompts, data flow

## Pending/Future Work
1. Implement two-stage prompting in the generator (currently uses single-stage)
2. Benchmark IKR vs SMT2 on StrategyQA
3. Add reasoning_chain support to IKR schema
4. Consider adding knowledge base caching for common background knowledge
