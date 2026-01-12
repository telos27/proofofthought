# VerifiedQA

VerifiedQA is a two-stage reasoning system that combines LLM chain-of-thought reasoning with formal verification using Z3.

## Overview

Unlike the standard `ProofOfThought` approach which generates Z3 programs directly, VerifiedQA:

1. **Stage 1**: Asks the LLM to answer the question naturally with chain-of-thought reasoning
2. **Stage 2**: Generates an SMT2 program encoding both the question facts AND the LLM's claimed answer
3. **Verification**: Z3 checks for contradictions between the facts and the claimed answer

If Z3 finds a contradiction (UNSAT), the LLM's answer is flipped. Otherwise, the LLM's answer is trusted.

## When to Use VerifiedQA

| Use Case | Recommended Approach |
|----------|---------------------|
| Complex logical reasoning | `ProofOfThought` |
| Questions with implicit facts | `VerifiedQA` |
| Simple yes/no questions | `VerifiedQA` |
| Multi-step proofs | `ProofOfThought` |
| Fact-checking LLM responses | `VerifiedQA` |

## Quick Start

```python
from openai import OpenAI
from z3adapter.reasoning import VerifiedQA

# Using Ollama (local)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

vqa = VerifiedQA(llm_client=client, model="qwen2.5-coder:32b")

result = vqa.query("If all birds fly and Tweety is a bird, can Tweety fly?")

print(f"LLM reasoning: {result.llm_answer}")
print(f"LLM verdict: {result.llm_verdict}")
print(f"Contradiction found: {result.contradiction_found}")
print(f"Final answer: {result.final_answer}")
```

## API Reference

### Constructor

```python
def __init__(
    self,
    llm_client: Any,
    model: str = "gpt-4o",
    cache_dir: str | None = None,
    z3_path: str = "z3",
) -> None
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm_client` | `Any` | required | OpenAI-compatible client |
| `model` | `str` | `"gpt-4o"` | Model name |
| `cache_dir` | `str \| None` | `tempfile.gettempdir()` | Directory for SMT2 program cache |
| `z3_path` | `str` | `"z3"` | Path to Z3 executable |

### query()

```python
def query(
    self,
    question: str,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> VerifiedQAResult
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | `str` | required | Natural language question |
| `temperature` | `float` | `0.1` | LLM temperature |
| `max_tokens` | `int` | `4096` | Max tokens for LLM response |

**Returns:** `VerifiedQAResult`

### VerifiedQAResult

```python
@dataclass
class VerifiedQAResult:
    question: str              # Input question
    llm_answer: str            # Full LLM response with reasoning
    llm_verdict: bool | None   # Extracted yes/no from LLM
    verified: bool             # Whether verification completed
    contradiction_found: bool  # True if Z3 returned UNSAT
    final_answer: bool | None  # Answer after verification
    smt2_program: str | None   # Generated SMT2 for inspection
    z3_output: str             # Raw Z3 output
    error: str | None          # Error message if failed
```

## How It Works

### Stage 1: LLM Reasoning

The LLM is prompted to:

1. Think through the question step by step
2. Provide chain-of-thought reasoning
3. Conclude with `ANSWER: YES` or `ANSWER: NO`

Example prompt:
```
Answer the following question. Think step by step, then provide your final answer.

Question: If all cats are mammals and Whiskers is a cat, is Whiskers a mammal?

Please reason through this step by step, then conclude with your answer.
At the end, clearly state "ANSWER: YES" or "ANSWER: NO".
```

### Stage 2: SMT2 Generation

A second LLM call generates an SMT2 program that:

1. Encodes all facts/premises from the question
2. Encodes the LLM's claimed answer as an assertion
3. Checks for satisfiability

Example generated SMT2:
```smt2
; Sorts
(declare-sort Entity 0)

; Predicates
(declare-fun is-cat (Entity) Bool)
(declare-fun is-mammal (Entity) Bool)

; Constants
(declare-const whiskers Entity)

; Facts from question
(assert (forall ((x Entity)) (=> (is-cat x) (is-mammal x))))  ; all cats are mammals
(assert (is-cat whiskers))                                    ; Whiskers is a cat

; LLM's claimed answer: TRUE (Whiskers is a mammal)
(assert (is-mammal whiskers))

; Check consistency
(check-sat)
```

### Stage 3: Verification

Z3 checks the combined assertions:

- **SAT**: Facts and claimed answer are consistent. Trust the LLM.
- **UNSAT**: Facts and claimed answer contradict. Flip the answer.

## Examples

### Catching LLM Errors

```python
# Question where LLM might make a mistake
result = vqa.query(
    "If no reptiles are mammals and all snakes are reptiles, "
    "is a snake a mammal?"
)

# If LLM incorrectly says YES:
# - SMT2 encodes: no reptiles are mammals, snakes are reptiles, snake is mammal
# - Z3 returns UNSAT (contradiction)
# - Final answer is flipped to NO (correct)
```

### Inspecting the SMT2 Program

```python
result = vqa.query("If all birds can fly, can a penguin fly?")

# View the generated verification program
print(result.smt2_program)

# Check Z3's output
print(result.z3_output)
```

### Handling Verification Failures

```python
result = vqa.query("Some complex question...")

if not result.verified:
    print(f"Verification failed: {result.error}")
    print(f"Falling back to LLM answer: {result.final_answer}")
else:
    print(f"Verified answer: {result.final_answer}")
```

## Comparison with ProofOfThought

| Aspect | ProofOfThought | VerifiedQA |
|--------|---------------|------------|
| **Approach** | Direct Z3 program generation | LLM answers, then verify |
| **LLM Calls** | 1-3 (with retries) | 2 (answer + verify) |
| **Reasoning** | Encoded in Z3 | Natural language |
| **Verification** | Z3 computes answer | Z3 checks consistency |
| **Best For** | Complex logical proofs | Fact-checking, simple Q&A |

## Limitations

- Requires questions with clear yes/no answers
- LLM must generate valid SMT2 for verification
- Cannot verify questions about quantities or specific values
- Verification quality depends on accurate SMT2 encoding

## Supported LLM Providers

Any OpenAI-compatible API works:

```python
# OpenAI
from openai import OpenAI
client = OpenAI()

# Azure OpenAI
from openai import AzureOpenAI
client = AzureOpenAI(...)

# Ollama
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```
