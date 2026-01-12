# Troubleshooting

This guide covers common issues and their solutions when using ProofOfThought.

## Installation Issues

### Z3 Not Found

**Error:**
```
FileNotFoundError: Z3 not found at z3
```

**Solution:**

Install Z3 solver:

```bash
# pip
pip install z3-solver

# Ubuntu/Debian
sudo apt install z3

# macOS
brew install z3

# Or specify path explicitly
pot = ProofOfThought(llm_client=client, z3_path="/usr/local/bin/z3")
```

### Import Errors

**Error:**
```python
ModuleNotFoundError: No module named 'proofofthought'
```

**Solution:**

The import name differs from the package name:

```python
# Wrong
from proofofthought import ProofOfThought

# Correct
from z3adapter.reasoning import ProofOfThought
```

### Missing Dependencies

**Error:**
```
ModuleNotFoundError: No module named 'openai'
```

**Solution:**

Install all dependencies:

```bash
pip install proofofthought[all]

# Or manually
pip install z3-solver openai scikit-learn numpy python-dotenv
```

## LLM Connection Issues

### OpenAI API Key

**Error:**
```
openai.AuthenticationError: Incorrect API key provided
```

**Solution:**

Set the API key:

```bash
export OPENAI_API_KEY="sk-..."
```

Or in Python:

```python
from openai import OpenAI
client = OpenAI(api_key="sk-...")
```

### Azure OpenAI Configuration

**Error:**
```
openai.NotFoundError: Resource not found
```

**Solution:**

Verify all Azure settings:

```python
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://YOUR-RESOURCE.openai.azure.com",
    api_key="YOUR-KEY",
    api_version="2024-12-01-preview",
)

# Use deployment name, not model name
pot = ProofOfThought(
    llm_client=client,
    model="YOUR-DEPLOYMENT-NAME",  # e.g., "gpt-4o-deployment"
)
```

### Ollama Connection

**Error:**
```
openai.APIConnectionError: Connection error
```

**Solution:**

1. Ensure Ollama is running:
   ```bash
   ollama serve
   ```

2. Check the model is available:
   ```bash
   ollama list
   ollama pull qwen2.5-coder:32b
   ```

3. Configure the client:
   ```python
   client = OpenAI(
       base_url="http://localhost:11434/v1",
       api_key="ollama",  # Required but unused
   )
   ```

## Execution Errors

### Timeout Exceeded

**Error:**
```
Z3 timeout: verification took longer than 10000ms
```

**Solution:**

Increase timeout for complex problems:

```python
pot = ProofOfThought(
    llm_client=client,
    model="gpt-4o",
    verify_timeout=60000,    # 60 seconds
    optimize_timeout=120000, # 2 minutes
)
```

### Invalid JSON Program

**Error:**
```
json.JSONDecodeError: Expecting property name
```

**Solution:**

This usually means the LLM generated invalid JSON. The system retries automatically, but you can:

1. Increase attempts:
   ```python
   pot = ProofOfThought(llm_client=client, max_attempts=5)
   ```

2. Use a more capable model:
   ```python
   pot = ProofOfThought(llm_client=client, model="gpt-4o")
   ```

3. Try the SMT2 backend (more forgiving):
   ```python
   pot = ProofOfThought(llm_client=client, backend="smt2")
   ```

### SMT2 Syntax Error

**Error:**
```
(error "line 5 column 10: unknown sort 'Entity'")
```

**Solution:**

The LLM generated invalid SMT-LIB code. Enable debugging:

```python
result = pot.query(
    question="...",
    save_program=True,
    program_path="debug_program.smt2",
)

# Examine the generated program
with open("debug_program.smt2") as f:
    print(f.read())
```

### Ambiguous Result

**Error:**
```
QueryResult(answer=None, ...)
```

**Cause:**

Z3 output contains both SAT and UNSAT, or neither.

**Solution:**

1. Simplify the question
2. Check the raw output:
   ```python
   result = pot.query("...")
   print(f"SAT count: {result.sat_count}")
   print(f"UNSAT count: {result.unsat_count}")
   print(f"Output: {result.output}")
   ```

## Model-Specific Issues

### GPT-5 Temperature Limitation

**Problem:** Setting `temperature` has no effect with GPT-5.

**Cause:** GPT-5 only supports `temperature=1.0`. The parameter is ignored.

**Solution:** This is expected behavior. For GPT-5, rely on other parameters like `max_tokens` or use postprocessors (e.g., self-consistency) for variance control.

```python
# Temperature is ignored for GPT-5
result = pot.query(question, temperature=0.1)  # Actually uses 1.0
```

### Model Not Supporting `max_completion_tokens`

**Problem:** Error about unsupported parameter.

**Cause:** Some older models use `max_tokens` instead of `max_completion_tokens`.

**Solution:** Use a model that supports the newer parameter, or modify the generator if needed for legacy models.

## Common Mistakes

### Using Wrong Backend for Question Type

**Problem:** Poor accuracy on certain question types.

**Solution:**

| Question Type | Recommended Backend |
|---------------|-------------------|
| Logical syllogisms | SMT2 |
| Optimization problems | JSON |
| Simple yes/no | SMT2 or VerifiedQA |
| Complex constraints | JSON |

### Not Handling Failures

**Problem:** Code crashes on failed queries.

**Solution:**

Always check the result:

```python
result = pot.query("...")

if not result.success:
    print(f"Query failed: {result.error}")
    # Handle failure
else:
    print(f"Answer: {result.answer}")
```

### Enum Sort Name Conflicts

**Problem:** Z3 error about duplicate sort names.

**Cause:** Z3 enum sorts are global. Reusing names causes conflicts.

**Solution:**

Use unique sort names, especially in tests:

```python
# Wrong: may conflict
{"name": "Animal", "type": "enum", "values": ["cat", "dog"]}

# Better: include context in name
{"name": "Test1_Animal", "type": "enum", "values": ["cat", "dog"]}
```

## Debugging Tips

### Enable Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all ProofOfThought operations are logged
pot = ProofOfThought(llm_client=client, model="gpt-4o")
result = pot.query("...")
```

### Save Generated Programs

```python
result = pot.query(
    question="...",
    save_program=True,
    program_path="./debug/",
)
```

### Inspect LLM Response

For JSON backend:

```python
result = pot.query("...")
if result.json_program:
    import json
    print(json.dumps(result.json_program, indent=2))
```

### Understanding Program Extraction

The system extracts programs from LLM responses using these patterns:

**JSON Backend:**
1. First tries: ` ```json { ... } ``` `
2. Falls back to: any `{ ... }` in the response

**SMT2 Backend:**
1. First tries: ` ```smt2 ... ``` `
2. Falls back to: lines starting with `;` or `(`

If extraction fails, check the raw response:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# DEBUG logs show: "Raw LLM response: ..."
result = pot.query("...")
```

### Test Z3 Directly

Isolate Z3 issues by testing directly:

```bash
# Save the generated program, then:
z3 debug_program.smt2
```

Or in Python:

```python
from z3 import *

# Minimal test
s = Solver()
x = Int('x')
s.add(x > 0)
print(s.check())  # Should print 'sat'
```

## Performance Issues

### Slow LLM Responses

**Solution:**

1. Use a faster model for simple queries
2. Reduce `max_tokens` if answers are short
3. Use batch evaluation with workers:
   ```python
   pipeline = EvaluationPipeline(pot, num_workers=4)
   ```

### High Memory Usage

**Solution:**

1. Process queries one at a time
2. Clear Z3 solver state between queries (automatic)
3. Reduce batch sizes in evaluation

## Getting Help

If you're still stuck:

1. Search [existing issues](https://github.com/anthropics/claude-code/issues)
2. Check the [API Reference](api-reference.md)
3. Open a new issue with:
   - Python version
   - ProofOfThought version
   - Full error traceback
   - Minimal reproduction code
