# Security

ProofOfThought executes LLM-generated code, which requires careful security measures. This document describes the security model and best practices.

## Threat Model

When using LLM-generated Z3 programs, the following risks exist:

| Threat | Mitigation |
|--------|------------|
| Arbitrary code execution | Expression validation, sandboxed eval |
| Resource exhaustion | Timeout limits on Z3 operations |
| Information disclosure | Restricted built-ins, no file access |
| Denial of service | Configurable timeouts |

## Expression Validator

The `ExpressionValidator` class (`z3adapter/security/validator.py`) provides safe expression evaluation for the JSON backend.

### Blocked Constructs

The validator blocks dangerous Python constructs:

```python
# Blocked: Dunder attribute access
obj.__class__.__bases__  # ValueError

# Blocked: Import statements
import os  # ValueError

# Blocked: Function/class definitions
def malicious(): pass  # ValueError

# Blocked: Dangerous builtins
eval("code")     # ValueError
exec("code")     # ValueError
compile("code")  # ValueError
__import__("os") # ValueError
```

### Safe Evaluation

The `safe_eval` method evaluates expressions with restricted globals:

```python
from z3adapter.security.validator import ExpressionValidator

# Safe context with only Z3 operators
safe_globals = {"And": z3.And, "Or": z3.Or, "Not": z3.Not}
context = {"x": z3_var_x, "y": z3_var_y}

# Evaluate expression safely
result = ExpressionValidator.safe_eval(
    "And(x > 0, y < 10)",
    safe_globals,
    context
)
```

### Implementation Details

```python
def safe_eval(expr_str: str, safe_globals: dict, context: dict) -> Any:
    # 1. Parse expression to AST
    tree = ast.parse(expr_str, mode="eval")

    # 2. Check AST for dangerous constructs
    ExpressionValidator.check_safe_ast(tree, expr_str)

    # 3. Compile and evaluate with no builtins
    code = compile(tree, "<string>", "eval")
    return eval(code, {"__builtins__": {}}, {**safe_globals, **context})
```

Key security features:

- **AST inspection**: Catches dangerous patterns before execution
- **No builtins**: `{"__builtins__": {}}` removes access to dangerous functions
- **Restricted globals**: Only explicitly provided functions are available

## Timeout Protection

Both backends support configurable timeouts to prevent resource exhaustion:

### JSON Backend

```python
from z3adapter.reasoning import ProofOfThought

pot = ProofOfThought(
    llm_client=client,
    model="gpt-4o",
    backend="json",
    verify_timeout=10000,    # 10 seconds for verifications
    optimize_timeout=100000, # 100 seconds for optimization
)
```

### SMT2 Backend

The SMT2 backend uses subprocess timeouts:

```python
result = subprocess.run(
    [self.z3_path, smt2_path],
    capture_output=True,
    timeout=30,  # 30 second timeout
)
```

## Backend Security Comparison

| Aspect | JSON Backend | SMT2 Backend |
|--------|-------------|--------------|
| Code execution | Python eval (sandboxed) | Z3 CLI (subprocess) |
| Attack surface | Expression validator | SMT-LIB parser |
| Isolation | Process-level | Process-level |
| Timeout control | Z3 API | subprocess timeout |

### SMT2 Backend Advantages

The SMT2 backend is generally more secure because:

1. **Limited language**: SMT-LIB has no side effects
2. **Separate process**: Z3 runs in isolated subprocess
3. **No Python eval**: No risk of Python code injection

### JSON Backend Considerations

The JSON backend requires more caution:

1. **Python expressions**: Constraints are evaluated in Python
2. **Expression validation**: Must pass AST checks
3. **Limited globals**: Only Z3 operators available

## Best Practices

### 1. Prefer SMT2 Backend

For production use, prefer the SMT2 backend:

```python
pot = ProofOfThought(
    llm_client=client,
    model="gpt-4o",
    backend="smt2",  # More secure
)
```

### 2. Set Appropriate Timeouts

Prevent resource exhaustion with reasonable limits:

```python
pot = ProofOfThought(
    llm_client=client,
    model="gpt-4o",
    verify_timeout=30000,   # 30 seconds max
    optimize_timeout=60000, # 60 seconds max
)
```

### 3. Validate LLM Provider

Use trusted LLM providers to reduce risk of malicious program generation:

```python
# Trusted: Official OpenAI API
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Trusted: Azure OpenAI
client = AzureOpenAI(...)

# Review: Self-hosted models may have different behaviors
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

### 4. Monitor Generated Programs

For debugging and auditing, save generated programs:

```python
result = pot.query(
    question="...",
    save_program=True,
    program_path="/var/log/z3_programs/query_001.smt2",
)
```

### 5. Run in Sandboxed Environment

For untrusted inputs, consider additional isolation:

- Docker containers
- Virtual machines
- Restricted user accounts
- seccomp profiles (Linux)

## Known Limitations

1. **No network isolation**: Z3 subprocess could theoretically make network calls
2. **File system access**: Z3 can read files if paths are provided
3. **Memory limits**: No built-in memory limits (use cgroups/Docker)

## Reporting Security Issues

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email the maintainers directly
3. Include steps to reproduce
4. Allow time for a fix before disclosure
