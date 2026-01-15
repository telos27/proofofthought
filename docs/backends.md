# Backends

ProofOfThought supports multiple execution backends: the standard SMT-LIB 2.0 format, a custom JSON DSL, the Intermediate Knowledge Representation (IKR), and Souffle Datalog.

## SMT2Backend

The SMT2 backend leverages Z3's standard command-line interface.

**Implementation:** `z3adapter/backends/smt2_backend.py`

### Execution

```python
subprocess.run([z3_path, f"-T:{timeout_seconds}", program_path])
```

The execution process involves:

- Running Z3 as a CLI subprocess with a timeout flag
- Applying a hard timeout of `timeout_seconds + 10` to prevent hanging
- Capturing output from both stdout and stderr

### Result Parsing

```python
sat_pattern = r"(?<!un)\bsat\b"      # Negative lookbehind to exclude "unsat"
unsat_pattern = r"\bunsat\b"
```

The parser counts occurrences in Z3 output and applies the following answer logic:

- `sat_count > 0, unsat_count == 0` → `True`
- `unsat_count > 0, sat_count == 0` → `False`
- Otherwise → `None`

### Prompt Template

The prompt template guides LLM program generation.

**Source:** `z3adapter/reasoning/smt2_prompt_template.py`

The template provides instructions for generating SMT-LIB 2.0 programs with these key requirements:
- All commands as S-expressions: `(command args...)`
- Declare sorts before use
- Single `(check-sat)` per program
- Semantic: `sat` = constraint satisfiable, `unsat` = contradicts knowledge base

### File Extension

`.smt2`

## JSON Backend

The JSON backend uses Z3's Python API for direct programmatic access.

**Implementation:** `z3adapter/backends/json_backend.py`

### Execution Pipeline

```python
interpreter = Z3JSONInterpreter(program_path, verify_timeout, optimize_timeout)
interpreter.run()
sat_count, unsat_count = interpreter.get_verification_counts()
```

### Z3JSONInterpreter Pipeline

The interpreter processes JSON programs through three main stages:

**Step 1: SortManager** (`z3adapter/dsl/sorts.py`)

First, the system topologically sorts type definitions to handle dependencies, then creates Z3 sorts:

- Built-in: `BoolSort()`, `IntSort()`, `RealSort()` (pre-defined)
- Custom: `DeclareSort(name)`, `EnumSort(name, values)`, `BitVecSort(n)`, `ArraySort(domain, range)`

For example, an ArraySort creates dependencies:
```json
{"name": "IntArray", "type": "ArraySort(IntSort, IntSort)"}
```

This requires `IntSort` to be defined first (fortunately, it's built-in) before creating `IntArray`.

**Step 2: ExpressionParser** (`z3adapter/dsl/expressions.py`)

Next, the parser evaluates logical expressions from strings using a restricted `eval()`:

```python
safe_globals = {**Z3_OPERATORS, **functions}
context = {**functions, **constants, **variables, **quantified_vars}
ExpressionValidator.safe_eval(expr_str, safe_globals, context)
```

Only whitelisted operators are permitted:
```python
Z3_OPERATORS = {
    "And", "Or", "Not", "Implies", "If", "Distinct",
    "Sum", "Product", "ForAll", "Exists", "Function", "Array", "BitVecVal"
}
```

**Step 3: Verifier** (`z3adapter/verification/verifier.py`)

Finally, the verifier tests each verification condition:
```python
result = solver.check(condition)  # Adds condition as hypothesis to KB
if result == sat:
    sat_count += 1
elif result == unsat:
    unsat_count += 1
```

**Verification Semantics:**
When calling `solver.check(φ)`, the system asks: "Is KB ∧ φ satisfiable?"

- **SAT**: φ is consistent with the knowledge base (possible scenario)
- **UNSAT**: φ contradicts the knowledge base (impossible scenario)

### Prompt Template

The JSON prompt template is more comprehensive than its SMT2 counterpart.

**Source:** `z3adapter/reasoning/prompt_template.py`

This 546-line specification of the JSON DSL includes these key sections:

**Sorts:**
```json
{"name": "Person", "type": "DeclareSort"}
```

**Functions:**
```json
{"name": "supports", "domain": ["Person", "Issue"], "range": "BoolSort"}
```

**Constants:**
```json
{"persons": {"sort": "Person", "members": ["nancy_pelosi"]}}
```

**Variables:**
Free variables for quantifier binding:
```json
{"name": "p", "sort": "Person"}
```

**Knowledge Base:**
```json
["ForAll([p], Implies(is_democrat(p), supports_abortion(p)))"]
```

**Verifications:**
The DSL supports three types of verifications:

1. **Simple constraint:**
```json
{"name": "test", "constraint": "supports_abortion(nancy)"}
```

2. **Existential:**
```json
{"name": "test", "exists": [{"name": "x", "sort": "Int"}], "constraint": "x > 0"}
```

3. **Universal:**
```json
{"name": "test", "forall": [{"name": "x", "sort": "Int"}],
 "implies": {"antecedent": "x > 0", "consequent": "x >= 1"}}
```

**Critical constraint:** The prompt enforces a single verification per question to avoid ambiguous results from testing both φ and ¬φ.

### File Extension

`.json`

## IKR Backend

The IKR (Intermediate Knowledge Representation) backend introduces a structured intermediate layer between natural language and SMT2. Instead of generating SMT2 directly, the LLM generates a structured JSON representation that is then **deterministically compiled** to SMT2.

**Implementation:** `z3adapter/backends/ikr_backend.py`

### Why IKR?

Direct SMT2 generation requires the LLM to simultaneously handle:
1. Semantic understanding of the question
2. Knowledge structuring (types, entities, relations)
3. SMT2 syntax (parentheses, keywords, order)

This cognitive load leads to syntax errors, especially on complex questions. IKR separates these concerns:

- **LLM task:** Semantic understanding + knowledge structuring (produces IKR JSON)
- **Compiler task:** Deterministic SMT2 generation (no syntax errors)

### IKR Schema

The IKR schema is defined using Pydantic models in `z3adapter/ikr/schema.py`:

```json
{
  "meta": {
    "question": "Would a vegetarian eat a plant burger?",
    "question_type": "yes_no"
  },
  "types": [
    {"name": "Person", "description": "A human individual"},
    {"name": "Food", "description": "An edible item"}
  ],
  "entities": [
    {"name": "vegetarian_person", "type": "Person"},
    {"name": "plant_burger", "type": "Food"}
  ],
  "relations": [
    {"name": "is_vegetarian", "signature": ["Person"], "range": "Bool"},
    {"name": "would_eat", "signature": ["Person", "Food"], "range": "Bool"}
  ],
  "facts": [
    {"predicate": "is_vegetarian", "arguments": ["vegetarian_person"], "source": "explicit"}
  ],
  "rules": [
    {
      "name": "vegetarians avoid meat",
      "quantified_vars": [{"name": "p", "type": "Person"}],
      "antecedent": {"predicate": "is_vegetarian", "arguments": ["p"]},
      "consequent": {"predicate": "avoids_meat", "arguments": ["p"]},
      "justification": "By definition"
    }
  ],
  "query": {
    "predicate": "would_eat",
    "arguments": ["vegetarian_person", "plant_burger"]
  }
}
```

### Schema Components

| Component | Purpose |
|-----------|---------|
| `meta` | Question text and type (yes_no, comparison, possibility) |
| `types` | Domain declarations (Person, Food, Location) |
| `entities` | Named individuals with their types |
| `relations` | Predicates and functions with signatures |
| `facts` | Ground assertions (explicit from question or background knowledge) |
| `rules` | Universally quantified implications |
| `query` | The property to check for satisfiability |

### Special Relation Properties

Relations can declare special properties that generate axioms automatically:

**Symmetric relations:**
```json
{"name": "knows", "signature": ["Person", "Person"], "range": "Bool", "symmetric": true}
```
Generates: `(forall ((x Person) (y Person)) (= (knows x y) (knows y x)))`

**Transitive relations:**
```json
{"name": "greater_than", "signature": ["Number", "Number"], "range": "Bool", "transitive": true}
```
Generates: `(forall ((x Number) (y Number) (z Number)) (=> (and (greater_than x y) (greater_than y z)) (greater_than x z)))`

### Execution Pipeline

```python
# 1. Load and validate IKR JSON
ikr = IKR.model_validate(json_data)

# 2. Compile to SMT2 (deterministic, no syntax errors)
compiler = IKRCompiler()
smt2_code = compiler.compile(ikr)

# 3. Execute via Z3 CLI
result = subprocess.run([z3_path, program_path], ...)
```

### Two-Stage Prompting (Default)

IKR uses two-stage prompting by default for improved accuracy:

**Stage 1:** Extract explicit knowledge from the question
```python
prompt = build_ikr_stage1_prompt(question)
# LLM generates: types, entities, relations, explicit facts, query
```

**Stage 2:** Generate background knowledge given the explicit IKR
```python
prompt = build_ikr_stage2_prompt(current_ikr_json)
# LLM adds: background facts, rules with justifications
```

The two stages are then merged into a complete IKR.

**Benefits:**
- Reduces cognitive load per LLM call
- Makes background knowledge generation more targeted
- Enables debugging of each stage independently
- Stage 2 failures gracefully degrade (returns Stage 1 result with empty rules)

**Configuration:**
```python
# Two-stage (default)
pot = ProofOfThought(llm_client=client, backend="ikr", ikr_two_stage=True)

# Single-stage (for simpler questions or fewer API calls)
pot = ProofOfThought(llm_client=client, backend="ikr", ikr_two_stage=False)
```

The `GenerationResult` includes metadata about the two-stage process:
```python
result.two_stage        # True if two-stage was used
result.stage1_response  # Raw LLM response from Stage 1
result.stage2_response  # Raw LLM response from Stage 2
```

### Prompt Templates

**Source:** `z3adapter/reasoning/ikr_prompt_template.py`

The IKR prompts guide the LLM to produce valid IKR JSON with:
- Clear schema specification
- Worked examples
- Guidance on fact sources (explicit vs background)
- Rule structure with justifications

### File Extension

`.json` (IKR files are JSON, compiled to `.smt2` internally)

## Souffle Backend

The Souffle backend compiles IKR to Datalog and executes via the [Souffle](https://souffle-lang.github.io/) Datalog engine. This provides an alternative execution semantics based on **derivability** rather than **satisfiability**.

**Implementation:** `z3adapter/backends/souffle_backend.py`

### Why Souffle/Datalog?

While SMT2 uses satisfiability checking ("is this query consistent with the knowledge base?"), Datalog uses forward chaining derivation ("can this query be derived from the facts and rules?"). This is often more intuitive for knowledge-graph style reasoning:

| Aspect | SMT2 (Z3) | Datalog (Souffle) |
|--------|-----------|-------------------|
| **Semantics** | SAT/UNSAT (consistency) | Derivable/Not derivable |
| **Query model** | "Is φ consistent with KB?" | "Can φ be derived from KB?" |
| **Closed-world** | Open-world by default | Closed-world assumption |
| **Recursion** | Requires careful encoding | Native support |
| **Performance** | Best for complex constraints | Best for recursive queries |

### Execution Pipeline

```python
# 1. Load and validate IKR JSON
ikr = IKR.model_validate(json_data)

# 2. Compile to Souffle Datalog
compiler = IKRSouffleCompiler()
souffle_program = compiler.compile(ikr)

# 3. Write .dl program and .facts files
program_path, facts_dir = compiler.write_program(souffle_program, output_dir)

# 4. Execute via Souffle CLI
runner = OfficialSouffleRunner()
result = runner.run(program_path, facts_dir, output_dir)

# 5. Check if query_result relation has any tuples
answer = len(result.output_tuples["query_result"]) > 0
```

### IKR to Datalog Compilation

The compiler maps IKR components to Datalog:

| IKR Component | Datalog Equivalent |
|---------------|-------------------|
| Types | Type comments (mapped to `symbol`) |
| Relations | `.decl` statements |
| Entities | Constants in `.facts` files |
| Facts | Tab-separated `.facts` files |
| Rules | Horn clauses (`head :- body.`) |
| Query | Output relation derivation rule |

**Example IKR rule:**
```json
{
  "antecedent": {"and": [
    {"predicate": "is_vegetarian", "arguments": ["p"]},
    {"predicate": "is_plant_based", "arguments": ["f"]}
  ]},
  "consequent": {"predicate": "would_eat", "arguments": ["p", "f"]}
}
```

**Compiled Datalog:**
```prolog
would_eat(p, f) :- is_vegetarian(p), is_plant_based(f).
```

### Runner Abstraction

The Souffle backend uses a runner abstraction to enable future support for different Datalog engines:

```python
# Protocol definition
class SouffleRunner(Protocol):
    def run(self, program_path: Path, facts_dir: Path, output_dir: Path) -> RunResult: ...
    def is_available(self) -> bool: ...
    def get_version(self) -> str | None: ...

# Current implementation
runner = OfficialSouffleRunner()  # Uses souffle CLI

# Future: mini-souffle support
# runner = MiniSouffleRunner()  # Uses mini-souffle
```

**CLI invocation:**
```bash
souffle -F facts_dir -D output_dir program.dl
```

### Usage

**High-Level API (Recommended):**
```python
from z3adapter.reasoning import ProofOfThought

# Initialize with Souffle backend (requires Souffle installed)
pot = ProofOfThought(
    llm_client=client,
    model="gpt-4o",
    backend="souffle",
    ikr_two_stage=True,  # Uses IKR generation (default)
    souffle_path="/custom/path/to/souffle",  # Optional: custom path
)

# Query naturally
result = pot.query("Would a vegetarian eat a plant burger?")
print(result.answer)   # True (derivable) or False (not derivable)
print(result.success)  # True if execution succeeded
```

**Low-Level Backend API:**
```python
from z3adapter.backends import SouffleBackend

# Initialize (requires Souffle installed)
backend = SouffleBackend()

# Execute an IKR JSON file directly
result = backend.execute("path/to/ikr.json")

print(result.answer)   # True (derivable) or False (not derivable)
print(result.success)  # True if execution succeeded
```

### Negation Handling

Souffle uses **stratified negation**. The compiler generates negation-as-failure (`!`) for negated queries:

```prolog
// Negated query: is the sky NOT blue?
query_result() :- !is_blue(sky).
```

Note: Negated base facts require special handling and generate warnings.

### Prerequisites

Install Souffle:
- **Ubuntu:** `sudo apt-get install souffle`
- **macOS:** `brew install souffle`
- **From source:** https://souffle-lang.github.io/install

### File Extension

`.json` (Same IKR input format as IKR/SMT2 backend)

## Benchmark Performance

Performance comparison across datasets reveals notable differences between the backends.

**Results from** `experiments_pipeline.py` (100 samples per dataset, GPT-5, `max_attempts=3`):

| Dataset | SMT2 Accuracy | JSON Accuracy | SMT2 Success | JSON Success |
|---------|---------------|---------------|--------------|--------------|
| ProntoQA | 100% | 99% | 100% | 100% |
| FOLIO | 69% | 76% | 99% | 94% |
| ProofWriter | 99% | 96% | 99% | 96% |
| ConditionalQA | 83% | 76% | 100% | 89% |
| StrategyQA | 84% | 68% | 100% | 86% |

**Success Rate** represents the percentage of queries that complete without error (including both generation and execution).

Overall, SMT2 achieves higher accuracy on 4 out of 5 datasets, while JSON shows greater success rate variance (86-100% compared to SMT2's 99-100%).

**Note:** IKR backend benchmarks are pending. The IKR backend is expected to improve accuracy on StrategyQA (commonsense reasoning) by making background knowledge explicit.

## Implementation Differences

The backends differ in several implementation details.

### Program Generation

**SMT2:** Extracts programs from markdown via:
```python
pattern = r"```smt2\s*([\s\S]*?)\s*```"
```

**JSON:** Extracts and parses via:
```python
pattern = r"```json\s*(\{[\s\S]*?\})\s*```"
json.loads(match.group(1))
```

### Error Handling

Error handling varies significantly between backends.

**SMT2:**
- Subprocess timeout → `TimeoutExpired`
- Parse errors → regex mismatch → `answer=None`
- Z3 errors in stderr → still parsed

**JSON:**
- JSON parse error → extraction failure
- Z3 Python API exception → caught in `try/except`
- Invalid sort reference → `ValueError` during SortManager
- Expression eval error → `ValueError` during ExpressionParser

### Timeout Configuration

Timeout handling differs between the two backends.

**SMT2:**
- Uses a single timeout parameter: `verify_timeout` (ms)
- Converts to seconds for Z3 CLI: `verify_timeout // 1000`
- Applies a hard subprocess timeout: `timeout_seconds + 10`

**JSON:**
- Uses two separate timeouts: `verify_timeout` (ms) and `optimize_timeout` (ms)
- Sets timeout via `solver.set("timeout", verify_timeout)` in Verifier
- Timeout applies per individual `solver.check()` call

## Backend Selection Code

The system selects backends at runtime based on configuration:

```python
if backend == "json":
    from z3adapter.backends.json_backend import JSONBackend
    backend_instance = JSONBackend(verify_timeout, optimize_timeout)
elif backend == "ikr":
    from z3adapter.backends.ikr_backend import IKRBackend
    backend_instance = IKRBackend(verify_timeout, z3_path)
elif backend == "souffle":
    from z3adapter.backends.souffle_backend import SouffleBackend
    from z3adapter.runners import OfficialSouffleRunner
    runner = OfficialSouffleRunner(souffle_path=souffle_path)
    backend_instance = SouffleBackend(verify_timeout, runner=runner)
else:  # smt2
    from z3adapter.backends.smt2_backend import SMT2Backend
    backend_instance = SMT2Backend(verify_timeout, z3_path)
```

**File:** `z3adapter/reasoning/proof_of_thought.py:121-141`

**Note:** The Souffle backend uses the same IKR generation as the IKR backend, so `generator_backend` is mapped to `"ikr"` internally when `backend="souffle"`.

## Prompt Selection

The appropriate prompt template is chosen based on the selected backend:

```python
# IKR routes to two-stage by default
if self.backend == "ikr" and self.ikr_two_stage:
    return self._generate_ikr_two_stage(question, max_tokens)

# Single-stage prompting for JSON, SMT2, or IKR with ikr_two_stage=False
if self.backend == "json":
    prompt = build_prompt(question)
elif self.backend == "ikr":
    prompt = build_ikr_single_stage_prompt(question)
else:  # smt2
    prompt = build_smt2_prompt(question)
```

**File:** `z3adapter/reasoning/program_generator.py:108-119`

All prompts include few-shot examples and format specifications:
- **SMT2:** Emphasizes S-expression syntax
- **JSON:** Detailed guidance on variable scoping and quantifier semantics
- **IKR:** Structured schema with explicit/background knowledge separation
