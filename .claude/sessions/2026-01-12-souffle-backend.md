# Session: 2026-01-12 - Souffle Backend Implementation

## Summary
Implemented a Souffle/Datalog backend for IKR, enabling derivability-based reasoning as an alternative to Z3's satisfiability checking. The implementation includes a runner abstraction layer designed for future mini-souffle support.

## What was done

### Core Implementation

**Runner Abstraction (`z3adapter/runners/`)**
- Created `SouffleRunner` protocol for backend-agnostic Souffle execution
- Implemented `OfficialSouffleRunner` using Souffle CLI
- `RunResult` dataclass with output files, tuples, and error handling
- CLI invocation: `souffle -F facts_dir -D output_dir program.dl`

**IKR to Souffle Compiler (`z3adapter/ikr/souffle_compiler.py`)**
- `IKRSouffleCompiler` class compiling IKR to Datalog
- `SouffleProgram` dataclass with .dl source and facts dictionary
- Type mapping: custom types → `symbol`, Int → `number`, Real → `float`
- Relations → `.decl` statements with auto-generated argument names
- Facts → tab-separated `.facts` files
- Rules → Horn clauses (`head :- body.`)
- Query → derivation rule for `query_result()` output relation
- Symmetric/transitive relation axioms
- `write_program()` method for file output

**Souffle Backend (`z3adapter/backends/souffle_backend.py`)**
- `SouffleBackend` class implementing `Backend` protocol
- Takes IKR JSON, compiles to Souffle, executes, returns result
- Derivability semantics: non-empty query_result → True, empty → False
- Graceful error handling for JSON, schema, and compilation errors

### Schema Fix
- Added `model_config = {"populate_by_name": True}` to `RuleCondition`
- Enables constructing with `and_=` instead of requiring alias `"and"`

## Files Created
- `z3adapter/runners/__init__.py` - Runner module exports
- `z3adapter/runners/base.py` - RunResult, SouffleRunner protocol
- `z3adapter/runners/official.py` - OfficialSouffleRunner implementation
- `z3adapter/ikr/souffle_compiler.py` - IKR to Souffle compiler
- `z3adapter/backends/souffle_backend.py` - Souffle backend
- `tests/unit/test_souffle_compiler.py` - Compiler tests (16 tests)
- `tests/unit/test_souffle_runner.py` - Runner tests (11 tests)
- `tests/unit/test_souffle_backend.py` - Backend tests (14 tests)

## Files Modified
- `z3adapter/backends/__init__.py` - Added SouffleBackend export
- `z3adapter/ikr/__init__.py` - Added IKRSouffleCompiler, SouffleProgram exports
- `z3adapter/ikr/schema.py` - Added model_config to RuleCondition
- `docs/backends.md` - Added Souffle backend documentation
- `CHANGELOG.md` - Added Souffle backend to v1.1.0

## Test Coverage
All 204 tests pass including 41 new Souffle-related tests:
- Compiler: relation declarations, rules, facts, symmetric/transitive axioms
- Runner: availability check, execution, timeout, output parsing
- Backend: derivable/not derivable queries, error handling, integration

## Key Design Decisions

1. **Runner abstraction**: Designed for swapping official Souffle with mini-souffle
2. **File-based interface**: Both implementations use identical CLI flags and file formats
3. **Derivability semantics**: Different from SMT2's satisfiability semantics
4. **Same IKR input**: Uses same IKR JSON format as IKR/SMT2 backend
5. **Closed-world assumption**: Not derivable = False (unlike Z3's open-world)

## Usage Example
```python
from z3adapter.backends import SouffleBackend

# Requires Souffle installed
backend = SouffleBackend()
result = backend.execute("path/to/ikr.json")

print(result.answer)   # True (derivable) or False (not derivable)
```

## Pending/Future Work
1. Add `backend="souffle"` option to `ProofOfThought` class
2. Implement `MiniSouffleRunner` when mini-souffle is ready
3. Benchmark Souffle vs SMT2 on StrategyQA (recursive reasoning)
4. Support disjunction in rule bodies (currently warns and uses first disjunct)
5. Handle negated base facts properly (currently warns)
