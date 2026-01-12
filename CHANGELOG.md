# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026

### Added
- **IKR Backend**: Intermediate Knowledge Representation for improved SMT generation
  - Structured schema: types, entities, relations, facts, rules, query
  - Deterministic compilation to SMT2 (eliminates syntax errors)
  - Two-stage prompting (default): Stage 1 extracts explicit facts, Stage 2 generates background knowledge
  - Support for symmetric and transitive relation axioms
  - Pydantic validation for schema correctness
  - Graceful degradation: Stage 2 failures return Stage 1 result with empty rules
- IKR documentation and examples
- `ikr_two_stage` parameter on `ProofOfThought` and `Z3ProgramGenerator`
- `GenerationResult` metadata: `two_stage`, `stage1_response`, `stage2_response`
- Unit tests for two-stage IKR generation

### Changed
- `ProofOfThought` now accepts `backend="ikr"` option
- `Z3ProgramGenerator` supports IKR extraction from LLM responses
- IKR backend uses two-stage prompting by default (set `ikr_two_stage=False` for single-stage)

## [1.0.1] - 2025

### Added
- **VerifiedQA**: Two-stage LLM reasoning with formal verification
  - LLM answers naturally with chain-of-thought reasoning
  - Z3 verifies consistency between facts and claimed answer
  - Automatic answer correction when contradictions are detected
- Development documentation (`DEVELOPMENT.md`)
- Ollama support for local LLM inference
- GitHub issue templates for bug reports and feature requests

### Changed
- Updated README to reflect PyPI release

## [1.0.0] - 2025

### Added
- **SMT2 Backend**: Standard SMT-LIB 2.0 format execution via Z3 CLI
  - More portable and secure than JSON backend
  - Direct Z3 CLI integration
- **Postprocessors**: Enhancement techniques for improved accuracy
  - Self-refine: Iterative refinement through self-critique
  - Self-consistency: Majority voting across reasoning paths
  - Decomposed: Question decomposition
  - Least-to-most: Progressive problem solving
- **MkDocs Documentation Site**: Comprehensive documentation
  - DSL specification
  - API reference
  - Backend comparison
  - Benchmark results
- **Benchmark Suite**: Evaluation on multiple datasets
  - ProntoQA
  - FOLIO
  - ProofWriter
  - ConditionalQA
  - StrategyQA

### Changed
- Improved benchmark evaluation pipeline
- Enhanced error handling and retry logic

## [0.1.0] - 2024

### Added
- Initial release
- **ProofOfThought**: Main reasoning API
  - Natural language to Z3 program translation
  - Automatic retry with error feedback
  - JSON DSL backend
- **Z3JSONInterpreter**: JSON DSL execution engine
  - Sort management (enum, int, bool, real, array, datatype)
  - Expression parsing and evaluation
  - Constraint verification
  - Optimization support
- **EvaluationPipeline**: Batch evaluation for datasets
  - Parallel execution support
  - Result caching
  - Metrics computation (accuracy, precision, recall, F1)
- **Command-line interface**: Execute JSON programs directly
- **Security**: Expression validator for safe evaluation
- Azure OpenAI support
- Unit and integration test suite
