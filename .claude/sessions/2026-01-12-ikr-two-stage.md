# Session: 2026-01-12 - IKR Two-Stage Prompting Implementation

## Summary
Implemented two-stage prompting for the IKR backend. This was item #1 from the pending work in the previous IKR session.

## What was done

### Core Implementation
- Added `IKRStageResult` dataclass for individual stage results
- Added `_generate_ikr_two_stage()` method to `Z3ProgramGenerator`
- Added `_run_ikr_stage1()` for explicit knowledge extraction
- Added `_run_ikr_stage2()` for background knowledge generation
- Added `_merge_ikr_stages()` for combining Stage 1 and Stage 2 outputs
- Added `ikr_two_stage` parameter (default `True`) to control two-stage vs single-stage

### GenerationResult Updates
- Added `two_stage: bool` field to track if two-stage was used
- Added `stage1_response: str | None` for Stage 1 raw response
- Added `stage2_response: str | None` for Stage 2 raw response

### Error Handling
- Stage 1 failure: Returns failure with error details
- Stage 2 failure: Returns partial success with Stage 1 IKR (graceful degradation)
- Stage 1 missing fields: Returns failure with specific field names

### API Updates
- `ProofOfThought.__init__()` now accepts `ikr_two_stage` parameter
- Parameter is passed through to `Z3ProgramGenerator`

## Files Modified
- `z3adapter/reasoning/program_generator.py` - Core two-stage implementation
- `z3adapter/reasoning/proof_of_thought.py` - Added `ikr_two_stage` parameter
- `docs/backends.md` - Updated IKR documentation with two-stage details
- `CHANGELOG.md` - Added two-stage features to v1.1.0

## Files Created
- `tests/unit/test_ikr_two_stage.py` - Unit tests with mocked LLM calls

## Test Coverage
The new test file includes:
- Successful two-stage generation
- Stage 1 failure handling
- Stage 1 missing required fields
- Stage 2 failure with partial result
- Single-stage fallback (ikr_two_stage=False)
- Background facts merging
- Empty Stage 2 handling
- Prompt content verification
- GenerationResult metadata verification

## Usage Example
```python
from z3adapter.reasoning import ProofOfThought

# Two-stage (default)
pot = ProofOfThought(llm_client=client, model="gpt-4o", backend="ikr")

# Single-stage (for simpler questions)
pot = ProofOfThought(llm_client=client, backend="ikr", ikr_two_stage=False)

result = pot.query("Would a vegetarian eat a plant burger?")
print(result.answer)
```

## Pending/Future Work
1. Benchmark IKR two-stage vs single-stage on StrategyQA
2. Add reasoning_chain support to IKR schema
3. Consider knowledge base caching for common background knowledge
4. Add error feedback retry for IKR two-stage (currently only single-stage has retry)
