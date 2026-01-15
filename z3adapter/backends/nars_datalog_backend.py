"""NARS-Datalog backend using native Python inference.

This backend executes IKR programs using a Python-native Datalog engine
with integrated NARS truth value propagation. Unlike backends that rely
on external solvers, this provides:

- Truth values propagate through inference (not converted to weights)
- No external dependencies (pure Python)
- Query results include confidence scores
- Transparent inference with explanation

The backend follows the same interface as other backends (SMT2, Souffle)
so it can be used as a drop-in replacement.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from pydantic import ValidationError

from z3adapter.backends.abstract import Backend, VerificationResult
from z3adapter.ikr.schema import IKR
from z3adapter.ikr.nars_datalog import NARSDatalogEngine
from z3adapter.reasoning.ikr_prompt_template import IKR_SINGLE_STAGE_INSTRUCTIONS

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["NARSDatalogBackend"]


class NARSDatalogBackend(Backend):
    """Backend for executing IKR programs with NARS truth propagation.

    This backend:
    1. Parses IKR JSON input
    2. Validates the schema with Pydantic
    3. Runs native Python Datalog inference with NARS truth values
    4. Returns verification results with truth values

    Advantages over SMT2/Souffle backends:
    - Truth values propagate through inference naturally
    - No external dependencies (pure Python, no Z3 or Souffle needed)
    - Query results include confidence scores
    - Inference is transparent and debuggable
    """

    def __init__(
        self,
        max_iterations: int = 100,
        min_confidence: float = 0.01,
        truth_threshold: float = 0.5,
    ) -> None:
        """Initialize NARS-Datalog backend.

        Args:
            max_iterations: Maximum inference iterations before stopping
            min_confidence: Minimum confidence to keep derived facts
            truth_threshold: Frequency threshold for True/False answer
                If frequency >= threshold, answer is True; else False
        """
        self.max_iterations = max_iterations
        self.min_confidence = min_confidence
        self.truth_threshold = truth_threshold

    def execute(self, program_path: str) -> VerificationResult:
        """Execute an IKR program using NARS-Datalog inference.

        Args:
            program_path: Path to IKR JSON file

        Returns:
            VerificationResult with answer and truth values
        """
        try:
            # Step 1: Load and parse IKR JSON
            with open(program_path, "r", encoding="utf-8") as f:
                ikr_data = json.load(f)

            # Step 2: Validate with Pydantic
            ikr = IKR.model_validate(ikr_data)

            # Step 3: Create and run engine
            engine = NARSDatalogEngine(
                max_iterations=self.max_iterations,
                min_confidence=self.min_confidence,
            )
            engine.load_ikr(ikr)

            # Step 4: Query
            result = engine.query(ikr.query)

            # Step 5: Determine answer from truth value
            if result.found and result.truth_value:
                tv = result.truth_value
                if tv.frequency >= self.truth_threshold:
                    answer = True
                    sat_count = 1
                    unsat_count = 0
                else:
                    answer = False
                    sat_count = 0
                    unsat_count = 1
            else:
                # Not found = False under closed-world assumption
                answer = False
                sat_count = 0
                unsat_count = 1

            # Build output info
            output_lines = [
                "[NARS-Datalog inference completed]",
                f"Iterations: {result.iterations}",
                f"Facts derived: {result.facts_derived}",
                f"Query: {result.query_atom}",
                f"Found: {result.found}",
            ]

            if result.truth_value:
                output_lines.append(
                    f"Truth value: f={result.truth_value.frequency:.3f}, "
                    f"c={result.truth_value.confidence:.3f}"
                )
                output_lines.append(
                    f"Expectation: {result.truth_value.expectation():.3f}"
                )

            output_lines.append(f"Answer: {answer}")
            output_lines.append(f"Explanation: {result.explanation}")

            return VerificationResult(
                answer=answer,
                sat_count=sat_count,
                unsat_count=unsat_count,
                output="\n".join(output_lines),
                success=True,
            )

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in IKR file: {e}"
            logger.error(error_msg)
            return VerificationResult(
                answer=None,
                sat_count=0,
                unsat_count=0,
                output="",
                success=False,
                error=error_msg,
            )
        except ValidationError as e:
            error_msg = f"IKR schema validation failed:\n{e}"
            logger.error(error_msg)
            return VerificationResult(
                answer=None,
                sat_count=0,
                unsat_count=0,
                output="",
                success=False,
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Inference error: {e}"
            logger.exception(error_msg)
            return VerificationResult(
                answer=None,
                sat_count=0,
                unsat_count=0,
                output="",
                success=False,
                error=error_msg,
            )

    def get_file_extension(self) -> str:
        """Get the file extension for IKR programs.

        Returns:
            ".json" since IKR uses JSON format
        """
        return ".json"

    def get_prompt_template(self) -> str:
        """Get the prompt template for IKR generation.

        Uses the same IKR prompt as other IKR-based backends since
        the input format is identical.

        Returns:
            IKR prompt template string
        """
        return IKR_SINGLE_STAGE_INSTRUCTIONS
