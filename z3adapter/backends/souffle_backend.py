"""Souffle backend using Datalog execution.

This backend takes IKR JSON files, compiles them to Souffle Datalog,
and executes via the Souffle CLI. Unlike the SMT2 backend which uses
satisfiability checking, this backend uses Datalog's derivability semantics:
- Query derivable → True
- Query not derivable → False
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

from pydantic import ValidationError

from z3adapter.backends.abstract import Backend, VerificationResult
from z3adapter.ikr.schema import IKR
from z3adapter.ikr.souffle_compiler import IKRSouffleCompiler
from z3adapter.reasoning.ikr_prompt_template import IKR_SINGLE_STAGE_INSTRUCTIONS
from z3adapter.runners import OfficialSouffleRunner, SouffleRunner

logger = logging.getLogger(__name__)


class SouffleBackend(Backend):
    """Backend for executing IKR programs via Souffle Datalog.

    This backend:
    1. Parses IKR JSON input
    2. Validates the schema with Pydantic
    3. Compiles to Souffle Datalog (.dl + .facts files)
    4. Executes via Souffle CLI
    5. Returns verification results based on derivability

    Semantics difference from SMT2:
    - SMT2: SAT = query consistent with facts, UNSAT = contradiction
    - Souffle: derivable = query follows from facts, not derivable = unknown/false
    """

    def __init__(
        self,
        verify_timeout: int = 30000,
        runner: SouffleRunner | None = None,
        keep_files: bool = False,
    ) -> None:
        """Initialize Souffle backend.

        Args:
            verify_timeout: Timeout for verification in milliseconds
            runner: Souffle runner to use (default: OfficialSouffleRunner)
            keep_files: If True, keep generated files for debugging

        Raises:
            RuntimeError: If Souffle is not available
        """
        self.verify_timeout = verify_timeout
        self.keep_files = keep_files
        self._compiler = IKRSouffleCompiler()

        # Initialize runner
        if runner is not None:
            self._runner = runner
        else:
            self._runner = OfficialSouffleRunner()

        # Validate Souffle is available
        if not self._runner.is_available():
            raise RuntimeError(
                "Souffle is not available.\n"
                "Please install Souffle:\n"
                "  - Ubuntu: sudo apt-get install souffle\n"
                "  - macOS: brew install souffle\n"
                "  - Or build from source: https://souffle-lang.github.io/install"
            )

    def execute(self, program_path: str) -> VerificationResult:
        """Execute an IKR program by compiling to Souffle and running.

        Args:
            program_path: Path to IKR JSON file

        Returns:
            VerificationResult with answer and execution details
        """
        work_dir = None
        try:
            # Step 1: Load and parse IKR JSON
            with open(program_path, "r", encoding="utf-8") as f:
                ikr_data = json.load(f)

            # Step 2: Validate with Pydantic
            ikr = IKR.model_validate(ikr_data)

            # Step 3: Compile to Souffle
            souffle_program = self._compiler.compile(ikr)
            logger.debug(f"Compiled Souffle program:\n{souffle_program.program}")

            # Step 4: Write to directory
            if self.keep_files:
                work_dir = Path(program_path).parent / "souffle_output"
                work_dir.mkdir(exist_ok=True)
            else:
                work_dir = Path(tempfile.mkdtemp(prefix="souffle_"))

            program_path_dl, facts_dir = self._compiler.write_program(
                souffle_program, work_dir
            )
            output_dir = work_dir / "output"
            output_dir.mkdir(exist_ok=True)

            # Step 5: Execute Souffle
            timeout_seconds = self.verify_timeout / 1000.0
            run_result = self._runner.run(
                program_path=program_path_dl,
                facts_dir=facts_dir,
                output_dir=output_dir,
                timeout=timeout_seconds,
            )

            if not run_result.success:
                return VerificationResult(
                    answer=None,
                    sat_count=0,
                    unsat_count=0,
                    output=f"stdout: {run_result.stdout}\nstderr: {run_result.stderr}",
                    success=False,
                    error=run_result.error,
                )

            # Step 6: Determine answer from query_result
            query_tuples = run_result.output_tuples.get(
                souffle_program.query_relation, []
            )

            # In Datalog: if query_result has any tuples, query is derivable (True)
            # If empty, query is not derivable (False for closed-world assumption)
            if query_tuples:
                answer = True
                sat_count = 1
                unsat_count = 0
            else:
                answer = False
                sat_count = 0
                unsat_count = 1

            output_info = [
                "[IKR -> Souffle compilation successful]",
                f"Query relation: {souffle_program.query_relation}",
                f"Query result tuples: {len(query_tuples)}",
                f"Answer: {'derivable (True)' if answer else 'not derivable (False)'}",
            ]

            if run_result.stdout:
                output_info.append(f"\nSouffle stdout:\n{run_result.stdout}")
            if run_result.stderr:
                output_info.append(f"\nSouffle stderr:\n{run_result.stderr}")

            return VerificationResult(
                answer=answer,
                sat_count=sat_count,
                unsat_count=unsat_count,
                output="\n".join(output_info),
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
        except ValueError as e:
            error_msg = f"IKR compilation error: {e}"
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
            error_msg = f"Unexpected error processing IKR: {e}"
            logger.error(error_msg)
            return VerificationResult(
                answer=None,
                sat_count=0,
                unsat_count=0,
                output="",
                success=False,
                error=error_msg,
            )
        finally:
            # Clean up temp directory if not keeping
            if work_dir and not self.keep_files:
                try:
                    shutil.rmtree(work_dir)
                except OSError:
                    pass

    def get_file_extension(self) -> str:
        """Get the file extension for IKR programs."""
        return ".json"

    def get_prompt_template(self) -> str:
        """Get the prompt template for IKR generation.

        Uses the same IKR prompt as the IKR/SMT2 backend since
        the input format is identical.
        """
        return IKR_SINGLE_STAGE_INSTRUCTIONS
