"""IKR backend using intermediate knowledge representation.

This backend takes IKR JSON files, compiles them to SMT2, and executes
via Z3. The key benefit is deterministic compilation - eliminating
the SMT2 syntax errors common with direct LLM generation.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from pydantic import ValidationError

from z3adapter.backends.abstract import Backend, VerificationResult
from z3adapter.ikr.compiler import IKRCompiler
from z3adapter.ikr.schema import IKR
from z3adapter.reasoning.ikr_prompt_template import IKR_SINGLE_STAGE_INSTRUCTIONS

logger = logging.getLogger(__name__)


class IKRBackend(Backend):
    """Backend for executing IKR programs via compilation to SMT2.

    This backend:
    1. Parses IKR JSON input
    2. Validates the schema with Pydantic
    3. Compiles to SMT2 deterministically
    4. Executes via Z3 CLI
    5. Returns verification results
    """

    def __init__(
        self,
        verify_timeout: int = 10000,
        z3_path: str = "z3",
        keep_smt2: bool = False,
    ) -> None:
        """Initialize IKR backend.

        Args:
            verify_timeout: Timeout for verification in milliseconds
            z3_path: Path to Z3 executable
            keep_smt2: If True, keep generated SMT2 files for debugging

        Raises:
            FileNotFoundError: If Z3 executable is not found
        """
        self.verify_timeout = verify_timeout
        self.z3_path = z3_path
        self.keep_smt2 = keep_smt2
        self._compiler = IKRCompiler()

        # Validate Z3 is available
        if not shutil.which(z3_path):
            raise FileNotFoundError(
                f"Z3 executable not found: '{z3_path}'\n"
                f"Please install Z3:\n"
                f"  - pip install z3-solver\n"
                f"  - Or download from: https://github.com/Z3Prover/z3/releases"
            )

    def execute(self, program_path: str) -> VerificationResult:
        """Execute an IKR program by compiling to SMT2 and running Z3.

        Args:
            program_path: Path to IKR JSON file

        Returns:
            VerificationResult with answer and execution details
        """
        smt2_path = None
        try:
            # Step 1: Load and parse IKR JSON
            with open(program_path, "r", encoding="utf-8") as f:
                ikr_data = json.load(f)

            # Step 2: Validate with Pydantic
            ikr = IKR.model_validate(ikr_data)

            # Step 3: Compile to SMT2
            smt2_code = self._compiler.compile(ikr)
            logger.debug(f"Compiled SMT2:\n{smt2_code}")

            # Step 4: Write SMT2 to temporary file
            if self.keep_smt2:
                # Keep next to the IKR file
                smt2_path = str(Path(program_path).with_suffix(".smt2"))
            else:
                # Use temp directory
                fd, smt2_path = tempfile.mkstemp(suffix=".smt2")
                os.close(fd)

            with open(smt2_path, "w", encoding="utf-8") as f:
                f.write(smt2_code)

            # Step 5: Execute Z3
            result = self._run_z3(smt2_path)
            result.output = f"[IKR -> SMT2 compilation successful]\n{result.output}"
            return result

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
            # Clean up temp file if not keeping
            if smt2_path and not self.keep_smt2 and os.path.exists(smt2_path):
                try:
                    os.remove(smt2_path)
                except OSError:
                    pass

    def _run_z3(self, smt2_path: str) -> VerificationResult:
        """Run Z3 on the compiled SMT2 file.

        Args:
            smt2_path: Path to SMT2 file

        Returns:
            VerificationResult from Z3 execution
        """
        try:
            timeout_seconds = self.verify_timeout // 1000

            result = subprocess.run(
                [self.z3_path, f"-T:{timeout_seconds}", smt2_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 10,
            )

            output = result.stdout + result.stderr

            # Parse Z3 output
            sat_count, unsat_count = self._parse_z3_output(output)
            answer = self.determine_answer(sat_count, unsat_count)

            return VerificationResult(
                answer=answer,
                sat_count=sat_count,
                unsat_count=unsat_count,
                output=output,
                success=True,
            )

        except subprocess.TimeoutExpired:
            error_msg = f"Z3 execution timed out after {timeout_seconds}s"
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
            error_msg = f"Z3 execution error: {e}"
            logger.error(error_msg)
            return VerificationResult(
                answer=None,
                sat_count=0,
                unsat_count=0,
                output="",
                success=False,
                error=error_msg,
            )

    def _parse_z3_output(self, output: str) -> tuple[int, int]:
        """Parse Z3 output to count sat/unsat results."""
        sat_pattern = r"(?<!un)\bsat\b"
        unsat_pattern = r"\bunsat\b"

        sat_matches = re.findall(sat_pattern, output, re.IGNORECASE)
        unsat_matches = re.findall(unsat_pattern, output, re.IGNORECASE)

        return len(sat_matches), len(unsat_matches)

    def get_file_extension(self) -> str:
        """Get the file extension for IKR programs."""
        return ".json"

    def get_prompt_template(self) -> str:
        """Get the prompt template for IKR generation."""
        return IKR_SINGLE_STAGE_INSTRUCTIONS
