"""Z3 DSL program generator using LLM."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from z3adapter.reasoning.prompt_template import build_prompt
from z3adapter.reasoning.smt2_prompt_template import build_smt2_prompt
from z3adapter.reasoning.ikr_prompt_template import (
    build_ikr_stage1_prompt,
    build_ikr_stage2_prompt,
    build_ikr_single_stage_prompt,
)

logger = logging.getLogger(__name__)

BackendType = Literal["json", "smt2", "ikr"]


@dataclass
class IKRStageResult:
    """Result from a single stage of IKR generation."""

    output: dict[str, Any] | None
    raw_response: str
    success: bool
    error: str | None = None


@dataclass
class GenerationResult:
    """Result of program generation."""

    program: dict[str, Any] | str | None  # JSON dict or SMT2 string
    raw_response: str
    success: bool
    backend: BackendType
    error: str | None = None
    # Two-stage IKR metadata
    stage1_response: str | None = None
    stage2_response: str | None = None
    two_stage: bool = False

    # Backward compatibility
    @property
    def json_program(self) -> dict[str, Any] | None:
        """Get JSON program (for backward compatibility)."""
        if self.backend == "json" and isinstance(self.program, dict):
            return self.program
        return None

    @property
    def smt2_program(self) -> str | None:
        """Get SMT2 program text."""
        if self.backend == "smt2" and isinstance(self.program, str):
            return self.program
        return None

    @property
    def ikr_program(self) -> dict[str, Any] | None:
        """Get IKR program (JSON dict)."""
        if self.backend == "ikr" and isinstance(self.program, dict):
            return self.program
        return None


class Z3ProgramGenerator:
    """Generate Z3 DSL programs from natural language questions using LLM."""

    def __init__(
        self,
        llm_client: Any,
        model: str = "gpt-4o",
        backend: BackendType = "smt2",
        ikr_two_stage: bool = True,
    ) -> None:
        """Initialize the program generator.

        Args:
            llm_client: LLM client (OpenAI, Anthropic, etc.)
            model: Model name to use
            backend: Backend type ("json", "smt2", or "ikr")
            ikr_two_stage: For IKR backend, use two-stage prompting (default True)
        """
        self.llm_client = llm_client
        self.model = model
        self.backend = backend
        self.ikr_two_stage = ikr_two_stage

    def generate(
        self,
        question: str,
        temperature: float = 0.1,
        max_tokens: int = 16384,
    ) -> GenerationResult:
        """Generate a Z3 DSL program from a question.

        Args:
            question: Natural language question
            temperature: LLM temperature
            max_tokens: Maximum tokens for response (default 16384 for GPT-5)

        Returns:
            GenerationResult with program or error
        """
        # Route IKR to two-stage when configured
        if self.backend == "ikr" and self.ikr_two_stage:
            return self._generate_ikr_two_stage(question, max_tokens)

        try:
            # Select prompt based on backend
            if self.backend == "json":
                prompt = build_prompt(question)
            elif self.backend == "ikr":
                prompt = build_ikr_single_stage_prompt(question)
            else:  # smt2
                prompt = build_smt2_prompt(question)

            # Make LLM API call (compatible with both OpenAI and Azure OpenAI)
            # Azure OpenAI requires content as string, not list
            # GPT-5 only supports temperature=1 (default), so don't pass it
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
            )

            raw_response = response.choices[0].message.content

            # Extract program based on backend
            if self.backend == "json":
                program: dict[str, Any] | str | None = self._extract_json(raw_response)
                error_msg = "Failed to extract valid JSON from response"
            elif self.backend == "ikr":
                program = self._extract_json(raw_response)  # IKR is also JSON
                error_msg = "Failed to extract valid IKR JSON from response"
            else:  # smt2
                program = self._extract_smt2(raw_response)
                error_msg = "Failed to extract valid SMT2 from response"

            if program:
                return GenerationResult(
                    program=program,
                    raw_response=raw_response,
                    success=True,
                    backend=self.backend,
                )
            else:
                # Log the raw response to help debug extraction failures
                logger.debug(f"Raw LLM response:\n{raw_response[:1000]}...")
                return GenerationResult(
                    program=None,
                    raw_response=raw_response,
                    success=False,
                    backend=self.backend,
                    error=error_msg,
                )

        except Exception as e:
            logger.error(f"Error generating program: {e}")
            return GenerationResult(
                program=None,
                raw_response="",
                success=False,
                backend=self.backend,
                error=str(e),
            )

    def generate_with_feedback(
        self,
        question: str,
        error_trace: str,
        previous_response: str,
        temperature: float = 0.1,
        max_tokens: int = 16384,
    ) -> GenerationResult:
        """Regenerate program with error feedback.

        Args:
            question: Original question
            error_trace: Error message from previous attempt
            previous_response: Previous LLM response
            temperature: LLM temperature
            max_tokens: Maximum tokens (default 16384 for GPT-5)

        Returns:
            GenerationResult with corrected program
        """
        try:
            # Select prompt based on backend
            if self.backend == "json":
                prompt = build_prompt(question)
                format_msg = "Please fix the JSON accordingly."
            elif self.backend == "ikr":
                prompt = build_ikr_single_stage_prompt(question)
                format_msg = "Please fix the IKR JSON accordingly."
            else:  # smt2
                prompt = build_smt2_prompt(question)
                format_msg = "Please fix the SMT2 program accordingly."

            feedback_message = (
                f"There was an error processing your response:\n{error_trace}\n{format_msg}"
            )

            # Multi-turn conversation with error feedback
            # Compatible with both OpenAI and Azure OpenAI
            # GPT-5 only supports temperature=1 (default), so don't pass it
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": previous_response},
                    {"role": "user", "content": feedback_message},
                ],
                max_completion_tokens=max_tokens,
            )

            raw_response = response.choices[0].message.content

            # Extract program based on backend
            if self.backend == "json":
                program: dict[str, Any] | str | None = self._extract_json(raw_response)
                error_msg = "Failed to extract valid JSON from feedback response"
            elif self.backend == "ikr":
                program = self._extract_json(raw_response)
                error_msg = "Failed to extract valid IKR JSON from feedback response"
            else:  # smt2
                program = self._extract_smt2(raw_response)
                error_msg = "Failed to extract valid SMT2 from feedback response"

            if program:
                return GenerationResult(
                    program=program,
                    raw_response=raw_response,
                    success=True,
                    backend=self.backend,
                )
            else:
                # Log the raw response to help debug extraction failures
                logger.debug(f"Raw LLM feedback response:\n{raw_response[:1000]}...")
                return GenerationResult(
                    program=None,
                    raw_response=raw_response,
                    success=False,
                    backend=self.backend,
                    error=error_msg,
                )

        except Exception as e:
            logger.error(f"Error generating program with feedback: {e}")
            return GenerationResult(
                program=None,
                raw_response="",
                success=False,
                backend=self.backend,
                error=str(e),
            )

    def _generate_ikr_two_stage(
        self,
        question: str,
        max_tokens: int = 16384,
    ) -> GenerationResult:
        """Generate IKR using two-stage prompting.

        Stage 1: Extract explicit facts, types, entities, relations, and query
        Stage 2: Generate background knowledge (facts and rules) given Stage 1 output

        Args:
            question: Natural language question
            max_tokens: Maximum tokens for each LLM response

        Returns:
            GenerationResult with merged IKR or error
        """
        # Stage 1: Extract explicit knowledge
        stage1_result = self._run_ikr_stage1(question, max_tokens)
        if not stage1_result.success or stage1_result.output is None:
            return GenerationResult(
                program=None,
                raw_response=stage1_result.raw_response,
                success=False,
                backend="ikr",
                error=f"Stage 1 failed: {stage1_result.error}",
                stage1_response=stage1_result.raw_response,
                two_stage=True,
            )

        stage1_ikr = stage1_result.output

        # Stage 2: Generate background knowledge
        stage2_result = self._run_ikr_stage2(stage1_ikr, max_tokens)
        if not stage2_result.success or stage2_result.output is None:
            # Return partial IKR from Stage 1 with empty rules
            logger.warning("Stage 2 failed, returning Stage 1 IKR without background knowledge")
            stage1_ikr.setdefault("rules", [])
            return GenerationResult(
                program=stage1_ikr,
                raw_response=stage1_result.raw_response,
                success=True,  # Partial success
                backend="ikr",
                error=f"Stage 2 failed (partial result): {stage2_result.error}",
                stage1_response=stage1_result.raw_response,
                stage2_response=stage2_result.raw_response,
                two_stage=True,
            )

        # Merge Stage 1 and Stage 2 outputs
        merged_ikr = self._merge_ikr_stages(stage1_ikr, stage2_result.output)

        return GenerationResult(
            program=merged_ikr,
            raw_response=f"Stage 1:\n{stage1_result.raw_response}\n\nStage 2:\n{stage2_result.raw_response}",
            success=True,
            backend="ikr",
            stage1_response=stage1_result.raw_response,
            stage2_response=stage2_result.raw_response,
            two_stage=True,
        )

    def _run_ikr_stage1(
        self, question: str, max_tokens: int
    ) -> IKRStageResult:
        """Run Stage 1: Extract explicit knowledge from question.

        Args:
            question: Natural language question
            max_tokens: Maximum tokens for response

        Returns:
            IKRStageResult with extracted explicit IKR
        """
        try:
            prompt = build_ikr_stage1_prompt(question)

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
            )

            raw_response = response.choices[0].message.content
            output = self._extract_json(raw_response)

            if output is None:
                return IKRStageResult(
                    output=None,
                    raw_response=raw_response,
                    success=False,
                    error="Failed to extract JSON from Stage 1 response",
                )

            # Validate Stage 1 output has required fields
            required_fields = ["meta", "types", "entities", "relations", "facts", "query"]
            missing = [f for f in required_fields if f not in output]
            if missing:
                return IKRStageResult(
                    output=output,
                    raw_response=raw_response,
                    success=False,
                    error=f"Stage 1 missing required fields: {missing}",
                )

            return IKRStageResult(
                output=output,
                raw_response=raw_response,
                success=True,
            )

        except Exception as e:
            logger.error(f"Stage 1 error: {e}")
            return IKRStageResult(
                output=None,
                raw_response="",
                success=False,
                error=str(e),
            )

    def _run_ikr_stage2(
        self, stage1_ikr: dict[str, Any], max_tokens: int
    ) -> IKRStageResult:
        """Run Stage 2: Generate background knowledge given Stage 1 output.

        Args:
            stage1_ikr: IKR from Stage 1
            max_tokens: Maximum tokens for response

        Returns:
            IKRStageResult with background facts and rules
        """
        try:
            # Format Stage 1 IKR for the prompt
            current_ikr_json = json.dumps(stage1_ikr, indent=2)
            prompt = build_ikr_stage2_prompt(current_ikr_json)

            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
            )

            raw_response = response.choices[0].message.content
            output = self._extract_json(raw_response)

            if output is None:
                return IKRStageResult(
                    output=None,
                    raw_response=raw_response,
                    success=False,
                    error="Failed to extract JSON from Stage 2 response",
                )

            # Stage 2 should have background_facts and/or rules
            if "background_facts" not in output and "rules" not in output:
                return IKRStageResult(
                    output=output,
                    raw_response=raw_response,
                    success=False,
                    error="Stage 2 missing both background_facts and rules",
                )

            return IKRStageResult(
                output=output,
                raw_response=raw_response,
                success=True,
            )

        except Exception as e:
            logger.error(f"Stage 2 error: {e}")
            return IKRStageResult(
                output=None,
                raw_response="",
                success=False,
                error=str(e),
            )

    def _merge_ikr_stages(
        self,
        stage1: dict[str, Any],
        stage2: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge Stage 1 and Stage 2 outputs into complete IKR.

        Args:
            stage1: Explicit IKR (meta, types, entities, relations, explicit facts, query)
            stage2: Background knowledge (background_facts, rules)

        Returns:
            Complete merged IKR
        """
        merged = stage1.copy()

        # Add background facts to the facts list
        background_facts = stage2.get("background_facts", [])
        if background_facts:
            merged["facts"] = merged.get("facts", []) + background_facts

        # Add rules (Stage 1 shouldn't have rules, but handle gracefully)
        stage2_rules = stage2.get("rules", [])
        merged["rules"] = merged.get("rules", []) + stage2_rules

        return merged

    def _extract_json(self, markdown_content: str) -> dict[str, Any] | None:
        """Extract JSON from markdown code block.

        Args:
            markdown_content: Markdown text potentially containing JSON

        Returns:
            Parsed JSON dict or None if extraction failed
        """
        # Pattern to match ```json ... ``` code blocks
        json_pattern = r"```json\s*(\{[\s\S]*?\})\s*```"
        match = re.search(json_pattern, markdown_content)

        if match:
            try:
                json_str = match.group(1)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                return None

        # Try to find JSON without code block markers
        try:
            # Look for { ... } pattern
            brace_pattern = r"\{[\s\S]*\}"
            match = re.search(brace_pattern, markdown_content)
            if match:
                return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

        return None

    def _extract_smt2(self, markdown_content: str) -> str | None:
        """Extract SMT2 from markdown code block.

        Args:
            markdown_content: Markdown text potentially containing SMT2

        Returns:
            SMT2 program text or None if extraction failed
        """
        # Pattern to match ```smt2 ... ``` code blocks
        smt2_pattern = r"```smt2\s*([\s\S]*?)\s*```"
        match = re.search(smt2_pattern, markdown_content)

        if match:
            smt2_text = match.group(1).strip()
            if smt2_text:
                return smt2_text
            logger.error("Found empty SMT2 code block")
            return None

        # Try to find SMT2 without code block markers (starts with comment or paren)
        # Look for lines that start with ';' or '('
        lines = markdown_content.split("\n")
        smt2_lines = []
        in_smt2 = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(";") or stripped.startswith("("):
                in_smt2 = True
            if in_smt2:
                smt2_lines.append(line)

        if smt2_lines:
            return "\n".join(smt2_lines).strip()

        logger.error("Could not extract SMT2 from response")
        return None
