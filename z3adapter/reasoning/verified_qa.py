"""VerifiedQA: Two-stage LLM reasoning with formal verification.

Flow:
1. LLM answers question naturally (chain-of-thought)
2. SMT2 generated from BOTH question AND answer
3. Z3 checks for contradictions between facts and claimed answer
"""

import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class VerifiedQAResult:
    """Result of verified Q&A pipeline."""

    question: str
    llm_answer: str  # Natural language answer from LLM
    llm_verdict: bool | None  # LLM's yes/no answer (extracted)
    verified: bool  # Whether verification succeeded
    contradiction_found: bool  # True if Z3 found contradiction
    final_answer: bool | None  # Final answer after verification
    smt2_program: str | None  # Generated SMT2 for inspection
    z3_output: str  # Raw Z3 output
    error: str | None = None


class VerifiedQA:
    """Two-stage verified Q&A: LLM answers, then Z3 verifies.

    Example:
        >>> client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        >>> vqa = VerifiedQA(llm_client=client, model="qwen2.5-coder:32b")
        >>> result = vqa.query("If all birds fly and penguins are birds, can penguins fly?")
        >>> print(f"LLM said: {result.llm_answer}")
        >>> print(f"Contradiction found: {result.contradiction_found}")
        >>> print(f"Final answer: {result.final_answer}")
    """

    def __init__(
        self,
        llm_client: Any,
        model: str = "gpt-4o",
        cache_dir: str | None = None,
        z3_path: str = "z3",
    ) -> None:
        """Initialize VerifiedQA.

        Args:
            llm_client: LLM client (OpenAI-compatible)
            model: Model name
            cache_dir: Directory for caching SMT2 programs
            z3_path: Path to Z3 executable
        """
        self.llm_client = llm_client
        self.model = model
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self.z3_path = z3_path

        os.makedirs(self.cache_dir, exist_ok=True)

    def query(
        self,
        question: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> VerifiedQAResult:
        """Run verified Q&A pipeline.

        Args:
            question: Natural language question
            temperature: LLM temperature
            max_tokens: Max tokens for LLM response

        Returns:
            VerifiedQAResult with answer and verification status
        """
        logger.info(f"VerifiedQA: Processing question: {question}")

        # Step 1: Get natural language answer from LLM
        logger.info("Step 1: Getting natural language answer from LLM")
        llm_answer, llm_verdict = self._get_llm_answer(
            question=question,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info(f"LLM answer: {llm_answer[:100]}...")
        logger.info(f"LLM verdict: {llm_verdict}")

        if llm_verdict is None:
            return VerifiedQAResult(
                question=question,
                llm_answer=llm_answer,
                llm_verdict=None,
                verified=False,
                contradiction_found=False,
                final_answer=None,
                smt2_program=None,
                z3_output="",
                error="Could not extract yes/no answer from LLM response",
            )

        # Step 2: Generate SMT2 from question + answer
        logger.info("Step 2: Generating SMT2 from question + answer")
        smt2_program = self._generate_verification_smt2(
            question=question,
            llm_answer=llm_answer,
            llm_verdict=llm_verdict,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if smt2_program is None:
            return VerifiedQAResult(
                question=question,
                llm_answer=llm_answer,
                llm_verdict=llm_verdict,
                verified=False,
                contradiction_found=False,
                final_answer=llm_verdict,  # Trust LLM if verification fails
                smt2_program=None,
                z3_output="",
                error="Failed to generate SMT2 verification program",
            )

        logger.info(f"Generated SMT2 program:\n{smt2_program[:500]}...")

        # Step 3: Run Z3 and check for contradictions
        logger.info("Step 3: Running Z3 contradiction check")
        z3_output, contradiction_found = self._run_z3_check(smt2_program)
        logger.info(f"Z3 output: {z3_output}")
        logger.info(f"Contradiction found: {contradiction_found}")

        # Determine final answer
        if contradiction_found:
            # LLM's answer contradicts the facts - flip it
            final_answer = not llm_verdict
            logger.info(f"Contradiction detected! Flipping answer from {llm_verdict} to {final_answer}")
        else:
            # No contradiction - trust LLM's answer
            final_answer = llm_verdict

        return VerifiedQAResult(
            question=question,
            llm_answer=llm_answer,
            llm_verdict=llm_verdict,
            verified=True,
            contradiction_found=contradiction_found,
            final_answer=final_answer,
            smt2_program=smt2_program,
            z3_output=z3_output,
        )

    def _get_llm_answer(
        self,
        question: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, bool | None]:
        """Step 1: Get natural language answer from LLM.

        Returns:
            Tuple of (full_answer, extracted_verdict)
        """
        prompt = f"""Answer the following question. Think step by step, then provide your final answer.

Question: {question}

Please reason through this step by step, then conclude with your answer.
At the end, clearly state "ANSWER: YES" or "ANSWER: NO" (or "ANSWER: TRUE" / "ANSWER: FALSE").
"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
            )
            answer = response.choices[0].message.content or ""

            # Extract yes/no verdict
            verdict = self._extract_verdict(answer)

            return answer, verdict

        except Exception as e:
            logger.error(f"Error getting LLM answer: {e}")
            return f"Error: {e}", None

    def _extract_verdict(self, answer: str) -> bool | None:
        """Extract yes/no verdict from LLM answer."""
        answer_lower = answer.lower()

        # Look for explicit ANSWER: markers first
        if "answer: yes" in answer_lower or "answer: true" in answer_lower:
            return True
        if "answer: no" in answer_lower or "answer: false" in answer_lower:
            return False

        # Look for conclusion patterns
        if "the answer is yes" in answer_lower or "the answer is true" in answer_lower:
            return True
        if "the answer is no" in answer_lower or "the answer is false" in answer_lower:
            return False

        # Last resort: check last sentence
        lines = [l.strip() for l in answer.split('\n') if l.strip()]
        if lines:
            last_line = lines[-1].lower()
            if "yes" in last_line and "no" not in last_line:
                return True
            if "no" in last_line and "yes" not in last_line:
                return False

        return None

    def _generate_verification_smt2(
        self,
        question: str,
        llm_answer: str,
        llm_verdict: bool,
        temperature: float,
        max_tokens: int,
    ) -> str | None:
        """Step 2: Generate SMT2 that encodes question facts AND claimed answer.

        The SMT2 program should:
        1. Encode the facts/premises from the question
        2. Encode the LLM's claimed answer as an assertion
        3. Check if these are consistent (SAT) or contradictory (UNSAT)
        """
        verdict_str = "TRUE" if llm_verdict else "FALSE"

        prompt = f"""You are a formal verification expert. Your task is to check if an LLM's answer contradicts the facts in a question.

QUESTION: {question}

LLM'S REASONING: {llm_answer}

LLM'S ANSWER: {verdict_str}

Generate an SMT-LIB 2.0 program that:
1. Encodes all facts/premises from the question as assertions
2. Encodes the LLM's claimed answer ({verdict_str}) as an assertion
3. Checks for satisfiability

IMPORTANT LOGIC:
- If the facts PLUS the claimed answer are SATISFIABLE (sat), there is NO contradiction
- If the facts PLUS the claimed answer are UNSATISFIABLE (unsat), there IS a contradiction

Structure your SMT2 program as:
1. Declare sorts for entities mentioned
2. Declare predicates/functions for properties
3. Assert the facts from the question
4. Assert the LLM's claimed answer
5. (check-sat)

Example for "If all birds fly and Tweety is a bird, can Tweety fly?" with answer TRUE:
```smt2
; Sorts
(declare-sort Bird 0)

; Predicates
(declare-fun can-fly (Bird) Bool)

; Constants
(declare-const tweety Bird)

; Facts from question
(assert (forall ((b Bird)) (can-fly b)))  ; all birds fly

; LLM's claimed answer: TRUE (Tweety can fly)
(assert (can-fly tweety))

; Check consistency
(check-sat)
; If sat: no contradiction, answer is consistent
; If unsat: contradiction, answer conflicts with facts
```

Now generate the SMT2 program for the given question and answer.
Output ONLY the SMT2 code wrapped in ```smt2 ... ``` markers.

```smt2
"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
            )
            raw_response = response.choices[0].message.content or ""

            # Extract SMT2 from response
            smt2 = self._extract_smt2(raw_response)
            return smt2

        except Exception as e:
            logger.error(f"Error generating verification SMT2: {e}")
            return None

    def _extract_smt2(self, response: str) -> str | None:
        """Extract SMT2 code from LLM response."""
        import re

        # Try to find ```smt2 ... ``` block
        pattern = r"```smt2\s*([\s\S]*?)\s*```"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()

        # Try ``` ... ``` block
        pattern = r"```\s*([\s\S]*?)\s*```"
        match = re.search(pattern, response)
        if match:
            code = match.group(1).strip()
            # Check if it looks like SMT2
            if code.startswith(";") or code.startswith("("):
                return code

        # Try to find SMT2 without markers
        lines = response.split("\n")
        smt2_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(";") or stripped.startswith("("):
                smt2_lines.append(line)

        if smt2_lines:
            return "\n".join(smt2_lines).strip()

        return None

    def _run_z3_check(self, smt2_program: str) -> tuple[str, bool]:
        """Step 3: Run Z3 and check for contradictions.

        Returns:
            Tuple of (z3_output, contradiction_found)
            - contradiction_found is True if Z3 returns UNSAT
        """
        import subprocess
        import tempfile

        # Write SMT2 to temp file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".smt2",
            dir=self.cache_dir,
            delete=False,
        ) as f:
            f.write(smt2_program)
            smt2_path = f.name

        try:
            # Run Z3
            result = subprocess.run(
                [self.z3_path, smt2_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            output = result.stdout.strip()
            stderr = result.stderr.strip()

            if stderr:
                logger.warning(f"Z3 stderr: {stderr}")

            # Check for UNSAT (contradiction)
            # UNSAT means facts + claimed answer are inconsistent
            contradiction_found = "unsat" in output.lower()

            full_output = output
            if stderr:
                full_output += f"\nSTDERR: {stderr}"

            return full_output, contradiction_found

        except subprocess.TimeoutExpired:
            logger.error("Z3 timed out")
            return "TIMEOUT", False
        except FileNotFoundError:
            logger.error(f"Z3 not found at {self.z3_path}")
            return f"ERROR: Z3 not found at {self.z3_path}", False
        except Exception as e:
            logger.error(f"Error running Z3: {e}")
            return f"ERROR: {e}", False
        finally:
            # Clean up temp file
            try:
                os.unlink(smt2_path)
            except Exception:
                pass
