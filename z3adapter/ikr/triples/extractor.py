"""Triple extraction from text using LLM.

Extracts semantic triples from natural language text using a language model.
Uses 7 generic predicates following Wikidata's philosophy.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from z3adapter.ikr.schema import TruthValue
from z3adapter.ikr.triples.schema import Predicate, Triple

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Protocol for OpenAI-compatible LLM client."""

    @property
    def chat(self) -> Any:
        """Chat completions interface."""
        ...


@dataclass
class ExtractionResult:
    """Result of triple extraction from text.

    Attributes:
        triples: List of extracted triples
        raw_response: Raw LLM response for debugging
        source: Optional source/provenance for the text
    """

    triples: list[Triple]
    raw_response: str
    source: Optional[str] = None


SYSTEM_PROMPT = '''You are a knowledge extraction system. Extract semantic triples from the given text.

## Predicates

Use ONLY these 7 predicates:
- is_a: X is a type/kind of Y (taxonomy, classification)
- part_of: X is part of Y (structure, composition)
- has: X has property/attribute Y
- causes: X causes/leads to/results in Y
- prevents: X prevents/stops/inhibits Y
- believes: X believes/claims/states Y (for attributed statements)
- related_to: X is related to Y (use only when others don't fit)

## Output Format

Return a JSON object with a "triples" array:
```json
{
  "triples": [
    {
      "id": "t1",
      "subject": "entity_name",
      "predicate": "predicate_name",
      "object": "entity_name",
      "negated": false,
      "surface_form": "original text snippet"
    }
  ]
}
```

## Rules

1. **Entity naming**: Use lowercase with underscores (working_memory, not "Working Memory")
2. **Triple references**: For beliefs about beliefs, reference other triples with "t:" prefix:
   - First: {"id": "t1", "subject": "stress", "predicate": "causes", "object": "anxiety"}
   - Then: {"id": "t2", "subject": "alice", "predicate": "believes", "object": "t:t1"}
3. **Negation**: Set negated=true for negative statements ("X does NOT cause Y")
4. **Specificity**: Prefer specific predicates over related_to
5. **Completeness**: Extract ALL meaningful relationships, not just the main one
6. **Surface form**: Include the original text snippet that supports the triple

## Examples

Input: "Chronic stress leads to elevated cortisol levels, which can impair memory formation."

Output:
```json
{
  "triples": [
    {"id": "t1", "subject": "chronic_stress", "predicate": "causes", "object": "elevated_cortisol", "negated": false, "surface_form": "Chronic stress leads to elevated cortisol levels"},
    {"id": "t2", "subject": "elevated_cortisol", "predicate": "causes", "object": "memory_impairment", "negated": false, "surface_form": "elevated cortisol levels, which can impair memory formation"}
  ]
}
```

Input: "Dr. Smith argues that meditation does not reduce anxiety in all cases."

Output:
```json
{
  "triples": [
    {"id": "t1", "subject": "meditation", "predicate": "prevents", "object": "anxiety", "negated": true, "surface_form": "meditation does not reduce anxiety in all cases"},
    {"id": "t2", "subject": "dr_smith", "predicate": "believes", "object": "t:t1", "negated": false, "surface_form": "Dr. Smith argues that"}
  ]
}
```
'''


class TripleExtractor:
    """Extract semantic triples from text using LLM.

    Uses an OpenAI-compatible LLM client to extract structured knowledge
    from natural language text.

    Example:
        from openai import OpenAI
        client = OpenAI()
        extractor = TripleExtractor(client, model="gpt-4o")

        result = extractor.extract(
            "Stress causes anxiety and depression.",
            source="Psychology 101 textbook"
        )

        for triple in result.triples:
            print(f"{triple.subject} {triple.predicate} {triple.object}")
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "gpt-4o",
        max_tokens: int = 2000,
    ):
        """Initialize extractor.

        Args:
            llm_client: OpenAI-compatible LLM client
            model: Model name to use
            max_tokens: Maximum tokens for response
        """
        self.client = llm_client
        self.model = model
        self.max_tokens = max_tokens

    def extract(
        self,
        text: str,
        source: Optional[str] = None,
        context: Optional[str] = None,
    ) -> ExtractionResult:
        """Extract triples from text.

        Args:
            text: Text to extract triples from
            source: Optional source/provenance (e.g., "Smith 2020 p.42")
            context: Optional context to help extraction (e.g., domain info)

        Returns:
            ExtractionResult with extracted triples and raw response
        """
        user_prompt = f"Extract triples from the following text:\n\n{text}"
        if context:
            user_prompt = f"Context: {context}\n\n{user_prompt}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=self.max_tokens,
            )
            raw_response = response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return ExtractionResult(triples=[], raw_response=str(e), source=source)

        # Parse response
        triples = self._parse_response(raw_response, source)
        return ExtractionResult(triples=triples, raw_response=raw_response, source=source)

    def extract_batch(
        self,
        texts: list[str],
        source: Optional[str] = None,
    ) -> list[ExtractionResult]:
        """Extract triples from multiple texts.

        Args:
            texts: List of texts to extract from
            source: Optional shared source for all texts

        Returns:
            List of ExtractionResults, one per input text
        """
        results = []
        for i, text in enumerate(texts):
            text_source = f"{source} (chunk {i + 1})" if source else None
            result = self.extract(text, source=text_source)
            results.append(result)
        return results

    def _parse_response(
        self,
        raw_response: str,
        source: Optional[str] = None,
    ) -> list[Triple]:
        """Parse LLM response into Triple objects.

        Args:
            raw_response: Raw response from LLM
            source: Source to attach to triples

        Returns:
            List of parsed Triple objects
        """
        json_data = self._extract_json(raw_response)
        if json_data is None:
            logger.warning("Failed to extract JSON from response")
            return []

        triples = []
        raw_triples = json_data.get("triples", [])

        for raw in raw_triples:
            try:
                triple = self._parse_triple(raw, source)
                if triple:
                    triples.append(triple)
            except Exception as e:
                logger.warning(f"Failed to parse triple {raw}: {e}")
                continue

        return triples

    def _parse_triple(
        self,
        raw: dict[str, Any],
        source: Optional[str] = None,
    ) -> Optional[Triple]:
        """Parse a single triple from JSON dict.

        Args:
            raw: Raw triple dict from JSON
            source: Source to attach

        Returns:
            Parsed Triple or None if invalid
        """
        # Required fields
        triple_id = raw.get("id")
        subject = raw.get("subject")
        predicate_str = raw.get("predicate")
        obj = raw.get("object")

        if not all([triple_id, subject, predicate_str, obj]):
            logger.warning(f"Missing required field in triple: {raw}")
            return None

        # Parse predicate
        try:
            predicate = Predicate(predicate_str)
        except ValueError:
            logger.warning(f"Invalid predicate '{predicate_str}', using related_to")
            predicate = Predicate.RELATED_TO

        # Optional fields
        negated = raw.get("negated", False)
        surface_form = raw.get("surface_form")

        # Parse truth value if present
        truth = None
        if "truth" in raw and raw["truth"]:
            truth_data = raw["truth"]
            truth = TruthValue(
                frequency=truth_data.get("frequency", 1.0),
                confidence=truth_data.get("confidence", 0.9),
            )

        return Triple(
            id=triple_id,
            subject=self._normalize_entity(subject),
            predicate=predicate,
            object=self._normalize_entity(obj),
            negated=negated,
            truth=truth,
            source=source,
            surface_form=surface_form,
        )

    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity name.

        Converts to lowercase, replaces spaces/hyphens with underscores.
        Does NOT normalize triple references (t:xxx).

        Args:
            entity: Raw entity string

        Returns:
            Normalized entity string
        """
        if entity.startswith("t:"):
            return entity  # Don't normalize triple references
        return entity.lower().strip().replace(" ", "_").replace("-", "_")

    def _extract_json(self, text: str) -> Optional[dict[str, Any]]:
        """Extract JSON from text, handling markdown code blocks.

        Args:
            text: Text potentially containing JSON

        Returns:
            Parsed JSON dict or None
        """
        # Try markdown code block first
        json_pattern = r"```json\s*(\{[\s\S]*?\})\s*```"
        match = re.search(json_pattern, text)

        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON from code block: {e}")

        # Try raw JSON
        try:
            brace_pattern = r"\{[\s\S]*\}"
            match = re.search(brace_pattern, text)
            if match:
                return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

        return None


def extract_triples(
    text: str,
    llm_client: LLMClient,
    model: str = "gpt-4o",
    source: Optional[str] = None,
) -> list[Triple]:
    """Convenience function to extract triples from text.

    Args:
        text: Text to extract from
        llm_client: OpenAI-compatible LLM client
        model: Model name
        source: Optional source/provenance

    Returns:
        List of extracted triples
    """
    extractor = TripleExtractor(llm_client, model=model)
    result = extractor.extract(text, source=source)
    return result.triples
