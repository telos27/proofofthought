"""Triple extraction pipeline for knowledge extraction.

This package provides tools for extracting semantic triples from text,
following Wikidata's philosophy: predicates are fixed schema, entities
emerge from content.

Core Components:
    - Triple: A semantic triple with subject, predicate, object
    - TripleStore: In-memory store with indexing
    - Predicate: Enum of 7 generic predicates
    - TripleExtractor: LLM-based triple extraction
    - ExtractionResult: Result of extraction with triples and raw response

Example:
    from z3adapter.ikr.triples import Triple, TripleStore, Predicate

    store = TripleStore()
    store.add(Triple(
        id="t1",
        subject="stress",
        predicate=Predicate.CAUSES,
        object="anxiety"
    ))

    # Query by predicate
    causal = store.query(predicate=Predicate.CAUSES)

Example (extraction):
    from openai import OpenAI
    from z3adapter.ikr.triples import TripleExtractor

    client = OpenAI()
    extractor = TripleExtractor(client, model="gpt-4o")
    result = extractor.extract("Stress causes anxiety.")

    for triple in result.triples:
        print(f"{triple.subject} {triple.predicate} {triple.object}")
"""

from z3adapter.ikr.triples.schema import (
    Predicate,
    PREDICATE_OPPOSITES,
    Triple,
    TripleStore,
)
from z3adapter.ikr.triples.extractor import (
    ExtractionResult,
    TripleExtractor,
    extract_triples,
)

__all__ = [
    # Schema
    "Predicate",
    "PREDICATE_OPPOSITES",
    "Triple",
    "TripleStore",
    # Extraction
    "ExtractionResult",
    "TripleExtractor",
    "extract_triples",
]
