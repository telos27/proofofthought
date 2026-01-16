"""End-to-end triple extraction pipeline.

This module provides the ExtractionPipeline class that orchestrates the full
text-to-verified-triples workflow:
1. Extract triples from text using LLM
2. Resolve entities to canonical forms
3. Store triples for later querying
4. Verify claims against stored knowledge

Example:
    from openai import OpenAI
    from z3adapter.ikr.triples import ExtractionPipeline

    # Create pipeline with LLM client
    client = OpenAI()
    pipeline = ExtractionPipeline.create(client, model="gpt-4o")

    # Ingest knowledge from text
    pipeline.ingest(
        "Chronic stress causes elevated cortisol levels.",
        source="Psychology 101"
    )

    # Query the knowledge base
    result = pipeline.query("Does stress cause cortisol?")
    print(result.verdict)  # VerificationVerdict.SUPPORTED
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Union

from z3adapter.ikr.fuzzy_nars import (
    VerificationResult,
    VerificationVerdict,
    combined_lexical_similarity,
)
from z3adapter.ikr.triples.schema import Predicate, Triple, TripleStore
from z3adapter.ikr.triples.extractor import (
    ExtractionResult,
    LLMClient,
    TripleExtractor,
)
from z3adapter.ikr.triples.entity_resolver import EntityMatch, EntityResolver
from z3adapter.ikr.triples.verification import (
    verify_triple_against_store,
    verify_triples_against_store,
)
from z3adapter.ikr.triples.storage import SQLiteTripleStorage


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class IngestResult:
    """Result of ingesting text into the pipeline.

    Attributes:
        triples: List of triples that were stored
        extraction_result: Raw extraction result from LLM
        entities_resolved: Number of unique entities resolved
        new_entities: Number of new entities added
    """

    triples: list[Triple]
    extraction_result: ExtractionResult
    entities_resolved: int = 0
    new_entities: int = 0

    @property
    def triple_count(self) -> int:
        """Number of triples stored."""
        return len(self.triples)


@dataclass
class QueryResult:
    """Result of querying the knowledge base.

    Attributes:
        question: The original question
        query_triple: Triple extracted from the question
        verification: Verification result from fuzzy-NARS
        extraction_result: Raw extraction result from LLM
    """

    question: str
    query_triple: Optional[Triple]
    verification: Optional[VerificationResult]
    extraction_result: ExtractionResult

    @property
    def verdict(self) -> Optional[VerificationVerdict]:
        """Get verification verdict."""
        return self.verification.verdict if self.verification else None

    @property
    def is_supported(self) -> bool:
        """Check if query is supported by knowledge base."""
        return self.verdict == VerificationVerdict.SUPPORTED

    @property
    def is_contradicted(self) -> bool:
        """Check if query contradicts knowledge base."""
        return self.verdict == VerificationVerdict.CONTRADICTED


# =============================================================================
# Pipeline
# =============================================================================


@dataclass
class ExtractionPipeline:
    """End-to-end pipeline for text → verified triples.

    Orchestrates:
    - Triple extraction from text via LLM
    - Entity resolution to canonical forms
    - Storage (in-memory or SQLite)
    - Verification of claims against stored knowledge

    Attributes:
        extractor: LLM-based triple extractor
        resolver: Entity resolver for fuzzy matching
        store: In-memory triple store
        storage: Optional SQLite storage (if using persistence)

    Example:
        # Create with factory method
        pipeline = ExtractionPipeline.create(llm_client, model="gpt-4o")

        # Or with SQLite persistence
        pipeline = ExtractionPipeline.create(
            llm_client,
            model="gpt-4o",
            db_path="knowledge.db"
        )

        # Ingest text
        result = pipeline.ingest("Stress causes anxiety.", source="Book")
        print(f"Stored {result.triple_count} triples")

        # Query
        result = pipeline.query("Does stress cause worry?")
        print(result.verdict)  # SUPPORTED (fuzzy match)
    """

    extractor: TripleExtractor
    resolver: EntityResolver
    store: TripleStore = field(default_factory=TripleStore)
    storage: Optional[SQLiteTripleStorage] = None

    # Configuration
    resolve_entities: bool = True
    auto_persist: bool = True  # Automatically sync to SQLite storage
    sim_fn: Callable[[str, str], float] = field(
        default=combined_lexical_similarity, repr=False
    )

    # ID generation
    _next_triple_id: int = field(default=1, repr=False)

    @classmethod
    def create(
        cls,
        llm_client: LLMClient,
        model: str = "gpt-4o",
        db_path: Optional[str] = None,
        entity_threshold: float = 0.8,
        resolve_entities: bool = True,
    ) -> "ExtractionPipeline":
        """Factory method to create a pipeline with lexical similarity.

        Args:
            llm_client: OpenAI-compatible LLM client
            model: Model name for extraction
            db_path: Optional path to SQLite database for persistence
            entity_threshold: Similarity threshold for entity resolution
            resolve_entities: Whether to resolve entities (default True)

        Returns:
            Configured ExtractionPipeline
        """
        extractor = TripleExtractor(llm_client, model=model)
        resolver = EntityResolver(threshold=entity_threshold)
        store = TripleStore()
        storage = SQLiteTripleStorage(db_path) if db_path else None

        return cls(
            extractor=extractor,
            resolver=resolver,
            store=store,
            storage=storage,
            resolve_entities=resolve_entities,
            auto_persist=db_path is not None,
        )

    @classmethod
    def create_with_embeddings(
        cls,
        llm_client: LLMClient,
        model: str = "gpt-4o",
        db_path: Optional[str] = None,
        entity_threshold: float = 0.7,
        resolve_entities: bool = True,
        use_mock_embeddings: bool = False,
        use_hybrid: bool = False,
        lexical_weight: float = 0.3,
    ) -> "ExtractionPipeline":
        """Factory method to create a pipeline with embedding-based similarity.

        Creates a pipeline that uses semantic embeddings for entity resolution,
        providing better matching for synonyms and related terms.

        Args:
            llm_client: OpenAI-compatible LLM client
            model: Model name for extraction
            db_path: Optional path to SQLite database for persistence
            entity_threshold: Similarity threshold (default: 0.7, lower than lexical)
            resolve_entities: Whether to resolve entities (default True)
            use_mock_embeddings: If True, use mock embeddings for testing
            use_hybrid: If True, use hybrid (lexical + embedding) similarity
            lexical_weight: Weight for lexical component when use_hybrid=True

        Returns:
            Configured ExtractionPipeline with embedding-based resolution

        Example:
            # Production (requires OPENAI_API_KEY)
            pipeline = ExtractionPipeline.create_with_embeddings(
                client, model="gpt-4o"
            )

            # Testing (no API calls)
            pipeline = ExtractionPipeline.create_with_embeddings(
                client, model="gpt-4o", use_mock_embeddings=True
            )

            # Hybrid similarity (best of both worlds)
            pipeline = ExtractionPipeline.create_with_embeddings(
                client, model="gpt-4o", use_hybrid=True
            )
        """
        from z3adapter.ikr.triples.embeddings import (
            MockEmbedding,
            OpenAIEmbedding,
            make_embedding_similarity,
            make_hybrid_similarity,
        )

        extractor = TripleExtractor(llm_client, model=model)
        store = TripleStore()
        storage = SQLiteTripleStorage(db_path) if db_path else None

        # Create embedding backend
        backend = MockEmbedding() if use_mock_embeddings else OpenAIEmbedding()

        # Create resolver with embedding similarity
        if use_hybrid:
            resolver = EntityResolver.with_hybrid_similarity(
                backend=backend,
                threshold=entity_threshold,
                lexical_weight=lexical_weight,
            )
            sim_fn = make_hybrid_similarity(backend, lexical_weight=lexical_weight)
        else:
            resolver = EntityResolver.with_embeddings(
                backend=backend,
                threshold=entity_threshold,
            )
            sim_fn = make_embedding_similarity(backend)

        return cls(
            extractor=extractor,
            resolver=resolver,
            store=store,
            storage=storage,
            resolve_entities=resolve_entities,
            auto_persist=db_path is not None,
            sim_fn=sim_fn,
        )

    # =========================================================================
    # Core Operations
    # =========================================================================

    def ingest(
        self,
        text: str,
        source: Optional[str] = None,
        context: Optional[str] = None,
    ) -> IngestResult:
        """Extract, resolve, and store triples from text.

        Args:
            text: Text to extract triples from
            source: Optional source/provenance
            context: Optional context for extraction

        Returns:
            IngestResult with stored triples and statistics
        """
        # 1. Extract triples from text
        extraction_result = self.extractor.extract(text, source=source, context=context)

        # 2. Process and store triples
        stored_triples = []
        entities_resolved = set()
        new_entities = 0

        for triple in extraction_result.triples:
            # Reassign ID to ensure uniqueness across ingestions
            new_id = self._generate_id()

            # Resolve entities if enabled
            if self.resolve_entities:
                subject_match = self._resolve_entity(triple.subject)
                object_match = self._resolve_entity(triple.object)

                resolved_subject = subject_match.canonical if subject_match else triple.subject
                resolved_object = object_match.canonical if object_match else triple.object

                entities_resolved.add(resolved_subject)
                entities_resolved.add(resolved_object)

                if subject_match and subject_match.is_new:
                    new_entities += 1
                if object_match and object_match.is_new:
                    new_entities += 1
            else:
                resolved_subject = triple.subject
                resolved_object = triple.object

            # Create resolved triple
            resolved_triple = Triple(
                id=new_id,
                subject=resolved_subject,
                predicate=triple.predicate,
                object=resolved_object,
                negated=triple.negated,
                truth=triple.truth,
                source=triple.source,
                surface_form=triple.surface_form,
            )

            # Store in memory
            self.store.add(resolved_triple)
            stored_triples.append(resolved_triple)

        # 3. Persist to SQLite if enabled
        if self.auto_persist and self.storage:
            self.storage.add_triples(stored_triples)

        return IngestResult(
            triples=stored_triples,
            extraction_result=extraction_result,
            entities_resolved=len(entities_resolved),
            new_entities=new_entities,
        )

    def ingest_batch(
        self,
        texts: list[str],
        source: Optional[str] = None,
    ) -> list[IngestResult]:
        """Ingest multiple texts.

        Args:
            texts: List of texts to ingest
            source: Optional shared source

        Returns:
            List of IngestResults, one per text
        """
        results = []
        for i, text in enumerate(texts):
            text_source = f"{source} (chunk {i + 1})" if source else None
            result = self.ingest(text, source=text_source)
            results.append(result)
        return results

    def query(
        self,
        question: str,
        match_threshold: float = 0.5,
        support_threshold: float = 0.5,
    ) -> QueryResult:
        """Query the knowledge base with a question.

        Extracts a query triple from the question and verifies it
        against the stored knowledge using fuzzy matching.

        Args:
            question: Natural language question
            match_threshold: Minimum similarity for matching
            support_threshold: Frequency threshold for support

        Returns:
            QueryResult with verification verdict
        """
        # 1. Extract query triple from question
        extraction_result = self.extractor.extract(question)

        if not extraction_result.triples:
            return QueryResult(
                question=question,
                query_triple=None,
                verification=None,
                extraction_result=extraction_result,
            )

        # 2. Get the first/main triple as the query
        query_triple = extraction_result.triples[0]

        # 3. Resolve entities if enabled
        if self.resolve_entities:
            subject_match = self._resolve_entity(query_triple.subject, auto_add=False)
            object_match = self._resolve_entity(query_triple.object, auto_add=False)

            query_triple = Triple(
                id=query_triple.id,
                subject=subject_match.canonical if subject_match else query_triple.subject,
                predicate=query_triple.predicate,
                object=object_match.canonical if object_match else query_triple.object,
                negated=query_triple.negated,
                truth=query_triple.truth,
                source=query_triple.source,
                surface_form=query_triple.surface_form,
            )

        # 4. Verify against store
        verification = verify_triple_against_store(
            query_triple,
            self.store,
            sim_fn=self.sim_fn,
            match_threshold=match_threshold,
            support_threshold=support_threshold,
        )

        return QueryResult(
            question=question,
            query_triple=query_triple,
            verification=verification,
            extraction_result=extraction_result,
        )

    def verify(
        self,
        triple: Triple,
        match_threshold: float = 0.5,
        support_threshold: float = 0.5,
    ) -> VerificationResult:
        """Verify a triple against the knowledge base.

        Direct verification without LLM extraction.

        Args:
            triple: Triple to verify
            match_threshold: Minimum similarity for matching
            support_threshold: Frequency threshold for support

        Returns:
            VerificationResult from fuzzy-NARS
        """
        return verify_triple_against_store(
            triple,
            self.store,
            sim_fn=self.sim_fn,
            match_threshold=match_threshold,
            support_threshold=support_threshold,
        )

    # =========================================================================
    # Entity Management
    # =========================================================================

    def add_entity(
        self,
        canonical: str,
        surface_forms: Optional[list[str]] = None,
    ) -> None:
        """Add an entity to the resolver.

        Args:
            canonical: Canonical entity name
            surface_forms: Optional list of aliases
        """
        self.resolver.add_entity(canonical, surface_forms)
        if self.auto_persist and self.storage:
            self.storage.add_entity(canonical, surface_forms)

    def get_entity_forms(self, canonical: str) -> set[str]:
        """Get all surface forms for an entity.

        Args:
            canonical: Canonical entity name

        Returns:
            Set of known surface forms
        """
        return self.resolver.get_surface_forms(canonical)

    def list_entities(self) -> list[str]:
        """List all canonical entity names."""
        return self.resolver.get_all_entities()

    def _resolve_entity(
        self,
        surface_form: str,
        auto_add: bool = True,
    ) -> Optional[EntityMatch]:
        """Resolve an entity, skipping triple references.

        Args:
            surface_form: Surface form to resolve
            auto_add: Whether to auto-add new entities

        Returns:
            EntityMatch or None for triple references
        """
        # Don't resolve triple references
        if Triple.is_triple_reference(surface_form):
            return None

        if auto_add:
            return self.resolver.resolve(surface_form)
        else:
            return self.resolver.resolve_or_none(surface_form)

    # =========================================================================
    # Store Access
    # =========================================================================

    def get_triple(self, triple_id: str) -> Optional[Triple]:
        """Get a triple by ID."""
        return self.store.get(triple_id)

    def query_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[Predicate] = None,
        obj: Optional[str] = None,
    ) -> list[Triple]:
        """Query triples by pattern.

        Args:
            subject: Filter by subject
            predicate: Filter by predicate
            obj: Filter by object

        Returns:
            List of matching triples
        """
        return self.store.query(subject=subject, predicate=predicate, obj=obj)

    def count_triples(self) -> int:
        """Get total number of triples in store."""
        return len(self.store)

    # =========================================================================
    # Persistence
    # =========================================================================

    def save(self) -> None:
        """Save current state to SQLite storage.

        Syncs in-memory store and resolver to persistent storage.
        """
        if not self.storage:
            raise ValueError("No SQLite storage configured")

        self.storage.from_triple_store(self.store, clear_existing=True)
        self.storage.from_entity_resolver(self.resolver, clear_existing=True)

    def load(self) -> None:
        """Load state from SQLite storage.

        Replaces in-memory store and resolver with persistent data.
        """
        if not self.storage:
            raise ValueError("No SQLite storage configured")

        self.store = self.storage.to_triple_store()
        self.resolver = self.storage.to_entity_resolver()

    def clear(self) -> None:
        """Clear all triples and entities from memory and storage."""
        self.store = TripleStore()
        self.resolver.clear()
        self._next_triple_id = 1

        if self.auto_persist and self.storage:
            self.storage.clear_all()

    # =========================================================================
    # Utilities
    # =========================================================================

    def _generate_id(self) -> str:
        """Generate a unique triple ID."""
        triple_id = f"t{self._next_triple_id}"
        self._next_triple_id += 1
        return triple_id

    def __len__(self) -> int:
        """Return number of triples in store."""
        return len(self.store)

    def __contains__(self, triple_id: str) -> bool:
        """Check if triple ID exists in store."""
        return triple_id in self.store
