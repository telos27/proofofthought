"""SQLite persistence for triples and entities.

This module provides persistent storage for the triple extraction pipeline,
storing triples and entities with their surface forms in a SQLite database.

Key features:
- CRUD operations for triples
- Entity storage with surface form tracking
- Indexed queries by subject/predicate/object
- Conversion to/from in-memory TripleStore and EntityResolver

Example:
    from z3adapter.ikr.triples import SQLiteTripleStorage

    # Create database
    storage = SQLiteTripleStorage("knowledge.db")

    # Add triples
    triple = Triple(id="t1", subject="stress", predicate=Predicate.CAUSES, object="anxiety")
    storage.add_triple(triple)

    # Query by predicate
    causal = storage.query_triples(predicate=Predicate.CAUSES)

    # Entity management
    storage.add_entity("working_memory", ["WM", "short-term memory"])
"""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

from z3adapter.ikr.schema import TruthValue
from z3adapter.ikr.triples.schema import Predicate, Triple, TripleStore
from z3adapter.ikr.triples.entity_resolver import EntityMatch, EntityResolver


# =============================================================================
# SQL Schema
# =============================================================================

SCHEMA_SQL = """
-- Entities (Layer 1)
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    canonical_name TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS entity_surface_forms (
    entity_id TEXT NOT NULL,
    surface_form TEXT NOT NULL,
    PRIMARY KEY (entity_id, surface_form),
    FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE
);

-- Triples (Layer 2)
CREATE TABLE IF NOT EXISTS triples (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    negated INTEGER DEFAULT 0,
    frequency REAL,
    confidence REAL,
    source TEXT,
    surface_form TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for fast lookup
CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject);
CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate);
CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object);
CREATE INDEX IF NOT EXISTS idx_entity_surface_forms ON entity_surface_forms(surface_form);
CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_name);
"""


# =============================================================================
# Storage Classes
# =============================================================================


class SQLiteTripleStorage:
    """SQLite-based storage for triples and entities.

    Provides persistent storage with:
    - Full CRUD operations for triples
    - Entity management with surface form tracking
    - Efficient indexed queries
    - Import/export to in-memory stores

    Attributes:
        db_path: Path to SQLite database file (":memory:" for in-memory)

    Example:
        storage = SQLiteTripleStorage("knowledge.db")

        # Add a triple
        triple = Triple(id="t1", subject="cat", predicate=Predicate.IS_A, object="mammal")
        storage.add_triple(triple)

        # Query triples
        mammals = storage.query_triples(predicate=Predicate.IS_A)

        # Export to TripleStore
        store = storage.to_triple_store()
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        """Initialize storage with database path.

        Args:
            db_path: Path to SQLite database file (":memory:" for in-memory)
        """
        self.db_path = db_path
        self._is_memory = str(db_path) == ":memory:"
        self._persistent_conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Create database schema if not exists."""
        with self._connection() as conn:
            conn.executescript(SCHEMA_SQL)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections.

        For in-memory databases, maintains a persistent connection.
        For file databases, opens and closes connections as needed.

        Yields:
            SQLite connection with foreign keys enabled
        """
        if self._is_memory:
            # In-memory databases need a persistent connection
            if self._persistent_conn is None:
                self._persistent_conn = sqlite3.connect(":memory:")
                self._persistent_conn.execute("PRAGMA foreign_keys = ON")
                self._persistent_conn.row_factory = sqlite3.Row
            try:
                yield self._persistent_conn
                self._persistent_conn.commit()
            except Exception:
                self._persistent_conn.rollback()
                raise
        else:
            # File databases can open/close connections
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA foreign_keys = ON")
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # =========================================================================
    # Triple CRUD
    # =========================================================================

    def add_triple(self, triple: Triple) -> None:
        """Add or update a triple.

        If a triple with the same ID exists, it will be replaced.

        Args:
            triple: The triple to add
        """
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO triples
                (id, subject, predicate, object, negated, frequency, confidence, source, surface_form)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    triple.id,
                    triple.subject,
                    triple.predicate.value,
                    triple.object,
                    1 if triple.negated else 0,
                    triple.truth.frequency if triple.truth else None,
                    triple.truth.confidence if triple.truth else None,
                    triple.source,
                    triple.surface_form,
                ),
            )

    def add_triples(self, triples: list[Triple]) -> None:
        """Add multiple triples in a batch.

        More efficient than calling add_triple() repeatedly.

        Args:
            triples: List of triples to add
        """
        with self._connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO triples
                (id, subject, predicate, object, negated, frequency, confidence, source, surface_form)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        t.id,
                        t.subject,
                        t.predicate.value,
                        t.object,
                        1 if t.negated else 0,
                        t.truth.frequency if t.truth else None,
                        t.truth.confidence if t.truth else None,
                        t.source,
                        t.surface_form,
                    )
                    for t in triples
                ],
            )

    def get_triple(self, triple_id: str) -> Optional[Triple]:
        """Get a triple by ID.

        Args:
            triple_id: The triple ID

        Returns:
            Triple if found, None otherwise
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM triples WHERE id = ?",
                (triple_id,),
            ).fetchone()

            if row is None:
                return None
            return self._row_to_triple(row)

    def remove_triple(self, triple_id: str) -> bool:
        """Remove a triple by ID.

        Args:
            triple_id: The triple ID to remove

        Returns:
            True if triple was removed, False if not found
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM triples WHERE id = ?",
                (triple_id,),
            )
            return cursor.rowcount > 0

    def query_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[Predicate] = None,
        obj: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[Triple]:
        """Query triples by subject/predicate/object pattern.

        Args:
            subject: Filter by subject (exact match)
            predicate: Filter by predicate
            obj: Filter by object (exact match)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching triples
        """
        conditions = []
        params: list = []

        if subject is not None:
            conditions.append("subject = ?")
            params.append(subject)
        if predicate is not None:
            conditions.append("predicate = ?")
            params.append(predicate.value)
        if obj is not None:
            conditions.append("object = ?")
            params.append(obj)

        query = "SELECT * FROM triples"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"

        if limit is not None:
            query += f" LIMIT {limit} OFFSET {offset}"

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_triple(row) for row in rows]

    def count_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[Predicate] = None,
        obj: Optional[str] = None,
    ) -> int:
        """Count triples matching a pattern.

        Args:
            subject: Filter by subject (exact match)
            predicate: Filter by predicate
            obj: Filter by object (exact match)

        Returns:
            Number of matching triples
        """
        conditions = []
        params: list = []

        if subject is not None:
            conditions.append("subject = ?")
            params.append(subject)
        if predicate is not None:
            conditions.append("predicate = ?")
            params.append(predicate.value)
        if obj is not None:
            conditions.append("object = ?")
            params.append(obj)

        query = "SELECT COUNT(*) FROM triples"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with self._connection() as conn:
            return conn.execute(query, params).fetchone()[0]

    def _row_to_triple(self, row: sqlite3.Row) -> Triple:
        """Convert a database row to a Triple.

        Args:
            row: SQLite row

        Returns:
            Triple object
        """
        truth = None
        if row["frequency"] is not None and row["confidence"] is not None:
            truth = TruthValue(frequency=row["frequency"], confidence=row["confidence"])

        return Triple(
            id=row["id"],
            subject=row["subject"],
            predicate=Predicate(row["predicate"]),
            object=row["object"],
            negated=bool(row["negated"]),
            truth=truth,
            source=row["source"],
            surface_form=row["surface_form"],
        )

    # =========================================================================
    # Entity CRUD
    # =========================================================================

    def add_entity(
        self,
        canonical: str,
        surface_forms: Optional[list[str]] = None,
        entity_id: Optional[str] = None,
    ) -> str:
        """Add an entity with optional surface forms.

        Args:
            canonical: The canonical entity name
            surface_forms: Optional list of surface forms/aliases
            entity_id: Optional entity ID (auto-generated if not provided)

        Returns:
            The entity ID
        """
        if entity_id is None:
            entity_id = str(uuid.uuid4())

        with self._connection() as conn:
            # Insert entity
            conn.execute(
                """
                INSERT OR IGNORE INTO entities (id, canonical_name)
                VALUES (?, ?)
                """,
                (entity_id, canonical),
            )

            # Get actual entity ID (in case of conflict)
            row = conn.execute(
                "SELECT id FROM entities WHERE canonical_name = ?",
                (canonical,),
            ).fetchone()
            actual_id = row["id"]

            # Add surface forms
            if surface_forms:
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO entity_surface_forms (entity_id, surface_form)
                    VALUES (?, ?)
                    """,
                    [(actual_id, sf) for sf in surface_forms],
                )

            return actual_id

    def get_entity(self, canonical: str) -> Optional[dict]:
        """Get an entity by canonical name.

        Args:
            canonical: The canonical entity name

        Returns:
            Dict with 'id', 'canonical_name', 'surface_forms' if found, None otherwise
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM entities WHERE canonical_name = ?",
                (canonical,),
            ).fetchone()

            if row is None:
                return None

            # Get surface forms
            sf_rows = conn.execute(
                "SELECT surface_form FROM entity_surface_forms WHERE entity_id = ?",
                (row["id"],),
            ).fetchall()

            return {
                "id": row["id"],
                "canonical_name": row["canonical_name"],
                "surface_forms": [r["surface_form"] for r in sf_rows],
            }

    def get_entity_by_id(self, entity_id: str) -> Optional[dict]:
        """Get an entity by ID.

        Args:
            entity_id: The entity ID

        Returns:
            Dict with 'id', 'canonical_name', 'surface_forms' if found, None otherwise
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM entities WHERE id = ?",
                (entity_id,),
            ).fetchone()

            if row is None:
                return None

            # Get surface forms
            sf_rows = conn.execute(
                "SELECT surface_form FROM entity_surface_forms WHERE entity_id = ?",
                (entity_id,),
            ).fetchall()

            return {
                "id": row["id"],
                "canonical_name": row["canonical_name"],
                "surface_forms": [r["surface_form"] for r in sf_rows],
            }

    def remove_entity(self, canonical: str) -> bool:
        """Remove an entity by canonical name.

        Surface forms are automatically deleted due to CASCADE.

        Args:
            canonical: The canonical entity name

        Returns:
            True if entity was removed, False if not found
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM entities WHERE canonical_name = ?",
                (canonical,),
            )
            return cursor.rowcount > 0

    def add_surface_form(self, canonical: str, surface_form: str) -> bool:
        """Add a surface form to an existing entity.

        Args:
            canonical: The canonical entity name
            surface_form: The surface form to add

        Returns:
            True if surface form was added, False if entity not found
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT id FROM entities WHERE canonical_name = ?",
                (canonical,),
            ).fetchone()

            if row is None:
                return False

            conn.execute(
                """
                INSERT OR IGNORE INTO entity_surface_forms (entity_id, surface_form)
                VALUES (?, ?)
                """,
                (row["id"], surface_form),
            )
            return True

    def find_entity_by_surface_form(self, surface_form: str) -> Optional[str]:
        """Find canonical entity name by surface form.

        Args:
            surface_form: The surface form to look up

        Returns:
            Canonical entity name if found, None otherwise
        """
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT e.canonical_name
                FROM entities e
                JOIN entity_surface_forms sf ON e.id = sf.entity_id
                WHERE sf.surface_form = ?
                """,
                (surface_form,),
            ).fetchone()

            if row is None:
                return None
            return row["canonical_name"]

    def list_entities(self, limit: Optional[int] = None, offset: int = 0) -> list[str]:
        """List all canonical entity names.

        Args:
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of canonical entity names
        """
        query = "SELECT canonical_name FROM entities ORDER BY canonical_name"
        if limit is not None:
            query += f" LIMIT {limit} OFFSET {offset}"

        with self._connection() as conn:
            rows = conn.execute(query).fetchall()
            return [row["canonical_name"] for row in rows]

    def count_entities(self) -> int:
        """Count total number of entities.

        Returns:
            Number of entities
        """
        with self._connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]

    # =========================================================================
    # Import/Export
    # =========================================================================

    def to_triple_store(self) -> TripleStore:
        """Export all triples to an in-memory TripleStore.

        Returns:
            TripleStore with all triples
        """
        store = TripleStore()
        for triple in self.query_triples():
            store.add(triple)
        return store

    def from_triple_store(self, store: TripleStore, clear_existing: bool = False) -> int:
        """Import triples from an in-memory TripleStore.

        Args:
            store: The TripleStore to import from
            clear_existing: If True, remove all existing triples first

        Returns:
            Number of triples imported
        """
        if clear_existing:
            self.clear_triples()

        triples = list(store)
        self.add_triples(triples)
        return len(triples)

    def to_entity_resolver(self) -> EntityResolver:
        """Export all entities to an in-memory EntityResolver.

        Returns:
            EntityResolver with all entities and surface forms
        """
        resolver = EntityResolver()

        with self._connection() as conn:
            # Get all entities
            entities = conn.execute("SELECT id, canonical_name FROM entities").fetchall()

            for entity in entities:
                # Get surface forms for this entity
                sf_rows = conn.execute(
                    "SELECT surface_form FROM entity_surface_forms WHERE entity_id = ?",
                    (entity["id"],),
                ).fetchall()
                surface_forms = [r["surface_form"] for r in sf_rows]

                resolver.add_entity(entity["canonical_name"], surface_forms)

        return resolver

    def from_entity_resolver(
        self, resolver: EntityResolver, clear_existing: bool = False
    ) -> int:
        """Import entities from an in-memory EntityResolver.

        Args:
            resolver: The EntityResolver to import from
            clear_existing: If True, remove all existing entities first

        Returns:
            Number of entities imported
        """
        if clear_existing:
            self.clear_entities()

        count = 0
        for canonical in resolver.get_all_entities():
            surface_forms = list(resolver.get_surface_forms(canonical))
            self.add_entity(canonical, surface_forms)
            count += 1

        return count

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def clear_triples(self) -> int:
        """Remove all triples.

        Returns:
            Number of triples removed
        """
        with self._connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM triples").fetchone()[0]
            conn.execute("DELETE FROM triples")
            return count

    def clear_entities(self) -> int:
        """Remove all entities and their surface forms.

        Returns:
            Number of entities removed
        """
        with self._connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            conn.execute("DELETE FROM entities")
            return count

    def clear_all(self) -> tuple[int, int]:
        """Remove all triples and entities.

        Returns:
            Tuple of (triples_removed, entities_removed)
        """
        triples = self.clear_triples()
        entities = self.clear_entities()
        return triples, entities

    def __len__(self) -> int:
        """Return number of triples in storage."""
        return self.count_triples()

    def __contains__(self, triple_id: str) -> bool:
        """Check if triple ID exists in storage."""
        return self.get_triple(triple_id) is not None

    def __bool__(self) -> bool:
        """Storage is always truthy (unlike __len__ which returns triple count)."""
        return True

    def close(self) -> None:
        """Close the database connection (for in-memory databases)."""
        if self._persistent_conn is not None:
            self._persistent_conn.close()
            self._persistent_conn = None

    def __del__(self) -> None:
        """Clean up connection on garbage collection."""
        self.close()
