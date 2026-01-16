"""EntityStore: SQLite-backed storage for entities and links.

Provides CRUD operations for entities, links, surface forms, and embeddings
with efficient indexing for fast lookups.
"""

import json
import sqlite3
import struct
from datetime import datetime
from pathlib import Path
from typing import Optional

from z3adapter.ikr.entities.schema import (
    Entity,
    EntityEmbedding,
    EntityLink,
    LinkType,
    SurfaceForm,
)


class EntityStore:
    """SQLite-backed storage for entities and links.

    Manages entities, entity links, surface forms, and embeddings with
    efficient indexing for fast lookups.

    Example:
        store = EntityStore("knowledge.db")

        # Add entity
        entity = Entity(name="anxiety", entity_type="emotion")
        store.add(entity)

        # Look up by name
        found = store.get_by_name("anxiety")

        # Add link
        link = EntityLink(
            source_id=entity1.id,
            target_id=entity2.id,
            link_type=LinkType.SIMILAR_TO,
            score=0.85
        )
        store.add_link(link)

        # Get similar entities
        similar = store.get_similar(entity.id, min_score=0.5)
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize EntityStore.

        Args:
            db_path: Path to SQLite database. If None, uses in-memory database.
        """
        self.db_path = db_path or ":memory:"
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection, creating if needed."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript(
            """
            -- Entities
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                entity_type TEXT,
                description TEXT,
                external_ids TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

            -- Entity Links (pre-computed similarities)
            CREATE TABLE IF NOT EXISTS entity_links (
                source_id TEXT NOT NULL REFERENCES entities(id),
                target_id TEXT NOT NULL REFERENCES entities(id),
                link_type TEXT NOT NULL,
                score REAL NOT NULL,
                method TEXT,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (source_id, target_id, link_type)
            );

            CREATE INDEX IF NOT EXISTS idx_links_source ON entity_links(source_id);
            CREATE INDEX IF NOT EXISTS idx_links_target ON entity_links(target_id);
            CREATE INDEX IF NOT EXISTS idx_links_score ON entity_links(score);

            -- Entity Embeddings
            CREATE TABLE IF NOT EXISTS entity_embeddings (
                entity_id TEXT PRIMARY KEY REFERENCES entities(id),
                embedding BLOB NOT NULL,
                model TEXT NOT NULL,
                computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Surface Forms (learned mappings for O(1) lookup)
            CREATE TABLE IF NOT EXISTS surface_forms (
                surface_form TEXT PRIMARY KEY,
                entity_id TEXT NOT NULL REFERENCES entities(id),
                score REAL NOT NULL,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_surface_forms_entity ON surface_forms(entity_id);
            """
        )
        conn.commit()

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ==================== Entity CRUD ====================

    def add(self, entity: Entity) -> str:
        """Add entity to store.

        Args:
            entity: Entity to add

        Returns:
            Entity ID

        Raises:
            sqlite3.IntegrityError: If entity with same name already exists
        """
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO entities (id, name, entity_type, description, external_ids, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entity.id,
                entity.name,
                entity.entity_type,
                entity.description,
                json.dumps(entity.external_ids) if entity.external_ids else None,
                entity.source,
                entity.created_at.isoformat(),
            ),
        )
        conn.commit()
        return entity.id

    def get(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID.

        Args:
            entity_id: Entity ID to look up

        Returns:
            Entity if found, None otherwise
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def get_by_name(self, name: str) -> Optional[Entity]:
        """Get entity by canonical name.

        Args:
            name: Entity name (will be normalized)

        Returns:
            Entity if found, None otherwise
        """
        normalized = Entity._normalize_name(name)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM entities WHERE name = ?", (normalized,)
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def get_or_create(self, name: str, **kwargs) -> tuple[Entity, bool]:
        """Get existing entity by name or create new one.

        Args:
            name: Entity name
            **kwargs: Additional entity attributes (entity_type, description, etc.)

        Returns:
            Tuple of (entity, is_new) where is_new indicates if entity was created
        """
        existing = self.get_by_name(name)
        if existing:
            return existing, False

        entity = Entity(name=name, **kwargs)
        self.add(entity)
        return entity, True

    def search(self, pattern: str, limit: int = 10) -> list[Entity]:
        """Search entities by name pattern.

        Args:
            pattern: SQL LIKE pattern (use % for wildcards)
            limit: Maximum number of results

        Returns:
            List of matching entities
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM entities WHERE name LIKE ? LIMIT ?",
            (pattern, limit),
        ).fetchall()
        return [self._row_to_entity(row) for row in rows]

    def count(self) -> int:
        """Return total number of entities."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) FROM entities").fetchone()
        return row[0] if row else 0

    def all_entities(self) -> list[Entity]:
        """Return all entities."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM entities").fetchall()
        return [self._row_to_entity(row) for row in rows]

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        """Convert database row to Entity object."""
        external_ids = json.loads(row["external_ids"]) if row["external_ids"] else {}
        created_at = datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now()
        return Entity(
            id=row["id"],
            name=row["name"],
            entity_type=row["entity_type"],
            description=row["description"],
            external_ids=external_ids,
            source=row["source"],
            created_at=created_at,
        )

    # ==================== Entity Links ====================

    def add_link(self, link: EntityLink) -> None:
        """Add entity link.

        Args:
            link: EntityLink to add

        Note:
            If link already exists (same source, target, type), it will be replaced.
        """
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO entity_links
            (source_id, target_id, link_type, score, method, computed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                link.source_id,
                link.target_id,
                link.link_type.value,
                link.score,
                link.method,
                link.computed_at.isoformat(),
            ),
        )
        conn.commit()

    def get_links(
        self,
        entity_id: str,
        link_type: Optional[LinkType] = None,
        direction: str = "outgoing",
    ) -> list[EntityLink]:
        """Get links for an entity.

        Args:
            entity_id: Entity ID
            link_type: Optional filter by link type
            direction: "outgoing" (from entity), "incoming" (to entity), or "both"

        Returns:
            List of EntityLinks
        """
        conn = self._get_conn()
        links = []

        if direction in ("outgoing", "both"):
            if link_type:
                rows = conn.execute(
                    "SELECT * FROM entity_links WHERE source_id = ? AND link_type = ?",
                    (entity_id, link_type.value),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM entity_links WHERE source_id = ?",
                    (entity_id,),
                ).fetchall()
            links.extend(self._row_to_link(row) for row in rows)

        if direction in ("incoming", "both"):
            if link_type:
                rows = conn.execute(
                    "SELECT * FROM entity_links WHERE target_id = ? AND link_type = ?",
                    (entity_id, link_type.value),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM entity_links WHERE target_id = ?",
                    (entity_id,),
                ).fetchall()
            links.extend(self._row_to_link(row) for row in rows)

        return links

    def get_similar(
        self,
        entity_id: str,
        min_score: float = 0.5,
        limit: int = 10,
    ) -> list[tuple[Entity, float]]:
        """Get entities similar to the given entity.

        Args:
            entity_id: Entity ID to find similar entities for
            min_score: Minimum similarity score threshold
            limit: Maximum number of results

        Returns:
            List of (Entity, score) tuples, sorted by score descending
        """
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT e.*, el.score
            FROM entity_links el
            JOIN entities e ON e.id = el.target_id
            WHERE el.source_id = ?
              AND el.link_type = ?
              AND el.score >= ?
            ORDER BY el.score DESC
            LIMIT ?
            """,
            (entity_id, LinkType.SIMILAR_TO.value, min_score, limit),
        ).fetchall()

        return [(self._row_to_entity(row), row["score"]) for row in rows]

    def _row_to_link(self, row: sqlite3.Row) -> EntityLink:
        """Convert database row to EntityLink object."""
        computed_at = datetime.fromisoformat(row["computed_at"]) if row["computed_at"] else datetime.now()
        return EntityLink(
            source_id=row["source_id"],
            target_id=row["target_id"],
            link_type=LinkType(row["link_type"]),
            score=row["score"],
            method=row["method"],
            computed_at=computed_at,
        )

    # ==================== Surface Forms ====================

    def add_surface_form(
        self,
        form: str,
        entity_id: str,
        score: float,
        source: str = "exact",
    ) -> None:
        """Add surface form mapping.

        Args:
            form: Surface form text (will be normalized)
            entity_id: Entity ID this form maps to
            score: Confidence score [0, 1]
            source: How the mapping was learned
        """
        normalized = form.lower().strip()
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO surface_forms
            (surface_form, entity_id, score, source, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (normalized, entity_id, score, source, datetime.now().isoformat()),
        )
        conn.commit()

    def lookup_surface_form(self, form: str) -> Optional[tuple[str, float]]:
        """Look up entity ID by surface form.

        Args:
            form: Surface form text (will be normalized)

        Returns:
            Tuple of (entity_id, score) if found, None otherwise
        """
        normalized = form.lower().strip()
        conn = self._get_conn()
        row = conn.execute(
            "SELECT entity_id, score FROM surface_forms WHERE surface_form = ?",
            (normalized,),
        ).fetchone()
        return (row["entity_id"], row["score"]) if row else None

    def get_surface_forms(self, entity_id: str) -> list[SurfaceForm]:
        """Get all surface forms for an entity.

        Args:
            entity_id: Entity ID

        Returns:
            List of SurfaceForm objects
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM surface_forms WHERE entity_id = ?",
            (entity_id,),
        ).fetchall()
        return [
            SurfaceForm(
                form=row["surface_form"],
                entity_id=row["entity_id"],
                score=row["score"],
                source=row["source"],
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
            )
            for row in rows
        ]

    # ==================== Embeddings ====================

    def save_embedding(
        self,
        entity_id: str,
        embedding: list[float],
        model: str,
    ) -> None:
        """Save embedding for an entity.

        Args:
            entity_id: Entity ID
            embedding: Embedding vector
            model: Model used to generate embedding
        """
        conn = self._get_conn()
        # Store as binary blob for efficiency
        blob = self._embedding_to_blob(embedding)
        conn.execute(
            """
            INSERT OR REPLACE INTO entity_embeddings
            (entity_id, embedding, model, computed_at)
            VALUES (?, ?, ?, ?)
            """,
            (entity_id, blob, model, datetime.now().isoformat()),
        )
        conn.commit()

    def get_embedding(self, entity_id: str) -> Optional[EntityEmbedding]:
        """Get embedding for an entity.

        Args:
            entity_id: Entity ID

        Returns:
            EntityEmbedding if found, None otherwise
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM entity_embeddings WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()
        if not row:
            return None

        embedding = self._blob_to_embedding(row["embedding"])
        computed_at = datetime.fromisoformat(row["computed_at"]) if row["computed_at"] else datetime.now()
        return EntityEmbedding(
            entity_id=row["entity_id"],
            embedding=embedding,
            model=row["model"],
            computed_at=computed_at,
        )

    def get_all_embeddings(self) -> dict[str, list[float]]:
        """Get all embeddings as a dict mapping entity_id to embedding.

        Returns:
            Dict mapping entity_id to embedding vector
        """
        conn = self._get_conn()
        rows = conn.execute("SELECT entity_id, embedding FROM entity_embeddings").fetchall()
        return {row["entity_id"]: self._blob_to_embedding(row["embedding"]) for row in rows}

    def get_entities_without_embeddings(self) -> list[Entity]:
        """Get entities that don't have embeddings yet.

        Returns:
            List of entities missing embeddings
        """
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT e.* FROM entities e
            LEFT JOIN entity_embeddings ee ON e.id = ee.entity_id
            WHERE ee.entity_id IS NULL
            """
        ).fetchall()
        return [self._row_to_entity(row) for row in rows]

    @staticmethod
    def _embedding_to_blob(embedding: list[float]) -> bytes:
        """Convert embedding list to binary blob."""
        return struct.pack(f"{len(embedding)}f", *embedding)

    @staticmethod
    def _blob_to_embedding(blob: bytes) -> list[float]:
        """Convert binary blob to embedding list."""
        count = len(blob) // 4  # 4 bytes per float
        return list(struct.unpack(f"{count}f", blob))

    # ==================== Utilities ====================

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity and all associated data.

        Args:
            entity_id: Entity ID to delete

        Returns:
            True if entity was deleted, False if not found
        """
        conn = self._get_conn()
        # Delete in order due to foreign keys
        conn.execute("DELETE FROM entity_embeddings WHERE entity_id = ?", (entity_id,))
        conn.execute("DELETE FROM surface_forms WHERE entity_id = ?", (entity_id,))
        conn.execute(
            "DELETE FROM entity_links WHERE source_id = ? OR target_id = ?",
            (entity_id, entity_id),
        )
        cursor = conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
        conn.commit()
        return cursor.rowcount > 0

    def clear(self) -> None:
        """Clear all data from the store."""
        conn = self._get_conn()
        conn.execute("DELETE FROM entity_embeddings")
        conn.execute("DELETE FROM surface_forms")
        conn.execute("DELETE FROM entity_links")
        conn.execute("DELETE FROM entities")
        conn.commit()
