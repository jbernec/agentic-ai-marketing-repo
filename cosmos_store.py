from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from collections.abc import Iterable
from typing import cast  # if not already imported
from azure.cosmos import CosmosClient, PartitionKey, exceptions

from langgraph.store.base import (
    BaseStore,
    Item,
    SearchItem,
    IndexConfig,
    ensure_embeddings,
    get_text_at_path,
)

from langgraph.store.base import (
    BaseStore,
    Item,
    SearchItem,
    IndexConfig,
    ensure_embeddings,
    get_text_at_path,
    GetOp,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchOp,
)

Namespace = Tuple[str, ...]


class CosmosDBStore(BaseStore):
    """
    Cosmos DB NoSQL-backed implementation of LangGraph BaseStore.

    Intended to be conceptually similar to PostgresStore, but using Cosmos:

    - Each item is stored as a JSON document with:
        id:           str   (same as namespaceKey)
        namespaceKey: str   (partition key)
        namespace:    list[str]
        key:          str
        value:        dict
        created_at:   ISO datetime string
        updated_at:   ISO datetime string
        ttl:          int   (seconds, optional)
        contentVector: list[float] (optional, for vector search)

    - The container should be created with:
        partition key = /namespaceKey
        optional defaultTtl
        optional vectorEmbeddingPolicy + vectorIndexes for /contentVector
    """

    def __init__(
        self,
        client: CosmosClient,
        database_id: str,
        container_id: str,
        *,
        partition_key_path: str = "/namespaceKey",
        index: Optional[IndexConfig] = None,
    ) -> None:
        """
        Args:
            client:       Azure CosmosClient (NoSQL API)
            database_id:  Name of the database
            container_id: Name of the container
            partition_key_path: Partition key path, default "/namespaceKey"
            index:        Optional IndexConfig for semantic search:
                          {"dims": int, "embed": Embeddings, "fields": list[str] | None}
        """
        self.client = client
        self.database_id = database_id
        self.container_id = container_id
        self.partition_key_path = partition_key_path

        # Optional semantic index (dims, embed, fields)
        self._index_cfg: Optional[IndexConfig] = (
            ensure_embeddings(index) if index is not None else None
        )

        # DB/container clients will be bound after setup()
        self._db = self.client.get_database_client(database_id)
        self._container = self._db.get_container_client(container_id)
        if index is not None:
            cfg = dict(index)
            embed_cfg = cfg.get("embed")
        if embed_cfg is None:
            raise ValueError("index config must include an 'embed' entry")
            cfg["embed"] = ensure_embeddings(embed_cfg)
            cfg.setdefault("fields", None)
            self._index_cfg = cast(IndexConfig, cfg)
        else:
            self._index_cfg = None

    # ----------------------------------------------------------------------
    # Setup: create DB + container (similar purpose to PostgresStore.setup)
    # ----------------------------------------------------------------------
    def setup(self, *, ttl_seconds: Optional[int] = None) -> None:
        """
        Ensure the database and container exist with:

        - Correct partition key
        - Optional default TTL
        - Optional vector embedding policy + vector index for /contentVector

        NOTE:
        - Vector policy / vector index parameters are evolving in the SDK.
          You may need to adjust names/structure to match your SDK and
          portal configuration.
        """

        # 1. Create database if needed
        self.client.create_database_if_not_exists(id=self.database_id)
        self._db = self.client.get_database_client(self.database_id)

        # 2. Build optional policies
        indexing_policy = None
        vector_embedding_policy = None

        # 2a. Vector embedding policy & index if index config is provided
        if self._index_cfg is not None:
            dims = self._index_cfg["dims"]

            # These shapes are representative; confirm with your SDK/docs.
            vector_embedding_policy = {
                "vectorEmbeddings": [
                    {
                        "path": "/contentVector",
                        "dataType": "float32",
                        "dimensions": dims,
                        "distanceFunction": "cosine",
                    }
                ]
            }

            indexing_policy = {
                "includedPaths": [{"path": "/*"}],
                "excludedPaths": [{"path": "/\"_etag\"/?"}],
                "vectorIndexes": [
                    {
                        "path": "/contentVector",
                        "type": "quantizedFlat",  # or "flat", depending on your needs
                    }
                ],
            }

        # 3. Create container if needed
        # NOTE: Some SDK versions may not accept vectorEmbeddingPolicy/indexingPolicy
        # directly in create_container_if_not_exists. If that happens,
        # you can configure vector/index via ARM or the Azure Portal instead.
        kwargs: Dict[str, Any] = {}
        if ttl_seconds is not None:
            kwargs["default_ttl"] = ttl_seconds  # default TTL in seconds

        if indexing_policy is not None:
            kwargs["indexing_policy"] = indexing_policy

        if vector_embedding_policy is not None:
            # Depending on SDK version, this arg name may differ or be unsupported.
            # Check your SDK docs; otherwise manage vector policy via Portal.
            kwargs["vector_embedding_policy"] = vector_embedding_policy

        self._db.create_container_if_not_exists(
            id=self.container_id,
            partition_key=PartitionKey(path=self.partition_key_path),
            **kwargs,
        )

        # Refresh container client
        self._container = self._db.get_container_client(self.container_id)

    # ----------------------------------------------------------------------
    # Helper utilities
    # ----------------------------------------------------------------------
    
    @staticmethod
    def _ns_to_str(namespace: Namespace) -> str:
        return "|".join(namespace)
    
    @staticmethod
    def _make_namespace_key(namespace: Namespace, key: str) -> str:
        ns_str = CosmosDBStore._ns_to_str(namespace)
        return f"{ns_str}::{key}" if ns_str else key

    # @staticmethod
    # def _make_namespace_key(namespace: Namespace, key: str) -> str:
    #     if namespace:
    #         return f"{'/'.join(namespace)}::{key}"
    #     return key

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def _to_item(self, doc: Dict[str, Any]) -> Item:
        return Item(
            namespace=tuple(doc["namespace"]),
            key=doc["key"],
            value=doc["value"],
            created_at=datetime.fromisoformat(doc["created_at"]),
            updated_at=datetime.fromisoformat(doc["updated_at"]),
        )

    # ----------------------------------------------------------------------
    # Core sync BaseStore API
    # ----------------------------------------------------------------------
    def get(
        self,
        namespace: Namespace,
        key: str,
        *,
        refresh_ttl: Optional[bool] = None,  # unused; TTL handled by Cosmos
    ) -> Optional[Item]:
        ns_key = self._make_namespace_key(namespace, key)
        try:
            doc = self._container.read_item(item=ns_key, partition_key=ns_key)
        except exceptions.CosmosResourceNotFoundError:
            return None
        return self._to_item(doc)

    def put(
        self,
        namespace: Namespace,
        key: str,
        value: Optional[Dict[str, Any]],
        *,
        index: Optional[List[str] | bool] = None,
        ttl: Optional[float] = None,  # minutes; we convert to seconds
    ) -> None:
        ns_key = self._make_namespace_key(namespace, key)

        # Deletion semantic: value=None => delete
        if value is None:
            try:
                self._container.delete_item(item=ns_key, partition_key=ns_key)
            except exceptions.CosmosResourceNotFoundError:
                pass
            return

        now = self._now()
        doc: Dict[str, Any] = {
            "id": ns_key,
            "namespaceKey": ns_key,
            "namespace": list(namespace),
            "key": key,
            "value": value,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }

        # Optional per-item TTL (in seconds)
        if ttl is not None:
            doc["ttl"] = int(ttl * 60)

        # Optional semantic index: compute embedding into contentVector
        if self._index_cfg is not None and index is not False:
            fields = index if isinstance(index, list) else self._index_cfg["fields"]

            text_chunks: List[str] = []
            if fields is None:
                # embed entire value
                text_chunks.append(str(value))
            else:
                for path in fields:
                    text_at_path = get_text_at_path(value, path)
                    if isinstance(text_at_path, str):
                        text_chunks.append(text_at_path)

            text_for_embedding = "\n".join(text_chunks) if text_chunks else ""
            if text_for_embedding:
                embed_model = self._index_cfg["embed"]
                [vec] = embed_model.embed_documents([text_for_embedding])
                doc["contentVector"] = vec

        self._container.upsert_item(doc)

    def delete(self, namespace: Namespace, key: str) -> None:
        self.put(namespace, key, value=None)

    def search(
        self,
        namespace_prefix: Namespace,
        *,
        query: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: Optional[bool] = None,  # unused
    ) -> List[SearchItem]:
        """
        Namespace-aware search.

        - If `query` and index config are provided, perform vector search
          using /contentVector.
        - Otherwise, perform a simple structured search using filters and
          namespace prefix.
        """
        ns_prefix_str = self._ns_to_str(namespace_prefix)
        items: List[SearchItem] = []

        # ---------------- Vector search path ----------------
        if query and self._index_cfg is not None:
            embed_model = self._index_cfg["embed"]
            query_vec = embed_model.embed_query(query)

            # NOTE: The exact Cosmos SQL for vector search depends on SDK & account.
            # This is representative; adjust to your exact VectorDistance/ANN support.
            sql = """
            SELECT TOP @limit
                c.id,
                c.namespace,
                c.key,
                c["value"] AS docValue,
                c.created_at,
                c.updated_at,
                VectorDistance(c.contentVector, @embedding) AS similarity
            FROM c
            WHERE STARTSWITH(c.namespaceKey, @ns_prefix)
            ORDER BY VectorDistance(c.contentVector, @embedding)
            """

            params = [
                {"name": "@limit", "value": limit + offset},
                {"name": "@embedding", "value": query_vec},
                {"name": "@ns_prefix", "value": ns_prefix_str},
            ]

            results = list(
                self._container.query_items(
                    query=sql,
                    parameters=params,
                    enable_cross_partition_query=True,
                )
            )

            for doc in results[offset : offset + limit]:
                items.append(
                    SearchItem(
                        namespace=tuple(doc["namespace"]),
                        key=doc["key"],
                        value=doc["value"],
                        created_at=datetime.fromisoformat(doc["created_at"]),
                        updated_at=datetime.fromisoformat(doc["updated_at"]),
                        score=float(doc.get("similarity", 0.0)),
                    )
                )

            return items

        # ---------------- Non-vector / simple search path ----------------
        where_clauses = ["STARTSWITH(c.namespaceKey, @ns_prefix)"]
        params = [{"name": "@ns_prefix", "value": ns_prefix_str}]

        if filter:
            i = 0
            for k, v in filter.items():
                pname = f"@p{i}"
                # Very simple exact-match filter on value.{k}
                # where_clauses.append(f"c.value.{k} = {pname}")
                path_parts = k.split(".")
                field_path = "".join(f'["{part}"]' for part in path_parts)
                where_clauses.append(f'c["value"]{field_path} = {pname}')
                params.append({"name": pname, "value": v})
                i += 1

        where_sql = " AND ".join(where_clauses)
        sql = f"""
        SELECT c.id, c.namespace, c.key, c["value"] AS docValue, c.created_at, c.updated_at
        FROM c
        WHERE {where_sql}
        """

        results = list(
            self._container.query_items(
                query=sql,
                parameters=params,
                enable_cross_partition_query=True,
            )
        )

        for doc in results[offset : offset + limit]:
            items.append(
                SearchItem(
                    namespace=tuple(doc["namespace"]),
                    key=doc["key"],
                    value=doc["docValue"],
                    created_at=datetime.fromisoformat(doc["created_at"]),
                    updated_at=datetime.fromisoformat(doc["updated_at"]),
                    score=None,
                )
            )

        return items

    def list_namespaces(
        self,
        *,
        match_conditions=None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Namespace]:
        """
        Simple namespace listing: DISTINCT namespace arrays.

        - `match_conditions` / `max_depth` can be added if you need more
          structured namespace filtering later.
        """
        sql = "SELECT DISTINCT c.namespace FROM c"
        results = list(
            self._container.query_items(
                query=sql,
                enable_cross_partition_query=True,
            )
        )
        namespaces: List[Namespace] = [tuple(doc["namespace"]) for doc in results]

        return namespaces[offset : offset + limit]

    # ----------------------------------------------------------------------
    # Async shims
    # ----------------------------------------------------------------------
    async def aget(
        self,
        namespace: Namespace,
        key: str,
        *,
        refresh_ttl: Optional[bool] = None,
    ) -> Optional[Item]:
        return self.get(namespace, key, refresh_ttl=refresh_ttl)

    async def aput(
        self,
        namespace: Namespace,
        key: str,
        value: Optional[Dict[str, Any]],
        *,
        index: Optional[List[str] | bool] = None,
        ttl: Optional[float] = None,
    ) -> None:
        self.put(namespace, key, value, index=index, ttl=ttl)

    async def adelete(self, namespace: Namespace, key: str) -> None:
        self.delete(namespace, key)

    async def asearch(
        self,
        namespace_prefix: Namespace,
        *,
        query: Optional[str] = None,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: Optional[bool] = None,
    ) -> List[SearchItem]:
        return self.search(
            namespace_prefix,
            query=query,
            filter=filter,
            limit=limit,
            offset=offset,
            refresh_ttl=refresh_ttl,
        )

    async def alist_namespaces(
        self,
        *,
        match_conditions=None,
        max_depth: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Namespace]:
        return self.list_namespaces(
            match_conditions=match_conditions,
            max_depth=max_depth,
            limit=limit,
            offset=offset,
        )
    def batch(self, ops: Iterable[Op]) -> List[Result]:
        results: List[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self.get(op.namespace, op.key, refresh_ttl=op.refresh_ttl))
            elif isinstance(op, PutOp):
                self.put(
                    op.namespace,
                    op.key,
                    op.value,
                    index=op.index,
                    ttl=op.ttl,
                )
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(
                    self.search(
                        op.namespace_prefix,
                        query=op.query,
                        filter=op.filter,
                        limit=op.limit,
                        offset=op.offset,
                        refresh_ttl=op.refresh_ttl,
                    )
                )
            elif isinstance(op, ListNamespacesOp):
                results.append(
                    self.list_namespaces(
                        match_conditions=op.match_conditions,
                        max_depth=op.max_depth,
                        limit=op.limit,
                        offset=op.offset,
                    )
                )
            else:
                raise ValueError(f"Unsupported op type: {type(op)}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> List[Result]:
        ops_list = list(ops)
        return self.batch(ops_list)
