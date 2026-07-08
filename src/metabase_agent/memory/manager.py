from __future__ import annotations

import hashlib
from typing import Any

from metabase_agent.config.settings import Settings
from metabase_agent.memory.models import (
    CandidateMemory,
    MemoryContext,
    MemoryRecord,
    MemoryStatus,
    MemoryType,
    utc_now_iso,
)
from metabase_agent.memory.namespaces import user_namespace
from metabase_agent.memory.prompt import build_context
from metabase_agent.memory.repository import (
    MemoryRepository,
    MongoMemoryRepository,
    NullMemoryRepository,
)
from metabase_agent.memory.vector import (
    EmbeddingProvider,
    HashEmbeddingProvider,
    NullVectorIndex,
    OpenAIEmbeddingProvider,
    PgVectorIndex,
    SiliconFlowEmbeddingProvider,
    VectorIndex,
)


class MemoryManager:
    def __init__(self, repository: MemoryRepository, vector_index: VectorIndex, embedding_provider: EmbeddingProvider) -> None:
        self.repository = repository
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider

    def load_context(self, *, tenant_id: str, user_id: str, query: str, limit: int = 5) -> MemoryContext:
        semantic_ns = user_namespace(tenant_id, user_id, MemoryType.SEMANTIC)
        procedural_ns = user_namespace(tenant_id, user_id, MemoryType.PROCEDURAL)

        profile_keys = (
            "profile.language",
            "profile.answer_style",
            "profile.default_database",
            "profile.default_schema",
            "analytics.table_context",
        )
        profile = [record for key in profile_keys if (record := self.repository.get(semantic_ns, key)) is not None and record.status == MemoryStatus.ACTIVE]
        active_rules = self.repository.list_namespace(procedural_ns, status=MemoryStatus.ACTIVE, limit=6)
        related: list[MemoryRecord] = []
        if query.strip():
            embedding = self.embedding_provider.embed(query)
            ids = self.vector_index.search(
                tenant_id,
                user_id,
                embedding,
                memory_types=[MemoryType.SEMANTIC.value, MemoryType.EPISODIC.value],
                limit=limit,
            )
            seen = {record.id for record in profile}
            related = [record for record in self.repository.get_many(ids) if record.status == MemoryStatus.ACTIVE and record.id not in seen]
        return build_context(profile, active_rules, related)

    def record_interaction(
        self,
        *,
        tenant_id: str,
        user_id: str,
        question: str,
        answer: str,
        query_result: dict[str, Any] | None = None,
        query_plan: dict[str, Any] | None = None,
    ) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for candidate in self._extract_candidates(question=question, answer=answer, query_result=query_result, query_plan=query_plan):
            record = self._upsert_candidate(tenant_id=tenant_id, user_id=user_id, candidate=candidate)
            if record is not None:
                records.append(record)
        return records

    def list_memories(
        self,
        *,
        tenant_id: str,
        user_id: str,
        memory_type: MemoryType | None = None,
        status: MemoryStatus | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        return self.repository.list_user(
            tenant_id,
            user_id,
            memory_type=memory_type.value if memory_type is not None else None,
            status=status,
            limit=limit,
        )

    def put_memory(
        self,
        *,
        tenant_id: str,
        user_id: str,
        memory_type: MemoryType,
        content: str,
        key: str | None = None,
        value: Any = None,
        metadata: dict[str, Any] | None = None,
        confidence: float = 1.0,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        source: str = "manual",
    ) -> MemoryRecord | None:
        return self._upsert_candidate(
            tenant_id=tenant_id,
            user_id=user_id,
            candidate=CandidateMemory(
                memory_type=memory_type,
                key=key,
                content=content,
                value=value,
                metadata=metadata or {},
                confidence=confidence,
                status=status,
                source=source,
            ),
        )

    def update_status(self, *, tenant_id: str, user_id: str, record_id: str, status: MemoryStatus) -> MemoryRecord | None:
        record = self.repository.get_by_id(record_id)
        if record is None or record.tenant_id != tenant_id or record.user_id != user_id:
            return None
        now = utc_now_iso()
        record.status = status
        record.updated_at = now
        record.last_seen = now
        self.repository.put(record)
        if record.memory_type in {MemoryType.SEMANTIC, MemoryType.EPISODIC}:
            self.vector_index.upsert(record, self.embedding_provider.embed(record.content))
        return record

    def _upsert_candidate(self, *, tenant_id: str, user_id: str, candidate: CandidateMemory) -> MemoryRecord | None:
        if candidate.confidence < 0.6 or not candidate.content.strip():
            return None
        namespace = user_namespace(tenant_id, user_id, candidate.memory_type)
        key = candidate.key or self._default_key(candidate)
        old = self.repository.get(namespace, key)
        now = utc_now_iso()
        if old is not None:
            old.content = candidate.content
            old.value = candidate.value
            old.metadata = {**old.metadata, **candidate.metadata}
            old.confidence = max(old.confidence, candidate.confidence)
            old.status = candidate.status
            old.updated_at = now
            old.last_seen = now
            record = old
        else:
            record = MemoryRecord(
                id=self._record_id(namespace, key),
                tenant_id=tenant_id,
                user_id=user_id,
                namespace=namespace,
                key=key,
                memory_type=candidate.memory_type,
                content=candidate.content,
                value=candidate.value,
                metadata=candidate.metadata,
                confidence=candidate.confidence,
                status=candidate.status,
                source=candidate.source,
                created_at=now,
                updated_at=now,
                last_seen=now,
            )
        self.repository.put(record)
        if record.status == MemoryStatus.ACTIVE and record.memory_type in {MemoryType.SEMANTIC, MemoryType.EPISODIC}:
            self.vector_index.upsert(record, self.embedding_provider.embed(record.content))
        return record

    def _extract_candidates(
        self,
        *,
        question: str,
        answer: str,
        query_result: dict[str, Any] | None,
        query_plan: dict[str, Any] | None,
    ) -> list[CandidateMemory]:
        candidates: list[CandidateMemory] = []
        if any(marker in question for marker in ("以后", "记住", "默认", "偏好")):
            if "中文" in question:
                candidates.append(
                    CandidateMemory(
                        memory_type=MemoryType.SEMANTIC,
                        key="profile.language",
                        content="用户偏好中文回答。",
                        value="zh-CN",
                        confidence=0.9,
                    )
                )
            if any(marker in question for marker in ("简洁", "直接", "详细", "工程")):
                candidates.append(
                    CandidateMemory(
                        memory_type=MemoryType.SEMANTIC,
                        key="profile.answer_style",
                        content=f"用户表达了回答风格偏好：{question}",
                        value=question,
                        confidence=0.78,
                    )
                )
        if query_plan:
            database = query_plan.get("database_name")
            schema = query_plan.get("schema_name")
            table = query_plan.get("table_name")
            if database:
                candidates.append(CandidateMemory(MemoryType.SEMANTIC, key="profile.default_database", content=f"用户最近使用的默认数据库是 `{database}`。", value=database, confidence=0.65))
            if schema:
                candidates.append(CandidateMemory(MemoryType.SEMANTIC, key="profile.default_schema", content=f"用户最近使用的默认 schema/dataset 是 `{schema}`。", value=schema, confidence=0.65))
            if table:
                candidates.append(
                    CandidateMemory(
                        MemoryType.SEMANTIC,
                        key="analytics.table_context",
                        content=f"用户最近分析的表是 `{schema + '.' if schema else ''}{table}`。",
                        value={"schema_name": schema, "table_name": table},
                        confidence=0.7,
                    )
                )
        if query_result and query_result.get("status") in {"completed", "requires_approval", "rejected", "failed"}:
            status = str(query_result.get("status"))
            summary = f"用户问题：{question}\n处理结果：{answer or status}"
            candidates.append(
                CandidateMemory(
                    memory_type=MemoryType.EPISODIC,
                    key=f"event.analysis.{_hash_text(summary)[:16]}",
                    content=summary,
                    value={"question": question, "answer": answer, "status": status},
                    metadata={"query_result_status": status},
                    confidence=0.72,
                )
            )
        return candidates

    def _default_key(self, candidate: CandidateMemory) -> str:
        prefix = "note" if candidate.memory_type == MemoryType.SEMANTIC else candidate.memory_type.value
        return f"{prefix}.{_hash_text(candidate.content)[:16]}"

    def _record_id(self, namespace: tuple[str, ...], key: str) -> str:
        return _hash_text("/".join(namespace) + ":" + key)


def build_memory_manager(settings: Settings) -> MemoryManager:
    if not settings.agent_long_term_memory_enabled:
        return MemoryManager(NullMemoryRepository(), NullVectorIndex(), HashEmbeddingProvider())

    repository: MemoryRepository
    if settings.agent_mongodb_uri:
        repository = MongoMemoryRepository(
            settings.agent_mongodb_uri,
            database=settings.agent_mongodb_database,
            collection=settings.agent_memory_collection,
        )
    else:
        repository = NullMemoryRepository()

    vector_index: VectorIndex = NullVectorIndex()
    if settings.agent_pgvector_dsn:
        vector_index = PgVectorIndex(settings.agent_pgvector_dsn, table=settings.agent_pgvector_table)

    embedding_provider: EmbeddingProvider
    if settings.agent_embedding_provider == "openai":
        embedding_provider = OpenAIEmbeddingProvider(settings)
    elif settings.agent_embedding_provider == "siliconflow":
        embedding_provider = SiliconFlowEmbeddingProvider(settings)
    else:
        embedding_provider = HashEmbeddingProvider(settings.agent_embedding_dimensions)

    return MemoryManager(repository, vector_index, embedding_provider)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
