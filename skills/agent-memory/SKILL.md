---
name: agent-memory
description: Use this skill for Agent memory, LangGraph memory, MongoDBStore, MongoDBSaver, pgvector, semantic memory, episodic memory, procedural memory, skills, prompt injection, or long-term memory design.
---

# agent-memory

## Memory model

1. MongoDBSaver stores thread checkpoints and short-term graph runtime state.
2. MongoDBStore stores structured long-term memory records.
3. pgvector stores embeddings for semantic retrieval and returns memory IDs.
4. MongoDBStore remains the source of truth for memory metadata, status, confidence, last_seen, and conflicts.
5. Skills are task workflows, not user memory.

## Memory types

- Semantic memory: stable facts, user preferences, table口径, default database/schema.
- Episodic memory: completed analysis events, SQL approval/rejection events, failures and resolutions.
- Procedural memory: rules that affect future behavior. Keep risky rules pending until reviewed.

## Retrieval

1. Use key lookup for stable profile slots.
2. Use pgvector retrieval for related events and non-slot semantic notes.
3. Inject only concise, relevant memory into the prompt.
4. Do not inject pending_review procedural memory.

