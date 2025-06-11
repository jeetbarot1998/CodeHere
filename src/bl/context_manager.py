import asyncio
import asyncpg
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Dict, Optional, Tuple
import uuid
from datetime import datetime
import logging
from langchain_ollama import OllamaEmbeddings
import json

logger = logging.getLogger(__name__)


class ContextManager:
    def __init__(self,
                 postgres_url: str = "postgresql://ai_proxy_user:ai_proxy_password@localhost:5432/ai_proxy_db",
                 qdrant_url: str = "localhost",
                 qdrant_port: int = 6333):

        self.postgres_url = postgres_url
        self.qdrant_client = QdrantClient(host=qdrant_url, port=qdrant_port)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.collection_name = "ai_conversations"
        self.pool = None

        # Initialize Qdrant collection
        self._init_qdrant_collection()

    def _init_qdrant_collection(self):
        """Initialize Qdrant collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections().collections
            if not any(col.name == self.collection_name for col in collections):
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)  # nomic-embed-text dimension
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")

    async def init_postgres(self):
        """Initialize PostgreSQL connection pool"""
        self.pool = await asyncpg.create_pool(self.postgres_url)

    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()

    async def ensure_project_exists(self, project_name: str) -> None:
        """Ensure project exists in database"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO projects (project_name) 
                VALUES ($1) 
                ON CONFLICT (project_name) DO NOTHING
            """, project_name)

    async def ensure_user_exists(self, user_id: str, team_id: Optional[str] = None) -> None:
        """Ensure user exists in database"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO users (user_id, team_id) 
                VALUES ($1, $2) 
                ON CONFLICT (user_id) DO NOTHING
            """, user_id, team_id)

    async def save_message(self,
                           project_name: str,
                           user_id: str,
                           session_id: str,  # Added session_id parameter
                           role: str,
                           content: str,
                           model_used: Optional[str] = None,
                           parent_message_id: Optional[str] = None,
                           metadata: Optional[dict] = None) -> str:
        """Save a message to both PostgreSQL and Qdrant"""

        # Ensure project and user exist
        await self.ensure_project_exists(project_name)
        await self.ensure_user_exists(user_id)

        # Generate embedding
        embedding = await asyncio.to_thread(
            self.embeddings.embed_query, content
        )

        # Save to PostgreSQL
        message_id = str(uuid.uuid4())
        vector_id = str(uuid.uuid4())

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO messages 
                (id, project_name, user_id, session_id, role, content, parent_message_id, 
                 model_used, metadata, vector_id)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, uuid.UUID(message_id), project_name, user_id, session_id, role, content,
                               uuid.UUID(parent_message_id) if parent_message_id else None,
                               model_used, json.dumps(metadata or {}), vector_id)

        # Save to Qdrant with project filtering
        point = PointStruct(
            id=vector_id,
            vector=embedding,
            payload={
                "message_id": message_id,
                "project_name": project_name,
                "user_id": user_id,
                "session_id": session_id,  # Include session_id in payload
                "role": role,
                "content": content,
                "parent_message_id": parent_message_id,
                "created_at": datetime.now().isoformat()
            }
        )

        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

        logger.info(f"Saved message {message_id} for project {project_name}, session {session_id}")
        return message_id

    async def search_similar_context(self,
                                     query: str,
                                     project_name: str,
                                     limit: int = 3) -> List[Dict]:
        """Search for similar messages within a project using Qdrant only"""

        # Generate embedding for query
        query_embedding = await asyncio.to_thread(
            self.embeddings.embed_query, query
        )

        # Search in Qdrant with project filter
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="project_name",
                        match=MatchValue(value=project_name)
                    )
                ]
            ),
            limit=limit * 2,  # Get more to ensure we have enough Q&A pairs
            score_threshold=0.7  # Optional: can remove this
        )

        # Group by parent-child relationships
        context_items = []
        seen_pairs = set()

        """
            When we search for "What is singleton pattern?", Qdrant might return:

            An answer about singleton (without its question)
            A similar question (without its answer)
            Random matches from the middle of conversations


            We need complete Q&A pairs for context, so the code:

            If it finds an answer → looks up its question in PostgreSQL or Qdrant
            If it finds a question → looks up its answer in PostgreSQL or Qdrant
        """
        for hit in search_result:
            payload = hit.payload

            # If this is an answer, get its question from Qdrant
            if payload["role"] == "assistant" and payload.get("parent_message_id"):
                pair_id = payload["parent_message_id"]

                if pair_id not in seen_pairs:
                    seen_pairs.add(pair_id)

                    # Retrieve the parent question directly from Qdrant
                    parent_results = self.qdrant_client.retrieve(
                        collection_name=self.collection_name,
                        ids=[payload["parent_message_id"]]
                    )

                    if parent_results:
                        parent_payload = parent_results[0].payload
                        context_items.append({
                            "question": parent_payload["content"],
                            "answer": payload["content"],
                            "score": hit.score
                        })

            # If this is a question, find its answer in Qdrant
            elif payload["role"] == "user":
                message_id = payload["message_id"]

                if message_id not in seen_pairs:
                    # Use scroll to find messages with exact parent_message_id match
                    answer_results, _ = self.qdrant_client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(
                                    key="parent_message_id",
                                    match=MatchValue(value=message_id)
                                ),
                                FieldCondition(
                                    key="project_name",
                                    match=MatchValue(value=project_name)
                                ),
                                FieldCondition(
                                    key="role",
                                    match=MatchValue(value="assistant")
                                )
                            ]
                        ),
                        limit=1,
                        with_payload=True
                    )

                    if answer_results:
                        seen_pairs.add(message_id)
                        context_items.append({
                            "question": payload["content"],
                            "answer": answer_results[0].payload["content"],
                            "score": hit.score
                        })

            if len(context_items) >= limit:
                break

        # Sort by score and return top results
        context_items.sort(key=lambda x: x["score"], reverse=True)
        return context_items[:limit]

    async def get_recent_messages(self,
                                  project_name: str,
                                  user_id: str,
                                  session_id: str,  # Added session_id parameter
                                  limit: int = 3) -> List[Dict]:
        """Get recent messages from the current session"""

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT role, content, created_at
                FROM messages
                WHERE project_name = $1 AND user_id = $2 AND session_id = $3
                ORDER BY created_at DESC
                LIMIT $4
            """, project_name, user_id, session_id, limit * 2)  # Get both Q&A pairs

        # Convert to list and reverse to maintain chronological order
        messages = [
            {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["created_at"].isoformat()
            }
            for row in reversed(rows)
        ]

        return messages[-limit:] if len(messages) > limit else messages

    async def get_session_messages(self,
                                   project_name: str,
                                   user_id: str,
                                   session_id: str) -> List[Dict]:
        """Get all messages from a specific session"""

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT role, content, created_at, model_used, metadata
                FROM messages
                WHERE project_name = $1 AND user_id = $2 AND session_id = $3
                ORDER BY created_at ASC
            """, project_name, user_id, session_id)

        return [
            {
                "role": row["role"],
                "content": row["content"],
                "timestamp": row["created_at"].isoformat(),
                "model_used": row["model_used"],
                "metadata": row["metadata"]
            }
            for row in rows
        ]

    async def star_message(self, message_id: str) -> None:
        """Star a message for quality curation"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE messages 
                SET starred = TRUE 
                WHERE id = $1::uuid
            """, uuid.UUID(message_id))

    async def get_starred_messages(self, project_name: str) -> List[Dict]:
        """Get starred messages for a project (for training data)"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT m1.content as question, m2.content as answer
                FROM messages m1
                JOIN messages m2 ON m2.parent_message_id = m1.id
                WHERE m1.project_name = $1 
                AND (m1.starred = TRUE OR m2.starred = TRUE)
                AND m1.role = 'user' 
                AND m2.role = 'assistant'
                ORDER BY m1.created_at DESC
            """, project_name)

        return [{"question": row["question"], "answer": row["answer"]} for row in rows]

    async def tag_message(self, message_id: str, tag: str) -> None:
        """Tag a message for training data curation"""
        async with self.pool.acquire() as conn:
            # Get current metadata
            current_metadata = await conn.fetchval("""
                SELECT metadata FROM messages WHERE id = $1::uuid
            """, uuid.UUID(message_id))

            if current_metadata:
                metadata = json.loads(current_metadata)
            else:
                metadata = {}

            # Add tag to metadata
            if "tags" not in metadata:
                metadata["tags"] = []

            if tag not in metadata["tags"]:
                metadata["tags"].append(tag)

            # Update message with new metadata
            await conn.execute("""
                UPDATE messages 
                SET metadata = $1
                WHERE id = $2::uuid
            """, json.dumps(metadata), uuid.UUID(message_id))

    async def get_project_code_statistics(self, project_name: str) -> Dict:
        """Get coding statistics for a project"""
        async with self.pool.acquire() as conn:
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_messages,
                    COUNT(CASE WHEN role = 'user' THEN 1 END) as user_messages,
                    COUNT(CASE WHEN role = 'assistant' THEN 1 END) as assistant_messages,
                    COUNT(CASE WHEN metadata->>'is_error_related' = 'true' THEN 1 END) as error_discussions,
                    COUNT(CASE WHEN metadata->>'has_code_blocks' = 'true' THEN 1 END) as messages_with_code,
                    AVG(CAST(metadata->>'complexity_score' AS INTEGER)) as avg_complexity,
                    COUNT(DISTINCT session_id) as total_sessions,
                    COUNT(DISTINCT metadata->>'language') as languages_used
                FROM messages 
                WHERE project_name = $1
            """, project_name)

            # Get top languages
            language_stats = await conn.fetch("""
                SELECT 
                    metadata->>'language' as language,
                    COUNT(*) as count
                FROM messages 
                WHERE project_name = $1 AND metadata->>'language' IS NOT NULL
                GROUP BY metadata->>'language'
                ORDER BY count DESC
                LIMIT 5
            """, project_name)

            # Get top task types
            task_stats = await conn.fetch("""
                SELECT 
                    metadata->>'task_type' as task_type,
                    COUNT(*) as count
                FROM messages 
                WHERE project_name = $1 AND metadata->>'task_type' IS NOT NULL
                GROUP BY metadata->>'task_type'
                ORDER BY count DESC
                LIMIT 5
            """, project_name)

            return {
                "total_messages": stats["total_messages"],
                "user_messages": stats["user_messages"],
                "assistant_messages": stats["assistant_messages"],
                "error_discussions": stats["error_discussions"],
                "messages_with_code": stats["messages_with_code"],
                "avg_complexity": float(stats["avg_complexity"]) if stats["avg_complexity"] else 0,
                "total_sessions": stats["total_sessions"],
                "languages_used": stats["languages_used"],
                "top_languages": [{"language": row["language"], "count": row["count"]} for row in language_stats],
                "top_task_types": [{"task_type": row["task_type"], "count": row["count"]} for row in task_stats]
            }

    async def get_project_libraries(self, project_name: str) -> List[Dict]:
        """Get most used libraries in a project"""
        async with self.pool.acquire() as conn:
            # This is a simplified version - for full implementation,
            # you'd need to parse the code_analysis JSON from metadata
            rows = await conn.fetch("""
                SELECT 
                    metadata,
                    created_at
                FROM messages 
                WHERE project_name = $1 
                AND metadata->>'code_analysis' IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 100
            """, project_name)

            # Count library usage across messages
            library_counts = {}
            for row in rows:
                metadata = row["metadata"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)

                code_analysis = metadata.get("code_analysis", {})
                imports = code_analysis.get("imports", {})
                all_libraries = imports.get("all_libraries", [])

                for lib in all_libraries:
                    library_counts[lib] = library_counts.get(lib, 0) + 1

            # Sort by usage count
            sorted_libraries = sorted(library_counts.items(), key=lambda x: x[1], reverse=True)

            return [{"library": lib, "usage_count": count} for lib, count in sorted_libraries[:10]]