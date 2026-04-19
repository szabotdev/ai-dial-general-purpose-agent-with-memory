import os
os.environ['OMP_NUM_THREADS'] = '1'

import json
from datetime import datetime, UTC, timedelta
import numpy as np
import faiss
from aidial_client import AsyncDial
from sentence_transformers import SentenceTransformer

from task.tools.memory._models import Memory, MemoryData, MemoryCollection


class LongTermMemoryStore:
    """
    Manages long-term memory storage for users.

    Storage format: Single JSON file per user in DIAL bucket
    - File: {user_id}/long-memories.json
    - Caching: In-memory cache with conversation_id as key
    - Deduplication: O(n log n) using FAISS batch search
    """

    DEDUP_INTERVAL_HOURS = 24

    def __init__(self, endpoint: str):
        #TODO:
        # 1. Set endpoint
        # 2. Create SentenceTransformer as model, model name is `all-MiniLM-L6-v2`
        # 3. Create cache, doct of str and MemoryCollection (it is imitation of cache, normally such cache should be set aside)
        # 4. Make `faiss.omp_set_num_threads(1)` (without this set up you won't be able to work in debug mode in `_deduplicate_fast` method
        self.endpoint = endpoint
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache: dict[str, MemoryCollection] = {}
        faiss.omp_set_num_threads(1)

    async def _get_memory_file_path(self, dial_client: AsyncDial) -> str:
        """Get the path to the memory file in DIAL bucket."""
        #TODO:
        # 1. Get DIAL app home path
        # 2. Return string with path in such format: `files/{bucket_with_app_home}/__long-memories/data.json`
        #    The memories will persist in appdata for this agent in `__long-memories` folder and `data.json` file
        #    (You will be able to check it also in Chat UI in attachments)
        appdata_home = await dial_client.my_appdata_home()
        return f"files/{(appdata_home / '__long-memories/data.json').as_posix()}"

    async def _load_memories(self, api_key: str) -> MemoryCollection:
        #TODO:
        # 1. Create AsyncDial client (api_version is 2025-01-01-preview)
        # 2. Get memory file path
        # 3. Check cache: cache is dict of str and MemoryCollection, for the key we will use `memory file path` to make
        #    it simple. Such key will be unique for user and will allow to access memories across different
        #    conversations and only user can access them. In case if cache is present return its MemoryCollection.
        # ---
        # Below is logic when cache is not present:
        # 4. Open try-except block:
        #   - in try:
        #       - download file content
        #       - in response get content and decode it with 'utf-8'
        #       - load content with `json`
        #       - create MemoryCollection (it is pydentic model, use `model_validate` method)
        #   - in except:
        #       - create MemoryCollection (it will have empty memories, set up time for updated_at, more detailed take
        #         a look at MemoryCollection pydentic model and it Fields)
        # 5. Return created MemoryCollection
        dial_client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
            api_version='2025-01-01-preview'
        )
        memory_file_path = await self._get_memory_file_path(dial_client)

        if memory_file_path in self.cache:
            return self.cache[memory_file_path]

        try:
            response = await dial_client.files.download(memory_file_path)
            content = await response.aget_content()
            data = json.loads(content.decode('utf-8'))
            collection = MemoryCollection.model_validate(data)
        except Exception:
            collection = MemoryCollection(updated_at=datetime.now(UTC))

        self.cache[memory_file_path] = collection
        return collection

    async def _save_memories(self, api_key: str, memories: MemoryCollection):
        """Save memories to DIAL bucket and update cache."""
        #TODO:
        # 1. Create AsyncDial client
        # 2. Get memory file path
        # 3. Update `updated_at` of memories (now)
        # 4. Converts memories to json string (it's pydentic model and it have model dump json method for this). Don't
        #    make any indentations because it will make file 'bigger'. Here is the point that we store all the memories
        #    in one file and 'one memory' with its embeddings takes ~6-8Kb, we expect that there are won't be more that
        #    1000 memories but anyway for 1000 memories it will be ~6-8Mb, so, we need to make at least these small
        #    efforts to make it smaller 😉
        # 5. Put to cache (kind reminder the key is memory file path)
        dial_client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key
        )

        memory_file_path = await self._get_memory_file_path(dial_client)
        memories.updated_at = datetime.now(UTC)
        json_str = memories.model_dump_json()
        file_bytes = json_str.encode('utf-8')
        await dial_client.files.upload(memory_file_path, file_bytes)
        self.cache[memory_file_path] = memories
        print(f"Saved memories to {memory_file_path}")

    async def add_memory(self, api_key: str, content: str, importance: float, category: str, topics: list[str]) -> str:
        """Add a new memory to storage."""
        #TODO:
        # 1. Load memories
        # 2. Make encodings for content with embedding model.
        #    Hint: provide content as list, and after encoding get first result (encode wil return list) and convertit `tolist`
        # 3. Create Memory
        #    - for id use `int(datetime.now(UTC).timestamp())` it will provide time now as int, it will be super enough
        #      to avoid collisions. Also, we won't use id but we added it because maybe in future you will make enhanced
        #      version of long-term memory and after that it will be additional 'headache' to add such ids 😬
        # 4. Add to memories created memory
        # 5. Save memories (it is PUT request bzw, -> https://dialx.ai/dial_api#tag/Files/operation/uploadFile)
        # 6. Return information that content has benn successfully stored
        collection = await self._load_memories(api_key)
        embedding = self.model.encode([content])[0].tolist()
        memory = Memory(
            data=MemoryData(
                id=int(datetime.now(UTC).timestamp()),
                content=content,
                importance=importance,
                category=category,
                topics=topics,
            ),
            embedding=embedding,
        )
        collection.memories.append(memory)
        await self._save_memories(api_key, collection)
        return f"Memory stored successfully: '{content}'"

    async def search_memories(self, api_key: str, query: str, top_k: int = 5) -> list[MemoryData]:
        """
        Search memories using semantic similarity.

        Returns:
            List of MemoryData objects (without embeddings)
        """
        #TODO:
        # 1. Load memories
        # 2. If they are empty return empty array
        # ---
        # 3. Check if they needs_deduplication, if yes then deduplicate_and_save (need to implements both of these methods)
        # 4. Make vector search (embeddings are part of memory)😈
        # 5. Return `top_k` MemoryData based on vector search
        collection = await self._load_memories(api_key)
        if not collection.memories:
            return []

        if self._needs_deduplication(collection):
            print("Deduplication in progress...")
            collection = await self._deduplicate_and_save(api_key, collection)

        query_embedding = np.array(self.model.encode([query]), dtype='float32')
        faiss.normalize_L2(query_embedding)

        embeddings = np.array([m.embedding for m in collection.memories], dtype='float32')
        faiss.normalize_L2(embeddings)

        n, dim = embeddings.shape
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        k = min(top_k, n)
        _, result_indices = index.search(query_embedding, k)

        return [collection.memories[i].data for i in result_indices[0]]

    def _needs_deduplication(self, collection: MemoryCollection) -> bool:
        """Check if deduplication is needed (>24 hours since last deduplication)."""
        #TODO:
        # The criteria for deduplication (collection length > 10 and >24 hours since last deduplication) or
        # (collection length > 10 last deduplication is None)
        if len(collection.memories) <= 10:
            return False
        if collection.last_deduplicated_at is None:
            return True
        elapsed = datetime.now(UTC) - collection.last_deduplicated_at
        return elapsed > timedelta(hours=self.DEDUP_INTERVAL_HOURS)

    async def _deduplicate_and_save(self, api_key: str, collection: MemoryCollection) -> MemoryCollection:
        """
        Deduplicate memories synchronously and save the result.
        Returns the updated collection.
        """
        #TODO:
        # 1. Make fast deduplication (need to implement)
        # 2. Update last_deduplicated_at as now
        # 3. Save deduplicated memories
        # 4. Return deduplicated collection
        collection.memories = self._deduplicate_fast(collection.memories)
        collection.last_deduplicated_at = datetime.now(UTC)
        await self._save_memories(api_key, collection)
        return collection

    def _deduplicate_fast(self, memories: list[Memory]) -> list[Memory]:
        """
        Fast deduplication using FAISS batch search with cosine similarity.

        Strategy:
        - Find k nearest neighbors for each memory using cosine similarity
        - Mark duplicates based on similarity threshold (cosine similarity > 0.75)
        - Keep memory with higher importance
        """
        #TODO:
        # This is the hard part 🔥🔥🔥
        # You need to deduplicate memories, duplicates are the memories that have 75% similarity.
        # Among duplicates remember about `importance`, most important have more priorities to survive
        # It must be fast, it is possible to do for O(n log n), probably you can find faster way (share with community if do 😉)
        # Return deduplicated memories
        if len(memories) <= 1:
            return memories

        SIMILARITY_THRESHOLD = 0.75

        # Build L2-normalised embedding matrix for cosine similarity via inner product
        embeddings = np.array([m.embedding for m in memories], dtype='float32')
        faiss.normalize_L2(embeddings)

        n, dim = embeddings.shape
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Batch search: each memory's k nearest neighbours (including itself)
        k = min(n, 10)
        similarities, indices = index.search(embeddings, k)

        # Union-Find: group memories that are near-duplicates transitively
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j_pos in range(k):
                j = int(indices[i][j_pos])
                if j != i and similarities[i][j_pos] > SIMILARITY_THRESHOLD:
                    union(i, j)

        # Collect groups and keep the highest-importance representative
        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        result = []
        for group_indices in groups.values():
            best = max(group_indices, key=lambda i: memories[i].data.importance)
            result.append(memories[best])

        return result

    async def delete_all_memories(self, api_key: str, ) -> str:
        """
        Delete all memories for the user.

        Removes the memory file from DIAL bucket and clears the cache
        for the current conversation.
        """
        #TODO:
        # 1. Create AsyncDial client
        # 2. Get memory file path
        # 3. Delete file
        # 4. Return info about successful memory deletion
        dial_client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key
        )
        memory_file_path = await self._get_memory_file_path(dial_client)
        try:
            await dial_client.files.delete(memory_file_path)
        except Exception:
            pass
        self.cache.pop(memory_file_path, None)
        return "All long-term memories have been successfully deleted."
