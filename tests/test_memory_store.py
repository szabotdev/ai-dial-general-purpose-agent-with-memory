import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import PurePosixPath
from task.tools.memory.memory_store import LongTermMemoryStore


def test_init_sets_endpoint():
    store = LongTermMemoryStore(endpoint="http://localhost:8080")
    assert store.endpoint == "http://localhost:8080"


def test_init_creates_empty_cache():
    store = LongTermMemoryStore(endpoint="http://localhost:8080")
    assert store.cache == {}


def test_init_creates_sentence_transformer():
    store = LongTermMemoryStore(endpoint="http://localhost:8080")
    assert store.model is not None


@pytest.mark.asyncio
async def test_get_memory_file_path():
    store = LongTermMemoryStore(endpoint="http://localhost:8080")
    mock_dial = AsyncMock()
    mock_dial.my_appdata_home = AsyncMock(
        return_value=PurePosixPath("files/bucket123/appdata/general-purpose-agent")
    )
    path = await store._get_memory_file_path(mock_dial)
    assert path == "files/bucket123/appdata/general-purpose-agent/__long-memories/data.json"


from task.tools.memory._models import MemoryCollection, Memory, MemoryData


@pytest.mark.asyncio
async def test_load_memories_returns_cached_collection():
    """Cache hit: should return cached value without hitting the network."""
    store = LongTermMemoryStore(endpoint="http://localhost:8080")
    cached = MemoryCollection()
    store.cache["files/bucket/appdata/agent/__long-memories/data.json"] = cached

    mock_dial = AsyncMock()
    mock_dial.my_appdata_home = AsyncMock(
        return_value=PurePosixPath("files/bucket/appdata/agent")
    )

    with patch("task.tools.memory.memory_store.AsyncDial", return_value=mock_dial):
        result = await store._load_memories(api_key="test-key")

    assert result is cached
    mock_dial.files.download.assert_not_called()


@pytest.mark.asyncio
async def test_load_memories_returns_empty_collection_on_not_found():
    """File doesn't exist yet: should return empty MemoryCollection."""
    store = LongTermMemoryStore(endpoint="http://localhost:8080")

    mock_dial = AsyncMock()
    mock_dial.my_appdata_home = AsyncMock(
        return_value=PurePosixPath("files/bucket/appdata/agent")
    )
    mock_dial.files.download = AsyncMock(side_effect=Exception("404 not found"))

    with patch("task.tools.memory.memory_store.AsyncDial", return_value=mock_dial):
        result = await store._load_memories(api_key="test-key")

    assert isinstance(result, MemoryCollection)
    assert result.memories == []


@pytest.mark.asyncio
async def test_add_memory_appends_and_returns_success():
    store = LongTermMemoryStore(endpoint="http://localhost:8080")

    mock_dial = AsyncMock()
    mock_dial.my_appdata_home = AsyncMock(
        return_value=PurePosixPath("files/bucket/appdata/agent")
    )
    mock_dial.files.download = AsyncMock(side_effect=Exception("not found"))
    mock_dial.files.upload = AsyncMock()

    with patch("task.tools.memory.memory_store.AsyncDial", return_value=mock_dial):
        result = await store.add_memory(
            api_key="key",
            content="User lives in Budapest",
            importance=0.9,
            category="personal_info",
            topics=["location"]
        )

    assert "successfully" in result.lower()
    path = "files/bucket/appdata/agent/__long-memories/data.json"
    assert len(store.cache[path].memories) == 1
    assert store.cache[path].memories[0].data.content == "User lives in Budapest"
    assert len(store.cache[path].memories[0].embedding) == 384


from datetime import datetime, UTC, timedelta


def _make_collection_with_n_memories(n: int) -> MemoryCollection:
    memories = []
    for i in range(n):
        memories.append(Memory(
            data=MemoryData(id=i, content=f"fact {i}", importance=0.5, category="test", topics=[]),
            embedding=[1.0] + [0.0] * 383
        ))
    return MemoryCollection(memories=memories)


def test_needs_dedup_false_when_few_memories():
    store = LongTermMemoryStore(endpoint="http://localhost:8080")
    col = _make_collection_with_n_memories(5)
    assert store._needs_deduplication(col) is False


def test_needs_dedup_true_when_many_and_never_deduped():
    store = LongTermMemoryStore(endpoint="http://localhost:8080")
    col = _make_collection_with_n_memories(15)
    col.last_deduplicated_at = None
    assert store._needs_deduplication(col) is True


def test_needs_dedup_false_when_recently_deduped():
    store = LongTermMemoryStore(endpoint="http://localhost:8080")
    col = _make_collection_with_n_memories(15)
    col.last_deduplicated_at = datetime.now(UTC) - timedelta(hours=1)
    assert store._needs_deduplication(col) is False


def test_needs_dedup_true_when_dedup_overdue():
    store = LongTermMemoryStore(endpoint="http://localhost:8080")
    col = _make_collection_with_n_memories(15)
    col.last_deduplicated_at = datetime.now(UTC) - timedelta(hours=25)
    assert store._needs_deduplication(col) is True


def _make_memory(content: str, importance: float, embedding: list[float]) -> Memory:
    return Memory(
        data=MemoryData(id=1, content=content, importance=importance, category="test", topics=[]),
        embedding=embedding
    )


def test_deduplicate_fast_keeps_all_distinct_memories():
    store = LongTermMemoryStore(endpoint="http://localhost:8080")
    # Orthogonal unit vectors — cosine similarity = 0
    e1 = [1.0] + [0.0] * 383
    e2 = [0.0, 1.0] + [0.0] * 382
    e3 = [0.0, 0.0, 1.0] + [0.0] * 381
    memories = [
        _make_memory("fact A", 0.5, e1),
        _make_memory("fact B", 0.5, e2),
        _make_memory("fact C", 0.5, e3),
    ]
    result = store._deduplicate_fast(memories)
    assert len(result) == 3


def test_deduplicate_fast_removes_near_duplicate_keeps_higher_importance():
    store = LongTermMemoryStore(endpoint="http://localhost:8080")
    # Identical embeddings → cosine similarity = 1.0 (well above 0.75)
    e = [1.0] + [0.0] * 383
    memories = [
        _make_memory("duplicate A", importance=0.3, embedding=list(e)),
        _make_memory("duplicate B", importance=0.9, embedding=list(e)),
    ]
    result = store._deduplicate_fast(memories)
    assert len(result) == 1
    assert result[0].data.importance == 0.9


def test_deduplicate_fast_handles_single_memory():
    store = LongTermMemoryStore(endpoint="http://localhost:8080")
    e = [1.0] + [0.0] * 383
    memories = [_make_memory("only one", 0.5, list(e))]
    result = store._deduplicate_fast(memories)
    assert len(result) == 1


@pytest.mark.asyncio
async def test_search_memories_returns_empty_list_when_no_memories():
    store = LongTermMemoryStore(endpoint="http://localhost:8080")
    mock_dial = AsyncMock()
    mock_dial.my_appdata_home = AsyncMock(
        return_value=PurePosixPath("files/bucket/appdata/agent")
    )
    mock_dial.files.download = AsyncMock(side_effect=Exception("not found"))

    with patch("task.tools.memory.memory_store.AsyncDial", return_value=mock_dial):
        result = await store.search_memories(api_key="key", query="anything")

    assert result == []


@pytest.mark.asyncio
async def test_search_memories_returns_top_k_results():
    store = LongTermMemoryStore(endpoint="http://localhost:8080")

    # Pre-populate cache with 3 known memories using real embeddings
    path = "files/bucket/appdata/agent/__long-memories/data.json"
    col = MemoryCollection()
    contents = ["lives in Budapest", "works as a developer", "likes Python"]
    for i, text in enumerate(contents):
        emb = store.model.encode([text])[0].tolist()
        col.memories.append(Memory(
            data=MemoryData(id=i, content=text, importance=0.5, category="info", topics=[]),
            embedding=emb
        ))
    store.cache[path] = col

    mock_dial = AsyncMock()
    mock_dial.my_appdata_home = AsyncMock(
        return_value=PurePosixPath("files/bucket/appdata/agent")
    )

    with patch("task.tools.memory.memory_store.AsyncDial", return_value=mock_dial):
        results = await store.search_memories(api_key="key", query="where does the user live", top_k=2)

    assert len(results) == 2
    # The most relevant result should be about Budapest
    assert any("Budapest" in r.content for r in results)


@pytest.mark.asyncio
async def test_delete_all_memories_clears_cache_and_returns_success():
    store = LongTermMemoryStore(endpoint="http://localhost:8080")
    path = "files/bucket/appdata/agent/__long-memories/data.json"
    store.cache[path] = MemoryCollection()

    mock_dial = AsyncMock()
    mock_dial.my_appdata_home = AsyncMock(
        return_value=PurePosixPath("files/bucket/appdata/agent")
    )
    mock_dial.files.delete = AsyncMock()

    with patch("task.tools.memory.memory_store.AsyncDial", return_value=mock_dial):
        result = await store.delete_all_memories(api_key="key")

    assert path not in store.cache
    mock_dial.files.delete.assert_called_once_with(path)
    assert "deleted" in result.lower()
