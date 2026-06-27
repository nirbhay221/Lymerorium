"""Vector memory for debate verdicts and per-agent position history."""


import json
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

# Anchor to vision-app/ (parent of swarm_core/) so paths work regardless of CWD
_MEM_DIR = Path(__file__).resolve().parent.parent
_INDEX_PATH = _MEM_DIR / "vector_memory.usearch"
_VERDICTS_PATH = _MEM_DIR / "vector_memory.json"


MAX_VERDICTS = 50


class VectorMemory:
    """
    Verdicts stored in a {key: verdict} dict - keys are monotonic so they stay
    consistent with the usearch index even after eviction.
    """

    def __init__(self):
        self._verdicts: dict[int, dict] = {}
        self._next_key: int = 0
        self._encoder = None          # lazy-loaded SentenceTransformer
        self._index = None            # lazy-loaded usearch Index
        self._lock = threading.Lock()
        self._encode_lock = threading.Lock()  # HF tokenizer is not thread-safe
        self._load_from_disk()        # restore persisted state on startup

    # Encoder / index (loaded once, reused forever)

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            print("[Memory] Loading embedding model (first time only)...")
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            print("[Memory] Embedding model ready")
        return self._encoder

    def _get_index(self):
        if self._index is None:
            from usearch.index import Index
            self._index = Index(ndim=EMBEDDING_DIM, metric="cos")
            self._verify_distance_convention()
        return self._index

    def _verify_distance_convention(self):
        """Sanity-check that usearch cosine distance = 1 - similarity (not similarity itself)."""
        try:
            import numpy as np
            from usearch.index import Index
            idx = Index(ndim=4, metric="cos")
            v = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            idx.add(0, v)
            result = idx.search(v, 1)
            dist = float(result[0].distance)
            # v dot v = 1.0, so similarity = 1.0 and distance should be ~0.0
            if dist > 0.1:
                print(f"[Memory] WARNING: usearch cosine distance={dist:.4f} for identical vectors. "
                      f"Expected ~0.0. Check similarity = 1 - distance assumption.")
        except Exception:
            pass

    def _encode(self, text: str) -> np.ndarray:
        with self._encode_lock:
            return self._get_encoder().encode(
                text, convert_to_numpy=True, normalize_embeddings=True
            )

    # Disk persistence (atomic writes)

    def _load_from_disk(self):
        try:
            if _INDEX_PATH.exists() and _VERDICTS_PATH.exists():
                from usearch.index import Index
                self._index = Index.restore(str(_INDEX_PATH), view=False)
                raw = json.loads(_VERDICTS_PATH.read_text(encoding="utf-8"))
                # New format: {"next_key": N, "verdicts": {"0": {...}, "1": {...}}}
                if isinstance(raw, dict) and "verdicts" in raw:
                    self._verdicts = {int(k): v for k, v in raw["verdicts"].items()}
                    self._next_key = int(raw.get("next_key",
                        (max(self._verdicts) + 1) if self._verdicts else 0))
                # Old format (list) - migrate
                elif isinstance(raw, list):
                    self._verdicts = {i: v for i, v in enumerate(raw)}
                    self._next_key = len(raw)
                    print("[Memory] Migrated old list format to keyed dict")
                print(f"[Memory] Restored {len(self._verdicts)} verdicts (next_key={self._next_key})")
        except Exception as e:
            print(f"[Memory] Disk restore failed (starting fresh): {e}")
            self._index = None
            self._verdicts = {}
            self._next_key = 0

    def _save_locked(self):
        """Must be called while holding self._lock. Writes are atomic (temp + rename)."""
        try:
            tmp_index = _INDEX_PATH.with_suffix(".tmp")
            tmp_verdicts = _VERDICTS_PATH.with_suffix(".json.tmp")
            self._get_index().save(str(tmp_index))
            payload = {
                "next_key": self._next_key,
                "verdicts": {str(k): v for k, v in self._verdicts.items()},
            }
            tmp_verdicts.write_text(
                json.dumps(payload, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            # Atomic rename - only visible to readers once fully written
            tmp_index.replace(_INDEX_PATH)
            tmp_verdicts.replace(_VERDICTS_PATH)
        except Exception as e:
            print(f"[Memory] Disk save failed: {e}")

    # Store a verdict

    def add(self, verdict: dict):
        """Embed and store a debate verdict, then persist to disk atomically."""
        text = verdict.get("topic", "") + ". " + verdict.get("verdict", "")[:300]
        embedding = self._encode(text)

        with self._lock:
            key = self._next_key
            self._next_key += 1
            self._get_index().add(key, embedding)
            self._verdicts[key] = {**verdict, "_key": key, "timestamp": time.time()}

            # Evict oldest when over cap - remove from BOTH dict and usearch
            while len(self._verdicts) > MAX_VERDICTS:
                oldest_key = min(self._verdicts)
                del self._verdicts[oldest_key]
                try:
                    self._get_index().remove(oldest_key)
                except Exception:
                    pass   # usearch returns error if key not present - ignore

            self._save_locked()

    # Semantic search

    def search(self, query: str, threshold: float = 0.42) -> Optional[dict]:
        """
        Find the most semantically similar stored verdict.
        threshold: cosine similarity (0-1). 0.42 = loosely related, 0.75 = very similar.
        """
        with self._lock:
            if not self._verdicts:
                return None

        embedding = self._encode(query)

        with self._lock:
            results = self._get_index().search(embedding, 1)

        if not results or len(results) == 0:
            return None

        similarity = 1.0 - float(results[0].distance)
        if similarity >= threshold:
            with self._lock:
                key = int(results[0].key)
                return self._verdicts.get(key)
        return None

    # Convergence check (replaces Jaccard)

    def convergence_score(self, messages: list[dict]) -> float:
        """
        Average cosine similarity between consecutive recent messages.
        Understands meaning - 'robots destroy jobs' and 'automation eliminates employment'
        score high. Pure word-overlap Jaccard would score zero for those.
        """
        recent = [m["content"] for m in messages[-8:] if not m.get("error")]
        if len(recent) < 2:
            return 0.0

        with self._encode_lock:
            embeddings = self._get_encoder().encode(
                recent, convert_to_numpy=True, normalize_embeddings=True
            )
        scores = [
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]
        return sum(scores) / len(scores) if scores else 0.0

    # Status

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._verdicts)

    @property
    def last_topic(self) -> Optional[str]:
        with self._lock:
            if not self._verdicts:
                return None
            latest_key = max(self._verdicts)
            return self._verdicts[latest_key]["topic"][:60]


# Module-level singleton shared by background loop + oracle chat
MEMORY = VectorMemory()


# Per-agent persistent memory

class AgentMemory:
    """
    Each agent has a journal of their past CLAIM positions from previous debates.
    Persisted to disk as JSON so it survives restarts.
    On retrieval: embed the current topic and find the most relevant past claims.
    """

    _SAVE_PATH = Path(__file__).resolve().parent.parent / "agent_memories.json"
    _MAX_PER_AGENT = 50

    def __init__(self):
        # {agent_name: [{topic, claim, timestamp}, ...]}
        self._data: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        self._load()

    # Disk persistence

    def _load(self):
        try:
            if self._SAVE_PATH.exists():
                self._data = json.loads(self._SAVE_PATH.read_text(encoding="utf-8"))
                print(f"[AgentMemory] Loaded {sum(len(v) for v in self._data.values())} positions")
        except Exception as e:
            print(f"[AgentMemory] Load failed: {e}")
            self._data = {}

    def _save(self):
        try:
            self._SAVE_PATH.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            print(f"[AgentMemory] Save failed: {e}")

    # Store a position

    def save_position(self, agent_name: str, topic: str, claim: str):
        """Save an agent's CLAIM from a debate round."""
        if not claim:
            return
        entry = {"topic": topic, "claim": claim, "timestamp": time.time()}
        with self._lock:
            bucket = self._data.setdefault(agent_name, [])
            bucket.append(entry)
            if len(bucket) > self._MAX_PER_AGENT:
                bucket.pop(0)
        self._save()

    # Retrieve relevant past positions

    def get_relevant(self, agent_name: str, topic: str, top_k: int = 3) -> list[str]:
        """
        Return the agent's top-k most semantically relevant past claims for a topic.
        Returns empty list if the agent has no history yet.
        """
        with self._lock:
            entries = list(self._data.get(agent_name, []))

        if not entries:
            return []

        with MEMORY._encode_lock:
            encoder = MEMORY._get_encoder()
            topic_emb = encoder.encode(topic, convert_to_numpy=True, normalize_embeddings=True)
            past_topics = [e["topic"] for e in entries]
            past_embs = encoder.encode(past_topics, convert_to_numpy=True, normalize_embeddings=True)

        similarities = np.dot(past_embs, topic_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [entries[i]["claim"] for i in top_indices if similarities[i] > 0.3]

    # MARS reflections
    # Separate from claim journals - stores principle-level and procedure-level
    # meta-cognitive reflections produced after each debate.

    _REFL_PATH = Path(__file__).resolve().parent.parent / "agent_reflections.json"
    _MAX_REFL_PER_AGENT = 30

    def _load_reflections(self):
        try:
            if self._REFL_PATH.exists():
                self._reflections: dict[str, list[dict]] = json.loads(
                    self._REFL_PATH.read_text(encoding="utf-8")
                )
            else:
                self._reflections = {}
        except Exception:
            self._reflections = {}

    def save_reflection(self, agent_name: str, topic: str, rtype: str, content: str):
        """
        Store a MARS meta-cognitive reflection.
        rtype: "principle" (generalizable lesson) | "procedure" (specific adjustment)
        """
        if not content or content.startswith("["):
            return
        if not hasattr(self, "_reflections"):
            self._load_reflections()
        entry = {"topic": topic, "type": rtype, "content": content, "timestamp": time.time()}
        with self._lock:
            bucket = self._reflections.setdefault(agent_name, [])
            bucket.append(entry)
            if len(bucket) > self._MAX_REFL_PER_AGENT:
                bucket.pop(0)
        try:
            self._REFL_PATH.write_text(
                json.dumps(self._reflections, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            print(f"[AgentMemory] Reflection save failed: {e}")

    def get_reflections(self, agent_name: str, topic: str,
                        rtype: str | None = None, top_k: int = 2) -> list[str]:
        """
        Retrieve the agent's most topic-relevant past reflections.
        rtype: filter by "principle" or "procedure"; None = both.
        """
        if not hasattr(self, "_reflections"):
            self._load_reflections()
        with self._lock:
            entries = [
                e for e in self._reflections.get(agent_name, [])
                if rtype is None or e.get("type") == rtype
            ]
        if not entries:
            return []
        with MEMORY._encode_lock:
            enc = MEMORY._get_encoder()
            topic_emb = enc.encode(topic, convert_to_numpy=True, normalize_embeddings=True)
            embs = enc.encode([e["topic"] for e in entries],
                              convert_to_numpy=True, normalize_embeddings=True)
        sims = np.dot(embs, topic_emb)
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [entries[i]["content"] for i in top_idx if sims[i] > 0.25]

    # Status

    def stats(self) -> dict:
        with self._lock:
            return {name: len(entries) for name, entries in self._data.items()}


AGENT_MEMORY = AgentMemory()
