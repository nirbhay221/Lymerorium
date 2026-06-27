"""Episodic memory consolidation: clusters high-quality verdicts and distills multi-debate insights."""


import json
import re
import threading
from pathlib import Path
from typing import Optional

import numpy as np

_BASE = Path(__file__).resolve().parent.parent
_PATH = _BASE / "consolidated_memory.json"
_lock = threading.Lock()

_MIN_CLUSTER_SIZE = 3
_CLUSTER_SIM_THRESH = 0.60  # min similarity for two verdicts to share a cluster
_QUALITY_CONV = 0.65  # min convergence_score to include a verdict
_QUALITY_CONF = 60    # min CONFIDENCE_SCORE to include a verdict

_DISTILL_SYSTEM = """\
You are a knowledge distiller. You have seen multiple AI-swarm debates on similar topics.
Write a 3-5 sentence insight capturing:
1. What the debates consistently agreed on
2. What remained genuinely contested
3. One practical implication for future questions on this topic
Be specific, cite concrete points, avoid generic statements."""


class ConsolidatedMemory:
    """High-confidence retrieval tier built from multi-debate distillation."""

    def __init__(self):
        self._insights: list[dict] = []    # [{topics, insight, embedding, source_count}]
        self._embeddings: np.ndarray | None = None  # shape (N, 384)
        self._load()

    # Persistence

    def _load(self) -> None:
        try:
            if _PATH.exists():
                raw = json.loads(_PATH.read_text(encoding="utf-8"))
                self._insights = raw
                if self._insights:
                    self._embeddings = np.array(
                        [ins["embedding"] for ins in self._insights], dtype=np.float32
                    )
                    print(f"[Consolidation] Loaded {len(self._insights)} insights")
        except Exception as e:
            print(f"[Consolidation] Load failed: {e}")

    def _save(self) -> None:
        try:
            tmp = _PATH.with_suffix(".json.tmp")
            # Store embeddings inline (384 floats × ~20 insights is small)
            tmp.write_text(
                json.dumps(self._insights, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            tmp.replace(_PATH)
        except Exception as e:
            print(f"[Consolidation] Save failed: {e}")

    # Main API

    def consolidate_from(self) -> int:
        """
        Read all verdicts from VectorMemory, cluster by topic, distill each cluster.
        Returns number of new insights written.
        Called from background.py every 10 verdicts.
        """
        from memory import MEMORY

        with MEMORY._lock:
            entries = list(MEMORY._verdicts.values())

        quality = _filter_quality(entries)
        if len(quality) < _MIN_CLUSTER_SIZE:
            print(f"[Consolidation] Only {len(quality)} quality verdicts - skipping")
            return 0

        clusters = _cluster(quality, MEMORY)
        if not clusters:
            return 0

        new_insights: list[dict] = []
        for cluster_entries in clusters:
            ins = _distill_cluster(cluster_entries, MEMORY)
            if ins:
                new_insights.append(ins)

        if not new_insights:
            return 0

        with _lock:
            self._insights = new_insights   # full rebuild - old insights replaced
            if new_insights:
                self._embeddings = np.array(
                    [ins["embedding"] for ins in new_insights], dtype=np.float32
                )
            else:
                self._embeddings = None
            self._save()

        print(f"[Consolidation] Wrote {len(new_insights)} insights from "
              f"{len(quality)} quality verdicts")
        return len(new_insights)

    def search(self, query: str, threshold: float = 0.75) -> Optional[dict]:
        """
        Find the most relevant distilled insight.
        Returns None if no insight exceeds threshold.
        """
        with _lock:
            if not self._insights or self._embeddings is None:
                return None
            embs = self._embeddings.copy()
            insights_copy = list(self._insights)

        from memory import MEMORY
        with MEMORY._encode_lock:
            enc = MEMORY._get_encoder()
            q_emb = enc.encode(query, convert_to_numpy=True, normalize_embeddings=True)

        sims = np.dot(embs, q_emb)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= threshold:
            return insights_copy[best_idx]
        return None

    @property
    def count(self) -> int:
        with _lock:
            return len(self._insights)

    def get_all(self) -> list[dict]:
        with _lock:
            return [
                {k: v for k, v in ins.items() if k != "embedding"}
                for ins in self._insights
            ]


# Private helpers

def _filter_quality(entries: list[dict]) -> list[dict]:
    out = []
    for e in entries:
        if e.get("convergence_score", 0) < _QUALITY_CONV:
            continue
        m = re.search(r"CONFIDENCE_SCORE:\s*(\d+)", e.get("verdict", ""))
        if m and int(m.group(1)) < _QUALITY_CONF:
            continue
        out.append(e)
    return out


def _cluster(entries: list[dict], memory) -> list[list[dict]]:
    texts = [e.get("topic", "") for e in entries]
    with memory._encode_lock:
        enc = memory._get_encoder()
        embs = enc.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    sim_matrix = np.dot(embs, embs.T)  # (N, N)
    assigned = [False] * len(entries)
    clusters: list[list[dict]] = []

    for i in range(len(entries)):
        if assigned[i]:
            continue
        cluster_idxs = [i]
        for j in range(i + 1, len(entries)):
            if not assigned[j] and sim_matrix[i, j] >= _CLUSTER_SIM_THRESH:
                cluster_idxs.append(j)
                assigned[j] = True
        assigned[i] = True

        if len(cluster_idxs) >= _MIN_CLUSTER_SIZE:
            clusters.append([entries[k] for k in cluster_idxs])

    return clusters


def _distill_cluster(entries: list[dict], memory) -> Optional[dict]:
    from agents import call_llm

    combined = "\n\n".join(
        f"Topic: {e.get('topic', '?')}\n"
        f"Verdict: {e.get('verdict', '')[:400]}"
        for e in entries[:8]   # cap at 8 to stay within context
    )
    prompt = (
        f"You have seen {len(entries)} debates on related topics. "
        f"Here are the verdicts:\n\n{combined}\n\n"
        f"Write a distilled insight (3-5 sentences) for future reference."
    )

    insight_text = call_llm(
        _DISTILL_SYSTEM, prompt,
        max_tokens=300, enable_thinking=False, temperature=0.3,
    )
    if insight_text.startswith("[") and len(insight_text) < 200:
        return None

    # Embed the insight itself for search
    topic_summary = "; ".join(e.get("topic", "")[:50] for e in entries[:4])
    search_text = topic_summary + ". " + insight_text[:200]
    with memory._encode_lock:
        enc = memory._get_encoder()
        emb = enc.encode(search_text, convert_to_numpy=True, normalize_embeddings=True)

    return {
        "topics": [e.get("topic", "") for e in entries],
        "insight": insight_text,
        "source_count": len(entries),
        "embedding": emb.tolist(),
    }


# Module-level singleton
CONSOLIDATED_MEMORY = ConsolidatedMemory()
