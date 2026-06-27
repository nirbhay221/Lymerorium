"""Continuous background swarm - always watching, always debating, storing verdicts."""


import os
import queue as _queue
import sys
import threading
import time
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(__file__))

from memory import MEMORY   # shared vector memory


class BackgroundSwarm:
    def __init__(self, interval_seconds: int = 300):
        self.interval = interval_seconds
        self._lock = threading.Lock()
        self._busy = False
        self._paused = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._get_frame: Optional[Callable] = None
        self._describe: Optional[Callable] = None
        self._topic_source: Optional[Callable] = None   # text-only topic generator
        self._verdict_count = 0
        self._topic_queue: _queue.Queue = _queue.Queue()   # externally injected topics
        self._last_frame_description: str = ""             # temporal diff (Level #6)

    # Lifecycle

    def start(self,
              get_frame_fn: Optional[Callable] = None,
              describe_fn: Optional[Callable] = None,
              topic_source: Optional[Callable] = None):
        """
        Three operating modes:
          - Vision:   pass get_frame_fn + describe_fn (camera-driven topics)
          - Text:     pass topic_source returning a string (e.g. RSS / queue / static list)
          - Standby:  pass nothing - loop runs but skips cycles until a source is set
        """
        self._get_frame = get_frame_fn
        self._describe = describe_fn
        self._topic_source = topic_source
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        mode = (
            "vision" if (get_frame_fn and describe_fn)
            else "text" if topic_source
            else "standby"
        )
        print(f"[BackgroundSwarm] Started - mode={mode}, interval={self.interval}s")

    def stop(self):
        self._running = False

    # Internal loop

    def _loop(self):
        time.sleep(600)  # let camera connect and user interact first
        while self._running:
            if not self._busy and not self._paused:
                threading.Thread(target=self._run_once, daemon=True).start()
            time.sleep(self.interval)

    def push_topic(self, topic: str) -> None:
        """Inject a topic directly into the debate queue (POST /swarm_queue)."""
        self._topic_queue.put(topic.strip())
        print(f"[BackgroundSwarm] Topic queued: {topic[:60]}")

    def _run_once(self):
        with self._lock:
            if self._busy:
                return
            self._busy = True
        try:
            img = ""
            topic = None

            # Priority 0: externally injected topics (POST /swarm_queue) drain first
            if not self._topic_queue.empty():
                try:
                    topic = self._topic_queue.get_nowait()
                    print(f"[BackgroundSwarm] Dequeued injected topic: {topic[:60]}")
                except _queue.Empty:
                    pass

            if not topic:
                # Mode A: camera available - describe frame with temporal diff
                if self._get_frame and self._describe:
                    img = self._get_frame() or ""
                    if img:
                        current_desc = self._describe(img)
                        if current_desc and not current_desc.startswith(("Error", "[")):
                            if self._last_frame_description:
                                # LAVAD-style temporal diff: what changed between frames?
                                topic = (
                                    f"Scene change detected. Previously the camera showed: "
                                    f"{self._last_frame_description[:200]}. "
                                    f"Now it shows: {current_desc[:200]}. "
                                    f"What changed, why does it matter, and what does it imply?"
                                )
                                print(f"[BackgroundSwarm] Temporal diff topic: {topic[:80]}...")
                            else:
                                topic = current_desc
                            self._last_frame_description = current_desc
                    # Camera offline - fall through to topic_source if configured
                    if not topic and self._topic_source:
                        try:
                            topic = self._topic_source()
                            print(f"[BackgroundSwarm] Camera offline - using topic_source fallback")
                        except Exception as e:
                            print(f"[BackgroundSwarm] topic_source fallback error: {e}")
                # Mode B: text-only topic source (no camera configured)
                elif self._topic_source:
                    try:
                        topic = self._topic_source()
                    except Exception as e:
                        print(f"[BackgroundSwarm] topic_source error: {e}")
                        return
                # Mode C: standby - nothing to debate this tick
                else:
                    return

            if not topic or topic.startswith("Error") or topic.startswith("["):
                return

            print(f"[BackgroundSwarm] Debating: {topic[:70]}...")

            from api import run_simulation
            result = run_simulation(topic, max_rounds=2, image_b64=img)

            # Skip storing if this was a cache hit - verdict is already in MEMORY,
            # re-adding it would create a duplicate with a fresh timestamp every 300s
            if result.get("cache_hit"):
                print(f"[BackgroundSwarm] Cache hit for '{topic[:50]}' - skipping MEMORY.add")
                return

            verdict = result.get("verdict", "")
            if not verdict or verdict.startswith("VERDICT: Debate could not complete"):
                print(f"[BackgroundSwarm] No useful verdict for '{topic[:50]}' - skipping store")
                return

            entry = {
                "topic": topic,
                "entities": result.get("entities", []),
                "verdict": verdict,
                "convergence_score": result.get("convergence_score", 0.0),
                "timestamp": time.time(),
            }

            # Store in vector memory (semantic search replaces keyword search)
            MEMORY.add(entry)
            self._verdict_count = MEMORY.count
            print(f"[BackgroundSwarm] Stored verdict #{self._verdict_count}: {topic[:50]}")

            # Episodic Memory Consolidation: every 10 verdicts,
            # cluster high-quality debates and distill multi-debate insights.
            if self._verdict_count > 0 and self._verdict_count % 10 == 0:
                threading.Thread(target=self._consolidate, daemon=True).start()

        except Exception as e:
            print(f"[BackgroundSwarm] Error: {e}")
        finally:
            self._busy = False

    def _consolidate(self):
        try:
            from consolidation import CONSOLIDATED_MEMORY
            n = CONSOLIDATED_MEMORY.consolidate_from()
            print(f"[BackgroundSwarm] Consolidation complete: {n} insights written")
        except Exception as e:
            print(f"[BackgroundSwarm] Consolidation error: {e}")

    # Search - delegates to vector memory

    def search(self, query: str) -> Optional[dict]:
        """Semantic search over stored verdicts. Understands meaning not just keywords."""
        return MEMORY.search(query)

    # Pause / resume

    def pause(self, wait_for_current: bool = True, timeout: float = 10.0):
        """Pause next cycle. If wait_for_current=True, block until in-flight run finishes (max timeout s)."""
        self._paused = True
        if wait_for_current and self._busy:
            deadline = time.time() + timeout
            while self._busy and time.time() < deadline:
                time.sleep(0.2)

    def resume(self):
        self._paused = False

    # Status

    @property
    def status(self) -> dict:
        return {
            "running": self._running,
            "busy": self._busy,
            "paused": self._paused,
            "verdicts_stored": MEMORY.count,
            "last_topic": MEMORY.last_topic,
            "queue_size": self._topic_queue.qsize(),
        }


BACKGROUND_SWARM = BackgroundSwarm(interval_seconds=300)
