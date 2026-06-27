
import hashlib as _hashlib
import threading
import time as _time
import networkx as nx
from typing import Optional
from config import LOCAL_NEO4J_URI, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# In-memory graph - persists across simulation runs (never wiped, only grows)
_local_graph: nx.DiGraph = nx.DiGraph()
_opinion_seq: int = 0   # monotonic counter so opinion nodes are unique across debates
_MAX_OPINIONS_PER_AGENT = 30  # cap per agent to prevent unbounded graph growth

# RLock (reentrant) protects all reads AND writes to _local_graph.
# NetworkX DiGraph is not thread-safe; RLock lets the same thread re-enter
# nested calls (seed_topic, add_entity, add_relationship) without deadlocking.
_graph_lock = threading.RLock()


def topic_node_id(topic: str) -> str:
    """Stable unique node ID per topic - hash-derived so same topic reuses same node."""
    h = _hashlib.md5(topic.encode("utf-8")).hexdigest()[:10]
    return f"topic:{h}"

# Neo4j driver - optional, only if credentials provided
_driver = None


def _get_driver():
    global _driver
    if _driver is not None:
        return _driver
    from neo4j import GraphDatabase
    # Try local Neo4j first, then fall back to Jetson
    for uri, label in [(LOCAL_NEO4J_URI, "local"), (NEO4J_URI, "Jetson")]:
        if not uri:
            continue
        try:
            d = GraphDatabase.driver(uri, auth=(NEO4J_USER, NEO4J_PASSWORD))
            d.verify_connectivity()
            _driver = d
            print(f"[KG] Connected to Neo4j ({label}) at {uri}")
            return _driver
        except Exception as e:
            print(f"[KG] Neo4j ({label}) at {uri} unavailable: {e}")
    print("[KG] No Neo4j available - using in-memory graph only")
    return None


# Local (NetworkX) operations

def add_entity(name: str, entity_type: str = "concept", **props) -> None:
    with _graph_lock:
        _local_graph.add_node(name, type=entity_type, **props)
    _sync_node_to_neo4j(name, entity_type, props)


def add_relationship(src: str, rel: str, dst: str, **props) -> None:
    with _graph_lock:
        _local_graph.add_edge(src, dst, relation=rel, **props)
    _sync_edge_to_neo4j(src, rel, dst, props)


def add_agent_opinion(agent: str, topic_entity: str, opinion: str, round_num: int) -> None:
    global _opinion_seq
    with _graph_lock:
        _opinion_seq += 1
        opinion_id = f"{agent}:r{round_num}:{_opinion_seq}"
        _local_graph.add_node(opinion_id, type="opinion", agent=agent, text=opinion,
                              round=round_num, timestamp=_time.time())
        _local_graph.add_edge(agent, opinion_id, relation="expressed")
        _local_graph.add_edge(opinion_id, topic_entity, relation="about")

        # Evict oldest opinions per agent to keep graph bounded
        agent_opinions = [
            n for _, n, d in _local_graph.out_edges(agent, data=True)
            if d.get("relation") == "expressed"
        ]
        if len(agent_opinions) > _MAX_OPINIONS_PER_AGENT:
            oldest = sorted(
                agent_opinions,
                key=lambda n: _local_graph.nodes[n].get("timestamp", 0)
            )
            for old_id in oldest[:len(agent_opinions) - _MAX_OPINIONS_PER_AGENT]:
                _local_graph.remove_node(old_id)


def get_entity_context(entities: list[str]) -> str:
    with _graph_lock:
        lines = []
        for name in entities:
            if name in _local_graph:
                neighbors = [
                    n for n in _local_graph.neighbors(name)
                    if not str(n).startswith(("topic:", "opinion_"))
                    and ":" not in str(n)
                ]
                if neighbors:
                    lines.append(f"• {name}: related to {', '.join(str(n) for n in neighbors[:5])}")
        return "\n".join(lines) if lines else "No prior knowledge available."


def get_agent_history(agent: str) -> list[str]:
    with _graph_lock:
        opinions = []
        for _, node, data in _local_graph.out_edges(agent, data=True):
            if data.get("relation") == "expressed":
                node_data = _local_graph.nodes.get(node, {})
                if node_data.get("type") == "opinion":
                    opinions.append(f"Round {node_data['round']}: {node_data['text']}")
        return opinions


def query_neo4j(cypher: str) -> list[dict]:
    driver = _get_driver()
    if not driver:
        return []
    with driver.session() as session:
        result = session.run(cypher)
        return [dict(record) for record in result]


def seed_topic(topic: str, entities: list[str]) -> str:
    """Seed a per-topic node + entity relationships. Returns the unique topic node ID."""
    tid = topic_node_id(topic)
    with _graph_lock:
        _local_graph.add_node(tid, type="topic", text=topic, timestamp=_time.time())
        for entity in entities:
            _local_graph.add_node(entity, type="extracted")
            _local_graph.add_edge(tid, entity, relation="mentions")
    _sync_topic_to_neo4j(topic, entities)
    return tid


# Neo4j sync (fire-and-forget)

def _sync_node_to_neo4j(name: str, entity_type: str, props: dict) -> None:
    driver = _get_driver()
    if not driver:
        return
    try:
        with driver.session() as session:
            session.run(
                "MERGE (n:Entity {name: $name}) SET n.type = $type, n += $props",
                name=name, type=entity_type, props=props,
            )
    except Exception as e:
        print(f"[KG] Neo4j node sync failed for '{name}': {e}")


def _sync_edge_to_neo4j(src: str, rel: str, dst: str, props: dict) -> None:
    driver = _get_driver()
    if not driver:
        return
    try:
        with driver.session() as session:
            session.run(
                """
                MERGE (a:Entity {name: $src})
                MERGE (b:Entity {name: $dst})
                MERGE (a)-[r:RELATES {type: $rel}]->(b)
                SET r += $props
                """,
                src=src, dst=dst, rel=rel, props=props,
            )
    except Exception as e:
        print(f"[KG] Neo4j edge sync failed ({src} to {dst}): {e}")


def _sync_topic_to_neo4j(topic: str, entities: list[str]) -> None:
    driver = _get_driver()
    if not driver:
        return
    try:
        with driver.session() as session:
            session.run(
                "MERGE (t:Topic {text: $topic})",
                topic=topic,
            )
            for entity in entities:
                session.run(
                    """
                    MERGE (t:Topic {text: $topic})
                    MERGE (e:Entity {name: $entity})
                    MERGE (t)-[:MENTIONS]->(e)
                    """,
                    topic=topic, entity=entity,
                )
    except Exception as e:
        print(f"[KG] Neo4j topic sync failed: {e}")


def soft_reset() -> None:
    """
    Called between simulation runs.
    Keeps the accumulated graph, agent history, and Neo4j connection.
    The neo4j driver manages its own connection pool - no need to recycle it.
    """
    pass


def reset() -> None:
    """Full wipe - only use for testing. Destroys all accumulated knowledge."""
    global _driver, _opinion_seq
    with _graph_lock:
        _local_graph.clear()
        _opinion_seq = 0
    if _driver:
        try:
            _driver.close()
        except Exception:
            pass
        _driver = None


def close() -> None:
    global _driver
    if _driver:
        _driver.close()
        _driver = None
