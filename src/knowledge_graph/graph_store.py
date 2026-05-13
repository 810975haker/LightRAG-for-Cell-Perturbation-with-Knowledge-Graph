from __future__ import annotations
from typing import Any, Dict, List
import math
import csv
from pathlib import Path
import re

import networkx as nx
try:
    from py2neo import Graph
except Exception:  # pragma: no cover - fallback when neo4j client is unavailable
    Graph = None

import config
import traceback

KG_BACKEND = config.KG_BACKEND
KG_HOST = config.KG_HOST
KG_PORT = config.KG_PORT
KG_USERNAME = config.KG_USERNAME
KG_PASSWORD = config.KG_PASSWORD
KG_NEO4J_BATCH_SIZE = int(getattr(config, "KG_NEO4J_BATCH_SIZE", 2000))
KG_RANK_MIN_SCORE = float(getattr(config, "KG_RANK_MIN_SCORE", 0.01))
KG_RANK_DISTANCE_DECAY = float(getattr(config, "KG_RANK_DISTANCE_DECAY", 0.78))
KG_RANK_PATHWAY_BOOST = float(getattr(config, "KG_RANK_PATHWAY_BOOST", 1.12))
KG_LOCAL_TRIPLES_CSV = str(getattr(config, "KG_LOCAL_TRIPLES_CSV", "data/processed/lung_cancer/kg_triples.csv"))


class KnowledgeGraphStore:
    def __init__(self):
        self.backend = KG_BACKEND.lower()
        self.local_graph = nx.MultiDiGraph()
        self.graph = None
        self._local_bootstrapped = False
        if self.backend == "neo4j" and Graph is not None:
            try:
                self.graph = Graph(
                    host=KG_HOST,
                    port=KG_PORT,
                    user=KG_USERNAME,
                    password=KG_PASSWORD,
                )
                # Fail fast when Neo4j is reachable but database is unavailable.
                self.graph.run("RETURN 1 AS ok").data()
            except Exception:
                self.backend = "networkx"
                self.graph = None
        elif self.backend == "neo4j" and Graph is None:
            self.backend = "networkx"

    @staticmethod
    def _infer_node_type(node_name: str) -> str:
        text = str(node_name or "")
        lower = text.lower()
        if lower.startswith("gene::"):
            return "gene"
        if lower.startswith("protein::"):
            return "protein"
        if lower.startswith("pathway::"):
            return "pathway"
        if lower.startswith("mirna::"):
            return "mirna"
        if lower.startswith("sample::"):
            return "sample"
        if lower.startswith("condition::"):
            return "condition"
        if lower.startswith("cell::"):
            return "Cell"
        return "entity"

    def _bootstrap_local_graph_from_triples(self):
        if self._local_bootstrapped:
            return
        self._local_bootstrapped = True
        csv_path = Path(KG_LOCAL_TRIPLES_CSV)
        if not csv_path.exists():
            return
        try:
            loaded = nx.MultiDiGraph()
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    head = str(row.get("head", "")).strip()
                    tail = str(row.get("tail", "")).strip()
                    if not head or not tail:
                        continue
                    if head not in loaded:
                        loaded.add_node(head, type=self._infer_node_type(head), is_seed=False)
                    if tail not in loaded:
                        loaded.add_node(tail, type=self._infer_node_type(tail), is_seed=False)
                    loaded.add_edge(
                        head,
                        tail,
                        relation=str(row.get("relation", "associated_with") or "associated_with"),
                        source=str(row.get("source", "unknown") or "unknown"),
                        version=str(row.get("version", "unknown") or "unknown"),
                        evidence=str(row.get("evidence", "") or ""),
                        weight=float(row.get("weight", 1.0) or 1.0),
                        confidence=float(row.get("confidence", 0.5) or 0.5),
                        effect_sign=float(row.get("effect_sign", 1.0) or 1.0),
                    )
            if loaded.number_of_edges() > 0:
                self.local_graph = loaded
        except Exception:
            pass

    def _ensure_available_graph(self):
        if self.backend == "networkx" and self.local_graph.number_of_edges() == 0:
            self._bootstrap_local_graph_from_triples()

    @staticmethod
    def _chunks(rows: List[Dict[str, Any]], batch_size: int):
        step = max(100, int(batch_size or 2000))
        for idx in range(0, len(rows), step):
            yield rows[idx: idx + step]

    @staticmethod
    def _relation_effect_sign(relation: str, fallback: float = 1.0) -> float:
        text = str(relation or "").lower()
        if any(token in text for token in ["inhibit", "suppress", "down", "decrease", "repress", "block", "negative"]):
            return -1.0
        if any(token in text for token in ["activ", "enhanc", "increase", "up", "induce", "promote", "positive"]):
            return 1.0
        return float(fallback)

    @staticmethod
    def _relation_to_neo4j_type(relation: str) -> str:
        text = str(relation or "").strip()
        if not text:
            return "RELATED_TO"
        normalized = re.sub(r"[^A-Za-z0-9_]", "_", text).upper()
        normalized = normalized.strip("_") or "RELATED_TO"
        if normalized[0].isdigit():
            normalized = "REL_{}".format(normalized)
        return normalized

    def _ensure_constraints(self):
        if self.graph is None:
            return
        try:
            self.graph.run(
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )
            return
        except Exception:
            pass
        # Neo4j 4.x compatible fallback.
        try:
            self.graph.run("CREATE CONSTRAINT ON (e:Entity) ASSERT e.name IS UNIQUE")
        except Exception:
            pass

    def _reset_schema(self):
        if self.graph is None:
            return
        try:
            constraints = self.graph.run("SHOW CONSTRAINTS").data()
            for row in constraints:
                name = str(row.get("name", "")).strip()
                if name:
                    self.graph.run("DROP CONSTRAINT `{}` IF EXISTS".format(name))
        except Exception:
            pass
        try:
            indexes = self.graph.run("SHOW INDEXES").data()
            for row in indexes:
                name = str(row.get("name", "")).strip()
                if name:
                    self.graph.run("DROP INDEX `{}` IF EXISTS".format(name))
        except Exception:
            pass

    def _clear_graph_in_batches(self, batch_size: int = 5000):
        if self.graph is None:
            return
        size = max(500, int(batch_size))
        # 先删关系（轻量），再删节点，避免 DETACH DELETE 内存爆栈
        while True:
            result = self.graph.run(
                "MATCH ()-[r]->() WITH r LIMIT $limit DELETE r RETURN count(r) AS deleted",
                limit=size,
            ).data()
            if not result or int((result[0].get("deleted") if result else 0) or 0) == 0:
                break
        while True:
            result = self.graph.run(
                "MATCH (n) WITH n LIMIT $limit DELETE n RETURN count(n) AS deleted",
                limit=size,
            ).data()
            if not result or int((result[0].get("deleted") if result else 0) or 0) == 0:
                break

    def _save_graph_neo4j_batch(self, nx_graph):
        if self.graph is None:
            return
        self._ensure_constraints()

        node_rows: List[Dict[str, Any]] = []
        for node_name, attrs in nx_graph.nodes(data=True):
            node_rows.append(
                {
                    "name": str(node_name),
                    "type": str(attrs.get("type", "Cell")),
                    "is_seed": bool(attrs.get("is_seed", False)),
                    "pathway_name": str(attrs.get("pathway_name", "")),
                }
            )

        node_query = (
            "UNWIND $rows AS row "
            "MERGE (n:Entity {name: row.name}) "
            "SET n.type = row.type, "
            "n.is_seed = (coalesce(n.is_seed, false) OR row.is_seed), "
            "n.pathway_name = CASE WHEN row.pathway_name <> '' THEN row.pathway_name ELSE n.pathway_name END"
        )
        for chunk in self._chunks(node_rows, KG_NEO4J_BATCH_SIZE):
            self.graph.run(node_query, rows=chunk)

        edge_rows: List[Dict[str, Any]] = []
        for head, tail, data in nx_graph.edges(data=True):
            inferred_effect_sign = self._relation_effect_sign(data.get("relation", ""), fallback=1.0)
            edge_rows.append(
                {
                    "head": str(head),
                    "tail": str(tail),
                    "relation": str(data.get("relation", "associated_with")),
                    "source": str(data.get("source", "unknown")),
                    "version": str(data.get("version", "unknown")),
                    "evidence": str(data.get("evidence", "")),
                    "weight": float(data.get("weight", 1.0) or 1.0),
                    "confidence": float(data.get("confidence", 0.5) or 0.5),
                    "effect_sign": float(data.get("effect_sign", inferred_effect_sign) or inferred_effect_sign),
                }
            )

        grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
        for row in edge_rows:
            rel_type = self._relation_to_neo4j_type(row.get("relation", ""))
            grouped_rows.setdefault(rel_type, []).append(row)

        for rel_type, rows in grouped_rows.items():
            edge_query = (
                "UNWIND $rows AS row "
                "MATCH (h:Entity {name: row.head}) "
                "MATCH (t:Entity {name: row.tail}) "
                "CREATE (h)-[:" + rel_type + " {"
                "relation: row.relation, source: row.source, version: row.version, "
                "evidence: row.evidence, weight: row.weight, confidence: row.confidence, effect_sign: row.effect_sign"
                "}]->(t)"
            )
            for chunk in self._chunks(rows, KG_NEO4J_BATCH_SIZE):
                self.graph.run(edge_query, rows=chunk)

    def _save_node_rows(self, node_rows: List[Dict[str, Any]]) -> None:
        if self.graph is None or not node_rows:
            return
        node_query = (
            "UNWIND $rows AS row "
            "MERGE (n:Entity {name: row.name}) "
            "SET n.type = row.type, "
            "n.is_seed = (coalesce(n.is_seed, false) OR row.is_seed), "
            "n.pathway_name = CASE WHEN row.pathway_name <> '' THEN row.pathway_name ELSE n.pathway_name END"
        )
        for chunk in self._chunks(node_rows, KG_NEO4J_BATCH_SIZE):
            self.graph.run(node_query, rows=chunk)

    def _save_pathway_rows(self, pathway_rows: List[Dict[str, Any]]) -> None:
        if self.graph is None or not pathway_rows:
            return
        pathway_query = (
            "UNWIND $rows AS row "
            "MERGE (n:Entity {name: row.name}) "
            "SET n.pathway_name = CASE WHEN row.pathway_name <> '' THEN row.pathway_name ELSE n.pathway_name END"
        )
        for chunk in self._chunks(pathway_rows, KG_NEO4J_BATCH_SIZE):
            self.graph.run(pathway_query, rows=chunk)

    def _save_edge_rows(self, edge_rows: List[Dict[str, Any]]) -> None:
        if self.graph is None or not edge_rows:
            return
        grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
        for row in edge_rows:
            rel_type = self._relation_to_neo4j_type(row.get("relation", ""))
            grouped_rows.setdefault(rel_type, []).append(row)

        for rel_type, rows in grouped_rows.items():
            edge_query = (
                "UNWIND $rows AS row "
                "MATCH (h:Entity {name: row.head}) "
                "MATCH (t:Entity {name: row.tail}) "
                "CREATE (h)-[:" + rel_type + " {"
                "relation: row.relation, source: row.source, version: row.version, "
                "evidence: row.evidence, weight: row.weight, confidence: row.confidence, effect_sign: row.effect_sign"
                "}]->(t)"
            )
            for chunk in self._chunks(rows, KG_NEO4J_BATCH_SIZE):
                self.graph.run(edge_query, rows=chunk)

    def save_triples_streaming(self, triple_rows, replace: bool = False) -> None:
        if self.backend != "neo4j" or self.graph is None:
            return
        if replace:
            self._reset_schema()
            self._clear_graph_in_batches(batch_size=5000)
        self._ensure_constraints()

        node_rows: List[Dict[str, Any]] = []
        edge_rows: List[Dict[str, Any]] = []
        pathway_rows: List[Dict[str, Any]] = []

        for row in triple_rows:
            if row.get("node_update"):
                pathway_rows.append(
                    {
                        "name": str(row.get("node_name", "")),
                        "pathway_name": str(row.get("pathway_name", "")),
                    }
                )
                if len(pathway_rows) >= KG_NEO4J_BATCH_SIZE:
                    self._save_pathway_rows(pathway_rows)
                    pathway_rows = []
                continue

            head = str(row.get("head", ""))
            tail = str(row.get("tail", ""))
            if not head or not tail:
                continue

            node_rows.append(
                {
                    "name": head,
                    "type": str(row.get("head_type", "Cell")),
                    "is_seed": bool(row.get("head_is_seed", False)),
                    "pathway_name": "",
                }
            )
            node_rows.append(
                {
                    "name": tail,
                    "type": str(row.get("tail_type", "Cell")),
                    "is_seed": bool(row.get("tail_is_seed", False)),
                    "pathway_name": "",
                }
            )

            edge_rows.append(
                {
                    "head": head,
                    "tail": tail,
                    "relation": str(row.get("relation", "associated_with")),
                    "source": str(row.get("source", "unknown")),
                    "version": str(row.get("version", "unknown")),
                    "evidence": str(row.get("evidence", "")),
                    "weight": float(row.get("weight", 1.0) or 1.0),
                    "confidence": float(row.get("confidence", 0.5) or 0.5),
                    "effect_sign": float(row.get("effect_sign", 0.0) or 0.0),
                }
            )

            if len(edge_rows) >= KG_NEO4J_BATCH_SIZE:
                self._save_node_rows(node_rows)
                self._save_edge_rows(edge_rows)
                node_rows = []
                edge_rows = []

        if node_rows:
            self._save_node_rows(node_rows)
        if edge_rows:
            self._save_edge_rows(edge_rows)
        if pathway_rows:
            self._save_pathway_rows(pathway_rows)

    def save_graph(self, nx_graph, replace: bool = False):
        """Persist graph to configured backend."""
        if replace:
            self.local_graph = nx_graph.copy()
        else:
            self.local_graph = nx.compose(self.local_graph, nx_graph)
        if self.backend != "neo4j" or self.graph is None:
            return
        try:
            if replace:
                self._reset_schema()
                self._clear_graph_in_batches(batch_size=5000)
            self._save_graph_neo4j_batch(nx_graph)
        except Exception as exc:
            print("[KG_SAVE_ERROR] Neo4j import failed, switching to networkx: {}".format(exc))
            traceback.print_exc()
            # Keep the service available even if Neo4j is temporarily unreachable.
            self.backend = "networkx"

    def query_graph(self, query):
        if self.backend == "neo4j" and self.graph is not None:
            return self.graph.run(query).data()
        raise NotImplementedError("query_graph is only available for neo4j backend")

    def get_entities(self):
        if self.backend == "neo4j" and self.graph is not None:
            return self.graph.nodes.match("Entity").all()
        return list(self.local_graph.nodes(data=True))

    def get_relations(self):
        if self.backend == "neo4j" and self.graph is not None:
            return self.graph.relationships.match().all()
        return list(self.local_graph.edges(data=True))

    def get_neighbors(self, node_id: str, depth: int = 1) -> List[dict]:
        self._ensure_available_graph()
        if self.backend == "neo4j" and self.graph is not None:
            try:
                max_depth = max(1, int(depth))
                query = (
                    "MATCH p=(s:Entity {name:$node_id})-[r*1.." + str(max_depth) + "]->(n:Entity) "
                    "UNWIND relationships(p) AS rel "
                    "WITH DISTINCT startNode(rel) AS a, endNode(rel) AS b, rel "
                    "RETURN a.name AS from_name, b.name AS to_name, "
                    "coalesce(rel.relation, 'associated_with') AS relation, "
                    "coalesce(rel.weight, 1.0) AS score, "
                    "coalesce(rel.weight, 1.0) AS weight, "
                    "coalesce(rel.confidence, 0.5) AS confidence, "
                    "coalesce(rel.effect_sign, "
                    "CASE "
                    "WHEN toLower(coalesce(rel.relation,'')) CONTAINS 'inhibit' OR toLower(coalesce(rel.relation,'')) CONTAINS 'suppress' OR toLower(coalesce(rel.relation,'')) CONTAINS 'repress' THEN -1.0 "
                    "WHEN toLower(coalesce(rel.relation,'')) CONTAINS 'activ' OR toLower(coalesce(rel.relation,'')) CONTAINS 'enhanc' OR toLower(coalesce(rel.relation,'')) CONTAINS 'increase' OR toLower(coalesce(rel.relation,'')) CONTAINS 'promote' THEN 1.0 "
                    "ELSE 1.0 END) AS effect_sign "
                    "LIMIT 2000"
                )
                rows = self.graph.run(query, node_id=node_id).data()
                return [
                    {
                        "from": row.get("from_name", ""),
                        "to": row.get("to_name", ""),
                        "relation": row.get("relation", "associated_with"),
                        "score": float(row.get("score", 1.0) or 1.0),
                        "weight": float(row.get("weight", 1.0) or 1.0),
                        "confidence": float(row.get("confidence", 0.5) or 0.5),
                        "effect_sign": float(row.get("effect_sign", 1.0) or 1.0),
                        "signal": round(float(row.get("weight", 1.0) or 1.0) * float(row.get("confidence", 0.5) or 0.5), 6),
                    }
                    for row in rows
                ]
            except Exception:
                pass
        if node_id not in self.local_graph:
            return []
        neighbors = []
        frontier = {node_id}
        visited = {node_id}
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for _, neighbor, edge_data in self.local_graph.out_edges(node, data=True):
                    weight = float(edge_data.get("weight", 1.0) or 1.0)
                    confidence = float(edge_data.get("confidence", 0.5) or 0.5)
                    effect_sign = float(edge_data.get("effect_sign", self._relation_effect_sign(edge_data.get("relation", ""), 1.0)) or 1.0)
                    neighbors.append({
                        "from": node,
                        "to": neighbor,
                        "relation": edge_data.get("relation", "associated_with"),
                        "score": float(weight),
                        "weight": weight,
                        "confidence": confidence,
                        "effect_sign": effect_sign,
                        "signal": round(weight * confidence, 6),
                    })
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier
        return neighbors

    def stats(self) -> Dict[str, int]:
        self._ensure_available_graph()
        if self.backend == "neo4j" and self.graph is not None:
            try:
                node_data = self.graph.run("MATCH (n:Entity) RETURN count(n) AS c").data()
                edge_data = self.graph.run("MATCH ()-[r]->() RETURN count(r) AS c").data()
                return {
                    "entities": int((node_data[0].get("c") if node_data else 0) or 0),
                    "relations": int((edge_data[0].get("c") if edge_data else 0) or 0),
                }
            except Exception:
                pass
        return {
            "entities": self.local_graph.number_of_nodes(),
            "relations": self.local_graph.number_of_edges(),
        }

    def visualization_data(self, max_nodes: int = 80, max_edges: int = 200) -> Dict[str, List[Dict]]:
        self._ensure_available_graph()
        if self.backend == "neo4j" and self.graph is not None:
            try:
                edge_rows = self.graph.run(
                    "MATCH (a:Entity)-[r]->(b:Entity) "
                    "RETURN a.name AS from_name, b.name AS to_name, "
                    "coalesce(a.type, 'entity') AS from_type, coalesce(b.type, 'entity') AS to_type, "
                    "coalesce(a.is_seed, false) AS from_seed, coalesce(b.is_seed, false) AS to_seed, "
                    "coalesce(r.relation, 'associated_with') AS relation, "
                    "coalesce(r.weight, 1.0) AS weight, "
                    "coalesce(r.confidence, 0.5) AS confidence "
                    "LIMIT $max_edges",
                    max_edges=int(max_edges),
                ).data()

                nodes_map: Dict[str, Dict[str, Any]] = {}
                edges: List[Dict[str, Any]] = []
                for row in edge_rows:
                    from_name = str(row.get("from_name", ""))
                    to_name = str(row.get("to_name", ""))
                    if not from_name or not to_name:
                        continue
                    if from_name not in nodes_map and len(nodes_map) < int(max_nodes):
                        nodes_map[from_name] = {
                            "id": from_name,
                            "label": from_name,
                            "type": str(row.get("from_type", "entity")),
                            "is_seed": bool(row.get("from_seed", False)),
                        }
                    if to_name not in nodes_map and len(nodes_map) < int(max_nodes):
                        nodes_map[to_name] = {
                            "id": to_name,
                            "label": to_name,
                            "type": str(row.get("to_type", "entity")),
                            "is_seed": bool(row.get("to_seed", False)),
                        }
                    if from_name in nodes_map and to_name in nodes_map:
                        edges.append(
                            {
                                "from": from_name,
                                "to": to_name,
                                "relation": str(row.get("relation", "associated_with")),
                                "weight": float(row.get("weight", 1.0) or 1.0),
                                "confidence": float(row.get("confidence", 0.5) or 0.5),
                            }
                        )

                if nodes_map:
                    return {"nodes": list(nodes_map.values()), "edges": edges}
            except Exception:
                pass
        nodes = []
        edges = []

        selected_nodes = list(self.local_graph.nodes(data=True))[:max_nodes]
        selected_names = {name for name, _ in selected_nodes}

        for node_name, attrs in selected_nodes:
            nodes.append({
                "id": node_name,
                "label": node_name,
                "type": attrs.get("type", "entity"),
                "is_seed": bool(attrs.get("is_seed", False)),
            })

        for head, tail, data in self.local_graph.edges(data=True):
            if len(edges) >= max_edges:
                break
            if head not in selected_names or tail not in selected_names:
                continue
            edges.append({
                "from": head,
                "to": tail,
                "relation": data.get("relation", "associated_with"),
                "weight": float(data.get("weight", 1.0) or 1.0),
                "confidence": float(data.get("confidence", 0.5) or 0.5),
            })

        return {"nodes": nodes, "edges": edges}

    def subgraph_data(
        self,
        node_keyword: str = "",
        relation_keyword: str = "",
        query_mode: str = "and",
        view_mode: str = "stack",
        node_query: str = "",
        relation_query: str = "",
        match_mode: str = "and",
        max_nodes: int = 120,
        max_edges: int = 300,
    ) -> Dict[str, Any]:
        self._ensure_available_graph()
        node_q = str(node_keyword or node_query or "").strip().lower()
        rel_q = str(relation_keyword or relation_query or "").strip().lower()
        mode = str(query_mode or match_mode or "or").strip().lower()
        if mode not in {"and", "or"}:
            mode = "or"
        view = str(view_mode or "stack").strip().lower()
        if view not in {"stack", "replace"}:
            view = "stack"
        max_nodes = max(10, int(max_nodes or 120))
        max_edges = max(10, int(max_edges or 300))
        has_node = bool(node_q)
        has_rel = bool(rel_q)

        def _matched_by(node_hit: bool, rel_hit: bool) -> str:
            if node_hit and rel_hit:
                return "both"
            if node_hit:
                return "node"
            if rel_hit:
                return "relation"
            return "none"

        stats = {
            "node_hit_edges": 0,
            "relation_hit_edges": 0,
            "both_hit_edges": 0,
            "returned_edges": 0,
        }

        if self.backend == "neo4j" and self.graph is not None:
            try:
                edge_rows = self.graph.run(
                    "MATCH (a:Entity)-[r]->(b:Entity) "
                    "WITH a, b, r, "
                    "(toLower(a.name) CONTAINS $node_q OR toLower(b.name) CONTAINS $node_q) AS node_hit, "
                    "(toLower(coalesce(r.relation, '')) CONTAINS $rel_q) AS rel_hit "
                    "WHERE CASE WHEN $mode = 'or' "
                    "THEN (CASE WHEN $has_node AND $has_rel THEN (node_hit OR rel_hit) WHEN $has_node THEN node_hit WHEN $has_rel THEN rel_hit ELSE true END) "
                    "ELSE (CASE WHEN $has_node AND $has_rel THEN (node_hit AND rel_hit) WHEN $has_node THEN node_hit WHEN $has_rel THEN rel_hit ELSE true END) END "
                    "RETURN a.name AS from_name, b.name AS to_name, "
                    "coalesce(a.type, 'entity') AS from_type, coalesce(b.type, 'entity') AS to_type, "
                    "coalesce(a.is_seed, false) AS from_seed, coalesce(b.is_seed, false) AS to_seed, "
                    "coalesce(r.relation, 'associated_with') AS relation, "
                    "coalesce(r.weight, 1.0) AS weight, coalesce(r.confidence, 0.5) AS confidence, "
                    "node_hit, rel_hit "
                    "LIMIT $max_edges",
                    node_q=node_q,
                    rel_q=rel_q,
                    mode=mode,
                    has_node=has_node,
                    has_rel=has_rel,
                    max_edges=max_edges,
                ).data()

                nodes_map: Dict[str, Dict[str, Any]] = {}
                edges: List[Dict[str, Any]] = []
                for row in edge_rows:
                    from_name = str(row.get("from_name", ""))
                    to_name = str(row.get("to_name", ""))
                    if not from_name or not to_name:
                        continue
                    node_hit = bool(row.get("node_hit", False)) if has_node else False
                    rel_hit = bool(row.get("rel_hit", False)) if has_rel else False
                    if node_hit:
                        stats["node_hit_edges"] += 1
                    if rel_hit:
                        stats["relation_hit_edges"] += 1
                    if node_hit and rel_hit:
                        stats["both_hit_edges"] += 1
                    if len(nodes_map) < max_nodes or (from_name in nodes_map and to_name in nodes_map):
                        if from_name not in nodes_map and len(nodes_map) < max_nodes:
                            nodes_map[from_name] = {
                                "id": from_name,
                                "label": from_name,
                                "type": str(row.get("from_type", "entity")),
                                "is_seed": bool(row.get("from_seed", False)),
                            }
                        if to_name not in nodes_map and len(nodes_map) < max_nodes:
                            nodes_map[to_name] = {
                                "id": to_name,
                                "label": to_name,
                                "type": str(row.get("to_type", "entity")),
                                "is_seed": bool(row.get("to_seed", False)),
                            }
                        if from_name in nodes_map and to_name in nodes_map:
                            edges.append(
                                {
                                    "from": from_name,
                                    "to": to_name,
                                    "relation": str(row.get("relation", "associated_with")),
                                    "weight": float(row.get("weight", 1.0) or 1.0),
                                    "confidence": float(row.get("confidence", 0.5) or 0.5),
                                    "matched_by": _matched_by(node_hit, rel_hit),
                                }
                            )
                stats["returned_edges"] = len(edges)
                return {
                    "nodes": list(nodes_map.values()),
                    "edges": edges,
                    "meta": {
                        "query": {
                            "node_keyword": node_q,
                            "relation_keyword": rel_q,
                            "query_mode": mode,
                            "view_mode": view,
                        },
                        "match_stats": stats,
                        "compat": {
                            "node_query": node_q,
                            "relation_query": rel_q,
                            "match_mode": mode,
                        },
                    },
                }
            except Exception:
                pass

        nodes_map: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []
        for head, tail, data in self.local_graph.edges(data=True):
            rel = str(data.get("relation", "associated_with"))
            joined = "{} {}".format(str(head).lower(), str(tail).lower())
            node_hit = bool(has_node and (node_q in joined))
            rel_hit = bool(has_rel and (rel_q in rel.lower()))
            if mode == "or":
                if has_node and has_rel:
                    if not (node_hit or rel_hit):
                        continue
                elif has_node and not node_hit:
                    continue
                elif has_rel and not rel_hit:
                    continue
            else:
                if has_node and has_rel:
                    if not (node_hit and rel_hit):
                        continue
                elif has_node and not node_hit:
                    continue
                elif has_rel and not rel_hit:
                    continue
            if len(edges) >= max_edges:
                break

            if node_hit:
                stats["node_hit_edges"] += 1
            if rel_hit:
                stats["relation_hit_edges"] += 1
            if node_hit and rel_hit:
                stats["both_hit_edges"] += 1

            for node_name in (head, tail):
                if node_name not in nodes_map and len(nodes_map) < max_nodes:
                    attrs = self.local_graph.nodes.get(node_name, {})
                    nodes_map[node_name] = {
                        "id": node_name,
                        "label": node_name,
                        "type": attrs.get("type", "entity"),
                        "is_seed": bool(attrs.get("is_seed", False)),
                    }
            if head in nodes_map and tail in nodes_map:
                edges.append(
                    {
                        "from": head,
                        "to": tail,
                        "relation": rel,
                        "weight": float(data.get("weight", 1.0) or 1.0),
                        "confidence": float(data.get("confidence", 0.5) or 0.5),
                        "matched_by": _matched_by(node_hit, rel_hit),
                    }
                )

        stats["returned_edges"] = len(edges)
        return {
            "nodes": list(nodes_map.values()),
            "edges": edges,
            "meta": {
                "query": {
                    "node_keyword": node_q,
                    "relation_keyword": rel_q,
                    "query_mode": mode,
                    "view_mode": view,
                },
                "match_stats": stats,
                "compat": {
                    "node_query": node_q,
                    "relation_query": rel_q,
                    "match_mode": mode,
                },
            },
        }

    def one_hop_subgraph(self, node_id: str, max_edges: int = 300) -> Dict[str, List[Dict[str, Any]]]:
        self._ensure_available_graph()
        target = str(node_id or "").strip()
        if not target:
            return {"nodes": [], "edges": []}

        if self.backend == "neo4j" and self.graph is not None:
            try:
                rows = self.graph.run(
                    "MATCH (n:Entity {name:$node_id})-[r]-(m:Entity) "
                    "RETURN n.name AS n_name, coalesce(n.type, 'entity') AS n_type, coalesce(n.is_seed, false) AS n_seed, "
                    "m.name AS m_name, coalesce(m.type, 'entity') AS m_type, coalesce(m.is_seed, false) AS m_seed, "
                    "coalesce(r.relation, 'associated_with') AS relation, "
                    "coalesce(r.weight, 1.0) AS weight, coalesce(r.confidence, 0.5) AS confidence "
                    "LIMIT $max_edges",
                    node_id=target,
                    max_edges=max(10, int(max_edges or 300)),
                ).data()
                nodes_map: Dict[str, Dict[str, Any]] = {}
                edges: List[Dict[str, Any]] = []
                for row in rows:
                    n_name = str(row.get("n_name", ""))
                    m_name = str(row.get("m_name", ""))
                    if not n_name or not m_name:
                        continue
                    if n_name not in nodes_map:
                        nodes_map[n_name] = {
                            "id": n_name,
                            "label": n_name,
                            "type": str(row.get("n_type", "entity")),
                            "is_seed": bool(row.get("n_seed", False)),
                        }
                    if m_name not in nodes_map:
                        nodes_map[m_name] = {
                            "id": m_name,
                            "label": m_name,
                            "type": str(row.get("m_type", "entity")),
                            "is_seed": bool(row.get("m_seed", False)),
                        }
                    edges.append(
                        {
                            "from": n_name,
                            "to": m_name,
                            "relation": str(row.get("relation", "associated_with")),
                            "weight": float(row.get("weight", 1.0) or 1.0),
                            "confidence": float(row.get("confidence", 0.5) or 0.5),
                        }
                    )
                return {"nodes": list(nodes_map.values()), "edges": edges}
            except Exception:
                pass

        if target not in self.local_graph:
            return {"nodes": [], "edges": []}

        nodes_map: Dict[str, Dict[str, Any]] = {}
        center_attrs = self.local_graph.nodes.get(target, {})
        nodes_map[target] = {
            "id": target,
            "label": target,
            "type": center_attrs.get("type", "entity"),
            "is_seed": bool(center_attrs.get("is_seed", False)),
        }
        edges: List[Dict[str, Any]] = []

        for head, tail, data in self.local_graph.out_edges(target, data=True):
            attrs = self.local_graph.nodes.get(tail, {})
            nodes_map[tail] = {
                "id": tail,
                "label": tail,
                "type": attrs.get("type", "entity"),
                "is_seed": bool(attrs.get("is_seed", False)),
            }
            edges.append(
                {
                    "from": head,
                    "to": tail,
                    "relation": str(data.get("relation", "associated_with")),
                    "weight": float(data.get("weight", 1.0) or 1.0),
                    "confidence": float(data.get("confidence", 0.5) or 0.5),
                }
            )
            if len(edges) >= max_edges:
                break

        if len(edges) < max_edges:
            for head, tail, data in self.local_graph.in_edges(target, data=True):
                attrs = self.local_graph.nodes.get(head, {})
                nodes_map[head] = {
                    "id": head,
                    "label": head,
                    "type": attrs.get("type", "entity"),
                    "is_seed": bool(attrs.get("is_seed", False)),
                }
                edges.append(
                    {
                        "from": head,
                        "to": tail,
                        "relation": str(data.get("relation", "associated_with")),
                        "weight": float(data.get("weight", 1.0) or 1.0),
                        "confidence": float(data.get("confidence", 0.5) or 0.5),
                    }
                )
                if len(edges) >= max_edges:
                    break

        return {"nodes": list(nodes_map.values()), "edges": edges}

    def get_ranked_candidates(
        self,
        node_id: str,
        max_hops: int = 3,
        min_score: float = KG_RANK_MIN_SCORE,
        include_prefixes: tuple[str, ...] = ("gene::", "pathway::", "protein::"),
    ) -> List[Dict[str, Any]]:
        self._ensure_available_graph()
        hops_limit = max(1, int(max_hops))
        best_by_node: Dict[str, Dict[str, Any]] = {}

        allowed_types = {
            str(prefix).replace("::", "").strip().lower()
            for prefix in (include_prefixes or ())
            if str(prefix).endswith("::")
        }

        def _register(node_name: str, node_type: str, hops: int, signal: float, direction_sign: float = 1.0):
            if not node_name or node_name == node_id:
                return
            if include_prefixes:
                has_prefix = any(str(node_name).startswith(p) for p in include_prefixes)
                if not has_prefix and str(node_type or "").strip().lower() not in allowed_types:
                    return
            decay = math.pow(max(0.01, KG_RANK_DISTANCE_DECAY), max(0, int(hops) - 1))
            score = float(signal) * decay
            if str(node_name).startswith("pathway::"):
                score *= max(0.1, KG_RANK_PATHWAY_BOOST)
            if score < float(min_score):
                return
            prev = best_by_node.get(node_name)
            payload = {
                "node": node_name,
                "type": node_type or "entity",
                "hops": int(hops),
                "raw_signal": round(float(signal), 6),
                "score": round(float(score), 6),
                "direction_sign": float(direction_sign or 1.0),
            }
            if prev is None or float(payload["score"]) > float(prev["score"]) or (
                float(payload["score"]) == float(prev["score"]) and int(payload["hops"]) < int(prev["hops"])
            ):
                best_by_node[node_name] = payload

        if self.backend == "neo4j" and self.graph is not None:
            try:
                query = (
                    "MATCH p=(s:Entity {name:$node_id})-[r*1.." + str(hops_limit) + "]->(n:Entity) "
                    "WITH n, length(p) AS hops, "
                    "reduce(sig=1.0, rel IN r | sig * coalesce(rel.weight, 1.0) * coalesce(rel.confidence, 0.5)) AS signal, "
                    "reduce(ds=1.0, rel IN r | ds * coalesce(rel.effect_sign, "
                    "CASE "
                    "WHEN toLower(coalesce(rel.relation,'')) CONTAINS 'inhibit' OR toLower(coalesce(rel.relation,'')) CONTAINS 'suppress' OR toLower(coalesce(rel.relation,'')) CONTAINS 'repress' THEN -1.0 "
                    "WHEN toLower(coalesce(rel.relation,'')) CONTAINS 'activ' OR toLower(coalesce(rel.relation,'')) CONTAINS 'enhanc' OR toLower(coalesce(rel.relation,'')) CONTAINS 'increase' OR toLower(coalesce(rel.relation,'')) CONTAINS 'promote' THEN 1.0 "
                    "ELSE 1.0 END)) AS direction_sign "
                    "RETURN n.name AS node_name, coalesce(n.type, 'entity') AS node_type, hops, signal, direction_sign "
                    "LIMIT 8000"
                )
                rows = self.graph.run(query, node_id=node_id).data()
                for row in rows:
                    _register(
                        node_name=str(row.get("node_name", "")),
                        node_type=str(row.get("node_type", "entity")),
                        hops=int(row.get("hops", 1) or 1),
                        signal=float(row.get("signal", 0.0) or 0.0),
                        direction_sign=float(row.get("direction_sign", 1.0) or 1.0),
                    )
                return sorted(best_by_node.values(), key=lambda x: (-float(x["score"]), int(x["hops"]), x["node"]))
            except Exception:
                pass

        if node_id not in self.local_graph:
            return []

        frontier: List[tuple[str, int, float, float]] = [(node_id, 0, 1.0, 1.0)]
        while frontier:
            current, hops, accum_signal, accum_direction = frontier.pop(0)
            if hops >= hops_limit:
                continue
            next_hops = hops + 1
            for _, neighbor, edge_data in self.local_graph.out_edges(current, data=True):
                weight = float(edge_data.get("weight", 1.0) or 1.0)
                confidence = float(edge_data.get("confidence", 0.5) or 0.5)
                edge_signal = max(0.0, weight * confidence)
                path_signal = accum_signal * edge_signal
                edge_sign = float(edge_data.get("effect_sign", self._relation_effect_sign(edge_data.get("relation", ""), 1.0)) or 1.0)
                path_direction = float(accum_direction) * edge_sign
                neighbor_type = str(self.local_graph.nodes[neighbor].get("type", "entity"))
                _register(neighbor, neighbor_type, next_hops, path_signal, path_direction)
                frontier.append((neighbor, next_hops, path_signal, path_direction))

        return sorted(best_by_node.values(), key=lambda x: (-float(x["score"]), int(x["hops"]), x["node"]))

    @staticmethod
    def _extract_gene_token(node_name: str) -> str:
        if not str(node_name).startswith("gene::"):
            return ""
        return str(node_name).split("::", 1)[-1]

    @staticmethod
    def _is_symbol_like(token: str) -> bool:
        text = str(token or "")
        if not text:
            return False
        has_alpha = any(ch.isalpha() for ch in text)
        return has_alpha and len(text) >= 2

    def resolve_gene_label(self, node_name: str) -> str:
        token = self._extract_gene_token(node_name)
        if not token:
            return str(node_name or "")
        if self._is_symbol_like(token):
            return token

        # Try to map ID-like gene nodes (e.g., NCBI/Ensembl IDs) back to symbol-like aliases.
        if self.backend == "neo4j" and self.graph is not None:
            try:
                rows = self.graph.run(
                    "MATCH (n:Entity {name:$node_name})-[r]-(m:Entity) "
                    "WHERE m.name STARTS WITH 'gene::' "
                    "RETURN m.name AS alias_name "
                    "LIMIT 20",
                    node_name=node_name,
                ).data()
                for row in rows:
                    alias = self._extract_gene_token(str(row.get("alias_name", "")))
                    if self._is_symbol_like(alias):
                        return alias
            except Exception:
                pass

        if node_name in self.local_graph:
            try:
                for _, neighbor, edge_data in self.local_graph.out_edges(node_name, data=True):
                    alias = self._extract_gene_token(str(neighbor))
                    if self._is_symbol_like(alias):
                        return alias
                for neighbor, _, edge_data in self.local_graph.in_edges(node_name, data=True):
                    alias = self._extract_gene_token(str(neighbor))
                    if self._is_symbol_like(alias):
                        return alias
            except Exception:
                pass

        return token

    def get_pathway_linked_genes(
        self,
        pathway_node: str,
        max_genes: int = 40,
        min_edge_signal: float = 0.02,
    ) -> List[Dict[str, Any]]:
        self._ensure_available_graph()
        limit = max(1, int(max_genes))
        min_sig = float(min_edge_signal)
        results: List[Dict[str, Any]] = []

        if self.backend == "neo4j" and self.graph is not None:
            try:
                rows = self.graph.run(
                    "MATCH (p:Entity {name:$pathway_node})-[r]-(g:Entity) "
                    "WHERE g.name STARTS WITH 'gene::' "
                    "RETURN g.name AS gene_name, "
                    "coalesce(r.weight, 1.0) AS w, "
                    "coalesce(r.confidence, 0.5) AS c, "
                    "coalesce(r.relation, 'associated_with') AS rel "
                    "LIMIT $limit",
                    pathway_node=pathway_node,
                    limit=limit * 3,
                ).data()
                for row in rows:
                    signal = float(row.get("w", 1.0) or 1.0) * float(row.get("c", 0.5) or 0.5)
                    if signal < min_sig:
                        continue
                    results.append(
                        {
                            "gene": str(row.get("gene_name", "")),
                            "signal": round(signal, 6),
                            "relation": str(row.get("rel", "associated_with")),
                        }
                    )
                results.sort(key=lambda x: (-float(x.get("signal", 0.0)), x.get("gene", "")))
                return results[:limit]
            except Exception:
                pass

        if pathway_node not in self.local_graph:
            return []

        for _, neighbor, edge_data in self.local_graph.out_edges(pathway_node, data=True):
            if not str(neighbor).startswith("gene::"):
                continue
            signal = float(edge_data.get("weight", 1.0) or 1.0) * float(
                edge_data.get("confidence", 0.5) or 0.5
            )
            if signal < min_sig:
                continue
            results.append({"gene": str(neighbor), "signal": round(signal, 6), "relation": str(edge_data.get("relation", "associated_with"))})

        for neighbor, _, edge_data in self.local_graph.in_edges(pathway_node, data=True):
            if not str(neighbor).startswith("gene::"):
                continue
            signal = float(edge_data.get("weight", 1.0) or 1.0) * float(
                edge_data.get("confidence", 0.5) or 0.5
            )
            if signal < min_sig:
                continue
            results.append({"gene": str(neighbor), "signal": round(signal, 6), "relation": str(edge_data.get("relation", "associated_with"))})

        uniq = {}
        for item in results:
            g = item.get("gene", "")
            prev = uniq.get(g)
            if prev is None or float(item.get("signal", 0.0)) > float(prev.get("signal", 0.0)):
                uniq[g] = item
        merged = sorted(uniq.values(), key=lambda x: (-float(x.get("signal", 0.0)), x.get("gene", "")))
        return merged[:limit]

    def get_gene_linked_pathways(
        self,
        gene_node: str,
        max_pathways: int = 20,
        min_edge_signal: float = 0.02,
    ) -> List[Dict[str, Any]]:
        self._ensure_available_graph()
        limit = max(1, int(max_pathways))
        min_sig = float(min_edge_signal)
        results: List[Dict[str, Any]] = []

        if self.backend == "neo4j" and self.graph is not None:
            try:
                rows = self.graph.run(
                    "MATCH (g:Entity {name:$gene_node})-[r]-(p:Entity) "
                    "WHERE p.name STARTS WITH 'pathway::' "
                    "RETURN p.name AS pathway_name, "
                    "coalesce(r.weight, 1.0) AS w, "
                    "coalesce(r.confidence, 0.5) AS c, "
                    "coalesce(r.relation, 'associated_with') AS rel "
                    "LIMIT $limit",
                    gene_node=gene_node,
                    limit=limit * 3,
                ).data()
                for row in rows:
                    signal = float(row.get("w", 1.0) or 1.0) * float(row.get("c", 0.5) or 0.5)
                    if signal < min_sig:
                        continue
                    results.append(
                        {
                            "pathway": str(row.get("pathway_name", "")),
                            "signal": round(signal, 6),
                            "relation": str(row.get("rel", "associated_with")),
                        }
                    )
                results.sort(key=lambda x: (-float(x.get("signal", 0.0)), x.get("pathway", "")))
                return results[:limit]
            except Exception:
                pass

        if gene_node not in self.local_graph:
            return []

        for _, neighbor, edge_data in self.local_graph.out_edges(gene_node, data=True):
            if not str(neighbor).startswith("pathway::"):
                continue
            signal = float(edge_data.get("weight", 1.0) or 1.0) * float(edge_data.get("confidence", 0.5) or 0.5)
            if signal < min_sig:
                continue
            results.append({"pathway": str(neighbor), "signal": round(signal, 6), "relation": str(edge_data.get("relation", "associated_with"))})

        for neighbor, _, edge_data in self.local_graph.in_edges(gene_node, data=True):
            if not str(neighbor).startswith("pathway::"):
                continue
            signal = float(edge_data.get("weight", 1.0) or 1.0) * float(edge_data.get("confidence", 0.5) or 0.5)
            if signal < min_sig:
                continue
            results.append({"pathway": str(neighbor), "signal": round(signal, 6), "relation": str(edge_data.get("relation", "associated_with"))})

        uniq = {}
        for item in results:
            p = item.get("pathway", "")
            prev = uniq.get(p)
            if prev is None or float(item.get("signal", 0.0)) > float(prev.get("signal", 0.0)):
                uniq[p] = item
        merged = sorted(uniq.values(), key=lambda x: (-float(x.get("signal", 0.0)), x.get("pathway", "")))
        return merged[:limit]

    def is_seed_gene(self, node_name: str) -> bool:
        self._ensure_available_graph()
        if not str(node_name).startswith("gene::"):
            return False
        if self.backend == "neo4j" and self.graph is not None:
            try:
                rows = self.graph.run(
                    "MATCH (n:Entity {name:$name}) RETURN coalesce(n.is_seed, false) AS is_seed LIMIT 1",
                    name=str(node_name),
                ).data()
                if rows:
                    return bool(rows[0].get("is_seed", False))
            except Exception:
                pass
        if node_name in self.local_graph:
            return bool(self.local_graph.nodes[node_name].get("is_seed", False))
        return False

    def get_filter_options(self, max_items: int = 200) -> Dict[str, List[str]]:
        self._ensure_available_graph()
        limit = max(10, int(max_items or 200))
        if self.backend == "neo4j" and self.graph is not None:
            try:
                node_rows = self.graph.run(
                    "MATCH (n:Entity) "
                    "WHERE n.name STARTS WITH 'gene::' OR n.name STARTS WITH 'protein::' OR n.name STARTS WITH 'pathway::' "
                    "RETURN DISTINCT n.name AS name "
                    "LIMIT $limit",
                    limit=limit,
                ).data()
                relation_rows = self.graph.run(
                    "MATCH ()-[r]->() "
                    "RETURN DISTINCT coalesce(r.relation, 'associated_with') AS relation "
                    "LIMIT $limit",
                    limit=limit,
                ).data()
                return {
                    "node_options": sorted([str(row.get("name", "")) for row in node_rows if row.get("name")]),
                    "relation_options": sorted([str(row.get("relation", "")) for row in relation_rows if row.get("relation")]),
                }
            except Exception:
                pass

        node_options = sorted(
            [
                str(node_name)
                for node_name, attrs in self.local_graph.nodes(data=True)
                if str(node_name).startswith(("gene::", "protein::", "pathway::"))
                or str(attrs.get("type", "")).lower() in {"gene", "protein", "pathway"}
            ]
        )[:limit]
        relation_options = sorted({str(data.get("relation", "associated_with")) for _, _, data in self.local_graph.edges(data=True)})[:limit]
        return {"node_options": node_options, "relation_options": relation_options}

