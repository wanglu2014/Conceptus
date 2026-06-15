#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5: Knowledge Pump Module - Enhanced Version

Generate Knowledge Pump context organized by Reach/Span/Scale dimensions.
This module merges functionality from:
  - knowledge_pump_module.py (base SBERT vectorization, clustering)
  - knowledge_pump_enhanced.py (metric-concept-metric relations, dimension classification)

Key Features:
  - Build metric-concept-metric relationship network
  - SBERT embedding and K-Means clustering
  - Dimension-based organization (Reach/Span/Scale/Cross)
  - Output: 5000-8000 character natural language context

Input Files:
    - outputs/step4_subgraph_info_{timestamp}.json
    - outputs/step3_seed_metrics_{timestamp}.json
    - GML knowledge graph (full or star subgraph)

Output Files:
    - outputs/step5_kp_context_{timestamp}.txt
    - outputs/step5_metric_only_graph_{timestamp}.gml
    - outputs/step5_kp_stats_{timestamp}.json

Usage:
    python step5_knowledge_pump.py --step3-output outputs/step3_seed_metrics_*.json

Author: Pipeline Modularization Project
Date: 2024-12
"""

import os
import sys
import json
import argparse
import traceback
import numpy as np
import networkx as nx
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field

# Import configuration
from config import CONFIG, DIMENSION_CATEGORIES, get_timestamp, get_output_path

# Import SBERT components
try:
    from knowledge_pump_module import SBERTVectorizer, EvidenceClusterer, Evidence
    SBERT_AVAILABLE = True
except ImportError:
    print("[Warning] knowledge_pump_module not found, some features disabled")
    SBERT_AVAILABLE = False


# ============================================================
# Dimension Classification (Unsupervised Clustering)
# ============================================================

# Global variables to store clustering results
_METRIC_CLUSTERS: Dict[str, int] = {}
_N_DIMENSION_CLUSTERS: int = 3  # Default number of clusters


def auto_discover_dimensions(seed_metrics: List[str],
                              n_clusters: int = 3,
                              model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> Dict[str, int]:
    """
    Auto-discover metric dimensions using SBERT embedding + K-Means clustering

    Args:
        seed_metrics: List of seed metric names
        n_clusters: Number of clusters (default 3)
        model_name: SBERT model name

    Returns:
        Dict[metric_name, cluster_id]
    """
    if not seed_metrics:
        return {}

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
    except ImportError:
        print("[KP] Warning: sentence-transformers or sklearn not available, using fallback")
        return {m: 0 for m in seed_metrics}

    # 1. Load SBERT model
    print(f"[KP] Performing dimension clustering using SBERT (k={n_clusters})...")
    model = SentenceTransformer(model_name)

    # 2. Encode metric names (replace underscores with spaces for better semantic understanding)
    metric_texts = [m.replace('_', ' ') for m in seed_metrics]
    embeddings = model.encode(metric_texts, show_progress_bar=False)

    # 3. K-Means clustering
    actual_k = min(n_clusters, len(seed_metrics))
    kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # 4. Build mapping
    metric_to_cluster = {m: int(cluster_labels[i]) for i, m in enumerate(seed_metrics)}

    return metric_to_cluster


def init_metric_clusters(seed_metrics: List[str], n_clusters: int = 3):
    """
    Initialize metric dimension clustering

    Called once at the start of Step5 to perform SBERT clustering

    Args:
        seed_metrics: List of seed metrics
        n_clusters: Number of clusters
    """
    global _METRIC_CLUSTERS, _N_DIMENSION_CLUSTERS
    _N_DIMENSION_CLUSTERS = n_clusters
    _METRIC_CLUSTERS = auto_discover_dimensions(seed_metrics, n_clusters)

    if _METRIC_CLUSTERS:
        print(f"[KP] Auto-discovered {n_clusters} semantic dimensions:")
        for c in range(n_clusters):
            metrics_in_c = [m for m, cid in _METRIC_CLUSTERS.items() if cid == c]
            preview = metrics_in_c[:5]
            suffix = f"... (+{len(metrics_in_c)-5})" if len(metrics_in_c) > 5 else ""
            print(f"    cluster_{c}: {preview}{suffix}")


def get_metric_dimension(metric_name: str) -> str:
    """
    Get the dimension of a metric (based on unsupervised clustering)

    Args:
        metric_name: Name of the metric

    Returns:
        str: Dimension name ('cluster_0', 'cluster_1', ..., or 'unknown')
    """
    if metric_name in _METRIC_CLUSTERS:
        return f"cluster_{_METRIC_CLUSTERS[metric_name]}"
    return 'unknown'


# ============================================================
# Data Classes
# ============================================================

@dataclass
class MetricRelation:
    """Represents a metric-concept-metric relationship"""
    metric_a: str
    metric_b: str
    concept: str
    concept_type: str
    relation_a_to_concept: str
    relation_concept_to_b: str
    description_a: str
    description_b: str
    path_nodes: List[str] = field(default_factory=list)
    dimension_a: str = ""
    dimension_b: str = ""
    embedding: Optional[np.ndarray] = None
    cluster_id: int = -1

    def __post_init__(self):
        if not self.dimension_a:
            self.dimension_a = get_metric_dimension(self.metric_a)
        if not self.dimension_b:
            self.dimension_b = get_metric_dimension(self.metric_b)

    def to_text(self) -> str:
        """Convert to text for SBERT encoding"""
        parts = [f"{self.metric_a} relates to {self.metric_b} through {self.concept}"]
        if self.description_a:
            parts.append(f": {self.relation_a_to_concept} - {self.description_a[:200]}")
        if self.description_b:
            parts.append(f"; {self.relation_concept_to_b} - {self.description_b[:200]}")
        return "".join(parts)

    def is_cross_dimension(self) -> bool:
        """Check if this is a cross-dimension relationship"""
        return (self.dimension_a != self.dimension_b and
                self.dimension_a != 'unknown' and
                self.dimension_b != 'unknown')


# ============================================================
# Metric Relation Builder
# ============================================================

class MetricRelationBuilder:
    """
    Build metric-concept-metric relationship network

    Unlike star graphs that only have seed->neighbor edges,
    this extracts paths: metric_A -> concept -> metric_B
    """

    def __init__(self, gml_path: str, matched_nodes: Dict[str, List]):
        """
        Args:
            gml_path: Path to the knowledge graph GML file
            matched_nodes: Mapping from metric names to node IDs
        """
        self.gml_path = gml_path
        self.matched_nodes = matched_nodes
        self.G = None
        self.relations: List[MetricRelation] = []

    def load_graph(self):
        """Load the knowledge graph"""
        print(f"[KP] Loading graph: {self.gml_path}")
        self.G = nx.read_gml(self.gml_path, label='id')
        print(f"[KP] Loading complete: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return self.G

    def build_1hop_relations(self, seed_metrics: List[str]) -> List[MetricRelation]:
        """
        Build metric relationships based on shared 1-hop neighbors (concepts)

        Logic:
        1. For each seed metric, get its direct neighbors
        2. For each pair of metrics, find common neighbors
        3. Create relationships from each common neighbor

        Args:
            seed_metrics: List of seed metrics

        Returns:
            List[MetricRelation]: List of relationships based on shared 1-hop neighbors
        """
        if self.G is None:
            self.load_graph()

        print(f"[KP] Building 1-hop relations for {len(seed_metrics)} seed metrics...")

        # Get 1-hop neighbors for each metric
        metric_neighbors = {}
        for metric in seed_metrics:
            if metric in self.matched_nodes:
                neighbors = set()
                nodes = self.matched_nodes[metric]
                if isinstance(nodes, list):
                    for node_id in nodes:
                        if node_id in self.G:
                            neighbors.update(self.G.neighbors(node_id))
                elif nodes in self.G:
                    neighbors.update(self.G.neighbors(nodes))
                metric_neighbors[metric] = neighbors

        print(f"[KP] Found neighbors for {len(metric_neighbors)} metrics")

        # Find common neighbors and create relationships
        relations = []
        metrics_list = list(metric_neighbors.keys())

        for i, metric_a in enumerate(metrics_list):
            for metric_b in metrics_list[i+1:]:
                common = metric_neighbors.get(metric_a, set()) & metric_neighbors.get(metric_b, set())
                for concept_node in list(common)[:5]:  # Limit to max 5 concepts per pair
                    relation = self._create_relation_from_common_neighbor(
                        metric_a, metric_b, concept_node
                    )
                    if relation:
                        relations.append(relation)

        print(f"[KP] Found {len(relations)} 1-hop relations")

        # New: Weighted sorting for cross-dimension relationships
        # Cross-dimension relationships have higher priority
        cross_dim_relations = [r for r in relations if r.is_cross_dimension()]
        same_dim_relations = [r for r in relations if not r.is_cross_dimension()]

        # Print cross-dimension statistics
        print(f"[KP] Cross-dimension relations: {len(cross_dim_relations)}, Same-dimension relations: {len(same_dim_relations)}")

        # Cross-dimension relationships come first
        relations = cross_dim_relations + same_dim_relations

        self.relations = relations
        return relations

    def _create_relation_from_common_neighbor(
        self,
        metric_a: str,
        metric_b: str,
        concept_node
    ) -> Optional[MetricRelation]:
        """Create MetricRelation from common neighbor, prioritizing edge description, fallback to node description"""
        if self.G is None:
            return None

        # Get concept information
        concept_data = self.G.nodes.get(concept_node, {})
        concept_label = concept_data.get('label', str(concept_node))
        concept_type = concept_data.get('type', 'concept')
        concept_description = concept_data.get('description', '')  # Node description as fallback

        # Find edge information
        edge_a_info = {'relation': 'related_to', 'description': ''}
        edge_b_info = {'relation': 'related_to', 'description': ''}

        # Edge from metric_a node to concept
        nodes_a = self.matched_nodes.get(metric_a, [])
        if not isinstance(nodes_a, list):
            nodes_a = [nodes_a]

        for node_a in nodes_a:
            if self.G.has_edge(node_a, concept_node):
                edge_data = self.G.edges[node_a, concept_node]
                edge_a_info = {
                    'relation': edge_data.get('relation', edge_data.get('type', 'related_to')),
                    'description': edge_data.get('description', '')
                }
                break
            elif self.G.has_edge(concept_node, node_a):
                edge_data = self.G.edges[concept_node, node_a]
                edge_a_info = {
                    'relation': edge_data.get('relation', edge_data.get('type', 'related_to')),
                    'description': edge_data.get('description', '')
                }
                break

        # Edge from metric_b node to concept
        nodes_b = self.matched_nodes.get(metric_b, [])
        if not isinstance(nodes_b, list):
            nodes_b = [nodes_b]

        for node_b in nodes_b:
            if self.G.has_edge(node_b, concept_node):
                edge_data = self.G.edges[node_b, concept_node]
                edge_b_info = {
                    'relation': edge_data.get('relation', edge_data.get('type', 'related_to')),
                    'description': edge_data.get('description', '')
                }
                break
            elif self.G.has_edge(concept_node, node_b):
                edge_data = self.G.edges[concept_node, node_b]
                edge_b_info = {
                    'relation': edge_data.get('relation', edge_data.get('type', 'related_to')),
                    'description': edge_data.get('description', '')
                }
                break

        # When edge description is empty, use node description as supplement
        desc_a = edge_a_info['description']
        desc_b = edge_b_info['description']

        if not desc_a and not desc_b and concept_description:
            # Build semantic description using concept node's description
            desc_a = f"{metric_a} relates to {concept_label}: {concept_description}"

        return MetricRelation(
            metric_a=metric_a,
            metric_b=metric_b,
            concept=concept_label,
            concept_type=concept_type,
            relation_a_to_concept=edge_a_info['relation'],
            relation_concept_to_b=edge_b_info['relation'],
            description_a=desc_a,
            description_b=desc_b,
            path_nodes=[metric_a, str(concept_node), metric_b],
        )

    def get_direct_edges(self, seed_metrics: List[str]) -> List[MetricRelation]:
        """Get direct edges between seed metrics (1-hop), prioritizing edge description, fallback to node description"""
        if self.G is None:
            self.load_graph()

        direct_relations = []

        # Get all seed node IDs
        all_seed_nodes = set()
        metric_node_map = {}
        for metric in seed_metrics:
            if metric in self.matched_nodes:
                nodes = self.matched_nodes[metric]
                if isinstance(nodes, list):
                    all_seed_nodes.update(nodes)
                    for node in nodes:
                        metric_node_map[node] = metric
                else:
                    all_seed_nodes.add(nodes)
                    metric_node_map[nodes] = metric

        # Find direct edges between seed nodes
        for u, v, data in self.G.edges(data=True):
            if u in all_seed_nodes and v in all_seed_nodes:
                metric_u = metric_node_map.get(u)
                metric_v = metric_node_map.get(v)
                if metric_u and metric_v and metric_u != metric_v:
                    # Get edge description
                    edge_desc = data.get('description', '').strip()

                    # When edge description is empty, try to build semantic description using node description
                    if not edge_desc:
                        node_u_data = self.G.nodes.get(u, {})
                        node_v_data = self.G.nodes.get(v, {})
                        node_u_desc = node_u_data.get('description', '')
                        node_v_desc = node_v_data.get('description', '')

                        if node_u_desc:
                            edge_desc = f"{metric_u}: {node_u_desc}"
                        elif node_v_desc:
                            edge_desc = f"{metric_v}: {node_v_desc}"
                        else:
                            # Build basic description
                            relation_type = data.get('relation', data.get('type', 'related_to'))
                            edge_desc = f"{metric_u} is {relation_type} to {metric_v} in network topology"

                    relation = MetricRelation(
                        metric_a=metric_u,
                        metric_b=metric_v,
                        concept="direct_connection",
                        concept_type="direct",
                        relation_a_to_concept=data.get('relation', data.get('type', 'related_to')),
                        relation_concept_to_b="",
                        description_a=edge_desc,
                        description_b="",
                        path_nodes=[str(u), str(v)],
                    )
                    direct_relations.append(relation)

        print(f"[KP] Found {len(direct_relations)} direct metric-metric edges")
        return direct_relations


# ============================================================
# Enhanced Clustering
# ============================================================

class EnhancedClusterer:
    """
    Enhanced Clusterer

    Features:
    - 15-20 clusters (vs 5 in V3)
    - 2-3 representatives per cluster (vs 1)
    """

    def __init__(
        self,
        n_clusters: int = 15,
        representatives_per_cluster: int = 2
    ):
        self.n_clusters = n_clusters
        self.representatives_per_cluster = representatives_per_cluster
        self.vectorizer = None
        self.clusters: Dict[int, List[MetricRelation]] = {}
        self.representatives: List[MetricRelation] = []

    def cluster_relations(
        self,
        relations: List[MetricRelation]
    ) -> Dict[int, List[MetricRelation]]:
        """
        Cluster relationships

        Args:
            relations: List of MetricRelation

        Returns:
            Dict[int, List[MetricRelation]]: Clustering results
        """
        if not relations:
            return {}

        if not SBERT_AVAILABLE:
            print("[KP] SBERT not available, skipping clustering")
            return {0: relations}

        print(f"[KP] SBERT clustering on {len(relations)} relations...")

        # Vectorization
        if self.vectorizer is None:
            self.vectorizer = SBERTVectorizer()

        texts = [rel.to_text() for rel in relations]

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(CONFIG.get("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
            embeddings = model.encode(texts, show_progress_bar=False)

            # Assign embeddings
            for rel, emb in zip(relations, embeddings):
                rel.embedding = emb

        except Exception as e:
            print(f"[KP] SBERT encoding failed: {e}")
            return {0: relations}

        # K-Means clustering
        from sklearn.cluster import KMeans

        n_clusters = min(self.n_clusters, len(relations))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Group by cluster
        clusters = defaultdict(list)
        for rel, label in zip(relations, labels):
            rel.cluster_id = int(label)
            clusters[label].append(rel)

        self.clusters = dict(clusters)
        print(f"[KP] Created {len(self.clusters)} clusters")

        return self.clusters

    def get_representatives(self) -> List[MetricRelation]:
        """Get representative relationships for each cluster"""
        representatives = []

        for cluster_id, rels in self.clusters.items():
            if not rels:
                continue

            # Get embeddings
            embeddings_with_idx = [
                (i, rel.embedding) for i, rel in enumerate(rels)
                if rel.embedding is not None
            ]

            if not embeddings_with_idx:
                representatives.append(rels[0])
                continue

            # Calculate cluster centroid
            emb_array = np.vstack([e[1] for e in embeddings_with_idx])
            centroid = np.mean(emb_array, axis=0)

            # Find relationships closest to centroid
            distances = [
                (idx, np.linalg.norm(emb - centroid))
                for idx, emb in embeddings_with_idx
            ]
            distances.sort(key=lambda x: x[1])

            # Select top-k representatives
            for i in range(min(self.representatives_per_cluster, len(distances))):
                idx = distances[i][0]
                representatives.append(rels[idx])

        self.representatives = representatives
        print(f"[KP] Selected {len(representatives)} representative relations")
        return representatives


# ============================================================
# Context Formatter
# ============================================================

class ContextFormatter:
    """
    Format relationships into natural language context organized by dimensions
    Supports dynamic clustering dimensions (unsupervised discovery)
    """

    def __init__(
        self,
        max_chars: int = 8000,
        min_chars: int = 5000
    ):
        self.max_chars = max_chars
        self.min_chars = min_chars

    def format_context(
        self,
        relations: List[MetricRelation],
        include_cross_dimension: bool = True
    ) -> str:
        """
        Format relationships into natural language context

        Args:
            relations: List of MetricRelation
            include_cross_dimension: Whether to include cross-dimension relationships

        Returns:
            str: Formatted context text
        """
        # Group by dimension (dynamic dimensions)
        dimension_groups = defaultdict(list)
        cross_relations = []

        for rel in relations:
            if rel.is_cross_dimension():
                cross_relations.append(rel)
            else:
                # Use dimension_a as grouping key
                dimension_groups[rel.dimension_a].append(rel)

        # Build context
        sections = []

        # Output each group by dimension
        for dim_name in sorted(dimension_groups.keys()):
            rels = dimension_groups[dim_name]
            if rels:
                # Generate dimension description
                metrics_in_dim = set()
                for r in rels:
                    metrics_in_dim.add(r.metric_a)
                    metrics_in_dim.add(r.metric_b)
                metrics_preview = list(metrics_in_dim)[:5]

                section = self._format_dimension_section(
                    f"Dimension {dim_name}",
                    f"Metrics in this semantic cluster: {', '.join(metrics_preview)}{'...' if len(metrics_in_dim) > 5 else ''}",
                    rels
                )
                sections.append(section)

        # Cross-dimension relationships
        if include_cross_dimension and cross_relations:
            section = self._format_dimension_section(
                "Cross-Dimension Relations",
                "Important relationships connecting metrics from different semantic clusters - key for discovering synergistic combinations",
                cross_relations
            )
            sections.append(section)

        context = "\n\n".join(sections)

        # Check length
        if len(context) > self.max_chars:
            context = self._truncate_context(context)

        return context

    def _format_dimension_section(
        self,
        title: str,
        description: str,
        relations: List[MetricRelation]
    ) -> str:
        """Format a single dimension section"""
        lines = [f"## {title}", description, ""]

        for i, rel in enumerate(relations[:10], 1):  # Max 10 per dimension
            line = self._format_relation(rel, i)
            lines.append(line)

        if len(relations) > 10:
            lines.append(f"... and {len(relations) - 10} more relations")

        return "\n".join(lines)

    def _format_relation(self, rel: MetricRelation, idx: int) -> str:
        """Format a single relationship"""
        parts = [f"{idx}. {rel.metric_a} <-> {rel.metric_b}"]

        if rel.concept and rel.concept != "direct_connection":
            parts.append(f" (via {rel.concept})")

        parts.append(f" [{rel.dimension_a}/{rel.dimension_b}]")

        desc = rel.description_a or rel.description_b
        if desc:
            desc_short = desc[:150] + "..." if len(desc) > 150 else desc
            parts.append(f"\n   {desc_short}")

        return "".join(parts)

    def _truncate_context(self, context: str) -> str:
        """Truncate context to maximum length"""
        if len(context) <= self.max_chars:
            return context

        # Split by section
        sections = context.split("\n\n")

        # Keep header, truncate content
        result = []
        current_len = 0

        for section in sections:
            if current_len + len(section) + 2 <= self.max_chars:
                result.append(section)
                current_len += len(section) + 2
            else:
                # Truncate this section
                remaining = self.max_chars - current_len - 50
                if remaining > 100:
                    truncated = section[:remaining] + "\n... (truncated)"
                    result.append(truncated)
                break

        return "\n\n".join(result)


# ============================================================
# Main Knowledge Pump Class
# ============================================================

class KnowledgePump:
    """
    Enhanced Knowledge Pump

    Usage:
        kp = KnowledgePump(gml_path, matched_nodes)
        context = kp.generate_context(seed_metrics)
        kp.save_results()
    """

    def __init__(
        self,
        gml_path: str,
        matched_nodes: Dict[str, List],
        config: Dict = None
    ):
        """Initialize Knowledge Pump"""
        self.gml_path = gml_path
        self.matched_nodes = matched_nodes
        self.config = config or CONFIG

        # Components
        self.relation_builder = MetricRelationBuilder(gml_path, matched_nodes)
        self.clusterer = EnhancedClusterer(
            n_clusters=self.config.get("ENHANCED_KP_N_CLUSTERS", 15),
            representatives_per_cluster=self.config.get("ENHANCED_KP_REPRESENTATIVES_PER_CLUSTER", 2)
        )
        self.formatter = ContextFormatter(
            max_chars=self.config.get("ENHANCED_KP_MAX_CHARS", 8000),
            min_chars=self.config.get("ENHANCED_KP_MIN_CHARS", 5000)
        )

        # Result storage
        self.relations: List[MetricRelation] = []
        self.representatives: List[MetricRelation] = []
        self.context: str = ""

        # Statistics
        self.stats: Dict[str, Any] = {
            "timestamp": None,
            "status": "not_started",
            "total_relations": 0,
            "clusters": 0,
            "representatives": 0,
            "context_length": 0,
            "dimension_breakdown": {}
        }

    def generate_context(
        self,
        seed_metrics: List[str],
        max_hop: int = None,
        include_cross_dimension: bool = True
    ) -> str:
        """
        Generate Knowledge Pump context

        Args:
            seed_metrics: List of seed metrics
            max_hop: Maximum hops (currently using 1-hop)
            include_cross_dimension: Whether to include cross-dimension relationships

        Returns:
            str: Formatted context text
        """
        print("\n" + "=" * 60)
        print("Step 5: Knowledge Pump Context Generation")
        print("=" * 60)

        self.stats["timestamp"] = datetime.now().isoformat()

        # Step 0: Initialize dimension clustering (unsupervised)
        n_dim_clusters = self.config.get("KP_N_DIMENSION_CLUSTERS", 3)
        print(f"\n[Step 5.0] Initializing dimension clustering (k={n_dim_clusters})...")
        init_metric_clusters(seed_metrics, n_clusters=n_dim_clusters)

        # Step 1: Build relationships
        print("\n[Step 5.1] Building metric-concept-metric relationships...")
        self.relations = self.relation_builder.build_1hop_relations(seed_metrics)

        # Add direct edges
        direct_edges = self.relation_builder.get_direct_edges(seed_metrics)
        self.relations.extend(direct_edges)

        self.stats["total_relations"] = len(self.relations)
        print(f"[Step5] Total relations: {len(self.relations)}")

        if not self.relations:
            print("[Step5] Warning: No relations found, returning empty context")
            self.context = "No metric relations found in knowledge graph."
            self.stats["status"] = "no_relations"
            return self.context

        # Step 2: Clustering
        print("\n[Step 5.2] SBERT clustering...")
        self.clusterer.cluster_relations(self.relations)
        self.stats["clusters"] = len(self.clusterer.clusters)

        # Step 3: Select representatives
        print("\n[Step 5.3] Selecting representative relations...")
        self.representatives = self.clusterer.get_representatives()
        self.stats["representatives"] = len(self.representatives)

        # Step 4: Format context
        print("\n[Step 5.4] Formatting context...")
        self.context = self.formatter.format_context(
            self.representatives,
            include_cross_dimension=include_cross_dimension
        )
        self.stats["context_length"] = len(self.context)

        # Statistics for dimension distribution (dynamic dimensions)
        dim_counts = defaultdict(int)
        cross_count = 0
        for rel in self.representatives:
            if rel.is_cross_dimension():
                cross_count += 1
            else:
                dim_counts[rel.dimension_a] += 1

        dim_counts['cross'] = cross_count
        self.stats["dimension_breakdown"] = dict(dim_counts)
        self.stats["status"] = "success"

        print("\n" + "=" * 60)
        print("Knowledge Pump Summary")
        print("=" * 60)
        print(f"  Total relations: {self.stats['total_relations']}")
        print(f"  Clusters: {self.stats['clusters']}")
        print(f"  Representatives: {self.stats['representatives']}")
        print(f"  Context length: {self.stats['context_length']} chars")
        dim_str = ", ".join([f"{k}={v}" for k, v in sorted(dim_counts.items())])
        print(f"  Dimension distribution: {dim_str}")

        return self.context

    def save_context(self, output_path: str = None) -> str:
        """Save context to text file"""
        if output_path is None:
            output_path = get_output_path("step5_kp_context", extension="txt")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Knowledge Pump Context\n")
            f.write(f"# Generated: {self.stats.get('timestamp', datetime.now().isoformat())}\n")
            f.write(f"# Relations: {self.stats.get('total_relations', 0)}\n")
            f.write(f"# Representatives: {self.stats.get('representatives', 0)}\n")
            f.write(f"# Context Length: {len(self.context)} chars\n")
            f.write("=" * 60 + "\n\n")
            f.write(self.context)

        print(f"[Step5] Context saved to: {output_path}")
        return output_path

    def save_stats(self, output_path: str = None) -> str:
        """Save statistics to JSON file"""
        if output_path is None:
            output_path = get_output_path("step5_kp_stats")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        print(f"[Step5] Statistics saved to: {output_path}")
        return output_path

    def save_metric_only_graph(self, output_path: str = None) -> str:
        """Save graph with only metric nodes"""
        if output_path is None:
            output_path = get_output_path("step5_metric_only_graph", extension="gml")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create directed graph for consistency with JSON output
        G = nx.DiGraph()

        # Collect all metrics
        metrics = set()
        for rel in self.representatives:
            metrics.add(rel.metric_a)
            metrics.add(rel.metric_b)

        # Add nodes
        for metric in metrics:
            dim = get_metric_dimension(metric)
            G.add_node(metric, type='metric', dimension=dim)

        # Add edges
        for rel in self.representatives:
            if G.has_edge(rel.metric_a, rel.metric_b):
                continue  # Avoid duplicate edges

            desc = rel.description_a or rel.description_b or ''
            G.add_edge(
                rel.metric_a, rel.metric_b,
                concept=rel.concept,
                relation=rel.relation_a_to_concept,
                description=desc[:200]
            )

        nx.write_gml(G, output_path)
        print(f"[Step5] Metric graph saved to: {output_path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
        return output_path

    def save_full_relation_graph(self, relations: List[MetricRelation],
                                  output_path: str) -> str:
        """
        Export complete metric-concept-metric tripartite graph to GML format

        Args:
            relations: List of MetricRelation (representatives or all relations)
            output_path: Output path

        Returns:
            Saved file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        G = nx.DiGraph()  # Use directed graph to match JSON output

        for rel in relations:
            # Add metric nodes (with dimension attribute)
            if rel.metric_a not in G:
                G.add_node(rel.metric_a,
                          node_type='metric',
                          dimension=rel.dimension_a)
            if rel.metric_b not in G:
                G.add_node(rel.metric_b,
                          node_type='metric',
                          dimension=rel.dimension_b)

            # Add concept node
            concept_id = str(rel.concept)
            if concept_id not in G:
                G.add_node(concept_id,
                          node_type='concept',
                          concept_type=rel.concept_type or 'unknown')

            # Add edge: metric_a -- concept
            edge_key_a = (rel.metric_a, concept_id)
            if not G.has_edge(*edge_key_a):
                G.add_edge(rel.metric_a, concept_id,
                          relation_type=rel.relation_a_to_concept or 'related_to',
                          description=rel.description_a[:200] if rel.description_a else '',
                          cluster_id=rel.cluster_id)

            # Add edge: concept -- metric_b
            edge_key_b = (concept_id, rel.metric_b)
            if not G.has_edge(*edge_key_b):
                G.add_edge(concept_id, rel.metric_b,
                          relation_type=rel.relation_concept_to_b or 'related_to',
                          description=rel.description_b[:200] if rel.description_b else '',
                          cluster_id=rel.cluster_id)

        nx.write_gml(G, output_path)

        # Count node types
        n_metrics = sum(1 for n, d in G.nodes(data=True) if d.get('node_type') == 'metric')
        n_concepts = sum(1 for n, d in G.nodes(data=True) if d.get('node_type') == 'concept')

        print(f"[Step5] Full relation graph saved: {output_path}")
        print(f"        ({n_metrics} metrics, {n_concepts} concepts, {G.number_of_edges()} edges)")
        return output_path

    def save_graph_json(self, relations: List[MetricRelation] = None,
                        output_path: str = None) -> str:
        """
        Save JSON file for Cytoscape.js visualization - deduplicated tripartite graph matching GML

        Nodes distinguished by node_type (metric/concept)
        Edges deduplicated, matching save_full_relation_graph GML output
        Full description preserved for click display

        Args:
            relations: List of MetricRelation, defaults to self.representatives
            output_path: Output path, auto-generated if not provided

        Returns:
            Saved file path
        """
        if output_path is None:
            output_path = get_output_path("step5_kp_graph")

        relations = relations or self.representatives

        nodes = []
        edges = []
        seen_nodes = set()
        seen_edges = set()  # Edge deduplication

        for rel in relations:
            # Add metric_a node
            if rel.metric_a not in seen_nodes:
                seen_nodes.add(rel.metric_a)
                nodes.append({
                    "id": rel.metric_a,
                    "label": rel.metric_a.replace("_", " "),
                    "nodeType": "metric",
                    "dimension": rel.dimension_a
                })

            # Add metric_b node
            if rel.metric_b not in seen_nodes:
                seen_nodes.add(rel.metric_b)
                nodes.append({
                    "id": rel.metric_b,
                    "label": rel.metric_b.replace("_", " "),
                    "nodeType": "metric",
                    "dimension": rel.dimension_b
                })

            # Add concept node
            concept_id = str(rel.concept)
            if concept_id not in seen_nodes:
                seen_nodes.add(concept_id)
                # Truncate long labels for display, but keep full ID
                label = concept_id[:30] + "..." if len(concept_id) > 30 else concept_id
                nodes.append({
                    "id": concept_id,
                    "label": label,
                    "nodeType": "concept",
                    "conceptType": rel.concept_type or "unknown"
                })

            # Add edge: metric_a -> concept (deduplicated)
            edge_key_a = (rel.metric_a, concept_id)
            if edge_key_a not in seen_edges:
                seen_edges.add(edge_key_a)
                edges.append({
                    "source": rel.metric_a,
                    "target": concept_id,
                    "relationType": rel.relation_a_to_concept or "related_to",
                    "label": (rel.description_a or "")[:50],  # Short label for display
                    "description": rel.description_a or ""    # Full description for click
                })

            # Add edge: concept -> metric_b (deduplicated)
            edge_key_b = (concept_id, rel.metric_b)
            if edge_key_b not in seen_edges:
                seen_edges.add(edge_key_b)
                edges.append({
                    "source": concept_id,
                    "target": rel.metric_b,
                    "relationType": rel.relation_concept_to_b or "related_to",
                    "label": (rel.description_b or "")[:50],  # Short label for display
                    "description": rel.description_b or ""    # Full description for click
                })

        # Build output data
        graph_data = {
            "timestamp": self.stats.get("timestamp"),
            "total_relations": len(self.relations),
            "representatives": len(relations),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": nodes,
            "edges": edges
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        # Count node types
        n_metrics = sum(1 for n in nodes if n.get('nodeType') == 'metric')
        n_concepts = sum(1 for n in nodes if n.get('nodeType') == 'concept')

        print(f"[Step5] Visualization JSON saved: {output_path}")
        print(f"        ({n_metrics} metrics, {n_concepts} concepts, {len(edges)} edges)")
        return output_path


# ============================================================
# Main Entry Point
# ============================================================

def load_step3_data(step3_path: str) -> Dict:
    """Load Step3 output data"""
    with open(step3_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Step 5: Knowledge Pump')
    parser.add_argument('--step3-output', type=str, help='Step 3 output JSON file path')
    parser.add_argument('--gml-path', type=str, help='GML knowledge graph path')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--max-chars', type=int, default=8000, help='Maximum context characters')
    parser.add_argument('--n-clusters', type=int, default=15, help='Number of SBERT clusters')
    parser.add_argument('--dim-clusters', type=int, default=3, help='Number of dimension clusters for unsupervised metric grouping')

    args = parser.parse_args()

    # Determine GML path
    gml_path = args.gml_path
    if gml_path is None:
        if CONFIG.get("USE_STAR_SUBGRAPH", True):
            gml_path = CONFIG["STAR_SUBGRAPH_PATH"]
        else:
            gml_path = CONFIG["GML_FILE_PATH"]

    print("\n" + "=" * 60)
    print("Step 5: Knowledge Pump")
    print("=" * 60)

    # Load Step3 data
    seed_metrics = []
    matched_nodes = {}

    if args.step3_output:
        print(f"\n[Load] Loading Step3 data: {args.step3_output}")
        step3_data = load_step3_data(args.step3_output)
        seed_metrics = step3_data.get("selected_seeds", [])
        matched_nodes = step3_data.get("seed_nodes", {})
        print(f"[Load] Loaded {len(seed_metrics)} seed metrics")
    else:
        print("[Warning] Step3 output not provided")

    # Update configuration
    config = CONFIG.copy()
    config["ENHANCED_KP_MAX_CHARS"] = args.max_chars
    config["ENHANCED_KP_N_CLUSTERS"] = args.n_clusters
    config["KP_N_DIMENSION_CLUSTERS"] = args.dim_clusters

    # Create Knowledge Pump
    kp = KnowledgePump(gml_path, matched_nodes, config)

    # Generate context
    context = kp.generate_context(seed_metrics)

    # Save results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    timestamp = get_timestamp()
    kp.save_context(os.path.join(output_dir, f"step5_kp_context_{timestamp}.txt"))
    kp.save_stats(os.path.join(output_dir, f"step5_kp_stats_{timestamp}.json"))
    kp.save_metric_only_graph(os.path.join(output_dir, f"step5_metric_only_graph_{timestamp}.gml"))

    # Save full relation graph GML (representatives)
    kp.save_full_relation_graph(
        kp.representatives,
        os.path.join(output_dir, f"step5_kp_full_graph_{timestamp}.gml")
    )

    # Save all relations graph GML
    kp.save_full_relation_graph(
        kp.relations,
        os.path.join(output_dir, f"step5_kp_all_relations_{timestamp}.gml")
    )

    # Save Cytoscape.js visualization JSON (representatives)
    kp.save_graph_json(
        kp.representatives,
        os.path.join(output_dir, f"step5_kp_graph_{timestamp}.json")
    )

    print("\n" + "=" * 60)
    print("Step 5 Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
