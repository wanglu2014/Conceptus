#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4: Subgraph Builder Module

Build unified subgraph from seed metrics and entity nodes with:
  - K-hop neighborhood expansion
  - SBERT edge clustering (K-Means) for deduplication
  - Impact factor sorting and filtering

Input Files:
    - outputs/step3_seed_metrics_{timestamp}.json
    - outputs/step2_matched_nodes_{timestamp}.json
    - GML knowledge graph

Output Files:
    - outputs/step4_subgraph_{timestamp}.gml
    - outputs/step4_subgraph_info_{timestamp}.json

Usage:
    python step4_subgraph_builder.py --step3-output outputs/step3_seed_metrics_*.json

Author: Pipeline Modularization Project
Date: 2024-12
"""

import os
import sys
import json
import argparse
import traceback
import numpy as np
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field

import networkx as nx

# Import configuration
from config import CONFIG, get_timestamp, get_output_path, DIMENSION_CATEGORIES

# Import SBERT components (from knowledge_pump_module)
try:
    from knowledge_pump_module import SBERTVectorizer, EvidenceClusterer, Evidence
    SBERT_AVAILABLE = True
except ImportError:
    print("[Warning] knowledge_pump_module not found, SBERT clustering disabled")
    SBERT_AVAILABLE = False


# ============================================================
# K-Hop Neighborhood Expansion
# ============================================================

def get_k_hop_neighbors(
    G: nx.Graph,
    seed_nodes: Set,
    k: int = 2
) -> Set:
    """
    Get k-hop neighbors of seed nodes

    Args:
        G: NetworkX graph
        seed_nodes: Set of seed nodes
        k: Number of hops

    Returns:
        set: Set of nodes including seeds and their k-hop neighbors
    """
    all_nodes = set(seed_nodes)
    current_frontier = set(seed_nodes)

    for hop in range(k):
        next_frontier = set()
        for node in current_frontier:
            if node in G:
                neighbors = set(G.neighbors(node))
                # For directed graphs, also get predecessor nodes
                if G.is_directed():
                    try:
                        neighbors |= set(G.predecessors(node))
                    except:
                        pass
                next_frontier |= neighbors

        # Remove already visited nodes
        next_frontier -= all_nodes
        all_nodes |= next_frontier
        current_frontier = next_frontier

        print(f"[Subgraph] Hop {hop+1}: Added {len(next_frontier)} nodes, total {len(all_nodes)} nodes")

        if not next_frontier:
            break

    return all_nodes


# ============================================================
# Edge Extraction and Processing
# ============================================================

def extract_edges_with_metadata(
    subgraph: nx.Graph,
    original_graph: nx.Graph
) -> List[Dict]:
    """
    Extract edges and their metadata from subgraph, including node descriptions

    Args:
        subgraph: Subgraph
        original_graph: Original graph (for getting node labels and descriptions)

    Returns:
        List[Dict]: List of edge information
    """
    edges_data = []

    for u, v, data in subgraph.edges(data=True):
        # Process MultiGraph edge data
        if isinstance(data, dict) and 0 in data:
            edge_data = data[0]
        else:
            edge_data = data if isinstance(data, dict) else {}

        # Get node attributes
        source_attrs = original_graph.nodes[u] if u in original_graph.nodes else {}
        target_attrs = original_graph.nodes[v] if v in original_graph.nodes else {}

        edge_info = {
            'source': u,
            'target': v,
            'impact_factor': edge_data.get('edge_impact_factor', 0.0),
            'citation_count': edge_data.get('edge_citation_count', 0),
            'edge_type': edge_data.get('type', edge_data.get('relation', 'related')),
            'description': edge_data.get('description', ''),
            'source_label': source_attrs.get('label', str(u)),
            'target_label': target_attrs.get('label', str(v)),
            # New: Node descriptions to supplement edge descriptions
            'source_description': source_attrs.get('description', ''),
            'target_description': target_attrs.get('description', ''),
            'source_type': source_attrs.get('type', 'unknown'),
            'target_type': target_attrs.get('type', 'unknown'),
        }
        edges_data.append(edge_info)

    return edges_data


def cluster_edges_with_sbert(
    edges: List[Dict],
    n_clusters: int = None
) -> List[Dict]:
    """
    Cluster edges using SBERT and return representative edge for each cluster

    Args:
        edges: List of edge information
        n_clusters: Number of clusters

    Returns:
        List[Dict]: List of representative edges
    """
    if not SBERT_AVAILABLE:
        print("[Subgraph] SBERT unavailable, skipping clustering")
        return edges

    if len(edges) == 0:
        return edges

    # Determine number of clusters
    if n_clusters is None:
        n_clusters = min(max(5, len(edges) // 20), 20)

    print(f"[Subgraph] SBERT clustering: {len(edges)} edges -> {n_clusters} clusters")

    try:
        # Convert to Evidence objects
        evidence_list = []
        for edge in edges:
            ev = Evidence(
                source_node=edge['source'] if isinstance(edge['source'], int) else hash(str(edge['source'])) % 10000000,
                target_node=edge['target'] if isinstance(edge['target'], int) else hash(str(edge['target'])) % 10000000,
                source_label=edge['source_label'],
                target_label=edge['target_label'],
                relation=edge['edge_type'],
                description=edge['description'] if edge['description'] else f"{edge['source_label']} -> {edge['target_label']}",
                domain="network_metric",
                source_type="subgraph"
            )
            ev._original_edge = edge
            evidence_list.append(ev)

        # SBERT vectorization
        vectorizer = SBERTVectorizer()
        evidence_with_emb = vectorizer.vectorize(evidence_list)

        # K-Means clustering
        clusterer = EvidenceClusterer(n_clusters=n_clusters)
        clusters = clusterer.cluster(evidence_with_emb)

        # Get representative edges
        representative_evidence = clusterer.get_representative_evidence()

        # Convert back to edge dictionary
        representative_edges = []
        for ev in representative_evidence:
            if hasattr(ev, '_original_edge'):
                representative_edges.append(ev._original_edge)

        print(f"[Subgraph] SBERT clustering complete: {len(edges)} -> {len(representative_edges)} representative edges")
        return representative_edges

    except Exception as e:
        print(f"[Subgraph] SBERT clustering failed: {e}, returning original edges")
        traceback.print_exc()
        return edges


def filter_edges_by_impact_factor(
    edges: List[Dict],
    max_edges: int = 50
) -> List[Dict]:
    """
    Sort by impact factor and filter Top-N edges

    Args:
        edges: List of edge information
        max_edges: Maximum number of edges

    Returns:
        List[Dict]: Filtered list of edges
    """
    # Sort by impact factor and citation count
    sorted_edges = sorted(
        edges,
        key=lambda x: (x.get('impact_factor', 0), x.get('citation_count', 0)),
        reverse=True
    )

    filtered = sorted_edges[:max_edges]

    if filtered:
        avg_if = np.mean([e.get('impact_factor', 0) for e in filtered])
        max_if = max([e.get('impact_factor', 0) for e in filtered])
        high_count = sum(1 for e in filtered if e.get('impact_factor', 0) > 5.0)
        print(f"[Subgraph] After filtering: {len(filtered)} edges, IF avg={avg_if:.2f}, max={max_if:.2f}, high impact(>5)={high_count}")

    return filtered


# ============================================================
# Edge Description Formatting
# ============================================================

def _get_metric_dimension(metric_name: str) -> Optional[str]:
    """Get the dimension that a metric belongs to"""
    if not metric_name:
        return None
    # Ensure it's a string type
    if not isinstance(metric_name, str):
        return None
    metric_lower = metric_name.lower()

    for dim, keywords in DIMENSION_CATEGORIES.items():
        for kw in keywords:
            if kw in metric_lower:
                return dim
    return None


def _is_cross_dimension_edge(edge: Dict, node_to_metric: Dict) -> bool:
    """Detect if an edge connects metrics from different dimensions"""
    source = edge.get('source', '')
    target = edge.get('target', '')

    # Get the metrics corresponding to source and target
    source_metric = node_to_metric.get(source, source)
    target_metric = node_to_metric.get(target, target)

    source_dim = _get_metric_dimension(source_metric)
    target_dim = _get_metric_dimension(target_metric)

    # If both ends are in different dimensions, it's a cross-dimension edge
    return source_dim != target_dim and source_dim is not None and target_dim is not None


def select_edges_fairly(
    edges: List[Dict],
    seed_metrics: List[str],
    matched_nodes: Dict[str, List],
    max_edges: int = 50
) -> List[Dict]:
    """
    Fairly select edges, ensuring each seed metric has at least one edge selected

    Args:
        edges: List of all edges
        seed_metrics: List of seed metrics
        matched_nodes: Mapping from metrics to nodes
        max_edges: Maximum number of edges

    Returns:
        List[Dict]: List of fairly selected edges
    """
    # Create reverse mapping
    node_to_metric = {}
    for metric, nodes in matched_nodes.items():
        if isinstance(nodes, list):
            for node in nodes:
                node_to_metric[node] = metric
        else:
            node_to_metric[nodes] = metric

    # Group edges by metric
    metric_edges = {metric: [] for metric in seed_metrics}
    other_edges = []

    for edge in edges:
        source = edge['source']
        target = edge['target']
        source_metric = node_to_metric.get(source)
        target_metric = node_to_metric.get(target)

        # Prioritize edges with description
        has_desc = bool(edge.get('description', '').strip())
        edge['has_description'] = has_desc

        # New: Detect wiki source
        source_type = str(edge.get('source_type', '')).lower()
        edge['is_wiki'] = 'wikipedia' in source_type or 'wiki' in source_type

        # New: Detect cross-dimension
        edge['is_cross_dimension'] = _is_cross_dimension_edge(edge, node_to_metric)

        if source_metric:
            metric_edges[source_metric].append(edge)
        if target_metric and target_metric != source_metric:
            metric_edges[target_metric].append(edge)
        if not source_metric and not target_metric:
            other_edges.append(edge)

    # Fair selection: at least 1 edge per metric
    # New sorting priority: wiki cross-dimension > wiki > has description > IF descending
    selected = []
    edges_per_metric = max(1, max_edges // len(seed_metrics)) if seed_metrics else 1

    for metric in seed_metrics:
        metric_edge_list = metric_edges.get(metric, [])
        # New sorting rules: wiki cross-dimension first, then wiki, then has description, finally IF
        sorted_edges = sorted(metric_edge_list, key=lambda x: (
            -int(x.get('is_wiki', False) and x.get('is_cross_dimension', False)),  # Wiki cross-dimension highest priority
            -int(x.get('is_wiki', False)),                                          # Wiki second priority
            -int(x.get('is_cross_dimension', False)),                               # Cross-dimension third
            -int(x.get('has_description', False)),                                  # Has description
            -x.get('impact_factor', 0)                                              # IF descending
        ))
        selected.extend(sorted_edges[:edges_per_metric])

    # Deduplicate
    seen = set()
    unique_selected = []
    for edge in selected:
        key = (edge['source'], edge['target'])
        if key not in seen:
            seen.add(key)
            unique_selected.append(edge)

    # If there's remaining capacity, add other high-quality edges
    remaining = max_edges - len(unique_selected)
    if remaining > 0:
        # Supplement from edges with description
        other_with_desc = [e for e in other_edges if e.get('has_description')]
        for edge in sorted(other_with_desc, key=lambda x: -x.get('impact_factor', 0))[:remaining]:
            key = (edge['source'], edge['target'])
            if key not in seen:
                seen.add(key)
                unique_selected.append(edge)

    return unique_selected


def format_edges_for_prompt(
    edges: List[Dict],
    seed_metrics: List[str],
    matched_nodes: Dict[str, List]
) -> List[str]:
    """
    Format edge descriptions for Knowledge Pump Context
    Output primarily in sentence descriptions rather than simple node relationships

    Args:
        edges: List of edge information
        seed_metrics: List of seed metrics
        matched_nodes: Mapping from metrics to nodes

    Returns:
        List[str]: Formatted edge descriptions (in sentence form)
    """
    # Create reverse mapping: node_id -> metric_name
    node_to_metric = {}
    for metric, nodes in matched_nodes.items():
        if isinstance(nodes, list):
            for node in nodes:
                node_to_metric[node] = metric
        else:
            node_to_metric[nodes] = metric

    descriptions = []

    for i, edge in enumerate(edges, 1):
        source = edge['source']
        target = edge['target']
        source_label = edge['source_label']
        target_label = edge['target_label']
        edge_type = edge['edge_type']
        description = edge.get('description', '').strip()
        impact_factor = edge.get('impact_factor', 0)

        # Check if it's a seed metric
        source_metric = node_to_metric.get(source)
        target_metric = node_to_metric.get(target)

        # Build relation tag
        if source_metric and target_metric:
            relation_tag = "Seed-to-Seed"
            source_name = source_metric
            target_name = target_metric
        elif source_metric:
            relation_tag = "Seed-related"
            source_name = source_metric
            target_name = f"concept_{target_label}"
        elif target_metric:
            relation_tag = "Seed-related"
            source_name = f"concept_{source_label}"
            target_name = target_metric
        else:
            relation_tag = "Context"
            source_name = f"concept_{source_label}"
            target_name = f"concept_{target_label}"

        # Build output: sentence descriptions as primary
        if description:
            # When description exists, output in sentence form
            line = f"[{relation_tag}] {source_name} → {target_name} ({edge_type}):\n"
            line += f"  \"{description}\""
        else:
            # When no description, construct simple explanation
            line = f"[{relation_tag}] {source_name} --[{edge_type}]--> {target_name}"
            # Try to construct sentence from node description
            node_desc = edge.get('source_description') or edge.get('target_description')
            if node_desc:
                line += f"\n  Context: {node_desc[:150]}"

        if impact_factor > 0:
            line += f" (IF={impact_factor:.1f})"

        descriptions.append(line)

    return descriptions


# ============================================================
# Main Subgraph Builder Class
# ============================================================

class SubgraphBuilder:
    """
    Unified Subgraph Builder

    Usage:
        builder = SubgraphBuilder(graph)
        builder.build(seed_metrics, matched_nodes, entity_nodes)
        builder.save_subgraph()
        builder.save_info()
    """

    def __init__(self, graph: nx.Graph, config: Dict = None):
        """Initialize subgraph builder"""
        self.G = graph
        self.config = config or CONFIG

        # Result storage
        self.subgraph: Optional[nx.Graph] = None
        self.subgraph_nodes: Set = set()
        self.all_edges: List[Dict] = []
        self.filtered_edges: List[Dict] = []
        self.edge_descriptions: List[str] = []

        # Node name mapping (node_id -> metric_name)
        self.node_to_metric: Dict = {}
        self.metric_to_nodes: Dict = {}

        # Statistics
        self.stats: Dict[str, Any] = {
            "timestamp": None,
            "status": "not_started",
            "total_seed_nodes": 0,
            "total_entity_nodes": 0,
            "subgraph_nodes": 0,
            "total_edges": 0,
            "edges_before_sbert": 0,
            "edges_after_sbert": 0,
            "filtered_edges": 0,
        }

    def _extract_star_subgraph(
        self,
        all_nodes: Set,
        seed_nodes: Set
    ) -> nx.Graph:
        """
        Extract star subgraph - only keep seed->neighbor edges, remove neighbor->neighbor edges
        Also add readable metric_name attribute to nodes

        Args:
            all_nodes: Set of all nodes (including neighbors)
            seed_nodes: Set of seed nodes

        Returns:
            nx.Graph: Star subgraph
        """
        # Create new graph
        star_graph = nx.Graph() if not self.G.is_directed() else nx.DiGraph()

        # Add all nodes with their attributes, also add metric_name
        for node in all_nodes:
            if node in self.G:
                node_attrs = dict(self.G.nodes[node])
                # Add metric_name attribute
                if node in self.node_to_metric:
                    node_attrs['metric_name'] = self.node_to_metric[node]
                    node_attrs['label'] = self.node_to_metric[node]  # Update label to readable name
                    node_attrs['is_seed'] = 1
                else:
                    node_attrs['is_seed'] = 0
                star_graph.add_node(node, **node_attrs)

        # Only add edges where at least one end is a seed node
        edges_added = 0
        edges_skipped = 0

        for u, v, data in self.G.edges(data=True):
            if u in all_nodes and v in all_nodes:
                # At least one end must be a seed node
                if u in seed_nodes or v in seed_nodes:
                    star_graph.add_edge(u, v, **data)
                    edges_added += 1
                else:
                    edges_skipped += 1

        print(f"[Step4] Star subgraph: kept {edges_added} edges, skipped {edges_skipped} neighbor-neighbor edges")

        return star_graph

    def build(
        self,
        seed_metrics: List[str],
        matched_nodes: Dict[str, List],
        entity_nodes: List = None,
        max_hop: int = None,
        max_edges: int = 50,
        use_sbert: bool = True,
        keyword_nodes: Set = None,
        star_mode: bool = False
    ) -> Dict:
        """
        Build unified subgraph

        Args:
            seed_metrics: List of seed metrics
            matched_nodes: Mapping from metrics to nodes
            entity_nodes: List of entity nodes (e.g., antifragility concept nodes)
            max_hop: Maximum number of hops
            max_edges: Maximum number of edges to keep
            use_sbert: Whether to use SBERT clustering
            keyword_nodes: Set of nodes from Step3.5 keyword matching
            star_mode: Whether to extract star subgraph (only keep seed-neighbor edges, remove neighbor-neighbor edges)

        Returns:
            Dict: Build result
        """
        if max_hop is None:
            max_hop = self.config.get("MAX_HOP", 2)

        mode_str = "Star" if star_mode else "K-hop"
        print("\n" + "=" * 60)
        print(f"Step 4: Building {mode_str} Subgraph (max_hop={max_hop})")
        print("=" * 60)

        self.stats["timestamp"] = datetime.now().isoformat()
        self.stats["star_mode"] = star_mode

        # Step 0: Build node name mapping
        self.metric_to_nodes = matched_nodes
        self.node_to_metric = {}
        for metric, nodes in matched_nodes.items():
            if isinstance(nodes, list):
                for node in nodes:
                    self.node_to_metric[node] = metric
            else:
                self.node_to_metric[nodes] = metric

        # Step 1: Collect all seed nodes
        all_seed_nodes = set()
        for metric in seed_metrics:
            if metric in matched_nodes:
                nodes = matched_nodes[metric]
                if isinstance(nodes, list):
                    all_seed_nodes.update(nodes)
                else:
                    all_seed_nodes.add(nodes)

        self.stats["total_seed_nodes"] = len(all_seed_nodes)
        print(f"[Step4] Collected {len(all_seed_nodes)} seed nodes (from {len(seed_metrics)} metrics)")

        # Step 1.5: Merge keyword nodes from Step3.5
        if keyword_nodes:
            keyword_set = set(keyword_nodes) if not isinstance(keyword_nodes, set) else keyword_nodes
            all_seed_nodes = all_seed_nodes | keyword_set
            self.stats["keyword_nodes"] = len(keyword_set)
            print(f"[Step4] Merged {len(keyword_set)} keyword nodes, total {len(all_seed_nodes)} seed nodes")
        else:
            self.stats["keyword_nodes"] = 0

        # Step 2: Add entity nodes
        entity_node_set = set(entity_nodes) if entity_nodes else set()
        target_nodes = all_seed_nodes | entity_node_set
        self.stats["total_entity_nodes"] = len(entity_node_set)

        print(f"[Step4] Total target nodes: {len(target_nodes)} (seeds={len(all_seed_nodes)}, entities={len(entity_node_set)})")

        # Step 3: K-hop neighbor expansion
        print(f"\n[Step4] K-hop neighbor expansion (k={max_hop})...")
        self.subgraph_nodes = get_k_hop_neighbors(self.G, target_nodes, k=max_hop)
        self.stats["subgraph_nodes"] = len(self.subgraph_nodes)

        # Step 4: Extract subgraph
        valid_nodes = {n for n in self.subgraph_nodes if n in self.G}

        if star_mode:
            # Star subgraph: only keep seed->neighbor edges, remove neighbor->neighbor edges
            print("[Step4] Extracting star subgraph (only keeping seed-neighbor edges)...")
            self.subgraph = self._extract_star_subgraph(valid_nodes, target_nodes)
        else:
            # K-hop subgraph: includes all edges
            self.subgraph = self.G.subgraph(valid_nodes).copy()

        self.stats["total_edges"] = self.subgraph.number_of_edges()

        print(f"[Step4] Subgraph: {self.subgraph.number_of_nodes()} nodes, {self.subgraph.number_of_edges()} edges")

        # Step 5: Extract edge information
        print("\n[Step4] Extracting edge metadata...")
        self.all_edges = extract_edges_with_metadata(self.subgraph, self.G)
        self.stats["edges_before_sbert"] = len(self.all_edges)

        # Step 6: SBERT clustering (optional)
        edges_for_filter = self.all_edges
        if use_sbert and SBERT_AVAILABLE and len(self.all_edges) > max_edges:
            print("\n[Step4] SBERT edge clustering...")
            edges_for_filter = cluster_edges_with_sbert(self.all_edges)
            self.stats["edges_after_sbert"] = len(edges_for_filter)
        else:
            self.stats["edges_after_sbert"] = len(self.all_edges)

        # Step 7: Fair edge selection (ensure each metric has representation)
        print("\n[Step4] Fair edge selection (at least 1 per metric, prioritizing edges with description)...")
        self.filtered_edges = select_edges_fairly(
            edges_for_filter,
            seed_metrics,
            matched_nodes,
            max_edges
        )
        self.stats["filtered_edges"] = len(self.filtered_edges)

        # Count edges with description
        with_desc = sum(1 for e in self.filtered_edges if e.get('description', '').strip())
        print(f"[Step4] Selected {len(self.filtered_edges)} edges, {with_desc} have description")

        # Step 8: Format edge descriptions
        print("\n[Step4] Formatting edge descriptions...")
        self.edge_descriptions = format_edges_for_prompt(
            self.filtered_edges,
            seed_metrics,
            matched_nodes
        )

        self.stats["status"] = "success"

        print("\n" + "=" * 60)
        print("Subgraph Build Summary")
        print("=" * 60)
        print(f"  Seed nodes: {self.stats['total_seed_nodes']}")
        print(f"  Entity nodes: {self.stats['total_entity_nodes']}")
        print(f"  Subgraph nodes: {self.stats['subgraph_nodes']}")
        print(f"  Total edges: {self.stats['total_edges']}")
        print(f"  After SBERT: {self.stats['edges_after_sbert']}")
        print(f"  Final edges: {self.stats['filtered_edges']}")

        return {
            'subgraph_nodes': self.subgraph_nodes,
            'filtered_edges': self.filtered_edges,
            'edge_descriptions': self.edge_descriptions,
            'stats': self.stats
        }

    def save_subgraph(self, output_path: str = None) -> str:
        """
        Save subgraph GML file

        Args:
            output_path: Output file path

        Returns:
            str: Output file path
        """
        if self.subgraph is None:
            print("[Step4] Error: Subgraph not yet built")
            return ""

        if output_path is None:
            output_path = get_output_path("step4_subgraph", extension="gml")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nx.write_gml(self.subgraph, output_path)

        print(f"[Step4] Subgraph saved to: {output_path}")
        return output_path

    def save_info(self, output_path: str = None) -> str:
        """
        Save subgraph information JSON file

        Args:
            output_path: Output file path

        Returns:
            str: Output file path
        """
        if output_path is None:
            output_path = get_output_path("step4_subgraph_info")

        # Prepare output data
        output_data = {
            **self.stats,
            "subgraph_nodes": list(self.subgraph_nodes)[:100],  # Only save first 100 node IDs
            "node_to_metric": {str(k): v for k, v in list(self.node_to_metric.items())[:50]},  # Node ID to metric name mapping
            "filtered_edges": [
                {
                    "source": str(e['source']),
                    "target": str(e['target']),
                    "source_label": e['source_label'],
                    "target_label": e['target_label'],
                    "source_metric": self.node_to_metric.get(e['source'], None),
                    "target_metric": self.node_to_metric.get(e['target'], None),
                    "edge_type": e['edge_type'],
                    "impact_factor": e.get('impact_factor', 0),
                    "description": e.get('description', '')[:300],
                    "source_description": e.get('source_description', '')[:150],
                    "target_description": e.get('target_description', '')[:150],
                }
                for e in self.filtered_edges
            ],
            "edge_descriptions": self.edge_descriptions,
            "config_used": {
                "max_hop": self.config.get("MAX_HOP"),
                "sbert_model": self.config.get("SBERT_MODEL"),
            }
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[Step4] Subgraph info saved to: {output_path}")
        return output_path


# ============================================================
# Main Entry Point
# ============================================================

def load_step3_data(step3_path: str) -> Dict:
    """Load Step3 output data"""
    with open(step3_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_step2_data(step2_path: str) -> Dict:
    """Load Step2 output data"""
    with open(step2_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Step 4: Subgraph Builder')
    parser.add_argument('--step3-output', type=str, help='Step 3 output JSON file path')
    parser.add_argument('--step2-output', type=str, help='Step 2 output JSON file path')
    parser.add_argument('--gml-path', type=str, help='GML knowledge graph path')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--max-hop', type=int, default=2, help='Maximum hop distance')
    parser.add_argument('--max-edges', type=int, default=50, help='Maximum edges to keep')
    parser.add_argument('--no-sbert', action='store_true', help='Disable SBERT clustering')

    args = parser.parse_args()

    # Determine GML path
    gml_path = args.gml_path
    if gml_path is None:
        if CONFIG.get("USE_STAR_SUBGRAPH", True):
            gml_path = CONFIG["STAR_SUBGRAPH_PATH"]
        else:
            gml_path = CONFIG["GML_FILE_PATH"]

    print("\n" + "=" * 60)
    print("Step 4: Subgraph Builder")
    print("=" * 60)

    # Load knowledge graph
    print(f"\n[Load] Loading knowledge graph: {gml_path}")
    G = nx.read_gml(gml_path, label='id')
    print(f"[Load] Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load Step3 data
    seed_metrics = []
    seed_nodes_mapping = {}
    if args.step3_output:
        print(f"\n[Load] Loading Step3 data: {args.step3_output}")
        step3_data = load_step3_data(args.step3_output)
        seed_metrics = step3_data.get("selected_seeds", [])
        seed_nodes_mapping = step3_data.get("seed_nodes", {})
        print(f"[Load] Loaded {len(seed_metrics)} seed metrics")
    else:
        print("[Warning] Step3 output not provided, using empty seed list")

    # Load Step2 data
    matched_nodes = seed_nodes_mapping
    entity_nodes = []
    if args.step2_output:
        print(f"\n[Load] Loading Step2 data: {args.step2_output}")
        step2_data = load_step2_data(args.step2_output)
        if not matched_nodes:
            matched_nodes = step2_data.get("metrics_to_node_ids", {})
        entity_nodes = step2_data.get("combined_seed_nodes", [])
        print(f"[Load] Loaded {len(entity_nodes)} entity nodes")

    # Create builder
    builder = SubgraphBuilder(G)

    # Build subgraph
    builder.build(
        seed_metrics=seed_metrics,
        matched_nodes=matched_nodes,
        entity_nodes=entity_nodes,
        max_hop=args.max_hop,
        max_edges=args.max_edges,
        use_sbert=not args.no_sbert
    )

    # Save results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    timestamp = get_timestamp()
    gml_path = os.path.join(output_dir, f"step4_subgraph_{timestamp}.gml")
    info_path = os.path.join(output_dir, f"step4_subgraph_info_{timestamp}.json")

    builder.save_subgraph(gml_path)
    builder.save_info(info_path)

    print("\n" + "=" * 60)
    print("Step 4 Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
