#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2: Node Matcher Module

Two-stage node matching for knowledge graph:
  Stage 1: String matching (exact keyword match in node labels/attributes)
  Stage 2: Semantic embedding matching (SapBERT cosine similarity)

This module integrates logic from:
  - extract_antifragility_subgraph.py (string + semantic matching)
  - modular_with_enhanced_rag.py (entity extraction)

Input Files:
    - outputs/step1_data_loaded_{timestamp}.json
    - GML knowledge graph
    - SapBERT embeddings JSON

Output File:
    - outputs/step2_matched_nodes_{timestamp}.json

Usage:
    python step2_node_matcher.py --step1-output outputs/step1_data_loaded_*.json

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

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# Import configuration
from config import CONFIG, STRING_MATCH_TERMS, FIXED_METRICS, get_timestamp, get_output_path


# ============================================================
# Stage 1: String Matching
# ============================================================

def find_nodes_by_string_match(
    G: nx.Graph,
    target_terms: List[str] = None,
    check_all_attrs: bool = True
) -> Set:
    """
    Find nodes by string matching

    Prioritizes matching label attribute, optionally matches other attributes

    Args:
        G: NetworkX graph
        target_terms: Target keyword list
        check_all_attrs: Whether to check all attributes (not just label)

    Returns:
        set: Set of matched node IDs
    """
    if target_terms is None:
        target_terms = STRING_MATCH_TERMS

    # Convert to lowercase for matching
    target_terms_lower = [t.lower() for t in target_terms]

    matched_nodes = set()

    for node_id, attrs in G.nodes(data=True):
        matched = False

        # Priority 1: Check label attribute
        label = attrs.get('label', '')
        if isinstance(label, str):
            label_lower = label.lower()
            for term in target_terms_lower:
                if term in label_lower:
                    matched_nodes.add(node_id)
                    matched = True
                    break

        # Priority 2: Check other attributes
        if not matched and check_all_attrs:
            for key, value in attrs.items():
                if key == 'label':
                    continue  # Already checked
                if isinstance(value, str):
                    value_lower = value.lower()
                    for term in target_terms_lower:
                        if term in value_lower:
                            matched_nodes.add(node_id)
                            matched = True
                            break
                if matched:
                    break

    return matched_nodes


def match_metrics_to_nodes(
    G: nx.Graph,
    metrics: List[str]
) -> Dict[str, List]:
    """
    Match network metric names to knowledge graph nodes

    Args:
        G: NetworkX graph
        metrics: Metric name list

    Returns:
        dict: {metric_name: [node_ids]}
    """
    metric_to_nodes = {}

    # Build metric name variants mapping (handling different naming conventions)
    metric_variants = {}
    for metric in metrics:
        variants = [
            metric.lower(),
            metric.lower().replace('_', ' '),
            metric.lower().replace('_', ''),
        ]
        # Add more variants to improve match rate
        # e.g., "average_degree" -> ["average degree", "mean degree", "avg degree"]
        if 'average' in metric.lower():
            variants.append(metric.lower().replace('average', 'mean').replace('_', ' '))
            variants.append(metric.lower().replace('average', 'avg').replace('_', ' '))
        if 'number_of' in metric.lower():
            variants.append(metric.lower().replace('number_of_', '').replace('_', ' ') + ' count')
            variants.append(metric.lower().replace('number_of_', 'total ').replace('_', ' '))
        metric_variants[metric] = variants

    for node_id, attrs in G.nodes(data=True):
        label = attrs.get('label', '')
        if isinstance(label, str):
            label = label.lower()
        else:
            label = str(label).lower()
        extracted_term = attrs.get('extracted_term', '').lower() if attrs.get('extracted_term') else ''
        # Important: Also check description attribute since graph node labels are numeric IDs
        description = attrs.get('description', '').lower() if attrs.get('description') else ''

        for metric, variants in metric_variants.items():
            for variant in variants:
                # Check label, extracted_term and description
                if variant in label or variant in extracted_term or variant in description:
                    if metric not in metric_to_nodes:
                        metric_to_nodes[metric] = []
                    if node_id not in metric_to_nodes[metric]:
                        metric_to_nodes[metric].append(node_id)
                    break

    return metric_to_nodes


# ============================================================
# Stage 2: Semantic Embedding Matching
# ============================================================

def load_embeddings_from_npy(config: Dict) -> Tuple[np.ndarray, List, List, Dict]:
    """
    Load embeddings and mapping from NPY file

    Args:
        config: Configuration dictionary

    Returns:
        Tuple: (embeddings matrix, node_ids list, node_labels list, node_to_idx mapping)
    """
    npy_path = config.get("EMBEDDING_NPY_PATH")
    mapping_path = config.get("NODE_MAPPING_PATH")

    if not npy_path or not os.path.exists(npy_path):
        print(f"[Step2] NPY file does not exist: {npy_path}")
        return None, [], [], {}

    if not mapping_path or not os.path.exists(mapping_path):
        print(f"[Step2] Mapping file does not exist: {mapping_path}")
        return None, [], [], {}

    print(f"[Step2] Loading NPY embedding: {npy_path}")
    embeddings = np.load(npy_path)
    print(f"[Step2] Embedding shape: {embeddings.shape}")

    print(f"[Step2] Loading node mapping: {mapping_path}")
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)

    node_ids = mapping['node_ids']
    node_labels = mapping['node_labels']

    # Build node_to_idx mapping
    node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    print(f"[Step2] Loading complete: {len(node_ids)} nodes")

    return embeddings, node_ids, node_labels, node_to_idx


def build_node_to_idx_mapping(embeddings_data: Dict) -> Dict:
    """
    Build node-to-index mapping from embeddings data

    Args:
        embeddings_data: Embeddings JSON data

    Returns:
        dict: {node_id: index}
    """
    node_to_idx = {}

    if 'embeddings' in embeddings_data:
        # Format: {"embeddings": {"node_id": [vector], ...}}
        for idx, node_id in enumerate(embeddings_data['embeddings'].keys()):
            node_to_idx[node_id] = idx
    else:
        # Direct format: {"node_id": [vector], ...}
        for idx, node_id in enumerate(embeddings_data.keys()):
            node_to_idx[node_id] = idx

    return node_to_idx


def get_embeddings_matrix(embeddings_data: Dict) -> np.ndarray:
    """
    Extract embeddings matrix from embeddings data

    Args:
        embeddings_data: Embeddings JSON data

    Returns:
        numpy.ndarray: Embeddings matrix (n_nodes, embedding_dim)
    """
    if 'embeddings' in embeddings_data:
        vectors = list(embeddings_data['embeddings'].values())
    else:
        vectors = list(embeddings_data.values())

    return np.array(vectors)


def find_nodes_by_semantic_similarity(
    G: nx.Graph,
    embeddings_data: Dict,
    seed_nodes: Set,
    top_k: int = None,
    similarity_threshold: float = None
) -> Tuple[Set, Dict]:
    """
    Find related nodes by semantic embedding similarity

    Uses mean of seed node embeddings as query vector to find most similar nodes

    Args:
        G: NetworkX graph
        embeddings_data: Embeddings data dictionary
        seed_nodes: Seed node set (used to compute query vector)
        top_k: Number of Top-K similar nodes to return
        similarity_threshold: Similarity threshold

    Returns:
        Tuple[set, dict]: (matched node set, similarity score dictionary)
    """
    if top_k is None:
        top_k = CONFIG.get("SEMANTIC_TOP_K", 100)
    if similarity_threshold is None:
        similarity_threshold = CONFIG.get("SEMANTIC_SIMILARITY_THRESHOLD", 0.4)

    # Build mapping
    node_to_idx = build_node_to_idx_mapping(embeddings_data)
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    embeddings = get_embeddings_matrix(embeddings_data)

    print(f"[Step2] Embeddings matrix shape: {embeddings.shape}")
    print(f"[Step2] Node mapping count: {len(node_to_idx)}")

    # Collect seed node embeddings
    seed_embeddings = []
    for node_id in seed_nodes:
        # Try different node ID formats
        node_id_str = str(node_id)
        if node_id_str in node_to_idx:
            idx = node_to_idx[node_id_str]
            seed_embeddings.append(embeddings[idx])
        elif node_id in node_to_idx:
            idx = node_to_idx[node_id]
            seed_embeddings.append(embeddings[idx])

    if not seed_embeddings:
        print("[Step2] Warning: No embeddings found for seed nodes!")
        return set(), {}

    seed_embeddings = np.array(seed_embeddings)
    print(f"[Step2] Seed embeddings count: {seed_embeddings.shape[0]}")

    # Compute query vector (mean of seed node embeddings)
    query_embedding = np.mean(seed_embeddings, axis=0).reshape(1, -1)

    # Calculate cosine similarity with all nodes
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Get Top-K similar nodes
    top_indices = np.argsort(similarities)[::-1][:top_k]

    semantic_matched_nodes = set()
    similarity_scores = {}

    print(f"\n[Step2] Top semantically similar nodes (threshold={similarity_threshold}):")
    displayed = 0
    for idx in top_indices:
        sim = similarities[idx]
        if sim >= similarity_threshold:
            node_id = idx_to_node.get(idx)
            if node_id and node_id in G:
                semantic_matched_nodes.add(node_id)
                similarity_scores[node_id] = float(sim)

                if displayed < 10:
                    label = G.nodes[node_id].get('label', str(node_id))[:50]
                    print(f"    [{displayed+1}] sim={sim:.3f}: {label}")
                    displayed += 1

    print(f"[Step2] Semantic matched nodes count (sim >= {similarity_threshold}): {len(semantic_matched_nodes)}")

    return semantic_matched_nodes, similarity_scores


def find_nodes_by_semantic_similarity_npy(
    G: nx.Graph,
    embeddings: np.ndarray,
    node_to_idx: Dict,
    node_labels: List,
    seed_nodes: Set,
    top_k: int = None,
    similarity_threshold: float = None
) -> Tuple[Set, Dict]:
    """
    Perform semantic similarity matching using NPY format embeddings

    Args:
        G: NetworkX graph
        embeddings: NPY embedding matrix (n_nodes, 768)
        node_to_idx: Node ID to index mapping
        node_labels: Node labels list
        seed_nodes: Seed node set
        top_k: Number of Top-K similar nodes to return
        similarity_threshold: Similarity threshold

    Returns:
        Tuple[set, dict]: (matched node set, similarity score dictionary)
    """
    if top_k is None:
        top_k = CONFIG.get("SEMANTIC_TOP_K", 100)
    if similarity_threshold is None:
        similarity_threshold = CONFIG.get("SEMANTIC_SIMILARITY_THRESHOLD", 0.4)

    idx_to_node = {v: k for k, v in node_to_idx.items()}

    print(f"[Step2-NPY] Embeddings matrix shape: {embeddings.shape}")
    print(f"[Step2-NPY] Node mapping count: {len(node_to_idx)}")

    # Collect seed node embeddings
    seed_embeddings = []
    seed_found = 0
    for node_id in seed_nodes:
        # Try different node ID formats (integer and string)
        node_id_str = str(node_id)
        node_id_int = int(node_id) if isinstance(node_id, str) and node_id.isdigit() else node_id

        if node_id_str in node_to_idx:
            idx = node_to_idx[node_id_str]
            seed_embeddings.append(embeddings[idx])
            seed_found += 1
        elif node_id_int in node_to_idx:
            idx = node_to_idx[node_id_int]
            seed_embeddings.append(embeddings[idx])
            seed_found += 1
        elif node_id in node_to_idx:
            idx = node_to_idx[node_id]
            seed_embeddings.append(embeddings[idx])
            seed_found += 1

    print(f"[Step2-NPY] Found {seed_found}/{len(seed_nodes)} seed node embeddings")

    if not seed_embeddings:
        print("[Step2-NPY] Warning: No embeddings found for seed nodes!")
        return set(), {}

    seed_embeddings = np.array(seed_embeddings)
    print(f"[Step2-NPY] Seed embeddings count: {seed_embeddings.shape[0]}")

    # Compute query vector (mean of seed node embeddings)
    query_embedding = np.mean(seed_embeddings, axis=0).reshape(1, -1)

    # Calculate cosine similarity with all nodes
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Get Top-K similar nodes
    top_indices = np.argsort(similarities)[::-1][:top_k]

    semantic_matched_nodes = set()
    similarity_scores = {}

    print(f"\n[Step2-NPY] Top semantically similar nodes (threshold={similarity_threshold}):")
    displayed = 0
    for idx in top_indices:
        sim = similarities[idx]
        if sim >= similarity_threshold:
            node_id = idx_to_node.get(idx)
            if node_id:
                # Check if node is in graph (may use different ID format)
                node_in_graph = node_id in G or str(node_id) in G
                if node_in_graph:
                    semantic_matched_nodes.add(node_id)
                    similarity_scores[node_id] = float(sim)

                    if displayed < 10:
                        label = node_labels[idx] if idx < len(node_labels) else str(node_id)
                        print(f"    [{displayed+1}] sim={sim:.3f}: {label[:50]}")
                        displayed += 1

    print(f"[Step2-NPY] Semantic matched nodes count (sim >= {similarity_threshold}): {len(semantic_matched_nodes)}")

    return semantic_matched_nodes, similarity_scores


# ============================================================
# Main Node Matcher Class
# ============================================================

class NodeMatcher:
    """
    Two-stage Node Matcher

    Usage:
        # NPY format (recommended)
        matcher = NodeMatcher(graph, config=CONFIG)
        matcher.load_npy_embeddings()  # Load NPY from config
        matcher.run_two_stage_matching()

        # Or JSON format (legacy)
        matcher = NodeMatcher(graph, embeddings_data=json_data)
        matcher.run_two_stage_matching()
    """

    def __init__(self, graph: nx.Graph, embeddings_data: Dict = None, config: Dict = None):
        """Initialize node matcher"""
        self.G = graph
        self.embeddings_data = embeddings_data  # JSON format (legacy)
        self.config = config or CONFIG

        # NPY format embedding storage
        self.embeddings_npy: np.ndarray = None
        self.npy_node_ids: List = []
        self.npy_node_labels: List = []
        self.npy_node_to_idx: Dict = {}
        self.use_npy = False  # Whether to use NPY format

        # Results storage
        self.string_matched_nodes: Set = set()
        self.semantic_matched_nodes: Set = set()
        self.combined_seed_nodes: Set = set()
        self.metrics_to_node_ids: Dict[str, List] = {}
        self.similarity_scores: Dict = {}

        # Status information
        self.match_status: Dict[str, Any] = {
            "timestamp": None,
            "status": "not_started",
            "string_matched_count": 0,
            "semantic_matched_count": 0,
            "combined_count": 0,
            "metrics_matched_count": 0,
            "embedding_format": "none",
        }

    def load_npy_embeddings(self) -> bool:
        """
        Load NPY format embeddings from config

        Returns:
            bool: Whether loading was successful
        """
        result = load_embeddings_from_npy(self.config)
        self.embeddings_npy, self.npy_node_ids, self.npy_node_labels, self.npy_node_to_idx = result

        if self.embeddings_npy is not None:
            self.use_npy = True
            self.match_status["embedding_format"] = "npy"
            print(f"[NodeMatcher] NPY embedding loaded: {self.embeddings_npy.shape}")
            return True
        else:
            self.use_npy = False
            print("[NodeMatcher] NPY embedding loading failed, will use JSON format or skip semantic matching")
            return False

    def run_string_matching(self, target_terms: List[str] = None) -> Set:
        """
        Run string matching (Stage 1)

        Args:
            target_terms: Target keyword list

        Returns:
            set: Matched node set
        """
        print("\n[Stage 1] String matching...")

        self.string_matched_nodes = find_nodes_by_string_match(
            self.G,
            target_terms=target_terms,
            check_all_attrs=True
        )

        self.match_status["string_matched_count"] = len(self.string_matched_nodes)
        print(f"[Stage 1] String matched nodes count: {len(self.string_matched_nodes)}")

        # Display partial match results
        if self.string_matched_nodes:
            print("[Stage 1] Sample matched nodes:")
            for i, node_id in enumerate(list(self.string_matched_nodes)[:5]):
                label = self.G.nodes[node_id].get('label', str(node_id))[:60]
                print(f"    - {label}")
            if len(self.string_matched_nodes) > 5:
                print(f"    ... and {len(self.string_matched_nodes) - 5} more nodes")

        return self.string_matched_nodes

    def run_semantic_matching(
        self,
        seed_nodes: Set = None,
        top_k: int = None,
        similarity_threshold: float = None
    ) -> Set:
        """
        Run semantic embedding matching (Stage 2)

        Prioritizes NPY format embeddings (if loaded), otherwise uses JSON format

        Args:
            seed_nodes: Seed nodes (for computing query vector), defaults to string matching results
            top_k: Number of Top-K similar nodes to return
            similarity_threshold: Similarity threshold

        Returns:
            set: Matched node set
        """
        print("\n[Stage 2] Semantic embedding matching...")

        # Use string matching results as seeds
        if seed_nodes is None:
            seed_nodes = self.string_matched_nodes

        if not seed_nodes:
            print("[Stage 2] Warning: No seed nodes, skipping semantic matching")
            return set()

        # Prioritize NPY format embeddings
        if self.use_npy and self.embeddings_npy is not None:
            print("[Stage 2] Using NPY format embeddings for semantic matching...")
            self.semantic_matched_nodes, self.similarity_scores = find_nodes_by_semantic_similarity_npy(
                self.G,
                self.embeddings_npy,
                self.npy_node_to_idx,
                self.npy_node_labels,
                seed_nodes,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
        elif self.embeddings_data is not None:
            # Fall back to JSON format
            print("[Stage 2] Using JSON format embeddings for semantic matching...")
            self.semantic_matched_nodes, self.similarity_scores = find_nodes_by_semantic_similarity(
                self.G,
                self.embeddings_data,
                seed_nodes,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
        else:
            print("[Stage 2] Warning: No embeddings data available, skipping semantic matching")
            return set()

        self.match_status["semantic_matched_count"] = len(self.semantic_matched_nodes)

        return self.semantic_matched_nodes

    def run_two_stage_matching(
        self,
        target_terms: List[str] = None,
        top_k: int = None,
        similarity_threshold: float = None
    ) -> Set:
        """
        Run complete two-stage matching

        Args:
            target_terms: Target keywords for string matching
            top_k: Top-K for semantic matching
            similarity_threshold: Similarity threshold for semantic matching

        Returns:
            set: Combined matched node set
        """
        print("\n" + "=" * 60)
        print("Step 2: Two-Stage Node Matching")
        print("=" * 60)

        self.match_status["timestamp"] = datetime.now().isoformat()

        # Stage 1: String matching
        self.run_string_matching(target_terms)

        # Stage 2: Semantic matching
        self.run_semantic_matching(
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        # Merge results
        self.combined_seed_nodes = self.string_matched_nodes | self.semantic_matched_nodes
        self.match_status["combined_count"] = len(self.combined_seed_nodes)

        print(f"\n[Step2] Merged results:")
        print(f"    String matching: {len(self.string_matched_nodes)}")
        print(f"    Semantic matching: {len(self.semantic_matched_nodes)}")
        print(f"    After merging: {len(self.combined_seed_nodes)}")

        self.match_status["status"] = "success"

        return self.combined_seed_nodes

    def match_metrics(self, metrics: List[str] = None, use_sapbert: bool = True) -> Dict[str, List]:
        """
        Match network metrics to graph nodes

        Args:
            metrics: Metric name list, defaults to FIXED_METRICS
            use_sapbert: Whether to use SapBERT semantic matching (recommended)

        Returns:
            dict: {metric_name: [node_ids]}
        """
        print("\n[Step2] Matching network metrics to graph nodes...")

        if metrics is None:
            metrics = FIXED_METRICS

        # Method 1: String matching
        self.metrics_to_node_ids = match_metrics_to_nodes(self.G, metrics)
        string_matched = len([m for m, nodes in self.metrics_to_node_ids.items() if nodes])
        print(f"[Step2] String matching: {string_matched}/{len(metrics)} metrics")

        # Method 2: SapBERT semantic matching (supplement unmatched metrics)
        if use_sapbert and self.use_npy and self.embeddings_npy is not None:
            print("[Step2] Using SapBERT for supplementary matching...")
            unmatched_metrics = [m for m in metrics if not self.metrics_to_node_ids.get(m)]

            if unmatched_metrics:
                sapbert_matched = self._match_metrics_by_sapbert(unmatched_metrics)

                # Merge results
                for metric, nodes in sapbert_matched.items():
                    if nodes and metric not in self.metrics_to_node_ids:
                        self.metrics_to_node_ids[metric] = nodes

                sapbert_count = len([m for m, nodes in sapbert_matched.items() if nodes])
                print(f"[Step2] SapBERT supplementary matching: {sapbert_count}/{len(unmatched_metrics)} metrics")

        matched_count = len([m for m, nodes in self.metrics_to_node_ids.items() if nodes])
        self.match_status["metrics_matched_count"] = matched_count

        print(f"[Step2] Total successfully matched {matched_count}/{len(metrics)} metrics")

        # Display partial match results
        print("[Step2] Sample matched metrics:")
        displayed = 0
        for metric, nodes in self.metrics_to_node_ids.items():
            if nodes and displayed < 5:
                print(f"    - {metric}: {len(nodes)} nodes")
                displayed += 1

        return self.metrics_to_node_ids

    def _match_metrics_by_sapbert(
        self,
        metrics: List[str],
        threshold: float = 0.85,
        top_k: int = 3
    ) -> Dict[str, List]:
        """
        Match metrics to nodes using SapBERT semantic matching

        Args:
            metrics: List of metrics to match
            threshold: Similarity threshold
            top_k: Maximum number of nodes to match per metric

        Returns:
            dict: {metric_name: [node_ids]}
        """
        try:
            from sentence_transformers import SentenceTransformer

            # Use SapBERT model
            model_name = self.config.get("SAPBERT_MODEL", "cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
            model = SentenceTransformer(model_name)

            matched = {}

            for metric in metrics:
                # Generate metric variants for matching
                query_text = metric.replace('_', ' ')

                # Get metric embedding
                metric_emb = model.encode([query_text], show_progress_bar=False)[0]
                metric_emb = metric_emb / np.linalg.norm(metric_emb)

                # Calculate similarity with all nodes
                norms = np.linalg.norm(self.embeddings_npy, axis=1, keepdims=True)
                norms[norms == 0] = 1
                normalized_emb = self.embeddings_npy / norms

                similarities = np.dot(normalized_emb, metric_emb)

                # Get top matches
                top_indices = np.argsort(similarities)[::-1][:top_k * 2]

                matches = []
                for idx in top_indices:
                    sim = similarities[idx]
                    if sim >= threshold and len(matches) < top_k:
                        node_id = self.npy_node_ids[idx]
                        matches.append(node_id)

                if matches:
                    matched[metric] = matches
                    label = self.npy_node_labels[top_indices[0]] if top_indices[0] < len(self.npy_node_labels) else "?"
                    print(f"    [SapBERT] {metric} -> {len(matches)} nodes (top: {similarities[top_indices[0]]:.3f}, '{label}')")

            return matched

        except Exception as e:
            print(f"[Step2] SapBERT matching failed: {e}")
            return {}

    def get_matched_nodes_for_metric(self, metric: str) -> List:
        """Get node list for a specific metric"""
        return self.metrics_to_node_ids.get(metric, [])

    def save_results(self, output_path: str = None) -> str:
        """
        Save matching results to JSON file

        Args:
            output_path: Output file path

        Returns:
            str: Output file path
        """
        if output_path is None:
            output_path = get_output_path("step2_matched_nodes")

        # Prepare output data
        output_data = {
            **self.match_status,
            "string_matched_nodes": list(self.string_matched_nodes),
            "semantic_matched_nodes": list(self.semantic_matched_nodes),
            "combined_seed_nodes": list(self.combined_seed_nodes),
            "metrics_to_node_ids": self.metrics_to_node_ids,
            "similarity_scores": {
                str(k): v for k, v in list(self.similarity_scores.items())[:100]
            },  # Only save top 100 similarity scores
            "config_used": {
                "semantic_top_k": self.config.get("SEMANTIC_TOP_K"),
                "similarity_threshold": self.config.get("SEMANTIC_SIMILARITY_THRESHOLD"),
            }
        }

        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[Step2] Results saved to: {output_path}")
        return output_path


# ============================================================
# Main Entry Point
# ============================================================

def load_step1_data(step1_path: str) -> Dict:
    """Load Step1 output data"""
    with open(step1_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Step 2: Node Matcher')
    parser.add_argument('--step1-output', type=str, help='Step 1 output JSON file path')
    parser.add_argument('--gml-path', type=str, help='GML knowledge graph path')
    parser.add_argument('--embeddings-path', type=str, help='SapBERT embeddings JSON path')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--top-k', type=int, default=100, help='Top-K for semantic matching')
    parser.add_argument('--threshold', type=float, default=0.4, help='Similarity threshold')

    args = parser.parse_args()

    # Determine GML path
    gml_path = args.gml_path
    if gml_path is None:
        if CONFIG.get("USE_STAR_SUBGRAPH", True):
            gml_path = CONFIG["STAR_SUBGRAPH_PATH"]
        else:
            gml_path = CONFIG["GML_FILE_PATH"]

    # Determine embeddings path
    embeddings_path = args.embeddings_path or CONFIG["EMBEDDING_PATH"]

    print("\n" + "=" * 60)
    print("Step 2: Node Matcher")
    print("=" * 60)

    # Load knowledge graph
    print(f"\n[Load] Loading knowledge graph: {gml_path}")
    G = nx.read_gml(gml_path, label='id')
    print(f"[Load] Graph loading complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load embeddings
    embeddings_data = None
    try:
        print(f"\n[Load] Loading embeddings: {embeddings_path}")
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            embeddings_data = json.load(f)
        print(f"[Load] Embeddings loading complete")
    except Exception as e:
        print(f"[Load] Warning: Failed to load embeddings: {e}")
        print("[Load] Will use string matching only")

    # Create matcher
    matcher = NodeMatcher(G, embeddings_data)

    # Run two-stage matching
    matcher.run_two_stage_matching(
        top_k=args.top_k,
        similarity_threshold=args.threshold
    )

    # Match network metrics
    matcher.match_metrics()

    # Save results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"step2_matched_nodes_{get_timestamp()}.json")
    matcher.save_results(output_path)

    print("\n" + "=" * 60)
    print("Step 2 Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
