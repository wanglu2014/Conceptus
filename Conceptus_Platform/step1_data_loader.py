#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Data Loader Module

This module handles loading and validating all input files:
- GML knowledge graph file
- CSV network metrics data
- JSON metric-to-node mappings
- SapBERT embeddings

Input Files:
    - deduplicated_graph_with_metrics.gml (or star subgraph)
    - filtered_phyloseq_network_data.csv
    - metric_to_nodes.json
    - sapbert_embeddings.json

Output File:
    - outputs/step1_data_loaded_{timestamp}.json

Usage:
    python step1_data_loader.py --output-dir outputs/

Author: Pipeline Modularization Project
Date: 2024-12
"""

import os
import sys
import json
import argparse
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import networkx as nx
import numpy as np

# Import configuration
from config import CONFIG, FIXED_METRICS, get_timestamp, get_output_path


# ============================================================
# Data Loading Functions
# ============================================================

def load_network_data(csv_path: str = None) -> Optional[pd.DataFrame]:
    """
    Load network metrics data from CSV file

    Args:
        csv_path: CSV file path, defaults to CONFIG path if not provided

    Returns:
        pandas.DataFrame: Dataset containing network metrics and response variables
    """
    csv_path = csv_path or CONFIG["CSV_DATA_PATH"]

    try:
        # Load data
        df = pd.read_csv(csv_path)
        print(f"[Step1] CSV data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Get metric columns (excluding response variables and group identifiers)
        exclude_cols = ["combination_key", "response", "response_binary"]
        metrics = [col for col in df.columns if col not in exclude_cols]
        print(f"[Step1] Found {len(metrics)} network metrics")

        # Check or create response_binary
        if "response_binary" not in df.columns:
            if "response" in df.columns:
                # Create binary response variable
                if df["response"].dtype == 'object':
                    # If text type, set "good" to 1
                    df["response_binary"] = (df["response"] == "good").astype(int)
                else:
                    # If numeric, use median as threshold
                    median = df["response"].median()
                    df["response_binary"] = (df["response"] > median).astype(int)

                counts = df["response_binary"].value_counts().to_dict()
                print(f"[Step1] Converted response to binary: {counts}")
            else:
                print("[Step1] Warning: response column not found, cannot create response_binary")

        # Check or create combination_key
        if "combination_key" not in df.columns:
            print("[Step1] combination_key column not found, creating default group...")
            df["combination_key"] = "default_group"

        # Print group statistics
        groups = df["combination_key"].unique()
        print(f"[Step1] Data contains {len(groups)} datasets/groups")

        return df

    except Exception as e:
        print(f"[Step1] Error loading CSV data: {e}")
        traceback.print_exc()
        return None


def load_knowledge_graph(gml_path: str = None, use_star_subgraph: bool = None) -> Optional[nx.Graph]:
    """
    Load knowledge graph from GML file

    Args:
        gml_path: GML file path
        use_star_subgraph: Whether to use star subgraph (faster), uses CONFIG setting if None

    Returns:
        networkx.Graph: Knowledge graph
    """
    # Determine which GML file to use
    if gml_path is None:
        if use_star_subgraph is None:
            use_star_subgraph = CONFIG.get("USE_STAR_SUBGRAPH", True)

        if use_star_subgraph:
            gml_path = CONFIG["STAR_SUBGRAPH_PATH"]
            print(f"[Step1] Loading mode: Star subgraph (fast mode)")
        else:
            gml_path = CONFIG["GML_FILE_PATH"]
            print(f"[Step1] Loading mode: Full knowledge graph")

    try:
        print(f"[Step1] Loading: {gml_path}")
        G = nx.read_gml(gml_path, label='id')
        print(f"[Step1] Successfully loaded knowledge graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Count node type distribution
        node_types = {}
        for _, data in G.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1

        print(f"[Step1] Node type distribution: {dict(list(node_types.items())[:5])}")

        return G

    except Exception as e:
        print(f"[Step1] Error loading knowledge graph: {e}")
        traceback.print_exc()
        return None


def load_metrics_dictionary(dict_path: str = None,
                           knowledge_graph: nx.Graph = None) -> Dict[str, List]:
    """
    Load metric-to-node mapping dictionary

    Prioritizes extraction from knowledge graph node attributes, falls back to JSON file if failed

    Args:
        dict_path: JSON dictionary file path
        knowledge_graph: Knowledge graph object (optional, used for extracting from node attributes)

    Returns:
        dict: {metric_name: [node_ids]}
    """
    dict_path = dict_path or CONFIG["METRICS_DICT_PATH"]
    metrics_dict = {}

    # Method 1: Extract from knowledge graph node attributes (preferred)
    if knowledge_graph is not None:
        print("[Step1] Extracting metric mappings from knowledge graph node attributes...")
        for node_id, node_data in knowledge_graph.nodes(data=True):
            # Check if extracted_term attribute exists
            if 'extracted_term' in node_data:
                metric_name = node_data['extracted_term']
                if metric_name not in metrics_dict:
                    metrics_dict[metric_name] = []
                metrics_dict[metric_name].append(node_id)

        if metrics_dict:
            print(f"[Step1] Extracted mappings for {len(metrics_dict)} metrics from graph")
            return metrics_dict
        else:
            print("[Step1] Warning: extracted_term attribute not found in graph, trying to load from JSON...")

    # Method 2: Load from JSON file (fallback)
    try:
        with open(dict_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "node_dict" in data:
            # Old format: {"node_dict": {node_id: {"metric_term": metric, ...}, ...}}
            print(f"[Step1] Using node_dict format")
            for node_id, info in data["node_dict"].items():
                metric = info.get("metric_term")
                if metric:
                    if metric not in metrics_dict:
                        metrics_dict[metric] = []
                    metrics_dict[metric].append(node_id)

        elif "simplified_mapping" in data:
            # New format: {"simplified_mapping": {metric: [node_ids], ...}}
            print(f"[Step1] Using simplified_mapping format")
            metrics_dict = data["simplified_mapping"]

        else:
            # Direct format: {metric: [node_ids], ...} or {keyword: [[node_id, label], ...], ...}
            print(f"[Step1] Using direct format")
            for keyword, node_data in data.items():
                if keyword in FIXED_METRICS:
                    if isinstance(node_data, list) and len(node_data) > 0:
                        if isinstance(node_data[0], list):
                            metrics_dict[keyword] = [item[0] for item in node_data]
                        else:
                            metrics_dict[keyword] = node_data

        print(f"[Step1] Successfully loaded metrics dictionary: {len(metrics_dict)} metrics")
        return metrics_dict

    except Exception as e:
        print(f"[Step1] Error loading metrics dictionary: {e}")
        traceback.print_exc()
        return {}


def load_embeddings(embeddings_path: str = None) -> Dict[str, Any]:
    """
    Load pre-computed SapBERT embedding vectors

    Args:
        embeddings_path: Embeddings file path

    Returns:
        dict: Embeddings data dictionary
    """
    embeddings_path = embeddings_path or CONFIG["EMBEDDING_PATH"]

    try:
        print(f"[Step1] Loading embeddings: {embeddings_path}")
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Count number of embeddings
        if isinstance(data, dict):
            if 'embeddings' in data:
                count = len(data['embeddings'])
            else:
                count = len(data)
        else:
            count = len(data)

        print(f"[Step1] Successfully loaded {count} node embeddings")
        return data

    except Exception as e:
        print(f"[Step1] Error loading embeddings: {e}")
        traceback.print_exc()
        return {}


# ============================================================
# Main Data Loader Class
# ============================================================

class DataLoader:
    """
    Data Loader Class - Unified management for all data loading operations

    Usage:
        loader = DataLoader()
        loader.load_all()
        loader.save_status()
    """

    def __init__(self, config: Dict = None):
        """Initialize data loader"""
        self.config = config or CONFIG

        # Data storage
        self.df: Optional[pd.DataFrame] = None
        self.graph: Optional[nx.Graph] = None
        self.metrics_dict: Dict[str, List] = {}
        self.embeddings: Dict[str, Any] = {}

        # Status information
        self.load_status: Dict[str, Any] = {
            "timestamp": None,
            "status": "not_started",
            "csv_loaded": False,
            "graph_loaded": False,
            "metrics_dict_loaded": False,
            "embeddings_loaded": False,
        }

    def load_all(self,
                csv_path: str = None,
                gml_path: str = None,
                dict_path: str = None,
                embeddings_path: str = None,
                use_star_subgraph: bool = None) -> bool:
        """
        Load all data files

        Returns:
            bool: Whether all files loaded successfully
        """
        print("\n" + "=" * 60)
        print("Step 1: Data Loading")
        print("=" * 60)

        self.load_status["timestamp"] = datetime.now().isoformat()
        success = True

        # 1. Load CSV data
        print("\n[1/4] Loading CSV data...")
        self.df = load_network_data(csv_path)
        if self.df is not None:
            self.load_status["csv_loaded"] = True
            self.load_status["csv_stats"] = {
                "rows": int(self.df.shape[0]),
                "columns": int(self.df.shape[1]),
                "groups": list(self.df["combination_key"].unique()) if "combination_key" in self.df.columns else [],
            }
        else:
            success = False

        # 2. Load knowledge graph
        print("\n[2/4] Loading knowledge graph...")
        self.graph = load_knowledge_graph(gml_path, use_star_subgraph)
        if self.graph is not None:
            self.load_status["graph_loaded"] = True
            self.load_status["graph_stats"] = {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
            }
        else:
            success = False

        # 3. Load metrics dictionary
        print("\n[3/4] Loading metrics dictionary...")
        self.metrics_dict = load_metrics_dictionary(dict_path, self.graph)
        if self.metrics_dict:
            self.load_status["metrics_dict_loaded"] = True
            self.load_status["metrics_count"] = len(self.metrics_dict)
        else:
            print("[Warning] Metrics dictionary is empty, continuing execution...")

        # 4. Load embeddings
        print("\n[4/4] Loading embeddings...")
        self.embeddings = load_embeddings(embeddings_path)
        if self.embeddings:
            self.load_status["embeddings_loaded"] = True
            if isinstance(self.embeddings, dict):
                if 'embeddings' in self.embeddings:
                    self.load_status["embeddings_count"] = len(self.embeddings['embeddings'])
                else:
                    self.load_status["embeddings_count"] = len(self.embeddings)

        # 5. Determine available metrics
        if self.df is not None:
            exclude_cols = ["combination_key", "response", "response_binary"]
            available_metrics = [col for col in self.df.columns if col not in exclude_cols]
            self.load_status["available_metrics"] = available_metrics
            print(f"\n[Step1] Available metrics: {len(available_metrics)}")

        # Update status
        self.load_status["status"] = "success" if success else "partial_failure"

        print("\n" + "=" * 60)
        print(f"Data Loading Complete: {self.load_status['status']}")
        print("=" * 60)

        return success

    def get_available_metrics(self) -> List[str]:
        """Get list of available metrics from CSV"""
        if self.df is None:
            return []
        exclude_cols = ["combination_key", "response", "response_binary"]
        return [col for col in self.df.columns if col not in exclude_cols]

    def get_matched_metrics(self) -> List[str]:
        """Get metrics that exist in both CSV and metrics dictionary"""
        available = set(self.get_available_metrics())
        in_dict = set(self.metrics_dict.keys())
        return list(available & in_dict)

    def save_status(self, output_path: str = None) -> str:
        """
        Save loading status to JSON file

        Args:
            output_path: Output file path

        Returns:
            str: Output file path
        """
        if output_path is None:
            output_path = get_output_path("step1_data_loaded")

        # Prepare output data
        output_data = {
            **self.load_status,
            "config_used": {
                "csv_path": self.config.get("CSV_DATA_PATH"),
                "gml_path": self.config.get("GML_FILE_PATH"),
                "metrics_dict_path": self.config.get("METRICS_DICT_PATH"),
                "embeddings_path": self.config.get("EMBEDDING_PATH"),
            }
        }

        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[Step1] Status saved to: {output_path}")
        return output_path


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Step 1: Data Loader')
    parser.add_argument('--csv-path', type=str, help='CSV data file path')
    parser.add_argument('--gml-path', type=str, help='GML knowledge graph path')
    parser.add_argument('--dict-path', type=str, help='Metrics dictionary JSON path')
    parser.add_argument('--embeddings-path', type=str, help='SapBERT embeddings JSON path')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--use-star-subgraph', action='store_true', help='Use star subgraph (faster)')
    parser.add_argument('--use-full-graph', action='store_true', help='Use full knowledge graph')

    args = parser.parse_args()

    # Determine whether to use star subgraph
    use_star = None
    if args.use_star_subgraph:
        use_star = True
    elif args.use_full_graph:
        use_star = False

    # Create data loader
    loader = DataLoader()

    # Load all data
    success = loader.load_all(
        csv_path=args.csv_path,
        gml_path=args.gml_path,
        dict_path=args.dict_path,
        embeddings_path=args.embeddings_path,
        use_star_subgraph=use_star
    )

    # Save status
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"step1_data_loaded_{get_timestamp()}.json")
    loader.save_status(output_path)

    # Return status code
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
