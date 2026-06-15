#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modular Knowledge Graph Analysis Pipeline - Global Configuration

This module contains all configuration parameters for the modular pipeline.
Each step module imports from this file to ensure consistency.

Author: Pipeline Modularization Project
Date: 2024-12
"""

import os
from typing import Dict, List, Any

# ============================================================
# Base Paths Configuration
# ============================================================
# Base directory - using relative paths for portability
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Main Configuration Dictionary
# ============================================================
CONFIG: Dict[str, Any] = {
    # ========== Data File Paths (relative paths) ==========
    "CSV_DATA_PATH": os.path.join(DATA_DIR, "filtered_phyloseq_network_data.csv"),
    "GML_FILE_PATH": os.path.join(DATA_DIR, "deduplicated_graph_with_metrics_v2.gml"),
    "STAR_SUBGRAPH_PATH": os.path.join(DATA_DIR, "antifragility_STAR_1hop_20251203_235150.gml"),
    "METRICS_DICT_PATH": os.path.join(DATA_DIR, "metric_to_nodes.json"),
    # Legacy JSON format (157 nodes, for compatibility)
    "EMBEDDING_PATH": os.path.join(DATA_DIR, "sapbert_embeddings.json"),

    # New NPY format (820166 nodes, complete)
    "EMBEDDING_NPY_PATH": os.path.join(DATA_DIR, "node_embeddings.npy"),
    "NODE_MAPPING_PATH": os.path.join(DATA_DIR, "node_mapping.json"),
    "KEYS_FILE_PATH": os.path.join(DATA_DIR, "keys.csv"),  # Users must configure their own API keys

    # ========== Output Directory ==========
    "OUTPUT_DIR": OUTPUT_DIR,

    # ========== Star Subgraph Configuration ==========
    "USE_STAR_SUBGRAPH": False,  # True=use pre-extracted star subgraph for faster loading, False=use full graph (with wiki edges)

    # ========== RAG Parameters ==========
    "TOP_K_METRICS": 10,          # Select Top-K metrics with highest AUC for RAG
    "MAX_HOP": 2,                 # RAG max hops (0=direct, 1=allow 1 intermediate node, 2=allow 2)
    "MAX_CONTEXT_CHARS": 5000,    # Max characters to pass to agents

    # ========== Evolution Parameters ==========
    "MAX_ITERATIONS": 3,          # Max iterations per cross-validation fold
    "POPULATION_SIZE": 10,        # Number of best combinations to retain per evolution
    "MAX_FORMULA_VARIABLES": 3,   # Max variables allowed in formula
    "MUTATION_RATE": 0.2,         # Mutation rate
    "CROSSOVER_RATE": 0.7,        # Crossover rate
    "FORMULA_MAX_TOKENS": 100,    # Max tokens in formula
    "NUM_COMBINATIONS": 5,        # Number of combinations to generate per iteration
    "IMPROVEMENT_THRESHOLD": 0.01, # Evolution convergence threshold

    # ========== Node Matching Parameters ==========
    "SEMANTIC_TOP_K": 100,              # Number of candidate nodes returned by semantic matching
    "SEMANTIC_SIMILARITY_THRESHOLD": 0.4,  # Semantic similarity threshold

    # ========== Step 3.5 Keyword Expansion Configuration ==========
    "SIMILARITY_THRESHOLD": 0.9,        # SapBERT keyword matching threshold
    "KEYWORD_TOP_K_MATCHES": 5,         # Max nodes to match per keyword

    # ========== Seed Selection Parameters ==========
    "SEED_TOP_K": 25,             # Select Top-25 seed metrics

    # ========== API Configuration ==========
    # OpenAI-compatible API (primary)
    "API_BASE_URL": "https://api.openai.com/v1",
    "API_MODEL": "gpt-4",
    
    # DeepSeek (backup)
    "DEEPSEEK_BASE_URL": "https://api.deepseek.com/v1",
    "DEEPSEEK_MODEL": "deepseek-reasoner",
    
    # API Keys - Read from keys.csv (users must configure their own keys)
    "API_KEY_PRIMARY": "",  # Primary API key (set via keys.csv or environment variable)
    "API_KEY_BACKUP": "",   # Backup API key
    "API_KEY_THIRD": "",    # Third backup API key
    "API_TIMEOUT": 180,           # API request timeout (seconds)
    "API_MAX_RETRIES": 3,         # Maximum API retry attempts

    # ========== Problem Description ==========
    "DEFAULT_PROBLEM": "What multi-term arithmetic combinations of patient specific microbial network attributes in patients most accurately forecast microbiome antifragility, thereby predicting disease outcome?",

    # ========== Entity Extraction Configuration ==========
    "ENTITY_EXTRACTION": {
        "ENABLED": True,
        "SIMILARITY_THRESHOLD": 0.90,
        "TOP_K_MATCHES": 5,
    },

    # ========== Knowledge Pump Configuration ==========
    "ENABLE_KNOWLEDGE_PUMP": True,
    "KNOWLEDGE_PUMP_NODE_TYPES": ["metric", "concept", "variable"],
    "KNOWLEDGE_PUMP_EDGE_TYPES": ["Influence", "Calculation", "Definition"],
    "KNOWLEDGE_PUMP_MAX_TOKENS": 2000,

    # ========== Enhanced Knowledge Pump Configuration ==========
    "USE_ENHANCED_KNOWLEDGE_PUMP": True,
    "ENHANCED_KP_N_CLUSTERS": 15,          # SBERT clustering count
    "ENHANCED_KP_REPRESENTATIVES_PER_CLUSTER": 2,  # Representatives per cluster
    "ENHANCED_KP_MAX_CHARS": 8000,         # Max output characters
    "ENHANCED_KP_MIN_CHARS": 5000,         # Min output characters
    "ENHANCED_KP_MAX_HOP": 2,              # Max path hops
    "ENHANCED_KP_MAX_PATHS_PER_PAIR": 3,   # Max paths per metric pair

    # ========== Unsupervised Dimension Clustering Configuration ==========
    "KP_N_DIMENSION_CLUSTERS": 3,          # Metric dimension cluster count (default 3, configurable)
    "KP_USE_AUTO_CLUSTERING": True,        # Enable SBERT auto-clustering
    "KP_CLUSTERING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",  # SBERT model for clustering

    # ========== SBERT Configuration ==========
    # SapBERT: for entity embedding matching (node_embeddings.npy generated with this model)
    "SAPBERT_MODEL": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    # BioLORD-2023: for sentence embedding (edge description clustering)
    "SBERT_MODEL": "FremyCompany/BioLORD-2023",
    "SBERT_N_CLUSTERS": 15,       # Edge cluster count
    "SBERT_TOP_EDGES": 50,        # Top edges to retain

    # ========== Sample Dropout Configuration ==========
    "SAMPLE_DROPOUT_RATE": 0.0,      # Sample dropout rate (0.0-0.5)
    "SAMPLE_DROPOUT_TRIALS": 3,      # Dropout evaluation trials for averaging

    # ========== File Retention Configuration ==========
    "MAX_FILES_RETENTION": 5,        # Retain latest N output files

    # ========== Conceptus Platform Configuration ==========
    "CONCEPTUS_OUTPUT_DIR": os.path.join(OUTPUT_DIR, "conceptus/"),
    "THEORY_LIBRARY_PATH": os.path.join(OUTPUT_DIR, "theory_library.json"),
}

# ============================================================
# Fixed Metrics List (45 network metrics)
# ============================================================
FIXED_METRICS: List[str] = [
    "number_of_nodes", "number_of_edges", "is_directed", "is_connected",
    "number_of_connected_components", "density", "largest_cc_size", "largest_cc_ratio",
    "average_degree", "min_degree", "max_degree", "degree_distribution_entropy",
    "degree_variance", "degree_coef_variation", "network_heterogeneity",
    "average_clustering", "min_clustering", "max_clustering",
    "mean_degree_centrality", "max_degree_centrality",
    "mean_betweenness_centrality", "max_betweenness_centrality",
    "mean_closeness_centrality", "max_closeness_centrality",
    "mean_eigenvector_centrality", "max_eigenvector_centrality",
    "mean_core_number", "max_core_number",
    "number_of_communities", "mean_community_size", "max_community_size",
    "modularity", "mean_rich_club_coefficient", "max_rich_club_coefficient",
    "spectral_radius", "graph_energy", "min_eigenvalue", "max_eigenvalue",
    "algebraic_connectivity", "average_shortest_path_length", "diameter",
    "radius", "mean_eccentricity", "min_eccentricity", "max_eccentricity"
]

# ============================================================
# String Match Terms for Antifragility Node Matching
# ============================================================
STRING_MATCH_TERMS: List[str] = [
    'antifragility', 'antifragile', 'fragility', 'fragile',
    'resilience', 'resilient', 'robustness', 'robust',
    'stability', 'stable', 'instability', 'unstable',
    'vulnerability', 'vulnerable', 'perturbation',
    'stress response', 'adaptation', 'recovery',
    'fault tolerance', 'redundancy'
]

# ============================================================
# Metrics Keywords for Knowledge Graph Node Matching
# ============================================================
METRICS_KEYWORDS: List[str] = [
    "number_of_nodes", "number_of_edges", "is_directed", "is_connected",
    "number_of_connected_components", "network_density", "largest_cc_size", "largest_cc_ratio",
    "average_degree", "min_degree", "max_degree", "degree_distribution_entropy",
    "degree_variance", "degree_coef_variation", "network_heterogeneity",
    "average_clustering_coefficient", "min_clustering_coefficient", "max_clustering_coefficient",
    "mean_degree", "max_degree",
    "mean_betweenness", "max_betweenness",
    "mean_closeness", "max_closeness", "closeness_centrality",
    "mean_eigenvector", "max_eigenvector",
    "mean_core_number", "max_core_number",
    "number_of_communities", "mean_community_size", "max_community_size",
    "network_modularity", "mean_rich_club_coefficient", "max_rich_club_coefficient",
    "spectral_radius", "graph_energy", "min_eigenvalue_ratio", "max_eigenvalue_ratio",
    "algebraic_connectivity", "average_shortest_path_length", "network_diameter",
    "network_radius", "mean_eccentricity", "min_eccentricity", "max_eccentricity"
]

# ============================================================
# Dimension Categories for Knowledge Pump
# ============================================================
DIMENSION_CATEGORIES: Dict[str, List[str]] = {
    "reach": [  # Reach/Efficiency Dimension
        "closeness", "betweenness", "efficiency", "path_length",
        "shortest_path", "centrality"
    ],
    "span": [   # Span/Extent Dimension
        "diameter", "radius", "eccentricity", "spread",
        "extent", "coverage"
    ],
    "scale": [  # Scale/Capacity Dimension
        "nodes", "edges", "size", "capacity", "degree",
        "density", "communities"
    ],
    "cross": [  # Cross-Dimension Combinations
        "modularity", "clustering", "heterogeneity",
        "connectivity", "resilience"
    ]
}

# ============================================================
# Agent Prompts Configuration
# ============================================================
AGENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "MathAgent": {
        "name": "MathAgent",
        "role": "Mathematical Formula Expert",
        "focus": "orthogonal dimension combinations",
        "output_count": 3,
    },
    "BioAgent": {
        "name": "BioAgent",
        "role": "Biological Interpretation Expert",
        "focus": "biological meaning and mechanism",
        "output_count": 3,
    },
    "IntegrationAgent": {
        "name": "IntegrationAgent",
        "role": "Integration and Synthesis Expert",
        "focus": "combining Math and Bio perspectives",
        "output_count": 5,
    }
}

# ============================================================
# Validation Rules for Formula Parsing
# ============================================================
FORMULA_VALIDATION: Dict[str, Any] = {
    "allowed_operators": ['+', '-', '*', '/', '(', ')'],
    "forbidden_patterns": [
        r'\+\s*[\d.]+',      # Forbid constant addition: +1, +0.5
        r'-\s*[\d.]+',       # Forbid constant subtraction: -1, -0.5
        r'\*\s*[\d.]+',      # Forbid coefficient multiplication: *2, *3
        r'/\s*[\d.]+',       # Forbid coefficient division: /2, /3
        r'[\d.]+\s*\*',      # Forbid prefix coefficient: 2*, 0.5*
        r'[\d.]+\s*/',       # Forbid prefix division: 2/, 0.5/
    ],
    "min_variables": 2,       # Minimum variable count
    "max_variables": 3,       # Maximum variable count
}

# ============================================================
# Utility Functions
# ============================================================
def get_timestamp() -> str:
    """Generate timestamp string for file naming"""
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def get_output_path(prefix: str, extension: str = "json") -> str:
    """Generate output file path with timestamp"""
    timestamp = get_timestamp()
    filename = f"{prefix}_{timestamp}.{extension}"
    return os.path.join(OUTPUT_DIR, filename)


def validate_paths() -> Dict[str, bool]:
    """Validate that all required input files exist"""
    required_files = {
        "CSV_DATA_PATH": CONFIG["CSV_DATA_PATH"],
        "GML_FILE_PATH": CONFIG["GML_FILE_PATH"],
        "METRICS_DICT_PATH": CONFIG["METRICS_DICT_PATH"],
        "EMBEDDING_PATH": CONFIG["EMBEDDING_PATH"],
        "KEYS_FILE_PATH": CONFIG["KEYS_FILE_PATH"],
    }

    results = {}
    for name, path in required_files.items():
        exists = os.path.exists(path)
        results[name] = exists
        if not exists:
            print(f"[WARNING] File not found: {name} -> {path}")

    return results


# ============================================================
# Module Test
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Configuration Module Test")
    print("=" * 60)

    print(f"\nBase Directory: {BASE_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")

    print(f"\nFixed Metrics: {len(FIXED_METRICS)} items")
    print(f"String Match Terms: {len(STRING_MATCH_TERMS)} items")

    print("\n--- Path Validation ---")
    validation = validate_paths()
    for name, exists in validation.items():
        status = "OK" if exists else "MISSING"
        print(f"  {name}: {status}")

    print(f"\n--- Key Configuration Values ---")
    for key in ["MAX_ITERATIONS", "POPULATION_SIZE", "MAX_FORMULA_VARIABLES",
                "SEED_TOP_K", "SEMANTIC_SIMILARITY_THRESHOLD"]:
        print(f"  {key}: {CONFIG.get(key)}")

    print("\n" + "=" * 60)
    print("Configuration loaded successfully!")
    print("=" * 60)
