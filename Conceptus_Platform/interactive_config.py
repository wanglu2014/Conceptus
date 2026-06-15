# -*- coding: utf-8 -*-
"""
Interactive Pipeline Configuration
===================================
Configuration for interactive concept input -> KG generation -> formula discovery pipeline.
"""

INTERACTIVE_CONFIG = {
    # Mode configurations
    "MODES": {
        "demo": {
            "literature_max": 30,      # Max articles per keyword
            "iterations": 2,           # Evolution iterations
            "batch_size": 5,           # Batch size for LLM calls
            "description": "Quick demo mode (~30 minutes)"
        },
        "full": {
            "literature_max": 100,     # Max articles per keyword
            "iterations": 5,           # Evolution iterations
            "batch_size": 10,          # Batch size for LLM calls
            "description": "Full analysis mode (~2 hours)"
        }
    },

    # Progress bar settings
    "PROGRESS_BAR": True,              # Use tqdm for progress display

    # Literature sources
    "LITERATURE_SOURCES": {
        "pubmed": True,
        "arxiv": True,
        "wikipedia": True
    },

    # Output directory template
    "OUTPUT_DIR_TEMPLATE": "outputs/interactive/session_{timestamp}",

    # Subdirectory structure
    "SUBDIRS": {
        "literature": "literature_data",
        "kg": "kg_output",
        "pipeline": "pipeline_output"
    },

    # API configuration (inherited from main config)
    "API": {
        "timeout": 120,
        "max_retries": 3,
        "retry_delay": 5
    },

    # Default values
    "DEFAULTS": {
        "mode": "demo",
        "language": "en"
    }
}

# Phase descriptions for progress display
PHASE_DESCRIPTIONS = {
    "concept_extraction": "Phase 1: Extracting concepts from research question",
    "literature_download": "Phase 2: Downloading literature from databases",
    "kg_construction": "Phase 3: Building knowledge graph from literature",
    "formula_discovery": "Phase 4: Running formula discovery pipeline (Step 1-7)"
}

# Time estimates (for user information)
TIME_ESTIMATES = {
    "demo": {
        "concept_extraction": "~30 seconds",
        "literature_download": "~3 minutes",
        "kg_construction": "~5 minutes",
        "formula_discovery": "~20 minutes",
        "total": "~30 minutes"
    },
    "full": {
        "concept_extraction": "~30 seconds",
        "literature_download": "~30 minutes",
        "kg_construction": "~45 minutes",
        "formula_discovery": "~45 minutes",
        "total": "~2 hours"
    }
}
