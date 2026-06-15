#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3: Seed Selector Module

Select top seed metrics based on per-dataset AUC scores.

[IMPORTANT NOTE] AUC is ONLY used for seed selection, NOT the sole criterion for Agent formula design!
- AUC helps identify metrics with predictive signal for SEED SELECTION ONLY
- When agents design combination formulas, they should consider:
  * Theoretical/biological meaning (orthogonal dimensions: efficiency x extent x capacity)
  * Complementary information from different metric categories
  * NOT just AUC ranking - low-AUC metrics may contribute crucially when combined

Input Files:
    - outputs/step2_matched_nodes_{timestamp}.json
    - CSV network data file (filtered_phyloseq_network_data.csv)

Output File:
    - outputs/step3_seed_metrics_{timestamp}.json

Usage:
    python step3_seed_selector.py --step2-output outputs/step2_matched_nodes_*.json

Author: Pipeline Modularization Project
Date: 2024-12
"""

import os
import sys
import json
import argparse
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from sklearn.metrics import roc_auc_score, average_precision_score

# Import configuration
from config import CONFIG, FIXED_METRICS, get_timestamp, get_output_path


# ============================================================
# AUC Calculation Functions
# ============================================================

def calculate_metric_auc_per_dataset(
    df: pd.DataFrame,
    metric: str,
    target_col: str = "response_binary"
) -> Tuple[float, float, List]:
    """
    Calculate AUC for a single metric across all datasets.

    Args:
        df: DataFrame (contains combination_key column to identify datasets)
        metric: Metric name
        target_col: Target variable column name

    Returns:
        Tuple[mean_auc, std_auc, dataset_scores]
    """
    if metric not in df.columns:
        return 0.0, 0.0, []

    if "combination_key" not in df.columns:
        datasets = ["all_data"]
        df = df.copy()
        df["combination_key"] = "all_data"
    else:
        datasets = df["combination_key"].unique()

    dataset_scores = []

    for dataset in datasets:
        dataset_df = df[df["combination_key"] == dataset]
        y = dataset_df[target_col].values

        # Skip datasets with only one class
        if len(np.unique(y)) < 2:
            continue

        try:
            X = dataset_df[metric].values

            # Handle missing values
            if np.isnan(X).any():
                X = np.nan_to_num(X, nan=np.nanmean(X))

            # Skip cases with zero variance
            if np.std(X) == 0:
                continue

            auc = roc_auc_score(y, X)
            dataset_scores.append({
                "dataset": dataset,
                "auc": float(auc),
                "n_samples": len(dataset_df)
            })
        except Exception as e:
            continue

    if not dataset_scores:
        return 0.0, 0.0, []

    aucs = [s["auc"] for s in dataset_scores]
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs) if len(aucs) > 1 else 0.0

    return float(mean_auc), float(std_auc), dataset_scores


def calculate_all_metrics_auc(
    df: pd.DataFrame,
    metrics: List[str],
    matched_nodes: Dict[str, List] = None
) -> List[Dict]:
    """
    Calculate AUC scores for all metrics.

    Args:
        df: DataFrame
        metrics: List of metric names
        matched_nodes: Mapping from metrics to nodes (only calculate for metrics with matched nodes)

    Returns:
        List[Dict]: Sorted list of AUC results
    """
    auc_results = []

    for metric in metrics:
        # Only calculate for metrics with matched nodes (if mapping is provided)
        if matched_nodes is not None and metric not in matched_nodes:
            continue

        if metric not in df.columns:
            continue

        mean_auc, std_auc, dataset_scores = calculate_metric_auc_per_dataset(df, metric)

        if mean_auc > 0:
            auc_results.append({
                "metric": metric,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "valid_datasets": len(dataset_scores),
                "dataset_scores": dataset_scores
            })

    # Sort by mean_auc in descending order
    auc_results.sort(key=lambda x: x["mean_auc"], reverse=True)

    return auc_results


# ============================================================
# Seed Selection Functions
# ============================================================

def select_seed_metrics_by_auc(
    df: pd.DataFrame,
    metrics: List[str],
    matched_nodes: Dict[str, List],
    top_k: int = None,
    problem_description: str = None
) -> Tuple[List[str], List[Dict]]:
    """
    Select seed metrics based on per-dataset AUC.

    Args:
        df: DataFrame
        metrics: List of available metrics
        matched_nodes: Mapping from metrics to nodes
        top_k: Number of seeds to select (default 25)
        problem_description: Problem description

    Returns:
        Tuple[selected_metrics, auc_rankings]
    """
    if top_k is None:
        top_k = CONFIG.get("SEED_TOP_K", 25)

    if problem_description is None:
        problem_description = CONFIG.get("DEFAULT_PROBLEM", "")

    print("\n" + "=" * 60)
    print("Seed Metric Selection (by per-dataset AUC)")
    print("=" * 60)
    print(f"【NOTE】AUC guides seed selection only; agents should explore orthogonal dimension products")
    print(f"Research Question: {problem_description[:100]}...")

    # Check required columns
    if "response_binary" not in df.columns:
        print("[Error] 'response_binary' column does not exist")
        return [], []

    # Dataset statistics
    if "combination_key" in df.columns:
        datasets = df["combination_key"].unique()
        print(f"[Step3] Found {len(datasets)} datasets")
    else:
        print("[Step3] combination_key column not found, using entire dataset")

    # Calculate AUC for all metrics
    print("\n[Step3] Calculating AUC scores for each metric...")
    auc_rankings = calculate_all_metrics_auc(df, metrics, matched_nodes)

    print(f"[Step3] Successfully calculated AUC for {len(auc_rankings)} metrics")

    # Display Top-15
    print("\n" + "=" * 60)
    print("Top 15 Metrics by Mean AUC (per-dataset)")
    print("=" * 60)
    for i, result in enumerate(auc_rankings[:15], 1):
        metric = result["metric"]
        mean_auc = result["mean_auc"]
        std_auc = result["std_auc"]
        n_datasets = result["valid_datasets"]
        print(f"  {i:2d}. {metric}: AUC={mean_auc:.4f} (std={std_auc:.4f}, n={n_datasets})")

    # Select Top-K seed metrics
    selected_metrics = []
    for result in auc_rankings:
        metric = result["metric"]
        if metric in matched_nodes and len(selected_metrics) < top_k:
            selected_metrics.append(metric)

    print("\n" + "=" * 60)
    print(f"Selected {len(selected_metrics)} Seed Metrics")
    print("=" * 60)
    for i, metric in enumerate(selected_metrics, 1):
        result = next((r for r in auc_rankings if r["metric"] == metric), None)
        auc = result["mean_auc"] if result else 0
        print(f"  {i:2d}. {metric} (mean_AUC={auc:.4f})")

    return selected_metrics, auc_rankings


# ============================================================
# Main Seed Selector Class
# ============================================================

class SeedSelector:
    """
    Seed Metric Selector

    Usage:
        selector = SeedSelector(df, metrics, matched_nodes)
        selector.select_by_auc()
        selector.save_results()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        matched_nodes: Dict[str, List],
        config: Dict = None
    ):
        """Initialize the seed selector"""
        self.df = df
        self.metrics = metrics
        self.matched_nodes = matched_nodes
        self.config = config or CONFIG

        # Result storage
        self.selected_seeds: List[str] = []
        self.auc_rankings: List[Dict] = []

        # Status information
        self.selection_status: Dict[str, Any] = {
            "timestamp": None,
            "status": "not_started",
            "problem_description": self.config.get("DEFAULT_PROBLEM", ""),
            "total_metrics": len(metrics),
            "metrics_with_nodes": 0,
            "seed_count": 0,
        }

    def select_by_auc(
        self,
        top_k: int = None,
        problem_description: str = None
    ) -> List[str]:
        """
        Select seed metrics by AUC.

        Args:
            top_k: Number of seeds to select
            problem_description: Problem description

        Returns:
            List[str]: List of selected seed metrics
        """
        self.selection_status["timestamp"] = datetime.now().isoformat()

        if problem_description:
            self.selection_status["problem_description"] = problem_description

        # Count metrics with matched nodes
        metrics_with_nodes = [m for m in self.metrics if m in self.matched_nodes]
        self.selection_status["metrics_with_nodes"] = len(metrics_with_nodes)

        # Execute selection
        self.selected_seeds, self.auc_rankings = select_seed_metrics_by_auc(
            self.df,
            self.metrics,
            self.matched_nodes,
            top_k=top_k,
            problem_description=problem_description
        )

        self.selection_status["seed_count"] = len(self.selected_seeds)
        self.selection_status["status"] = "success"

        return self.selected_seeds

    def get_seed_nodes(self) -> Dict[str, List]:
        """
        Get node mapping for seed metrics.

        Returns:
            Dict[str, List]: {metric: [node_ids]}
        """
        return {
            metric: self.matched_nodes.get(metric, [])
            for metric in self.selected_seeds
            if metric in self.matched_nodes
        }

    def get_auc_for_metric(self, metric: str) -> Optional[Dict]:
        """Get AUC information for a specific metric"""
        for result in self.auc_rankings:
            if result["metric"] == metric:
                return result
        return None

    def save_results(self, output_path: str = None) -> str:
        """
        Save selection results to JSON file.

        Args:
            output_path: Output file path

        Returns:
            str: Output file path
        """
        if output_path is None:
            output_path = get_output_path("step3_seed_metrics")

        # Prepare output data
        output_data = {
            **self.selection_status,
            "selected_seeds": self.selected_seeds,
            "seed_nodes": self.get_seed_nodes(),
            "auc_rankings": self.auc_rankings[:50],  # Only save top 50
            "config_used": {
                "seed_top_k": self.config.get("SEED_TOP_K"),
                "max_hop": self.config.get("MAX_HOP"),
            }
        }

        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[Step3] Results saved to: {output_path}")
        return output_path


# ============================================================
# Main Entry Point
# ============================================================

def load_step2_data(step2_path: str) -> Dict:
    """Load Step2 output data"""
    with open(step2_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Step 3: Seed Selector')
    parser.add_argument('--step2-output', type=str, help='Step 2 output JSON file path')
    parser.add_argument('--csv-path', type=str, help='CSV network data file path')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--top-k', type=int, default=25, help='Number of seed metrics to select')
    parser.add_argument('--problem', type=str, help='Problem description')

    args = parser.parse_args()

    # Determine CSV path
    csv_path = args.csv_path or CONFIG["CSV_DATA_PATH"]

    print("\n" + "=" * 60)
    print("Step 3: Seed Selector")
    print("=" * 60)

    # Load CSV data
    print(f"\n[Load] Loading CSV data: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[Load] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Check or create response_binary
    if "response_binary" not in df.columns:
        if "response" in df.columns:
            if df["response"].dtype == 'object':
                df["response_binary"] = (df["response"] == "good").astype(int)
            else:
                median = df["response"].median()
                df["response_binary"] = (df["response"] > median).astype(int)
            print(f"[Load] Created response_binary column")
        else:
            print("[Error] Missing response or response_binary column!")
            return 1

    # Get metric list
    exclude_cols = ["combination_key", "response", "response_binary"]
    metrics = [col for col in df.columns if col not in exclude_cols]
    print(f"[Load] Found {len(metrics)} metric columns")

    # Load Step2 data or create default matched_nodes
    matched_nodes = {}
    if args.step2_output:
        print(f"\n[Load] Loading Step2 data: {args.step2_output}")
        step2_data = load_step2_data(args.step2_output)
        matched_nodes = step2_data.get("metrics_to_node_ids", {})
        print(f"[Load] Loaded node mapping for {len(matched_nodes)} metrics")
    else:
        # If no Step2 data, assume all metrics are matched (for testing)
        print("[Load] No Step2 output provided, assuming all metrics are valid")
        matched_nodes = {m: [f"node_{m}"] for m in metrics}

    # Create selector
    selector = SeedSelector(df, metrics, matched_nodes)

    # Execute selection
    problem = args.problem or CONFIG.get("DEFAULT_PROBLEM", "")
    selector.select_by_auc(top_k=args.top_k, problem_description=problem)

    # Save results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"step3_seed_metrics_{get_timestamp()}.json")
    selector.save_results(output_path)

    print("\n" + "=" * 60)
    print("Step 3 Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
