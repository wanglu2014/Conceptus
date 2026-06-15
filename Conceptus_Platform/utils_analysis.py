#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis Utilities Module

Provides formula family analysis and multi-dimensional evaluation.

Features:
  - calculate_additional_metrics: Add robustness, simplicity metrics
  - display_formula_families: Show formulas grouped by accuracy/robustness/simplicity
  - count_unique_metrics: Count variables in formula

Author: Pipeline Modularization Project
Date: 2024-12
"""

import re
from typing import List, Dict, Any


def count_unique_metrics(formula: str) -> int:
    """
    Count unique network metric variables in formula

    Args:
        formula: Formula string

    Returns:
        int: Number of unique metric variables
    """
    if not formula:
        return 0

    # Extract all variable names
    variables = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)', formula)

    # Exclude math functions
    math_funcs = {
        'log', 'sqrt', 'abs', 'max', 'min', 'exp',
        'sin', 'cos', 'tan', 'np', 'log10', 'pow'
    }

    unique_vars = set(v for v in variables if v not in math_funcs)
    return len(unique_vars)


def calculate_additional_metrics(combination: Dict) -> Dict:
    """
    Calculate additional evaluation metrics for a combination

    Adds:
      - robustness: Based on std_auc (lower std = higher robustness)
      - simplicity: Based on metric count (fewer = higher simplicity)
      - num_metrics: Number of unique metrics in formula

    Args:
        combination: Combination dictionary with 'formula', 'mean_auc', 'std_auc'

    Returns:
        Dict: Updated combination with additional metrics
    """
    # Metric 1: Accuracy (already exists)
    accuracy = combination.get("mean_auc", 0)

    # Metric 2: Robustness (inverse of std_auc)
    std_auc = combination.get("std_auc", 0.1)
    robustness = 1.0 / (1.0 + std_auc)  # Lower std = higher robustness

    # Metric 3: Simplicity (inverse of metric count)
    formula = combination.get("formula", "")
    num_metrics = count_unique_metrics(formula)
    simplicity = 1.0 / max(num_metrics, 1)  # Fewer metrics = higher simplicity

    combination["robustness"] = robustness
    combination["simplicity"] = simplicity
    combination["num_metrics"] = num_metrics

    return combination


def display_formula_families(combinations: List[Dict], top_n: int = 10):
    """
    Display formulas grouped by different characteristics (families)

    Families:
      1. High Accuracy (sorted by mean_auc)
      2. High Robustness (sorted by robustness, stable across datasets)
      3. High Simplicity (sorted by simplicity, fewer metrics)

    Args:
        combinations: List of combination dictionaries
        top_n: Number of formulas to show per family
    """
    if not combinations:
        print("No combinations to display")
        return

    # Calculate additional metrics for all combinations
    for comb in combinations:
        calculate_additional_metrics(comb)

    print("\n" + "=" * 80)
    print("【Formula Family Analysis】")
    print("=" * 80)

    # Family 1: High Accuracy Formulas (sorted by mean_auc)
    print("\n【Family 1: High Accuracy Formulas】(sorted by Accuracy)")
    print("-" * 80)
    top_accuracy = sorted(combinations, key=lambda x: x.get("mean_auc", 0), reverse=True)[:top_n]
    for i, comb in enumerate(top_accuracy, 1):
        name = comb.get('name', 'Unknown')[:50]
        print(f"{i}. {name}")
        print(f"   Formula: {comb.get('formula', 'N/A')}")
        print(f"   Accuracy={comb.get('mean_auc', 0):.4f}, "
              f"Robustness={comb.get('robustness', 0):.4f}, "
              f"Simplicity={comb.get('simplicity', 0):.4f}, "
              f"Metrics={comb.get('num_metrics', 0)}")
        print()

    # Family 2: High Robustness Formulas (sorted by robustness)
    print("\n【Family 2: High Robustness Formulas】(sorted by Robustness, most stable)")
    print("-" * 80)
    top_robustness = sorted(combinations, key=lambda x: x.get("robustness", 0), reverse=True)[:top_n]
    for i, comb in enumerate(top_robustness, 1):
        name = comb.get('name', 'Unknown')[:50]
        print(f"{i}. {name}")
        print(f"   Formula: {comb.get('formula', 'N/A')}")
        print(f"   Accuracy={comb.get('mean_auc', 0):.4f}, "
              f"Robustness={comb.get('robustness', 0):.4f} (std_auc={comb.get('std_auc', 0):.4f}), "
              f"Simplicity={comb.get('simplicity', 0):.4f}, "
              f"Metrics={comb.get('num_metrics', 0)}")
        print()

    # Family 3: High Simplicity Formulas (sorted by simplicity)
    print("\n【Family 3: High Simplicity Formulas】(sorted by Simplicity, fewest metrics)")
    print("-" * 80)
    top_simplicity = sorted(combinations, key=lambda x: x.get("simplicity", 0), reverse=True)[:top_n]
    for i, comb in enumerate(top_simplicity, 1):
        name = comb.get('name', 'Unknown')[:50]
        print(f"{i}. {name}")
        print(f"   Formula: {comb.get('formula', 'N/A')}")
        print(f"   Accuracy={comb.get('mean_auc', 0):.4f}, "
              f"Robustness={comb.get('robustness', 0):.4f}, "
              f"Simplicity={comb.get('simplicity', 0):.4f}, "
              f"Metrics={comb.get('num_metrics', 0)}")
        print()

    # Comprehensive Analysis
    print("\n【Comprehensive Analysis】")
    print("-" * 80)

    # Find formulas that rank highly in multiple families
    accuracy_set = set(c.get('formula') for c in top_accuracy[:5])
    robustness_set = set(c.get('formula') for c in top_robustness[:5])
    simplicity_set = set(c.get('formula') for c in top_simplicity[:5])

    # Formulas in top 5 of two or more families
    balanced_formulas = (
        (accuracy_set & robustness_set) |
        (accuracy_set & simplicity_set) |
        (robustness_set & simplicity_set)
    )

    if balanced_formulas:
        print("Formulas excelling in multiple dimensions:")
        for formula in balanced_formulas:
            comb = next((c for c in combinations if c.get('formula') == formula), None)
            if comb:
                # Count how many top-5 lists this formula appears in
                in_accuracy = formula in accuracy_set
                in_robustness = formula in robustness_set
                in_simplicity = formula in simplicity_set

                families = []
                if in_accuracy:
                    families.append("Accuracy")
                if in_robustness:
                    families.append("Robustness")
                if in_simplicity:
                    families.append("Simplicity")

                print(f"  - {comb.get('name', 'Unknown')[:40]}")
                print(f"    Formula: {formula}")
                print(f"    Top-5 in: {', '.join(families)}")
                print(f"    Scores: Acc={comb.get('mean_auc', 0):.4f}, "
                      f"Rob={comb.get('robustness', 0):.4f}, "
                      f"Simp={comb.get('simplicity', 0):.4f}")
    else:
        print("Different formulas excel in different dimensions.")
        print("Selection depends on application requirements:")
        print("  - Research: Prefer high accuracy")
        print("  - Production: Prefer high robustness")
        print("  - Interpretation: Prefer high simplicity")

    print("=" * 80)


def get_formula_family_stats(combinations: List[Dict]) -> Dict[str, Any]:
    """
    Get statistical summary of formula families

    Args:
        combinations: List of combination dictionaries

    Returns:
        Dict: Statistics for each family
    """
    if not combinations:
        return {}

    # Ensure additional metrics are calculated
    for comb in combinations:
        calculate_additional_metrics(comb)

    # Get top 5 for each family
    top_accuracy = sorted(combinations, key=lambda x: x.get("mean_auc", 0), reverse=True)[:5]
    top_robustness = sorted(combinations, key=lambda x: x.get("robustness", 0), reverse=True)[:5]
    top_simplicity = sorted(combinations, key=lambda x: x.get("simplicity", 0), reverse=True)[:5]

    return {
        "accuracy_family": {
            "top_formulas": [c.get("formula") for c in top_accuracy],
            "mean_auc_range": (
                min(c.get("mean_auc", 0) for c in top_accuracy),
                max(c.get("mean_auc", 0) for c in top_accuracy)
            )
        },
        "robustness_family": {
            "top_formulas": [c.get("formula") for c in top_robustness],
            "robustness_range": (
                min(c.get("robustness", 0) for c in top_robustness),
                max(c.get("robustness", 0) for c in top_robustness)
            )
        },
        "simplicity_family": {
            "top_formulas": [c.get("formula") for c in top_simplicity],
            "simplicity_range": (
                min(c.get("simplicity", 0) for c in top_simplicity),
                max(c.get("simplicity", 0) for c in top_simplicity)
            )
        },
        "total_combinations": len(combinations)
    }


# Module test
if __name__ == "__main__":
    # Test with sample data
    test_combinations = [
        {
            "name": "Formula A",
            "formula": "metric1 * metric2",
            "mean_auc": 0.75,
            "std_auc": 0.05
        },
        {
            "name": "Formula B",
            "formula": "metric1 + metric2 + metric3",
            "mean_auc": 0.72,
            "std_auc": 0.02
        },
        {
            "name": "Formula C",
            "formula": "metric1 / metric2",
            "mean_auc": 0.70,
            "std_auc": 0.10
        },
        {
            "name": "Formula D",
            "formula": "metric1",
            "mean_auc": 0.68,
            "std_auc": 0.03
        }
    ]

    print("Testing formula family analysis...")
    display_formula_families(test_combinations, top_n=3)

    print("\nFormula family stats:")
    stats = get_formula_family_stats(test_combinations)
    print(f"Total combinations: {stats.get('total_combinations', 0)}")
