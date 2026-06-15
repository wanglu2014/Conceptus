#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 7: Evaluation and Evolution Module (Merged)

This module combines formula evaluation and genetic evolution:
  - Phase 1: Evaluate combinations across all datasets (AUC/PRAUC)
  - Phase 2: Genetic evolution optimization (crossover, mutation)

Key Features:
  - Cross-dataset evaluation for robustness
  - Sample dropout for stability testing
  - Variable count limits (MAX_FORMULA_VARIABLES)
  - Crossover: (formula1 + formula2) / 2
  - Mutation: structural changes (add→mul, mul→div, etc.)

Input Files:
    - outputs/step6_combinations_{timestamp}.json
    - filtered_phyloseq_network_data.csv

Output Files:
    - outputs/step7_initial_eval_{timestamp}.json
    - outputs/step7_evolved_{timestamp}.json
    - outputs/step7_final_results_{timestamp}.json
    - outputs/step7_details_{timestamp}.csv

Usage:
    python step7_eval_and_evolve.py --combinations outputs/step6_combinations_*.json

Author: Pipeline Modularization Project
Date: 2024-12
"""

import os
import sys
import json
import argparse
import traceback
import re
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from sklearn.metrics import roc_auc_score, average_precision_score

# Import config
from config import CONFIG, FIXED_METRICS, get_timestamp, get_output_path


# ============================================================
# Formula Calculation
# ============================================================

def calculate_formula(formula: str, data: pd.DataFrame, verbose: bool = False) -> Optional[pd.Series]:
    """
    Calculate combination formula values

    Args:
        formula: Formula string
        data: DataFrame containing metrics
        verbose: Print calculation info

    Returns:
        pd.Series: Calculated values or None if failed
    """
    try:
        # Get all variable names
        var_names = re.findall(r'([a-zA-Z_]+[a-zA-Z0-9_]*)', formula)

        # Exclude math functions
        math_functions = ['max', 'min', 'sqrt', 'log', 'abs', 'sin', 'cos', 'tan', 'exp', 'log10', 'np']
        var_names = [var for var in var_names if var not in math_functions]

        # Check all variables exist
        for var in var_names:
            if var not in data.columns:
                if verbose:
                    print(f"[Eval] Warning: Variable '{var}' not in dataset")
                return None

        # Create local namespace with variables and allowed functions
        local_vars = {
            'np': np,
            'min': min,
            'max': max,
            'abs': abs,
            'sum': sum,
            'len': len,
            'log': np.log,
            'log10': np.log10,
            'sqrt': np.sqrt,
            'exp': np.exp,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan
        }

        # Add data columns
        for var in var_names:
            local_vars[var] = data[var].values

        # Replace power operator
        safe_formula = formula.replace("^", "**")

        # Execute calculation
        if verbose:
            print(f"[Eval] Executing: {safe_formula}")
        result = eval(safe_formula, {"__builtins__": {}}, local_vars)

        # Convert to Series
        if not isinstance(result, pd.Series):
            result = pd.Series(result)

        # Handle inf and NaN
        result = result.replace([np.inf, -np.inf], np.nan)

        # Fill NaN with mean
        if result.isna().any():
            if verbose:
                print(f"[Eval] Warning: Formula '{formula}' produced {result.isna().sum()} NaN values")
            result = result.fillna(result.mean())

        return result

    except Exception as e:
        if verbose:
            print(f"[Eval] Error calculating '{formula}': {e}")
            traceback.print_exc()
        return None


def count_formula_variables(formula: str) -> int:
    """Count unique network metric variables in formula"""
    math_funcs = {'log', 'sqrt', 'abs', 'max', 'min', 'exp', 'sin', 'cos', 'tan', 'np', 'log10', 'pow'}
    variables = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)', formula)
    unique_vars = set(v for v in variables if v not in math_funcs)
    return len(unique_vars)


# ============================================================
# Evaluation Functions
# ============================================================

def evaluate_combination_across_all_datasets(
    formula: str,
    df: pd.DataFrame,
    groups: List[str],
    dropout_rate: float = None,
    num_trials: int = None
) -> Dict:
    """
    Evaluate metric combination performance across all datasets

    Supports sample dropout for robustness testing.

    Args:
        formula: Combination formula string
        df: Full DataFrame
        groups: List of dataset group names
        dropout_rate: Sample dropout rate
        num_trials: Number of dropout trials

    Returns:
        dict: Evaluation results with per-dataset scores and averages
    """
    if dropout_rate is None:
        dropout_rate = CONFIG.get("SAMPLE_DROPOUT_RATE", 0.0)
    if num_trials is None:
        num_trials = CONFIG.get("SAMPLE_DROPOUT_TRIALS", 3)

    dropout_rate = max(0.0, min(0.5, dropout_rate))

    try:
        results = {
            "formula": formula,
            "dataset_scores": {},
            "mean_auc": 0,
            "mean_prauc": 0,
            "std_auc": 0,
            "std_prauc": 0,
            "valid_datasets": 0,
            "dropout_rate": dropout_rate
        }

        aucs = []
        praucs = []

        for group in groups:
            group_data = df[df["combination_key"] == group].copy()

            if len(group_data) < 10:
                continue

            # Dropout evaluation
            if dropout_rate > 0 and len(group_data) >= 15:
                trial_aucs = []
                trial_praucs = []

                for trial in range(num_trials):
                    n_keep = int(len(group_data) * (1 - dropout_rate))
                    sampled_data = group_data.sample(n=n_keep, random_state=42 + trial)

                    combo_values = calculate_formula(formula, sampled_data)
                    if combo_values is None:
                        continue

                    y_true = sampled_data["response_binary"].values
                    if len(set(y_true)) < 2:
                        continue

                    try:
                        trial_aucs.append(roc_auc_score(y_true, combo_values))
                        trial_praucs.append(average_precision_score(y_true, combo_values))
                    except:
                        continue

                if trial_aucs:
                    auc_score = np.mean(trial_aucs)
                    prauc_score = np.mean(trial_praucs)
                    aucs.append(auc_score)
                    praucs.append(prauc_score)

                    results["dataset_scores"][group] = {
                        "auc": auc_score,
                        "prauc": prauc_score,
                        "n_samples": len(group_data),
                        "dropout_trials": len(trial_aucs)
                    }
            else:
                # Standard evaluation without dropout
                combo_values = calculate_formula(formula, group_data)
                if combo_values is None:
                    continue

                y_true = group_data["response_binary"].values
                if len(set(y_true)) < 2:
                    continue

                try:
                    auc_score = roc_auc_score(y_true, combo_values)
                    prauc_score = average_precision_score(y_true, combo_values)

                    aucs.append(auc_score)
                    praucs.append(prauc_score)

                    results["dataset_scores"][group] = {
                        "auc": auc_score,
                        "prauc": prauc_score,
                        "n_samples": len(group_data)
                    }
                except Exception as e:
                    continue

        # Calculate statistics
        if aucs:
            results["mean_auc"] = float(np.mean(aucs))
            results["mean_prauc"] = float(np.mean(praucs))
            results["std_auc"] = float(np.std(aucs))
            results["std_prauc"] = float(np.std(praucs))
            results["valid_datasets"] = len(aucs)
            results["score"] = results["mean_auc"]
        else:
            return {"error": "Could not calculate valid scores on any dataset"}

        return results

    except Exception as e:
        return {"error": str(e)}


def evaluate_all_combinations(
    combinations: List[Dict],
    df: pd.DataFrame,
    groups: List[str],
    verbose: bool = True
) -> List[Dict]:
    """
    Evaluate all combinations across datasets

    Args:
        combinations: List of combination dictionaries
        df: Full DataFrame
        groups: Dataset group names
        verbose: Print progress

    Returns:
        List[Dict]: Evaluated combinations with scores
    """
    evaluated = []

    for i, combo in enumerate(combinations, 1):
        formula = combo.get("formula", "")
        name = combo.get("name", f"Combination_{i}")

        if verbose:
            print(f"\n[Eval] {i}/{len(combinations)}: {name}")
            print(f"       Formula: {formula}")

        evaluation = evaluate_combination_across_all_datasets(formula, df, groups)

        if "error" not in evaluation:
            combo_result = {
                **combo,
                "mean_auc": evaluation.get("mean_auc", 0),
                "mean_prauc": evaluation.get("mean_prauc", 0),
                "std_auc": evaluation.get("std_auc", 0),
                "std_prauc": evaluation.get("std_prauc", 0),
                "valid_datasets": evaluation.get("valid_datasets", 0),
                "score": evaluation.get("score", 0),
                "dataset_scores": evaluation.get("dataset_scores", {})
            }
            evaluated.append(combo_result)

            if verbose:
                print(f"       AUC: {combo_result['mean_auc']:.4f} ± {combo_result['std_auc']:.4f}")
                print(f"       Valid datasets: {combo_result['valid_datasets']}")
        else:
            if verbose:
                print(f"       Error: {evaluation.get('error', 'Unknown')}")

    # Sort by score
    evaluated.sort(key=lambda x: x.get("score", 0), reverse=True)

    return evaluated


# ============================================================
# Evolution Functions
# ============================================================

def evolution_phase(
    combinations: List[Dict],
    df: pd.DataFrame,
    groups: List[str],
    population_size: int = 10,
    mutation_rate: float = 0.2,
    verbose: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Genetic evolution phase for optimizing formulas

    Args:
        combinations: Evaluated combinations
        df: Full DataFrame
        groups: Dataset group names
        population_size: Selection size
        mutation_rate: Mutation probability
        verbose: Print progress

    Returns:
        Tuple[List[Dict], Dict]: (evolved combinations, evolution stats)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Evolution Phase (Genetic Optimization)")
        print("=" * 60)

    max_vars = CONFIG.get("MAX_FORMULA_VARIABLES", 3)
    stats = {
        "crossover_generated": 0,
        "crossover_skipped": 0,
        "mutation_generated": 0,
        "mutation_skipped": 0,
        "evaluated": 0,
        "successful": 0
    }

    # Rank by score
    ranked = sorted(combinations, key=lambda x: x.get("score", 0), reverse=True)

    # Selection
    selected = ranked[:population_size]
    if verbose:
        print(f"[Evolve] Selected top {len(selected)} combinations")

    # Crossover
    crossover_combinations = []
    for i in range(len(selected) - 1):
        for j in range(i + 1, len(selected)):
            try:
                formula1 = selected[i]["formula"]
                formula2 = selected[j]["formula"]

                crossover_formula = f"({formula1} + {formula2}) / 2"

                var_count = count_formula_variables(crossover_formula)
                if var_count > max_vars:
                    stats["crossover_skipped"] += 1
                    continue

                name1 = selected[i].get('name', 'F1')[:10]
                name2 = selected[j].get('name', 'F2')[:10]

                crossover_combinations.append({
                    "formula": crossover_formula,
                    "name": f"{name1}-{name2}_Crossover",
                    "description": f"Crossover of '{selected[i].get('name', '')}' and '{selected[j].get('name', '')}'"
                })
                stats["crossover_generated"] += 1

            except Exception as e:
                if verbose:
                    print(f"[Evolve] Crossover error: {e}")

    if verbose:
        print(f"[Evolve] Generated {stats['crossover_generated']} crossover combinations "
              f"(skipped {stats['crossover_skipped']})")

    # Mutation
    mutation_combinations = []
    for i, combo in enumerate(selected[:3]):
        try:
            formula = combo["formula"]
            name = combo.get("name", f"Combo_{i}")

            mutation = None
            mutation_name = None
            mutation_desc = None

            if "+" in formula and "*" not in formula:
                # Add -> Multiply
                parts = formula.split("+")
                if len(parts) >= 2:
                    mutation = f"{parts[0].strip()} * {parts[1].strip()}"
                    mutation_name = f"{name}_MulMutation"
                    mutation_desc = f"Structural mutation: addition to multiplication"

            elif "*" in formula and "+" not in formula:
                # Multiply -> Divide
                parts = formula.split("*")
                if len(parts) >= 2:
                    mutation = f"{parts[0].strip()} / {parts[1].strip()}"
                    mutation_name = f"{name}_DivMutation"
                    mutation_desc = f"Structural mutation: multiplication to division"

            elif "/" in formula:
                # Divide -> Invert
                parts = formula.split("/")
                if len(parts) >= 2:
                    mutation = f"{parts[1].strip()} / {parts[0].strip()}"
                    mutation_name = f"{name}_InvMutation"
                    mutation_desc = f"Structural mutation: fraction inversion"

            else:
                # Add another high-ranking formula
                if i + 1 < len(selected):
                    other_formula = selected[i + 1]["formula"]
                    mutation = f"{formula} + {other_formula}"
                    mutation_name = f"{name}_AddMutation"
                    mutation_desc = f"Structural mutation: adding another formula"

            if mutation is not None:
                var_count = count_formula_variables(mutation)
                if var_count > max_vars:
                    stats["mutation_skipped"] += 1
                    continue

                mutation_combinations.append({
                    "formula": mutation,
                    "name": mutation_name,
                    "description": mutation_desc
                })
                stats["mutation_generated"] += 1

        except Exception as e:
            if verbose:
                print(f"[Evolve] Mutation error: {e}")

    if verbose:
        print(f"[Evolve] Generated {stats['mutation_generated']} mutation combinations "
              f"(skipped {stats['mutation_skipped']})")

    # Combine all new combinations
    new_combinations = crossover_combinations + mutation_combinations

    # Evaluate new combinations
    evaluated_new = []
    for combo in new_combinations:
        formula = combo["formula"]
        name = combo["name"]

        if verbose:
            print(f"\n[Evolve] Evaluating: {name}")

        evaluation = evaluate_combination_across_all_datasets(formula, df, groups)
        stats["evaluated"] += 1

        if "error" not in evaluation:
            combo["score"] = evaluation.get("score", 0)
            combo["mean_auc"] = evaluation.get("mean_auc", 0)
            combo["mean_prauc"] = evaluation.get("mean_prauc", 0)
            combo["std_auc"] = evaluation.get("std_auc", 0)
            combo["std_prauc"] = evaluation.get("std_prauc", 0)
            combo["valid_datasets"] = evaluation.get("valid_datasets", 0)
            combo["dataset_scores"] = evaluation.get("dataset_scores", {})
            evaluated_new.append(combo)
            stats["successful"] += 1

            if verbose:
                print(f"         AUC: {combo['mean_auc']:.4f}")

    if verbose:
        print(f"\n[Evolve] Successfully evaluated {stats['successful']}/{stats['evaluated']} new combinations")

    # Combine original top 3 with new top 3
    best_original = ranked[:3]
    best_new = sorted(evaluated_new, key=lambda x: x.get("score", 0), reverse=True)[:3] if evaluated_new else []

    final = best_original + best_new
    final = sorted(final, key=lambda x: x.get("score", 0), reverse=True)

    # Print best result
    if final and verbose:
        best = final[0]
        print(f"\n[Evolve] Best combination: {best.get('name', 'Unknown')}")
        print(f"         Formula: {best['formula']}")
        print(f"         AUC: {best.get('mean_auc', 0):.4f} ± {best.get('std_auc', 0):.4f}")

    return final, stats


def run_iterative_evolution(
    combinations: List[Dict],
    df: pd.DataFrame,
    groups: List[str],
    max_iterations: int = 3,
    population_size: int = 10,
    improvement_threshold: float = 0.01,
    verbose: bool = True
) -> Tuple[List[Dict], Dict]:
    """
    Run multiple iterations of evolution

    Args:
        combinations: Initial combinations
        df: Full DataFrame
        groups: Dataset group names
        max_iterations: Maximum evolution iterations
        population_size: Selection size
        improvement_threshold: Stop if improvement is below this
        verbose: Print progress

    Returns:
        Tuple[List[Dict], Dict]: (final combinations, evolution history)
    """
    history = {
        "iterations": [],
        "initial_best_auc": 0,
        "final_best_auc": 0,
        "improvement": 0,
        "converged": False,
        "convergence_reason": ""
    }

    current_population = combinations.copy()

    if current_population:
        history["initial_best_auc"] = max(c.get("score", 0) for c in current_population)

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evolution Iteration {iteration + 1}/{max_iterations}")
            print("=" * 60)

        best_before = max(c.get("score", 0) for c in current_population) if current_population else 0

        evolved, stats = evolution_phase(
            current_population, df, groups,
            population_size=population_size,
            verbose=verbose
        )

        best_after = max(c.get("score", 0) for c in evolved) if evolved else 0

        iteration_info = {
            "iteration": iteration + 1,
            "best_auc_before": best_before,
            "best_auc_after": best_after,
            "improvement": best_after - best_before,
            "stats": stats
        }
        history["iterations"].append(iteration_info)

        if verbose:
            print(f"\n[Iter {iteration + 1}] Best AUC: {best_before:.4f} -> {best_after:.4f} "
                  f"(+{best_after - best_before:.4f})")

        # Check convergence
        if best_after - best_before < improvement_threshold:
            history["converged"] = True
            history["convergence_reason"] = "improvement_below_threshold"
            if verbose:
                print(f"[Converged] Improvement below threshold ({improvement_threshold})")
            break

        current_population = evolved

    history["final_best_auc"] = max(c.get("score", 0) for c in current_population) if current_population else 0
    history["improvement"] = history["final_best_auc"] - history["initial_best_auc"]

    return current_population, history


# ============================================================
# Main Evaluator Class
# ============================================================

class FormulaEvaluator:
    """Formula evaluation and evolution manager"""

    def __init__(self, df: pd.DataFrame, config: Dict = None):
        """Initialize evaluator"""
        self.df = df
        self.config = config or CONFIG

        # Get groups
        if "combination_key" in df.columns:
            self.groups = list(df["combination_key"].unique())
        else:
            self.groups = ["all_data"]
            self.df["combination_key"] = "all_data"

        # Results storage
        self.initial_results: List[Dict] = []
        self.evolved_results: List[Dict] = []
        self.final_results: List[Dict] = []
        self.evolution_history: Dict = {}

        # Statistics
        self.stats: Dict[str, Any] = {
            "timestamp": None,
            "status": "not_started",
            "total_combinations": 0,
            "evaluated": 0,
            "successful": 0,
            "iterations": 0,
            "best_formula": "",
            "best_auc": 0
        }

    def evaluate_combinations(self, combinations: List[Dict], verbose: bool = True) -> List[Dict]:
        """Phase 1: Initial evaluation"""
        print("\n" + "=" * 60)
        print("Step 7 Phase 1: Initial Evaluation")
        print("=" * 60)
        print(f"Evaluating {len(combinations)} combinations across {len(self.groups)} datasets")

        self.stats["timestamp"] = datetime.now().isoformat()
        self.stats["total_combinations"] = len(combinations)

        self.initial_results = evaluate_all_combinations(
            combinations, self.df, self.groups, verbose=verbose
        )

        self.stats["evaluated"] = len(self.initial_results)

        if self.initial_results:
            self.stats["best_formula"] = self.initial_results[0]["formula"]
            self.stats["best_auc"] = self.initial_results[0].get("mean_auc", 0)

        print(f"\n[Phase 1] Evaluated {len(self.initial_results)} combinations")
        if self.initial_results:
            print(f"[Phase 1] Best: {self.initial_results[0].get('name', 'Unknown')}")
            print(f"          AUC: {self.initial_results[0].get('mean_auc', 0):.4f}")

        return self.initial_results

    def evolve_combinations(
        self,
        max_iterations: int = None,
        population_size: int = None,
        verbose: bool = True
    ) -> List[Dict]:
        """Phase 2: Evolution optimization"""
        if not self.initial_results:
            print("[Phase 2] Error: No initial results to evolve")
            return []

        if max_iterations is None:
            max_iterations = self.config.get("MAX_ITERATIONS", 3)
        if population_size is None:
            population_size = self.config.get("POPULATION_SIZE", 10)

        print("\n" + "=" * 60)
        print("Step 7 Phase 2: Evolution Optimization")
        print("=" * 60)

        self.evolved_results, self.evolution_history = run_iterative_evolution(
            self.initial_results,
            self.df,
            self.groups,
            max_iterations=max_iterations,
            population_size=population_size,
            verbose=verbose
        )

        self.stats["iterations"] = len(self.evolution_history.get("iterations", []))

        if self.evolved_results:
            self.stats["best_formula"] = self.evolved_results[0]["formula"]
            self.stats["best_auc"] = self.evolved_results[0].get("mean_auc", 0)

        return self.evolved_results

    def get_final_results(self) -> List[Dict]:
        """Get final results (evolved if available, otherwise initial)"""
        if self.evolved_results:
            self.final_results = self.evolved_results
        else:
            self.final_results = self.initial_results

        self.stats["status"] = "success"
        return self.final_results

    def save_initial_eval(self, output_path: str = None) -> str:
        """Save initial evaluation results"""
        if output_path is None:
            output_path = get_output_path("step7_initial_eval")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        output_data = {
            "timestamp": self.stats["timestamp"],
            "total_evaluated": len(self.initial_results),
            "groups": self.groups,
            "results": self.initial_results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[Step7] Initial evaluation saved to: {output_path}")
        return output_path

    def save_evolved(self, output_path: str = None) -> str:
        """Save evolution results"""
        if output_path is None:
            output_path = get_output_path("step7_evolved")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        output_data = {
            "timestamp": self.stats["timestamp"],
            "config": {
                "max_iterations": self.config.get("MAX_ITERATIONS"),
                "population_size": self.config.get("POPULATION_SIZE"),
                "max_formula_variables": self.config.get("MAX_FORMULA_VARIABLES")
            },
            "evolution_history": self.evolution_history,
            "evolved_results": self.evolved_results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[Step7] Evolution results saved to: {output_path}")
        return output_path

    def save_final_results(self, output_path: str = None) -> str:
        """Save final results"""
        if output_path is None:
            output_path = get_output_path("step7_final_results")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        final = self.get_final_results()

        # Get dropout configuration
        dropout_rate = self.config.get("SAMPLE_DROPOUT_RATE", 0.0)
        dropout_trials = self.config.get("SAMPLE_DROPOUT_TRIALS", 3)

        output_data = {
            **self.stats,
            "dropout_config": {
                "enabled": dropout_rate > 0,
                "dropout_rate": dropout_rate,
                "dropout_trials": dropout_trials,
                "description": (
                    f"Sample dropout strategy: randomly removed {dropout_rate*100:.0f}% of samples, "
                    f"repeated {dropout_trials} times per dataset, reported average AUC. "
                    "This approach mitigates batch effects and provides more generalizable estimates."
                ) if dropout_rate > 0 else "Dropout disabled (rate=0)"
            },
            "best_formula": final[0] if final else None,
            "top_10_formulas": final[:10],
            "full_ranking": final,
            "evolution_summary": {
                "initial_best": self.evolution_history.get("initial_best_auc", 0),
                "final_best": self.evolution_history.get("final_best_auc", 0),
                "improvement": self.evolution_history.get("improvement", 0),
                "converged": self.evolution_history.get("converged", False)
            } if self.evolution_history else None
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[Step7] Final results saved to: {output_path}")
        return output_path

    def save_details_csv(self, output_path: str = None) -> str:
        """Save detailed results as CSV"""
        if output_path is None:
            output_path = get_output_path("step7_details", extension="csv")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        final = self.get_final_results()

        # Get dropout configuration
        dropout_rate = self.config.get("SAMPLE_DROPOUT_RATE", 0.0)
        dropout_trials = self.config.get("SAMPLE_DROPOUT_TRIALS", 3)

        csv_data = []
        for i, result in enumerate(final, 1):
            # Get per-result dropout info if available
            dataset_scores = result.get("dataset_scores", {})
            result_dropout_trials = None
            if dataset_scores:
                # Get dropout_trials from first dataset that has it
                for ds_info in dataset_scores.values():
                    if isinstance(ds_info, dict) and "dropout_trials" in ds_info:
                        result_dropout_trials = ds_info.get("dropout_trials")
                        break

            csv_data.append({
                "Rank": i,
                "Name": result.get("name", ""),
                "Formula": result.get("formula", ""),
                "Agent": result.get("agent_name", "Unknown"),
                "Mean_AUC": result.get("mean_auc", 0),
                "Std_AUC": result.get("std_auc", 0),
                "Mean_PRAUC": result.get("mean_prauc", 0),
                "Std_PRAUC": result.get("std_prauc", 0),
                "Valid_Datasets": result.get("valid_datasets", 0),
                "Dropout_Rate": dropout_rate,
                "Dropout_Trials": result_dropout_trials if result_dropout_trials else (dropout_trials if dropout_rate > 0 else 0),
                "Description": result.get("description", "")[:200]
            })

        df_csv = pd.DataFrame(csv_data)
        df_csv.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"[Step7] Details CSV saved to: {output_path}")
        return output_path


# ============================================================
# Main Entry Point
# ============================================================

def load_combinations(json_path: str) -> List[Dict]:
    """Load combinations from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("all_combinations", [])


def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Load CSV data file"""
    df = pd.read_csv(csv_path)

    # Ensure response_binary exists
    if "response_binary" not in df.columns:
        if "response" in df.columns:
            if df["response"].dtype == 'object':
                df["response_binary"] = (df["response"] == "good").astype(int)
            else:
                median = df["response"].median()
                df["response_binary"] = (df["response"] > median).astype(int)

    # Ensure combination_key exists
    if "combination_key" not in df.columns:
        df["combination_key"] = "all_data"

    return df


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Step 7: Evaluation and Evolution')
    parser.add_argument('--combinations', type=str, help='Step 6 combinations JSON file path')
    parser.add_argument('--data-csv', type=str, help='CSV data file path')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--max-iterations', type=int, default=3, help='Max evolution iterations')
    parser.add_argument('--population-size', type=int, default=10, help='Evolution population size')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation, skip evolution')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Step 7: Evaluation and Evolution")
    print("=" * 60)

    # Load CSV data
    csv_path = args.data_csv or CONFIG.get("CSV_DATA_PATH")
    print(f"\n[Load] Loading CSV data: {csv_path}")
    df = load_csv_data(csv_path)
    print(f"[Load] Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Load combinations
    combinations = []
    if args.combinations:
        print(f"\n[Load] Loading combinations: {args.combinations}")
        combinations = load_combinations(args.combinations)
        print(f"[Load] Loaded {len(combinations)} combinations")
    else:
        print("[Warning] No combinations file provided")

    if not combinations:
        print("[Error] No combinations to evaluate")
        return 1

    # Create evaluator
    evaluator = FormulaEvaluator(df)

    # Phase 1: Initial evaluation
    evaluator.evaluate_combinations(combinations, verbose=args.verbose)

    # Phase 2: Evolution (unless --eval-only)
    if not args.eval_only:
        evaluator.evolve_combinations(
            max_iterations=args.max_iterations,
            population_size=args.population_size,
            verbose=args.verbose
        )

    # Save results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    timestamp = get_timestamp()

    evaluator.save_initial_eval(os.path.join(output_dir, f"step7_initial_eval_{timestamp}.json"))

    if not args.eval_only:
        evaluator.save_evolved(os.path.join(output_dir, f"step7_evolved_{timestamp}.json"))

    evaluator.save_final_results(os.path.join(output_dir, f"step7_final_results_{timestamp}.json"))
    evaluator.save_details_csv(os.path.join(output_dir, f"step7_details_{timestamp}.csv"))

    # Print summary
    final = evaluator.get_final_results()
    print("\n" + "=" * 60)
    print("Step 7 Complete!")
    print("=" * 60)

    if final:
        print(f"\nTop 5 Formulas:")
        for i, result in enumerate(final[:5], 1):
            print(f"  {i}. {result.get('name', 'Unknown')}")
            print(f"     Formula: {result['formula']}")
            print(f"     AUC: {result.get('mean_auc', 0):.4f} ± {result.get('std_auc', 0):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
