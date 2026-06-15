#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Pipeline Orchestrator

This module orchestrates the complete modular pipeline:
  Step 1: Data Loading
  Step 2: Node Matching
  Step 3: Seed Selection
  Step 4: Subgraph Building
  Step 5: Knowledge Pump Context
  Step 6: Expert System
  Step 7: Evaluation & Evolution

Features:
  - Run complete pipeline or individual steps
  - Resume from any step (using previous outputs)
  - Checkpoint saving for fault tolerance
  - Human-readable final report

Usage:
    # Run complete pipeline
    python main_pipeline.py

    # Start from specific step
    python main_pipeline.py --start-step 4

    # Run only a specific step
    python main_pipeline.py --only-step 6

    # Resume with existing outputs
    python main_pipeline.py --resume outputs/checkpoint_*.json

Author: Pipeline Modularization Project
Date: 2024-12
"""

import os
import sys
import json
import argparse
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any
from glob import glob

# Import config
from config import CONFIG, FIXED_METRICS, get_timestamp, get_output_path

# Import step modules
from step1_data_loader import DataLoader
from step2_node_matcher import NodeMatcher
from step3_seed_selector import SeedSelector
from step4_subgraph_builder import SubgraphBuilder
from step5_knowledge_pump import KnowledgePump
from step6_expert_system import ExpertSystem, APIPool
from step7_eval_and_evolve import FormulaEvaluator
from step3_5_keyword_expander import KeywordExpander

# Import utility modules
from utils_logging import PipelineLogger, get_logger
from utils_analysis import display_formula_families, get_formula_family_stats


# ============================================================
# Pipeline State Management
# ============================================================

class PipelineState:
    """Manage pipeline state and checkpoints"""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.timestamp = get_timestamp()

        # Step outputs
        self.step1_output: Optional[str] = None
        self.step2_output: Optional[str] = None
        self.step3_output: Optional[str] = None
        self.step4_output: Optional[str] = None
        self.step4_gml: Optional[str] = None
        self.step5_output: Optional[str] = None
        self.step5_context: Optional[str] = None
        self.step6_output: Optional[str] = None
        self.step7_output: Optional[str] = None

        # Step 3.5 outputs
        self.step3_5_output: Optional[str] = None
        self.expanded_keywords: Dict = {}
        self.keyword_node_ids: set = set()

        # Loaded data (for passing between steps)
        self.df = None
        self.graph = None
        self.embeddings = None
        self.matched_nodes: Dict = {}
        self.seed_metrics: List[str] = []
        self.combinations: List[Dict] = []
        self.final_results: List[Dict] = []

        # Status
        self.completed_steps: List[int] = []
        self.current_step: int = 0
        self.error: Optional[str] = None

    def save_checkpoint(self, step: int) -> str:
        """Save checkpoint after completing a step"""
        checkpoint_path = os.path.join(
            self.output_dir,
            f"checkpoint_step{step}_{self.timestamp}.json"
        )

        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "completed_steps": self.completed_steps,
            "current_step": self.current_step,
            "outputs": {
                "step1": self.step1_output,
                "step2": self.step2_output,
                "step3": self.step3_output,
                "step3_5": self.step3_5_output,
                "step4": self.step4_output,
                "step4_gml": self.step4_gml,
                "step5": self.step5_output,
                "step5_context": self.step5_context,
                "step6": self.step6_output,
                "step7": self.step7_output
            },
            "seed_metrics": self.seed_metrics,
            "expanded_keywords": self.expanded_keywords,
            "keyword_node_ids": list(self.keyword_node_ids),
            "config_used": {
                "csv_path": CONFIG.get("CSV_DATA_PATH"),
                "gml_path": CONFIG.get("GML_FILE_PATH"),
                "max_hop": CONFIG.get("MAX_HOP"),
                "seed_top_k": CONFIG.get("SEED_TOP_K")
            }
        }

        os.makedirs(self.output_dir, exist_ok=True)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

        print(f"[Pipeline] Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load state from checkpoint"""
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.completed_steps = data.get("completed_steps", [])
            self.current_step = data.get("current_step", 0)

            outputs = data.get("outputs", {})
            self.step1_output = outputs.get("step1")
            self.step2_output = outputs.get("step2")
            self.step3_output = outputs.get("step3")
            self.step3_5_output = outputs.get("step3_5")
            self.step4_output = outputs.get("step4")
            self.step4_gml = outputs.get("step4_gml")
            self.step5_output = outputs.get("step5")
            self.step5_context = outputs.get("step5_context")
            self.step6_output = outputs.get("step6")
            self.step7_output = outputs.get("step7")

            self.seed_metrics = data.get("seed_metrics", [])
            self.expanded_keywords = data.get("expanded_keywords", {})
            self.keyword_node_ids = set(data.get("keyword_node_ids", []))

            print(f"[Pipeline] Checkpoint loaded: {checkpoint_path}")
            print(f"[Pipeline] Completed steps: {self.completed_steps}")

            return True

        except Exception as e:
            print(f"[Pipeline] Error loading checkpoint: {e}")
            return False

    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint file"""
        pattern = os.path.join(self.output_dir, "checkpoint_step*.json")
        checkpoints = glob(pattern)
        if checkpoints:
            return max(checkpoints, key=os.path.getctime)
        return None


# ============================================================
# Pipeline Steps
# ============================================================

def run_step1(state: PipelineState, config: Dict) -> bool:
    """Step 1: Data Loading"""
    print("\n" + "=" * 70)
    print("STEP 1: DATA LOADING")
    print("=" * 70)

    try:
        loader = DataLoader(config)
        success = loader.load_all()

        if success:
            state.df = loader.df
            state.graph = loader.graph
            state.embeddings = loader.embeddings

            # Save output
            output_path = os.path.join(
                state.output_dir,
                f"step1_data_loaded_{state.timestamp}.json"
            )
            state.step1_output = loader.save_status(output_path)
            state.completed_steps.append(1)

            print(f"[Step 1] Success - Output: {state.step1_output}")
            return True
        else:
            state.error = "Step 1 failed: Data loading incomplete"
            return False

    except Exception as e:
        state.error = f"Step 1 error: {str(e)}"
        traceback.print_exc()
        return False


def run_step2(state: PipelineState, config: Dict) -> bool:
    """Step 2: Node Matching"""
    print("\n" + "=" * 70)
    print("STEP 2: NODE MATCHING")
    print("=" * 70)

    try:
        if state.graph is None:
            print("[Step 2] Error: Graph not loaded")
            return False

        matcher = NodeMatcher(state.graph, state.embeddings, config)

        # Load NPY format embeddings for SapBERT matching
        matcher.load_npy_embeddings()

        matcher.run_two_stage_matching()
        matcher.match_metrics(use_sapbert=True)  # Enable SapBERT matching

        state.matched_nodes = matcher.metrics_to_node_ids

        # Save output
        output_path = os.path.join(
            state.output_dir,
            f"step2_matched_nodes_{state.timestamp}.json"
        )
        state.step2_output = matcher.save_results(output_path)
        state.completed_steps.append(2)

        print(f"[Step 2] Success - Output: {state.step2_output}")
        return True

    except Exception as e:
        state.error = f"Step 2 error: {str(e)}"
        traceback.print_exc()
        return False


def run_step3(state: PipelineState, config: Dict) -> bool:
    """Step 3: Seed Selection"""
    print("\n" + "=" * 70)
    print("STEP 3: SEED SELECTION")
    print("=" * 70)

    try:
        if state.df is None:
            print("[Step 3] Error: Data not loaded")
            return False

        # Get available metrics
        exclude_cols = ["combination_key", "response", "response_binary"]
        metrics = [col for col in state.df.columns if col not in exclude_cols]

        selector = SeedSelector(state.df, metrics, state.matched_nodes, config)
        selector.select_by_auc()

        state.seed_metrics = selector.selected_seeds

        # Save output
        output_path = os.path.join(
            state.output_dir,
            f"step3_seed_metrics_{state.timestamp}.json"
        )
        state.step3_output = selector.save_results(output_path)
        state.completed_steps.append(3)

        print(f"[Step 3] Success - Selected {len(state.seed_metrics)} seeds")
        print(f"[Step 3] Output: {state.step3_output}")
        return True

    except Exception as e:
        state.error = f"Step 3 error: {str(e)}"
        traceback.print_exc()
        return False


def run_step3_5(state: PipelineState, config: Dict) -> bool:
    """Step 3.5: Keyword Expansion via API"""
    print("\n" + "=" * 70)
    print("STEP 3.5: KEYWORD EXPANSION")
    print("=" * 70)

    try:
        # Get problem description
        problem = config.get("DEFAULT_PROBLEM", "")
        if not problem:
            print("[Step 3.5] Warning: No problem description, skipping")
            state.completed_steps.append(3.5)
            return True

        # Create expander
        expander = KeywordExpander(config=config)

        # Expand keywords from problem
        keywords = expander.expand_from_problem(problem)

        if keywords:
            state.expanded_keywords = keywords

            # Match keywords to KG nodes
            threshold = config.get("SIMILARITY_THRESHOLD", 0.7)
            state.keyword_node_ids = expander.match_to_nodes(threshold=threshold)

            # Save output
            output_path = os.path.join(
                state.output_dir,
                f"step3_5_keywords_{state.timestamp}.json"
            )
            state.step3_5_output = expander.save_results(output_path)

            print(f"[Step 3.5] Success - Keywords: {keywords}")
            print(f"[Step 3.5] Matched nodes: {len(state.keyword_node_ids)}")
        else:
            print("[Step 3.5] Warning: No keywords extracted")

        state.completed_steps.append(3.5)
        return True

    except Exception as e:
        state.error = f"Step 3.5 error: {str(e)}"
        traceback.print_exc()
        return False


def run_step4(state: PipelineState, config: Dict) -> bool:
    """Step 4: Subgraph Building"""
    print("\n" + "=" * 70)
    print("STEP 4: SUBGRAPH BUILDING")
    print("=" * 70)

    try:
        if state.graph is None:
            print("[Step 4] Error: Graph not loaded")
            return False

        builder = SubgraphBuilder(state.graph, config)

        # Get star_mode from config (default True for star subgraph)
        star_mode = config.get("USE_STAR_SUBGRAPH", True)

        builder.build(
            seed_metrics=state.seed_metrics,
            matched_nodes=state.matched_nodes,
            max_hop=config.get("MAX_HOP", 2),
            keyword_nodes=state.keyword_node_ids if state.keyword_node_ids else None,
            star_mode=star_mode
        )

        # Save outputs
        gml_path = os.path.join(
            state.output_dir,
            f"step4_subgraph_{state.timestamp}.gml"
        )
        state.step4_gml = builder.save_subgraph(gml_path)

        info_path = os.path.join(
            state.output_dir,
            f"step4_subgraph_info_{state.timestamp}.json"
        )
        state.step4_output = builder.save_info(info_path)
        state.completed_steps.append(4)

        print(f"[Step 4] Success - Output: {state.step4_output}")
        return True

    except Exception as e:
        state.error = f"Step 4 error: {str(e)}"
        traceback.print_exc()
        return False


def run_step5(state: PipelineState, config: Dict) -> bool:
    """Step 5: Knowledge Pump Context"""
    print("\n" + "=" * 70)
    print("STEP 5: KNOWLEDGE PUMP CONTEXT")
    print("=" * 70)

    try:
        # Determine GML path - prefer Step4 output, fallback to config
        if state.step4_gml and os.path.exists(state.step4_gml):
            gml_path = state.step4_gml
            print(f"[Step 5] Using Step4 GML output: {gml_path}")
        elif config.get("USE_STAR_SUBGRAPH", True):
            gml_path = config["STAR_SUBGRAPH_PATH"]
            print(f"[Step 5] Using config STAR_SUBGRAPH_PATH: {gml_path}")
        else:
            gml_path = config["GML_FILE_PATH"]
            print(f"[Step 5] Using config GML_FILE_PATH: {gml_path}")

        kp = KnowledgePump(gml_path, state.matched_nodes, config)
        context = kp.generate_context(state.seed_metrics)

        # Save outputs
        context_path = os.path.join(
            state.output_dir,
            f"step5_kp_context_{state.timestamp}.txt"
        )
        state.step5_context = kp.save_context(context_path)

        stats_path = os.path.join(
            state.output_dir,
            f"step5_kp_stats_{state.timestamp}.json"
        )
        state.step5_output = kp.save_stats(stats_path)
        state.completed_steps.append(5)

        print(f"[Step 5] Success - Context: {state.step5_context}")
        return True

    except Exception as e:
        state.error = f"Step 5 error: {str(e)}"
        traceback.print_exc()
        return False


def run_step6(state: PipelineState, config: Dict) -> bool:
    """Step 6: Expert System"""
    print("\n" + "=" * 70)
    print("STEP 6: EXPERT SYSTEM")
    print("=" * 70)

    try:
        # Load edge descriptions from step4
        edge_descriptions = []
        if state.step4_output and os.path.exists(state.step4_output):
            with open(state.step4_output, 'r', encoding='utf-8') as f:
                step4_data = json.load(f)
            edge_descriptions = step4_data.get("edge_descriptions", [])

        # Initialize API pool
        keys_file = config.get("KEYS_FILE_PATH")
        api_pool = APIPool(keys_file)

        # Use seed metrics as available metrics
        available_metrics = state.seed_metrics if state.seed_metrics else FIXED_METRICS

        # Create expert system
        expert = ExpertSystem(api_pool, available_metrics, config)

        # Get problem description
        problem = config.get("DEFAULT_PROBLEM", "")
        if not problem:
            problem = "What multi-term arithmetic combinations of network metrics best predict microbial community antifragility?"

        # Generate combinations
        state.combinations = expert.generate_combinations(
            problem_description=problem,
            edge_descriptions=edge_descriptions,
            save_prompts=True,
            output_dir=state.output_dir
        )

        # Save output
        output_path = os.path.join(
            state.output_dir,
            f"step6_combinations_{state.timestamp}.json"
        )
        state.step6_output = expert.save_results(output_path)
        state.completed_steps.append(6)

        print(f"[Step 6] Success - {len(state.combinations)} combinations generated")
        print(f"[Step 6] Output: {state.step6_output}")
        return True

    except Exception as e:
        state.error = f"Step 6 error: {str(e)}"
        traceback.print_exc()
        return False


def run_step7(state: PipelineState, config: Dict) -> bool:
    """Step 7: Evaluation & Evolution"""
    print("\n" + "=" * 70)
    print("STEP 7: EVALUATION & EVOLUTION")
    print("=" * 70)

    try:
        if state.df is None:
            print("[Step 7] Error: Data not loaded")
            return False

        if not state.combinations:
            # Try to load from step6 output
            if state.step6_output and os.path.exists(state.step6_output):
                with open(state.step6_output, 'r', encoding='utf-8') as f:
                    step6_data = json.load(f)
                state.combinations = step6_data.get("all_combinations", [])

        if not state.combinations:
            print("[Step 7] Error: No combinations to evaluate")
            return False

        # Create evaluator
        evaluator = FormulaEvaluator(state.df, config)

        # Phase 1: Evaluation
        evaluator.evaluate_combinations(state.combinations)

        # Phase 2: Evolution
        evaluator.evolve_combinations(
            max_iterations=config.get("MAX_ITERATIONS", 3),
            population_size=config.get("POPULATION_SIZE", 10)
        )

        state.final_results = evaluator.get_final_results()

        # Save outputs
        evaluator.save_initial_eval(
            os.path.join(state.output_dir, f"step7_initial_eval_{state.timestamp}.json")
        )
        evaluator.save_evolved(
            os.path.join(state.output_dir, f"step7_evolved_{state.timestamp}.json")
        )

        output_path = os.path.join(
            state.output_dir,
            f"step7_final_results_{state.timestamp}.json"
        )
        state.step7_output = evaluator.save_final_results(output_path)

        evaluator.save_details_csv(
            os.path.join(state.output_dir, f"step7_details_{state.timestamp}.csv")
        )

        state.completed_steps.append(7)

        print(f"[Step 7] Success - Output: {state.step7_output}")
        return True

    except Exception as e:
        state.error = f"Step 7 error: {str(e)}"
        traceback.print_exc()
        return False


# ============================================================
# Final Report Generation
# ============================================================

def generate_final_report(state: PipelineState) -> str:
    """Generate human-readable final report"""
    report_path = os.path.join(
        state.output_dir,
        f"final_report_{state.timestamp}.txt"
    )

    # Get dropout config from CONFIG
    dropout_rate = CONFIG.get("SAMPLE_DROPOUT_RATE", 0.0)
    dropout_trials = CONFIG.get("SAMPLE_DROPOUT_TRIALS", 3)

    lines = [
        "=" * 70,
        "MODULAR PIPELINE FINAL REPORT",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "PIPELINE SUMMARY",
        "-" * 40,
        f"Completed Steps: {state.completed_steps}",
        f"Seed Metrics: {len(state.seed_metrics)}",
        f"Combinations Generated: {len(state.combinations)}",
        f"Final Results: {len(state.final_results)}",
        "",
        "EVALUATION METHODOLOGY",
        "-" * 40,
    ]

    # Add dropout evaluation description
    if dropout_rate > 0:
        lines.extend([
            "Sample Dropout Strategy for Batch Effect Mitigation:",
            f"  - Dropout Rate: {dropout_rate*100:.0f}% of samples randomly removed",
            f"  - Dropout Trials: {dropout_trials} repetitions per dataset",
            "  - Reported AUC/PRAUC: Average across all dropout trials",
            "",
            "This approach mitigates batch effects and provides more",
            "generalizable performance estimates across heterogeneous datasets.",
            "(Politis et al., 1999; Efron & Tibshirani, 1994)",
            "",
        ])
    else:
        lines.extend([
            "Evaluation Mode: Standard (no sample dropout)",
            "  - Full dataset used for each evaluation",
            "",
        ])

    # Add seed metrics
    if state.seed_metrics:
        lines.append("SEED METRICS (Top 10)")
        lines.append("-" * 40)
        for i, metric in enumerate(state.seed_metrics[:10], 1):
            lines.append(f"  {i:2d}. {metric}")
        lines.append("")

    # Add final results
    if state.final_results:
        lines.append("TOP FORMULA COMBINATIONS")
        lines.append("-" * 40)
        for i, result in enumerate(state.final_results[:5], 1):
            lines.append(f"\n{i}. {result.get('name', 'Unknown')}")
            lines.append(f"   Formula: {result.get('formula', '')}")
            lines.append(f"   Agent: {result.get('agent_name', 'Unknown')}")
            lines.append(f"   AUC: {result.get('mean_auc', 0):.4f} ± {result.get('std_auc', 0):.4f}")
            lines.append(f"   PRAUC: {result.get('mean_prauc', 0):.4f}")
            lines.append(f"   Valid Datasets: {result.get('valid_datasets', 0)}")
        lines.append("")

    # Add formula family analysis
    if state.final_results:
        lines.append("FORMULA FAMILY ANALYSIS")
        lines.append("-" * 40)
        family_stats = get_formula_family_stats(state.final_results)
        if family_stats:
            lines.append(f"Total Combinations: {family_stats.get('total_combinations', 0)}")
            lines.append("")
            lines.append("Top Accuracy Family:")
            for f in family_stats.get("accuracy_family", {}).get("top_formulas", [])[:3]:
                lines.append(f"  - {f[:60]}")
            lines.append("")
            lines.append("Top Robustness Family:")
            for f in family_stats.get("robustness_family", {}).get("top_formulas", [])[:3]:
                lines.append(f"  - {f[:60]}")
            lines.append("")
            lines.append("Top Simplicity Family:")
            for f in family_stats.get("simplicity_family", {}).get("top_formulas", [])[:3]:
                lines.append(f"  - {f[:60]}")
        lines.append("")

    # Add output files
    lines.append("OUTPUT FILES")
    lines.append("-" * 40)
    if state.step1_output:
        lines.append(f"  Step 1: {state.step1_output}")
    if state.step2_output:
        lines.append(f"  Step 2: {state.step2_output}")
    if state.step3_output:
        lines.append(f"  Step 3: {state.step3_output}")
    if state.step4_output:
        lines.append(f"  Step 4: {state.step4_output}")
    if state.step5_output:
        lines.append(f"  Step 5: {state.step5_output}")
    if state.step6_output:
        lines.append(f"  Step 6: {state.step6_output}")
    if state.step7_output:
        lines.append(f"  Step 7: {state.step7_output}")
    lines.append("")

    # Add error if any
    if state.error:
        lines.append("ERRORS")
        lines.append("-" * 40)
        lines.append(f"  {state.error}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n[Pipeline] Final report saved: {report_path}")
    return report_path


# ============================================================
# Main Pipeline Runner
# ============================================================

def run_pipeline(
    start_step: int = 1,
    end_step: int = 7,
    only_step: int = None,
    output_dir: str = "outputs",
    resume_checkpoint: str = None,
    config: Dict = None,
    enable_logging: bool = True
) -> PipelineState:
    """
    Run the complete pipeline or specific steps

    Args:
        start_step: Step to start from (1-7)
        end_step: Step to end at (1-7)
        only_step: If set, run only this specific step
        output_dir: Output directory
        resume_checkpoint: Path to checkpoint file to resume from
        config: Configuration dictionary
        enable_logging: Enable dual output logging

    Returns:
        PipelineState: Final pipeline state
    """
    if config is None:
        config = CONFIG

    # Initialize state
    state = PipelineState(output_dir)

    # Initialize logging
    logger = None
    if enable_logging:
        logger = PipelineLogger(output_dir=output_dir)
        logger.start()

    # Load checkpoint if resuming
    if resume_checkpoint:
        if os.path.exists(resume_checkpoint):
            state.load_checkpoint(resume_checkpoint)
        else:
            print(f"[Pipeline] Checkpoint not found: {resume_checkpoint}")

    # Determine steps to run
    if only_step:
        steps_to_run = [only_step]
    else:
        # Build step list including 3.5 between 3 and 4
        steps_to_run = []
        for s in range(start_step, end_step + 1):
            steps_to_run.append(s)
            if s == 3 and start_step <= 3 and end_step >= 4:
                steps_to_run.append(3.5)

    print("\n" + "=" * 70)
    print("MODULAR KNOWLEDGE GRAPH ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Steps to run: {steps_to_run}")
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {state.timestamp}")
    print(f"Logging: {'Enabled' if enable_logging else 'Disabled'}")

    # Step runners
    step_runners = {
        1: run_step1,
        2: run_step2,
        3: run_step3,
        3.5: run_step3_5,
        4: run_step4,
        5: run_step5,
        6: run_step6,
        7: run_step7
    }

    # Run steps
    for step in steps_to_run:
        if step in state.completed_steps:
            print(f"\n[Pipeline] Step {step} already completed, skipping...")
            continue

        state.current_step = step
        runner = step_runners.get(step)

        if runner:
            success = runner(state, config)

            if success:
                state.save_checkpoint(step)
            else:
                print(f"\n[Pipeline] Step {step} failed: {state.error}")
                break
        else:
            print(f"\n[Pipeline] Unknown step: {step}")

    # Display formula family analysis if we have results
    if state.final_results and len(state.final_results) >= 3:
        print("\n")
        display_formula_families(state.final_results, top_n=5)

    # Generate final report
    if state.completed_steps:
        generate_final_report(state)

    # Stop logging and save logs
    if logger:
        logger.stop()
        logger.save(filename=f"complete_run_log_{state.timestamp}.json")

    return state


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Modular Knowledge Graph Analysis Pipeline'
    )
    parser.add_argument(
        '--start-step', type=int, default=1,
        help='Step to start from (1-7)'
    )
    parser.add_argument(
        '--end-step', type=int, default=7,
        help='Step to end at (1-7)'
    )
    parser.add_argument(
        '--only-step', type=int,
        help='Run only this specific step'
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--resume', type=str,
        help='Path to checkpoint file to resume from'
    )
    parser.add_argument(
        '--list-checkpoints', action='store_true',
        help='List available checkpoints'
    )
    parser.add_argument(
        '--disable-logging', action='store_true',
        help='Disable dual output logging'
    )

    args = parser.parse_args()

    # List checkpoints
    if args.list_checkpoints:
        pattern = os.path.join(args.output_dir, "checkpoint_*.json")
        checkpoints = glob(pattern)
        if checkpoints:
            print("Available checkpoints:")
            for cp in sorted(checkpoints):
                print(f"  - {cp}")
        else:
            print("No checkpoints found")
        return 0

    # Run pipeline
    state = run_pipeline(
        start_step=args.start_step,
        end_step=args.end_step,
        only_step=args.only_step,
        output_dir=args.output_dir,
        resume_checkpoint=args.resume,
        enable_logging=not args.disable_logging
    )

    # Print summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Completed steps: {state.completed_steps}")

    if state.final_results:
        print(f"\nBest Formula:")
        best = state.final_results[0]
        print(f"  Name: {best.get('name', 'Unknown')}")
        print(f"  Formula: {best.get('formula', '')}")
        print(f"  AUC: {best.get('mean_auc', 0):.4f}")

    if state.error:
        print(f"\nError: {state.error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
