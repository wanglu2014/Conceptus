#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6: Expert System Module

Three-agent expert system for generating network metric combination formulas:
  - MathAgent: Mathematical perspective on network metrics
  - BioAgent: Biological perspective on microbial communities
  - IntegrationAgent: Synthesize both perspectives

Key Design Principles (Prompt-Driven):
  - AUC is only used for seed selection, NOT for guiding formula design
  - Agents are guided by theoretical framework: Reach × Span × Scale
  - Orthogonal dimension products capture emergent system properties

Input Files:
    - outputs/step5_kp_context_{timestamp}.txt
    - outputs/step4_subgraph_info_{timestamp}.json
    - outputs/step3_seed_metrics_{timestamp}.json

Output Files:
    - outputs/step6_combinations_{timestamp}.json
    - outputs/step6_prompts/*.txt (optional)
    - outputs/step6_responses/*.txt (optional)

Usage:
    python step6_expert_system.py --kp-context outputs/step5_kp_context_*.txt

Author: Pipeline Modularization Project
Date: 2024-12
"""

import os
import sys
import json
import argparse
import traceback
import re
import requests
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# Import config
from config import CONFIG, FIXED_METRICS, get_timestamp, get_output_path


# ============================================================
# Metric Categories for Prompt Generation
# ============================================================

DISTANCE_METRICS = [
    "mean_closeness_centrality", "max_closeness_centrality",
    "mean_betweenness_centrality", "max_betweenness_centrality",
    "mean_eigenvector_centrality", "max_eigenvector_centrality",
    "mean_degree_centrality", "max_degree_centrality",
    "mean_core_number", "max_core_number"
]

SCALE_METRICS = [
    "number_of_nodes", "number_of_edges", "largest_cc_size", "largest_cc_ratio",
    "density", "average_degree", "min_degree", "max_degree",
    "degree_distribution_entropy", "degree_variance", "degree_coef_variation",
    "network_heterogeneity", "mean_rich_club_coefficient", "max_rich_club_coefficient"
]

PATH_METRICS = [
    "average_shortest_path_length", "diameter", "radius",
    "mean_eccentricity", "min_eccentricity", "max_eccentricity",
    "average_clustering", "min_clustering", "max_clustering",
    "modularity", "spectral_radius", "algebraic_connectivity",
    "graph_energy", "min_eigenvalue", "max_eigenvalue",
    "number_of_communities", "mean_community_size", "max_community_size"
]

# Advanced math functions (if enabled)
ADVANCED_MATH_FUNCTIONS = ["log", "sqrt", "exp", "abs"]
ENABLE_ADVANCED_MATH = CONFIG.get("ENABLE_ADVANCED_MATH", False)


# ============================================================
# Agent Memory Class
# ============================================================

class AgentMemory:
    """Agent memory for storing historical combinations and evaluation results"""

    def __init__(self, output_path: str = None):
        """Initialize memory object"""
        self.combinations: List[Dict] = []
        self.best_combinations: List[Dict] = []
        self.most_used_metrics: Dict[str, int] = {}
        self.iteration: int = 0
        self.output_path = output_path

    def add_combinations(self, combinations: List[Dict], scores: List[float] = None):
        """Add combinations to memory"""
        if not combinations:
            return

        for i, combo in enumerate(combinations):
            if "formula" not in combo:
                continue

            if scores and i < len(scores):
                combo["score"] = scores[i]

            combo["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            combo["iteration"] = self.iteration

            self.combinations.append(combo)

            # Update most used metrics
            formula = combo["formula"]
            var_names = re.findall(r'([a-zA-Z_]+[a-zA-Z0-9_]*)', formula)
            for var in var_names:
                if var not in ["np", "log", "sqrt", "exp", "sin", "cos", "tan"]:
                    self.most_used_metrics[var] = self.most_used_metrics.get(var, 0) + 1

        self.update_best_combinations()
        self.save_memory()

    def update_best_combinations(self, top_n: int = 5):
        """Update best combinations list"""
        if not self.combinations:
            return

        valid = [c for c in self.combinations if "score" in c and c["score"] is not None]
        if not valid:
            return

        sorted_combos = sorted(valid, key=lambda x: x["score"], reverse=True)
        self.best_combinations = sorted_combos[:top_n]

    def get_best_combinations(self, top_n: int = 5) -> List[Dict]:
        """Get best combinations"""
        return self.best_combinations[:top_n]

    def get_most_used_metrics(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently used metrics"""
        sorted_metrics = sorted(self.most_used_metrics.items(), key=lambda x: x[1], reverse=True)
        return sorted_metrics[:top_n]

    def summarize_for_prompt(self, max_combos: int = 3) -> str:
        """Generate memory summary for prompts"""
        if not self.best_combinations:
            return "No historical evaluation data available."

        summary = "Historical best combinations:\n"
        for i, combo in enumerate(self.best_combinations[:max_combos], 1):
            summary += f"{i}. {combo.get('name', 'Unknown')} ({combo['formula']}): Score = {combo.get('score', 0):.4f}\n"

        summary += "\nMost frequently used metrics:\n"
        top_metrics = self.get_most_used_metrics(5)
        for metric, count in top_metrics:
            summary += f"- {metric}: Used {count} times\n"

        return summary

    def save_memory(self):
        """Save memory to file"""
        if not self.output_path:
            return

        try:
            memory_path = os.path.join(self.output_path, "agent_memory.json")
            os.makedirs(os.path.dirname(memory_path), exist_ok=True)

            memory_data = {
                "combinations": self.combinations,
                "best_combinations": self.best_combinations,
                "most_used_metrics": self.most_used_metrics,
                "iteration": self.iteration,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(memory_path, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"[Step6] Error saving memory: {e}")

    def increment_iteration(self):
        """Increment iteration count"""
        self.iteration += 1


# ============================================================
# API Pool Management
# ============================================================

class APIPool:
    """API key pool for managing multiple API keys"""

    def __init__(self, keys_file: str = None, specific_key_index: int = None):
        """Initialize API pool from keys file"""
        self.deepseek_keys: List[str] = []
        self.openai_keys: List[str] = [
            CONFIG.get("API_KEY_PRIMARY", ""),
            CONFIG.get("API_KEY_BACKUP", ""),
            CONFIG.get("API_KEY_THIRD", "")
        ]
        # Filter empty keys
        self.openai_keys = [k for k in self.openai_keys if k]

        # Load from file if provided
        if keys_file and os.path.exists(keys_file):
            try:
                print(f"[Step6] Loading API keys from: {keys_file}")
                with open(keys_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if not line or not line.startswith('sk-'):
                        continue

                    parts = line.split(',')
                    if len(parts) >= 2:
                        key = parts[0].strip()
                        comment = parts[1].strip()

                        if 'deepseek' in comment.lower():
                            self.deepseek_keys.append(key)
                        elif 'gpt' in comment.lower() or 'openai' in comment.lower() or 'ai2api' in comment.lower():
                            self.openai_keys.append(key)
                        else:
                            # Default to OpenAI/GPT-5 for unknown types
                            self.openai_keys.append(key)

                print(f"[Step6] Loaded {len(self.deepseek_keys)} DeepSeek keys, {len(self.openai_keys)} OpenAI keys")

            except Exception as e:
                print(f"[Step6] Error loading keys file: {e}")

        # Specific key selection
        if specific_key_index is not None and specific_key_index < len(self.deepseek_keys):
            self.deepseek_keys = [self.deepseek_keys[specific_key_index]]

        self.current_deepseek_idx = 0
        self.current_openai_idx = 0

    def get_deepseek_key(self) -> Optional[str]:
        """Get next DeepSeek API key"""
        if not self.deepseek_keys:
            return None
        key = self.deepseek_keys[self.current_deepseek_idx]
        self.current_deepseek_idx = (self.current_deepseek_idx + 1) % len(self.deepseek_keys)
        return key

    def get_openai_key(self) -> Optional[str]:
        """Get next OpenAI API key"""
        if not self.openai_keys:
            return None
        key = self.openai_keys[self.current_openai_idx]
        self.current_openai_idx = (self.current_openai_idx + 1) % len(self.openai_keys)
        return key


# ============================================================
# API Calling Functions
# ============================================================

def call_openai_api(prompt: str, api_key: str, verbose: bool = True) -> Optional[str]:
    """Call OpenAI/GPT-5 API via AI2API"""
    if verbose:
        print(f"\n[API] Sending prompt ({len(prompt)} chars) to GPT-5 API...")

    url = CONFIG.get("API_BASE_URL", "https://hi.ai2api.dev/v1") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": CONFIG.get("API_MODEL", "gpt-5-2025-08-07"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 16000
    }

    try:
        # Disable proxy for AI2API - direct connection works better
        response = requests.post(url, headers=headers, json=data, timeout=180,
                                 proxies={"http": None, "https": None})
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            print(f"[API] Response received: {len(content)} chars")
            return content
        else:
            print(f"[API] Request failed: status {response.status_code}")
            print(f"[API] Response: {response.text[:500]}")
            return None
    except Exception as e:
        print(f"[API] Error calling API: {e}")
        return None


def call_deepseek_api(prompt: str, api_key: str, verbose: bool = True) -> Optional[str]:
    """Call DeepSeek API"""
    if verbose:
        print(f"\n[API] Sending prompt ({len(prompt)} chars) to DeepSeek API...")

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 4000
    }

    try:
        json_data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        # Disable proxy for direct connection
        response = requests.post(url, headers=headers, data=json_data, timeout=240,
                                 proxies={"http": None, "https": None})

        if response.status_code == 200:
            response_data = response.json()
            message = response_data["choices"][0]["message"]
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")

            if content and len(content.strip()) >= 10:
                return content
            elif reasoning and len(reasoning) > 100:
                return reasoning
            return None
        else:
            print(f"[API] Request failed: status {response.status_code}")
            return None
    except Exception as e:
        print(f"[API] Error calling API: {e}")
        return None


def call_api(prompt: str, api_pool: APIPool, model: str = "auto", max_retries: int = 3) -> Optional[str]:
    """Call API with model selection, automatic retry and fallback support
    
    Args:
        prompt: The prompt to send
        api_pool: API key pool
        model: 'deepseek', 'openai', or 'auto' (default: try GPT-5 first, then deepseek)
        max_retries: Maximum number of retry attempts per API type
    """
    result = None
    
    # Try GPT-5 (OpenAI API) first with retries
    if model in ("openai", "auto"):
        for attempt in range(max_retries):
            key = api_pool.get_openai_key()
            if key:
                print(f"[API] Trying GPT-5 (AI2API) - attempt {attempt + 1}/{max_retries}...")
                result = call_openai_api(prompt, key)
                if result:
                    return result
                else:
                    print(f"[API] GPT-5 attempt {attempt + 1} failed, will try next key...")
                    import time
                    time.sleep(1)  # Brief pause before retry
    
    # Try DeepSeek as fallback with retries
    if model in ("deepseek", "auto") or result is None:
        for attempt in range(max_retries):
            key = api_pool.get_deepseek_key()
            if key:
                print(f"[API] Falling back to DeepSeek - attempt {attempt + 1}/{max_retries}...")
                result = call_deepseek_api(prompt, key)
                if result:
                    return result
                else:
                    print(f"[API] DeepSeek attempt {attempt + 1} failed, will try next key...")
                    import time
                    time.sleep(1)
    
    print("[API] All API attempts exhausted, returning None")
    return None


# ============================================================
# Prompt Generation Functions
# ============================================================

def categorize_metrics(metrics: List[str]) -> Dict[str, List[str]]:
    """Categorize metrics into distance/scale/path groups"""
    return {
        'distance': [m for m in metrics if m in DISTANCE_METRICS],
        'scale': [m for m in metrics if m in SCALE_METRICS],
        'path': [m for m in metrics if m in PATH_METRICS],
        'other': [m for m in metrics if m not in DISTANCE_METRICS + SCALE_METRICS + PATH_METRICS]
    }


def create_math_expert_prompt(
    problem_description: str,
    metrics: List[str],
    edge_descriptions: List[str] = None,
    memory: AgentMemory = None,
    iteration: int = 1
) -> str:
    """Create MathAgent prompt"""
    cats = categorize_metrics(metrics)
    max_vars = str(CONFIG.get("MAX_FORMULA_VARIABLES", 3))

    if ENABLE_ADVANCED_MATH:
        math_note = f"- You can use mathematical functions: {', '.join(ADVANCED_MATH_FUNCTIONS)}"
    else:
        math_note = "- Formulas must only use basic arithmetic operations (+, -, *, /)"

    prompt = f"""You are a mathematical expert in network science named MathAgent, responsible for designing network metric combination formulas.

We are trying to generate new composite metrics from basic network metrics to better reflect network characteristics. The problem we need to solve is: {problem_description}

Available network metrics grouped by category (only metrics available in current dataset):
- Distance-related topological metrics: {', '.join(cats['distance']) if cats['distance'] else "No available metrics"}
- Scale and structural metrics: {', '.join(cats['scale']) if cats['scale'] else "No available metrics"}
- Path and connectivity metrics: {', '.join(cats['path']) if cats['path'] else "No available metrics"}
{f"- Other metrics: {', '.join(cats['other'])}" if cats['other'] else ""}

"""

    # Add knowledge graph info
    if edge_descriptions and len(edge_descriptions) > 0:
        prompt += "Based on knowledge graph analysis, we found the following relationship information:\n"
        for desc in edge_descriptions[:5]:
            prompt += f"{desc}\n"
    else:
        prompt += "Based on knowledge graph analysis, we did not find relevant knowledge graph information.\n"

    prompt += f"\nIteration: {iteration}\n"

    # Add memory
    if memory and memory.combinations:
        prompt += f"\nHistorical evaluation information:\n{memory.summarize_for_prompt()}\n"
    else:
        prompt += "\nHistorical evaluation information:\nNo historical combination evaluation data available.\n"

    prompt += f"""
Based on the above information, please create 3 network metric combination formulas to solve the problem: {problem_description}

For each combination formula, please provide:
1. Formula: Use basic mathematical operators such as +, -, *, /. Ensure the formula is concise and clear.
2. Name: Give this new metric a descriptive name.
3. Description: Explain the mathematical meaning and network characteristics.

【Important Constraints】:
- Must include at least one multiplication/division operation.
- Create diverse combinations including different types of network metrics.
{math_note}
- 【ABSOLUTELY FORBIDDEN】Do not add any constants (like +1, -0.5) or coefficients (like *0.5, *2) in formulas.
- 【ABSOLUTELY FORBIDDEN】Do not use any numerical values in operations with metrics.
- Each formula can contain at most {max_vars} different metrics.
- 【ABSOLUTELY FORBIDDEN】Do not use any metrics not in the above available metrics list.

【ORTHOGONAL DIMENSION GUIDANCE】
Antifragility = Reach × Scope × Scale

Effective formulas typically combine metrics from three different dimensions:
- Reach (mean_closeness_centrality, mean_eigenvector_centrality) - communication speed
- Scope (diameter, average_shortest_path_length) - network extent
- Scale (number_of_nodes, network_heterogeneity) - system capacity

Products across dimensions reveal emergent network properties.

Please output your suggestions like  the example following format:

COMBINATION_1:
Formula: metric1 * metric2
Name: Composite metric name
Description: This metric measures...

COMBINATION_2:
Formula: metric1 + metric2
Name: Composite metric name
Description: This metric measures...

COMBINATION_3:
Formula: metric1 * metric2 * metric3
Name: Composite metric name
Description: This metric measures...

Only use metrics from the available metrics list above.
"""
    return prompt


def create_biology_expert_prompt(
    problem_description: str,
    metrics: List[str],
    edge_descriptions: List[str] = None,
    memory: AgentMemory = None,
    iteration: int = 1
) -> str:
    """Create BioAgent prompt"""
    cats = categorize_metrics(metrics)
    max_vars = str(CONFIG.get("MAX_FORMULA_VARIABLES", 3))

    if ENABLE_ADVANCED_MATH:
        math_note = f"- You can use mathematical functions: {', '.join(ADVANCED_MATH_FUNCTIONS)}"
    else:
        math_note = "- Formulas must only use basic arithmetic operations (+, -, *, /)"

    prompt = f"""You are a biology expert in microbial community network analysis named BioAgent, responsible for explaining the biological significance of network metrics and designing metric combination formulas.

In microbial interaction network analysis, we are trying to find network metrics that can better characterize microbial community properties. The problem we need to solve is: {problem_description}

Available network metrics grouped by category (only metrics available in current dataset):
- Distance-related topological metrics: {', '.join(cats['distance']) if cats['distance'] else "No available metrics"}
- Scale and structural metrics: {', '.join(cats['scale']) if cats['scale'] else "No available metrics"}
- Path and connectivity metrics: {', '.join(cats['path']) if cats['path'] else "No available metrics"}
{f"- Other metrics: {', '.join(cats['other'])}" if cats['other'] else ""}

"""

    if edge_descriptions and len(edge_descriptions) > 0:
        prompt += "Based on knowledge graph analysis, we found the following relationship information:\n"
        for desc in edge_descriptions[:5]:
            prompt += f"{desc}\n"
    else:
        prompt += "Based on knowledge graph analysis, we did not find relevant knowledge graph information.\n"

    prompt += f"\nIteration: {iteration}\n"

    if memory and memory.combinations:
        prompt += f"\nHistorical evaluation information:\n{memory.summarize_for_prompt()}\n"
    else:
        prompt += "\nHistorical evaluation information:\nNo historical combination evaluation data available.\n"

    prompt += f"""
Based on the above information and your biological expertise, please create 3 network metric combination formulas to solve the problem: {problem_description}

For each combination formula, please provide:
1. Formula: Use basic mathematical operators. Ensure the formula is concise and clear.
2. Name: Give this new metric a descriptive name that reflects its biological significance.
3. Description: Explain the biological meaning and what characteristics of microbial communities it can capture.

【Important Constraints】:
- Must include at least one multiplication/division operation.
- Consider characteristics like inter-species interactions, community stability, functional redundancy.
{math_note}
- 【ABSOLUTELY FORBIDDEN】Do not add any constants or coefficients in formulas.
- Each formula can contain at most {max_vars} different metrics.
- 【ABSOLUTELY FORBIDDEN】Do not use any metrics not in the above available metrics list.

【CRITICAL BIOLOGICAL GUIDANCE - Prompt-Driven Design】
**Core Biological Insight: Antifragility shall combine metrics from different dimensions:
- Reach (mean_closeness_centrality, mean_eigenvector_centrality) - communication speed
- Scope (diameter, average_shortest_path_length) - network extent
- Scale (number_of_nodes, network_heterogeneity) - system capacity
- Communication/Efficiency: Metabolic exchange rate (closeness-related)
- Niche Breadth/Span: The physical or chemical landscape covered. A larger diameter represents a broader ecological niche and more diverse metabolic gradients (diameter-related)
- Diversity/Capacity: Taxonomic richness (node count-related)
Please output your suggestions like the following format:

COMBINATION_1:
Formula: metric1 * metric2
Name: Composite metric name
Description: This metric biologically measures...

COMBINATION_2:
Formula: metric1 / metric2
Name: Composite metric name
Description: This metric biologically measures...

COMBINATION_3:
Formula: metric1 * metric2 * metric3
Name: Composite metric name
Description: This metric biologically measures...

Only use metrics from the available metrics list above.
"""
    return prompt


def create_integration_prompt(
    math_response: str,
    bio_response: str,
    problem_description: str,
    metrics: List[str],
    memory: AgentMemory = None,
    iteration: int = 0,
    math_eval_results: List[Dict] = None,
    bio_eval_results: List[Dict] = None
) -> str:
    """Create IntegrationAgent prompt"""
    cats = categorize_metrics(metrics)
    max_vars = str(CONFIG.get("MAX_FORMULA_VARIABLES", 3))

    if ENABLE_ADVANCED_MATH:
        math_note = f"- You can use basic arithmetic operators and functions: {', '.join(ADVANCED_MATH_FUNCTIONS)}"
    else:
        math_note = "- Formulas must only use basic arithmetic operations (+, -, *, /)"

    memory_summary = memory.summarize_for_prompt() if memory else "No historical data."

    # Format evaluation results
    eval_summary = ""
    if math_eval_results or bio_eval_results:
        eval_summary = "\n### Expert Evaluation Results ###\n\n"
        if math_eval_results:
            eval_summary += "Math Expert Evaluation Results:\n"
            for i, r in enumerate(math_eval_results[:3], 1):
                name = r.get("name", "Unknown") if isinstance(r, dict) else r[0]
                score = r.get("score", 0) if isinstance(r, dict) else r[1]
                eval_summary += f"{i}. {name}: Score = {score:.4f}\n"
        if bio_eval_results:
            eval_summary += "Biology Expert Evaluation Results:\n"
            for i, r in enumerate(bio_eval_results[:3], 1):
                name = r.get("name", "Unknown") if isinstance(r, dict) else r[0]
                score = r.get("score", 0) if isinstance(r, dict) else r[1]
                eval_summary += f"{i}. {name}: Score = {score:.4f}\n"

    prompt = f"""You are an interdisciplinary expert in network science and microbial ecology named IntegrationAgent, responsible for integrating suggestions from mathematics and biology experts to form final network metric combination schemes.

We are trying to develop new composite network metrics for better analysis of microbial interaction networks. The problem we need to solve is: {problem_description}

Available network metrics grouped by category:
- Distance-related topological metrics: {', '.join(cats['distance'])}
- Scale and structural metrics: {', '.join(cats['scale'])}
- Path and connectivity metrics: {', '.join(cats['path'])}
{f"- Other metrics: {', '.join(cats['other'])}" if cats['other'] else ""}

Math Expert Suggestions:
{math_response}

Biology Expert Suggestions:
{bio_response}

{eval_summary}
Current iteration: {iteration + 1}

Historical evaluation information:
{memory_summary}

Based on the suggestions from both experts, please select and integrate 5 final metric combination schemes.

You can:
1. Directly adopt formulas proposed by experts
2. Synthesize ideas from both experts to create new formulas
3. Modify expert-proposed formulas to make them more reasonable

【CRITICAL CONSTRAINTS】:
{math_note}
- 【ABSOLUTELY FORBIDDEN】Do not add any constants or coefficients in formulas
- Each formula can contain at most {max_vars} different metrics
- 【ABSOLUTELY FORBIDDEN】Do not use any metrics not in the above available metrics list

【CRITICAL INTEGRATION GUIDANCE - Prompt-Driven Design】
**Key Principle: Prompt > AUC in Guiding Formula Design**
- AUC scores help identify SEED metrics, but do NOT dictate which metrics should be combined
- Theoretical meaning and complementary information are MORE important than individual AUC

Please output like this format:

COMBINATION_1:
Formula: metric1 * metric2 / metric3
Name: Composite Metric Name
Description: This metric mathematically and biologically means...

COMBINATION_2:
Formula: metric1 / metric2
Name: Composite Metric Name
Description: This metric mathematically and biologically means...

COMBINATION_3:
Formula: metric1 * metric2 * metric3
Name: Composite Metric Name
Description: This metric mathematically and biologically means...

Only use metrics from the available metrics list."""

    return prompt


# ============================================================
# Combination Parsing Functions
# ============================================================

def parse_combinations(response: str, available_metrics: List[str] = None) -> List[Dict]:
    """Parse combinations from API response"""
    print("[Step6] Parsing combinations...")

    if not response or not isinstance(response, str):
        print("[Step6] Invalid response")
        return []

    combinations = []

    # Try standardized format first (COMBINATION_X:)
    pattern = r"COMBINATION_(\d+):\s*\n\s*Formula:\s*([^\n]+)\s*\n\s*Name:\s*([^\n]+)\s*\n\s*Description:\s*(.*?)(?=\n\s*COMBINATION_|\Z)"
    matches = re.findall(pattern, response, re.DOTALL)

    if matches:
        print(f"[Step6] Found {len(matches)} combinations in standardized format")
        for match in matches:
            _, formula, name, description = match
            formula = formula.strip()
            name = name.strip()
            description = description.strip()

            if formula and not re.match(r'^[+\-*/^=:：\s]*$', formula) and len(formula.strip()) > 1:
                combinations.append({
                    "formula": formula,
                    "name": name,
                    "description": description
                })

        if combinations:
            return validate_and_clean_combinations(combinations, available_metrics)

    # Fallback parsing methods
    print("[Step6] Trying fallback parsing methods...")

    # Try backticks format
    formulas_backticks = re.findall(r"`([^`]+)`", response)
    if formulas_backticks:
        for i, formula in enumerate(formulas_backticks):
            formula = formula.strip()
            if any(op in formula for op in ['+', '-', '*', '/']) and not re.match(r'^[+\-*/^=\s]*$', formula):
                combinations.append({
                    "formula": formula,
                    "name": f"Combination{i+1}",
                    "description": f"Formula extracted: {formula}"
                })

    return validate_and_clean_combinations(combinations, available_metrics)


def validate_and_clean_combinations(
    combinations: List[Dict],
    available_metrics: List[str] = None
) -> List[Dict]:
    """Validate and clean parsed combinations"""
    # Filter obviously invalid formulas
    filtered = []
    for combo in combinations:
        formula = combo["formula"]
        if (formula and
            not re.match(r'^[+\-*/^=:：\s]*$', formula) and
            not formula.startswith(':') and
            len(formula.strip()) > 1):
            filtered.append(combo)
        else:
            print(f"[Step6] Filtering invalid formula: '{formula}'")

    combinations = filtered

    # Validate against available metrics
    if available_metrics is not None and combinations:
        print("[Step6] Validating against available metrics...")

        metric_set = set(available_metrics) if isinstance(available_metrics, list) else set()
        allowed_funcs = {"log", "sqrt", "abs", "max", "min", "exp", "sin", "cos", "tan", "np"}
        max_vars = CONFIG.get("MAX_FORMULA_VARIABLES", 3)

        valid_combinations = []
        for combo in combinations:
            formula = combo["formula"]

            # Check variables
            variables = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)', formula)
            invalid_vars = []
            valid_vars = []

            for var in variables:
                if var not in metric_set and var not in allowed_funcs:
                    invalid_vars.append(var)
                elif var in metric_set:
                    valid_vars.append(var)

            if invalid_vars:
                print(f"[Step6] Skipping '{formula}': invalid vars {invalid_vars}")
                continue

            # Check variable count
            unique_vars = set(valid_vars)
            if len(unique_vars) > max_vars:
                print(f"[Step6] Skipping '{formula}': {len(unique_vars)} vars (max: {max_vars})")
                continue

            # Clean formula
            clean_formula = formula.replace("^", "**")
            combo["formula"] = clean_formula
            valid_combinations.append(combo)

        print(f"[Step6] Validated: {len(valid_combinations)}/{len(combinations)} combinations")
        return valid_combinations

    return combinations


# ============================================================
# Main Expert System Class
# ============================================================

class ExpertSystem:
    """Three-agent expert system for generating metric combinations"""

    def __init__(
        self,
        api_pool: APIPool,
        metrics: List[str],
        config: Dict = None
    ):
        """Initialize expert system"""
        self.api_pool = api_pool
        self.metrics = metrics
        self.config = config or CONFIG
        self.memory = AgentMemory()

        # Results storage
        self.math_response: str = ""
        self.bio_response: str = ""
        self.integration_response: str = ""
        self.math_combinations: List[Dict] = []
        self.bio_combinations: List[Dict] = []
        self.integration_combinations: List[Dict] = []
        self.all_combinations: List[Dict] = []

        # Statistics
        self.stats: Dict[str, Any] = {
            "timestamp": None,
            "status": "not_started",
            "math_parsed": 0,
            "bio_parsed": 0,
            "integration_parsed": 0,
            "total_unique": 0,
            "api_calls": 0,
            "api_errors": 0
        }

    def generate_combinations(
        self,
        problem_description: str,
        edge_descriptions: List[str] = None,
        kp_context: str = None,
        iteration: int = 1,
        save_prompts: bool = False,
        output_dir: str = None
    ) -> List[Dict]:
        """Generate combinations using three-agent system"""
        print("\n" + "=" * 60)
        print("Step 6: Expert System - Three-Agent Combination Generation")
        print("=" * 60)
        print(f"【Prompt-Driven Design】Agents guided by theoretical framework (Reach × Span × Scale)")

        self.stats["timestamp"] = datetime.now().isoformat()

        # Step 1: MathAgent
        print("\n[Step 6.1] MathAgent generating formulas...")
        math_prompt = create_math_expert_prompt(
            problem_description, self.metrics, edge_descriptions, self.memory, iteration
        )

        if save_prompts and output_dir:
            self._save_prompt(math_prompt, "math", output_dir)

        self.math_response = call_api(math_prompt, self.api_pool)
        self.stats["api_calls"] += 1

        if self.math_response:
            if save_prompts and output_dir:
                self._save_response(self.math_response, "math", output_dir)
            self.math_combinations = parse_combinations(self.math_response, self.metrics)
            self.stats["math_parsed"] = len(self.math_combinations)
            for combo in self.math_combinations:
                combo["agent_name"] = "MathAgent"
            print(f"[Step 6.1] MathAgent: {len(self.math_combinations)} combinations parsed")
        else:
            self.stats["api_errors"] += 1
            print("[Step 6.1] MathAgent: No response")

        # Step 2: BioAgent
        print("\n[Step 6.2] BioAgent generating formulas...")
        bio_prompt = create_biology_expert_prompt(
            problem_description, self.metrics, edge_descriptions, self.memory, iteration
        )

        if save_prompts and output_dir:
            self._save_prompt(bio_prompt, "bio", output_dir)

        self.bio_response = call_api(bio_prompt, self.api_pool)
        self.stats["api_calls"] += 1

        if self.bio_response:
            if save_prompts and output_dir:
                self._save_response(self.bio_response, "bio", output_dir)
            self.bio_combinations = parse_combinations(self.bio_response, self.metrics)
            self.stats["bio_parsed"] = len(self.bio_combinations)
            for combo in self.bio_combinations:
                combo["agent_name"] = "BioAgent"
            print(f"[Step 6.2] BioAgent: {len(self.bio_combinations)} combinations parsed")
        else:
            self.stats["api_errors"] += 1
            print("[Step 6.2] BioAgent: No response")

        # Step 3: IntegrationAgent
        print("\n[Step 6.3] IntegrationAgent synthesizing...")
        integration_prompt = create_integration_prompt(
            self.math_response or "No math expert response",
            self.bio_response or "No bio expert response",
            problem_description,
            self.metrics,
            self.memory,
            iteration
        )

        if save_prompts and output_dir:
            self._save_prompt(integration_prompt, "integration", output_dir)

        self.integration_response = call_api(integration_prompt, self.api_pool)
        self.stats["api_calls"] += 1

        if self.integration_response:
            if save_prompts and output_dir:
                self._save_response(self.integration_response, "integration", output_dir)
            self.integration_combinations = parse_combinations(self.integration_response, self.metrics)
            self.stats["integration_parsed"] = len(self.integration_combinations)
            for combo in self.integration_combinations:
                combo["agent_name"] = "IntegrationAgent"
            print(f"[Step 6.3] IntegrationAgent: {len(self.integration_combinations)} combinations parsed")
        else:
            self.stats["api_errors"] += 1
            print("[Step 6.3] IntegrationAgent: No response")

        # Step 4: Merge and deduplicate
        print("\n[Step 6.4] Merging and deduplicating combinations...")
        self.all_combinations = self._merge_combinations()
        self.stats["total_unique"] = len(self.all_combinations)
        self.stats["status"] = "success"

        print("\n" + "=" * 60)
        print("Expert System Summary")
        print("=" * 60)
        print(f"  MathAgent: {self.stats['math_parsed']} combinations")
        print(f"  BioAgent: {self.stats['bio_parsed']} combinations")
        print(f"  IntegrationAgent: {self.stats['integration_parsed']} combinations")
        print(f"  Total Unique: {self.stats['total_unique']}")
        print(f"  API Calls: {self.stats['api_calls']}")

        return self.all_combinations

    def _merge_combinations(self) -> List[Dict]:
        """Merge and deduplicate combinations from all agents"""
        all_combos = (
            self.math_combinations +
            self.bio_combinations +
            self.integration_combinations
        )

        # Deduplicate by formula
        seen_formulas = set()
        unique = []
        for combo in all_combos:
            formula = combo["formula"].replace(" ", "").lower()
            if formula not in seen_formulas:
                seen_formulas.add(formula)
                unique.append(combo)

        return unique

    def _save_prompt(self, prompt: str, agent_type: str, output_dir: str):
        """Save prompt to file"""
        prompt_dir = os.path.join(output_dir, "step6_prompts")
        os.makedirs(prompt_dir, exist_ok=True)
        timestamp = get_timestamp()
        path = os.path.join(prompt_dir, f"{agent_type}_prompt_{timestamp}.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(prompt)

    def _save_response(self, response: str, agent_type: str, output_dir: str):
        """Save response to file"""
        response_dir = os.path.join(output_dir, "step6_responses")
        os.makedirs(response_dir, exist_ok=True)
        timestamp = get_timestamp()
        path = os.path.join(response_dir, f"{agent_type}_response_{timestamp}.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(response)

    def save_results(self, output_path: str = None) -> str:
        """Save all results to JSON file"""
        if output_path is None:
            output_path = get_output_path("step6_combinations")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        output_data = {
            **self.stats,
            "math_combinations": self.math_combinations,
            "bio_combinations": self.bio_combinations,
            "integration_combinations": self.integration_combinations,
            "all_combinations": self.all_combinations,
            "config_used": {
                "max_formula_variables": self.config.get("MAX_FORMULA_VARIABLES"),
                "enable_advanced_math": ENABLE_ADVANCED_MATH,
                "api_model": self.config.get("API_MODEL"),
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[Step6] Results saved to: {output_path}")
        return output_path


# ============================================================
# Main Entry Point
# ============================================================

def load_kp_context(kp_path: str) -> str:
    """Load knowledge pump context from file"""
    with open(kp_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_step_data(json_path: str) -> Dict:
    """Load step output JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Step 6: Expert System')
    parser.add_argument('--kp-context', type=str, help='Step 5 KP context file path')
    parser.add_argument('--subgraph-info', type=str, help='Step 4 subgraph info JSON path')
    parser.add_argument('--seed-metrics', type=str, help='Step 3 seed metrics JSON path')
    parser.add_argument('--keys-file', type=str, help='API keys CSV file path')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--problem', type=str, help='Problem description')
    parser.add_argument('--save-prompts', action='store_true', help='Save prompts and responses')
    parser.add_argument('--iteration', type=int, default=1, help='Current iteration number')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Step 6: Expert System")
    print("=" * 60)

    # Load data
    kp_context = ""
    edge_descriptions = []
    seed_metrics = []
    available_metrics = FIXED_METRICS.copy()

    if args.kp_context:
        print(f"\n[Load] Loading KP context: {args.kp_context}")
        kp_context = load_kp_context(args.kp_context)
        print(f"[Load] KP context loaded: {len(kp_context)} chars")

    if args.subgraph_info:
        print(f"\n[Load] Loading subgraph info: {args.subgraph_info}")
        subgraph_data = load_step_data(args.subgraph_info)
        edge_descriptions = subgraph_data.get("edge_descriptions", [])
        print(f"[Load] Loaded {len(edge_descriptions)} edge descriptions")

    if args.seed_metrics:
        print(f"\n[Load] Loading seed metrics: {args.seed_metrics}")
        seed_data = load_step_data(args.seed_metrics)
        seed_metrics = seed_data.get("selected_seeds", [])
        print(f"[Load] Loaded {len(seed_metrics)} seed metrics")
        if seed_metrics:
            available_metrics = seed_metrics

    # Get problem description
    problem = args.problem or CONFIG.get("DEFAULT_PROBLEM", "")
    if not problem:
        problem = "What multi-term arithmetic combinations of network metrics best predict microbial community antifragility?"

    # Initialize API pool
    keys_file = args.keys_file or CONFIG.get("KEYS_FILE_PATH")
    api_pool = APIPool(keys_file)

    # Create expert system
    expert_system = ExpertSystem(api_pool, available_metrics)

    # Generate combinations
    combinations = expert_system.generate_combinations(
        problem_description=problem,
        edge_descriptions=edge_descriptions,
        kp_context=kp_context,
        iteration=args.iteration,
        save_prompts=args.save_prompts,
        output_dir=args.output_dir
    )

    # Save results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    timestamp = get_timestamp()
    output_path = os.path.join(output_dir, f"step6_combinations_{timestamp}.json")
    expert_system.save_results(output_path)

    print("\n" + "=" * 60)
    print("Step 6 Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
