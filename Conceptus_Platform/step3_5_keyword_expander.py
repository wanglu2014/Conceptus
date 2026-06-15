#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3.5: Keyword Expander Module

Extract core concept from problem description and generate synonyms/antonyms via LLM API.
Then match keywords to knowledge graph nodes using SapBERT embeddings.

Input:
    - Problem description (from config)
    - Node embeddings (SapBERT)

Output:
    - outputs/step3_5_keywords_{timestamp}.json
    - Matched keyword nodes for Step4

Usage:
    python step3_5_keyword_expander.py --problem "Your research problem"

Author: Pipeline Modularization Project
Date: 2024-12
"""

import os
import sys
import json
import argparse
import re
import requests
import numpy as np
from datetime import datetime
from typing import Dict, List, Set, Optional, Any, Tuple

from config import CONFIG, get_timestamp, get_output_path


# ============================================================
# Prompt Template
# ============================================================

KEYWORD_EXPANSION_PROMPT = """You are a scientific terminology expert.

Given a research problem, extract the SINGLE most important concept and generate synonyms/antonyms.

## Research Problem:
{problem_description}

## Task:
Extract ONLY 1 core concept (the prediction target), then generate:
- Synonyms (2-3 terms with similar meaning)
- Antonyms (2-3 terms with opposite meaning)

## Output Format (JSON):
{{
    "concept": "antifragility",
    "synonyms": ["robustness", "resilience"],
    "antonyms": ["fragility", "vulnerability"]
}}

## Requirements:
- Extract ONLY the most important concept (quality over quantity)
- Synonyms/Antonyms: 2-3 each, no more
- Focus on scientific terminology
- Output ONLY valid JSON

## Your Response:"""


# ============================================================
# API Calling Functions
# ============================================================

def call_api(prompt: str, api_key: str = None, verbose: bool = True) -> Optional[str]:
    """
    Call OpenAI/GPT-5 API via AI2API

    Args:
        prompt: The prompt to send
        api_key: API key (uses config default if not provided)
        verbose: Print progress messages

    Returns:
        str: API response content or None if failed
    """
    if api_key is None:
        api_key = CONFIG.get("API_KEY_PRIMARY", "")

    if verbose:
        print(f"[Step3.5] Calling API ({len(prompt)} chars)...")

    url = CONFIG.get("API_BASE_URL", "https://hi.ai2api.dev/v1") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": CONFIG.get("API_MODEL", "gpt-5-2025-08-07"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 2000
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=data,
            timeout=CONFIG.get("API_TIMEOUT", 120)
        )
        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"]
            if verbose:
                print(f"[Step3.5] API response received: {len(content)} chars")
            return content
        else:
            print(f"[Step3.5] API request failed: status {response.status_code}")
            print(f"[Step3.5] Response: {response.text[:500]}")
            return None
    except Exception as e:
        print(f"[Step3.5] Error calling API: {e}")
        return None


def parse_json_response(response: str) -> Optional[Dict]:
    """
    Parse JSON from API response

    Args:
        response: Raw API response text

    Returns:
        Dict: Parsed JSON or None if failed
    """
    if not response:
        return None

    # Try to extract JSON from response
    try:
        # First try direct parsing
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON in code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find JSON-like pattern
    json_match = re.search(r'\{[\s\S]*"concept"[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    print(f"[Step3.5] Failed to parse JSON from response: {response[:200]}")
    return None


# ============================================================
# SapBERT Matching Functions
# ============================================================

def load_embeddings(
    npy_path: str = None,
    mapping_path: str = None
) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Load SapBERT embeddings from NPY file

    Args:
        npy_path: Path to embeddings .npy file
        mapping_path: Path to node mapping JSON

    Returns:
        Tuple[np.ndarray, Dict]: Embeddings array and node mapping (idx -> node_id)
    """
    if npy_path is None:
        npy_path = CONFIG.get("EMBEDDING_NPY_PATH")
    if mapping_path is None:
        mapping_path = CONFIG.get("NODE_MAPPING_PATH")

    try:
        print(f"[Step3.5] Loading embeddings from: {npy_path}")
        embeddings = np.load(npy_path)
        print(f"[Step3.5] Embeddings shape: {embeddings.shape}")

        print(f"[Step3.5] Loading node mapping from: {mapping_path}")
        with open(mapping_path, 'r', encoding='utf-8') as f:
            raw_mapping = json.load(f)

        # Convert to idx -> (node_id, label) format
        node_ids = raw_mapping.get("node_ids", [])
        node_labels = raw_mapping.get("node_labels", [])

        # Create mapping: index -> (node_id, label)
        node_mapping = {}
        for i, (nid, label) in enumerate(zip(node_ids, node_labels)):
            node_mapping[i] = {"node_id": nid, "label": label}

        print(f"[Step3.5] Node mapping: {len(node_mapping)} entries")

        return embeddings, node_mapping

    except Exception as e:
        print(f"[Step3.5] Error loading embeddings: {e}")
        return None, None


def get_sapbert_embedding(text: str) -> Optional[np.ndarray]:
    """
    Get SapBERT embedding for a single text

    Args:
        text: Text to encode

    Returns:
        np.ndarray: Embedding vector
    """
    try:
        from sentence_transformers import SentenceTransformer

        # Use SapBERT model (same model used to generate node_embeddings.npy)
        model_name = CONFIG.get("SAPBERT_MODEL", "cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        model = SentenceTransformer(model_name)

        embedding = model.encode([text], show_progress_bar=False)[0]
        return embedding

    except Exception as e:
        print(f"[Step3.5] Error getting embedding: {e}")
        return None


def find_similar_nodes(
    keyword: str,
    embeddings: np.ndarray,
    node_mapping: Dict,
    threshold: float = 0.7,
    top_k: int = 5
) -> List[Tuple[str, float, str]]:
    """
    Find nodes similar to keyword using cosine similarity

    Args:
        keyword: Keyword to match
        embeddings: Node embeddings array
        node_mapping: Mapping from index to {node_id, label}
        threshold: Similarity threshold
        top_k: Maximum number of matches to return

    Returns:
        List[Tuple[str, float, str]]: List of (node_id, similarity, label) tuples
    """
    # Get keyword embedding
    keyword_emb = get_sapbert_embedding(keyword)
    if keyword_emb is None:
        return []

    # Normalize for cosine similarity
    keyword_emb = keyword_emb / np.linalg.norm(keyword_emb)

    # Calculate similarities
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized_embeddings = embeddings / norms

    similarities = np.dot(normalized_embeddings, keyword_emb)

    # Get top matches above threshold
    matches = []
    sorted_indices = np.argsort(similarities)[::-1]

    for idx in sorted_indices[:top_k * 2]:  # Check more than top_k in case some are below threshold
        sim = similarities[idx]
        if sim >= threshold:
            node_info = node_mapping.get(idx, {})
            node_id = node_info.get("node_id", str(idx))
            label = node_info.get("label", "")
            matches.append((node_id, float(sim), label))

        if len(matches) >= top_k:
            break

    return matches


# ============================================================
# Main Keyword Expander Class
# ============================================================

class KeywordExpander:
    """
    Expand problem description into keywords and match to KG nodes

    Usage:
        expander = KeywordExpander()
        keywords = expander.expand_from_problem(problem)
        nodes = expander.match_to_nodes()
    """

    def __init__(self, api_key: str = None, config: Dict = None):
        """
        Initialize KeywordExpander

        Args:
            api_key: API key for LLM calls
            config: Configuration dictionary
        """
        self.api_key = api_key or CONFIG.get("API_KEY_PRIMARY")
        self.config = config or CONFIG

        # Results storage
        self.problem: str = ""
        self.keywords: Dict = {}  # {concept, synonyms, antonyms}
        self.all_keywords: List[str] = []
        self.matched_nodes: Dict[str, List[Tuple[str, float]]] = {}
        self.keyword_node_ids: Set[str] = set()

        # Embeddings (lazy loaded)
        self._embeddings: Optional[np.ndarray] = None
        self._node_mapping: Optional[Dict] = None

        # Statistics
        self.stats: Dict[str, Any] = {
            "timestamp": None,
            "status": "not_started",
            "problem_length": 0,
            "concept": "",
            "num_synonyms": 0,
            "num_antonyms": 0,
            "total_keywords": 0,
            "total_matched_nodes": 0,
            "api_call_success": False
        }

    def _load_embeddings(self):
        """Lazy load embeddings"""
        if self._embeddings is None:
            self._embeddings, self._node_mapping = load_embeddings()

    def expand_from_problem(self, problem: str) -> Dict:
        """
        Call API to extract concept and generate synonyms/antonyms

        Args:
            problem: Problem description

        Returns:
            Dict: {concept, synonyms, antonyms}
        """
        print("\n" + "=" * 60)
        print("Step 3.5: Keyword Expansion")
        print("=" * 60)

        self.problem = problem
        self.stats["timestamp"] = datetime.now().isoformat()
        self.stats["problem_length"] = len(problem)

        # Generate prompt
        prompt = KEYWORD_EXPANSION_PROMPT.format(problem_description=problem)

        print(f"[Step3.5] Problem: {problem[:100]}...")
        print(f"[Step3.5] Calling API for keyword expansion...")

        # Call API
        response = call_api(prompt, self.api_key)

        if response:
            self.stats["api_call_success"] = True
            parsed = parse_json_response(response)

            if parsed:
                self.keywords = {
                    "concept": parsed.get("concept", ""),
                    "synonyms": parsed.get("synonyms", []),
                    "antonyms": parsed.get("antonyms", [])
                }

                # Collect all keywords
                self.all_keywords = [self.keywords["concept"]]
                self.all_keywords.extend(self.keywords.get("synonyms", []))
                self.all_keywords.extend(self.keywords.get("antonyms", []))
                self.all_keywords = [k for k in self.all_keywords if k]  # Remove empty

                # Update stats
                self.stats["concept"] = self.keywords["concept"]
                self.stats["num_synonyms"] = len(self.keywords.get("synonyms", []))
                self.stats["num_antonyms"] = len(self.keywords.get("antonyms", []))
                self.stats["total_keywords"] = len(self.all_keywords)
                self.stats["status"] = "keywords_extracted"

                print(f"[Step3.5] Concept: {self.keywords['concept']}")
                print(f"[Step3.5] Synonyms: {self.keywords.get('synonyms', [])}")
                print(f"[Step3.5] Antonyms: {self.keywords.get('antonyms', [])}")
            else:
                print("[Step3.5] Failed to parse API response")
                self.stats["status"] = "parse_failed"
        else:
            print("[Step3.5] API call failed")
            self.stats["status"] = "api_failed"

        return self.keywords

    def match_to_nodes(
        self,
        threshold: float = None,
        top_k: int = 5
    ) -> Set[str]:
        """
        Match all keywords to KG nodes using SapBERT similarity

        Args:
            threshold: Similarity threshold (default from config)
            top_k: Max matches per keyword

        Returns:
            Set[str]: Set of matched node IDs
        """
        if threshold is None:
            threshold = self.config.get("SIMILARITY_THRESHOLD", 0.7)

        if not self.all_keywords:
            print("[Step3.5] No keywords to match")
            return set()

        print(f"\n[Step3.5] Matching {len(self.all_keywords)} keywords to KG nodes...")
        print(f"[Step3.5] Threshold: {threshold}, Top-K: {top_k}")

        # Load embeddings
        self._load_embeddings()

        if self._embeddings is None or self._node_mapping is None:
            print("[Step3.5] Embeddings not available, skipping node matching")
            return set()

        # Match each keyword
        self.matched_nodes = {}
        self.keyword_node_ids = set()

        for keyword in self.all_keywords:
            matches = find_similar_nodes(
                keyword,
                self._embeddings,
                self._node_mapping,
                threshold=threshold,
                top_k=top_k
            )

            if matches:
                self.matched_nodes[keyword] = matches
                for node_id, sim, label in matches:
                    self.keyword_node_ids.add(node_id)
                print(f"[Step3.5] '{keyword}' -> {len(matches)} matches (top: {matches[0][1]:.3f}, '{matches[0][2]}')")
            else:
                print(f"[Step3.5] '{keyword}' -> No matches above threshold")

        self.stats["total_matched_nodes"] = len(self.keyword_node_ids)
        self.stats["status"] = "completed"

        print(f"\n[Step3.5] Total unique matched nodes: {len(self.keyword_node_ids)}")

        return self.keyword_node_ids

    def save_results(self, output_path: str = None) -> str:
        """
        Save results to JSON file

        Args:
            output_path: Output file path

        Returns:
            str: Path to saved file
        """
        if output_path is None:
            output_path = get_output_path("step3_5_keywords")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        output_data = {
            **self.stats,
            "problem": self.problem,
            "keywords": self.keywords,
            "all_keywords": self.all_keywords,
            "matched_nodes": {
                k: [(n, s) for n, s, _ in v]
                for k, v in self.matched_nodes.items()
            },
            "keyword_node_ids": list(self.keyword_node_ids),
            "config_used": {
                "api_model": self.config.get("API_MODEL"),
                "similarity_threshold": self.config.get("SIMILARITY_THRESHOLD", 0.7)
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"[Step3.5] Results saved to: {output_path}")
        return output_path


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Step 3.5: Keyword Expander')
    parser.add_argument('--problem', type=str, help='Problem description')
    parser.add_argument('--threshold', type=float, default=0.7, help='Similarity threshold')
    parser.add_argument('--top-k', type=int, default=5, help='Top-K matches per keyword')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--skip-matching', action='store_true', help='Skip node matching')

    args = parser.parse_args()

    # Get problem description
    problem = args.problem or CONFIG.get("DEFAULT_PROBLEM", "")
    if not problem:
        print("[Step3.5] Error: No problem description provided")
        return 1

    print("\n" + "=" * 60)
    print("Step 3.5: Keyword Expander")
    print("=" * 60)

    # Create expander
    expander = KeywordExpander()

    # Expand keywords
    keywords = expander.expand_from_problem(problem)

    if not keywords:
        print("[Step3.5] Failed to extract keywords")
        return 1

    # Match to nodes (optional)
    if not args.skip_matching:
        matched_nodes = expander.match_to_nodes(
            threshold=args.threshold,
            top_k=args.top_k
        )

    # Save results
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    timestamp = get_timestamp()
    output_path = os.path.join(output_dir, f"step3_5_keywords_{timestamp}.json")
    expander.save_results(output_path)

    print("\n" + "=" * 60)
    print("Step 3.5 Complete!")
    print("=" * 60)
    print(f"  Concept: {expander.keywords.get('concept', 'N/A')}")
    print(f"  Synonyms: {expander.keywords.get('synonyms', [])}")
    print(f"  Antonyms: {expander.keywords.get('antonyms', [])}")
    print(f"  Matched Nodes: {len(expander.keyword_node_ids)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
