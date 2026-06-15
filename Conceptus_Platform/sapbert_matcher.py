#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SapBERT Matcher for Antifragility Concept Matching
Uses pre-computed SapBERT embeddings to match concepts to knowledge graph nodes.
"""

import json
import numpy as np
from typing import List, Dict, Tuple

class SapBERTMatcher:
    """Match antifragility concepts to knowledge graph nodes using SapBERT embeddings"""

    def __init__(self, embeddings_path: str, similarity_threshold: float = 0.90):
        """
        Initialize the matcher

        Args:
            embeddings_path: Path to pre-computed SapBERT embeddings JSON
            similarity_threshold: Minimum cosine similarity for matching (default: 0.90)
        """
        self.embeddings_path = embeddings_path
        self.similarity_threshold = similarity_threshold
        self.node_embeddings = {}
        self.node_labels = {}
        self._load_embeddings()

    def _load_embeddings(self):
        """Load pre-computed embeddings from JSON file"""
        try:
            with open(self.embeddings_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different JSON formats
            if isinstance(data, dict):
                if 'embeddings' in data:
                    embeddings_data = data['embeddings']
                else:
                    embeddings_data = data

                for node_id, info in embeddings_data.items():
                    if isinstance(info, dict):
                        if 'embedding' in info:
                            self.node_embeddings[node_id] = np.array(info['embedding'])
                        if 'label' in info:
                            self.node_labels[node_id] = info['label']
                    elif isinstance(info, list):
                        # Direct embedding array
                        self.node_embeddings[node_id] = np.array(info)

            print(f"[SapBERT] Loaded {len(self.node_embeddings)} node embeddings")

        except Exception as e:
            print(f"[SapBERT] Failed to load embeddings: {e}")
            self.node_embeddings = {}

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def encode_concept(self, concept: str) -> np.ndarray:
        """
        Encode a concept string to embedding vector.
        Uses pre-computed embeddings if available, otherwise returns zero vector.
        """
        # Check if concept exists in pre-computed embeddings
        for node_id, label in self.node_labels.items():
            if label and concept.lower() in label.lower():
                return self.node_embeddings.get(node_id, np.zeros(768))

        # Return zero vector if not found (will not match anything above threshold)
        return np.zeros(768)

    def find_matching_nodes(self, concepts: List[str], top_k: int = 5) -> Dict[str, List[Tuple[str, float, str]]]:
        """
        Find knowledge graph nodes matching the given concepts

        Args:
            concepts: List of concept strings to match
            top_k: Maximum number of matches per concept

        Returns:
            Dict mapping concept -> [(node_id, similarity_score, label), ...]
        """
        matches = {}

        for concept in concepts:
            concept_matches = []
            concept_lower = concept.lower()

            # Direct label matching first (high confidence)
            for node_id, label in self.node_labels.items():
                if label:
                    label_lower = label.lower()
                    # Exact or substring match
                    if concept_lower in label_lower or label_lower in concept_lower:
                        concept_matches.append((node_id, 0.95, label))

            # If no direct matches, try embedding similarity
            if not concept_matches and self.node_embeddings:
                concept_vec = self.encode_concept(concept)
                if np.any(concept_vec):  # Non-zero vector
                    for node_id, node_vec in self.node_embeddings.items():
                        sim = self._cosine_similarity(concept_vec, node_vec)
                        if sim >= self.similarity_threshold:
                            label = self.node_labels.get(node_id, node_id)
                            concept_matches.append((node_id, sim, label))

            # Sort by similarity and take top_k
            concept_matches.sort(key=lambda x: x[1], reverse=True)
            matches[concept] = concept_matches[:top_k]

        # Log results
        total_matches = sum(len(m) for m in matches.values())
        print(f"[SapBERT] Found {total_matches} matches for {len(concepts)} concepts (threshold={self.similarity_threshold})")

        return matches


if __name__ == "__main__":
    # Test
    embeddings_path = "E:/Onedrive/LabYYc/chatgpt/.history/AgentConc/rep1114/merged_output_sapbert/sapbert_embeddings.json"
    matcher = SapBERTMatcher(embeddings_path, similarity_threshold=0.90)

    test_concepts = ["antifragility", "robustness", "resilience"]
    matches = matcher.find_matching_nodes(test_concepts)

    for concept, node_matches in matches.items():
        print(f"\n{concept}:")
        for node_id, score, label in node_matches:
            print(f"  {node_id}: {label} (score={score:.3f})")
