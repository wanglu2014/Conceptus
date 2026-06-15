#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entity Extractor for Antifragility Concepts
Extracts antifragility-related concept entities from problem descriptions.
"""

import re
from typing import List

class EntityExtractor:
    """Extract antifragility concept entities from problem descriptions"""

    # Only antifragility-related concepts
    ANTIFRAGILITY_CONCEPTS = [
        "antifragility", "anti-fragility", "antifragile",
        "robustness", "resilience", "stability",
        "stress response", "perturbation response",
        "recovery", "adaptation", "redundancy",
        "fault tolerance", "persistence", "durability",
        "network robustness", "structural stability",
        "functional redundancy", "ecological resilience"
    ]

    def __init__(self):
        """Initialize the entity extractor"""
        pass

    def extract_entities(self, question: str) -> List[str]:
        """
        Extract antifragility concepts from the question

        Args:
            question: The problem description string

        Returns:
            List of extracted antifragility concepts
        """
        question_lower = question.lower()
        extracted = []

        for concept in self.ANTIFRAGILITY_CONCEPTS:
            if concept.lower() in question_lower:
                extracted.append(concept)

        # Always include core concept if question mentions antifragility
        if "antifragil" in question_lower and "antifragility" not in extracted:
            extracted.append("antifragility")

        # If no concepts found but question is about prediction/forecast, add core concepts
        if not extracted and any(kw in question_lower for kw in ["forecast", "predict", "outcome"]):
            extracted = ["antifragility", "robustness", "resilience"]

        return list(set(extracted))

    def get_all_concepts(self) -> List[str]:
        """Return all antifragility concepts for embedding"""
        return self.ANTIFRAGILITY_CONCEPTS.copy()


if __name__ == "__main__":
    # Test
    extractor = EntityExtractor()
    test_question = "What multi-term arithmetic combinations of patient specific microbial network attributes in patients most accurately forecast microbiome antifragility, thereby predicting disease outcome?"

    entities = extractor.extract_entities(test_question)
    print(f"Extracted entities: {entities}")
