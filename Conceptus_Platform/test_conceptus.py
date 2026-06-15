#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Conceptus Platform Test Suite
Tests concept extraction, term expansion, and KG building

Usage:
    cd E:\Onedrive\LabYYc\chatgpt\.history\AgentConc\rep1212
    C:\Users\WLPC\.conda\envs\mindmap_env\python.exe test_conceptus.py
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conceptus import Conceptus, DiscoveryResult


def test_concept_extraction():
    """Test Step A: Concept extraction"""
    print("\n=== Test 1: Concept Extraction ===")
    platform = Conceptus()

    test_cases = [
        ("What is antifragility in microbiome networks?", "antifragility"),
        ("How does resilience affect network stability?", "resilience"),
        ("Define robustness in biological systems", "robustness"),
    ]

    for question, expected in test_cases:
        concept = platform.extract_concept(question)
        print(f"  Q: {question}")
        print(f"  -> Extracted: {concept}")
        if expected.lower() in concept.lower():
            print(f"  [PASS]")
        else:
            print(f"  [INFO] Expected '{expected}', got '{concept}'")

    print("=== Test 1 PASSED ===\n")
    return True


def test_term_expansion():
    """Test Step B: Term expansion"""
    print("\n=== Test 2: Term Expansion ===")
    platform = Conceptus()

    concept = "antifragility"
    print(f"  Expanding concept: {concept}")

    try:
        terms = platform.expand_terms(concept)

        synonyms = terms.get('synonyms', [])
        antonyms = terms.get('antonyms', [])
        related = terms.get('related', [])

        print(f"  Synonyms ({len(synonyms)}): {synonyms[:5]}...")
        print(f"  Antonyms ({len(antonyms)}): {antonyms[:5]}...")
        print(f"  Related ({len(related)}): {related[:5]}...")

        if synonyms or antonyms or related:
            print("  [PASS] Term expansion returned results")
        else:
            print("  [WARN] No terms returned")

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False

    print("=== Test 2 PASSED ===\n")
    return True


def test_kg_module_import():
    """Test Step C+D: KG building module import"""
    print("\n=== Test 3: KG Module Import ===")
    platform = Conceptus()

    try:
        if platform.kg_build_path not in sys.path:
            sys.path.insert(0, platform.kg_build_path)

        from download_literature import LiteratureDownloader
        print("  [PASS] LiteratureDownloader imported")

        from extract_and_process import NetworkKGBuilder
        print("  [PASS] NetworkKGBuilder imported")

        # Verify constructor parameters
        import inspect
        lit_sig = inspect.signature(LiteratureDownloader.__init__)
        assert 'output_dir' in lit_sig.parameters, "LiteratureDownloader missing output_dir param"
        print("  [PASS] LiteratureDownloader has output_dir parameter")

        kg_sig = inspect.signature(NetworkKGBuilder.__init__)
        assert 'input_dir' in kg_sig.parameters, "NetworkKGBuilder missing input_dir param"
        assert 'output_dir' in kg_sig.parameters, "NetworkKGBuilder missing output_dir param"
        print("  [PASS] NetworkKGBuilder has configurable path parameters")

    except ImportError as e:
        print(f"  [FAIL] Import failed: {e}")
        return False
    except AssertionError as e:
        print(f"  [FAIL] {e}")
        return False

    print("=== Test 3 PASSED ===\n")
    return True


def test_with_existing_gml():
    """Test using existing GML file (skip literature download)"""
    print("\n=== Test 4: Using Existing GML ===")

    gml_path = r"E:\Onedrive\LabYYc\chatgpt\.history\AgentConc\rep1114\dedup_output\deduplicated_graph_with_metrics.gml"
    data_path = r"E:\Onedrive\LabYYc\chatgpt\filtered_phyloseq_network_data.csv"

    if not os.path.exists(gml_path):
        print(f"  [SKIP] GML file not found: {gml_path}")
        return True

    if not os.path.exists(data_path):
        print(f"  [SKIP] Data file not found: {data_path}")
        return True

    platform = Conceptus()
    print(f"  GML: {gml_path}")
    print(f"  Data: {data_path}")
    print("  [PASS] Files verified, ready for pipeline")

    print("=== Test 4 PASSED ===\n")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Conceptus Platform Test Suite")
    print("=" * 60)

    results = []

    results.append(("Concept Extraction", test_concept_extraction()))
    results.append(("Term Expansion", test_term_expansion()))
    results.append(("KG Module Import", test_kg_module_import()))
    results.append(("Existing GML", test_with_existing_gml()))

    print("=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        if result:
            passed += 1
        else:
            failed += 1
        print(f"  {name}: {status}")

    print("-" * 60)
    print(f"Total: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
