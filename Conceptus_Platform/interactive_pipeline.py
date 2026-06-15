#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive Pipeline: Concept Input -> KG Generation -> Formula Discovery

This script provides an interactive interface for:
1. Input research question + CSV network metrics data
2. Extract concepts/keywords from the question (via LLM)
3. Download literature from PubMed/arXiv/Wikipedia
4. Generate a new GML knowledge graph
5. Run the formula discovery pipeline (Step 1-7)

Usage:
    # Interactive mode
    python interactive_pipeline.py

    # Command line mode
    python interactive_pipeline.py --question "Your research question" --csv data.csv --mode demo

Author: Pipeline Modularization Project
Date: 2024-12
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.dirname(PROJECT_ROOT)  # Conceptus_Student_Package
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PACKAGE_ROOT, "kg_build", "kg_build"))  # kg_build/kg_build

# Import configuration
from interactive_config import (
    INTERACTIVE_CONFIG,
    PHASE_DESCRIPTIONS,
    TIME_ESTIMATES
)
from config import CONFIG

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[Warning] tqdm not installed. Progress bars disabled.")


# ============================================================
# Progress Display Utilities
# ============================================================

class ProgressTracker:
    """Track and display progress across pipeline phases"""

    def __init__(self, mode: str = "demo", use_tqdm: bool = True):
        self.mode = mode
        self.use_tqdm = use_tqdm and TQDM_AVAILABLE
        self.current_phase = None
        self.phase_start_time = None

    def start_phase(self, phase_name: str, total_steps: int = 100):
        """Start a new phase with optional progress bar"""
        self.current_phase = phase_name
        self.phase_start_time = datetime.now()

        description = PHASE_DESCRIPTIONS.get(phase_name, phase_name)
        estimate = TIME_ESTIMATES.get(self.mode, {}).get(phase_name, "unknown")

        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"Estimated time: {estimate}")
        print(f"{'='*60}")

        if self.use_tqdm:
            self.pbar = tqdm(total=total_steps, desc=phase_name, unit="step")
        else:
            self.pbar = None

        return self

    def update(self, n: int = 1, desc: str = None):
        """Update progress"""
        if self.pbar:
            if desc:
                self.pbar.set_postfix_str(desc)
            self.pbar.update(n)

    def finish_phase(self):
        """Finish current phase"""
        if self.pbar:
            self.pbar.close()

        elapsed = datetime.now() - self.phase_start_time
        print(f"[{self.current_phase}] Completed in {elapsed.total_seconds():.1f}s")


# ============================================================
# Interactive Input Handler
# ============================================================

class InteractiveInput:
    """Handle interactive user input for research question and data"""

    def __init__(self):
        self.question: str = ""
        self.csv_path: str = ""
        self.mode: str = "demo"

    def collect_inputs(self) -> Dict[str, str]:
        """Interactively collect user inputs"""
        print("\n" + "="*60)
        print("INTERACTIVE FORMULA DISCOVERY PIPELINE")
        print("="*60)

        # Get research question
        print("\n[Step 1] Enter your research question:")
        print("Example: What network metrics predict disease severity?")
        self.question = input(">>> ").strip()

        if not self.question:
            raise ValueError("Research question cannot be empty")

        # Get CSV path
        print("\n[Step 2] Enter path to your CSV network metrics file:")
        print("Example: data/my_metrics.csv")
        self.csv_path = input(">>> ").strip()

        if not self.csv_path:
            raise ValueError("CSV path cannot be empty")

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        # Select mode
        print("\n[Step 3] Select mode:")
        print("  1. demo  - Quick demo (~30 minutes, 30 articles/keyword)")
        print("  2. full  - Full analysis (~2 hours, 100 articles/keyword)")
        mode_input = input(">>> Enter 1 or 2 [default=1]: ").strip()

        if mode_input == "2":
            self.mode = "full"
        else:
            self.mode = "demo"

        # Confirm
        print("\n" + "-"*60)
        print("Configuration Summary:")
        print(f"  Question: {self.question[:60]}...")
        print(f"  CSV Path: {self.csv_path}")
        print(f"  Mode: {self.mode} ({TIME_ESTIMATES[self.mode]['total']})")
        print("-"*60)

        confirm = input("Proceed? [Y/n]: ").strip().lower()
        if confirm == 'n':
            raise KeyboardInterrupt("User cancelled")

        return {
            "question": self.question,
            "csv_path": self.csv_path,
            "mode": self.mode
        }

    @staticmethod
    def from_args(question: str, csv_path: str, mode: str = "demo") -> Dict[str, str]:
        """Create input from command line arguments"""
        if not question:
            raise ValueError("Research question cannot be empty")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        if mode not in INTERACTIVE_CONFIG["MODES"]:
            raise ValueError(f"Invalid mode: {mode}. Choose from: {list(INTERACTIVE_CONFIG['MODES'].keys())}")

        return {
            "question": question,
            "csv_path": csv_path,
            "mode": mode
        }


# ============================================================
# Knowledge Graph Generator
# ============================================================

class KGGenerator:
    """Generate Knowledge Graph from research question"""

    def __init__(self, session_dir: str, mode: str = "demo"):
        self.session_dir = session_dir
        self.mode = mode
        self.mode_config = INTERACTIVE_CONFIG["MODES"][mode]

        # Output directories
        self.literature_dir = os.path.join(session_dir, INTERACTIVE_CONFIG["SUBDIRS"]["literature"])
        self.kg_dir = os.path.join(session_dir, INTERACTIVE_CONFIG["SUBDIRS"]["kg"])

        # Results storage
        self.keywords: Dict = {}
        self.gml_path: str = ""

    def extract_concepts(self, question: str, progress: ProgressTracker) -> Dict:
        """
        Phase 1: Extract concepts from research question

        Uses KeywordExpander from step3_5_keyword_expander.py
        """
        progress.start_phase("concept_extraction", total_steps=3)

        from step3_5_keyword_expander import KeywordExpander

        progress.update(1, "Initializing KeywordExpander")
        expander = KeywordExpander()

        progress.update(1, "Calling LLM for concept extraction")
        self.keywords = expander.expand_from_problem(question)

        progress.update(1, "Concept extraction complete")
        progress.finish_phase()

        # Save keywords
        keywords_path = os.path.join(self.session_dir, "extracted_keywords.json")
        with open(keywords_path, 'w', encoding='utf-8') as f:
            json.dump(self.keywords, f, indent=2, ensure_ascii=False)

        print(f"\n[Concepts Extracted]")
        print(f"  Core concept: {self.keywords.get('concept', 'N/A')}")
        print(f"  Synonyms: {self.keywords.get('synonyms', [])}")
        print(f"  Antonyms: {self.keywords.get('antonyms', [])}")

        return self.keywords

    def download_literature(self, progress: ProgressTracker) -> str:
        """
        Phase 2: Download literature from PubMed/arXiv/Wikipedia

        Uses LiteratureDownloader from kg_build/download_literature.py
        """
        # Collect all keywords
        all_terms = [self.keywords.get("concept", "")]
        all_terms.extend(self.keywords.get("synonyms", []))
        all_terms.extend(self.keywords.get("antonyms", []))
        all_terms = [t for t in all_terms if t]  # Remove empty

        if not all_terms:
            raise ValueError("No keywords extracted from question")

        progress.start_phase("literature_download", total_steps=4)

        # Import downloader
        from download_literature import LiteratureDownloader

        # Create output directory
        os.makedirs(self.literature_dir, exist_ok=True)

        progress.update(1, "Initializing downloader")

        # Initialize downloader - it will create literature_data subdirectory
        max_results = self.mode_config["literature_max"]
        downloader = LiteratureDownloader(
            max_results=max_results,
            output_dir=self.literature_dir
        )

        progress.update(1, f"Downloading for {len(all_terms)} terms")

        # Download all literature at once (handles all 3 sources)
        try:
            wiki_pages, pubmed_articles, arxiv_articles = downloader.download_all_literature(all_terms)

            wiki_count = len(wiki_pages)
            pubmed_count = sum(len(articles) for articles in pubmed_articles.values()) if isinstance(pubmed_articles, dict) else 0
            arxiv_count = len(arxiv_articles)

            progress.update(1, f"Wiki:{wiki_count} PubMed:{pubmed_count} arXiv:{arxiv_count}")
            print(f"\n[Literature Downloaded]")
            print(f"  Wikipedia: {wiki_count} pages")
            print(f"  PubMed: {pubmed_count} articles")
            print(f"  arXiv: {arxiv_count} articles")

        except Exception as e:
            print(f"[Warning] Literature download error: {e}")
            import traceback
            traceback.print_exc()

        progress.update(1, "Download complete")
        progress.finish_phase()

        # Return the actual literature_data directory path
        # LiteratureDownloader creates literature_data subdirectory inside output_dir
        actual_literature_dir = os.path.join(self.literature_dir, "literature_data")
        if os.path.exists(actual_literature_dir):
            return actual_literature_dir
        return self.literature_dir

    def build_kg(self, literature_data_dir: str, progress: ProgressTracker) -> str:
        """
        Phase 3: Build Knowledge Graph from downloaded literature

        Uses NetworkKGBuilder from kg_build/extract_and_process.py

        Args:
            literature_data_dir: Path to the literature_data directory containing
                                 wiki_pages, pubmed, arxiv subdirectories
        """
        progress.start_phase("kg_construction", total_steps=5)

        # Import KG builder
        from extract_and_process import NetworkKGBuilder

        # Create output directory
        os.makedirs(self.kg_dir, exist_ok=True)

        progress.update(1, "Initializing NetworkKGBuilder")

        # Collect terms
        all_terms = [self.keywords.get("concept", "")]
        all_terms.extend(self.keywords.get("synonyms", []))
        all_terms.extend(self.keywords.get("antonyms", []))
        all_terms = [t for t in all_terms if t]

        print(f"\n[KG Builder]")
        print(f"  Input dir: {literature_data_dir}")
        print(f"  Output dir: {self.kg_dir}")
        print(f"  Terms: {all_terms}")

        # Initialize builder with correct input directory
        builder = NetworkKGBuilder(
            max_results=self.mode_config["literature_max"],
            input_dir=literature_data_dir,  # Should contain wiki_pages, pubmed, arxiv
            output_dir=self.kg_dir
        )

        progress.update(1, "Processing literature files")

        # Process existing literature (skip download)
        try:
            builder.process_existing_literature(all_terms)
            progress.update(1, "Extracting entities and relations")
        except Exception as e:
            print(f"[Warning] Literature processing error: {e}")
            import traceback
            traceback.print_exc()
            # Try build_knowledge_graph as fallback
            try:
                builder.build_knowledge_graph(all_terms)
                progress.update(1, "Using build_knowledge_graph fallback")
            except Exception as e2:
                print(f"[Warning] Fallback also failed: {e2}")

        progress.update(1, "Building graph structure")

        # Find the generated GML file - look in graphs subdirectory
        graphs_dir = os.path.join(self.kg_dir, "graphs")
        gml_files = []

        if os.path.exists(graphs_dir):
            # Look for GML files recursively
            for root, dirs, files in os.walk(graphs_dir):
                for f in files:
                    if f.endswith('.gml'):
                        gml_files.append(os.path.join(root, f))

        # Also check kg_dir directly
        if os.path.exists(self.kg_dir):
            for f in os.listdir(self.kg_dir):
                if f.endswith('.gml'):
                    gml_files.append(os.path.join(self.kg_dir, f))

        if gml_files:
            # Prefer network_kg_*.gml (the merged graph)
            merged_gml = [f for f in gml_files if 'network_kg_' in os.path.basename(f)]
            if merged_gml:
                # Use the most recent one
                merged_gml.sort(key=os.path.getmtime, reverse=True)
                self.gml_path = merged_gml[0]
            else:
                # Use the most recent GML file
                gml_files.sort(key=os.path.getmtime, reverse=True)
                self.gml_path = gml_files[0]

            progress.update(1, "KG generation complete")
            progress.finish_phase()

            print(f"\n[Knowledge Graph Generated]")
            print(f"  GML Path: {self.gml_path}")
            print(f"  Total GML files found: {len(gml_files)}")

            return self.gml_path
        else:
            progress.finish_phase()
            raise RuntimeError(f"No GML file generated. Check literature download and extraction.\n"
                             f"  Checked directories: {graphs_dir}, {self.kg_dir}")



# ============================================================
# Formula Discovery Runner
# ============================================================

class FormulaDiscovery:
    """Run formula discovery pipeline with custom KG"""

    def __init__(self, session_dir: str, csv_path: str, gml_path: str, question: str, mode: str = "demo"):
        self.session_dir = session_dir
        self.csv_path = csv_path
        self.gml_path = gml_path
        self.question = question
        self.mode = mode
        self.mode_config = INTERACTIVE_CONFIG["MODES"][mode]

        # Output directory
        self.pipeline_dir = os.path.join(session_dir, INTERACTIVE_CONFIG["SUBDIRS"]["pipeline"])

        # Results
        self.final_results: List[Dict] = []

    def run(self, progress: ProgressTracker) -> List[Dict]:
        """
        Phase 4: Run formula discovery pipeline (Step 1-7)
        """
        progress.start_phase("formula_discovery", total_steps=7)

        # Import pipeline runner
        from main_pipeline import run_pipeline

        # Create custom config
        custom_config = CONFIG.copy()
        custom_config["CSV_DATA_PATH"] = self.csv_path
        custom_config["GML_FILE_PATH"] = self.gml_path
        custom_config["DEFAULT_PROBLEM"] = self.question
        custom_config["MAX_ITERATIONS"] = self.mode_config["iterations"]

        # Create output directory
        os.makedirs(self.pipeline_dir, exist_ok=True)

        print(f"\n[Formula Discovery Configuration]")
        print(f"  CSV: {self.csv_path}")
        print(f"  GML: {self.gml_path}")
        print(f"  Question: {self.question[:50]}...")
        print(f"  Iterations: {self.mode_config['iterations']}")

        # Run pipeline
        progress.update(1, "Starting Step 1: Data Loading")

        try:
            state = run_pipeline(
                start_step=1,
                end_step=7,
                output_dir=self.pipeline_dir,
                config=custom_config,
                enable_logging=True
            )

            progress.update(6, "Pipeline completed")

            # Extract final results
            if state.final_results:
                self.final_results = state.final_results

        except Exception as e:
            print(f"\n[Error] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()

        progress.finish_phase()

        return self.final_results


# ============================================================
# Main Interactive Pipeline Orchestrator
# ============================================================

class InteractivePipeline:
    """Main orchestrator for interactive formula discovery"""

    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = ""
        self.inputs: Dict = {}
        self.progress: Optional[ProgressTracker] = None

    def setup_session(self, inputs: Dict) -> str:
        """Setup session directory and configuration"""
        self.inputs = inputs

        # Create session directory
        template = INTERACTIVE_CONFIG["OUTPUT_DIR_TEMPLATE"]
        self.session_dir = template.format(timestamp=self.session_id)
        self.session_dir = os.path.join(PROJECT_ROOT, self.session_dir)
        os.makedirs(self.session_dir, exist_ok=True)

        # Save session configuration
        config_path = os.path.join(self.session_dir, "session_config.json")
        session_config = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "inputs": inputs,
            "interactive_config": INTERACTIVE_CONFIG
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(session_config, f, indent=2, ensure_ascii=False)

        print(f"\n[Session Created]")
        print(f"  Session ID: {self.session_id}")
        print(f"  Directory: {self.session_dir}")

        # Initialize progress tracker
        use_tqdm = INTERACTIVE_CONFIG.get("PROGRESS_BAR", True)
        self.progress = ProgressTracker(mode=inputs["mode"], use_tqdm=use_tqdm)

        return self.session_dir

    def run(self) -> Dict:
        """Run the complete interactive pipeline"""
        results = {
            "session_id": self.session_id,
            "success": False,
            "keywords": {},
            "gml_path": "",
            "formulas": [],
            "error": None
        }

        try:
            # Phase 1: Concept Extraction
            kg_gen = KGGenerator(self.session_dir, self.inputs["mode"])
            results["keywords"] = kg_gen.extract_concepts(
                self.inputs["question"],
                self.progress
            )

            # Phase 2: Literature Download
            literature_data_dir = kg_gen.download_literature(self.progress)

            # Phase 3: KG Construction
            results["gml_path"] = kg_gen.build_kg(literature_data_dir, self.progress)

            # Phase 4: Formula Discovery
            discovery = FormulaDiscovery(
                session_dir=self.session_dir,
                csv_path=self.inputs["csv_path"],
                gml_path=results["gml_path"],
                question=self.inputs["question"],
                mode=self.inputs["mode"]
            )
            results["formulas"] = discovery.run(self.progress)

            results["success"] = True

        except Exception as e:
            results["error"] = str(e)
            import traceback
            traceback.print_exc()

        # Save final results
        results_path = os.path.join(self.session_dir, "final_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict):
        """Print final summary"""
        print("\n" + "="*60)
        print("PIPELINE COMPLETED")
        print("="*60)

        if results["success"]:
            print(f"\nSession ID: {results['session_id']}")
            print(f"Output Directory: {self.session_dir}")

            print(f"\n[Extracted Concept]")
            print(f"  {results['keywords'].get('concept', 'N/A')}")

            if results["formulas"]:
                print(f"\n[Top Discovered Formulas]")
                for i, formula in enumerate(results["formulas"][:3], 1):
                    if isinstance(formula, dict):
                        name = formula.get("name", formula.get("formula", "Unknown"))
                        auc = formula.get("auc", formula.get("mean_auc", "N/A"))
                        print(f"  {i}. {name}")
                        if isinstance(auc, (int, float)):
                            print(f"     AUC: {auc:.4f}")

            print(f"\n[Output Files]")
            print(f"  Keywords: {self.session_dir}/extracted_keywords.json")
            print(f"  KG: {results['gml_path']}")
            print(f"  Results: {self.session_dir}/final_results.json")
        else:
            print(f"\n[Error] Pipeline failed: {results['error']}")

        print("\n" + "="*60)


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Interactive Formula Discovery Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python interactive_pipeline.py

  # Command line mode
  python interactive_pipeline.py --question "What predicts disease outcome?" --csv data.csv

  # Full mode (more thorough)
  python interactive_pipeline.py --question "..." --csv data.csv --mode full
"""
    )

    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Research question"
    )
    parser.add_argument(
        "--csv", "-c",
        type=str,
        help="Path to CSV file with network metrics"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["demo", "full"],
        default="demo",
        help="Pipeline mode: demo (~30min) or full (~2h)"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = InteractivePipeline()

    try:
        # Collect inputs
        if args.question and args.csv:
            # Command line mode
            inputs = InteractiveInput.from_args(
                question=args.question,
                csv_path=args.csv,
                mode=args.mode
            )
        else:
            # Interactive mode
            input_handler = InteractiveInput()
            inputs = input_handler.collect_inputs()

        # Setup session
        pipeline.setup_session(inputs)

        # Run pipeline
        results = pipeline.run()

        return 0 if results["success"] else 1

    except KeyboardInterrupt:
        print("\n[Cancelled] Pipeline cancelled by user")
        return 1
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
