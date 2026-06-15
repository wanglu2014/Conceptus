#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Knowledge Pump Module V3 - Integrated Version
Extract evidence from knowledge graph for GPT-5 API

Data Flow:
    [AUC Results] -> AUC Ranking -> Variable Matching
                                        |
    [Star GML] -> Type Filtering -> Edge Evidence
                                        |
    Sentence-BERT Embedding -> K-Means Clustering -> Summarization -> GPT-5 API

Literature Support:
    - Sentence-BERT: Reimers & Gurevych, 2019
      "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
"""

# Set offline mode before importing any HuggingFace related libraries
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import csv
import numpy as np
import networkx as nx
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import re

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    "GML_PATH": "E:/Onedrive/LabYYc/chatgpt/.history/AgentConc/rep1114/dedup_test/antifragility_STAR_1hop_20251203_235150.gml",
    "OUTPUT_DIR": "E:/Onedrive/LabYYc/chatgpt/.history/AgentConc/rep1114/dedup_test",

    # Sentence-BERT model (Reimers & Gurevych, 2019)
    "SBERT_MODEL": "sentence-transformers/all-MiniLM-L6-v2",

    # Clustering
    "N_CLUSTERS": 5,

    # API Configuration
    "GPT5_API_KEY": "sk-1SuNYVKIN077sIaTk5dZ5ZzdfsU6X7lc2ur03f4pQGTcNlXy",
    "GPT5_BASE_URL": "https://hi.ai2api.dev/v1",
    "GPT5_MODEL": "gpt-5-2025-08-07",
    "MAX_TOKENS": 3000,

    # Default filters (no weighting, just type filtering)
    "DEFAULT_NODE_TYPES": ["metric", "concept", "variable"],
    "DEFAULT_EDGE_TYPES": ["Influence", "Calculation", "Definition"],

    # File retention limit - keep only latest N files per type
    "MAX_FILES_RETENTION": 3,
}


# ============================================================
# File Retention Utility
# ============================================================
def cleanup_old_files(directory: str, pattern: str, max_files: int = 3) -> List[str]:
    """
    Keep only the latest N files matching pattern, delete older ones.

    Args:
        directory: Directory to clean up
        pattern: Glob pattern to match files (e.g., 'knowledge_pump_*.json')
        max_files: Maximum number of files to retain (default: 3)

    Returns:
        List of deleted file paths
    """
    import glob

    search_pattern = os.path.join(directory, pattern)
    matching_files = glob.glob(search_pattern)

    if len(matching_files) <= max_files:
        return []

    files_with_mtime = [(f, os.path.getmtime(f)) for f in matching_files]
    files_sorted = sorted(files_with_mtime, key=lambda x: x[1], reverse=True)

    files_to_delete = [f[0] for f in files_sorted[max_files:]]

    deleted = []
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            deleted.append(file_path)
            print(f"  [Cleanup] Deleted old file: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"  [Cleanup] Failed to delete {file_path}: {e}")

    return deleted


# ============================================================
# Data Classes
# ============================================================
@dataclass
class Evidence:
    """Single edge evidence from knowledge graph"""
    source_node: int
    target_node: int
    source_label: str
    target_label: str
    relation: str
    description: str
    domain: str
    source_type: str
    paper_id: str = ""
    paper_title: str = ""
    paper_url: str = ""
    embedding: Optional[np.ndarray] = None
    cluster_id: int = -1


@dataclass
class FormulaResult:
    """AUC result for a formula"""
    formula: str
    mean_auc: float
    std_auc: float
    mean_prauc: float
    dataset_scores: Dict[str, float]
    variables: List[str] = field(default_factory=list)


@dataclass
class EvidenceCluster:
    """Clustered evidence group"""
    cluster_id: int
    evidence_list: List[Evidence] = field(default_factory=list)
    centroid_idx: int = -1
    summary: str = ""


# ============================================================
# Step 1: AUC Ranking
# ============================================================
class AUCRanker:
    """Rank formulas by AUC scores from evaluation results"""

    def __init__(self, results_path: str = None):
        self.results_path = results_path
        self.formulas: List[FormulaResult] = []

    def load_results(self, results_path: str) -> List[FormulaResult]:
        """Load AUC results from final_results_all_datasets.json"""
        with open(results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        formulas = []
        for item in data:
            # Extract variables from formula string
            formula_str = item.get('formula', '')
            variables = self._extract_variables(formula_str)

            result = FormulaResult(
                formula=formula_str,
                mean_auc=item.get('mean_auc', 0.0),
                std_auc=item.get('std_auc', 0.0),
                mean_prauc=item.get('mean_prauc', 0.0),
                dataset_scores=item.get('dataset_scores', {}),
                variables=variables
            )
            formulas.append(result)

        self.formulas = formulas
        return formulas

    def _extract_variables(self, formula: str) -> List[str]:
        """Extract variable names from formula string"""
        # Remove operators and numbers
        clean = re.sub(r'[\+\-\*\/\(\)\^\d\.\s]', ' ', formula)
        # Split and filter
        tokens = [t.strip() for t in clean.split() if t.strip()]
        # Remove common math functions
        math_funcs = {'log', 'exp', 'sqrt', 'abs', 'sin', 'cos', 'tan'}
        variables = [t for t in tokens if t.lower() not in math_funcs]
        return list(set(variables))

    def get_top_formulas(self, top_n: int = 10, min_auc: float = 0.0) -> List[FormulaResult]:
        """Get top N formulas sorted by mean_auc"""
        filtered = [f for f in self.formulas if f.mean_auc >= min_auc]
        sorted_formulas = sorted(filtered, key=lambda x: x.mean_auc, reverse=True)
        return sorted_formulas[:top_n]

    def get_all_variables(self, formulas: List[FormulaResult] = None) -> List[str]:
        """Get all unique variables from formulas"""
        formulas = formulas or self.formulas
        all_vars = set()
        for f in formulas:
            all_vars.update(f.variables)
        return list(all_vars)


# ============================================================
# Step 2: Type Filtering (No Weighting)
# ============================================================
class TypeFilter:
    """Filter edges by node type and edge type - NO weighting"""

    def __init__(self, G: nx.Graph):
        self.G = G

    def filter_by_node_type(self,
                           node_types: List[str] = None) -> List[int]:
        """Filter nodes by type"""
        node_types = node_types or CONFIG["DEFAULT_NODE_TYPES"]

        matched_nodes = []
        for node_id, data in self.G.nodes(data=True):
            node_type = data.get('type', 'unknown')
            if node_type in node_types:
                matched_nodes.append(node_id)

        return matched_nodes

    def filter_by_edge_type(self,
                           edge_types: List[str] = None) -> List[Tuple]:
        """Filter edges by relation type"""
        edge_types = edge_types or CONFIG["DEFAULT_EDGE_TYPES"]

        matched_edges = []
        for u, v, data in self.G.edges(data=True):
            relation = data.get('relation', 'unknown')
            if relation in edge_types:
                matched_edges.append((u, v, data))

        return matched_edges

    def extract_evidence(self,
                        node_types: List[str] = None,
                        edge_types: List[str] = None,
                        variable_nodes: List[int] = None) -> List[Evidence]:
        """
        Extract evidence edges matching filters

        Args:
            node_types: Filter by source/target node types
            edge_types: Filter by edge relation types
            variable_nodes: If provided, only edges connected to these nodes
        """
        node_types = node_types or CONFIG["DEFAULT_NODE_TYPES"]
        edge_types = edge_types or CONFIG["DEFAULT_EDGE_TYPES"]

        evidence_list = []

        for u, v, data in self.G.edges(data=True):
            # Check edge type
            relation = data.get('relation', 'unknown')
            if relation not in edge_types:
                continue

            # Check node types
            u_type = self.G.nodes[u].get('type', 'unknown')
            v_type = self.G.nodes[v].get('type', 'unknown')

            if u_type not in node_types and v_type not in node_types:
                continue

            # Check if connected to variable nodes (if specified)
            if variable_nodes is not None:
                if u not in variable_nodes and v not in variable_nodes:
                    continue

            # Create evidence object
            ev = Evidence(
                source_node=u,
                target_node=v,
                source_label=self.G.nodes[u].get('label', str(u)),
                target_label=self.G.nodes[v].get('label', str(v)),
                relation=relation,
                description=data.get('description', ''),
                domain=data.get('domain', 'unknown'),
                source_type=data.get('source_type', 'unknown'),
                paper_id=data.get('paper_id', ''),
                paper_title=data.get('paper_title', ''),
                paper_url=data.get('paper_url', '')
            )
            evidence_list.append(ev)

        return evidence_list


# ============================================================
# Step 3: Sentence-BERT Vectorization
# ============================================================
class SBERTVectorizer:
    """
    Vectorize evidence descriptions using Sentence-BERT

    Model: all-MiniLM-L6-v2
    Reference: Reimers & Gurevych, 2019
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or CONFIG["SBERT_MODEL"]
        self.model = None

    def _load_model(self):
        """Lazy load Sentence-BERT model"""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                print(f"Loaded Sentence-BERT model: {self.model_name}")
            except ImportError:
                print("Warning: sentence-transformers not installed")
                print("Install with: pip install sentence-transformers")
                self.model = "fallback"

    def _fallback_tfidf(self, texts: List[str]) -> np.ndarray:
        """Fallback to TF-IDF if Sentence-BERT unavailable"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=384)
        return vectorizer.fit_transform(texts).toarray()

    def vectorize(self, evidence_list: List[Evidence]) -> List[Evidence]:
        """Add embeddings to evidence objects"""
        self._load_model()

        # Prepare texts
        texts = [f"{ev.relation}: {ev.description}" for ev in evidence_list]

        if not texts:
            return evidence_list

        # Generate embeddings
        if self.model == "fallback":
            embeddings = self._fallback_tfidf(texts)
        else:
            embeddings = self.model.encode(texts, show_progress_bar=True)

        # Assign embeddings
        for ev, emb in zip(evidence_list, embeddings):
            ev.embedding = emb

        return evidence_list


# ============================================================
# Step 4: K-Means Clustering
# ============================================================
class EvidenceClusterer:
    """Cluster evidence by embedding similarity using K-Means"""

    def __init__(self, n_clusters: int = None):
        self.n_clusters = n_clusters or CONFIG["N_CLUSTERS"]
        self.clusters: List[EvidenceCluster] = []

    def cluster(self, evidence_list: List[Evidence]) -> List[EvidenceCluster]:
        """Cluster evidence by embeddings"""
        # Filter evidence with embeddings
        ev_with_emb = [ev for ev in evidence_list if ev.embedding is not None]

        if len(ev_with_emb) < self.n_clusters:
            # Not enough evidence, return single cluster
            cluster = EvidenceCluster(
                cluster_id=0,
                evidence_list=ev_with_emb,
                centroid_idx=0
            )
            self.clusters = [cluster]
            return self.clusters

        # Stack embeddings
        embeddings = np.vstack([ev.embedding for ev in ev_with_emb])

        # K-Means clustering
        from sklearn.cluster import KMeans
        n_clusters = min(self.n_clusters, len(ev_with_emb))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Group evidence by cluster
        cluster_dict = defaultdict(list)
        for ev, label in zip(ev_with_emb, labels):
            ev.cluster_id = int(label)
            cluster_dict[label].append(ev)

        # Find centroid (most representative) in each cluster
        self.clusters = []
        for cid, ev_list in cluster_dict.items():
            # Find evidence closest to cluster centroid
            cluster_embs = np.vstack([ev.embedding for ev in ev_list])
            centroid = kmeans.cluster_centers_[cid]
            distances = np.linalg.norm(cluster_embs - centroid, axis=1)
            centroid_idx = np.argmin(distances)

            cluster = EvidenceCluster(
                cluster_id=cid,
                evidence_list=ev_list,
                centroid_idx=centroid_idx
            )
            self.clusters.append(cluster)

        return self.clusters

    def get_representative_evidence(self) -> List[Evidence]:
        """Get one representative evidence per cluster"""
        representatives = []
        for cluster in self.clusters:
            if cluster.evidence_list:
                rep = cluster.evidence_list[cluster.centroid_idx]
                representatives.append(rep)
        return representatives


# ============================================================
# Step 5: Summarization with Anti-Truncation Strategy
# ============================================================
class SafeTruncator:
    """
    Safe truncation utility with 4-layer protection:
    Layer 1: Accurate token counting (tiktoken)
    Layer 2: Priority-based selection (AUC ranking)
    Layer 3: Sentence-level truncation (preserve integrity)
    Layer 4: Safety buffer (15% margin)
    """

    def __init__(self, max_tokens: int = 3000, buffer_ratio: float = 0.15):
        self.max_tokens = max_tokens
        self.buffer_ratio = buffer_ratio
        self.target_tokens = int(max_tokens * (1 - buffer_ratio))
        self._encoder = None

    def _get_encoder(self):
        """Lazy load tiktoken encoder"""
        if self._encoder is None:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                print("Warning: tiktoken not installed, using approximate counting")
                self._encoder = "fallback"
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """Accurate token counting"""
        encoder = self._get_encoder()
        if encoder == "fallback":
            # Fallback: 1.5 chars per token for mixed Chinese/English
            return int(len(text) / 1.5)
        return len(encoder.encode(text))

    def truncate_to_sentence(self, text: str, max_tokens: int = None) -> Tuple[str, int]:
        """
        Truncate text at sentence boundary

        Returns:
            Tuple[str, int]: (truncated_text, actual_token_count)
        """
        max_tokens = max_tokens or self.target_tokens

        # Split by sentence boundaries
        sentence_pattern = r'(?<=[。！？.!?])\s*'
        sentences = re.split(sentence_pattern, text)

        result = []
        current_tokens = 0

        for sentence in sentences:
            if not sentence.strip():
                continue
            sentence_tokens = self.count_tokens(sentence)

            if current_tokens + sentence_tokens <= max_tokens:
                result.append(sentence)
                current_tokens += sentence_tokens
            else:
                # Stop at sentence boundary
                break

        truncated = ''.join(result)
        return truncated, current_tokens

    def safe_truncate(self, text: str) -> Tuple[str, int, bool]:
        """
        Main safe truncation method

        Returns:
            Tuple[str, int, bool]: (text, token_count, was_truncated)
        """
        original_tokens = self.count_tokens(text)

        if original_tokens <= self.target_tokens:
            return text, original_tokens, False

        truncated, actual_tokens = self.truncate_to_sentence(text)
        return truncated, actual_tokens, True


class EvidenceSummarizer:
    """Summarize clustered evidence for API output with anti-truncation protection"""

    def __init__(self, max_tokens: int = None, buffer_ratio: float = 0.15):
        self.max_tokens = max_tokens or CONFIG["MAX_TOKENS"]
        self.buffer_ratio = buffer_ratio
        self.truncator = SafeTruncator(self.max_tokens, buffer_ratio)
        # Deprecated but kept for compatibility
        self.chars_per_token = 4

    def summarize_cluster(self, cluster: EvidenceCluster, max_evidence: int = 3) -> str:
        """Generate summary for a cluster"""
        if not cluster.evidence_list:
            return ""

        # Get top evidence (by description length as proxy for informativeness)
        sorted_ev = sorted(cluster.evidence_list,
                          key=lambda x: len(x.description),
                          reverse=True)[:max_evidence]

        lines = [f"[Cluster {cluster.cluster_id}] (n={len(cluster.evidence_list)})"]
        for ev in sorted_ev:
            lines.append(f"  - [{ev.relation}] {ev.description}")

        cluster.summary = "\n".join(lines)
        return cluster.summary

    def format_for_api(self,
                      clusters: List[EvidenceCluster],
                      top_formulas: List[FormulaResult] = None,
                      query: str = "") -> Dict[str, Any]:
        """Format evidence for GPT-5 API input with safe truncation"""
        output = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "top_formulas": [],
            "evidence_clusters": [],
            "summary_text": "",
            "truncation_info": {
                "was_truncated": False,
                "original_tokens": 0,
                "final_tokens": 0,
                "target_tokens": self.truncator.target_tokens
            }
        }

        current_tokens = 0

        # Add top formulas (priority content - never truncate)
        if top_formulas:
            for f in top_formulas[:5]:
                formula_obj = {
                    "formula": f.formula,
                    "mean_auc": round(f.mean_auc, 4),
                    "std_auc": round(f.std_auc, 4) if hasattr(f, 'std_auc') else 0,
                    "mean_prauc": round(f.mean_prauc, 4) if hasattr(f, 'mean_prauc') else 0,
                    "variables": f.variables[:10]
                }
                output["top_formulas"].append(formula_obj)
                current_tokens += self.truncator.count_tokens(json.dumps(formula_obj))

        # Add cluster summaries with safe truncation
        summary_lines = []
        remaining_tokens = self.truncator.target_tokens - current_tokens

        for cluster in clusters:
            if not cluster.summary:
                self.summarize_cluster(cluster)

            summary_tokens = self.truncator.count_tokens(cluster.summary)

            if summary_tokens > remaining_tokens:
                # Truncate at sentence boundary
                truncated, actual_tokens, was_truncated = self.truncator.safe_truncate(cluster.summary)
                if actual_tokens <= remaining_tokens and truncated:
                    cluster_obj = {
                        "cluster_id": cluster.cluster_id,
                        "evidence_count": len(cluster.evidence_list),
                        "summary": truncated,
                        "truncated": was_truncated
                    }
                    output["evidence_clusters"].append(cluster_obj)
                    summary_lines.append(truncated)
                    remaining_tokens -= actual_tokens
                    output["truncation_info"]["was_truncated"] = True
                break
            else:
                cluster_obj = {
                    "cluster_id": cluster.cluster_id,
                    "evidence_count": len(cluster.evidence_list),
                    "summary": cluster.summary,
                    "truncated": False
                }
                output["evidence_clusters"].append(cluster_obj)
                summary_lines.append(cluster.summary)
                remaining_tokens -= summary_tokens

        output["summary_text"] = "\n\n".join(summary_lines)
        output["truncation_info"]["final_tokens"] = self.truncator.target_tokens - remaining_tokens
        output["truncation_info"]["original_tokens"] = self.truncator.count_tokens(
            "\n\n".join([c.summary for c in clusters if c.summary])
        )

        return output

    def to_prompt_context(self, api_output: Dict) -> str:
        """Convert to plain text for prompt injection"""
        lines = []

        lines.append("=== Knowledge Graph Evidence ===")

        # Top formulas
        if api_output.get("top_formulas"):
            lines.append("\n--- Top Performing Formulas ---")
            for f in api_output["top_formulas"]:
                lines.append(f"  AUC={f['mean_auc']:.3f}: {f['formula']}")

        # Evidence summaries
        lines.append("\n--- Evidence Clusters ---")
        lines.append(api_output.get("summary_text", ""))

        return "\n".join(lines)


# ============================================================
# Step 6: GPT-5 API Caller
# ============================================================
class GPT5Caller:
    """Call GPT-5 API with evidence context"""

    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or CONFIG["GPT5_API_KEY"]
        self.base_url = base_url or CONFIG["GPT5_BASE_URL"]
        self.model = model or CONFIG["GPT5_MODEL"]

    def call(self,
            system_prompt: str,
            user_prompt: str,
            evidence_context: str = "",
            max_tokens: int = 2000) -> str:
        """Call GPT-5 API"""
        import requests

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        if evidence_context:
            user_content = f"{evidence_context}\n\n---\n\n{user_prompt}"
        else:
            user_content = user_prompt

        messages.append({"role": "user", "content": user_content})

        # API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"GPT-5 API error: {e}")
            return f"Error: {str(e)}"


# ============================================================
# Main Knowledge Pump Class
# ============================================================
class KnowledgePump:
    """
    Main orchestrator for knowledge extraction pipeline

    Usage:
        pump = KnowledgePump(gml_path)
        pump.load_auc_results(auc_path)
        evidence = pump.extract(node_types, edge_types)
        context = pump.format_for_gpt5(evidence)
        response = pump.call_gpt5(context, query)
    """

    def __init__(self, gml_path: str = None, config: Dict = None):
        self.config = config or CONFIG
        self.gml_path = gml_path or self.config["GML_PATH"]

        # Load graph
        print(f"Loading graph from: {self.gml_path}")
        self.G = nx.read_gml(self.gml_path, label='id')
        print(f"Loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

        # Initialize components
        self.auc_ranker = AUCRanker()
        self.type_filter = TypeFilter(self.G)
        self.vectorizer = SBERTVectorizer()
        self.clusterer = EvidenceClusterer()
        self.summarizer = EvidenceSummarizer()
        self.api_caller = GPT5Caller()

        # State
        self.top_formulas: List[FormulaResult] = []
        self.evidence: List[Evidence] = []
        self.clusters: List[EvidenceCluster] = []

    def load_auc_results(self, results_path: str, top_n: int = 10, min_auc: float = 0.0):
        """Load and rank AUC results"""
        self.auc_ranker.load_results(results_path)
        self.top_formulas = self.auc_ranker.get_top_formulas(top_n, min_auc)
        print(f"Loaded {len(self.top_formulas)} top formulas")
        return self.top_formulas

    def extract(self,
               node_types: List[str] = None,
               edge_types: List[str] = None,
               use_formula_variables: bool = False) -> List[Evidence]:
        """
        Extract evidence from knowledge graph

        Args:
            node_types: Filter by node types
            edge_types: Filter by edge types
            use_formula_variables: If True, only extract edges connected to formula variables
        """
        print("\n[1/4] Extracting evidence by type...")

        variable_nodes = None
        if use_formula_variables and self.top_formulas:
            # Get all variable names from top formulas
            var_names = self.auc_ranker.get_all_variables(self.top_formulas)
            # Match to node IDs
            variable_nodes = self._match_variables_to_nodes(var_names)
            print(f"  Matched {len(variable_nodes)} variable nodes")

        self.evidence = self.type_filter.extract_evidence(
            node_types=node_types,
            edge_types=edge_types,
            variable_nodes=variable_nodes
        )
        print(f"  Extracted {len(self.evidence)} evidence edges")

        print("\n[2/4] Vectorizing with Sentence-BERT...")
        self.evidence = self.vectorizer.vectorize(self.evidence)

        print("\n[3/4] Clustering evidence...")
        self.clusters = self.clusterer.cluster(self.evidence)
        print(f"  Created {len(self.clusters)} clusters")

        print("\n[4/4] Summarizing clusters...")
        for cluster in self.clusters:
            self.summarizer.summarize_cluster(cluster)

        return self.evidence

    def _match_variables_to_nodes(self, var_names: List[str]) -> List[int]:
        """Match variable names to node IDs in graph"""
        matched = []
        var_names_lower = [v.lower() for v in var_names]

        for node_id, data in self.G.nodes(data=True):
            label = data.get('label', '').lower()
            if label in var_names_lower:
                matched.append(node_id)

        return matched

    def format_for_gpt5(self, query: str = "") -> Dict[str, Any]:
        """Format extracted evidence for GPT-5 API"""
        return self.summarizer.format_for_api(
            clusters=self.clusters,
            top_formulas=self.top_formulas,
            query=query
        )

    def get_prompt_context(self, query: str = "") -> str:
        """Get plain text context for prompt injection"""
        api_output = self.format_for_gpt5(query)
        return self.summarizer.to_prompt_context(api_output)

    def call_gpt5(self,
                 system_prompt: str,
                 user_prompt: str,
                 include_evidence: bool = True) -> str:
        """Call GPT-5 with evidence context"""
        evidence_context = ""
        if include_evidence:
            evidence_context = self.get_prompt_context()

        return self.api_caller.call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            evidence_context=evidence_context
        )

    def save_results(self, output_path: str = None):
        """Save extraction results to JSON"""
        if output_path is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(
                self.config["OUTPUT_DIR"],
                f"knowledge_pump_v3_{ts}.json"
            )

        output = {
            "timestamp": datetime.now().isoformat(),
            "gml_path": self.gml_path,
            "top_formulas": [
                {
                    "formula": f.formula,
                    "mean_auc": float(f.mean_auc),
                    "variables": f.variables
                }
                for f in self.top_formulas
            ],
            "evidence_count": len(self.evidence),
            "clusters": [
                {
                    "cluster_id": int(c.cluster_id),
                    "evidence_count": len(c.evidence_list),
                    "summary": c.summary
                }
                for c in self.clusters
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"Saved results to: {output_path}")
        return output_path

    def save_formulas_csv(self, output_path: str = None, generation: int = 1) -> str:
        """
        Save formula history to CSV with comprehensive metrics

        Columns:
        - timestamp: Extraction timestamp
        - generation: Iteration/generation number
        - rank: Rank by mean_auc (1 = best)
        - formula: Formula expression
        - mean_auc: Mean AUC score
        - std_auc: Standard deviation of AUC
        - mean_prauc: Mean PR-AUC score
        - variables: Extracted variables (semicolon-separated)
        - variable_count: Number of variables
        - agent_source: Source agent (KnowledgePump_V3)
        - gml_source: Source GML file
        - evidence_count: Related evidence count
        - dataset_scores: Per-dataset scores (JSON string)
        """
        if output_path is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(
                self.config["OUTPUT_DIR"],
                f"knowledge_pump_formulas_{ts}.csv"
            )

        fieldnames = [
            'timestamp', 'generation', 'rank', 'formula', 'mean_auc', 'std_auc',
            'mean_prauc', 'variables', 'variable_count', 'agent_source',
            'gml_source', 'evidence_count', 'dataset_scores'
        ]

        # Sort formulas by AUC for ranking
        sorted_formulas = sorted(self.top_formulas, key=lambda x: x.mean_auc, reverse=True)

        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for rank, formula in enumerate(sorted_formulas, start=1):
                row = {
                    'timestamp': datetime.now().isoformat(),
                    'generation': generation,
                    'rank': rank,
                    'formula': formula.formula,
                    'mean_auc': round(formula.mean_auc, 6),
                    'std_auc': round(formula.std_auc, 6),
                    'mean_prauc': round(formula.mean_prauc, 6),
                    'variables': ';'.join(formula.variables),
                    'variable_count': len(formula.variables),
                    'agent_source': 'KnowledgePump_V3',
                    'gml_source': os.path.basename(self.gml_path),
                    'evidence_count': len(self.evidence),
                    'dataset_scores': json.dumps(formula.dataset_scores)
                }
                writer.writerow(row)

        print(f"Saved formulas CSV to: {output_path}")
        return output_path

    def save_evidence_csv(self, output_path: str = None, generation: int = 1) -> str:
        """
        Save evidence details to CSV

        Columns:
        - timestamp: Extraction timestamp
        - generation: Iteration/generation number
        - evidence_rank: Rank within cluster (by description length as proxy)
        - source_node: Source node ID
        - target_node: Target node ID
        - source_label: Source node label
        - target_label: Target node label
        - relation: Edge relation type
        - description: Edge description
        - domain: Knowledge domain
        - source_type: Node type
        - paper_id: Associated paper ID
        - paper_title: Paper title
        - cluster_id: Assigned cluster ID
        - cluster_rank: Cluster rank (by size)
        - agent_source: Source agent
        """
        if output_path is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(
                self.config["OUTPUT_DIR"],
                f"knowledge_pump_evidence_{ts}.csv"
            )

        fieldnames = [
            'timestamp', 'generation', 'evidence_rank', 'source_node', 'target_node',
            'source_label', 'target_label', 'relation', 'description', 'domain',
            'source_type', 'paper_id', 'paper_title', 'cluster_id', 'cluster_rank',
            'agent_source'
        ]

        # Calculate cluster ranks (by size, largest = 1)
        cluster_sizes = {}
        for ev in self.evidence:
            cluster_sizes[ev.cluster_id] = cluster_sizes.get(ev.cluster_id, 0) + 1

        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        cluster_rank_map = {cid: rank for rank, (cid, _) in enumerate(sorted_clusters, start=1)}

        # Group evidence by cluster for ranking within cluster
        cluster_evidence = {}
        for ev in self.evidence:
            if ev.cluster_id not in cluster_evidence:
                cluster_evidence[ev.cluster_id] = []
            cluster_evidence[ev.cluster_id].append(ev)

        # Rank within each cluster by description length (longer = more informative)
        evidence_rank_map = {}
        for cid, ev_list in cluster_evidence.items():
            sorted_ev = sorted(ev_list, key=lambda x: len(x.description) if x.description else 0, reverse=True)
            for rank, ev in enumerate(sorted_ev, start=1):
                evidence_rank_map[(ev.source_node, ev.target_node)] = rank

        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for ev in self.evidence:
                row = {
                    'timestamp': datetime.now().isoformat(),
                    'generation': generation,
                    'evidence_rank': evidence_rank_map.get((ev.source_node, ev.target_node), 0),
                    'source_node': ev.source_node,
                    'target_node': ev.target_node,
                    'source_label': ev.source_label,
                    'target_label': ev.target_label,
                    'relation': ev.relation,
                    'description': ev.description[:500] if ev.description else '',
                    'domain': ev.domain,
                    'source_type': ev.source_type,
                    'paper_id': ev.paper_id,
                    'paper_title': ev.paper_title[:200] if ev.paper_title else '',
                    'cluster_id': ev.cluster_id,
                    'cluster_rank': cluster_rank_map.get(ev.cluster_id, 0),
                    'agent_source': 'KnowledgePump_V3'
                }
                writer.writerow(row)

        print(f"Saved evidence CSV to: {output_path}")
        return output_path

    def save_cluster_summary_csv(self, output_path: str = None, generation: int = 1) -> str:
        """
        Save cluster summaries to CSV

        Columns:
        - timestamp: Extraction timestamp
        - generation: Iteration/generation number
        - cluster_rank: Rank by cluster size (1 = largest)
        - cluster_id: Cluster ID
        - evidence_count: Number of evidence in cluster
        - summary: Cluster summary text
        - centroid_idx: Index of centroid evidence
        - agent_source: Source agent
        """
        if output_path is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(
                self.config["OUTPUT_DIR"],
                f"knowledge_pump_clusters_{ts}.csv"
            )

        fieldnames = [
            'timestamp', 'generation', 'cluster_rank', 'cluster_id',
            'evidence_count', 'summary', 'centroid_idx', 'agent_source'
        ]

        # Sort clusters by size for ranking
        sorted_clusters = sorted(self.clusters, key=lambda x: len(x.evidence_list), reverse=True)

        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for rank, cluster in enumerate(sorted_clusters, start=1):
                row = {
                    'timestamp': datetime.now().isoformat(),
                    'generation': generation,
                    'cluster_rank': rank,
                    'cluster_id': cluster.cluster_id,
                    'evidence_count': len(cluster.evidence_list),
                    'summary': cluster.summary[:1000] if cluster.summary else '',
                    'centroid_idx': cluster.centroid_idx,
                    'agent_source': 'KnowledgePump_V3'
                }
                writer.writerow(row)

        print(f"Saved cluster summary CSV to: {output_path}")
        return output_path

    def save_all_csv(self, generation: int = 1) -> Dict[str, str]:
        """Save all results to CSV files (formulas, evidence, clusters)"""
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = {}

        # Save formulas CSV (only if we have formulas)
        if self.top_formulas:
            results['formulas'] = self.save_formulas_csv(
                os.path.join(self.config["OUTPUT_DIR"], f"knowledge_pump_formulas_{ts}.csv"),
                generation=generation
            )

        # Save evidence CSV
        if self.evidence:
            results['evidence'] = self.save_evidence_csv(
                os.path.join(self.config["OUTPUT_DIR"], f"knowledge_pump_evidence_{ts}.csv"),
                generation=generation
            )

        # Save cluster summary CSV
        if self.clusters:
            results['clusters'] = self.save_cluster_summary_csv(
                os.path.join(self.config["OUTPUT_DIR"], f"knowledge_pump_clusters_{ts}.csv"),
                generation=generation
            )

        print(f"\n=== Saved {len(results)} CSV files (Generation {generation}) ===")
        for name, path in results.items():
            print(f"  {name}: {path}")

        # Cleanup old files - keep only latest MAX_FILES_RETENTION
        max_retention = self.config.get("MAX_FILES_RETENTION", 3)
        output_dir = self.config["OUTPUT_DIR"]

        print(f"\n[File Cleanup] Keeping only latest {max_retention} files per type...")
        cleanup_old_files(output_dir, "knowledge_pump_formulas_*.csv", max_retention)
        cleanup_old_files(output_dir, "knowledge_pump_evidence_*.csv", max_retention)
        cleanup_old_files(output_dir, "knowledge_pump_clusters_*.csv", max_retention)
        cleanup_old_files(output_dir, "knowledge_pump_v3_*.json", max_retention)

        return results


# ============================================================
# Integration Helper for modular_with_efficient_rag_test.py
# ============================================================
def create_enhanced_rag_context(
    gml_path: str,
    auc_results_path: str = None,
    node_types: List[str] = None,
    edge_types: List[str] = None,
    top_n_formulas: int = 10
) -> str:
    """
    One-shot function to create enhanced RAG context

    For integration with modular_with_efficient_rag_test.py:

    ```python
    from knowledge_pump_module import create_enhanced_rag_context

    context = create_enhanced_rag_context(
        gml_path="antifragility_STAR_1hop_*.gml",
        auc_results_path="final_results_all_datasets.json",
        node_types=["metric", "concept"],
        edge_types=["Influence", "Calculation", "Definition"]
    )
    ```
    """
    pump = KnowledgePump(gml_path)

    if auc_results_path:
        pump.load_auc_results(auc_results_path, top_n=top_n_formulas)

    pump.extract(
        node_types=node_types,
        edge_types=edge_types,
        use_formula_variables=bool(auc_results_path)
    )

    return pump.get_prompt_context()


# ============================================================
# Demo / Test
# ============================================================
def main():
    print("=" * 60)
    print("Knowledge Pump V3 - Integrated Version with CSV Output")
    print("=" * 60)

    # Initialize
    pump = KnowledgePump()

    # Extract evidence (without AUC for demo)
    pump.extract(
        node_types=["metric", "concept", "variable"],
        edge_types=["Influence", "Calculation", "Definition"]
    )

    # Get prompt context
    context = pump.get_prompt_context(query="antifragility metrics")
    print("\n" + "=" * 60)
    print("PROMPT CONTEXT:")
    print("=" * 60)
    print(context[:2000] + "..." if len(context) > 2000 else context)

    # Save JSON results
    pump.save_results()

    # Save CSV results (formulas, evidence, clusters) with generation info
    print("\n" + "=" * 60)
    print("SAVING CSV FILES:")
    print("=" * 60)
    csv_files = pump.save_all_csv(generation=1)  # Generation 1 for demo

    # Print truncation info
    api_output = pump.format_for_gpt5(query="antifragility metrics")
    print("\n" + "=" * 60)
    print("TRUNCATION INFO:")
    print("=" * 60)
    trunc_info = api_output.get("truncation_info", {})
    print(f"  Was truncated: {trunc_info.get('was_truncated', False)}")
    print(f"  Original tokens: {trunc_info.get('original_tokens', 0)}")
    print(f"  Final tokens: {trunc_info.get('final_tokens', 0)}")
    print(f"  Target tokens: {trunc_info.get('target_tokens', 0)}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
