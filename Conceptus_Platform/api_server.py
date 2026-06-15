#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conceptus API Server

FastAPI server that exposes the Expert System for web interface interaction.
Provides endpoints for running the three-agent system and streaming responses.

Usage:
    uvicorn api_server:app --reload --port 8000

Author: Conceptus Team
Date: 2024-12
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import traceback

# Import from existing modules
from config import CONFIG, FIXED_METRICS, get_timestamp, OUTPUT_DIR
from step6_expert_system import (
    ExpertSystem, APIPool, AgentMemory,
    create_math_expert_prompt, create_biology_expert_prompt, create_integration_prompt,
    call_api, parse_combinations, categorize_metrics
)

# Import authentication and database modules
try:
    from database import init_database, create_default_admin, get_db, User, AnalysisSession, Formula
    from auth import router as auth_router, get_current_user_optional
    HAS_AUTH = True
except ImportError:
    HAS_AUTH = False
    print("[Warning] Auth module not available, running without authentication")

# ============================================================
# FastAPI App Configuration
# ============================================================

app = FastAPI(
    title="Conceptus Pro API",
    description="Multi-Agent Concept Operationalization API with User Authentication",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register authentication router
if HAS_AUTH:
    app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize database on startup"""
        init_database()
        create_default_admin()
        print("[Conceptus Pro] Database initialized")

# Serve landing page
@app.get("/")
async def serve_landing():
    """Serve landing page"""
    landing_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(landing_path):
        return FileResponse(landing_path, media_type="text/html")
    return {"error": "Landing page not found", "redirect": "/app"}

# Serve main app (requires login)
@app.get("/app")
async def serve_frontend():
    """Serve Conceptus Pro frontend"""
    frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "Conceptus_Pro.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path, media_type="text/html")
    return {"error": "Frontend not found"}

# Global API pool
api_pool = APIPool(CONFIG.get("KEYS_FILE_PATH"))


# ============================================================
# Request/Response Models
# ============================================================

class InferenceRequest(BaseModel):
    """Request model for inference"""
    problem: str = CONFIG.get("DEFAULT_PROBLEM", "")
    metrics: Optional[List[str]] = None
    max_iterations: int = 1
    stream: bool = True


class AgentResponse(BaseModel):
    """Response from a single agent"""
    agent_name: str
    response: str
    combinations: List[Dict]
    timestamp: str


class PipelineResult(BaseModel):
    """Complete pipeline result"""
    status: str
    math_response: AgentResponse
    bio_response: AgentResponse
    integration_response: AgentResponse
    all_combinations: List[Dict]
    stats: Dict


# ============================================================
# API Endpoints
# ============================================================

@app.get("/")
async def root():
    """Root endpoint - API status"""
    return {
        "status": "running",
        "name": "Conceptus API",
        "version": "1.0.0",
        "endpoints": {
            "/api/infer": "POST - Run inference with three-agent system",
            "/api/infer/stream": "POST - Run inference with streaming",
            "/api/metrics": "GET - Get available metrics",
            "/api/config": "GET - Get current configuration",
            "/api/history": "GET - Get inference history",
            "/api/kp-graph": "GET - Get Knowledge Pump graph for visualization"
        }
    }


@app.get("/api/metrics")
async def get_metrics():
    """Get available network metrics"""
    from step6_expert_system import DISTANCE_METRICS, SCALE_METRICS, PATH_METRICS

    return {
        "total": len(FIXED_METRICS),
        "metrics": FIXED_METRICS,
        "categories": {
            "distance": DISTANCE_METRICS,
            "scale": SCALE_METRICS,
            "path": PATH_METRICS
        }
    }


@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "max_formula_variables": CONFIG.get("MAX_FORMULA_VARIABLES"),
        "max_iterations": CONFIG.get("MAX_ITERATIONS"),
        "api_model": CONFIG.get("API_MODEL"),
        "seed_top_k": CONFIG.get("SEED_TOP_K"),
        "default_problem": CONFIG.get("DEFAULT_PROBLEM")
    }


@app.post("/api/infer")
async def run_inference(request: InferenceRequest):
    """Run inference with three-agent system (non-streaming)"""
    try:
        metrics = request.metrics or FIXED_METRICS  # Use all metrics if not specified

        expert_system = ExpertSystem(api_pool, metrics)

        combinations = expert_system.generate_combinations(
            problem_description=request.problem,
            edge_descriptions=[],
            kp_context="",
            iteration=1,
            save_prompts=True,
            output_dir=OUTPUT_DIR
        )

        # Save results
        output_path = expert_system.save_results()

        return {
            "status": "success",
            "math_response": {
                "agent_name": "MathAgent",
                "response": expert_system.math_response[:500] + "..." if len(expert_system.math_response) > 500 else expert_system.math_response,
                "combinations": expert_system.math_combinations,
                "timestamp": datetime.now().isoformat()
            },
            "bio_response": {
                "agent_name": "BioAgent",
                "response": expert_system.bio_response[:500] + "..." if len(expert_system.bio_response) > 500 else expert_system.bio_response,
                "combinations": expert_system.bio_combinations,
                "timestamp": datetime.now().isoformat()
            },
            "integration_response": {
                "agent_name": "IntegrationAgent",
                "response": expert_system.integration_response[:500] + "..." if len(expert_system.integration_response) > 500 else expert_system.integration_response,
                "combinations": expert_system.integration_combinations,
                "timestamp": datetime.now().isoformat()
            },
            "all_combinations": combinations,
            "stats": expert_system.stats,
            "output_path": output_path
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/infer/stream")
async def run_inference_stream(request: InferenceRequest):
    """Run inference with streaming response"""

    async def generate():
        try:
            metrics = request.metrics or FIXED_METRICS
            cats = categorize_metrics(metrics)

            # Event: Start
            yield f"data: {json.dumps({'event': 'start', 'message': 'Starting Conceptus inference pipeline...'})}\n\n"
            await asyncio.sleep(0.1)

            # Event: MathAgent starting
            yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'MathAgent', 'message': 'MathAgent analyzing metric compositions...'})}\n\n"
            await asyncio.sleep(0.1)

            # Generate MathAgent prompt and call API
            math_prompt = create_math_expert_prompt(
                request.problem, metrics, [], None, 1
            )

            math_response = call_api(math_prompt, api_pool, "auto")

            if math_response:
                math_combinations = parse_combinations(math_response, metrics)
                for combo in math_combinations:
                    combo["agent_name"] = "MathAgent"

                yield f"data: {json.dumps({'event': 'agent_complete', 'agent': 'MathAgent', 'response': math_response[:1000], 'combinations': math_combinations})}\n\n"
            else:
                yield f"data: {json.dumps({'event': 'agent_error', 'agent': 'MathAgent', 'message': 'No response from MathAgent'})}\n\n"
                math_response = ""
                math_combinations = []

            await asyncio.sleep(0.1)

            # Event: BioAgent starting
            yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'BioAgent', 'message': 'BioAgent evaluating biological plausibility...'})}\n\n"
            await asyncio.sleep(0.1)

            # Generate BioAgent prompt and call API
            bio_prompt = create_biology_expert_prompt(
                request.problem, metrics, [], None, 1
            )

            bio_response = call_api(bio_prompt, api_pool, "auto")

            if bio_response:
                bio_combinations = parse_combinations(bio_response, metrics)
                for combo in bio_combinations:
                    combo["agent_name"] = "BioAgent"

                yield f"data: {json.dumps({'event': 'agent_complete', 'agent': 'BioAgent', 'response': bio_response[:1000], 'combinations': bio_combinations})}\n\n"
            else:
                yield f"data: {json.dumps({'event': 'agent_error', 'agent': 'BioAgent', 'message': 'No response from BioAgent'})}\n\n"
                bio_response = ""
                bio_combinations = []

            await asyncio.sleep(0.1)

            # Event: IntegrationAgent starting
            yield f"data: {json.dumps({'event': 'agent_start', 'agent': 'IntegrationAgent', 'message': 'IntegrationAgent synthesizing perspectives...'})}\n\n"
            await asyncio.sleep(0.1)

            # Generate IntegrationAgent prompt and call API
            integration_prompt = create_integration_prompt(
                math_response or "No math expert response",
                bio_response or "No bio expert response",
                request.problem,
                metrics,
                None,
                0
            )

            integration_response = call_api(integration_prompt, api_pool, "auto")

            if integration_response:
                integration_combinations = parse_combinations(integration_response, metrics)
                for combo in integration_combinations:
                    combo["agent_name"] = "IntegrationAgent"

                yield f"data: {json.dumps({'event': 'agent_complete', 'agent': 'IntegrationAgent', 'response': integration_response[:1000], 'combinations': integration_combinations})}\n\n"
            else:
                yield f"data: {json.dumps({'event': 'agent_error', 'agent': 'IntegrationAgent', 'message': 'No response from IntegrationAgent'})}\n\n"
                integration_combinations = []

            # Merge all combinations
            all_combos = math_combinations + bio_combinations + integration_combinations
            seen = set()
            unique_combos = []
            for c in all_combos:
                formula_key = c["formula"].replace(" ", "").lower()
                if formula_key not in seen:
                    seen.add(formula_key)
                    unique_combos.append(c)

            # Event: Complete
            yield f"data: {json.dumps({'event': 'complete', 'message': 'Pipeline complete!', 'total_combinations': len(unique_combos), 'combinations': unique_combos})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/kp-graph")
async def get_kp_graph():
    """Get the latest Knowledge Pump graph data for visualization"""
    import glob

    pattern = os.path.join(OUTPUT_DIR, "step5_kp_graph_*.json")
    files = glob.glob(pattern)

    if not files:
        raise HTTPException(status_code=404, detail="No KP graph found. Run step5_knowledge_pump.py first.")

    # Get the latest file by modification time
    latest = max(files, key=os.path.getmtime)

    with open(latest, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Add filename for reference
    data["file"] = os.path.basename(latest)
    return data


@app.get("/api/formulas")
async def get_formulas():
    """Get the latest evaluated formulas with AUC scores"""
    import glob

    # Try step7 first (has AUC scores)
    pattern = os.path.join(OUTPUT_DIR, "step7_final_results_*.json")
    files = glob.glob(pattern)

    if files:
        latest = max(files, key=os.path.getmtime)
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data["file"] = os.path.basename(latest)
        data["source"] = "step7"
        return data

    # Fallback to step6 (no AUC scores)
    pattern = os.path.join(OUTPUT_DIR, "step6_combinations_*.json")
    files = glob.glob(pattern)

    if not files:
        raise HTTPException(status_code=404, detail="No formulas found. Run the pipeline first.")

    latest = max(files, key=os.path.getmtime)
    with open(latest, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data["file"] = os.path.basename(latest)
    data["source"] = "step6"
    return data


@app.get("/api/history")
async def get_history():
    """Get recent inference results"""
    import glob

    pattern = os.path.join(OUTPUT_DIR, "step6_combinations_*.json")
    files = glob.glob(pattern)
    files.sort(key=os.path.getmtime, reverse=True)

    results = []
    for f in files[:10]:  # Last 10
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                results.append({
                    "file": os.path.basename(f),
                    "timestamp": data.get("timestamp"),
                    "total_combinations": data.get("total_unique", 0),
                    "api_calls": data.get("api_calls", 0)
                })
        except:
            pass

    return {"history": results}


# ============================================================
# Case 2: Interactive Pipeline Endpoints (Custom Question + KG Build)
# ============================================================

# Store active sessions
active_sessions: Dict[str, Dict] = {}

class InteractiveRequest(BaseModel):
    """Request for interactive pipeline (Case 2)"""
    question: str
    mode: str = "demo"  # "demo" or "full"
    use_existing_kg: bool = True  # True = use pre-built KG, False = build new KG
    csv_content: Optional[str] = None  # CSV file content as string
    csv_filename: Optional[str] = None  # Original filename
    session_id: Optional[str] = None  # Reuse existing session (skip KG build)


@app.post("/api/upload-csv")
async def upload_csv(file: Any = None):
    """Upload CSV file for custom analysis"""
    from fastapi import UploadFile, File
    # Note: This is a placeholder - in production, use proper file upload
    # For now, we'll use the existing demo CSV
    demo_csv = CONFIG.get("CSV_DATA_PATH")
    return {
        "status": "success",
        "csv_path": demo_csv,
        "message": "Using demo CSV. File upload will be implemented in production."
    }


@app.post("/api/interactive/start")
async def start_interactive_pipeline(request: InteractiveRequest):
    """Start interactive pipeline with streaming progress (Case 2)"""
    import uuid
    
    # Check if reusing existing session
    reuse_session = False
    if request.session_id:
        existing_session_dir = os.path.join(OUTPUT_DIR, "interactive", f"session_{request.session_id}")
        existing_kg = os.path.join(existing_session_dir, "kg_output", "graphs")
        if os.path.exists(existing_kg) and len(os.listdir(existing_kg)) > 0:
            session_id = request.session_id
            reuse_session = True
        else:
            session_id = str(uuid.uuid4())[:8]
    else:
        session_id = str(uuid.uuid4())[:8]
    
    async def generate():
        try:
            # Phase 0: Setup
            yield f"data: {json.dumps({'phase': 'setup', 'progress': 0, 'message': 'Initializing session...', 'session_id': session_id})}\n\n"
            
            if request.use_existing_kg:
                # Use existing pre-built KG (fast path)
                yield f"data: {json.dumps({'phase': 'setup', 'progress': 100, 'message': 'Using pre-built knowledge graph'})}\n\n"
                
                # Run standard inference
                metrics = FIXED_METRICS
                expert_system = ExpertSystem(api_pool, metrics)
                
                # Phase: Agent inference
                yield f"data: {json.dumps({'phase': 'inference', 'progress': 10, 'message': 'Starting MathAgent...'})}\n\n"
                
                combinations = expert_system.generate_combinations(
                    problem_description=request.question,
                    edge_descriptions=[],
                    kp_context="",
                    iteration=1,
                    save_prompts=True,
                    output_dir=OUTPUT_DIR
                )
                
                yield f"data: {json.dumps({'phase': 'inference', 'progress': 100, 'message': 'Inference complete'})}\n\n"
                
                # Final results
                result = {
                    "phase": "complete",
                    "session_id": session_id,
                    "success": True,
                    "combinations": combinations,
                    "stats": expert_system.stats
                }
                yield f"data: {json.dumps(result)}\n\n"
                
            else:
                # Check if we can reuse existing session
                session_dir = os.path.join(OUTPUT_DIR, "interactive", f"session_{session_id}")
                
                # Handle CSV
                user_csv_path = None
                if request.csv_content:
                    csv_filename = request.csv_filename or "user_data.csv"
                    user_csv_path = os.path.join(session_dir, csv_filename)
                    os.makedirs(session_dir, exist_ok=True)
                    with open(user_csv_path, 'w', encoding='utf-8') as f:
                        f.write(request.csv_content)
                    yield f"data: {json.dumps({'phase': 'setup', 'progress': 50, 'message': f'CSV saved: {csv_filename}'})}\n\n"
                
                if reuse_session:
                    # Skip KG build phases, just run formula discovery
                    yield f"data: {json.dumps({'phase': 'setup', 'progress': 100, 'message': f'Reusing session {session_id} - skipping KG build'})}\n\n"
                    
                    # Load existing keywords
                    keywords_file = os.path.join(session_dir, "extracted_keywords.json")
                    keywords = {"concept": "reused", "synonyms": [], "antonyms": []}
                    if os.path.exists(keywords_file):
                        with open(keywords_file, 'r', encoding='utf-8') as kf:
                            keywords = json.load(kf)
                else:
                    # Build new KG (slow path)
                    os.makedirs(session_dir, exist_ok=True)
                    
                    # Import interactive pipeline modules
                    PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    sys.path.insert(0, os.path.join(PACKAGE_ROOT, "kg_build", "kg_build"))
                    from interactive_pipeline import KGGenerator
                    
                    kg_gen = KGGenerator(session_dir, request.mode)
                    
                    # Dummy progress tracker
                    class DummyProgress:
                        def start_phase(self, *args, **kwargs): return self
                        def update(self, *args, **kwargs): pass
                        def finish_phase(self, *args, **kwargs): pass
                    
                    # Phase 1: Concept Extraction
                    yield f"data: {json.dumps({'phase': 'concept_extraction', 'progress': 0, 'message': 'Extracting concepts...'})}\n\n"
                    try:
                        keywords = kg_gen.extract_concepts(request.question, DummyProgress())
                        yield f"data: {json.dumps({'phase': 'concept_extraction', 'progress': 100, 'message': 'Concept: ' + keywords.get('concept', 'N/A')})}\n\n"
                        # Save keywords for future reuse
                        with open(os.path.join(session_dir, 'extracted_keywords.json'), 'w') as kf:
                            json.dump(keywords, kf)
                    except Exception as e:
                        keywords = {'concept': 'network', 'synonyms': [], 'antonyms': []}
                        yield f"data: {json.dumps({'phase': 'concept_extraction', 'progress': 100, 'message': 'Using defaults'})}\n\n"
                    
                    # Phase 2: Literature Download
                    yield f"data: {json.dumps({'phase': 'literature_download', 'progress': 0, 'message': 'Downloading...'})}\n\n"
                    try:
                        literature_dir = kg_gen.download_literature(DummyProgress())
                        yield f"data: {json.dumps({'phase': 'literature_download', 'progress': 100, 'message': 'Complete'})}\n\n"
                    except:
                        literature_dir = session_dir
                        yield f"data: {json.dumps({'phase': 'literature_download', 'progress': 100, 'message': 'Skipped'})}\n\n"
                    
                    # Phase 3: KG Construction
                    yield f"data: {json.dumps({'phase': 'kg_construction', 'progress': 0, 'message': 'Building KG...'})}\n\n"
                    try:
                        kg_gen.build_kg(literature_dir, DummyProgress())
                        yield f"data: {json.dumps({'phase': 'kg_construction', 'progress': 100, 'message': 'KG built'})}\n\n"
                    except:
                        yield f"data: {json.dumps({'phase': 'kg_construction', 'progress': 100, 'message': 'Using demo'})}\n\n"
                
                # Phase 4: Formula Discovery
                yield f"data: {json.dumps({'phase': 'formula_discovery', 'progress': 0, 'message': 'Running inference...'})}\n\n"
                
                metrics = FIXED_METRICS
                expert_system = ExpertSystem(api_pool, metrics)
                
                combinations = expert_system.generate_combinations(
                    problem_description=request.question,
                    edge_descriptions=[],
                    kp_context=str(keywords),
                    iteration=1,
                    save_prompts=True,
                    output_dir=session_dir
                )
                
                yield f"data: {json.dumps({'phase': 'formula_discovery', 'progress': 100, 'message': 'Complete'})}\n\n"
                
                # Add validation info
                validation_info = None
                if user_csv_path and os.path.exists(user_csv_path):
                    try:
                        import pandas as pd
                        df = pd.read_csv(user_csv_path)
                        validation_info = {'csv_file': os.path.basename(user_csv_path), 'rows': len(df), 'validated': True}
                    except:
                        validation_info = {'validated': False}
                
                result = {
                    'phase': 'complete',
                    'session_id': session_id,
                    'success': True,
                    'keywords': keywords,
                    'combinations': combinations,
                    'stats': expert_system.stats,
                    'validation': validation_info
                }
                yield f"data: {json.dumps(result)}\n\n"
                
        except Exception as e:
            error_result = {
                "phase": "error",
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            yield f"data: {json.dumps(error_result)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/api/interactive/modes")
async def get_interactive_modes():
    """Get available interactive pipeline modes"""
    from interactive_config import INTERACTIVE_CONFIG, TIME_ESTIMATES
    
    return {
        "modes": {
            "demo": {
                "name": "Demo Mode",
                "description": "Use pre-built knowledge graph (~5 min)",
                "use_existing_kg": True,
                "estimated_time": "5 minutes"
            },
            "custom": {
                "name": "Custom Mode", 
                "description": "Build new knowledge graph from literature (~30 min)",
                "use_existing_kg": False,
                "estimated_time": TIME_ESTIMATES.get("demo", {}).get("total", "30 minutes")
            }
        },
        "default_mode": "demo"
    }


@app.get("/api/interactive/sessions")
async def get_available_sessions():
    """Get list of available sessions that can be reused"""
    sessions = []
    interactive_dir = os.path.join(OUTPUT_DIR, "interactive")
    
    if os.path.exists(interactive_dir):
        for session_name in os.listdir(interactive_dir):
            session_path = os.path.join(interactive_dir, session_name)
            if os.path.isdir(session_path) and session_name.startswith("session_"):
                session_id = session_name.replace("session_", "")
                
                # Check what phases are complete
                kg_output = os.path.join(session_path, "kg_output", "graphs")
                keywords_file = os.path.join(session_path, "extracted_keywords.json")
                
                has_kg = os.path.exists(kg_output) and len(os.listdir(kg_output)) > 0 if os.path.exists(kg_output) else False
                has_keywords = os.path.exists(keywords_file)
                
                # Get creation time
                created = os.path.getmtime(session_path)
                
                sessions.append({
                    "session_id": session_id,
                    "path": session_path,
                    "has_kg": has_kg,
                    "has_keywords": has_keywords,
                    "created": created,
                    "reusable": has_kg  # Can reuse if KG is built
                })
    
    # Sort by creation time, newest first
    sessions.sort(key=lambda x: x["created"], reverse=True)
    
    return {
        "sessions": sessions[:10],  # Return top 10
        "total": len(sessions)
    }


# ============================================================
# SSN (Subject-Specific Network) API Endpoints
# ============================================================

# Import SSN service
try:
    from ssn_service import ssn_manager, get_ssn_status
    HAS_SSN = True
    print("[SSN] Module loaded successfully")
except Exception as e:
    HAS_SSN = False
    import traceback
    print(f"[Warning] SSN module not available: {type(e).__name__}: {e}")
    traceback.print_exc()

from fastapi import UploadFile, File, Form


@app.get("/api/ssn/status")
async def ssn_service_status():
    """Get SSN service status and R environment info"""
    if not HAS_SSN:
        return {"error": "SSN module not available", "available": False}
    
    status = get_ssn_status()
    status["available"] = True
    return status


@app.post("/api/ssn/submit")
async def ssn_submit(
    otu_file: UploadFile = File(..., description="OTU abundance table CSV"),
    meta_file: UploadFile = File(..., description="Metadata CSV with Sampleno and Subgroup columns"),
    threshold: float = Form(0.65, description="OTU filtering threshold"),
    seed: int = Form(73616, description="Random seed for reproducibility")
):
    """Submit SSN computation task
    
    Upload OTU and metadata CSV files to compute Subject-Specific Networks.
    Returns a task_id for tracking progress.
    """
    if not HAS_SSN:
        raise HTTPException(503, "SSN service not available")
    
    # Validate file types
    if not otu_file.filename.endswith('.csv'):
        raise HTTPException(400, "OTU file must be CSV format")
    if not meta_file.filename.endswith('.csv'):
        raise HTTPException(400, "Metadata file must be CSV format")
    
    # Create task
    task = ssn_manager.create_task(threshold=threshold, seed=seed)
    
    # Save uploaded files
    task_dir = ssn_manager.get_task_upload_dir(task.task_id)
    otu_path = task_dir / 'OTUcount_filter.csv'
    meta_path = task_dir / 'meta.csv'
    
    # Write files
    otu_content = await otu_file.read()
    meta_content = await meta_file.read()
    
    with open(otu_path, 'wb') as f:
        f.write(otu_content)
    with open(meta_path, 'wb') as f:
        f.write(meta_content)
    
    # Start background computation
    ssn_manager.run_computation(task.task_id, str(otu_path), str(meta_path))
    
    return {
        "task_id": task.task_id,
        "status": "submitted",
        "message": "SSN computation started. Use /api/ssn/task/{task_id} to check progress."
    }


@app.get("/api/ssn/task/{task_id}")
async def ssn_task_status(task_id: str):
    """Get SSN task status and results summary
    
    Returns task status, progress, and network summary if completed.
    """
    if not HAS_SSN:
        raise HTTPException(503, "SSN service not available")
    
    task = ssn_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Task {task_id} not found")
    
    result = task.to_dict()
    
    # Add detailed summary if completed
    if task.status == 'completed':
        summary = ssn_manager.get_task_summary(task_id)
        if summary:
            result['network_summary'] = summary
    
    return result


@app.get("/api/ssn/result/{task_id}")
async def ssn_download_result(task_id: str, file_type: str = "attrtable"):
    """Download SSN result file
    
    Args:
        task_id: Task ID
        file_type: "attrtable" for full results, "metrics" for Conceptus-compatible format
    """
    if not HAS_SSN:
        raise HTTPException(503, "SSN service not available")
    
    task = ssn_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Task {task_id} not found")
    
    if task.status != 'completed':
        raise HTTPException(400, f"Task not completed. Current status: {task.status}")
    
    if file_type == "attrtable" and task.attr_file:
        return FileResponse(
            task.attr_file,
            filename="attrtable_SSN.csv",
            media_type="text/csv"
        )
    elif file_type == "metrics" and task.metrics_file:
        return FileResponse(
            task.metrics_file,
            filename="metrics.csv",
            media_type="text/csv"
        )
    
    raise HTTPException(404, f"File type '{file_type}' not available")


@app.get("/api/ssn/attributes/{task_id}")
async def ssn_get_attributes(task_id: str):
    """Get network attributes as JSON for display
    
    Returns structured data with samples and groups network attributes.
    """
    if not HAS_SSN:
        raise HTTPException(503, "SSN service not available")
    
    task = ssn_manager.get_task(task_id)
    if not task:
        raise HTTPException(404, f"Task {task_id} not found")
    
    if task.status != 'completed':
        raise HTTPException(400, f"Task not completed. Current status: {task.status}")
    
    attributes = ssn_manager.get_task_attributes(task_id)
    if not attributes:
        raise HTTPException(500, "Failed to read attributes")
    
    return attributes


# ============================================================
# Main Entry
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("Conceptus API Server")
    print("=" * 60)
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"API Model: {CONFIG.get('API_MODEL')}")
    print(f"Available Metrics: {len(FIXED_METRICS)}")
    print("=" * 60)
    print("\nStarting server at http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)

