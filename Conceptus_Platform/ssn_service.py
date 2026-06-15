#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSN Service Module
Provides SSN (Subject-Specific Network) computation functionality 
via R script integration.

Date: 2025-12-28
"""

import os
import uuid
import subprocess
import threading
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).parent
R_SCRIPT_DIR = BASE_DIR / 'r_scripts'
SSN_SCRIPT = R_SCRIPT_DIR / 'SSN_compute.R'
UPLOADS_DIR = BASE_DIR / 'uploads' / 'ssn'
RESULTS_DIR = BASE_DIR / 'outputs' / 'ssn'

# R executable path - configurable via environment variable
# R executable path configuration
if os.name == 'nt':
    RSCRIPT_PATH = os.environ.get('RSCRIPT_PATH', r'Y:\_Software\R\R-4.5.2\bin\Rscript.exe')
else:
    # Linux: Use dedicated 'r45' Conda environment (has SpiecEasi and all deps)
    RSCRIPT_PATH = os.environ.get('RSCRIPT_PATH', '/root/miniconda3/envs/r45/bin/Rscript')
    if not os.path.exists(RSCRIPT_PATH):
        # Fallback chain
        RSCRIPT_PATH = '/root/miniconda3/envs/ssn/bin/Rscript'
        if not os.path.exists(RSCRIPT_PATH):
            RSCRIPT_PATH = '/root/miniconda3/bin/Rscript'
            if not os.path.exists(RSCRIPT_PATH):
                RSCRIPT_PATH = '/usr/bin/Rscript'


# ============================================================
# Data Classes
# ============================================================
@dataclass
class SSNTask:
    """Represents an SSN computation task"""
    task_id: str
    status: str = 'pending'  # pending, running, completed, failed
    progress: str = ''
    error: Optional[str] = None
    result_dir: Optional[str] = None
    attr_file: Optional[str] = None
    metrics_file: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    threshold: float = 0.65
    seed: int = 73616
    sample_count: int = 0
    group_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response"""
        return {
            'task_id': self.task_id,
            'status': self.status,
            'progress': self.progress,
            'error': self.error,
            'threshold': self.threshold,
            'seed': self.seed,
            'sample_count': self.sample_count,
            'group_count': self.group_count,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# ============================================================
# Task Manager
# ============================================================
class SSNTaskManager:
    """Manages SSN computation tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, SSNTask] = {}
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    def create_task(self, threshold: float = 0.65, seed: int = 73616) -> SSNTask:
        """Create a new SSN task"""
        task_id = str(uuid.uuid4())[:8]
        task = SSNTask(task_id=task_id, threshold=threshold, seed=seed)
        self.tasks[task_id] = task
        logger.info(f"[SSN] Created task {task_id}")
        return task
    
    def get_task(self, task_id: str) -> Optional[SSNTask]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_task_upload_dir(self, task_id: str) -> Path:
        """Get upload directory for a task"""
        task_dir = UPLOADS_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        return task_dir
    
    def get_task_result_dir(self, task_id: str) -> Path:
        """Get result directory for a task"""
        result_dir = RESULTS_DIR / task_id
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir
    
    def check_r_available(self) -> Tuple[bool, str]:
        """Check if R is available on the system"""
        try:
            result = subprocess.run(
                [RSCRIPT_PATH, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version_info = result.stderr or result.stdout
                return True, version_info.strip()
            return False, "R not found or not working"
        except FileNotFoundError:
            return False, f"Rscript not found at: {RSCRIPT_PATH}"
        except Exception as e:
            return False, str(e)
    
    def run_computation(self, task_id: str, otu_path: str, meta_path: str):
        """Run SSN computation in background thread"""
        thread = threading.Thread(
            target=self._run_ssn_script,
            args=(task_id, otu_path, meta_path),
            daemon=True
        )
        thread.start()
        logger.info(f"[SSN] Started computation thread for task {task_id}")
    
    def _run_ssn_script(self, task_id: str, otu_path: str, meta_path: str):
        """Execute SSN R script (runs in background thread)"""
        task = self.tasks.get(task_id)
        if not task:
            logger.error(f"[SSN] Task {task_id} not found")
            return
        
        task.status = 'running'
        task.progress = 'Checking R environment...'
        
        # Check R availability
        r_available, r_info = self.check_r_available()
        if not r_available:
            task.status = 'failed'
            task.error = f"R environment not available: {r_info}. Please install R and set RSCRIPT_PATH environment variable."
            logger.error(f"[SSN] R not available: {r_info}")
            return
        
        task.progress = 'Starting R computation...'
        result_dir = self.get_task_result_dir(task_id)
        
        try:
            # Build command
            cmd = [
                RSCRIPT_PATH,
                str(SSN_SCRIPT),
                otu_path,
                meta_path,
                str(task.threshold),
                str(task.seed),
                str(result_dir)
            ]
            
            # Prepare environment with Rtools (Windows only)
            env = os.environ.copy()
            if os.name == 'nt':
                # Use proper Windows paths with backslashes
                rtools_paths = [
                    r"Y:\_Software\R\rtools45\usr\bin",
                    r"Y:\_Software\R\rtools45\x86_64-w64-mingw32.static.posix\bin"
                ]
                current_path = env.get('PATH', '')
                # Prepend Rtools to PATH
                env['PATH'] = os.pathsep.join(rtools_paths + [current_path])
                logger.info(f"[SSN] Env PATH prepended with: {rtools_paths}")
            logger.info(f"[SSN] Running command: {' '.join(cmd)}")
            task.progress = 'Running SpiecEasi network construction...'
            
            # Execute R script
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd=str(R_SCRIPT_DIR),
                env=env
            )
            
            # Log output
            if result.stdout:
                logger.info(f"[SSN] R stdout (last 1000 chars): {result.stdout[-1000:]}")
            if result.stderr:
                logger.warning(f"[SSN] R stderr (last 500 chars): {result.stderr[-500:]}")
            
            if result.returncode == 0:
                # Check for output files
                attr_file = result_dir / 'attrtable_SSN.csv'
                metrics_file = result_dir / 'metrics.csv'
                
                if attr_file.exists():
                    task.status = 'completed'
                    task.result_dir = str(result_dir)
                    task.attr_file = str(attr_file)
                    task.metrics_file = str(metrics_file) if metrics_file.exists() else None
                    task.progress = 'Completed successfully'
                    
                    # Parse results for summary
                    try:
                        df = pd.read_csv(attr_file)
                        samples = df[~df['sample'].str.contains('_group', na=False)]
                        groups = df[df['sample'].str.contains('_group', na=False)]
                        task.sample_count = len(samples)
                        task.group_count = len(groups)
                    except Exception as e:
                        logger.warning(f"[SSN] Failed to parse results: {e}")
                    
                    logger.info(f"[SSN] Task {task_id} completed successfully")
                else:
                    task.status = 'failed'
                    task.error = 'R script completed but no output files generated'
                    logger.error(f"[SSN] No output files for task {task_id}")
            else:
                task.status = 'failed'
                # Try to read error from log file
                log_file = result_dir / 'ssn_debug.log'
                error_msg = result.stderr
                if log_file.exists():
                    try:
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            log_content = f.read()
                            # Get last 500 chars if log is too long
                            if len(log_content) > 1000:
                                log_tail = log_content[-1000:]
                            else:
                                log_tail = log_content
                            error_msg = f"{error_msg}\nLog tail:\n{log_tail}"
                    except Exception as e:
                        logger.error(f"Failed to read error log: {e}")
                
                task.error = error_msg or 'R script failed with no error message'
                logger.error(f"[SSN] R script failed for task {task_id}")
                
        except subprocess.TimeoutExpired:
            task.status = 'failed'
            task.error = 'Computation timed out (>1 hour)'
            logger.error(f"[SSN] Task {task_id} timed out")
        except Exception as e:
            task.status = 'failed'
            task.error = str(e)
            logger.error(f"[SSN] Task {task_id} failed with exception: {e}")
    
    def get_task_summary(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed summary for a completed task"""
        task = self.get_task(task_id)
        if not task or task.status != 'completed' or not task.attr_file:
            return None
        
        try:
            df = pd.read_csv(task.attr_file)
            samples_df = df[~df['sample'].str.contains('_group', na=False)]
            
            return {
                'total_samples': len(samples_df),
                'subgroups': samples_df['subgroup'].unique().tolist() if 'subgroup' in samples_df.columns else [],
                'avg_mean_degree': round(samples_df['mean_degree'].mean(), 4) if 'mean_degree' in samples_df.columns else None,
                'avg_edge_number': round(samples_df['edge_number'].mean(), 2) if 'edge_number' in samples_df.columns else None,
                'avg_node_number': round(samples_df['node_number'].mean(), 2) if 'node_number' in samples_df.columns else None
            }
        except Exception as e:
            logger.warning(f"[SSN] Failed to generate summary: {e}")
            return None
    
    def get_task_attributes(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get network attributes as structured data"""
        task = self.get_task(task_id)
        if not task or task.status != 'completed' or not task.attr_file:
            return None
        
        try:
            df = pd.read_csv(task.attr_file)
            samples_df = df[~df['sample'].str.contains('_group', na=False)]
            groups_df = df[df['sample'].str.contains('_group', na=False)]
            
            return {
                'samples': samples_df.to_dict(orient='records'),
                'groups': groups_df.to_dict(orient='records'),
                'columns': df.columns.tolist()
            }
        except Exception as e:
            logger.error(f"[SSN] Failed to get attributes: {e}")
            return None


# ============================================================
# Global Instance
# ============================================================
ssn_manager = SSNTaskManager()


# ============================================================
# Utility Functions
# ============================================================
def get_ssn_status() -> Dict[str, Any]:
    """Get SSN service status information"""
    r_available, r_info = ssn_manager.check_r_available()
    
    return {
        'service': 'SSN Analysis',
        'r_available': r_available,
        'r_info': r_info,
        'r_script_path': str(SSN_SCRIPT),
        'r_script_exists': SSN_SCRIPT.exists(),
        'rscript_path': RSCRIPT_PATH,
        'active_tasks': len(ssn_manager.tasks)
    }
