#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging Utilities Module

Provides dual-output logging (console + file) and log collection for the pipeline.

Features:
  - TeeOutput: Write to both console and log file simultaneously
  - LogCollector: Collect all logs with timestamps for JSON export

Author: Pipeline Modularization Project
Date: 2024-12
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Optional


class TeeOutput:
    """
    Dual output stream that writes to multiple file objects simultaneously.

    Used to capture all console output to both stdout and a log file.
    Handles encoding errors gracefully.
    """

    def __init__(self, *files):
        """
        Initialize with multiple file objects.

        Args:
            *files: Variable number of file-like objects to write to
        """
        self.files = files

    def write(self, text):
        """Write text to all file objects"""
        for f in self.files:
            try:
                f.write(text)
                f.flush()
            except UnicodeEncodeError:
                # Handle encoding errors by replacing problematic characters
                safe_text = text.encode('utf-8', errors='replace').decode('utf-8')
                f.write(safe_text)
                f.flush()
            except Exception:
                pass  # Silently ignore other write errors

    def flush(self):
        """Flush all file objects"""
        for f in self.files:
            try:
                f.flush()
            except Exception:
                pass


class LogCollector:
    """
    Global log collector that captures all output with timestamps.

    Collects log messages and can save them to a JSON file for later analysis.
    """

    def __init__(self):
        """Initialize the log collector"""
        self.logs: List[Dict] = []
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.start_time = datetime.now()
        self._active = False

    def write(self, text):
        """Write to original stdout and collect log"""
        # Write to original output
        try:
            self.original_stdout.write(text)
            self.original_stdout.flush()
        except Exception:
            pass

        # Collect non-empty logs
        if text.strip():
            self.logs.append({
                "timestamp": datetime.now().isoformat(),
                "message": text.strip()
            })

    def flush(self):
        """Flush original stdout"""
        try:
            self.original_stdout.flush()
        except Exception:
            pass

    def activate(self):
        """Activate log collection by replacing sys.stdout"""
        if not self._active:
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            self.start_time = datetime.now()
            self._active = True
            # Note: Don't replace sys.stdout here to avoid recursion

    def deactivate(self):
        """Restore original stdout"""
        if self._active:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            self._active = False

    def get_logs(self) -> List[Dict]:
        """Get all collected logs"""
        return self.logs.copy()

    def clear_logs(self):
        """Clear all collected logs"""
        self.logs = []
        self.start_time = datetime.now()

    def save_logs(self, output_path: str, filename: str = "complete_run_log.json") -> str:
        """
        Save all logs to a JSON file.

        Args:
            output_path: Directory to save the log file
            filename: Name of the log file

        Returns:
            str: Path to the saved log file
        """
        os.makedirs(output_path, exist_ok=True)
        log_file = os.path.join(output_path, filename)

        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "run_info": {
                        "start_time": self.start_time.isoformat() if self.logs else datetime.now().isoformat(),
                        "end_time": datetime.now().isoformat(),
                        "total_messages": len(self.logs),
                        "duration_seconds": (datetime.now() - self.start_time).total_seconds()
                    },
                    "logs": self.logs
                }, f, indent=2, ensure_ascii=False)

            print(f"\n[Log] Complete run log saved to: {log_file}")
            return log_file

        except Exception as e:
            print(f"[Log] Error saving logs: {e}")
            return ""


class PipelineLogger:
    """
    Pipeline-specific logger with dual output and collection.

    Usage:
        logger = PipelineLogger(output_dir="outputs")
        logger.start()

        # ... run pipeline ...

        logger.stop()
        logger.save()
    """

    def __init__(self, output_dir: str = "outputs", log_filename: str = "debug_log.txt"):
        """
        Initialize pipeline logger.

        Args:
            output_dir: Directory for log files
            log_filename: Name of the debug log file
        """
        self.output_dir = output_dir
        self.log_filename = log_filename
        self.log_file = None
        self.tee_output = None
        self.collector = LogCollector()
        self.original_stdout = None
        self.original_stderr = None
        self._active = False

    def start(self):
        """Start logging - redirect stdout to both console and file"""
        if self._active:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        log_path = os.path.join(self.output_dir, self.log_filename)

        try:
            self.log_file = open(log_path, 'w', encoding='utf-8')
            self.log_file.write(f"Pipeline started at: {datetime.now().isoformat()}\n")
            self.log_file.write("=" * 60 + "\n\n")
            self.log_file.flush()

            # Save original streams
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr

            # Create TeeOutput for dual logging
            self.tee_output = TeeOutput(sys.__stdout__, self.log_file)

            # Replace stdout
            sys.stdout = self.tee_output

            # Activate collector
            self.collector.activate()
            self.collector.original_stdout = self.tee_output

            self._active = True
            print(f"[Log] Logging started. Debug log: {log_path}")

        except Exception as e:
            print(f"[Log] Error starting logger: {e}")
            self._cleanup()

    def stop(self):
        """Stop logging and restore stdout"""
        if not self._active:
            return

        try:
            # Write footer
            if self.log_file:
                self.log_file.write("\n" + "=" * 60 + "\n")
                self.log_file.write(f"Pipeline ended at: {datetime.now().isoformat()}\n")
                self.log_file.flush()
        except Exception:
            pass

        # Restore stdout
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr

        # Deactivate collector
        self.collector.deactivate()

        self._cleanup()
        self._active = False

    def _cleanup(self):
        """Clean up resources"""
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass
            self.log_file = None

    def save(self, filename: str = "complete_run_log.json") -> str:
        """Save collected logs to JSON"""
        return self.collector.save_logs(self.output_dir, filename)

    def log(self, message: str, level: str = "INFO"):
        """
        Log a message with timestamp and level.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR, DEBUG)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        print(formatted)

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
        return False


# Global logger instance
_global_logger: Optional[PipelineLogger] = None


def get_logger(output_dir: str = "outputs") -> PipelineLogger:
    """Get or create global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = PipelineLogger(output_dir)
    return _global_logger


def setup_logging(output_dir: str = "outputs") -> PipelineLogger:
    """Setup and start global logging"""
    logger = get_logger(output_dir)
    logger.start()
    return logger


# Module test
if __name__ == "__main__":
    print("Testing logging utilities...")

    # Test with context manager
    with PipelineLogger(output_dir="outputs") as logger:
        print("This message goes to both console and log file")
        logger.log("This is an INFO message")
        logger.log("This is a WARNING", level="WARNING")
        print("Testing complete!")

    print("\nLogger test finished. Check outputs/debug_log.txt")
