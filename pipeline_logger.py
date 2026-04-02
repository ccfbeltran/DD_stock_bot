"""
pipeline_logger.py -- Shared logging setup for run_full.py and run_test.py
=========================================================================

PURPOSE:
    This module provides a centralized, dual-channel logging system for the
    entire DD stock bot pipeline. It solves several problems that arise when
    running a long (30+ minute) pipeline on Windows:

    1. DUAL LOGGING: We need both a clean console experience (INFO level, no
       timestamps) for the user watching the run, AND a verbose debug log file
       that captures everything for post-mortem debugging. The stdlib logging
       module handles this via separate handlers with different log levels.

    2. PRINT() CAPTURE: Many pipeline modules use print() instead of logging.
       The TeeWriter class intercepts sys.stdout/sys.stderr so that print()
       statements also appear in the log file -- without modifying every module.

    3. WINDOWS ENCODING: Windows consoles default to cp1252 (or similar),
       which crashes on Unicode characters (stock symbols, company names with
       accents, etc.). The SafeStream wrapper catches UnicodeEncodeError and
       replaces un-encodable characters instead of crashing the whole pipeline.

    4. RUN ISOLATION: Each pipeline run gets its own timestamped directory
       under logs/ (e.g., logs/full_2025-01-15_143022/). This directory holds
       the main pipeline.log plus per-stock Dexter logs and checkpoint data.
       The CURRENT_RUN_DIR module-level variable lets other modules (like
       dexter_runner.py) save their own files into the same run folder.

ARCHITECTURE:
    setup_logging() is called once at the start of main() in run_full.py or
    run_test.py. It:
      - Creates the timestamped run directory
      - Configures the root logger with file + console handlers
      - Replaces sys.stdout and sys.stderr with TeeWriter instances
      - Logs environment info (Python version, API keys present, etc.)

    Other modules import helper functions (log_step, log_error, log_dict,
    get_run_dir) without needing to know the logging configuration details.

FILE LAYOUT:
    logs/
      full_2025-01-15_143022/       <-- one directory per pipeline run
        pipeline.log                <-- main log file (DEBUG level, everything)
        dexter_NVDA.log             <-- per-stock Dexter subprocess stderr
        dexter_AAPL.log
        summary.json                <-- run summary written at completion
      test_2025-01-15_150000/
        pipeline.log
        ...
"""

import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── Module-level path constants ────────────────────────────────────────────
# REPO_ROOT points to the DD_stock_bot directory (where this file lives).
# All other paths are derived from it so the project is relocatable.
REPO_ROOT = Path(__file__).resolve().parent
LOGS_DIR = REPO_ROOT / "logs"

# CURRENT_RUN_DIR is set by setup_logging() and read by other modules via
# get_run_dir(). It points to the timestamped directory for THIS pipeline run.
# It starts as None before setup_logging() is called.
CURRENT_RUN_DIR: Path | None = None


def setup_logging(mode: str = "full") -> Path:
    """Set up dual logging: file (DEBUG) + console (INFO).

    Creates a timestamped run directory under DD_stock_bot/logs/ and configures
    the Python logging system with two handlers:

      - File handler (DEBUG level): Captures EVERYTHING -- debug messages,
        tracebacks, timing data, environment info. This is the forensic record
        for debugging failed runs.

      - Console handler (INFO level): Shows only user-facing progress messages.
        Uses a minimal format (no timestamps) to keep the terminal clean.

    Also installs TeeWriter on sys.stdout/sys.stderr so that print() calls
    from any module are captured in the log file too.

    Args:
        mode: Either "full" or "test". Used in the directory name
              (e.g., "full_2025-01-15_143022" vs "test_2025-01-15_143022")
              so you can tell which type of run produced each log folder.

    Returns:
        Path to the pipeline.log file inside the new run directory.
    """
    # Store the run directory in the module-level variable so other modules
    # (like dexter_runner.py) can access it via get_run_dir().
    global CURRENT_RUN_DIR
    LOGS_DIR.mkdir(exist_ok=True)

    # Timestamp format: YYYY-MM-DD_HHMMSS -- filesystem-safe, sortable
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = LOGS_DIR / f"{mode}_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    CURRENT_RUN_DIR = run_dir
    log_file = run_dir / "pipeline.log"

    # ── Configure the root logger ──────────────────────────────────────
    # We use the root logger so that ALL logging calls from ANY module
    # (including third-party libraries) are captured in the file handler.
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Accept all levels; handlers filter further

    # Clear any existing handlers to avoid duplicate output if setup_logging()
    # is called more than once (e.g., in tests or interactive use).
    root.handlers.clear()

    # ── File handler -- captures EVERYTHING (DEBUG+) ───────────────────
    # encoding="utf-8" is critical on Windows where the default file encoding
    # may be cp1252, which cannot represent all Unicode characters that appear
    # in stock data (company names, special symbols, etc.).
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(fh)

    # ── Console handler -- clean output (INFO+), no timestamp clutter ──
    # We wrap the real stdout in SafeStream to handle Windows encoding issues.

    class SafeStream:
        """Wrapper around a text stream that replaces un-encodable characters
        instead of raising UnicodeEncodeError.

        WHY THIS EXISTS:
            On Windows, sys.stdout is typically encoded as cp1252. If any log
            message contains characters outside cp1252 (e.g., Unicode arrows,
            accented company names, or special symbols), Python raises
            UnicodeEncodeError and the entire pipeline crashes.

            This wrapper catches that error and replaces the problematic
            characters with '?' (or the encoding's replacement character),
            so the pipeline continues running even with unusual characters.

        NOTE:
            We wrap sys.__stdout__ (the original stdout saved by Python at
            startup), NOT sys.stdout, because sys.stdout gets replaced by
            TeeWriter later. Using __stdout__ ensures the console handler
            always writes to the real terminal.
        """
        def __init__(self, stream):
            self.stream = stream
            # Preserve the stream's encoding for error-safe re-encoding
            self.encoding = getattr(stream, 'encoding', 'utf-8')

        def write(self, msg):
            """Write msg to the underlying stream, replacing un-encodable chars."""
            try:
                self.stream.write(msg)
            except UnicodeEncodeError:
                # Encode to the stream's native encoding with 'replace' error
                # handling (replaces bad chars with '?'), then decode back to
                # a string that the stream CAN write without error.
                self.stream.write(msg.encode(self.encoding, errors='replace').decode(self.encoding))

        def flush(self):
            """Flush the underlying stream (required by logging handler interface)."""
            self.stream.flush()

    # sys.__stdout__ is Python's saved reference to the original stdout,
    # which remains valid even after we replace sys.stdout with TeeWriter.
    ch = logging.StreamHandler(SafeStream(sys.__stdout__))
    ch.setLevel(logging.INFO)
    # Minimal format: no timestamp, no level -- just the message.
    # The user sees clean progress output; the file has the full details.
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    # ── Tee stdout/stderr to the log file ──────────────────────────────
    # Many pipeline modules use print() instead of logging.info(). Without
    # TeeWriter, those messages would appear on the console but NOT in the
    # log file. TeeWriter intercepts print() by replacing sys.stdout and
    # sys.stderr, writing each message to BOTH the original stream AND the
    # log file handler.

    class TeeWriter:
        """Intercepts writes to stdout/stderr and duplicates them to the log file.

        HOW IT WORKS:
            When Python executes print("hello"), it calls sys.stdout.write("hello").
            Because we replace sys.stdout with a TeeWriter instance, the write()
            method here runs instead. It:
              1. Writes to the original stream (so the user sees it on console)
              2. Creates a LogRecord and emits it through the file handler
                 (so it appears in pipeline.log)

        WHY WE NEED THIS:
            The pipeline spans multiple modules (fetch, filter, transcribe, etc.)
            that use print() extensively. Refactoring all of them to use logging
            would be a large change. TeeWriter captures their output without
            modifying any of those modules.

        ATTRIBUTES FORWARDED:
            encoding, errors, fileno(), isatty(), reconfigure() are all forwarded
            to the original stream because some libraries (tqdm, subprocess, etc.)
            check these attributes to decide how to format their output.
        """
        def __init__(self, original, log_handler):
            """Initialize the TeeWriter.

            Args:
                original:    The original stream (sys.__stdout__ or sys.__stderr__)
                             that should still receive all writes.
                log_handler: The file handler (logging.FileHandler) that should
                             also receive copies of all writes.
            """
            self.original = original
            self.log_handler = log_handler
            # Forward stream attributes that libraries like tqdm inspect
            # to determine terminal capabilities and encoding.
            self.encoding = getattr(original, 'encoding', 'utf-8')
            self.errors = getattr(original, 'errors', 'strict')

        def write(self, msg):
            """Write to both the original stream and the log file.

            Args:
                msg: The string to write. Empty/whitespace-only messages are
                     written to the original stream (to preserve formatting
                     like blank lines) but NOT to the log file (to avoid
                     cluttering it with empty log records).
            """
            self.original.write(msg)
            # Only log non-empty messages to avoid blank lines in the log file.
            # The .strip() check filters out the newline that print() appends.
            if msg.strip():
                self.log_handler.emit(logging.LogRecord(
                    name="stdout", level=logging.INFO,
                    pathname="", lineno=0, msg=msg.rstrip(),
                    args=None, exc_info=None,
                ))

        def flush(self):
            """Flush the original stream. Required for real-time output."""
            self.original.flush()

        def fileno(self):
            """Return the file descriptor of the original stream.

            Some libraries (subprocess, os.dup2) need the raw file descriptor.
            We delegate to the original stream so those libraries work correctly.
            """
            return self.original.fileno()

        def isatty(self):
            """Check if the original stream is a terminal.

            Libraries like tqdm use this to decide whether to show progress bars
            with ANSI escape codes. We delegate to the original stream.
            """
            return self.original.isatty()

        def reconfigure(self, **kwargs):
            """Reconfigure the original stream's encoding or error handling.

            Python 3.7+ allows reconfiguring sys.stdout (e.g., to change encoding).
            We forward this to the original stream and update our cached encoding
            so SafeStream (used by the console handler) stays consistent.

            Args:
                **kwargs: Passed directly to the original stream's reconfigure().
            """
            if hasattr(self.original, 'reconfigure'):
                self.original.reconfigure(**kwargs)
                # Keep our encoding attribute in sync after reconfiguration
                self.encoding = getattr(self.original, 'encoding', 'utf-8')

    # Replace sys.stdout and sys.stderr with TeeWriter instances.
    # sys.__stdout__ and sys.__stderr__ are Python's saved originals -- they
    # remain valid even after this replacement, so the TeeWriter can always
    # write to the real terminal.
    sys.stdout = TeeWriter(sys.__stdout__, fh)
    sys.stderr = TeeWriter(sys.__stderr__, fh)

    # ── Write log header ───────────────────────────────────────────────
    # The header provides essential context for debugging failed runs:
    # who ran it, when, what Python version, what environment variables
    # are set (without revealing actual secret values).
    logging.info("=" * 70)
    logging.info(f"LOG FILE: {log_file}")
    logging.info(f"STARTED:  {datetime.now().isoformat()}")
    logging.info(f"MODE:     {mode}")
    # DEBUG-level details: only visible in the log file, not on console
    logging.debug(f"PYTHON:   {sys.executable} ({sys.version.split()[0]})")
    logging.debug(f"CWD:      {os.getcwd()}")
    logging.debug(f"PLATFORM: {sys.platform}")
    logging.debug(f"ARGV:     {sys.argv}")
    # Log whether API keys are set (not their values!) for debugging
    # "my pipeline didn't call the API" issues.
    logging.debug(f"ENV DEEPSEEK_API_KEY: {'SET' if os.environ.get('DEEPSEEK_API_KEY') else 'NOT SET'}")
    logging.debug(f"ENV BUN_PATH: {os.environ.get('BUN_PATH', 'not set')}")
    logging.debug(f"REPO_ROOT: {REPO_ROOT}")
    logging.info("=" * 70)

    return log_file


def get_run_dir() -> Path | None:
    """Get the current run's log directory.

    Other modules (especially dexter_runner.py) use this to save per-stock
    log files and other artifacts into the same timestamped run folder.
    This keeps all outputs from a single run grouped together.

    Returns:
        Path to the current run directory (e.g., logs/full_2025-01-15_143022/),
        or None if setup_logging() has not been called yet.
    """
    return CURRENT_RUN_DIR


def log_step(step: str, detail: str = ""):
    """Log a major pipeline step with a visual separator.

    Used to mark transitions between pipeline phases (e.g., "YOUTUBE PIPELINE"
    or "DEXTER PIPELINE"). The separator lines make it easy to scan the log
    file and find where each phase started.

    Args:
        step:   Short label for the step (e.g., "YOUTUBE PIPELINE").
        detail: Optional longer description shown below the step label.
    """
    logging.info(f"\n{'-' * 60}")
    logging.info(f"  {step}")
    if detail:
        logging.info(f"  {detail}")
    logging.info(f"{'-' * 60}")
    # Timestamp at DEBUG level -- not shown on console, but useful in the
    # log file for measuring how long each step took.
    logging.debug(f"Step timestamp: {datetime.now().isoformat()}")


def log_error(context: str, error: Exception):
    """Log an error with full traceback to the log file.

    The error message is logged at ERROR level (visible on console), while
    the full traceback is logged at DEBUG level (only in the log file).
    This keeps the console clean while preserving full debugging info.

    Args:
        context: Description of where the error occurred
                 (e.g., "Step 4: Transcription" or "Dexter NVDA").
        error:   The caught exception object.
    """
    logging.error(f"ERROR in {context}: {error}")
    # Full traceback only in the log file (DEBUG level) to avoid
    # overwhelming the user with stack traces on the console.
    logging.debug(f"Full traceback:\n{traceback.format_exc()}")


def log_dict(label: str, data: dict, max_keys: int = 30):
    """Log a dictionary's contents at DEBUG level (log file only).

    Useful for inspecting configuration dicts, result dicts, etc. without
    cluttering the console. Handles large values gracefully:
      - Long strings are truncated to 100 chars
      - Nested dicts show key count instead of full contents
      - Lists show item count instead of full contents
      - After max_keys entries, remaining keys are summarized

    Args:
        label:    Human-readable label for the dict (e.g., "Stock config").
        data:     The dictionary to log.
        max_keys: Maximum number of keys to show before summarizing.
                  Prevents enormous dicts from filling up the log.
                  Defaults to 30.
    """
    logging.debug(f"{label}:")
    for i, (k, v) in enumerate(data.items()):
        # Stop after max_keys to avoid logging massive dicts
        if i >= max_keys:
            logging.debug(f"  ... ({len(data) - max_keys} more keys)")
            break
        # Truncate long string values to keep log lines readable
        if isinstance(v, str) and len(v) > 200:
            logging.debug(f"  {k}: ({len(v)} chars) {v[:100]}...")
        # Show nested dicts/lists as summaries, not full contents
        elif isinstance(v, dict):
            logging.debug(f"  {k}: {{...}} ({len(v)} keys)")
        elif isinstance(v, list):
            logging.debug(f"  {k}: [...] ({len(v)} items)")
        else:
            logging.debug(f"  {k}: {v}")
