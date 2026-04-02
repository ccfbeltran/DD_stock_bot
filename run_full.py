"""
Full Pipeline -- YouTube Signals -> Dexter Deep DD -> Investment Report
======================================================================

PURPOSE:
    This is the main entry point for the complete stock due diligence pipeline.
    It orchestrates a 12-step process that transforms raw YouTube video data
    into a professional PDF investment report with AI-generated verdicts.

ARCHITECTURE OVERVIEW:
    The pipeline has two major phases:

    Phase 1 -- YouTube Signal Pipeline (Steps 1-7):
        Runs inside the Stock_Due_Diligence_bot submodule. Fetches videos from
        43+ financial YouTube channels, filters out non-investment content,
        transcribes audio with Whisper, summarizes into investment memos using
        DeepSeek, extracts per-ticker insights, and ranks tickers by a
        multi-factor scoring system (mentions, sentiment, creator diversity).

    Phase 2 -- Dexter Deep DD Pipeline (Steps 8-12):
        Runs inside the dexter-duedeligencebot submodule. Takes the top N
        ranked tickers, runs a headless AI agent (Dexter) that calls 15+
        financial data APIs per stock, then synthesizes a deep due diligence
        report. Extracts verdicts (STRONG BUY to STRONG SELL), generates an
        executive summary via LLM, and builds a PDF + JSON report.

    Both phases use a checkpoint system so that crashes/restarts don't lose
    progress. Cost tracking estimates LLM token costs and data API costs.

MODULE LOADING STRATEGY:
    The two submodules (Stock_Due_Diligence_bot and dexter-duedeligencebot)
    both have a src/ directory. If we put both on sys.path, Python would
    confuse "from src.X import Y" between the two. To solve this:

    - YouTube modules: We temporarily add Stock_Due_Diligence_bot to sys.path,
      import what we need, then remove it from sys.path afterward.

    - Dexter modules: We use importlib.util.spec_from_file_location() to load
      each module by its absolute file path, avoiding sys.path entirely.
      This is the _load_dexter_module() helper function.

ENCODING NOTE (WINDOWS):
    All file reads/writes use encoding="utf-8" explicitly. On Windows, Python
    defaults to the system locale encoding (usually cp1252), which cannot
    represent many Unicode characters found in stock data (company names,
    special symbols, financial notation). Without explicit UTF-8, the pipeline
    would crash with UnicodeDecodeError/UnicodeEncodeError on Windows.

Usage:
    python run_full.py
    python run_full.py --top-n 15
    python run_full.py --skip-youtube        # Skip steps 1-7, use existing signals
    python run_full.py --skip-dexter         # Only run YouTube pipeline, no DD
    python run_full.py --retry-errors        # Re-run only failed Dexter stocks
    python run_full.py --keep-last 5         # Keep only 5 most recent reports
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from dotenv import load_dotenv
from pipeline_logger import setup_logging, log_step, log_error, log_dict, get_run_dir

# ── Resolve paths from sync_config.json ──────────────────────────────────────
# sync_config.json is the single source of truth for where the two submodules
# live and where their outputs go. This avoids hardcoded paths scattered
# throughout the codebase.
REPO_ROOT = Path(__file__).resolve().parent

# Load .env file for API keys (DEEPSEEK_API_KEY, FINANCIAL_DATASETS_API_KEY, etc.)
load_dotenv(REPO_ROOT / ".env")

SYNC_CONFIG = REPO_ROOT / "sync_config.json"

if not SYNC_CONFIG.exists():
    sys.exit("sync_config.json not found in repo root.")

# Read sync_config.json with explicit UTF-8 encoding (Windows safety).
# This file maps logical names to relative paths within the repo.
_sync = json.loads(SYNC_CONFIG.read_text(encoding="utf-8"))

# Path to the YouTube signal pipeline submodule
STOCK_DD_ROOT = REPO_ROOT / "Stock_Due_Diligence_bot"

# Path to the YouTube pipeline's output data (ticker_scores.json, insights.json, etc.)
STOCK_DATA_DIR = REPO_ROOT / _sync["stock_dd_output"]

# Path to the Dexter deep DD submodule
DEXTER_ROOT = REPO_ROOT / _sync["dexter_root"]

# Path where Dexter writes its output (checkpoint.json, per-stock results)
OUTPUT_DIR = DEXTER_ROOT / "report_generator" / "output"

# Path to the Bun JavaScript runtime (used to run dexter_analyze.ts).
# Checks environment variable first, then falls back to sync_config.json,
# then falls back to "bun" (assumes it is on PATH).
BUN_PATH = Path(os.environ.get("BUN_PATH", _sync.get("bun_path", "bun")))

# The main TypeScript entry point for the Dexter headless agent
DEXTER_SCRIPT = DEXTER_ROOT / "report_generator" / "dexter_analyze.ts"

# Final report output directory (in the parent repo, not inside a submodule)
REPORTS_DIR = REPO_ROOT / "reports"

# Directory containing Dexter's Python helper modules (stock_loader, dexter_runner, etc.)
DEXTER_REPORT_SRC = DEXTER_ROOT / "report_generator" / "src"


def _load_dexter_module(name: str):
    """Load a Python module from Dexter's report_generator/src/ by absolute file path.

    WHY WE USE importlib INSTEAD OF NORMAL IMPORTS:
        Both Stock_Due_Diligence_bot and dexter-duedeligencebot have a src/
        directory. If we added both to sys.path, Python's import system would
        get confused -- "from src.insights import X" could resolve to either
        submodule's src/ directory depending on sys.path order.

        By using importlib.util.spec_from_file_location(), we load each module
        by its exact file path, completely bypassing sys.path. This guarantees
        we get the right module even when both submodules are present.

    The module is loaded with a "dexter_" prefix in its internal name to avoid
    collisions with any same-named modules that might be on sys.path.

    Args:
        name: Module filename without .py extension (e.g., "dexter_runner",
              "stock_loader", "verdict_extractor").

    Returns:
        The loaded module object, ready to use (e.g., mod.run_dexter(...)).
    """
    spec = importlib.util.spec_from_file_location(
        f"dexter_{name}", DEXTER_REPORT_SRC / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Cost tracking ───────────────────────────────────────────────────────────
# These constants reflect DeepSeek and Financial Datasets API pricing as of
# the time this pipeline was written. They are used to estimate the dollar
# cost of each pipeline run.
#
# DeepSeek pricing (deepseek-chat model):
#   - Input tokens:  $0.27 per million tokens
#   - Output tokens: $1.10 per million tokens
#
# Financial Datasets API pricing:
#   - ~$0.29 per stock lookup (income statements, balance sheets, etc.)
#   - Free for NVDA, MSFT, AAPL (the "free tier" tickers)
COST_PER_1M_INPUT = 0.27
COST_PER_1M_OUTPUT = 1.10
COST_PER_STOCK_API = 0.29


def estimate_cost(results: dict, fresh_tickers: set[str] | None = None, free_tickers: set[str] | None = None) -> dict:
    """Estimate the dollar cost of a Dexter pipeline run.

    This function calculates approximate costs based on token counts reported
    by the Dexter subprocess and the number of Financial Datasets API calls.

    FRESH vs CACHED TICKERS:
        The checkpoint system means some stocks were analyzed in a previous run
        and their results were loaded from disk (cached). We should NOT count
        those cached results toward this run's cost, because no API calls or
        LLM tokens were consumed for them. The fresh_tickers parameter tells
        us which stocks were actually run this time.

    TOKEN SPLIT ASSUMPTION:
        DeepSeek reports total tokens but not the input/output breakdown.
        We assume an 80/20 split (80% input, 20% output) based on empirical
        observation -- the Dexter agent sends large prompts with financial data
        and receives shorter analysis responses.

    FREE TICKERS:
        Financial Datasets API provides free access for NVDA, MSFT, and AAPL.
        These are excluded from the API cost calculation.

    Args:
        results:       Dict of ticker -> Dexter result dict. Each result may
                       contain 'total_tokens', 'total_tools', 'error', and
                       'turn1_cached' fields.
        fresh_tickers: Set of ticker symbols that were actually run this time
                       (not loaded from checkpoint). If None, all results are
                       treated as fresh.
        free_tickers:  Set of ticker symbols that don't incur API costs.
                       Defaults to {"NVDA", "MSFT", "AAPL"}.

    Returns:
        Dict with cost breakdown:
          - total_tokens:  Total LLM tokens consumed (fresh stocks only)
          - input_tokens:  Estimated input tokens (80% of total)
          - output_tokens: Estimated output tokens (20% of total)
          - total_tools:   Total tool calls made by the Dexter agent
          - llm_cost:      Estimated LLM cost in USD
          - api_cost:      Estimated Financial Datasets API cost in USD
          - total_cost:    llm_cost + api_cost
          - paid_stocks:   Number of stocks that incurred API costs
          - has_api_key:   Whether FINANCIAL_DATASETS_API_KEY was set
          - fresh_stocks:  Count of stocks run this time
          - cached_stocks: Count of stocks loaded from checkpoint
    """
    free_tickers = free_tickers or {"NVDA", "MSFT", "AAPL"}
    fresh_tickers = fresh_tickers or set(results.keys())

    # Only count tokens/tools from stocks that were freshly run (not checkpointed).
    # This gives an accurate cost for THIS run, not cumulative across all runs.
    fresh_results = {t: r for t, r in results.items() if t in fresh_tickers}
    total_tokens = sum(r.get("total_tokens", 0) for r in fresh_results.values())
    total_tools = sum(r.get("total_tools", 0) for r in fresh_results.values())

    # Apply the 80/20 input/output split assumption
    input_tokens = int(total_tokens * 0.8)
    output_tokens = int(total_tokens * 0.2)

    # Calculate LLM cost using per-million-token pricing
    llm_cost = (input_tokens / 1_000_000 * COST_PER_1M_INPUT
                + output_tokens / 1_000_000 * COST_PER_1M_OUTPUT)

    # Calculate Financial Datasets API cost.
    # Only count stocks that:
    #   1. Were freshly run (in fresh_tickers)
    #   2. Are NOT free-tier tickers (NVDA, MSFT, AAPL)
    #   3. Did NOT error out (errored stocks may not have made API calls)
    #   4. Did NOT use cached Turn 1 data (turn1_cached means no new API calls)
    has_api_key = bool(os.environ.get("FINANCIAL_DATASETS_API_KEY"))
    if has_api_key:
        paid_stocks = sum(
            1 for t in fresh_tickers
            if t not in free_tickers
            and not results.get(t, {}).get("error")
            and not results.get(t, {}).get("turn1_cached")
        )
    else:
        # No API key means no API calls were made (Dexter skips data gathering)
        paid_stocks = 0
    api_cost = paid_stocks * COST_PER_STOCK_API

    return {
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tools": total_tools,
        "llm_cost": llm_cost,
        "api_cost": api_cost,
        "total_cost": llm_cost + api_cost,
        "paid_stocks": paid_stocks,
        "has_api_key": has_api_key,
        "fresh_stocks": len(fresh_tickers),
        "cached_stocks": len(results) - len(fresh_tickers),
    }


# ── Report cleanup ─────────────────────────────────────────────────────────

def cleanup_reports(keep_last: int) -> None:
    """Delete old report files, keeping only the N most recent.

    Reports accumulate over time in the reports/ directory. This function
    sorts them by modification time and removes the oldest ones, keeping
    the most recent 'keep_last' files of each type (PDF and JSON separately).

    Args:
        keep_last: Number of most recent reports to keep. Must be > 0.
                   PDFs and JSONs are cleaned up independently.
    """
    if not REPORTS_DIR.exists():
        return
    # Sort by modification time (newest first) so we can keep the first N
    pdfs = sorted(REPORTS_DIR.glob("*.pdf"), key=lambda f: f.stat().st_mtime, reverse=True)
    jsons = sorted(REPORTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    # Slice off everything after the first keep_last entries
    for old in pdfs[keep_last:]:
        old.unlink()
        print(f"  Removed old report: {old.name}")
    for old in jsons[keep_last:]:
        old.unlink()
        print(f"  Removed old report: {old.name}")


# ── Retry errors ────────────────────────────────────────────────────────────

def retry_failed_stocks(stock_list, results, bun_path, dexter_script, dexter_root, checkpoint_file, today, timeout, model):
    """Re-run the Dexter analysis only on stocks that previously failed.

    This is used with --retry-errors to recover from transient failures
    (API timeouts, rate limits, etc.) without re-running successful stocks.

    The function filters the stock list to only those with errors in their
    results, runs Dexter on each one, and saves the checkpoint after each
    stock completes (for crash safety).

    Args:
        stock_list:      Full list of stock dicts from stock_loader.
        results:         Dict of ticker -> result from the previous run.
        bun_path:        Path to the Bun executable.
        dexter_script:   Path to dexter_analyze.ts.
        dexter_root:     Path to the Dexter project root.
        checkpoint_file: Path to checkpoint.json for saving progress.
        today:           Date string (YYYY-MM-DD) for checkpoint metadata.
        timeout:         Max seconds per stock for the Dexter subprocess.
        model:           LLM model identifier (e.g., "deepseek-chat").

    Returns:
        Updated results dict with retried stocks' new results.
    """
    # Load Dexter runner module using importlib (avoids sys.path conflicts)
    _runner = _load_dexter_module("dexter_runner")
    run_dexter, save_checkpoint = _runner.run_dexter, _runner.save_checkpoint

    # Filter to only stocks that have an error in their current results
    failed = [s for s in stock_list if s["ticker"] in results and results[s["ticker"]].get("error")]
    if not failed:
        print("  No failed stocks to retry.")
        return results

    print(f"  Retrying {len(failed)} failed stocks...")
    for stock in tqdm(failed, desc="  Retrying", unit="stock"):
        ticker = stock["ticker"]
        result = run_dexter(stock, bun_path, dexter_script, dexter_root, timeout, model)
        results[ticker] = result
        # Save checkpoint after EVERY stock for crash safety
        save_checkpoint(results, checkpoint_file, today)
        # Show per-stock status inline with the tqdm progress bar
        status = "OK" if not result.get("error") else f"ERROR: {result['error'][:40]}"
        tqdm.write(f"    {ticker:<8} {status}")

    return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the full pipeline.

    Returns:
        Namespace with attributes:
          - top_n (int):        Number of top tickers for Dexter DD (default: 10)
          - timeout (int):      Dexter timeout per stock in seconds (default: 3600)
          - model (str):        LLM model identifier (default: "deepseek-chat")
          - skip_youtube (bool): Skip YouTube pipeline steps 1-7
          - skip_dexter (bool):  Skip Dexter DD steps 8-12
          - retry_errors (bool): Re-run only failed Dexter stocks
          - keep_last (int):     Keep only N most recent reports (0 = keep all)
    """
    parser = argparse.ArgumentParser(description="Full DD pipeline: YouTube -> Dexter -> Report")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top tickers for Dexter DD (default: 10)")
    parser.add_argument("--timeout", type=int, default=3600, help="Dexter timeout per stock in seconds (default: 3600)")
    parser.add_argument("--model", default="deepseek-chat", help="LLM model for Dexter (default: deepseek-chat)")
    parser.add_argument("--skip-youtube", action="store_true", help="Skip YouTube pipeline (steps 1-7), use existing signals")
    parser.add_argument("--skip-dexter", action="store_true", help="Skip Dexter DD (steps 8-12), only run YouTube pipeline")
    parser.add_argument("--retry-errors", action="store_true", help="Re-run only failed Dexter stocks from checkpoint")
    parser.add_argument("--keep-last", type=int, default=0, help="Keep only N most recent reports (0 = keep all)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# YouTube Signal Pipeline (Steps 1-7)
# ─────────────────────────────────────────────────────────────────────────────

def run_youtube_pipeline() -> None:
    """Run the full Stock DD Bot pipeline: fetch -> filter -> transcribe -> summarize -> insights -> aggregate -> rank.

    This function executes Steps 1-7 of the pipeline, all within the
    Stock_Due_Diligence_bot submodule. It temporarily changes the working
    directory and sys.path to that submodule because its internal imports
    assume they are running from the submodule root.

    WORKING DIRECTORY MANAGEMENT:
        The Stock_Due_Diligence_bot modules use relative paths (e.g.,
        "data/processed/videos.json") and expect to run from their own root.
        We save the original working directory, chdir to the submodule, and
        restore it in the finally block to avoid breaking Dexter's imports.

    SYS.PATH MANAGEMENT:
        We insert Stock_Due_Diligence_bot at the front of sys.path so its
        "from src.X import Y" and "from config import settings" imports work.
        After the YouTube pipeline finishes, we REMOVE it from sys.path to
        prevent import collisions when the Dexter pipeline runs next (both
        submodules have a src/ directory with different modules).

    TRANSCRIPTION ERROR HANDLING (Step 4):
        Transcription is run as a subprocess because it loads heavy models
        (Whisper) that can crash on cleanup. The exit code handling covers:
          - Code 0: Success
          - Code 3: Partial failure (some videos skipped, but others ok)
          - Other codes: Check if transcripts were saved before the crash.
            If yes, continue. If no, abort the pipeline.
    """
    # All pipeline modules expect to run from Stock_Due_Diligence_bot root.
    original_dir = os.getcwd()
    os.chdir(STOCK_DD_ROOT)
    sys.path.insert(0, str(STOCK_DD_ROOT))

    try:
        # ── Step 1: Load config ────────────────────────────────────────
        # Imports the settings module from the Stock DD Bot submodule.
        # This loads channel list, keywords, and lookback window.
        print("\n[Step 1/7] Loading configuration...")
        t0 = time.time()
        from config import settings as cfg
        print(f"  Channels: {len(cfg.CHANNEL_MAP)}  |  Keywords: {len(cfg.STOCK_KEYWORDS)}  |  Lookback: {cfg.DAYS_LOOKBACK}d")
        logging.debug(f"Step 1 completed in {time.time() - t0:.1f}s")
        logging.debug(f"Channel list: {list(cfg.CHANNEL_MAP.keys())}")

        # ── Step 2: Fetch videos ───────────────────────────────────────
        # Uses the YouTube Data API to fetch recent videos from all configured
        # channels within the lookback window.
        print("\n[Step 2/7] Fetching videos from YouTube channels...")
        t0 = time.time()
        from src.fetch import fetch_videos
        videos = fetch_videos()
        print(f"  Total: {len(videos)} videos fetched")
        logging.debug(f"Step 2 completed in {time.time() - t0:.1f}s -- {len(videos)} videos")

        # ── Step 3: Filter content ─────────────────────────────────────
        # Uses DeepSeek LLM to classify each video as investment-related
        # or not. Non-investment videos (lifestyle, vlogs, etc.) are removed.
        print("\n[Step 3/7] Filtering non-investment content...")
        t0 = time.time()
        from src.filter import filter_videos
        kept, removed = filter_videos()
        print(f"  Kept: {len(kept)}  |  Removed: {len(removed)}")
        logging.debug(f"Step 3 completed in {time.time() - t0:.1f}s -- kept={len(kept)}, removed={len(removed)}")

        # ── Step 4: Transcribe ─────────────────────────────────────────
        # Downloads audio from YouTube videos and transcribes them using
        # OpenAI Whisper. This is run as a SUBPROCESS because:
        #   1. Whisper loads large ML models that consume lots of memory
        #   2. Whisper sometimes crashes during model cleanup (atexit handlers)
        #   3. Running in a subprocess isolates those crashes from our pipeline
        #
        # The subprocess runs a one-liner that imports and calls transcribe_videos().
        print("\n[Step 4/7] Downloading audio & transcribing (Whisper)...")
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, "-c", "from src.transcribe import transcribe_videos; transcribe_videos()"],
            cwd=str(STOCK_DD_ROOT),
        )
        logging.debug(f"Step 4 subprocess exited with code {result.returncode} in {time.time() - t0:.1f}s")

        if result.returncode == 3:
            # Exit code 3 means partial failure: some videos couldn't be
            # transcribed (e.g., private, deleted, or too long), but others
            # were transcribed successfully. This is acceptable.
            print("  WARNING: Transcription had partial failures (some videos skipped)")
            logging.warning(f"Transcription exit code 3 -- partial failures")
        elif result.returncode != 0:
            # Non-zero exit code (other than 3): the subprocess crashed.
            # Check if transcripts were saved before the crash -- Whisper
            # sometimes crashes during model cleanup but AFTER saving results.
            logging.warning(f"Transcription exit code {result.returncode} -- checking for saved transcripts")
            videos_file = STOCK_DD_ROOT / "data" / "processed" / "videos.json"
            if videos_file.exists():
                vids = json.loads(videos_file.read_text(encoding="utf-8"))
                has_transcripts = any(v.get("transcription") or v.get("transcript") for v in vids)
                if has_transcripts:
                    # Crash happened after transcripts were saved -- safe to continue
                    print(f"  WARNING: Transcription process crashed on exit (code {result.returncode}), but transcripts were saved. Continuing...")
                    logging.warning(f"Transcription crashed on cleanup (code {result.returncode}) but transcripts saved -- continuing")
                else:
                    # Crash happened before any transcripts were saved -- must abort
                    logging.error(f"Transcription failed with code {result.returncode} -- no transcripts produced")
                    sys.exit(f"ERROR: Transcription failed with code {result.returncode} and no transcripts were produced.")
            else:
                # No videos.json at all -- something went very wrong
                logging.error(f"Transcription failed with code {result.returncode} -- no videos.json found")
                sys.exit(f"ERROR: Transcription failed with code {result.returncode}. Cannot continue.")

        # ── Step 5: Summarize ──────────────────────────────────────────
        # Uses DeepSeek to generate investment memos from transcripts.
        # Each transcript is condensed into a structured memo with key claims.
        print("\n[Step 5/7] Generating investment memos...")
        t0 = time.time()
        from src.summarize import summarize_videos
        summarize_videos()
        logging.debug(f"Step 5 completed in {time.time() - t0:.1f}s")

        # ── Step 6: Extract insights ───────────────────────────────────
        # Parses memos into structured per-ticker insights (sentiment,
        # confidence, theme, specific claims) using DeepSeek.
        print("\n[Step 6/7] Extracting structured insights...")
        t0 = time.time()
        from src.insights import extract_insights
        extract_insights()
        logging.debug(f"Step 6 completed in {time.time() - t0:.1f}s")

        # ── Step 7: Aggregate + Rank ───────────────────────────────────
        # Aggregates per-ticker insights across all videos, then ranks
        # tickers using a multi-factor scoring system that considers:
        #   - Number of mentions
        #   - Number of unique creators mentioning the ticker
        #   - Sentiment distribution (bull/bear/neutral)
        #   - Confidence levels of claims
        print("\n[Step 7/7] Aggregating & ranking tickers...")
        t0 = time.time()
        from src.aggregate import aggregate_insights
        aggregate_insights()

        from src.rank import rank_tickers
        ranking = rank_tickers()
        logging.debug(f"Step 7 completed in {time.time() - t0:.1f}s -- {len(ranking['rankings'])} tickers ranked")

        # Display the top 10 ranked tickers for quick visual confirmation
        print(f"\n  Top 10 ranked tickers:")
        for row in ranking["rankings"][:10]:
            print(
                f"    {row['ticker']:8s} | {row['label']:7s} | "
                f"score={row['rank_score']:.2f} | mentions={row['mentions']:>2} | "
                f"creators={row['unique_creators']:>2}"
            )
    finally:
        # Always restore the original working directory, even if an error occurred
        os.chdir(original_dir)
        # Remove Stock_Due_Diligence_bot from sys.path to prevent import collisions.
        # The Dexter pipeline has its own src/ directory, and leaving the YouTube
        # src/ on the path would cause "from src.X" to resolve to the wrong module.
        if str(STOCK_DD_ROOT) in sys.path:
            sys.path.remove(str(STOCK_DD_ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Dexter Deep DD + Report (Steps 8-12)
# ─────────────────────────────────────────────────────────────────────────────

def run_dexter_pipeline(top_n: int, timeout: int, model: str, retry_errors: bool, keep_last: int) -> None:
    """Run the Dexter deep due diligence pipeline and generate reports.

    This function executes Steps 8-12:
      Step 8:  Load the ranked stock signals from the YouTube pipeline output
      Step 9:  Run the Dexter headless agent on each stock (with checkpointing)
      Step 10: Extract investment verdicts from Dexter's analysis text
      Step 11: Generate an executive summary using DeepSeek LLM
      Step 12: Build PDF and JSON reports, copy to reports/ directory

    CHECKPOINT / RETRY FLOW:
        Before running Dexter, this function checks for existing checkpoint data.
        If --retry-errors is set, it clears the "_failed_date" flag on failed
        stocks so they will be re-run (normally, stocks that failed today are
        skipped to avoid wasting API calls on persistent failures).

    DATA FRESHNESS CHECK:
        The function reads pipeline_meta.json to determine when the YouTube
        signals were generated. If the data is more than 48 hours old, it
        prints a warning -- stale signals may lead to outdated DD reports.

    Args:
        top_n:        Number of top-ranked tickers to analyze with Dexter.
        timeout:      Maximum seconds to wait for each Dexter subprocess.
        model:        LLM model identifier (e.g., "deepseek-chat").
        retry_errors: If True, clear failed-today flags and re-run errored stocks.
        keep_last:    Number of most recent reports to keep (0 = keep all).
    """
    TODAY = datetime.now().strftime("%Y-%m-%d")
    CHECKPOINT = OUTPUT_DIR / "checkpoint.json"
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("ERROR: DEEPSEEK_API_KEY environment variable not set.")
    t_start = time.time()

    # Ensure the output directory exists (for checkpoint file and reports)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 8: Load signals ─────────────────────────────────────────────
    # Load the ranked stock data produced by the YouTube pipeline (Step 7).
    # This includes ticker scores, sentiment data, and top claims.
    print("\n[Step 8/12] Loading ranked stock signals...")
    logging.debug(f"Dexter pipeline config: top_n={top_n}, timeout={timeout}s, model={model}, retry={retry_errors}")
    t0 = time.time()

    # Load the stock_loader module from Dexter's src/ using importlib
    load_stocks = _load_dexter_module("stock_loader").load_stocks

    scores_path = STOCK_DATA_DIR / "ticker_scores.json"
    insights_path = STOCK_DATA_DIR / "insights.json"

    if not scores_path.exists():
        sys.exit(f"Missing ticker_scores.json at {scores_path}. Run YouTube pipeline first.")

    # load_stocks() merges ticker scores with detailed insights to create
    # enriched stock dicts with sentiment counts, top claims, etc.
    stock_list = load_stocks(
        ticker_scores_path=scores_path,
        insights_path=insights_path,
        top_n=top_n,
    )

    # Check how fresh the YouTube signal data is.
    # Stale data (>48h) may produce DD reports based on outdated claims.
    meta_path = STOCK_DATA_DIR / "pipeline_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        age_hours = (datetime.now() - datetime.fromisoformat(meta["completed_at"])).total_seconds() / 3600
        print(f"  Pipeline data: {meta['completed_at']} ({age_hours:.1f}h ago, {meta['ticker_count']} tickers)")
        if age_hours > 48:
            print("  WARNING: Stock data is more than 48 hours old!")
    else:
        print("  WARNING: No pipeline_meta.json -- data freshness unknown.")

    # Display the selected stocks for user confirmation
    print(f"  Selected top {len(stock_list)} stocks for deep DD:")
    for i, s in enumerate(stock_list, 1):
        print(
            f"    {i:<3} {s['ticker']:<8} {s['company'][:22]:<23} "
            f"{s['label']:<9} score={s['rank_score']:.2f}"
        )

    logging.debug(f"Step 8 completed in {time.time() - t0:.1f}s -- loaded {len(stock_list)} stocks")
    for s in stock_list:
        logging.debug(f"  Stock: {s['ticker']:<8} {s['company'][:30]:<31} label={s['label']} score={s['rank_score']:.2f} mentions={s.get('mentions',0)} creators={s.get('unique_creators',0)}")

    # ── Step 9: Run Dexter analysis ──────────────────────────────────────
    # Run the Dexter headless agent on each stock. Each stock spawns a
    # subprocess (bun run dexter_analyze.ts) that makes ~15 API calls
    # and produces a structured DD analysis.
    t0 = time.time()
    print(f"\n[Step 9/12] Running Dexter deep DD (model={model}, timeout={timeout}s)...")
    _runner = _load_dexter_module("dexter_runner")
    run_all_stocks = _runner.run_all_stocks

    # If retrying errors, clear the "_failed_date" field on failed stocks.
    # Normally, stocks that failed today are skipped on re-run to avoid
    # wasting API calls on persistent failures (e.g., unsupported tickers).
    # The --retry-errors flag overrides this by removing the date tag.
    if retry_errors:
        load_checkpoint = _runner.load_checkpoint
        save_checkpoint = _runner.save_checkpoint
        existing = load_checkpoint(CHECKPOINT)
        cleared = 0
        for ticker, result in existing.items():
            if result.get("error") and result.get("_failed_date"):
                del result["_failed_date"]
                cleared += 1
        if cleared:
            save_checkpoint(existing, CHECKPOINT, TODAY)
            print(f"  Cleared {cleared} failed-today flags for retry")

    # run_all_stocks handles checkpoint loading, per-stock execution, and
    # checkpoint saving. It returns results for ALL stocks (checkpoint + fresh).
    results = run_all_stocks(
        stock_list=stock_list,
        bun_path=BUN_PATH,
        dexter_script=DEXTER_SCRIPT,
        dexter_root=DEXTER_ROOT,
        checkpoint_file=CHECKPOINT,
        today=TODAY,
        timeout=timeout,
        model=model,
    )

    logging.debug(f"Step 9 completed in {time.time() - t0:.1f}s")
    # Log per-stock result summaries at DEBUG level for the log file
    for ticker in [s["ticker"] for s in stock_list]:
        r = results.get(ticker, {})
        ans_len = len(r.get("answer", ""))
        tools = r.get("total_tools", 0)
        time_ms = r.get("total_time_ms", 0)
        err = r.get("error", "")
        logging.debug(f"  {ticker:<8} answer={ans_len:>5} chars, tools={tools:>2}, time={time_ms/1000:.0f}s{f', ERROR: {err}' if err else ''}")

    # ── Step 10: Extract verdicts ────────────────────────────────────────
    # Parse each stock's Dexter analysis to extract a structured verdict
    # (STRONG BUY, BUY, HOLD, SELL, STRONG SELL) plus supporting data.
    t0 = time.time()
    print("\n[Step 10/12] Extracting verdicts...")
    build_verdicts = _load_dexter_module("verdict_extractor").build_verdicts

    analyses, verdicts, verdict_counts = build_verdicts(
        stock_list, results
    )
    # Display the verdict distribution as a visual histogram
    print("  VERDICT DISTRIBUTION:")
    for vn in ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]:
        c = verdict_counts.get(vn, 0)
        if c:
            print(f"    {vn:<12} {c:>2}  {'#' * c}")
    logging.debug(f"Step 10 completed in {time.time() - t0:.1f}s")
    for ticker in [s["ticker"] for s in stock_list]:
        v = verdicts.get(ticker, {})
        logging.debug(f"  {ticker:<8} verdict={v.get('verdict', 'N/A')}")

    # ── Step 11: Executive summary ───────────────────────────────────────
    # Uses DeepSeek to generate a portfolio-level executive summary that
    # synthesizes all individual stock verdicts into investment themes.
    t0 = time.time()
    print("\n[Step 11/12] Generating executive summary...")
    generate_executive_summary = _load_dexter_module("executive_summary").generate_executive_summary

    executive_summary = generate_executive_summary(
        stock_list=stock_list,
        verdicts=verdicts,
        analyses=analyses,
        today=TODAY,
        api_key=api_key,
    )
    print("  Done.")
    logging.debug(f"Step 11 completed in {time.time() - t0:.1f}s -- summary length: {len(executive_summary)} chars")

    # ── Step 12: Build report ────────────────────────────────────────────
    # Generate the final PDF and JSON reports from all the collected data.
    t0 = time.time()
    print("\n[Step 12/12] Building PDF + JSON report...")
    build_report = _load_dexter_module("pdf_report").build_report
    save_json_report = _load_dexter_module("json_report").save_json_report

    pdf_path = build_report(
        stock_list=stock_list,
        verdicts=verdicts,
        analyses=analyses,
        executive_summary=executive_summary,
        verdict_counts=verdict_counts,
        today=TODAY,
        output_dir=OUTPUT_DIR,
    )

    json_path = save_json_report(
        stock_list=stock_list,
        verdicts=verdicts,
        analyses=analyses,
        results=results,
        executive_summary=executive_summary,
        verdict_counts=verdict_counts,
        today=TODAY,
        output_dir=OUTPUT_DIR,
    )
    logging.debug(f"Step 12 completed in {time.time() - t0:.1f}s -- PDF: {pdf_path}, JSON: {json_path}")

    # ── Copy reports to parent repo reports/ folder ──────────────────────
    # Reports are first generated inside the Dexter submodule's output/ dir,
    # then copied to the parent repo's reports/ dir for easy access.
    REPORTS_DIR.mkdir(exist_ok=True)
    final_pdf = REPORTS_DIR / Path(pdf_path).name
    final_json = REPORTS_DIR / Path(json_path).name
    shutil.copy2(pdf_path, final_pdf)
    shutil.copy2(json_path, final_json)

    # ── Cleanup old reports ──────────────────────────────────────────────
    if keep_last > 0:
        print(f"\n  Cleaning up reports (keeping last {keep_last})...")
        cleanup_reports(keep_last)

    # ── Cost summary ────────────────────────────────────────────────────
    # Extract the _fresh_tickers tracking field that run_all_stocks() added.
    # This field is a list of tickers that were actually run this time (not
    # loaded from checkpoint). We pop it from results because it is internal
    # metadata, not a stock result.
    fresh = set(results.pop("_fresh_tickers", []))
    cost = estimate_cost(results, fresh_tickers=fresh)
    elapsed = time.time() - t_start

    # Display the final summary with report paths, timing, and cost breakdown
    print()
    print("=" * 60)
    print("  FULL REPORT COMPLETE")
    print(f"  PDF:  {final_pdf}")
    print(f"  JSON: {final_json}")
    print(f"  Chkp: {CHECKPOINT}")
    print(f"  Time: {elapsed / 60:.1f} min")
    print(f"  {'-' * 40}")
    print(f"  COST SUMMARY (this run only -- {cost['fresh_stocks']} fresh, {cost['cached_stocks']} cached)")
    print(f"    Tokens:     {cost['total_tokens']:,} ({cost['input_tokens']:,} in / {cost['output_tokens']:,} out)")
    print(f"    Tools:      {cost['total_tools']}")
    print(f"    LLM:        ${cost['llm_cost']:.2f}")
    if cost['has_api_key']:
        print(f"    Data API:   ${cost['api_cost']:.2f} ({cost['paid_stocks']} paid stocks x ${COST_PER_STOCK_API})")
    else:
        print(f"    Data API:   $0.00 (no FINANCIAL_DATASETS_API_KEY set)")
    print(f"    TOTAL:      ${cost['total_cost']:.2f}")
    print("=" * 60)

    # ── Save run summary to the log folder ──────────────────────────────
    # Write a machine-readable summary of the run into the timestamped log
    # directory. This JSON file can be used for historical analysis of runs,
    # cost tracking over time, and debugging.
    run_dir = get_run_dir()
    if run_dir:
        summary = {
            "completed_at": datetime.now().isoformat(),
            "elapsed_min": round(elapsed / 60, 1),
            "mode": "full",
            "top_n": top_n,
            "model": model,
            "timeout": timeout,
            "pdf": str(final_pdf),
            "json": str(final_json),
            "stocks": {},
            "cost": cost,
        }
        # Include per-stock results in the summary for easy lookup
        for s in stock_list:
            t = s["ticker"]
            r = results.get(t, {})
            v = verdicts.get(t, {})
            summary["stocks"][t] = {
                "company": s["company"],
                "verdict": v.get("verdict", "N/A"),
                "answer_chars": len(r.get("answer", "")),
                "tools": r.get("total_tools", 0),
                "time_ms": r.get("total_time_ms", 0),
                "error": r.get("error", None),
            }
        # Write with explicit UTF-8 encoding and ensure_ascii=False so that
        # company names with non-ASCII characters are preserved correctly.
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the full pipeline.

    Parses command-line arguments, sets up logging, then runs the YouTube
    signal pipeline and/or the Dexter DD pipeline based on the flags provided.
    Either phase can be skipped independently with --skip-youtube or --skip-dexter.
    """
    args = parse_args()
    t_start = time.time()

    # Set up logging -- all output goes to both console and log file
    log_file = setup_logging(mode="full")

    logging.debug(f"Parsed args: {vars(args)}")

    # Display run configuration banner
    print("=" * 60)
    print("  FULL PIPELINE - YouTube Signals -> Deep DD -> Report")
    print("=" * 60)
    print(f"  Date:      {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Top N:     {args.top_n}")
    print(f"  Model:     {args.model}")
    print(f"  YouTube:   {'SKIP' if args.skip_youtube else 'RUN'}")
    print(f"  Dexter:    {'SKIP' if args.skip_dexter else 'RUN'}")
    print(f"  Timeout:   {args.timeout}s ({args.timeout // 60} min)")
    print(f"  Log:       {log_file}")
    if args.retry_errors:
        print(f"  Retry:     ON")
    if args.keep_last > 0:
        print(f"  Cleanup:   keep last {args.keep_last}")

    # ── Phase 1: YouTube Signal Pipeline (Steps 1-7) ───────────────────
    if not args.skip_youtube:
        log_step("YOUTUBE PIPELINE", "Steps 1-7: Fetch -> Filter -> Transcribe -> Summarize -> Insights -> Rank")
        run_youtube_pipeline()
    else:
        print("\n  Skipping YouTube pipeline (--skip-youtube)")
        logging.debug("YouTube pipeline skipped via --skip-youtube flag")

    # ── Phase 2: Dexter Deep DD Pipeline (Steps 8-12) ──────────────────
    if not args.skip_dexter:
        log_step("DEXTER PIPELINE", f"Steps 8-12: Load -> DD -> Verdicts -> Summary -> Report (top {args.top_n})")
        run_dexter_pipeline(
            top_n=args.top_n,
            timeout=args.timeout,
            model=args.model,
            retry_errors=args.retry_errors,
            keep_last=args.keep_last,
        )
    else:
        print("\n  Skipping Dexter DD (--skip-dexter)")
        print("  YouTube signals saved to:", STOCK_DATA_DIR)

    elapsed = time.time() - t_start
    print(f"\nDone. Total time: {elapsed / 60:.1f} min")

    # Final log summary with completion timestamp
    logging.info(f"\n{'=' * 70}")
    logging.info(f"RUN COMPLETED: {datetime.now().isoformat()}")
    logging.info(f"Total time: {elapsed / 60:.1f} min")
    logging.info(f"Log file: {log_file}")
    logging.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
