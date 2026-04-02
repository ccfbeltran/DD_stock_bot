"""
Test Pipeline -- Run Dexter on 3 Free Tickers (NVDA, MSFT, AAPL)
================================================================

PURPOSE:
    This is a lightweight test harness for the Dexter deep DD pipeline.
    It skips the entire YouTube signal pipeline (Steps 1-7) and instead
    uses pre-built test data with three "free tier" tickers: NVDA, MSFT,
    and AAPL. These tickers are free on the Financial Datasets API, so
    the only cost is LLM tokens for the Dexter analysis.

DIFFERENCES FROM run_full.py:
    1. NO YOUTUBE PIPELINE: Skips video fetching, filtering, transcription,
       summarization, insight extraction, and ranking entirely. Uses
       pre-built ticker_scores.json and insights.json from test_data/.

    2. HARDCODED PARAMETERS: Uses fixed values for timeout (3600s), model
       ("deepseek-chat"), and top_n (3). No --top-n, --timeout, or --model
       CLI flags because the test always runs exactly 3 stocks.

    3. SIMPLIFIED COST TRACKING: All tickers are free-tier, so API cost
       is always $0.00. The estimate_cost() function is simpler and does
       not need fresh_tickers tracking or free_tickers filtering.

    4. SEPARATE CHECKPOINT: Uses checkpoint_test.json (not checkpoint.json)
       so test runs don't interfere with production runs.

    5. DIRECT IMPORTS: Unlike run_full.py which uses importlib to avoid
       sys.path conflicts, run_test.py adds Dexter's report_generator/
       to sys.path directly and uses normal imports. This is safe because
       run_test.py never loads the YouTube pipeline, so there is no
       conflicting src/ directory on the path.

    6. FEWER CLI FLAGS: Only supports --retry-errors and --keep-last.
       No --skip-youtube, --skip-dexter, --top-n, --timeout, or --model.

TYPICAL USE CASES:
    - Verifying that Dexter, the LLM, and the report generator work correctly
      before running the full (expensive) pipeline.
    - Testing changes to the report generation code without waiting for the
      YouTube pipeline to complete.
    - Quick sanity checks after code changes.

Usage:
    python run_test.py
    python run_test.py --retry-errors       # Re-run only failed stocks
    python run_test.py --keep-last 5        # Keep only 5 most recent reports
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from dotenv import load_dotenv
from pipeline_logger import setup_logging, log_step, log_error

# ── Resolve paths from sync_config.json ──────────────────────────────────────
# Same path resolution pattern as run_full.py. The sync_config.json file
# provides the relative paths to the Dexter submodule and other directories.
REPO_ROOT = Path(__file__).resolve().parent

# Load .env file for API keys (DEEPSEEK_API_KEY is required)
load_dotenv(REPO_ROOT / ".env")

SYNC_CONFIG = REPO_ROOT / "sync_config.json"

if not SYNC_CONFIG.exists():
    sys.exit("sync_config.json not found in repo root.")

# encoding="utf-8" is critical on Windows to avoid cp1252 encoding errors
# when reading JSON files that may contain non-ASCII characters.
_sync = json.loads(SYNC_CONFIG.read_text(encoding="utf-8"))

# Path to the Dexter deep DD submodule
DEXTER_ROOT = REPO_ROOT / _sync["dexter_root"]

# Pre-built test data directory containing ticker_scores.json and insights.json
# for NVDA, MSFT, and AAPL. This data simulates the YouTube pipeline output.
TEST_DATA_DIR = DEXTER_ROOT / "report_generator" / "test_data"

# Directory where Dexter writes output (checkpoint, report files)
OUTPUT_DIR = DEXTER_ROOT / "report_generator" / "output"

# Path to the Bun JavaScript runtime (for running dexter_analyze.ts)
BUN_PATH = Path(os.environ.get("BUN_PATH", _sync.get("bun_path", "bun")))

# The TypeScript entry point for the Dexter headless agent
DEXTER_SCRIPT = DEXTER_ROOT / "report_generator" / "dexter_analyze.ts"

# Separate checkpoint file for test runs so they don't overwrite production data.
# Production uses checkpoint.json; test uses checkpoint_test.json.
CHECKPOINT = OUTPUT_DIR / "checkpoint_test.json"

# Fixed parameters for test mode (no CLI overrides needed)
TODAY = datetime.now().strftime("%Y-%m-%d")
TIMEOUT = 3600   # 1 hour timeout per stock (generous for test)
MODEL = "deepseek-chat"  # Default LLM model

# Final report output directory (same location as run_full.py)
REPORTS_DIR = REPO_ROOT / "reports"

# Ensure the output directory exists for checkpoint and report files
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add Dexter's report_generator/ to sys.path so we can import its modules
# directly (e.g., "from src.dexter_runner import run_dexter").
# NOTE: Unlike run_full.py, we can do this safely here because run_test.py
# never loads the YouTube pipeline, so there is no conflicting src/ directory.
sys.path.insert(0, str(DEXTER_ROOT / "report_generator"))

# ── Validate test data ──────────────────────────────────────────────────────
# Fail fast if the test data directory or required files are missing.
# This gives a clear error message instead of a confusing FileNotFoundError
# deep in the stock_loader module.
if not TEST_DATA_DIR.exists():
    sys.exit(f"Test data not found: {TEST_DATA_DIR}")
if not (TEST_DATA_DIR / "ticker_scores.json").exists():
    sys.exit(f"Missing ticker_scores.json in {TEST_DATA_DIR}")


# ── Cost tracking ───────────────────────────────────────────────────────────
# DeepSeek pricing constants (same as run_full.py).
# The key difference is COST_PER_STOCK_API = 0.00 because NVDA, MSFT, and AAPL
# are free-tier tickers on the Financial Datasets API.
COST_PER_1M_INPUT = 0.27   # $0.27 per million input tokens (deepseek-chat)
COST_PER_1M_OUTPUT = 1.10  # $1.10 per million output tokens (deepseek-chat)
COST_PER_STOCK_API = 0.00  # Free tickers in test mode -- no API cost


def estimate_cost(results: dict) -> dict:
    """Estimate the dollar cost of a test pipeline run.

    Simplified version of run_full.py's estimate_cost(). Since all test
    tickers are free-tier, we don't need fresh_tickers tracking or
    free_tickers filtering. Every stock in results is counted for tokens.

    TOKEN SPLIT ASSUMPTION:
        Same as run_full.py: 80% input tokens, 20% output tokens, based on
        empirical observation of Dexter's prompt-heavy conversation pattern.

    Args:
        results: Dict of ticker -> Dexter result dict. Each result may
                 contain 'total_tokens' and 'total_tools' fields.

    Returns:
        Dict with cost breakdown:
          - total_tokens:  Total LLM tokens consumed
          - input_tokens:  Estimated input tokens (80% of total)
          - output_tokens: Estimated output tokens (20% of total)
          - total_tools:   Total tool calls made by Dexter
          - llm_cost:      Estimated LLM cost in USD
          - api_cost:      Always $0.00 for test mode (free tickers)
          - total_cost:    llm_cost + api_cost
    """
    total_tokens = sum(r.get("total_tokens", 0) for r in results.values())
    total_tools = sum(r.get("total_tools", 0) for r in results.values())

    # Apply the 80/20 input/output split assumption
    input_tokens = int(total_tokens * 0.8)
    output_tokens = int(total_tokens * 0.2)

    # Calculate LLM cost using per-million-token pricing
    llm_cost = (input_tokens / 1_000_000 * COST_PER_1M_INPUT
                + output_tokens / 1_000_000 * COST_PER_1M_OUTPUT)

    # API cost is always zero in test mode (free tickers only)
    api_cost = len(results) * COST_PER_STOCK_API

    return {
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tools": total_tools,
        "llm_cost": llm_cost,
        "api_cost": api_cost,
        "total_cost": llm_cost + api_cost,
    }


# ── Report cleanup ─────────────────────────────────────────────────────────

def cleanup_reports(keep_last: int) -> None:
    """Delete old report files, keeping only the N most recent.

    Identical to run_full.py's cleanup_reports(). Reports accumulate in
    the reports/ directory; this function removes the oldest ones.

    Args:
        keep_last: Number of most recent reports to keep. Must be > 0.
                   PDFs and JSONs are cleaned up independently.
    """
    if not REPORTS_DIR.exists():
        return
    # Sort by modification time (newest first) so we can keep the first N
    pdfs = sorted(REPORTS_DIR.glob("*.pdf"), key=lambda f: f.stat().st_mtime, reverse=True)
    jsons = sorted(REPORTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    # Slice off everything after the first keep_last entries and delete them
    for old in pdfs[keep_last:]:
        old.unlink()
        print(f"  Removed old report: {old.name}")
    for old in jsons[keep_last:]:
        old.unlink()
        print(f"  Removed old report: {old.name}")


# ── Retry errors ────────────────────────────────────────────────────────────

def retry_failed_stocks(stock_list, results, bun_path, dexter_script, dexter_root, checkpoint_file, today, timeout, model):
    """Re-run the Dexter analysis only on stocks that previously failed.

    Identical logic to run_full.py's retry_failed_stocks(). Filters the
    stock list to only those with errors, re-runs Dexter on each one,
    and saves the checkpoint after each stock.

    NOTE: Unlike run_full.py, this function uses direct imports from
    src.dexter_runner (because Dexter's report_generator/ is on sys.path).

    Args:
        stock_list:      Full list of stock dicts from stock_loader.
        results:         Dict of ticker -> result from the previous run.
        bun_path:        Path to the Bun executable.
        dexter_script:   Path to dexter_analyze.ts.
        dexter_root:     Path to the Dexter project root.
        checkpoint_file: Path to checkpoint_test.json for saving progress.
        today:           Date string (YYYY-MM-DD) for checkpoint metadata.
        timeout:         Max seconds per stock for the Dexter subprocess.
        model:           LLM model identifier (e.g., "deepseek-chat").

    Returns:
        Updated results dict with retried stocks' new results.
    """
    # Direct import (not importlib) -- safe because only Dexter's src/ is on sys.path
    from src.dexter_runner import run_dexter, save_checkpoint

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


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the test pipeline.

    Runs a 5-step pipeline:
      Step 1: Load pre-built test stock data (NVDA, MSFT, AAPL)
      Step 2: Run Dexter analysis with checkpoint/resume support
      Step 3: Extract investment verdicts from Dexter's analysis
      Step 4: Generate an executive summary via LLM
      Step 5: Build PDF + JSON reports

    This is a simplified version of run_full.py's main() that skips the
    YouTube pipeline entirely and uses fixed parameters.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Test pipeline: 3 free tickers")
    parser.add_argument("--retry-errors", action="store_true", help="Re-run only failed stocks from checkpoint")
    parser.add_argument("--keep-last", type=int, default=0, help="Keep only N most recent reports (0 = keep all)")
    args = parser.parse_args()

    t_start = time.time()

    # Set up logging with mode="test" so the log directory is named
    # "test_YYYY-MM-DD_HHMMSS" (distinguishable from full runs).
    log_file = setup_logging(mode="test")

    # Display run configuration banner
    print("=" * 60)
    print("  TEST PIPELINE - 3 Free Tickers")
    print("=" * 60)
    print(f"  Date:      {TODAY}")
    print(f"  Dexter:    {DEXTER_ROOT}")
    print(f"  Test data: {TEST_DATA_DIR}")
    print(f"  Output:    {OUTPUT_DIR}")
    print(f"  Log:       {log_file}")
    print()

    # ── Step 1: Load test stock data ─────────────────────────────────────
    # Load the pre-built test data (simulates YouTube pipeline output).
    # The test data contains ticker_scores.json and insights.json for
    # NVDA, MSFT, and AAPL with realistic signal data.
    print("[1/5] Loading test stock data...")

    # Direct import -- safe because only Dexter's report_generator/ is on sys.path
    from src.stock_loader import load_stocks

    stock_list = load_stocks(
        ticker_scores_path=TEST_DATA_DIR / "ticker_scores.json",
        insights_path=TEST_DATA_DIR / "insights.json",
        top_n=3,  # Always 3 stocks in test mode
    )
    print(f"  Loaded {len(stock_list)} stocks")
    for s in stock_list:
        print(
            f"    {s['ticker']:<6} {s['company']:<20} "
            f"mentions={s['mentions']}  creators={s['unique_creators']}"
        )
    print()

    # ── Step 2: Run Dexter analysis ──────────────────────────────────────
    # Run the Dexter headless agent on each test stock. Uses the same
    # checkpoint/resume system as run_full.py, but with a separate
    # checkpoint file (checkpoint_test.json) to avoid conflicts.
    print("[2/5] Running Dexter analysis (with checkpoint)...")
    from src.dexter_runner import run_all_stocks, load_checkpoint, save_checkpoint

    # If retrying errors, clear the same-day skip flag so they get re-run.
    # This is the same logic as in run_full.py -- the _failed_date field
    # normally prevents re-running stocks that already failed today.
    if args.retry_errors:
        existing = load_checkpoint(CHECKPOINT)
        cleared = 0
        for ticker, result in existing.items():
            if result.get("error") and result.get("_failed_date"):
                del result["_failed_date"]
                cleared += 1
        if cleared:
            save_checkpoint(existing, CHECKPOINT, TODAY)
            print(f"  Cleared {cleared} failed-today flags for retry")

    results = run_all_stocks(
        stock_list=stock_list,
        bun_path=BUN_PATH,
        dexter_script=DEXTER_SCRIPT,
        dexter_root=DEXTER_ROOT,
        checkpoint_file=CHECKPOINT,
        today=TODAY,
        timeout=TIMEOUT,
        model=MODEL,
    )

    # Retry failed stocks if --retry-errors was passed
    if args.retry_errors:
        results = retry_failed_stocks(
            stock_list, results, BUN_PATH, DEXTER_SCRIPT,
            DEXTER_ROOT, CHECKPOINT, TODAY, TIMEOUT, MODEL,
        )
    print()

    # ── Step 3: Extract verdicts ─────────────────────────────────────────
    # Parse each stock's analysis to extract structured verdicts.
    # Same as Steps 10 in run_full.py.
    print("[3/5] Extracting verdicts...")
    from src.verdict_extractor import build_verdicts

    analyses, verdicts, verdict_counts = build_verdicts(
        stock_list, results
    )
    # Display per-stock verdicts for quick visual confirmation
    print("  VERDICTS:")
    for s in stock_list:
        t = s["ticker"]
        v = verdicts.get(t, {})
        print(
            f"    {t:<6} {v.get('verdict', 'N/A'):<12}"
        )
    print()

    # ── Step 4: Executive summary ────────────────────────────────────────
    # Generate a portfolio-level executive summary via DeepSeek LLM.
    # Same as Step 11 in run_full.py.
    print("[4/5] Generating executive summary...")
    from src.executive_summary import generate_executive_summary

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        sys.exit("ERROR: DEEPSEEK_API_KEY environment variable not set.")
    executive_summary = generate_executive_summary(
        stock_list=stock_list,
        verdicts=verdicts,
        analyses=analyses,
        today=TODAY,
        api_key=api_key,
    )
    print("  Done.\n")

    # ── Step 5: Build PDF + JSON report ──────────────────────────────────
    # Generate the final reports. Same as Step 12 in run_full.py,
    # but with mode="test" passed to save_json_report for labeling.
    print("[5/5] Building report...")
    from src.pdf_report import build_report
    from src.json_report import save_json_report

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
        mode="test",  # Marks the JSON report as a test run (not in run_full.py)
    )

    # ── Copy reports to parent repo reports/ folder ──────────────────────
    # Same as run_full.py: copy from Dexter's output/ to the parent reports/ dir
    REPORTS_DIR.mkdir(exist_ok=True)
    final_pdf = REPORTS_DIR / Path(pdf_path).name
    final_json = REPORTS_DIR / Path(json_path).name
    shutil.copy2(pdf_path, final_pdf)
    shutil.copy2(json_path, final_json)

    # ── Cleanup old reports ──────────────────────────────────────────────
    if args.keep_last > 0:
        print(f"\n  Cleaning up reports (keeping last {args.keep_last})...")
        cleanup_reports(args.keep_last)

    # ── Cost summary ────────────────────────────────────────────────────
    # Remove the internal _fresh_tickers tracking field before calculating costs.
    # In test mode, we don't need fresh vs cached distinction because all
    # tickers are free-tier and the simplified estimate_cost() counts everything.
    results.pop("_fresh_tickers", None)
    cost = estimate_cost(results)
    elapsed = time.time() - t_start

    # Display the final summary with report paths, timing, and cost breakdown
    print()
    print("=" * 60)
    print("  TEST REPORT COMPLETE")
    print(f"  PDF:  {final_pdf}")
    print(f"  JSON: {final_json}")
    print(f"  Time: {elapsed / 60:.1f} min")
    print(f"  {'-' * 40}")
    print(f"  COST SUMMARY")
    print(f"    Tokens:   {cost['total_tokens']:,} ({cost['input_tokens']:,} in / {cost['output_tokens']:,} out)")
    print(f"    Tools:    {cost['total_tools']}")
    print(f"    LLM:      ${cost['llm_cost']:.4f}")
    print(f"    Data API: ${cost['api_cost']:.2f} (free tickers)")
    print(f"    TOTAL:    ${cost['total_cost']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
