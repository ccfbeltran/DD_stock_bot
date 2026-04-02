# DD Stock Bot - Automated Stock Due Diligence Pipeline

Automated investment research system that discovers stock signals from YouTube investment channels, validates them against real financial data, and produces professional PDF due diligence reports.

## Architecture

The system consists of two submodules working as a pipeline:

```
DD_stock_bot/                        <-- Parent repo (orchestrator)
  |
  +-- Stock_Due_Diligence_bot/       <-- Submodule 1: YouTube Signal Discovery
  |     Steps 1-7: Fetch -> Filter -> Transcribe -> Summarize -> Extract -> Aggregate -> Rank
  |     Output: ticker_scores.json + insights.json (ranked stock signals)
  |
  +-- dexter-duedeligencebot/        <-- Submodule 2: Deep Due Diligence
  |     Steps 8-12: Load Signals -> Dexter Agent -> Verdicts -> Summary -> Report
  |     Output: PDF + JSON investment report
  |
  +-- run_full.py                    <-- Full pipeline runner (Steps 1-12)
  +-- run_test.py                    <-- Test runner (3 free tickers, Steps 8-12 only)
  +-- pipeline_logger.py             <-- Shared logging system
  +-- sync_config.json               <-- Path configuration
  +-- requirements.txt               <-- Unified Python dependencies
  +-- reports/                        <-- Generated reports (PDF + JSON)
  +-- logs/                           <-- Per-run log folders
```

### Pipeline Flow

```
  YouTube Channels (43+)
         |
    [Step 1] Fetch videos (YouTube Data API)
    [Step 2] Filter non-investment content (DeepSeek LLM)
    [Step 3] Download audio & transcribe (yt-dlp + Whisper)
    [Step 4] Summarize into investment memos (DeepSeek LLM)
    [Step 5] Extract per-ticker structured insights (DeepSeek LLM)
    [Step 6] Aggregate insights by ticker & date
    [Step 7] Rank tickers with multi-factor scoring
         |
    ticker_scores.json + insights.json
         |
    [Step 8]  Load ranked signals
    [Step 9]  Run Dexter AI agent per stock (Financial Datasets API + DeepSeek)
    [Step 10] Extract verdicts (BUY / HOLD / SELL)
    [Step 11] Generate executive summary (DeepSeek)
    [Step 12] Build PDF + JSON report
         |
    Deep_DD_Report_YYYY-MM-DD.pdf
```

## Quick Start

### Prerequisites

- Python 3.11+ (tested on 3.13)
- [Bun](https://bun.sh) JavaScript runtime (for Dexter agent)
- API keys:
  - `DEEPSEEK_API_KEY` - Required for all LLM calls
  - `YOUTUBE_API_KEY` - Required for video fetching (Steps 1-2)
  - `FINANCIAL_DATASETS_API_KEY` - Optional but recommended for financial data

### Installation

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>
cd DD_stock_bot

# Install Python dependencies
pip install -r requirements.txt

# Install Dexter dependencies (TypeScript)
cd dexter-duedeligencebot
bun install
cd ..

# Set up environment variables
# Edit .env and dexter-duedeligencebot/.env with your API keys
```

### Configuration

Edit `.env` in the project root:

```env
DEEPSEEK_API_KEY=sk-your-key-here
FINANCIAL_DATASETS_API_KEY=your-key-here
```

Edit `dexter-duedeligencebot/.env`:

```env
DEEPSEEK_API_KEY=sk-your-key-here
FINANCIAL_DATASETS_API_KEY=your-key-here
```

The `sync_config.json` file configures paths between submodules:

```json
{
  "stock_dd_output": "Stock_Due_Diligence_bot/data/output",
  "dexter_root": "dexter-duedeligencebot",
  "bun_path": "C:\\Users\\username\\.bun\\bin\\bun.exe"
}
```

## Usage

### Full Pipeline (YouTube + Dexter + Report)

```bash
# Run everything (takes 30-60 min depending on video count)
python run_full.py

# Analyze top 15 stocks instead of default 10
python run_full.py --top-n 15

# Skip YouTube pipeline, use existing signals (for re-running Dexter only)
python run_full.py --skip-youtube

# Skip Dexter, only run YouTube pipeline
python run_full.py --skip-dexter

# Retry stocks that failed/timed out
python run_full.py --skip-youtube --retry-errors

# Keep only the 5 most recent reports
python run_full.py --keep-last 5
```

### Test Pipeline (3 Free Tickers)

```bash
# Run with pre-built test data (NVDA, MSFT, AAPL -- free API tier)
python run_test.py

# Retry failed test stocks
python run_test.py --retry-errors
```

## Output

### Reports

Reports are saved to `reports/`:

```
reports/
  Deep_DD_Report_2026-03-29.pdf    <-- Professional PDF report
  Deep_DD_Report_2026-03-29.json   <-- Machine-readable JSON data
```

The PDF report includes:
- Executive summary with market overview
- Per-stock deep analysis with:
  - Financial Health Scorecard (Revenue, Profitability, Balance Sheet, Cash Flow, Valuation)
  - Bull Case / Bear Case with CONFIRMED/UNCONFIRMED markers
  - Key Leadership (CEO, CFO, CTO, COO with ratings)
  - News & Catalysts attributed to responsible executives
  - Insider Activity with named insiders
  - DCF Valuation
  - Final Verdict (STRONG BUY / BUY / HOLD / SELL / STRONG SELL)
- Sentiment timeline charts
- YouTube creator insights and attribution
- Methodology appendix

### Logs

Each run creates a timestamped folder in `logs/`:

```
logs/
  full_2026-03-29_143022/
    pipeline.log          <-- Full pipeline log (DEBUG level)
    dexter_NVDA.log       <-- Per-stock Dexter subprocess output
    dexter_AAPL.log
    summary.json          <-- Run summary (stocks, verdicts, costs)
```

## Caching System

The pipeline uses a multi-layer caching system to avoid redundant API calls:

### Layer 1: YouTube Pipeline Cache

| Data | Cache Method | How |
|------|-------------|-----|
| Filtered videos | URL dedup | Already-filtered videos are skipped (no DeepSeek call) |
| Transcriptions | Status flag | `status.transcribed = true` skips re-download + re-transcribe |
| Summaries | Status flag | `status.summarized = true` skips re-summarization |
| Insights | Status flag | `status.insights_extracted = true` skips re-extraction |

### Layer 2: Dexter Report Cache (per ticker, disk-based)

| Cache | TTL | Purpose |
|-------|-----|---------|
| `gather_context` | 24h | Full Turn 1 output -- skips entire data gathering phase |
| `leadership` | 30 days | CEO/CFO/CTO bios -- executives rarely change |

### Layer 3: HTTP API Cache (per endpoint, disk-based)

| Data | TTL | Rationale |
|------|-----|-----------|
| Income/Balance/Cash Flow | 24h | Quarterly data, refreshed daily |
| Analyst Estimates | 12h | More volatile than fundamentals |
| Insider Trades | 12h | Filed throughout the day |
| Company News | 6h | News cycle is fast |
| Stock Price Snapshot | 1h | Intraday changes |
| Historical Prices (past) | Forever | Closed candles are immutable |
| Historical Prices (today) | 4h | Today's candle still forming |
| SEC Filing Content | Forever | Filed documents are immutable |
| Ticker Lists | 7 days | New IPOs/delistings are rare |

### Layer 4: Checkpoint System

| File | Purpose |
|------|---------|
| `checkpoint.json` | Completed Dexter analyses -- stocks with valid results are skipped on re-run |
| `checkpoint_test.json` | Same as above, for test pipeline (separate to avoid interference) |
| `_failed_date` flag | Stocks that failed today are skipped to avoid re-spending (cleared by `--retry-errors`) |

## Cost Tracking

Each run displays an accurate cost summary:

```
COST SUMMARY (this run only -- 3 fresh, 7 cached)
  Tokens:     691,311 (553,048 in / 138,262 out)
  Tools:      25
  LLM:        $0.30
  Data API:   $0.87 (3 paid stocks x $0.29)
  TOTAL:      $1.17
```

- **Checkpointed stocks**: $0 (not re-run)
- **Stocks with cached Turn 1**: LLM cost only (no API calls)
- **Fresh stocks**: LLM + API cost
- **Free tickers** (NVDA, MSFT, AAPL): No API cost regardless

## File Reference

### Parent Repo (DD_stock_bot/)

| File | Purpose |
|------|---------|
| `run_full.py` | Full pipeline orchestrator (Steps 1-12) |
| `run_test.py` | Test pipeline (3 free tickers, Steps 8-12) |
| `pipeline_logger.py` | Shared logging system (dual file+console, TeeWriter, SafeStream) |
| `sync_config.json` | Path configuration between submodules |
| `requirements.txt` | Unified Python dependencies |
| `.env` | API keys (DEEPSEEK, FINANCIAL_DATASETS) |

### Stock_Due_Diligence_bot/ (YouTube Pipeline)

| File | Purpose |
|------|---------|
| `src/fetch.py` | Fetch videos from YouTube channels |
| `src/filter.py` | Filter non-investment content via DeepSeek |
| `src/transcribe.py` | Download audio + transcribe with Whisper |
| `src/summarize.py` | Generate investment memos via DeepSeek |
| `src/insights.py` | Extract structured per-ticker insights |
| `src/aggregate.py` | Group insights by ticker and date |
| `src/rank.py` | Multi-factor ticker ranking |
| `config/settings.py` | Channels, keywords, API keys, thresholds |

### dexter-duedeligencebot/ (Deep DD Pipeline)

| File | Purpose |
|------|---------|
| `report_generator/dexter_analyze.ts` | 2-turn Dexter agent runner |
| `report_generator/dexter_cache.ts` | Disk-based financial data cache |
| `report_generator/src/dexter_runner.py` | Python subprocess manager for Dexter |
| `report_generator/src/stock_loader.py` | Load and enrich stock data |
| `report_generator/src/verdict_extractor.py` | Extract BUY/HOLD/SELL verdicts |
| `report_generator/src/executive_summary.py` | Generate CIO-style briefing |
| `report_generator/src/pdf_report.py` | Build professional PDF report |
| `report_generator/src/json_report.py` | Build machine-readable JSON report |
| `src/tools/finance/` | Financial data tools (15+ API endpoints) |
| `src/utils/cache.ts` | HTTP-level API response cache |

## Metrics

The pipeline tracks several quality metrics per stock:

| Metric | Reliability | How Calculated |
|--------|------------|----------------|
| **Score** | High | `net_signal * (0.5 + quality) * (0.75 + 0.25 * diversity) * coverage` |
| **Quality** | High | Mean of per-insight support scores (8 sub-factors) |
| **Freshness** | Excellent | Exponential decay: `0.5^(age_days / 14)` |

## Known Limitations

- Some YouTube videos are geo-restricted and cannot be downloaded (US-only channels)
- Whisper transcription may crash on Windows during cleanup (handled gracefully)
- DeepSeek API rate limits may cause slower processing with many stocks
- The Financial Datasets API free tier only covers NVDA, MSFT, AAPL

## License

Private repository. Not for distribution.
