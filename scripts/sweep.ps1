# scripts/sweep.ps1
# Usage:
#   pwsh -File scripts/sweep.ps1
# or (Windows PowerShell)
#   powershell -ExecutionPolicy Bypass -File scripts/sweep.ps1

$ErrorActionPreference = "Stop"

$RUN_PY = "src\run.py"

$SYMBOL = "BTCUSDT"
$START = "20250601"
$END = "20251201"
$DATA_ROOT = "D:\,,,"

# ===== sweep grid =====
$TF_LIST = @("5min", "15min", "1h")

$ATZ_PARAMS = @(
    @{ MinBars = 2; MergeGap = 0}
    @{ MinBars = 4; MergeGap = 1}
    @{ MinBars = 8; MergeGap = 2}
)
$HORIZON_HOURS = @(1, 4, 8, 12) # Evaluation horizon in TIME (not bars)

# =====fixed params=====
$Q = 0.90
$WINDOW_TIME = "1d"

$BASELINE_RATIO = 3.0
$SEED           = 7
$METRICS        = "mfe,mae,range"

# =====optional=====
$OVERWRITE = $false
$BY_DATE   = $false

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG_DIR = Join-Path -Path $DATA_ROOT -ChildPath ("sweep_logs\" + $ts)
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

Write-Host "SWEEP START $ts"
Write-Host "SYMBOL      $SYMBOL"
Write-Host "RANGE       $START ~ $END"
Write-Host "DATA_ROOT $DATA_ROOT"
Write-Host "LOG_DIR       $LOG_DIR"
Write-Host ""

function Run-One($tf)


Run-One()