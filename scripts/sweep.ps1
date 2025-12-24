# scripts/sweep.ps1
# Usage:
#   pwsh -File scripts/sweep.ps1
# or (Windows PowerShell)
#   powershell -ExecutionPolicy Bypass -File scripts/sweep.ps1

$ErrorActionPreference = "Stop"

# ====== EDIT THESE ======
$SYMBOL   = "BTCUSDT"
$START    = "20240601"     # YYYYMMDD (UTC)
$END      = "20241201"     # YYYYMMDD (UTC)
$DATA_ROOT = "D:\YOUR\DATA\ROOT"  # <-- CHANGE THIS

# ====== sweep grid ======
$TF_LIST      = @("5min", "15min")
$Q_LIST       = @(0.85, 0.90, 0.95)
$WINDOW_LIST  = @(48, 96, 192)

# fixed params
$ATZ_MIN_BARS   = 2
$ATZ_MERGE_GAP  = 1
$HORIZON        = 8
$BASELINE_RATIO = 3.0
$SEED           = 7
$METRICS        = "mfe,mae,range"

# Optional: add --overwrite to force rebuild
$OVERWRITE = $false   # set $true if you want to overwrite outputs
$BY_DATE   = $false   # set $true if you want analyze by-date

# ====== bookkeeping ======
$RUN_PY = "src\run.py"

# Create a timestamped log folder for this sweep
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG_DIR = Join-Path -Path $DATA_ROOT -ChildPath ("sweep_logs\" + $ts)
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

Write-Host "SWEEP START  $ts"
Write-Host "LOG_DIR      $LOG_DIR"
Write-Host "DATA_ROOT    $DATA_ROOT"
Write-Host ""

function Run-One($tf, $q, $win) {
    $tag = "sweep_tf=$tf`_q=$q`_w=$win"
    $logPath = Join-Path $LOG_DIR ("$tag.log")

    $args = @(
        $RUN_PY,
        "--symbol", $SYMBOL,
        "--start", $START,
        "--end", $END,
        "--data-root", $DATA_ROOT,
        "--tf", $tf,
        "--atz-window", $win,
        "--atz-q", $q,
        "--atz-min-bars", $ATZ_MIN_BARS,
        "--atz-merge-gap", $ATZ_MERGE_GAP,
        "--horizon", $HORIZON,
        "--baseline-ratio", $BASELINE_RATIO,
        "--seed", $SEED,
        "--metrics", $METRICS,
        "--tag", $tag
    )

    if ($OVERWRITE) { $args += "--overwrite" }
    if ($BY_DATE)   { $args += "--by-date" }

    Write-Host "RUN  tf=$tf  q=$q  win=$win  -> $logPath"

    # 실행 + 로그 저장 (stdout/stderr 모두)
    & python @args  *>> $logPath

    if ($LASTEXITCODE -ne 0) {
        throw "FAILED: tf=$tf q=$q win=$win (see $logPath)"
    }
}

# ====== main loops ======
foreach ($tf in $TF_LIST) {
    foreach ($q in $Q_LIST) {
        foreach ($win in $WINDOW_LIST) {
            Run-One $tf $q $win
        }
    }
}

Write-Host ""
Write-Host "SWEEP DONE ✅  logs at: $LOG_DIR"
