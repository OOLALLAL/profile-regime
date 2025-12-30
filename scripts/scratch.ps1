# scripts/sweep.ps1
# Run:
#   powershell -ExecutionPolicy Bypass -File scripts\sweep.ps1
# or:
#   pwsh -File scripts\sweep.ps1

$ErrorActionPreference = "Stop"

# ======================
# EDIT THESE
# ======================
$SYMBOL    = "BTCUSDT"
$START     = "20240601"          # YYYYMMDD (UTC)
$END       = "20241201"          # YYYYMMDD (UTC)
$DATA_ROOT = "D:\YOUR\DATA\ROOT" # <-- CHANGE THIS

# ======================
# GRID (fixed design)
# ======================
$TF_LIST = @("5min", "15min", "1h")

# ATZ activity threshold (fixed for stage-1)
$Q = 0.90

# Window time: 1 day (fixed)
$WINDOW_TIME = "1d"

# ATZ params (event definition)
$ATZ_PARAMS = @(
    @{ MinBars = 2; MergeGap = 0 },
    @{ MinBars = 4; MergeGap = 1 },
    @{ MinBars = 8; MergeGap = 2 }
)

# Evaluation horizons in TIME (not bars)
$HORIZON_HOURS = @(1, 4, 8, 12)

# Fixed params
$BASELINE_RATIO = 3.0
$SEED           = 7
$METRICS        = "mfe,mae,range"

# Optional
$OVERWRITE = $false
$BY_DATE   = $false

# Run entrypoint
$RUN_PY = "src\run.py"

# ======================
# Helpers: TF -> minutes
# ======================
function TfMinutes([string]$tf) {
    switch ($tf) {
        "5min"  { return 5 }
        "15min" { return 15 }
        "1h"    { return 60 }
        default { throw "Unsupported tf: $tf (use 5min|15min|1h)" }
    }
}

function WindowBars([string]$tf, [string]$windowTime) {
    $m = TfMinutes $tf
    switch ($windowTime) {
        "1d" { return [int](1440 / $m) }   # 1 day = 1440 minutes
        "8h" { return [int](480 / $m) }    # if you ever want it
        "3d" { return [int](4320 / $m) }
        default { throw "Unsupported window_time: $windowTime (use 1d)" }
    }
}

function HorizonBars([string]$tf, [int]$hours) {
    $m = TfMinutes $tf
    $mins = 60 * $hours
    $bars = [int]($mins / $m)
    if ($bars -lt 1) { $bars = 1 }
    return $bars
}

# ======================
# Log folder
# ======================
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$LOG_DIR = Join-Path -Path $DATA_ROOT -ChildPath ("sweep_logs\" + $ts)
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

Write-Host "SWEEP START  $ts"
Write-Host "SYMBOL       $SYMBOL"
Write-Host "RANGE        $START ~ $END"
Write-Host "DATA_ROOT    $DATA_ROOT"
Write-Host "LOG_DIR      $LOG_DIR"
Write-Host ""

function Run-One([string]$tf, [int]$winBars, [double]$q, [int]$minBars, [int]$gap, [int]$hBars, [string]$hLabel) {
    $tag = "tf=$tf`_w=$WINDOW_TIME`_q=$q`_mb=$minBars`_gap=$gap`_h=$hLabel"
    $logPath = Join-Path $LOG_DIR ("$tag.log")

    $args = @(
        $RUN_PY,
        "--symbol", $SYMBOL,
        "--start", $START,
        "--end", $END,
        "--data-root", $DATA_ROOT,

        "--tf", $tf,

        "--atz-window", $winBars,
        "--atz-q", $q,
        "--atz-min-bars", $minBars,
        "--atz-merge-gap", $gap,

        "--horizon", $hBars,
        "--baseline-ratio", $BASELINE_RATIO,
        "--seed", $SEED,

        "--metrics", $METRICS,
        "--tag", $tag
    )

    if ($OVERWRITE) { $args += "--overwrite" }
    if ($BY_DATE)   { $args += "--by-date" }

    Write-Host "RUN  $tag  (winBars=$winBars, hBars=$hBars) -> $logPath"

    & python @args  *>> $logPath

    if ($LASTEXITCODE -ne 0) {
        throw "FAILED: $tag (see $logPath)"
    }
}

# ======================
# Main sweep (36 runs)
# ======================
foreach ($tf in $TF_LIST) {
    $winBars = WindowBars $tf $WINDOW_TIME

    foreach ($p in $ATZ_PARAMS) {
        $mb  = [int]$p.MinBars
        $gap = [int]$p.MergeGap

        foreach ($hh in $HORIZON_HOURS) {
            $hBars = HorizonBars $tf $hh
            $hLabel = "$($hh)h"

            Run-One $tf $winBars $Q $mb $gap $hBars $hLabel
        }
    }
}

Write-Host ""
Write-Host "SWEEP DONE ✅  logs at: $LOG_DIR"
