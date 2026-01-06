# sweep.ps1
# .\.venv\Scripts\Activate.ps1
# powershell -ExecutionPolicy Bypass -File .\scripts\sweep.ps1
param(
    [string]$symbol = "BTCUSDT",
    [string]$start = "20250601",
    [string]$end = "20251220",
    [string]$data_root = "D:\data\profile-regime",

    # grid
    [string[]]$tf_list = @("15min", "1h"),
    [int[]]$window = @(96, 192, 384),
    [double[]]$quantile = @(0.90, 0.95),
    [int[]]$min_bars = @(2, 4, 8),
    [double[]]$merge_gap_ratio = @(0.25, 0),        # merge_gap = min_bars * merge_gap_ratio
    [int[]]$horizon_ratio = @(2, 4, 6, 8)              # horizon = min_bars * horizon_ratio
)
# default params
#   sleep-sec : 0.2
#   r-norm : range (range | vol_tanh)
#   vol-window : 96 (vol_tanh)
#   eps : 1e-12
#   baseline-ratio : 3.0
#   log-level : INFO
#   metrics : mfe,mae,range
#   atz-label : atz
#   base-label : baseline

# store true params
#   no-cv
#   no-trade-count
#   by-date
#   overwrite

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$rand = Get-Random -Minimum 0 -Maximum 1000
$randStr = $rand.ToString("D3")
$run_id = "${timestamp}_${randStr}"

$rows = New-Object System.Collections.Generic.List[object]
foreach ($tf in $tf_list){
    foreach ($w in $window){
        foreach ($q in $quantile){
            $qStr = "{0:0.00}" -f $q
            foreach ($m in $min_bars){
                foreach ($merge_gap_r in $merge_gap_ratio){
                    $g = [int][math]::Floor($m * $merge_gap_r)  | Select-Object -Unique
                    foreach ($horizon_r in $horizon_ratio){
                        $h = $m * $horizon_r

                        $rows.Add([pscustomobject]@{
                            tf = $tf
                            window = $w
                            quantile = $q
                            min_bars = $m
                            merge_gap = $g
                            horizon = $h
                            grid_id = "tf=${tf}__w=${w}__q=${qStr}__m=${m}__g=${g}__h=${h}"
                        })
                    }

                }
            }
        }
    }
}

$ROOT = Split-Path -Parent $PSScriptRoot   # project root
$PY   = Join-Path $ROOT ".venv\Scripts\python.exe"
$RUNPY = Join-Path $ROOT "src\run.py"

if (-not (Test-Path $PY))    { throw "python not found: $PY" }
if (-not (Test-Path $RUNPY)) { throw "run.py not found: $RUNPY" }

foreach ($row in $rows){
    $pyArgs = @(
        "--symbol", $symbol,
        "--start", $start,
        "--end", $end,
        "--data-root", $data_root,

        "--tf", $row.tf,
        "--window", $row.window,
        "--q", $row.quantile,
        "--min-bars", $row.min_bars,
        "--merge-gap", $row.merge_gap,
        "--horizon", $row.horizon,

        # "--overwrite",

        "--run-id", $run_id,
        "--grid-id", $row.grid_id
    )

    Push-Location $ROOT
    try {
        & $PY $RUNPY @pyArgs
    } finally {
        Pop-Location
    }
}