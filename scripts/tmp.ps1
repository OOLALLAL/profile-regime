param(
    [string]$symbol = "BTCUSDT",
    [string]$start = "20250601",
    [string]$end = "20251220",
    [string]$data_root = "D:\data\profile-regime",

    # grid
    [string[]]$tf_list = @("5min", "15min", "1h"),
    [int[]]$window = @(96, 192, 384, 768),
    [int[]]$quantile = @(0.90, 0.95),
    [int[]]$min_bars = @(2, 4, 8, 16),
    [float[]]$merge_gap_ratio = @(1, 0.5, 0.25),    # merge_gap = min_bars * merge_gap_ratio
    [int[]]$horizon_ratio = @(2, 4, 6, 8)           # horizon = min_bars * horizon_ratio
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

$rows = New-Object System.Collections.Generic.List[object]

foreach ($tf in $tf_list){
    foreach ($w in $window){
        foreach ($q in $quantile){
            foreach ($m in $min_bars){
                $merge_gap = $m * $merge_gap_ratio
                foreach ($g in $merge_gap){
                    $horizon = $m * $horizon_ratio
                    foreach ($h in $horizon){
                        $rows.Add([pscustomobject]@{
                            window = $w
                            quantile = $q
                            min_bars = $m
                            merge_gap = $g
                            horizon = $h
                        })
                    }
                }
            }
        }
    }
}
$pyArgs = @(
    "--symbol", $symbol,`
    "--start", $start,`
    "--end", $end,`
    "--data-root", $data_root,`

    "--tf", $rows.tf,`
    "--window", $rows.window,`
    "--q", $rows.quantile,`
    "--min-bars", $rows.m,`
    "--merge-gap", $rows.g,`
    "--horizon", $rows.h,`

    "--overwrite"
)
& python "src/run.py" @pyArgs

# run_id, combo_id