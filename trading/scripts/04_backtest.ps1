<#
.SYNOPSIS
    Backtests MLProbStrategy over the period the model never saw.
.DESCRIPTION
    Runs the exchange fee model, the order fills and the slot limits that the
    signal-level metrics from training ignore. This is the number that counts.
.EXAMPLE
    .\scripts\04_backtest.ps1
    .\scripts\04_backtest.ps1 -TimeRange 20240101-20250101 -Breakdown month
#>

param(
    [string]$TimeRange = "20240101-",
    [string]$Timeframe = "1h",
    [string]$Breakdown = "month",
    [switch]$SkipLookaheadCheck
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "Backtesting $TimeRange ..." -ForegroundColor Cyan
docker compose run --rm freqtrade backtesting `
    --config /freqtrade/user_data/config.dryrun.json `
    --strategy MLProbStrategy `
    --timeframe $Timeframe `
    --timerange $TimeRange `
    --breakdown $Breakdown `
    --cache none

if ($LASTEXITCODE -ne 0) { throw "backtesting failed" }

if (-not $SkipLookaheadCheck) {
    Write-Host ""
    Write-Host "Running Freqtrade's own lookahead-bias check..." -ForegroundColor Cyan
    Write-Host "It re-runs the strategy on shortened histories: if the results move," -ForegroundColor DarkGray
    Write-Host "the strategy is reading the future somewhere." -ForegroundColor DarkGray
    docker compose run --rm freqtrade lookahead-analysis `
        --config /freqtrade/user_data/config.dryrun.json `
        --strategy MLProbStrategy `
        --timeframe $Timeframe `
        --timerange $TimeRange
    if ($LASTEXITCODE -ne 0) { Write-Host "lookahead-analysis reported a problem — read the output above." -ForegroundColor Red }
}

Write-Host ""
Write-Host "Results in user_data\backtest_results" -ForegroundColor Green
Write-Host "Only move on if the drawdown is something you could actually sit through." -ForegroundColor Yellow
Write-Host "Next step:  .\scripts\05_dryrun.ps1" -ForegroundColor Green
