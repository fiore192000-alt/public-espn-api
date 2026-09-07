<#
.SYNOPSIS
    Trains the XGBoost entry model on the downloaded candles.
.DESCRIPTION
    Everything before -TestStart is used to fit and tune the model; everything
    from -TestStart onward is only ever *scored*, never trained on. The printed
    "test" numbers are therefore the honest ones — the validation numbers are
    not, because the threshold was chosen on them.
.EXAMPLE
    .\scripts\03_train.ps1
    .\scripts\03_train.ps1 -Timeframe 4h -Horizon 12 -TpAtr 2.5
#>

param(
    [string[]]$Pairs = @("BTC/USDT", "ETH/USDT", "SOL/USDT"),
    [string]$Timeframe = "1h",
    [string]$TrainEnd = "2024-01-01",
    [string]$TestStart = "2024-01-01",
    [int]$Horizon = 24,
    [double]$TpAtr = 2.0,
    [double]$SlAtr = 1.0
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "Training on data before $TrainEnd, testing from $TestStart onward." -ForegroundColor Cyan

docker compose run --rm --entrypoint python --workdir /freqtrade/user_data freqtrade `
    -m ml.train `
    --pairs $Pairs `
    --timeframe $Timeframe `
    --train-end $TrainEnd `
    --test-start $TestStart `
    --horizon $Horizon `
    --tp-atr $TpAtr `
    --sl-atr $SlAtr

if ($LASTEXITCODE -ne 0) { throw "training failed" }

Write-Host ""
Write-Host "Model written to user_data\models\entry" -ForegroundColor Green
Write-Host "Read the 'test' line before going further: if its expectancy is not" -ForegroundColor Yellow
Write-Host "clearly positive, the backtest will not save it." -ForegroundColor Yellow
Write-Host "Next step:  .\scripts\04_backtest.ps1" -ForegroundColor Green
