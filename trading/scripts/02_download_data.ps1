<#
.SYNOPSIS
    Downloads historical candles from the exchange into user_data/data.
.EXAMPLE
    .\scripts\02_download_data.ps1
    .\scripts\02_download_data.ps1 -Pairs "BTC/USDT","ETH/USDT" -Timeframes "1h" -Days 2000
.NOTES
    Public market data only — no API key, no account, nothing to pay.
#>

param(
    [string[]]$Pairs = @("BTC/USDT", "ETH/USDT", "SOL/USDT"),
    [string[]]$Timeframes = @("1h", "4h"),
    [int]$Days = 2200,
    [string]$Exchange = "binance"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host ("Downloading {0} days of {1} data for: {2}" -f $Days, ($Timeframes -join ", "), ($Pairs -join ", ")) -ForegroundColor Cyan
Write-Host "This can take several minutes on the first run." -ForegroundColor DarkGray

docker compose run --rm freqtrade download-data `
    --config /freqtrade/user_data/config.dryrun.json `
    --exchange $Exchange `
    --pairs $Pairs `
    --timeframes $Timeframes `
    --days $Days `
    --data-format-ohlcv feather

if ($LASTEXITCODE -ne 0) { throw "download-data failed" }

Write-Host ""
Write-Host "Files written to user_data\data\$Exchange" -ForegroundColor Green
Write-Host "Next step:  .\scripts\03_train.ps1" -ForegroundColor Green
