<#
.SYNOPSIS
    Starts the bot in dry-run: live prices, simulated orders, no money.
.DESCRIPTION
    dry_run is set to true in config.dryrun.json and the exchange keys are
    empty, so the bot cannot place a real order even if it wanted to.
    Let it run for weeks, not days, and compare the result with the backtest
    over the same window — if they disagree, believe the dry-run.
.EXAMPLE
    .\scripts\05_dryrun.ps1
    .\scripts\05_dryrun.ps1 -Follow
    .\scripts\05_dryrun.ps1 -Stop
#>

param(
    [switch]$Follow,
    [switch]$Stop
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if ($Stop) {
    docker compose down
    Write-Host "Bot stopped." -ForegroundColor Green
    return
}

$config = Get-Content "user_data/config.dryrun.json" -Raw | ConvertFrom-Json
if (-not $config.dry_run) {
    throw "config.dryrun.json has dry_run = false. Refusing to start."
}
if ($config.exchange.key -ne "") {
    throw "config.dryrun.json contains an exchange API key. Refusing to start."
}

Write-Host "Starting the bot in dry-run (simulated orders, no real funds)." -ForegroundColor Cyan
docker compose up -d
if ($LASTEXITCODE -ne 0) { throw "failed to start the bot" }

Write-Host "Running. Logs:  docker compose logs -f" -ForegroundColor Green
if ($Follow) { docker compose logs -f }
