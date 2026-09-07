<#
.SYNOPSIS
    Builds the Freqtrade image with the ML libraries baked in.
.DESCRIPTION
    Downloads a few hundred MB the first time and is near-instant afterwards.
    Re-run it whenever requirements-ml.txt changes.
#>

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "Building freqtrade-ml image (first run downloads ~1 GB)..." -ForegroundColor Cyan
docker compose build
if ($LASTEXITCODE -ne 0) { throw "docker compose build failed" }

Write-Host "Verifying the image can import the ML stack..." -ForegroundColor Cyan
docker compose run --rm --entrypoint python freqtrade -c "import freqtrade, xgboost, sklearn, pandas; print('freqtrade', freqtrade.__version__, '| xgboost', xgboost.__version__, '| sklearn', sklearn.__version__, '| pandas', pandas.__version__)"
if ($LASTEXITCODE -ne 0) { throw "the image is missing one of the required libraries" }

Write-Host ""
Write-Host "Done. Next step:  .\scripts\02_download_data.ps1" -ForegroundColor Green
