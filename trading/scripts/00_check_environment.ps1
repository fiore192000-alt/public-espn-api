<#
.SYNOPSIS
    Reports what is installed on this machine and what is still missing.
.DESCRIPTION
    Run this first. It never installs or changes anything — it only looks.
    Docker Desktop is the only hard requirement for the Docker route.
#>

$ErrorActionPreference = "Continue"

function Test-Tool {
    param(
        [string]$Name,
        [string]$Command,
        [string[]]$Arguments,
        [string]$Hint,
        [switch]$Optional
    )

    $found = Get-Command $Command -ErrorAction SilentlyContinue
    if (-not $found) {
        $label = if ($Optional) { "OPTIONAL" } else { "MISSING " }
        Write-Host ("[{0}] {1,-14} not found" -f $label, $Name) -ForegroundColor ($(if ($Optional) { "Yellow" } else { "Red" }))
        Write-Host ("            -> {0}" -f $Hint) -ForegroundColor DarkGray
        return $false
    }

    $version = (& $Command @Arguments 2>&1 | Select-Object -First 1)
    Write-Host ("[OK]      {0,-14} {1}" -f $Name, $version) -ForegroundColor Green
    return $true
}

Write-Host ""
Write-Host "Environment check" -ForegroundColor Cyan
Write-Host "-----------------"

$docker = Test-Tool -Name "Docker" -Command "docker" -Arguments @("--version") `
    -Hint "Install Docker Desktop: https://www.docker.com/products/docker-desktop/  (recommended route on Windows)"

$python = Test-Tool -Name "Python" -Command "python" -Arguments @("--version") `
    -Hint "Only needed if you train outside Docker. Freqtrade requires Python 3.11+." -Optional

$git = Test-Tool -Name "Git" -Command "git" -Arguments @("--version") `
    -Hint "Install from https://git-scm.com/download/win" -Optional

Test-Tool -Name "NVIDIA GPU" -Command "nvidia-smi" -Arguments @("--query-gpu=name,driver_version", "--format=csv,noheader") `
    -Hint "Not required. XGBoost trains on CPU here, and this project never needs a GPU." -Optional | Out-Null

if ($docker) {
    Write-Host ""
    Write-Host "Checking that the Docker engine is actually running..." -ForegroundColor Cyan
    docker info --format "{{.ServerVersion}}" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK]      Docker engine responding" -ForegroundColor Green
    }
    else {
        Write-Host "[MISSING] Docker is installed but the engine is not running." -ForegroundColor Red
        Write-Host "            -> Start Docker Desktop and wait for the whale icon to go steady." -ForegroundColor DarkGray
        $docker = $false
    }
}

Write-Host ""
if ($docker) {
    Write-Host "Ready. Next step:  .\scripts\01_build.ps1" -ForegroundColor Cyan
}
else {
    Write-Host "Install Docker Desktop first, then run this script again." -ForegroundColor Yellow
    Write-Host "Without Docker you can still run everything natively — see README.md, 'Route B'." -ForegroundColor DarkGray
}
Write-Host ""
