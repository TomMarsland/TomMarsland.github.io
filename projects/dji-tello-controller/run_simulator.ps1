$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Starting simulator from $PSScriptRoot"
Write-Host "Using cascade PID controller"

.\use_amr_clean.ps1 run.py
