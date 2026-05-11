param(
    [int]$Passes = 3,
    [double]$Horizon = 10.0,
    [switch]$ResetDefaults
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$tuneArgs = @(
    "autotune_cascade_pid.py",
    "--passes", $Passes,
    "--horizon", $Horizon
)

if ($ResetDefaults) {
    $tuneArgs += "--reset-defaults"
}

Write-Host "Auto-tuning cascade PID..."
.\use_amr_clean.ps1 @tuneArgs

$gainsPath = Join-Path $PSScriptRoot "src\cascade_pid_gains.json"
Write-Host "Tuned gains written to $gainsPath"
Write-Host "Launching simulator with tuned gains..."

.\use_amr_clean.ps1 run.py
