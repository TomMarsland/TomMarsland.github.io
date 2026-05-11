param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsToPython
)

$python = "D:\Users\tomar\Anaconda\envs\AMR_clean\python.exe"

if (-not (Test-Path $python)) {
    throw "AMR_clean Python was not found at $python"
}

if (-not $ArgsToPython -or $ArgsToPython.Count -eq 0) {
    Write-Host "Usage examples:"
    Write-Host "  .\use_amr_clean.ps1 run.py"
    Write-Host "  .\use_amr_clean.ps1 train_sac.py --total-steps 150000 --wind on --output models\sac_controller.pt"
    Write-Host "  .\use_amr_clean.ps1 evaluate_sac.py --episodes 10 --policy models\sac_controller.pt"
    exit 0
}

& $python @ArgsToPython
