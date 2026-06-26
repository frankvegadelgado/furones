# Removes the generated CAR bundle (JSON/CSV/report) from this car/ folder so a
# fresh run regenerates everything. Scripts and README.md are kept.
#
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File "D:\Papers\gfjournal_furones\car\clean_generated_bundle.ps1"

$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path

$generated = @(
    "CAR-001-ratio-constant.json",
    "CAR-002-strategy-ablation.json",
    "CAR-003-high-degree.json",
    "CAR-004-adversarial.json",
    "CAR.furones-v0.3.4.json",
    "CAR.furones-v0.3.5.json",
    "CAR.furones-v0.3.6.json",
    "car_benchmark_cases.csv",
    "family_summary.csv",
    "strategy_ablation_by_instance.csv",
    "strategy_ablation_summary.csv",
    "high_degree_by_instance.csv",
    "adversarial_by_instance.csv",
    "INTEGRITY_REPORT.md"
)

foreach ($name in $generated) {
    $path = Join-Path $dir $name
    if (Test-Path $path) {
        Remove-Item -Force $path
        Write-Host "deleted: $name"
    } else {
        Write-Host "skip (absent): $name"
    }
}

Write-Host ""
Write-Host "Remaining files in car/:"
Get-ChildItem -File $dir | Select-Object -ExpandProperty Name | Sort-Object | ForEach-Object { Write-Host "  $_" }
