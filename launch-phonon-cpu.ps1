#Requires -Version 7.0
# Bundled CPU fallback; isolated data, hotkey and PHONON_WIN_TRIM_IDLE are preserved.
# Examples: .\launch-phonon-cpu.ps1; .\launch-phonon-cpu.ps1 selftest
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'launch-phonon.ps1') -Cpu @args
exit $LASTEXITCODE
