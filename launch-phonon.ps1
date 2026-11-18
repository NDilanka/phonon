#Requires -Version 7.0
# GMI correction / CPU ASR by default when GMI_API_KEY is present. Local GPU
# correction remains available with PHONON_WIN_CORRECTION_BACKEND=local.
# Examples: .\launch-phonon.ps1; .\launch-phonon.ps1 selftest
# .\launch-phonon.ps1 bench --repeat 3 'C:\fixtures\dictation.wav'
param([switch] $Cpu)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$phononArgs = @($args)
$root = $PSScriptRoot
$profileRoot = Join-Path $root 'research\laptop'
$cpuLauncher = Join-Path $root 'launch-phonon-cpu.ps1'
$process = $null
$exitCode = 1

function Invoke-LocalProbe(
    [string] $File,
    [string[]] $ProbeArgs,
    [Diagnostics.ProcessStartInfo] $Profile
) {
    $info = [Diagnostics.ProcessStartInfo]::new()
    $info.FileName = $File
    $info.WorkingDirectory = [IO.Path]::GetDirectoryName($File)
    $info.UseShellExecute = $false
    $info.CreateNoWindow = $true
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.Environment.Clear()
    foreach ($entry in $Profile.Environment.GetEnumerator()) {
        $info.Environment[$entry.Key] = $entry.Value
    }
    foreach ($argument in $ProbeArgs) { $info.ArgumentList.Add($argument) }
    $probe = $null
    try {
        $probe = [Diagnostics.Process]::Start($info)
        if ($null -eq $probe) { throw "Could not start $File" }
        $stdout = $probe.StandardOutput.ReadToEndAsync()
        $stderr = $probe.StandardError.ReadToEndAsync()
        if (-not $probe.WaitForExit(15000)) {
            $probe.Kill($true)
            $probe.WaitForExit()
            throw "Probe timed out after 15 seconds: $File"
        }
        $text = $stdout.GetAwaiter().GetResult() + "`n" + $stderr.GetAwaiter().GetResult()
        if ($probe.ExitCode -ne 0) {
            $tail = $text.Substring([Math]::Max(0, $text.Length - 4096)).Trim()
            throw "$File exited with $($probe.ExitCode): $tail"
        }
        return $text
    } finally {
        if ($null -ne $probe) { $probe.Dispose() }
    }
}

try {
    if (-not $IsWindows) { throw 'These launchers require Windows and PowerShell 7.' }
    $executable = [IO.Path]::Combine($root, 'target', 'release', 'phonon-win.exe')
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Missing release executable: $executable"
    }
    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $executable
    $startInfo.WorkingDirectory = $root
    $startInfo.UseShellExecute = $false

    # Child-only reset; preserve PATH, PHONON_WIN_HOTKEY and the idle-trim opt-out.
    $resetPattern = '^(LLAMA_|GGML_|CUDA_|CUBLAS_|CUDNN_|NVIDIA_|ORT_|SHERPA_ONNX_|OMP_|GOMP_|KMP_|MKL_|OPENBLAS_|PHONON_WIN_(ASR_|CORRECTION_THREADS$|LLAMA_))'
    foreach ($name in @($startInfo.Environment.Keys)) {
        if ($name -match $resetPattern -or $name -in @('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY')) {
            [void] $startInfo.Environment.Remove($name)
        }
    }
    $startInfo.Environment['PHONON_WIN_HOME'] = Join-Path $profileRoot 'data'
    $startInfo.Environment['NO_PROXY'] = '127.0.0.1,localhost,::1'
    $startInfo.Environment['PHONON_WIN_ASR_PROVIDER'] = 'cpu'
    if (-not $startInfo.Environment.ContainsKey('GMI_API_KEY')) {
        $userKey = [Environment]::GetEnvironmentVariable('GMI_API_KEY', 'User')
        if (-not [string]::IsNullOrWhiteSpace($userKey)) {
            $startInfo.Environment['GMI_API_KEY'] = $userKey
        }
    }
    $backend = $startInfo.Environment['PHONON_WIN_CORRECTION_BACKEND']
    if ([string]::IsNullOrWhiteSpace($backend)) {
        $backend = if ($startInfo.Environment.ContainsKey('GMI_API_KEY')) { 'gmi' } else { 'local' }
    }
    $startInfo.Environment['PHONON_WIN_CORRECTION_BACKEND'] = $backend

    if ($Cpu -or $backend -in @('gmi', 'cloud')) {
        # GMI correction has no resident local llama server. ASR stays CPU so
        # cloud mode does not require CUDA or consume GPU memory.
        $startInfo.Environment['LLAMA_ARG_N_GPU_LAYERS'] = '0'
    } else {
        # Local correction profile: validate the pinned CUDA runtime before
        # launching, so a missing GPU never silently changes the backend.
        $runtimeRoot = Join-Path $profileRoot 'cuda-b10726'
        $server = Join-Path $runtimeRoot 'llama-server.exe'
        $profile = @{
            PHONON_WIN_LLAMA_SERVER = $server
            PHONON_WIN_CORRECTION_THREADS = '4'
            PHONON_WIN_ASR_THREADS = '4'
            LLAMA_ARG_N_GPU_LAYERS = 'all'
            LLAMA_ARG_N_PARALLEL = '1'
            LLAMA_ARG_CACHE_RAM = '0'
            LLAMA_ARG_CTX_CHECKPOINTS = '0'
            LLAMA_ARG_BATCH = '256'
            LLAMA_ARG_UBATCH = '64'
            LLAMA_ARG_LAZY_MODE = 'off'
            LLAMA_ARG_CACHE_TYPE_K = 'f16'
            LLAMA_ARG_CACHE_TYPE_V = 'f16'
            LLAMA_ARG_FIT = 'off'
        }
        foreach ($entry in $profile.GetEnumerator()) {
            $startInfo.Environment[$entry.Key] = $entry.Value
        }

        $required = @(
            'llama-server.exe', 'llama-server-impl.dll', 'llama-common.dll', 'llama.dll',
            'mtmd.dll', 'ggml.dll', 'ggml-base.dll', 'ggml-cpu-x64.dll', 'ggml-cuda.dll',
            'libomp.dll', 'cublas64_12.dll', 'cublasLt64_12.dll', 'cudart64_12.dll'
        )
        foreach ($name in $required) {
            $file = Join-Path $runtimeRoot $name
            if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
                throw "Missing adjacent CUDA runtime file: $file"
            }
        }
        $smi = Get-Command nvidia-smi.exe -CommandType Application -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($null -eq $smi) { throw 'nvidia-smi.exe is unavailable.' }
        $gpus = Invoke-LocalProbe -File $smi.Source -ProbeArgs @('--query-gpu=name', '--format=csv,noheader') -Profile $startInfo
        if ($gpus -notmatch '\bRTX\b') { throw 'nvidia-smi reported no available RTX GPU.' }
        $version = Invoke-LocalProbe -File $server -ProbeArgs @('--version') -Profile $startInfo
        if ($version -notmatch '(?m)^[ \t]*version:\s*(?:b?10726(?:\s|$)|[^\r\n]*\(build 10726,)') {
            throw 'The CUDA runtime did not report pinned llama.cpp build b10726.'
        }
        $devices = Invoke-LocalProbe -File $server -ProbeArgs @('--list-devices') -Profile $startInfo
        $device = [regex]::Match($devices, '(?im)^[ \t]*(CUDA[0-9]+):[^\r\n]*\bRTX\b')
        if (-not $device.Success) { throw "llama.cpp exposed no RTX CUDA device: $($devices.Trim())" }
        # A missing selected device is fatal in b10726, not a CPU fallback.
        $startInfo.Environment['LLAMA_ARG_DEVICE'] = $device.Groups[1].Value
    }

    foreach ($argument in $phononArgs) { $startInfo.ArgumentList.Add([string] $argument) }
    $process = [Diagnostics.Process]::Start($startInfo)
    if ($null -eq $process) { throw 'Windows did not return a Phonon process.' }
    $process.WaitForExit()
    $exitCode = $process.ExitCode
} catch {
    [Console]::Error.WriteLine("Phonon launcher: {0}", $_.Exception.Message)
    if (-not $Cpu) {
        [Console]::Error.WriteLine("CPU fallback: pwsh -File `"{0}`"", $cpuLauncher)
    }
} finally {
    if ($null -ne $process) { $process.Dispose() }
}

exit $exitCode
