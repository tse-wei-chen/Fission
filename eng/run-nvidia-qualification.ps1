param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $ModelPath,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $TokenizerPath,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $ModelId,

    [ValidateSet("fp32", "fp16")]
    [string] $ModelPrecision = "fp16",

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 2147483647)]
    [int] $NumHiddenLayers,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 2147483647)]
    [int] $NumKvHeads,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 2147483647)]
    [int] $HeadDim,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 2147483647)]
    [int] $VocabularySize,

    [ValidateNotNull()]
    [string] $EosTokenIds = "",

    [ValidateNotNull()]
    [string] $CudaRuntimeLibraryPath = "",

    [ValidateNotNull()]
    [string] $SampledTokenIdsOutput = "fission_sampled_token_ids",

    [ValidateRange(0, 2147483647)]
    [int] $CudaDeviceId = 0,

    [ValidateNotNullOrEmpty()]
    [string] $OutputDirectory = "artifacts/serving-gpu",

    [ValidateNotNullOrEmpty()]
    [string] $Label = "fission-cuda-fp16-qualification",

    [ValidateRange(1, 1000)]
    [int] $Repetitions = 3,

    [ValidateRange(100, 10000)]
    [int] $GpuTelemetryIntervalMilliseconds = 500,

    [switch] $GpuTelemetry,

    [switch] $NoBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$benchmarkRunner = Join-Path $repositoryRoot "eng/run-nvidia-serving-benchmark.ps1"
$resultValidator = Join-Path $repositoryRoot "eng/validate-serving-suite-results.ps1"
$manifest = Join-Path $repositoryRoot "benchmarks/serving/workloads.gpu-qualification.json"

foreach ($required in @($benchmarkRunner, $resultValidator, $manifest)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required qualification file '$required' does not exist."
    }
}

$outputRoot = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory
} else {
    Join-Path $repositoryRoot $OutputDirectory
}
$outputRoot = [System.IO.Path]::GetFullPath($outputRoot)
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$startedAt = [DateTimeOffset]::UtcNow
$telemetryJob = $null
$telemetryStopPath = Join-Path $outputRoot ".qualification-telemetry-$([Guid]::NewGuid().ToString('N')).stop"
$telemetryTempPath = Join-Path $outputRoot ".qualification-telemetry-$([Guid]::NewGuid().ToString('N')).csv"

if ($GpuTelemetry) {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        throw "nvidia-smi was not found; GPU telemetry requires a working NVIDIA driver."
    }

    $telemetryJob = Start-Job -ScriptBlock {
        param(
            [string] $NvidiaSmiPath,
            [int] $DeviceId,
            [string] $OutputPath,
            [string] $StopPath,
            [int] $IntervalMilliseconds
        )

        $ErrorActionPreference = "Continue"
        "timestamp_utc,memory_used_mib,memory_total_mib,gpu_utilization_percent,memory_utilization_percent,temperature_c" |
            Set-Content -LiteralPath $OutputPath -Encoding UTF8

        while (-not (Test-Path -LiteralPath $StopPath)) {
            $rows = @(
                & $NvidiaSmiPath "--id=$DeviceId" "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu" "--format=csv,noheader,nounits" 2>$null
            )
            if ($LASTEXITCODE -eq 0 -and $rows.Count -eq 1) {
                $parts = @([string] $rows[0] -split ",\s*", 5)
                if ($parts.Count -eq 5) {
                    $timestamp = [DateTimeOffset]::UtcNow.ToString("O")
                    "$timestamp,$($parts[0]),$($parts[1]),$($parts[2]),$($parts[3]),$($parts[4])" |
                        Add-Content -LiteralPath $OutputPath -Encoding UTF8
                }
            }

            Start-Sleep -Milliseconds $IntervalMilliseconds
        }
    } -ArgumentList @(
        $nvidiaSmi.Source,
        $CudaDeviceId,
        $telemetryTempPath,
        $telemetryStopPath,
        $GpuTelemetryIntervalMilliseconds)
}

$benchmarkSucceeded = $false
try {
    $runnerArguments = @{
        ModelPath = $ModelPath
        TokenizerPath = $TokenizerPath
        ModelId = $ModelId
        ModelPrecision = $ModelPrecision
        NumHiddenLayers = $NumHiddenLayers
        NumKvHeads = $NumKvHeads
        HeadDim = $HeadDim
        VocabularySize = $VocabularySize
        EosTokenIds = $EosTokenIds
        CudaDeviceId = $CudaDeviceId
        CudaRuntimeLibraryPath = $CudaRuntimeLibraryPath
        SampledTokenIdsOutput = $SampledTokenIdsOutput
        Manifest = $manifest
        OutputDirectory = $outputRoot
        Label = $Label
        Repetitions = $Repetitions
    }
    if ($NoBuild) {
        $runnerArguments["NoBuild"] = $true
    }

    & $benchmarkRunner @runnerArguments
    $benchmarkSucceeded = $true
}
finally {
    if ($null -ne $telemetryJob) {
        New-Item -ItemType File -Force -Path $telemetryStopPath | Out-Null
        Wait-Job -Job $telemetryJob -Timeout 15 | Out-Null
        Receive-Job -Job $telemetryJob -ErrorAction SilentlyContinue | Out-Null
        Remove-Job -Job $telemetryJob -Force -ErrorAction SilentlyContinue
        $telemetryJob = $null
    }
    Remove-Item -LiteralPath $telemetryStopPath -Force -ErrorAction SilentlyContinue
}

if (-not $benchmarkSucceeded) {
    throw "NVIDIA qualification benchmark did not complete."
}

$candidates = @(
    Get-ChildItem -LiteralPath $outputRoot -Directory |
        Where-Object {
            $environmentPath = Join-Path $_.FullName "environment.json"
            if (-not (Test-Path -LiteralPath $environmentPath -PathType Leaf)) {
                return $false
            }

            try {
                $environment = Get-Content -LiteralPath $environmentPath -Raw | ConvertFrom-Json
                $captured = [DateTimeOffset]::Parse([string] $environment.captured_at_utc)
                return [string] $environment.benchmark.label -eq $Label -and
                    $captured -ge $startedAt.AddSeconds(-5)
            }
            catch {
                return $false
            }
        } |
        Sort-Object LastWriteTimeUtc -Descending
)
if ($candidates.Count -eq 0) {
    throw "Could not locate the timestamped qualification run directory under '$outputRoot'."
}

$runRoot = $candidates[0].FullName
$resultsRoot = Join-Path $runRoot "results"
$validatorArguments = @{
    ManifestPath = $manifest
    ResultsRoot = $resultsRoot
    Label = $Label
    Repetitions = $Repetitions
    Model = $ModelId
}
& $resultValidator @validatorArguments

if ($GpuTelemetry) {
    if (-not (Test-Path -LiteralPath $telemetryTempPath -PathType Leaf)) {
        throw "GPU telemetry was requested but no telemetry CSV was produced."
    }

    $telemetryCsv = Join-Path $runRoot "gpu-telemetry.csv"
    Move-Item -LiteralPath $telemetryTempPath -Destination $telemetryCsv -Force
    $samples = @(Import-Csv -LiteralPath $telemetryCsv)
    if ($samples.Count -eq 0) {
        throw "GPU telemetry was requested but contained no samples."
    }

    $memoryUsed = @($samples | ForEach-Object { [double] $_.memory_used_mib })
    $memoryTotal = [double] $samples[0].memory_total_mib
    $gpuUtilization = @($samples | ForEach-Object { [double] $_.gpu_utilization_percent })
    $memoryUtilization = @($samples | ForEach-Object { [double] $_.memory_utilization_percent })
    $temperature = @($samples | ForEach-Object { [double] $_.temperature_c })

    $peakMemory = [double] (($memoryUsed | Measure-Object -Maximum).Maximum)
    $summary = [ordered]@{
        schema_version = 1
        sample_count = $samples.Count
        interval_milliseconds = $GpuTelemetryIntervalMilliseconds
        memory_total_mib = $memoryTotal
        memory_used_min_mib = [double] (($memoryUsed | Measure-Object -Minimum).Minimum)
        memory_used_max_mib = $peakMemory
        memory_headroom_at_peak_mib = $memoryTotal - $peakMemory
        gpu_utilization_max_percent = [double] (($gpuUtilization | Measure-Object -Maximum).Maximum)
        memory_utilization_max_percent = [double] (($memoryUtilization | Measure-Object -Maximum).Maximum)
        temperature_max_c = [double] (($temperature | Measure-Object -Maximum).Maximum)
        csv_path = $telemetryCsv
    }

    $summaryPath = Join-Path $runRoot "gpu-telemetry-summary.json"
    $summary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
    Write-Host ""
    Write-Host "GPU telemetry"
    Write-Host "  samples:       $($samples.Count)"
    Write-Host "  peak VRAM:     $("{0:F0}" -f $peakMemory) / $("{0:F0}" -f $memoryTotal) MiB"
    Write-Host "  peak headroom: $("{0:F0}" -f ($memoryTotal - $peakMemory)) MiB"
    Write-Host "  peak GPU util: $("{0:F0}" -f $summary.gpu_utilization_max_percent)%"
    Write-Host "  summary:       $summaryPath"
}
else {
    Remove-Item -LiteralPath $telemetryTempPath -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "NVIDIA production qualification passed completeness checks."
Write-Host "  run: $runRoot"
