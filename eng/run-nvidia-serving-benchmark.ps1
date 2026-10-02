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

    [ValidateSet("none", "chatml", "qwen2", "llama3")]
    [string] $ChatTemplate = "none",

    [ValidateRange(0, 2147483647)]
    [int] $CudaDeviceId = 0,

    [ValidateNotNull()]
    [string] $CudaRuntimeLibraryPath = "",

    [ValidateNotNull()]
    [string] $SampledTokenIdsOutput = "",

    [ValidateNotNullOrEmpty()]
    [string] $StartupProbePrompt = "Hello",

    [ValidateRange(1, 4096)]
    [int] $StartupProbeMaxTokens = 1,

    [ValidateRange(1, 3600)]
    [int] $StartupProbeTimeoutSeconds = 120,

    [ValidateNotNullOrEmpty()]
    [string] $Manifest = "benchmarks/serving/workloads.gpu-smoke.json",

    [ValidateNotNullOrEmpty()]
    [string] $OutputDirectory = "artifacts/serving-gpu",

    [ValidateNotNullOrEmpty()]
    [string] $Label = "fission-cuda",

    [ValidateRange(1, 1000)]
    [int] $Repetitions = 1,

    [ValidateRange(1, 65535)]
    [int] $Port = 18080,

    [ValidateRange(1, 3600)]
    [int] $ServerReadyTimeoutSeconds = 180,

    [ValidateNotNullOrEmpty()]
    [string] $Configuration = "Release",

    [ValidateRange(1, 300)]
    [int] $GracefulShutdownTimeoutSeconds = 30,

    [switch] $PageLockedDecodeLogits,

    [switch] $OrtProfile,

    [switch] $NoBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-RequiredFile {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Path,

        [Parameter(Mandatory = $true)]
        [string] $Label
    )

    $candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        $Path
    } else {
        Join-Path (Get-Location) $Path
    }

    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "$Label '$candidate' does not exist."
    }

    return (Resolve-Path -LiteralPath $candidate).Path
}

function Resolve-RepositoryFile {
    param(
        [Parameter(Mandatory = $true)]
        [string] $RepositoryRoot,

        [Parameter(Mandatory = $true)]
        [string] $Path,

        [Parameter(Mandatory = $true)]
        [string] $Label
    )

    $candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        $Path
    } else {
        Join-Path $RepositoryRoot $Path
    }

    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "$Label '$candidate' does not exist."
    }

    return (Resolve-Path -LiteralPath $candidate).Path
}

function Show-ServerLogs {
    param(
        [Parameter(Mandatory = $true)]
        [string] $StdOutPath,

        [Parameter(Mandatory = $true)]
        [string] $StdErrPath
    )

    foreach ($path in @($StdOutPath, $StdErrPath)) {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            Write-Host ""
            Write-Host "==> $path"
            Get-Content -LiteralPath $path -Tail 120
        }
    }
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$serverProject = Join-Path $repositoryRoot "src/Fission.Server/Fission.Server.csproj"
$loadGenProject = Join-Path $repositoryRoot "benchmarks/Fission.Serving.LoadGen/Fission.Serving.LoadGen.csproj"
$reporterProject = Join-Path $repositoryRoot "benchmarks/Fission.Serving.Reporter/Fission.Serving.Reporter.csproj"
$backendProject = Join-Path $repositoryRoot "src/Fission.Backends.OnnxRuntime/Fission.Backends.OnnxRuntime.csproj"
$suiteRunner = Join-Path $repositoryRoot "eng/run-serving-suite.ps1"
$ortProfileSummarizer = Join-Path $repositoryRoot "eng/summarize-ort-profile.ps1"

$model = Resolve-RequiredFile -Path $ModelPath -Label "ONNX model"
$tokenizer = Resolve-RequiredFile -Path $TokenizerPath -Label "Tokenizer"
$manifestPath = Resolve-RepositoryFile -RepositoryRoot $repositoryRoot -Path $Manifest -Label "Benchmark manifest"
$cudaRuntimeLibrary = if ([string]::IsNullOrWhiteSpace($CudaRuntimeLibraryPath)) {
    ""
} else {
    Resolve-RequiredFile -Path $CudaRuntimeLibraryPath -Label "CUDA Runtime library"
}
$cudaLibraryDirectory = if ([string]::IsNullOrWhiteSpace($cudaRuntimeLibrary)) {
    $null
} else {
    Split-Path -Parent $cudaRuntimeLibrary
}

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($null -eq $nvidiaSmi) {
    throw "nvidia-smi was not found. Run this benchmark on an NVIDIA host with a working driver."
}

$inventoryLines = @(
    & $nvidiaSmi.Source "--query-gpu=index,name,driver_version,memory.total" "--format=csv,noheader,nounits"
)
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi failed with exit code $LASTEXITCODE."
}

$gpuInventory = @()
foreach ($line in $inventoryLines) {
    $parts = @($line -split ",\s*", 4)
    if ($parts.Count -ne 4) {
        throw "Unexpected nvidia-smi inventory row: '$line'."
    }

    $gpuInventory += [pscustomobject]@{
        index = [int] $parts[0]
        name = [string] $parts[1]
        driver_version = [string] $parts[2]
        memory_total_mib = [int] $parts[3]
    }
}

$selectedGpu = @($gpuInventory | Where-Object { $_.index -eq $CudaDeviceId })
if ($selectedGpu.Count -ne 1) {
    throw "CUDA device $CudaDeviceId was not found in nvidia-smi inventory."
}
$selectedGpu = $selectedGpu[0]

if (-not $NoBuild) {
    foreach ($project in @($serverProject, $loadGenProject, $reporterProject)) {
        & dotnet build $project -c $Configuration
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to build '$project'."
        }
    }
}

$outputRoot = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory
} else {
    Join-Path $repositoryRoot $OutputDirectory
}
$outputRoot = [System.IO.Path]::GetFullPath($outputRoot)

$runName = [DateTimeOffset]::Now.ToString("yyyyMMdd-HHmmss")
$runRoot = Join-Path $outputRoot $runName
$resultsRoot = Join-Path $runRoot "results"
New-Item -ItemType Directory -Force -Path $resultsRoot | Out-Null

$stdoutPath = Join-Path $runRoot "server.stdout.log"
$stderrPath = Join-Path $runRoot "server.stderr.log"
$metadataPath = Join-Path $runRoot "environment.json"
$reportMarkdown = Join-Path $runRoot "report.md"
$reportCsv = Join-Path $runRoot "report.csv"
$ortProfilePrefix = Join-Path $runRoot "ort-profile"
$ortProfileSummaryMarkdown = Join-Path $runRoot "ort-profile-summary.md"
$ortProfileSummaryJson = Join-Path $runRoot "ort-profile-summary.json"
$baseUrl = "http://127.0.0.1:$Port"
$controlToken = [Guid]::NewGuid().ToString("N")

$gitCommit = $null
$git = Get-Command git -ErrorAction SilentlyContinue
if ($null -ne $git) {
    $gitValue = & $git.Source -C $repositoryRoot rev-parse HEAD 2>$null
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($gitValue)) {
        $gitCommit = ([string] $gitValue).Trim()
    }
}

[xml] $backendProjectXml = Get-Content -LiteralPath $backendProject -Raw
$onnxPackage = @(
    $backendProjectXml.Project.ItemGroup.PackageReference |
        Where-Object { $_.Include -eq "Microsoft.ML.OnnxRuntime.Gpu" }
)
$onnxRuntimeVersion = if ($onnxPackage.Count -eq 1) {
    [string] $onnxPackage[0].Version
} else {
    $null
}

$dotnetVersion = (& dotnet --version).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "dotnet --version failed."
}

$metadata = [ordered]@{
    schema_version = 1
    status = "starting"
    captured_at_utc = [DateTimeOffset]::UtcNow.ToString("O")
    git_commit = $gitCommit
    provider = "cuda"
    device = [ordered]@{
        logical_id = "cuda:$CudaDeviceId"
        ordinal = $selectedGpu.index
        name = $selectedGpu.name
        driver_version = $selectedGpu.driver_version
        memory_total_mib = $selectedGpu.memory_total_mib
    }
    runtime = [ordered]@{
        dotnet_sdk = $dotnetVersion
        onnxruntime_gpu = $onnxRuntimeVersion
        cuda_runtime_library = if ([string]::IsNullOrWhiteSpace($cudaRuntimeLibrary)) { $null } else { $cudaRuntimeLibrary }
        cuda_library_search_directory = $cudaLibraryDirectory
        page_locked_decode_logits = [bool] $PageLockedDecodeLogits
        ort_profile_enabled = [bool] $OrtProfile
        ort_profile_prefix = if ($OrtProfile) { $ortProfilePrefix } else { $null }
    }
    model = [ordered]@{
        id = $ModelId
        model_path = $model
        tokenizer_path = $tokenizer
        num_hidden_layers = $NumHiddenLayers
        num_kv_heads = $NumKvHeads
        head_dim = $HeadDim
        vocabulary_size = $VocabularySize
        eos_token_ids = $EosTokenIds
        chat_template = $ChatTemplate
        sampled_token_ids_output = if ([string]::IsNullOrWhiteSpace($SampledTokenIdsOutput)) { $null } else { $SampledTokenIdsOutput }
    }
    startup_probe = [ordered]@{
        prompt = $StartupProbePrompt
        max_tokens = $StartupProbeMaxTokens
        timeout_seconds = $StartupProbeTimeoutSeconds
    }
    benchmark = [ordered]@{
        label = $Label
        manifest = $manifestPath
        repetitions = $Repetitions
        base_url = $baseUrl
    }
}
$metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $metadataPath -Encoding UTF8

$settings = [ordered]@{
    "ASPNETCORE_URLS" = $baseUrl
    "Fission__Backend" = "onnx"
    "Fission__ExecutionProvider" = "cuda"
    "Fission__Device" = "cuda:$CudaDeviceId"
    "Fission__CudaDeviceId" = "$CudaDeviceId"
    "Fission__CudaRuntimeLibraryPath" = $cudaRuntimeLibrary
    "Fission__CudaPageLockedDecodeLogits" = if ($PageLockedDecodeLogits) { "true" } else { "false" }
    "Fission__SampledTokenIdsOutput" = $SampledTokenIdsOutput
    "Fission__OrtProfileOutputPathPrefix" = if ($OrtProfile) { $ortProfilePrefix } else { "" }
    "Fission__ControlToken" = $controlToken
    "Fission__ModelPath" = $model
    "Fission__ModelId" = $ModelId
    "Fission__NumHiddenLayers" = "$NumHiddenLayers"
    "Fission__NumKvHeads" = "$NumKvHeads"
    "Fission__HeadDim" = "$HeadDim"
    "Fission__VocabularySize" = "$VocabularySize"
    "Fission__EosTokenIds" = $EosTokenIds
    "Fission__Tokenizer" = "huggingface"
    "Fission__TokenizerPath" = $tokenizer
    "Fission__ChatTemplate" = $ChatTemplate
    "Fission__StartupProbeEnabled" = "true"
    "Fission__StartupProbeExitAfterSuccess" = "false"
    "Fission__StartupProbePrompt" = $StartupProbePrompt
    "Fission__StartupProbeModelId" = $ModelId
    "Fission__StartupProbeMaxTokens" = "$StartupProbeMaxTokens"
    "Fission__StartupProbeTimeoutSeconds" = "$StartupProbeTimeoutSeconds"
}

if (-not [string]::IsNullOrWhiteSpace($cudaLibraryDirectory)) {
    $isWindowsPlatform = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
    $librarySearchVariable = if ($isWindowsPlatform) { "PATH" } else { "LD_LIBRARY_PATH" }
    $currentLibrarySearchPath = [Environment]::GetEnvironmentVariable(
        $librarySearchVariable,
        [EnvironmentVariableTarget]::Process)
    $settings[$librarySearchVariable] = if ([string]::IsNullOrWhiteSpace($currentLibrarySearchPath)) {
        $cudaLibraryDirectory
    } else {
        "$cudaLibraryDirectory$([System.IO.Path]::PathSeparator)$currentLibrarySearchPath"
    }
}

$previous = @{}
$serverProcess = $null
$startedAt = [DateTimeOffset]::UtcNow

try {
    foreach ($entry in $settings.GetEnumerator()) {
        $previous[$entry.Key] = [Environment]::GetEnvironmentVariable(
            $entry.Key,
            [EnvironmentVariableTarget]::Process)
        [Environment]::SetEnvironmentVariable(
            $entry.Key,
            [string] $entry.Value,
            [EnvironmentVariableTarget]::Process)
    }

    try {
        $serverProcess = Start-Process -FilePath "dotnet" -ArgumentList @(
            "run",
            "--project", "src/Fission.Server/Fission.Server.csproj",
            "-c", $Configuration,
            "--no-build"
        ) -WorkingDirectory $repositoryRoot -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    }
    finally {
        foreach ($entry in $settings.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable(
                $entry.Key,
                $previous[$entry.Key],
                [EnvironmentVariableTarget]::Process)
        }
    }

    Write-Host "Fission NVIDIA serving benchmark"
    Write-Host "  model:       $ModelId"
    Write-Host "  device:      cuda:$CudaDeviceId ($($selectedGpu.name))"
    Write-Host "  driver:      $($selectedGpu.driver_version)"
    Write-Host "  pinned logits: $([bool] $PageLockedDecodeLogits)"
    Write-Host "  ORT profile: $([bool] $OrtProfile)"
    Write-Host "  target:      $baseUrl"
    Write-Host "  manifest:    $manifestPath"
    Write-Host "  output:      $runRoot"
    Write-Host ""

    $deadline = [DateTimeOffset]::UtcNow.AddSeconds($ServerReadyTimeoutSeconds)
    $ready = $false

    while (-not $ready -and [DateTimeOffset]::UtcNow -lt $deadline) {
        if ($serverProcess.HasExited) {
            Show-ServerLogs -StdOutPath $stdoutPath -StdErrPath $stderrPath
            throw "Fission.Server exited before becoming ready with code $($serverProcess.ExitCode)."
        }

        try {
            $response = Invoke-WebRequest -Uri "$baseUrl/healthz" -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                $ready = $true
                break
            }
        }
        catch {
        }

        Start-Sleep -Seconds 1
    }

    if (-not $ready) {
        Show-ServerLogs -StdOutPath $stdoutPath -StdErrPath $stderrPath
        throw "Fission.Server did not become ready within $ServerReadyTimeoutSeconds second(s)."
    }

    $metadata["status"] = "server-ready"
    $metadata["server_ready_at_utc"] = [DateTimeOffset]::UtcNow.ToString("O")
    $metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $metadataPath -Encoding UTF8

    & $suiteRunner -Label $Label -BaseUrl $baseUrl -Model $ModelId -Manifest $manifestPath -OutputDirectory $resultsRoot -Repetitions $Repetitions -Configuration $Configuration -NoBuild
    if ($LASTEXITCODE -ne 0) {
        throw "Serving benchmark suite failed with exit code $LASTEXITCODE."
    }

    & dotnet run --project $reporterProject -c $Configuration --no-build -- --input $resultsRoot --markdown $reportMarkdown --csv $reportCsv
    if ($LASTEXITCODE -ne 0) {
        throw "Serving benchmark reporter failed with exit code $LASTEXITCODE."
    }

    $shutdownResponse = Invoke-WebRequest `
        -Method Post `
        -Uri "$baseUrl/internal/control/shutdown" `
        -Headers @{ "X-Fission-Control-Token" = $controlToken } `
        -UseBasicParsing `
        -TimeoutSec 10
    if ($shutdownResponse.StatusCode -ne 202) {
        throw "Fission.Server graceful shutdown returned HTTP $($shutdownResponse.StatusCode)."
    }

    $shutdownDeadline = [DateTimeOffset]::UtcNow.AddSeconds(
        $GracefulShutdownTimeoutSeconds)
    while (-not $serverProcess.HasExited -and
           [DateTimeOffset]::UtcNow -lt $shutdownDeadline) {
        Start-Sleep -Milliseconds 100
    }

    if (-not $serverProcess.HasExited) {
        throw "Fission.Server did not stop gracefully within $GracefulShutdownTimeoutSeconds second(s)."
    }

    $serverProcess.WaitForExit()
    if ($serverProcess.ExitCode -ne 0) {
        Show-ServerLogs -StdOutPath $stdoutPath -StdErrPath $stderrPath
        throw "Fission.Server exited with code $($serverProcess.ExitCode) during graceful shutdown."
    }

    if ($OrtProfile) {
        $profileFiles = @(
            Get-ChildItem -LiteralPath $runRoot -Filter "ort-profile*.json" -File |
                Sort-Object LastWriteTimeUtc -Descending
        )
        if ($profileFiles.Count -eq 0) {
            throw "ORT profiling was enabled but no profile JSON was produced under '$runRoot'."
        }

        $ortProfilePath = $profileFiles[0].FullName
        & $ortProfileSummarizer `
            -ProfilePath $ortProfilePath `
            -MarkdownPath $ortProfileSummaryMarkdown `
            -JsonPath $ortProfileSummaryJson
        if ($LASTEXITCODE -ne 0) {
            throw "ORT profile summarizer failed with exit code $LASTEXITCODE."
        }

        $metadata["ort_profile"] = [ordered]@{
            profile_path = $ortProfilePath
            summary_markdown = $ortProfileSummaryMarkdown
            summary_json = $ortProfileSummaryJson
        }
    }

    $metadata["status"] = "passed"
    $metadata["completed_at_utc"] = [DateTimeOffset]::UtcNow.ToString("O")
    $metadata["elapsed_seconds"] = ([DateTimeOffset]::UtcNow - $startedAt).TotalSeconds
    $metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $metadataPath -Encoding UTF8

    Write-Host ""
    Write-Host "Fission NVIDIA serving benchmark passed."
    Write-Host "  environment: $metadataPath"
    Write-Host "  report:      $reportMarkdown"
    Write-Host "  csv:         $reportCsv"
    if ($OrtProfile) {
        Write-Host "  ORT profile: $ortProfilePath"
        Write-Host "  ORT summary: $ortProfileSummaryMarkdown"
    }
    Write-Host ""
    Get-Content -LiteralPath $reportMarkdown
}
catch {
    $metadata["status"] = "failed"
    $metadata["completed_at_utc"] = [DateTimeOffset]::UtcNow.ToString("O")
    $metadata["failure"] = $_.Exception.Message
    $metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $metadataPath -Encoding UTF8
    Show-ServerLogs -StdOutPath $stdoutPath -StdErrPath $stderrPath
    throw
}
finally {
    if ($null -ne $serverProcess -and -not $serverProcess.HasExited) {
        Stop-Process -Id $serverProcess.Id -Force
        $serverProcess.WaitForExit()
    }
}
