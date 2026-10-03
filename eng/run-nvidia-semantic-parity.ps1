param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $BaselineModelPath,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $CandidateModelPath,

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

    [ValidateNotNullOrEmpty()]
    [string] $CandidateSampledTokenIdsOutput = "fission_sampled_token_ids",

    [ValidateNotNullOrEmpty()]
    [string] $Corpus = "benchmarks/serving/semantic-parity.json",

    [ValidateNotNullOrEmpty()]
    [string] $OutputDirectory = "artifacts/semantic-parity-gpu",

    [ValidateRange(1, 65535)]
    [int] $BaselinePort = 18080,

    [ValidateRange(1, 65535)]
    [int] $CandidatePort = 18081,

    [ValidateRange(1, 3600)]
    [int] $ServerReadyTimeoutSeconds = 180,

    [ValidateRange(1, 300)]
    [int] $GracefulShutdownTimeoutSeconds = 30,

    [ValidateNotNullOrEmpty()]
    [string] $Configuration = "Release",

    [switch] $RequireExactTokens,

    [switch] $NoBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-RequiredFile {
    param([string] $Path, [string] $Label)
    $candidate = if ([System.IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path (Get-Location) $Path }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "$Label '$candidate' does not exist."
    }
    return (Resolve-Path -LiteralPath $candidate).Path
}

function Show-ServerLogs {
    param([object] $Server)
    foreach ($path in @($Server.StdOutPath, $Server.StdErrPath)) {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            Write-Host ""
            Write-Host "==> $path"
            Get-Content -LiteralPath $path -Tail 120
        }
    }
}

function Start-FissionServer {
    param(
        [string] $Name,
        [string] $BaseUrl,
        [string] $ControlToken,
        [string] $ModelPath,
        [string] $Precision,
        [bool] $PageLockedDecodeLogits,
        [string] $SampledTokenIdsOutput,
        [string] $StdOutPath,
        [string] $StdErrPath,
        [string] $RepositoryRoot,
        [string] $Tokenizer,
        [string] $CudaRuntime,
        [string] $LibrarySearchVariable,
        [string] $LibrarySearchValue
    )

    $settings = [ordered]@{
        "ASPNETCORE_URLS" = $BaseUrl
        "Fission__Backend" = "onnx"
        "Fission__ExecutionProvider" = "cuda"
        "Fission__Device" = "cuda:$CudaDeviceId"
        "Fission__CudaDeviceId" = "$CudaDeviceId"
        "Fission__CudaRuntimeLibraryPath" = $CudaRuntime
        "Fission__CudaPageLockedDecodeLogits" = if ($PageLockedDecodeLogits) { "true" } else { "false" }
        "Fission__SampledTokenIdsOutput" = $SampledTokenIdsOutput
        "Fission__ModelPrecision" = $Precision
        "Fission__ControlToken" = $ControlToken
        "Fission__ModelPath" = $ModelPath
        "Fission__ModelId" = $ModelId
        "Fission__NumHiddenLayers" = "$NumHiddenLayers"
        "Fission__NumKvHeads" = "$NumKvHeads"
        "Fission__HeadDim" = "$HeadDim"
        "Fission__VocabularySize" = "$VocabularySize"
        "Fission__EosTokenIds" = $EosTokenIds
        "Fission__Tokenizer" = "huggingface"
        "Fission__TokenizerPath" = $Tokenizer
        "Fission__ChatTemplate" = $ChatTemplate
        "Fission__StartupProbeEnabled" = "true"
        "Fission__StartupProbeExitAfterSuccess" = "false"
        "Fission__StartupProbePrompt" = "Hello"
        "Fission__StartupProbeModelId" = $ModelId
        "Fission__StartupProbeMaxTokens" = "1"
        "Fission__StartupProbeTimeoutSeconds" = "120"
    }
    if (-not [string]::IsNullOrWhiteSpace($LibrarySearchVariable)) {
        $settings[$LibrarySearchVariable] = $LibrarySearchValue
    }

    $previous = @{}
    try {
        foreach ($entry in $settings.GetEnumerator()) {
            $previous[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, [EnvironmentVariableTarget]::Process)
            [Environment]::SetEnvironmentVariable($entry.Key, [string] $entry.Value, [EnvironmentVariableTarget]::Process)
        }

        $process = Start-Process -FilePath "dotnet" -ArgumentList @(
            "run", "--project", "src/Fission.Server/Fission.Server.csproj", "-c", $Configuration, "--no-build"
        ) -WorkingDirectory $RepositoryRoot -RedirectStandardOutput $StdOutPath -RedirectStandardError $StdErrPath -PassThru
    }
    finally {
        foreach ($entry in $settings.GetEnumerator()) {
            [Environment]::SetEnvironmentVariable($entry.Key, $previous[$entry.Key], [EnvironmentVariableTarget]::Process)
        }
    }

    return [pscustomobject]@{
        Name = $Name
        BaseUrl = $BaseUrl
        ControlToken = $ControlToken
        Process = $process
        StdOutPath = $StdOutPath
        StdErrPath = $StdErrPath
    }
}

function Wait-ServerReady {
    param([object] $Server)
    $deadline = [DateTimeOffset]::UtcNow.AddSeconds($ServerReadyTimeoutSeconds)
    while ([DateTimeOffset]::UtcNow -lt $deadline) {
        if ($Server.Process.HasExited) {
            Show-ServerLogs -Server $Server
            throw "$($Server.Name) exited before becoming ready with code $($Server.Process.ExitCode)."
        }
        try {
            $response = Invoke-WebRequest -Uri "$($Server.BaseUrl)/healthz" -UseBasicParsing -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                return
            }
        }
        catch {
        }
        Start-Sleep -Seconds 1
    }
    Show-ServerLogs -Server $Server
    throw "$($Server.Name) did not become ready within $ServerReadyTimeoutSeconds second(s)."
}

function Stop-ServerBestEffort {
    param([object] $Server)
    if ($null -eq $Server -or $Server.Process.HasExited) {
        return
    }
    try {
        $response = Invoke-WebRequest -Method Post -Uri "$($Server.BaseUrl)/internal/control/shutdown" `
            -Headers @{ "X-Fission-Control-Token" = $Server.ControlToken } -UseBasicParsing -TimeoutSec 10
        if ($response.StatusCode -eq 202) {
            $deadline = [DateTimeOffset]::UtcNow.AddSeconds($GracefulShutdownTimeoutSeconds)
            while (-not $Server.Process.HasExited -and [DateTimeOffset]::UtcNow -lt $deadline) {
                Start-Sleep -Milliseconds 100
            }
        }
    }
    catch {
        Write-Warning "Graceful shutdown failed for $($Server.Name): $($_.Exception.Message)"
    }
    if (-not $Server.Process.HasExited) {
        Stop-Process -Id $Server.Process.Id -Force -ErrorAction SilentlyContinue
    }
    $Server.Process.WaitForExit()
}

if ($BaselinePort -eq $CandidatePort) {
    throw "BaselinePort and CandidatePort must differ."
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$serverProject = Join-Path $repositoryRoot "src/Fission.Server/Fission.Server.csproj"
$comparator = Join-Path $repositoryRoot "eng/compare-serving-semantic-parity.ps1"
$baselineModel = Resolve-RequiredFile -Path $BaselineModelPath -Label "FP32 baseline model"
$candidateModel = Resolve-RequiredFile -Path $CandidateModelPath -Label "FP16 candidate model"
$tokenizer = Resolve-RequiredFile -Path $TokenizerPath -Label "Tokenizer"
$corpusPath = Resolve-RequiredFile -Path (Join-Path $repositoryRoot $Corpus) -Label "Semantic parity corpus"
$cudaRuntime = if ([string]::IsNullOrWhiteSpace($CudaRuntimeLibraryPath)) { "" } else { Resolve-RequiredFile -Path $CudaRuntimeLibraryPath -Label "CUDA runtime library" }

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi was not found. Run this gate on an NVIDIA host."
}
if (-not $NoBuild) {
    & dotnet build $serverProject -c $Configuration
    if ($LASTEXITCODE -ne 0) { throw "Failed to build Fission.Server." }
}

$cudaDirectory = if ([string]::IsNullOrWhiteSpace($cudaRuntime)) { "" } else { Split-Path -Parent $cudaRuntime }
$librarySearchVariable = ""
$librarySearchValue = ""
if (-not [string]::IsNullOrWhiteSpace($cudaDirectory)) {
    $isWindows = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
    $librarySearchVariable = if ($isWindows) { "PATH" } else { "LD_LIBRARY_PATH" }
    $existing = [Environment]::GetEnvironmentVariable($librarySearchVariable, [EnvironmentVariableTarget]::Process)
    $librarySearchValue = if ([string]::IsNullOrWhiteSpace($existing)) { $cudaDirectory } else { "$cudaDirectory$([System.IO.Path]::PathSeparator)$existing" }
}

$outputRoot = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else { Join-Path $repositoryRoot $OutputDirectory }
$runRoot = Join-Path ([System.IO.Path]::GetFullPath($outputRoot)) ([DateTimeOffset]::Now.ToString("yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
$reportJson = Join-Path $runRoot "report.json"
$reportMarkdown = Join-Path $runRoot "report.md"
$metadataPath = Join-Path $runRoot "environment.json"

$baseline = $null
$candidate = $null
$metadata = [ordered]@{
    schema_version = 1
    status = "starting"
    generated_at_utc = [DateTimeOffset]::UtcNow.ToString("O")
    model = $ModelId
    device = "cuda:$CudaDeviceId"
    baseline = [ordered]@{ precision = "fp32"; model_path = $baselineModel; port = $BaselinePort; page_locked_decode_logits = $true }
    candidate = [ordered]@{ precision = "fp16"; model_path = $candidateModel; port = $CandidatePort; sampled_token_ids_output = $CandidateSampledTokenIdsOutput }
    tokenizer_path = $tokenizer
    corpus = $corpusPath
}
$metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $metadataPath -Encoding UTF8

try {
    $baseline = Start-FissionServer -Name "FP32 baseline" -BaseUrl "http://127.0.0.1:$BaselinePort" `
        -ControlToken ([Guid]::NewGuid().ToString("N")) -ModelPath $baselineModel -Precision "fp32" `
        -PageLockedDecodeLogits $true -SampledTokenIdsOutput "" `
        -StdOutPath (Join-Path $runRoot "baseline.stdout.log") -StdErrPath (Join-Path $runRoot "baseline.stderr.log") `
        -RepositoryRoot $repositoryRoot -Tokenizer $tokenizer -CudaRuntime $cudaRuntime `
        -LibrarySearchVariable $librarySearchVariable -LibrarySearchValue $librarySearchValue
    Wait-ServerReady -Server $baseline

    $candidate = Start-FissionServer -Name "FP16 candidate" -BaseUrl "http://127.0.0.1:$CandidatePort" `
        -ControlToken ([Guid]::NewGuid().ToString("N")) -ModelPath $candidateModel -Precision "fp16" `
        -PageLockedDecodeLogits $false -SampledTokenIdsOutput $CandidateSampledTokenIdsOutput `
        -StdOutPath (Join-Path $runRoot "candidate.stdout.log") -StdErrPath (Join-Path $runRoot "candidate.stderr.log") `
        -RepositoryRoot $repositoryRoot -Tokenizer $tokenizer -CudaRuntime $cudaRuntime `
        -LibrarySearchVariable $librarySearchVariable -LibrarySearchValue $librarySearchValue
    Wait-ServerReady -Server $candidate

    $compareArgs = @{
        BaselineBaseUrl = $baseline.BaseUrl
        CandidateBaseUrl = $candidate.BaseUrl
        BaselineControlToken = $baseline.ControlToken
        CandidateControlToken = $candidate.ControlToken
        ModelId = $ModelId
        Corpus = $corpusPath
        OutputJson = $reportJson
        OutputMarkdown = $reportMarkdown
    }
    if ($RequireExactTokens) {
        $compareArgs["RequireExactTokens"] = $true
    }

    & $comparator @compareArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Semantic parity comparator failed with exit code $LASTEXITCODE."
    }

    $metadata["status"] = "passed"
    $metadata["completed_at_utc"] = [DateTimeOffset]::UtcNow.ToString("O")
    $metadata["report_json"] = $reportJson
    $metadata["report_markdown"] = $reportMarkdown
    $metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $metadataPath -Encoding UTF8

    Write-Host ""
    Write-Host "NVIDIA semantic parity run completed."
    Write-Host "  report: $reportMarkdown"
    Write-Host "  json:   $reportJson"
}
catch {
    $metadata["status"] = "failed"
    $metadata["completed_at_utc"] = [DateTimeOffset]::UtcNow.ToString("O")
    $metadata["failure"] = $_.Exception.Message
    $metadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $metadataPath -Encoding UTF8
    if ($null -ne $baseline) { Show-ServerLogs -Server $baseline }
    if ($null -ne $candidate) { Show-ServerLogs -Server $candidate }
    throw
}
finally {
    Stop-ServerBestEffort -Server $candidate
    Stop-ServerBestEffort -Server $baseline
}
