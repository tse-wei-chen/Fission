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
    [ValidateRange(1, [int]::MaxValue)]
    [int] $NumHiddenLayers,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, [int]::MaxValue)]
    [int] $NumKvHeads,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, [int]::MaxValue)]
    [int] $HeadDim,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, [int]::MaxValue)]
    [int] $VocabularySize,

    [ValidateNotNull()]
    [string] $EosTokenIds = "",

    [ValidateSet("none", "chatml", "qwen2", "llama3")]
    [string] $ChatTemplate = "none",

    [ValidateRange(0, [int]::MaxValue)]
    [int] $CudaDeviceId = 0,

    [ValidateRange(1, 4096)]
    [int] $MaxTokens = 1,

    [ValidateRange(1, 3600)]
    [int] $TimeoutSeconds = 120,

    [ValidateNotNullOrEmpty()]
    [string] $Prompt = "Hello",

    [ValidateNotNull()]
    [string] $CudaRuntimeLibraryPath = "",

    [ValidateNotNullOrEmpty()]
    [string] $Configuration = "Release",

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

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$serverProject = Join-Path $repositoryRoot "src/Fission.Server/Fission.Server.csproj"
$model = Resolve-RequiredFile -Path $ModelPath -Label "ONNX model"
$tokenizer = Resolve-RequiredFile -Path $TokenizerPath -Label "Tokenizer"

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($null -eq $nvidiaSmi) {
    throw "nvidia-smi was not found. Run this smoke on an NVIDIA host with a working driver."
}

Write-Host "NVIDIA device inventory"
& $nvidiaSmi.Source "--query-gpu=index,name,driver_version,memory.total" "--format=csv,noheader"
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi failed with exit code $LASTEXITCODE."
}

if (-not $NoBuild) {
    & dotnet build $serverProject -c $Configuration
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build Fission.Server."
    }
}

$settings = [ordered]@{
    "Fission__Backend" = "onnx"
    "Fission__ExecutionProvider" = "cuda"
    "Fission__Device" = "cuda:$CudaDeviceId"
    "Fission__CudaDeviceId" = "$CudaDeviceId"
    "Fission__CudaRuntimeLibraryPath" = $CudaRuntimeLibraryPath
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
    "Fission__StartupProbeExitAfterSuccess" = "true"
    "Fission__StartupProbePrompt" = $Prompt
    "Fission__StartupProbeModelId" = $ModelId
    "Fission__StartupProbeMaxTokens" = "$MaxTokens"
    "Fission__StartupProbeTimeoutSeconds" = "$TimeoutSeconds"
}

$previous = @{}

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

    Write-Host ""
    Write-Host "Fission NVIDIA real-model smoke"
    Write-Host "  model:       $ModelId"
    Write-Host "  model path:  $model"
    Write-Host "  tokenizer:   $tokenizer"
    Write-Host "  device:      cuda:$CudaDeviceId"
    Write-Host "  max tokens:  $MaxTokens"
    Write-Host "  timeout:     $TimeoutSeconds s"
    Write-Host ""

    & dotnet run --project $serverProject -c $Configuration --no-build
    if ($LASTEXITCODE -ne 0) {
        throw "Fission NVIDIA smoke failed with exit code $LASTEXITCODE."
    }
}
finally {
    foreach ($entry in $settings.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable(
            $entry.Key,
            $previous[$entry.Key],
            [EnvironmentVariableTarget]::Process)
    }
}

Write-Host ""
Write-Host "Fission NVIDIA real-model smoke passed."
