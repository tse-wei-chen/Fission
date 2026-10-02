param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $Label,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $BaseUrl,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $Model,

    [string] $Manifest = "benchmarks/serving/workloads.json",

    [string] $OutputDirectory = "artifacts/serving",

    [ValidateRange(1, 1000)]
    [int] $Repetitions = 1,

    [ValidateNotNullOrEmpty()]
    [string] $Configuration = "Release",

    [switch] $NoBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($Label -notmatch '^[A-Za-z0-9._-]+$') {
    throw "Label '$Label' may contain only letters, digits, '.', '_' and '-'."
}

if (-not [Uri]::IsWellFormedUriString($BaseUrl, [UriKind]::Absolute)) {
    throw "BaseUrl '$BaseUrl' must be an absolute URI."
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$loadGenProject = Join-Path $repositoryRoot "benchmarks/Fission.Serving.LoadGen/Fission.Serving.LoadGen.csproj"
$manifestPath = if ([System.IO.Path]::IsPathRooted($Manifest)) {
    $Manifest
} else {
    Join-Path $repositoryRoot $Manifest
}

if (-not (Test-Path -LiteralPath $loadGenProject -PathType Leaf)) {
    throw "Load-generator project '$loadGenProject' does not exist."
}

if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw "Benchmark manifest '$manifestPath' does not exist."
}

$manifestPath = (Resolve-Path -LiteralPath $manifestPath).Path
$manifestDirectory = Split-Path -Parent $manifestPath
$definition = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

if ($definition.schema_version -ne 1) {
    throw "Unsupported benchmark manifest schema_version '$($definition.schema_version)'. Expected 1."
}

$workloads = @($definition.workloads)
if ($workloads.Count -eq 0) {
    throw "Benchmark manifest contains no workloads."
}

if (-not $NoBuild) {
    & dotnet build $loadGenProject -c $Configuration
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build serving load generator."
    }
}

$outputRoot = if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory
} else {
    Join-Path $repositoryRoot $OutputDirectory
}
$outputRoot = [System.IO.Path]::GetFullPath($outputRoot)

Write-Host "Serving benchmark suite"
Write-Host "  label:       $Label"
Write-Host "  target:      $BaseUrl"
Write-Host "  model:       $Model"
Write-Host "  manifest:    $manifestPath"
Write-Host "  repetitions: $Repetitions"
Write-Host "  output:      $outputRoot"

foreach ($workload in $workloads) {
    $name = [string] $workload.name
    if ([string]::IsNullOrWhiteSpace($name) -or $name -notmatch '^[A-Za-z0-9._-]+$') {
        throw "Workload name '$name' must use only letters, digits, '.', '_' and '-'."
    }

    $endpoint = [string] $workload.endpoint
    if ($endpoint -notin @("completions", "chat")) {
        throw "Workload '$name' endpoint must be 'completions' or 'chat'."
    }

    $requests = [int] $workload.requests
    $maxTokens = [int] $workload.max_tokens
    $warmup = [int] $workload.warmup
    $timeoutSeconds = [int] $workload.timeout_seconds

    if ($requests -le 0 -or $maxTokens -le 0 -or $warmup -lt 0 -or $timeoutSeconds -le 0) {
        throw "Workload '$name' contains invalid request/token/warmup/timeout values."
    }

    $concurrencyValues = @($workload.concurrency | ForEach-Object { [int] $_ })
    if ($concurrencyValues.Count -eq 0 -or ($concurrencyValues | Where-Object { $_ -le 0 })) {
        throw "Workload '$name' must define one or more positive concurrency values."
    }

    $promptFile = [string] $workload.prompt_file
    if ([string]::IsNullOrWhiteSpace($promptFile)) {
        throw "Workload '$name' must define prompt_file."
    }

    $promptPath = if ([System.IO.Path]::IsPathRooted($promptFile)) {
        $promptFile
    } else {
        Join-Path $manifestDirectory $promptFile
    }

    if (-not (Test-Path -LiteralPath $promptPath -PathType Leaf)) {
        throw "Prompt file '$promptPath' for workload '$name' does not exist."
    }

    $promptPath = (Resolve-Path -LiteralPath $promptPath).Path

    foreach ($concurrency in $concurrencyValues) {
        for ($repetition = 1; $repetition -le $Repetitions; $repetition++) {
            $workloadOutput = Join-Path $outputRoot "$Label/$name"
            New-Item -ItemType Directory -Force -Path $workloadOutput | Out-Null
            $outputPath = Join-Path $workloadOutput "c$($concurrency)-r$($repetition).json"

            Write-Host ""
            Write-Host "==> $Label / $name / concurrency=$concurrency / repetition=$repetition"

            $arguments = @(
                "run",
                "--project", $loadGenProject,
                "-c", $Configuration,
                "--no-build",
                "--",
                "--url", $BaseUrl,
                "--model", $Model,
                "--endpoint", $endpoint,
                "--requests", $requests,
                "--concurrency", $concurrency,
                "--max-tokens", $maxTokens,
                "--warmup", $warmup,
                "--timeout-seconds", $timeoutSeconds,
                "--prompt-file", $promptPath,
                "--output", $outputPath
            )

            & dotnet @arguments
            if ($LASTEXITCODE -ne 0) {
                throw "Serving benchmark failed for '$name' at concurrency $concurrency, repetition $repetition."
            }
        }
    }
}

Write-Host ""
Write-Host "Serving benchmark suite completed: $outputRoot"
