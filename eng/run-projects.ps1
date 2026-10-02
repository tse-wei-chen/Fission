param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("samples", "tests")]
    [string] $Root,

    [ValidateNotNullOrEmpty()]
    [string] $Configuration = "Release",

    [switch] $NoBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$solutionPath = Join-Path $repositoryRoot "Fission.slnx"

if (-not (Test-Path -LiteralPath $solutionPath -PathType Leaf)) {
    throw "Solution manifest '$solutionPath' does not exist."
}

[xml] $solution = Get-Content -LiteralPath $solutionPath -Raw
$prefix = "$Root/"

$projectPaths = @(
    $solution.Solution.Project |
        ForEach-Object { [string] $_.Path } |
        Where-Object {
            $_.StartsWith($prefix, [StringComparison]::Ordinal) -and
            ($_.EndsWith(".csproj", [StringComparison]::OrdinalIgnoreCase) -or
             $_.EndsWith(".fsproj", [StringComparison]::OrdinalIgnoreCase))
        } |
        Sort-Object
)

if ($projectPaths.Count -eq 0) {
    throw "No solution projects found under '$Root'."
}

Write-Host "Running $($projectPaths.Count) solution project(s) under '$Root' in $Configuration configuration."

foreach ($relativePath in $projectPaths) {
    $projectPath = Join-Path $repositoryRoot $relativePath
    if (-not (Test-Path -LiteralPath $projectPath -PathType Leaf)) {
        throw "Solution project '$relativePath' does not exist on disk."
    }

    Write-Host ""
    Write-Host "==> $relativePath"

    $arguments = @(
        "run",
        "--project", $relativePath,
        "-c", $Configuration
    )

    if ($NoBuild) {
        $arguments += "--no-build"
    }

    & dotnet @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "dotnet run failed for '$relativePath' with exit code $LASTEXITCODE."
    }
}

Write-Host ""
Write-Host "Completed $($projectPaths.Count) solution project(s) under '$Root'."
