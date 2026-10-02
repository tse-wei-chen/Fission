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
$projectRoot = Join-Path $repositoryRoot $Root

if (-not (Test-Path -LiteralPath $projectRoot -PathType Container)) {
    throw "Project root '$projectRoot' does not exist."
}

$projects = @(
    Get-ChildItem -LiteralPath $projectRoot -Recurse -File |
        Where-Object {
            ($_.Extension -eq ".csproj" -or $_.Extension -eq ".fsproj") -and
            $_.FullName -notmatch "[\\/](bin|obj)[\\/]"
        } |
        Sort-Object FullName
)

if ($projects.Count -eq 0) {
    throw "No .csproj or .fsproj files found under '$Root'."
}

Write-Host "Running $($projects.Count) project(s) under '$Root' in $Configuration configuration."

foreach ($project in $projects) {
    $relativePath = [System.IO.Path]::GetRelativePath($repositoryRoot, $project.FullName)
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
Write-Host "Completed $($projects.Count) project(s) under '$Root'."
