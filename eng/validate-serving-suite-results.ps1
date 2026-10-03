param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $ManifestPath,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $ResultsRoot,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $Label,

    [ValidateRange(1, 1000)]
    [int] $Repetitions = 1,

    [ValidateNotNull()]
    [string] $Model = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$manifest = (Resolve-Path -LiteralPath $ManifestPath).Path
$results = [System.IO.Path]::GetFullPath($ResultsRoot)
if (-not (Test-Path -LiteralPath $results -PathType Container)) {
    throw "Serving results root '$results' does not exist."
}

$definition = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
if ($definition.schema_version -ne 1) {
    throw "Unsupported benchmark manifest schema_version '$($definition.schema_version)'. Expected 1."
}

$expected = 0
$validated = 0
$failures = [System.Collections.Generic.List[string]]::new()

foreach ($workload in @($definition.workloads)) {
    $name = [string] $workload.name
    $requests = [int] $workload.requests
    $maxTokens = [int] $workload.max_tokens

    foreach ($concurrencyValue in @($workload.concurrency)) {
        $concurrency = [int] $concurrencyValue
        for ($repetition = 1; $repetition -le $Repetitions; $repetition++) {
            $expected++
            $path = Join-Path $results "$Label/$name/c$($concurrency)-r$($repetition).json"
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                $failures.Add("Missing result: $path")
                continue
            }

            try {
                $report = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
                if ([int] $report.concurrency -ne $concurrency) {
                    $failures.Add("$path: concurrency=$($report.concurrency), expected $concurrency.")
                    continue
                }
                if ([int] $report.max_tokens -ne $maxTokens) {
                    $failures.Add("$path: max_tokens=$($report.max_tokens), expected $maxTokens.")
                    continue
                }
                if ([int] $report.total_requests -ne $requests) {
                    $failures.Add("$path: total_requests=$($report.total_requests), expected $requests.")
                    continue
                }
                if ([int] $report.successful_requests -ne $requests -or
                    [int] $report.failed_requests -ne 0) {
                    $failures.Add(
                        "$path: successful=$($report.successful_requests), failed=$($report.failed_requests); expected $requests/0.")
                    continue
                }
                if (-not [bool] $report.all_token_counts_from_usage) {
                    $failures.Add("$path: token usage accounting is approximate, expected exact usage.")
                    continue
                }
                if (-not [string]::IsNullOrWhiteSpace($Model) -and
                    [string] $report.model -ne $Model) {
                    $failures.Add("$path: model='$($report.model)', expected '$Model'.")
                    continue
                }

                $validated++
            }
            catch {
                $failures.Add("$path: $($_.Exception.Message)")
            }
        }
    }
}

Write-Host "Serving suite completeness gate"
Write-Host "  manifest:  $manifest"
Write-Host "  label:     $Label"
Write-Host "  expected:  $expected"
Write-Host "  validated: $validated"

if ($failures.Count -gt 0) {
    foreach ($failure in $failures) {
        Write-Error $failure
    }
    throw "Serving suite completeness gate failed: $($failures.Count) issue(s)."
}

if ($validated -ne $expected) {
    throw "Serving suite completeness gate validated $validated/$expected result(s)."
}

Write-Host "Serving suite completeness gate passed."
