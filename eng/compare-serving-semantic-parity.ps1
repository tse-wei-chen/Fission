param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $BaselineBaseUrl,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $CandidateBaseUrl,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $BaselineControlToken,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $CandidateControlToken,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $ModelId,

    [ValidateNotNullOrEmpty()]
    [string] $Corpus = "benchmarks/serving/semantic-parity.json",

    [ValidateNotNullOrEmpty()]
    [string] $OutputJson = "artifacts/semantic-parity/report.json",

    [ValidateNotNullOrEmpty()]
    [string] $OutputMarkdown = "artifacts/semantic-parity/report.md",

    [switch] $RequireExactTokens
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Resolve-RepositoryFile {
    param([string] $RepositoryRoot, [string] $Path)

    $candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        $Path
    } else {
        Join-Path $RepositoryRoot $Path
    }

    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "File '$candidate' does not exist."
    }

    return (Resolve-Path -LiteralPath $candidate).Path
}

function Invoke-ControlGenerate {
    param(
        [string] $BaseUrl,
        [string] $ControlToken,
        [string] $Model,
        [string] $Prompt,
        [int] $MaxTokens
    )

    $uri = [Uri]::new([Uri] $BaseUrl, "/internal/control/generate")
    $body = [ordered]@{
        model = $Model
        prompt = $Prompt
        max_tokens = $MaxTokens
        priority = 0
    } | ConvertTo-Json -Depth 4

    return Invoke-RestMethod `
        -Method Post `
        -Uri $uri `
        -Headers @{ "X-Fission-Control-Token" = $ControlToken } `
        -ContentType "application/json" `
        -Body $body `
        -TimeoutSec 600
}

function Get-CommonPrefixLength {
    param([int[]] $Left, [int[]] $Right)

    $limit = [Math]::Min($Left.Length, $Right.Length)
    for ($index = 0; $index -lt $limit; $index++) {
        if ($Left[$index] -ne $Right[$index]) {
            return $index
        }
    }

    return $limit
}

function Escape-Markdown {
    param([string] $Value)
    if ($null -eq $Value) {
        return ""
    }
    return $Value.Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

if (-not [Uri]::IsWellFormedUriString($BaselineBaseUrl, [UriKind]::Absolute)) {
    throw "BaselineBaseUrl '$BaselineBaseUrl' must be an absolute URI."
}
if (-not [Uri]::IsWellFormedUriString($CandidateBaseUrl, [UriKind]::Absolute)) {
    throw "CandidateBaseUrl '$CandidateBaseUrl' must be an absolute URI."
}

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$corpusPath = Resolve-RepositoryFile -RepositoryRoot $repositoryRoot -Path $Corpus
$definition = Get-Content -LiteralPath $corpusPath -Raw | ConvertFrom-Json
if ($definition.schema_version -ne 1) {
    throw "Unsupported semantic parity corpus schema_version '$($definition.schema_version)'. Expected 1."
}
$cases = @($definition.cases)
if ($cases.Count -eq 0) {
    throw "Semantic parity corpus contains no cases."
}

$results = [System.Collections.Generic.List[object]]::new()
foreach ($case in $cases) {
    $name = [string] $case.name
    $prompt = [string] $case.prompt
    $maxTokens = [int] $case.max_tokens
    if ([string]::IsNullOrWhiteSpace($name) -or
        [string]::IsNullOrWhiteSpace($prompt) -or
        $maxTokens -le 0) {
        throw "Semantic parity case '$name' is invalid."
    }

    Write-Host "==> semantic parity / $name"
    $baseline = Invoke-ControlGenerate `
        -BaseUrl $BaselineBaseUrl `
        -ControlToken $BaselineControlToken `
        -Model $ModelId `
        -Prompt $prompt `
        -MaxTokens $maxTokens
    $candidate = Invoke-ControlGenerate `
        -BaseUrl $CandidateBaseUrl `
        -ControlToken $CandidateControlToken `
        -Model $ModelId `
        -Prompt $prompt `
        -MaxTokens $maxTokens

    $baselineTokens = @($baseline.token_ids | ForEach-Object { [int] $_ })
    $candidateTokens = @($candidate.token_ids | ForEach-Object { [int] $_ })
    $prefix = Get-CommonPrefixLength -Left $baselineTokens -Right $candidateTokens
    $minLength = [Math]::Min($baselineTokens.Length, $candidateTokens.Length)
    $prefixRatio = if ($minLength -eq 0) {
        if ($baselineTokens.Length -eq 0 -and $candidateTokens.Length -eq 0) { 1.0 } else { 0.0 }
    } else {
        [double] $prefix / [double] $minLength
    }
    $exactTokens = $baselineTokens.Length -eq $candidateTokens.Length -and
        $prefix -eq $baselineTokens.Length
    $firstDivergence = if ($exactTokens) { $null } else { $prefix }

    $results.Add([pscustomobject]@{
        name = $name
        max_tokens = $maxTokens
        baseline_token_count = $baselineTokens.Length
        candidate_token_count = $candidateTokens.Length
        common_prefix_tokens = $prefix
        common_prefix_ratio = $prefixRatio
        first_divergence_index = $firstDivergence
        exact_tokens = $exactTokens
        exact_text = [string] $baseline.text -eq [string] $candidate.text
        baseline_finish_reason = [string] $baseline.finish_reason
        candidate_finish_reason = [string] $candidate.finish_reason
        baseline_token_ids = $baselineTokens
        candidate_token_ids = $candidateTokens
        baseline_text = [string] $baseline.text
        candidate_text = [string] $candidate.text
    })
}

$exactTokenCases = @($results | Where-Object { $_.exact_tokens }).Count
$exactTextCases = @($results | Where-Object { $_.exact_text }).Count
$meanPrefixRatio = [double] (($results | Measure-Object -Property common_prefix_ratio -Average).Average)

$report = [ordered]@{
    schema_version = 1
    generated_at_utc = [DateTimeOffset]::UtcNow.ToString("O")
    model = $ModelId
    corpus = $corpusPath
    baseline_base_url = $BaselineBaseUrl
    candidate_base_url = $CandidateBaseUrl
    cases = $results
    summary = [ordered]@{
        case_count = $results.Count
        exact_token_cases = $exactTokenCases
        exact_text_cases = $exactTextCases
        mean_common_prefix_ratio = $meanPrefixRatio
    }
}

$jsonFullPath = [System.IO.Path]::GetFullPath($OutputJson)
$jsonDirectory = Split-Path -Parent $jsonFullPath
if (-not [string]::IsNullOrWhiteSpace($jsonDirectory)) {
    New-Item -ItemType Directory -Force -Path $jsonDirectory | Out-Null
}
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonFullPath -Encoding UTF8

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Serving semantic parity report")
$lines.Add("")
$lines.Add("Model: `$ModelId`")
$lines.Add("")
$lines.Add("- Cases: $($results.Count)")
$lines.Add("- Exact token cases: $exactTokenCases/$($results.Count)")
$lines.Add("- Exact text cases: $exactTextCases/$($results.Count)")
$lines.Add("- Mean common token-prefix ratio: $("{0:P2}" -f $meanPrefixRatio)")
$lines.Add("")
$lines.Add("| Case | Baseline tokens | Candidate tokens | Common prefix | Prefix ratio | Exact tokens | Exact text |")
$lines.Add("| --- | ---: | ---: | ---: | ---: | --- | --- |")
foreach ($result in $results) {
    $lines.Add(
        "| $(Escape-Markdown $result.name) | $($result.baseline_token_count) | $($result.candidate_token_count) | $($result.common_prefix_tokens) | $("{0:P2}" -f $result.common_prefix_ratio) | $($result.exact_tokens) | $($result.exact_text) |")
}
$lines.Add("")
$lines.Add("Token IDs and full decoded text for both endpoints are preserved in the JSON report.")

$markdownFullPath = [System.IO.Path]::GetFullPath($OutputMarkdown)
$markdownDirectory = Split-Path -Parent $markdownFullPath
if (-not [string]::IsNullOrWhiteSpace($markdownDirectory)) {
    New-Item -ItemType Directory -Force -Path $markdownDirectory | Out-Null
}
$lines | Set-Content -LiteralPath $markdownFullPath -Encoding UTF8

Get-Content -LiteralPath $markdownFullPath

if ($RequireExactTokens -and $exactTokenCases -ne $results.Count) {
    throw "Semantic parity failed exact-token gate: $exactTokenCases/$($results.Count) case(s) matched exactly."
}
