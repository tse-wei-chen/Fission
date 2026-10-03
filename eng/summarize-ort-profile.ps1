param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $ProfilePath,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $MarkdownPath,

    [ValidateNotNull()]
    [string] $JsonPath = "",

    [ValidateNotNull()]
    [string[]] $RequireCudaOp = @(),

    [switch] $PartialTrace
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Get-OptionalProperty {
    param(
        [Parameter(Mandatory = $true)]
        [object] $Object,

        [Parameter(Mandatory = $true)]
        [string] $Name
    )

    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }

    return $property.Value
}

function Format-Percent {
    param(
        [double] $Part,
        [double] $Total
    )

    if ($Total -le 0) {
        return "0.00%"
    }

    return "{0:F2}%" -f (100.0 * $Part / $Total)
}

function Escape-Markdown {
    param([string] $Value)

    if ($null -eq $Value) {
        return ""
    }

    return $Value.Replace("|", "\|")
}

function Add-Stat {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable] $Table,

        [Parameter(Mandatory = $true)]
        [string] $Key,

        [Parameter(Mandatory = $true)]
        [double] $DurationUs,

        [string] $Provider = "",

        [string] $OpName = ""
    )

    if (-not $Table.ContainsKey($Key)) {
        $Table[$Key] = [pscustomobject]@{
            provider = $Provider
            op_name = $OpName
            events = 0L
            duration_us = 0.0
        }
    }

    $stat = $Table[$Key]
    $stat.events = [long] $stat.events + 1L
    $stat.duration_us = [double] $stat.duration_us + $DurationUs
}

function Get-MultiLineBraceDelta {
    param([string] $Text)

    $delta = 0
    $inString = $false
    $escaped = $false
    foreach ($character in $Text.ToCharArray()) {
        if ($inString) {
            if ($escaped) {
                $escaped = $false
                continue
            }
            if ($character -eq '\\') {
                $escaped = $true
                continue
            }
            if ($character -eq '"') {
                $inString = $false
            }
            continue
        }

        if ($character -eq '"') {
            $inString = $true
            continue
        }
        if ($character -eq '{') {
            $delta++
        } elseif ($character -eq '}') {
            $delta--
        }
    }

    return $delta
}

$resolvedProfile = (Resolve-Path -LiteralPath $ProfilePath).Path
$providerStats = @{}
$opStats = @{}
$nodeEventCount = 0L
$totalDurationUs = 0.0
$memcpyEventCount = 0L
$memcpyDurationUs = 0.0
$detectedEventLimit = [bool] $PartialTrace

function Add-ProfileEvent {
    param([object] $Event)

    $category = [string] (Get-OptionalProperty -Object $Event -Name "cat")
    $duration = Get-OptionalProperty -Object $Event -Name "dur"
    if ($category -ne "Node" -or $null -eq $duration) {
        return
    }

    $durationUs = [double] $duration
    if ($durationUs -lt 0) {
        return
    }

    $args = Get-OptionalProperty -Object $Event -Name "args"
    $provider = "unknown"
    $opName = "unknown"
    if ($null -ne $args) {
        $providerValue = Get-OptionalProperty -Object $args -Name "provider"
        if (-not [string]::IsNullOrWhiteSpace([string] $providerValue)) {
            $provider = [string] $providerValue
        }

        $opValue = Get-OptionalProperty -Object $args -Name "op_name"
        if (-not [string]::IsNullOrWhiteSpace([string] $opValue)) {
            $opName = [string] $opValue
        }
    }

    $eventName = [string] (Get-OptionalProperty -Object $Event -Name "name")
    if ([string]::IsNullOrWhiteSpace($eventName)) {
        $eventName = "unknown"
    }

    $isMemcpy =
        $opName.Contains("Memcpy", [StringComparison]::OrdinalIgnoreCase) -or
        $eventName.Contains("Memcpy", [StringComparison]::OrdinalIgnoreCase)

    $script:nodeEventCount++
    $script:totalDurationUs += $durationUs
    if ($isMemcpy) {
        $script:memcpyEventCount++
        $script:memcpyDurationUs += $durationUs
    }

    Add-Stat -Table $script:providerStats -Key $provider -DurationUs $durationUs -Provider $provider
    Add-Stat -Table $script:opStats -Key "${provider}::$opName" -DurationUs $durationUs -Provider $provider -OpName $opName
}

$reader = [System.IO.StreamReader]::new($resolvedProfile)
$multiLineBuffer = $null
$multiLineDepth = 0
try {
    while ($null -ne ($line = $reader.ReadLine())) {
        if ($line.Contains("event limit", [StringComparison]::OrdinalIgnoreCase) -and
            $line.Contains("reach", [StringComparison]::OrdinalIgnoreCase)) {
            $detectedEventLimit = $true
        }

        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed -eq "[" -or $trimmed -eq "]") {
            continue
        }

        if ($null -eq $multiLineBuffer -and
            $trimmed.StartsWith("{", [StringComparison]::Ordinal) -and
            ($trimmed.EndsWith("}", [StringComparison]::Ordinal) -or
             $trimmed.EndsWith("},", [StringComparison]::Ordinal))) {
            $jsonText = if ($trimmed.EndsWith(",", [StringComparison]::Ordinal)) {
                $trimmed.Substring(0, $trimmed.Length - 1)
            } else {
                $trimmed
            }
            Add-ProfileEvent -Event ($jsonText | ConvertFrom-Json)
            continue
        }

        if ($null -eq $multiLineBuffer) {
            $multiLineBuffer = [System.Text.StringBuilder]::new()
            $multiLineDepth = 0
        }

        [void] $multiLineBuffer.AppendLine($trimmed.TrimEnd(','))
        $multiLineDepth += Get-MultiLineBraceDelta -Text $trimmed
        if ($multiLineDepth -eq 0 -and $multiLineBuffer.Length -gt 0) {
            $jsonText = $multiLineBuffer.ToString().Trim().TrimEnd(',')
            if ($jsonText.StartsWith("{", [StringComparison]::Ordinal)) {
                Add-ProfileEvent -Event ($jsonText | ConvertFrom-Json)
            }
            $multiLineBuffer = $null
        }
    }
}
finally {
    $reader.Dispose()
}

if ($null -ne $multiLineBuffer) {
    throw "ORT profile ended with an incomplete JSON event."
}

$providers = @(
    $providerStats.Values |
        ForEach-Object {
            [pscustomobject]@{
                provider = $_.provider
                events = $_.events
                duration_us = $_.duration_us
                duration_ms = $_.duration_us / 1000.0
                percent = if ($totalDurationUs -gt 0) {
                    100.0 * $_.duration_us / $totalDurationUs
                } else {
                    0.0
                }
            }
        } |
        Sort-Object duration_us -Descending
)

$allOps = @(
    $opStats.Values |
        ForEach-Object {
            [pscustomobject]@{
                provider = $_.provider
                op_name = $_.op_name
                events = $_.events
                duration_us = $_.duration_us
                duration_ms = $_.duration_us / 1000.0
                percent = if ($totalDurationUs -gt 0) {
                    100.0 * $_.duration_us / $totalDurationUs
                } else {
                    0.0
                }
            }
        }
)
$topOps = @($allOps | Sort-Object duration_us -Descending | Select-Object -First 20)

$gateResults = [System.Collections.Generic.List[object]]::new()
$gateFailures = [System.Collections.Generic.List[string]]::new()
foreach ($requiredOp in @($RequireCudaOp | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)) {
    $matching = @($allOps | Where-Object { $_.op_name -eq $requiredOp })
    $cudaEvents = [long] (($matching | Where-Object { $_.provider -eq "CUDAExecutionProvider" } | Measure-Object -Property events -Sum).Sum ?? 0)
    $nonCudaEvents = [long] (($matching | Where-Object { $_.provider -ne "CUDAExecutionProvider" } | Measure-Object -Property events -Sum).Sum ?? 0)
    $passed = $cudaEvents -gt 0 -and $nonCudaEvents -eq 0
    $gateResults.Add([pscustomobject]@{
        op_name = $requiredOp
        required_provider = "CUDAExecutionProvider"
        cuda_events = $cudaEvents
        non_cuda_events = $nonCudaEvents
        passed = $passed
    })
    if (-not $passed) {
        $gateFailures.Add(
            "Operator '$requiredOp' requires exclusive CUDA placement, but profile contains $cudaEvents CUDA event(s) and $nonCudaEvents non-CUDA event(s).")
    }
}

$summary = [ordered]@{
    schema_version = 2
    profile_path = $resolvedProfile
    partial_trace = $detectedEventLimit
    node_event_count = $nodeEventCount
    node_duration_us = $totalDurationUs
    memcpy_event_count = $memcpyEventCount
    memcpy_duration_us = $memcpyDurationUs
    memcpy_percent_of_node_duration = if ($totalDurationUs -gt 0) {
        100.0 * $memcpyDurationUs / $totalDurationUs
    } else {
        0.0
    }
    providers = $providers
    top_ops = $topOps
    cuda_op_gates = $gateResults
    gate_passed = $gateFailures.Count -eq 0
}

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# ONNX Runtime profile summary")
$lines.Add("")
$lines.Add(("Profile: {0}{1}{0}" -f [char]96, (Escape-Markdown $resolvedProfile)))
$lines.Add("")
if ($detectedEventLimit) {
    $lines.Add("WARNING: ORT reported its event limit was reached. This is a partial trace, not the full workload.")
    $lines.Add("")
}
$lines.Add("- Node execution events: $nodeEventCount")
$lines.Add("- Summed node duration: $("{0:F2}" -f ($totalDurationUs / 1000.0)) ms")
$lines.Add("- Memcpy execution events: $memcpyEventCount")
$lines.Add("- Summed Memcpy duration: $("{0:F2}" -f ($memcpyDurationUs / 1000.0)) ms ($(Format-Percent $memcpyDurationUs $totalDurationUs) of summed node duration)")
$lines.Add("")
$lines.Add("Node durations are summed ORT trace event durations, not wall-clock request latency.")
$lines.Add("")

if ($gateResults.Count -gt 0) {
    $lines.Add("## Structural gates")
    $lines.Add("")
    $lines.Add("| Op | Required provider | CUDA events | Non-CUDA events | Result |")
    $lines.Add("| --- | --- | ---: | ---: | --- |")
    foreach ($gate in $gateResults) {
        $result = if ($gate.passed) { "PASS" } else { "FAIL" }
        $lines.Add("| $(Escape-Markdown $gate.op_name) | CUDAExecutionProvider | $($gate.cuda_events) | $($gate.non_cuda_events) | $result |")
    }
    $lines.Add("")
}

$lines.Add("## Execution providers")
$lines.Add("")
$lines.Add("| Provider | Events | Duration ms | Share |")
$lines.Add("| --- | ---: | ---: | ---: |")
foreach ($provider in $providers) {
    $lines.Add(
        "| $(Escape-Markdown $provider.provider) | $($provider.events) | $("{0:F2}" -f $provider.duration_ms) | $("{0:F2}%" -f $provider.percent) |")
}
$lines.Add("")

$lines.Add("## Top operator groups")
$lines.Add("")
$lines.Add("| Provider | Op | Events | Duration ms | Share |")
$lines.Add("| --- | --- | ---: | ---: | ---: |")
foreach ($op in $topOps) {
    $lines.Add(
        "| $(Escape-Markdown $op.provider) | $(Escape-Markdown $op.op_name) | $($op.events) | $("{0:F2}" -f $op.duration_ms) | $("{0:F2}%" -f $op.percent) |")
}
$lines.Add("")

$markdownDirectory = Split-Path -Parent ([System.IO.Path]::GetFullPath($MarkdownPath))
if (-not [string]::IsNullOrWhiteSpace($markdownDirectory)) {
    New-Item -ItemType Directory -Force -Path $markdownDirectory | Out-Null
}
$lines | Set-Content -LiteralPath $MarkdownPath -Encoding UTF8

if (-not [string]::IsNullOrWhiteSpace($JsonPath)) {
    $jsonDirectory = Split-Path -Parent ([System.IO.Path]::GetFullPath($JsonPath))
    if (-not [string]::IsNullOrWhiteSpace($jsonDirectory)) {
        New-Item -ItemType Directory -Force -Path $jsonDirectory | Out-Null
    }

    $summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $JsonPath -Encoding UTF8
}

Get-Content -LiteralPath $MarkdownPath

if ($gateFailures.Count -gt 0) {
    throw ($gateFailures -join [Environment]::NewLine)
}
