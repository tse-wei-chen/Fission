param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $ProfilePath,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string] $MarkdownPath,

    [ValidateNotNull()]
    [string] $JsonPath = ""
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

$resolvedProfile = (Resolve-Path -LiteralPath $ProfilePath).Path
$events = @(Get-Content -LiteralPath $resolvedProfile -Raw | ConvertFrom-Json)

$records = @(
    foreach ($event in $events) {
        $category = [string] (Get-OptionalProperty -Object $event -Name "cat")
        $duration = Get-OptionalProperty -Object $event -Name "dur"
        if ($category -ne "Node" -or $null -eq $duration) {
            continue
        }

        $durationUs = [double] $duration
        if ($durationUs -lt 0) {
            continue
        }

        $args = Get-OptionalProperty -Object $event -Name "args"
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

        $eventName = [string] (Get-OptionalProperty -Object $event -Name "name")
        if ([string]::IsNullOrWhiteSpace($eventName)) {
            $eventName = "unknown"
        }

        $isMemcpy =
            $opName.Contains("Memcpy", [StringComparison]::OrdinalIgnoreCase) -or
            $eventName.Contains("Memcpy", [StringComparison]::OrdinalIgnoreCase)

        [pscustomobject]@{
            provider = $provider
            op_name = $opName
            event_name = $eventName
            duration_us = $durationUs
            is_memcpy = $isMemcpy
            op_key = "${provider}::$opName"
        }
    }
)

$totalDurationUs = [double] (($records | Measure-Object -Property duration_us -Sum).Sum ?? 0)
$memcpyRecords = @($records | Where-Object { $_.is_memcpy })
$memcpyDurationUs = [double] (($memcpyRecords | Measure-Object -Property duration_us -Sum).Sum ?? 0)

$providers = @(
    $records |
        Group-Object provider |
        ForEach-Object {
            $durationUs = [double] (($_.Group | Measure-Object -Property duration_us -Sum).Sum ?? 0)
            [pscustomobject]@{
                provider = $_.Name
                events = $_.Count
                duration_us = $durationUs
                duration_ms = $durationUs / 1000.0
                percent = if ($totalDurationUs -gt 0) {
                    100.0 * $durationUs / $totalDurationUs
                } else {
                    0.0
                }
            }
        } |
        Sort-Object duration_us -Descending
)

$topOps = @(
    $records |
        Group-Object op_key |
        ForEach-Object {
            $first = $_.Group[0]
            $durationUs = [double] (($_.Group | Measure-Object -Property duration_us -Sum).Sum ?? 0)
            [pscustomobject]@{
                provider = $first.provider
                op_name = $first.op_name
                events = $_.Count
                duration_us = $durationUs
                duration_ms = $durationUs / 1000.0
                percent = if ($totalDurationUs -gt 0) {
                    100.0 * $durationUs / $totalDurationUs
                } else {
                    0.0
                }
            }
        } |
        Sort-Object duration_us -Descending |
        Select-Object -First 20
)

$summary = [ordered]@{
    schema_version = 1
    profile_path = $resolvedProfile
    node_event_count = $records.Count
    node_duration_us = $totalDurationUs
    memcpy_event_count = $memcpyRecords.Count
    memcpy_duration_us = $memcpyDurationUs
    memcpy_percent_of_node_duration = if ($totalDurationUs -gt 0) {
        100.0 * $memcpyDurationUs / $totalDurationUs
    } else {
        0.0
    }
    providers = $providers
    top_ops = $topOps
}

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# ONNX Runtime profile summary")
$lines.Add("")
$lines.Add(("Profile: {0}{1}{0}" -f [char]96, (Escape-Markdown $resolvedProfile)))
$lines.Add("")
$lines.Add("- Node execution events: $($records.Count)")
$lines.Add("- Summed node duration: $("{0:F2}" -f ($totalDurationUs / 1000.0)) ms")
$lines.Add("- Memcpy execution events: $($memcpyRecords.Count)")
$lines.Add("- Summed Memcpy duration: $("{0:F2}" -f ($memcpyDurationUs / 1000.0)) ms ($(Format-Percent $memcpyDurationUs $totalDurationUs) of summed node duration)")
$lines.Add("")
$lines.Add("Node durations are summed ORT trace event durations, not wall-clock request latency.")
$lines.Add("")

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
