using System.Globalization;
using System.Text;
using System.Text.Json;

try
{
    var options = ReporterOptions.Parse(args);
    if (options.ShowHelp)
    {
        ReporterOptions.PrintHelp();
        return 0;
    }

    var inputRoot = Path.GetFullPath(options.InputDirectory);
    if (!Directory.Exists(inputRoot))
    {
        throw new DirectoryNotFoundException($"Input directory '{inputRoot}' does not exist.");
    }

    var files = Directory
        .EnumerateFiles(inputRoot, "*.json", SearchOption.AllDirectories)
        .Order(StringComparer.Ordinal)
        .ToArray();

    if (files.Length == 0)
    {
        throw new InvalidOperationException($"No JSON benchmark reports were found under '{inputRoot}'.");
    }

    var runs = new List<LabeledRun>(files.Length);
    foreach (var file in files)
    {
        var relative = Path.GetRelativePath(inputRoot, file);
        var parts = relative.Split(
            new[] { Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar },
            StringSplitOptions.RemoveEmptyEntries);

        if (parts.Length < 3)
        {
            throw new InvalidOperationException(
                $"Report '{relative}' must use '<engine>/<workload>/<run>.json' layout.");
        }

        runs.Add(new LabeledRun(
            Engine: parts[0],
            Workload: parts[1],
            Report: BenchmarkRun.Load(file)));
    }

    var summaries = runs
        .GroupBy(static run => new SummaryKey(
            run.Engine,
            run.Workload,
            run.Report.Endpoint,
            run.Report.Model,
            run.Report.Concurrency,
            run.Report.MaxTokens))
        .Select(static group => BenchmarkSummary.Create(group.Key, group.Select(static item => item.Report).ToArray()))
        .OrderBy(static summary => summary.Key.Workload, StringComparer.Ordinal)
        .ThenBy(static summary => summary.Key.Concurrency)
        .ThenBy(static summary => summary.Key.MaxTokens)
        .ThenBy(static summary => summary.Key.Engine, StringComparer.Ordinal)
        .ToArray();

    var baseline = string.IsNullOrWhiteSpace(options.BaselineEngine)
        ? null
        : options.BaselineEngine;

    if (baseline is not null &&
        !summaries.Any(summary => string.Equals(summary.Key.Engine, baseline, StringComparison.Ordinal)))
    {
        throw new InvalidOperationException(
            $"Baseline engine '{baseline}' was not found under '{inputRoot}'.");
    }

    var comparisons = summaries
        .Select(summary => new SummaryComparison(
            summary,
            FindBaseline(summary, summaries, baseline)))
        .ToArray();

    var markdown = RenderMarkdown(comparisons, baseline);
    var csv = RenderCsv(comparisons, baseline);

    Console.WriteLine(markdown);

    if (!string.IsNullOrWhiteSpace(options.MarkdownPath))
    {
        await WriteTextAsync(options.MarkdownPath, markdown);
        Console.WriteLine($"Markdown: {Path.GetFullPath(options.MarkdownPath)}");
    }

    if (!string.IsNullOrWhiteSpace(options.CsvPath))
    {
        await WriteTextAsync(options.CsvPath, csv);
        Console.WriteLine($"CSV: {Path.GetFullPath(options.CsvPath)}");
    }

    return 0;
}
catch (Exception exception)
{
    Console.Error.WriteLine(exception.Message);
    return 2;
}

static BenchmarkSummary? FindBaseline(
    BenchmarkSummary summary,
    IReadOnlyList<BenchmarkSummary> all,
    string? baseline)
{
    if (baseline is null ||
        string.Equals(summary.Key.Engine, baseline, StringComparison.Ordinal))
    {
        return null;
    }

    return all.FirstOrDefault(candidate =>
        string.Equals(candidate.Key.Engine, baseline, StringComparison.Ordinal) &&
        string.Equals(candidate.Key.Workload, summary.Key.Workload, StringComparison.Ordinal) &&
        string.Equals(candidate.Key.Endpoint, summary.Key.Endpoint, StringComparison.Ordinal) &&
        string.Equals(candidate.Key.Model, summary.Key.Model, StringComparison.Ordinal) &&
        candidate.Key.Concurrency == summary.Key.Concurrency &&
        candidate.Key.MaxTokens == summary.Key.MaxTokens);
}

static string RenderMarkdown(
    IReadOnlyList<SummaryComparison> comparisons,
    string? baseline)
{
    var builder = new StringBuilder();
    builder.AppendLine("# Serving benchmark report");
    builder.AppendLine();
    builder.AppendLine($"Generated: {DateTimeOffset.UtcNow:O}");
    builder.AppendLine();

    if (baseline is not null)
    {
        builder.AppendLine($"Baseline: `{baseline}`. Delta columns are relative to the matching baseline row; positive means the metric value is higher.");
        builder.AppendLine();
    }

    builder.Append("| Engine | Workload | Endpoint | C | Max tokens | Runs | Success | Failed | Req/s mean | Req/s sd | Output tok/s mean | TTFT p50 avg ms | TTFT p95 avg ms | TPOT p50 avg ms | TPOT p95 avg ms | E2E p95 avg ms | Exact usage |");
    if (baseline is not null)
    {
        builder.Append(" Δ req/s | Δ output tok/s | Δ TTFT p50 | Δ TPOT p50 |");
    }
    builder.AppendLine();
    builder.Append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
    if (baseline is not null)
    {
        builder.Append(" ---: | ---: | ---: | ---: |");
    }
    builder.AppendLine();

    foreach (var comparison in comparisons)
    {
        var summary = comparison.Summary;
        builder.Append($"| {EscapeMarkdown(summary.Key.Engine)}");
        builder.Append($" | {EscapeMarkdown(summary.Key.Workload)}");
        builder.Append($" | {EscapeMarkdown(summary.Key.Endpoint)}");
        builder.Append($" | {summary.Key.Concurrency}");
        builder.Append($" | {summary.Key.MaxTokens}");
        builder.Append($" | {summary.Runs}");
        builder.Append($" | {summary.SuccessfulRequests}");
        builder.Append($" | {summary.FailedRequests}");
        builder.Append($" | {Format(summary.RequestsPerSecondMean)}");
        builder.Append($" | {Format(summary.RequestsPerSecondStdDev)}");
        builder.Append($" | {Format(summary.OutputTokensPerSecondMean)}");
        builder.Append($" | {Format(summary.TtftP50Mean)}");
        builder.Append($" | {Format(summary.TtftP95Mean)}");
        builder.Append($" | {Format(summary.TpotP50Mean)}");
        builder.Append($" | {Format(summary.TpotP95Mean)}");
        builder.Append($" | {Format(summary.E2eP95Mean)}");
        builder.Append($" | {summary.ExactUsageRuns}/{summary.Runs} |");

        if (baseline is not null)
        {
            builder.Append($" {FormatPercent(PercentDelta(summary.RequestsPerSecondMean, comparison.Baseline?.RequestsPerSecondMean))} |");
            builder.Append($" {FormatPercent(PercentDelta(summary.OutputTokensPerSecondMean, comparison.Baseline?.OutputTokensPerSecondMean))} |");
            builder.Append($" {FormatPercent(PercentDelta(summary.TtftP50Mean, comparison.Baseline?.TtftP50Mean))} |");
            builder.Append($" {FormatPercent(PercentDelta(summary.TpotP50Mean, comparison.Baseline?.TpotP50Mean))} |");
        }

        builder.AppendLine();
    }

    builder.AppendLine();
    builder.AppendLine("Repeated-run percentile cells are arithmetic means of the percentile reported by each run; they are not pooled percentiles across raw requests.");
    return builder.ToString();
}

static string RenderCsv(
    IReadOnlyList<SummaryComparison> comparisons,
    string? baseline)
{
    var builder = new StringBuilder();
    var headers = new List<string>
    {
        "engine",
        "workload",
        "endpoint",
        "model",
        "concurrency",
        "max_tokens",
        "runs",
        "successful_requests",
        "failed_requests",
        "request_throughput_mean",
        "request_throughput_stddev",
        "output_tokens_per_second_mean",
        "ttft_p50_mean_ms",
        "ttft_p95_mean_ms",
        "tpot_p50_mean_ms",
        "tpot_p95_mean_ms",
        "e2e_p95_mean_ms",
        "exact_usage_runs"
    };

    if (baseline is not null)
    {
        headers.AddRange(new[]
        {
            "request_throughput_delta_percent",
            "output_tokens_per_second_delta_percent",
            "ttft_p50_delta_percent",
            "tpot_p50_delta_percent"
        });
    }

    builder.AppendLine(string.Join(",", headers));

    foreach (var comparison in comparisons)
    {
        var summary = comparison.Summary;
        var values = new List<string>
        {
            Csv(summary.Key.Engine),
            Csv(summary.Key.Workload),
            Csv(summary.Key.Endpoint),
            Csv(summary.Key.Model),
            summary.Key.Concurrency.ToString(CultureInfo.InvariantCulture),
            summary.Key.MaxTokens.ToString(CultureInfo.InvariantCulture),
            summary.Runs.ToString(CultureInfo.InvariantCulture),
            summary.SuccessfulRequests.ToString(CultureInfo.InvariantCulture),
            summary.FailedRequests.ToString(CultureInfo.InvariantCulture),
            CsvNumber(summary.RequestsPerSecondMean),
            CsvNumber(summary.RequestsPerSecondStdDev),
            CsvNumber(summary.OutputTokensPerSecondMean),
            CsvNumber(summary.TtftP50Mean),
            CsvNumber(summary.TtftP95Mean),
            CsvNumber(summary.TpotP50Mean),
            CsvNumber(summary.TpotP95Mean),
            CsvNumber(summary.E2eP95Mean),
            summary.ExactUsageRuns.ToString(CultureInfo.InvariantCulture)
        };

        if (baseline is not null)
        {
            values.Add(CsvNumber(PercentDelta(summary.RequestsPerSecondMean, comparison.Baseline?.RequestsPerSecondMean)));
            values.Add(CsvNumber(PercentDelta(summary.OutputTokensPerSecondMean, comparison.Baseline?.OutputTokensPerSecondMean)));
            values.Add(CsvNumber(PercentDelta(summary.TtftP50Mean, comparison.Baseline?.TtftP50Mean)));
            values.Add(CsvNumber(PercentDelta(summary.TpotP50Mean, comparison.Baseline?.TpotP50Mean)));
        }

        builder.AppendLine(string.Join(",", values));
    }

    return builder.ToString();
}

static double? PercentDelta(double? value, double? baseline)
{
    if (!value.HasValue || !baseline.HasValue || baseline.Value == 0)
    {
        return null;
    }

    return ((value.Value / baseline.Value) - 1.0) * 100.0;
}

static string Format(double? value) =>
    value.HasValue
        ? value.Value.ToString("F2", CultureInfo.InvariantCulture)
        : "n/a";

static string FormatPercent(double? value) =>
    value.HasValue
        ? value.Value.ToString("+0.00;-0.00;0.00", CultureInfo.InvariantCulture) + "%"
        : "n/a";

static string CsvNumber(double? value) =>
    value.HasValue
        ? value.Value.ToString("G17", CultureInfo.InvariantCulture)
        : string.Empty;

static string Csv(string value)
{
    if (!value.Contains(',') && !value.Contains('"') && !value.Contains('\n') && !value.Contains('\r'))
    {
        return value;
    }

    return "\"" + value.Replace("\"", "\"\"", StringComparison.Ordinal) + "\"";
}

static string EscapeMarkdown(string value) =>
    value.Replace("|", "\\|", StringComparison.Ordinal);

static async Task WriteTextAsync(string path, string content)
{
    var fullPath = Path.GetFullPath(path);
    var directory = Path.GetDirectoryName(fullPath);
    if (!string.IsNullOrEmpty(directory))
    {
        Directory.CreateDirectory(directory);
    }

    await File.WriteAllTextAsync(fullPath, content);
}

internal sealed record ReporterOptions(
    string InputDirectory,
    string? MarkdownPath,
    string? CsvPath,
    string? BaselineEngine,
    bool ShowHelp)
{
    public static ReporterOptions Parse(string[] args)
    {
        if (args.Any(static arg => arg is "-h" or "--help"))
        {
            return new ReporterOptions(".", null, null, null, ShowHelp: true);
        }

        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (var index = 0; index < args.Length; index++)
        {
            var key = args[index];
            if (!key.StartsWith("--", StringComparison.Ordinal))
            {
                throw new ArgumentException($"Unexpected argument '{key}'. Use --help for usage.");
            }

            if (index + 1 >= args.Length)
            {
                throw new ArgumentException($"Missing value for '{key}'.");
            }

            values[key] = args[++index];
        }

        if (!values.TryGetValue("--input", out var input) ||
            string.IsNullOrWhiteSpace(input))
        {
            throw new ArgumentException("--input is required.");
        }

        values.TryGetValue("--markdown", out var markdown);
        values.TryGetValue("--csv", out var csv);
        values.TryGetValue("--baseline", out var baseline);

        return new ReporterOptions(input, markdown, csv, baseline, ShowHelp: false);
    }

    public static void PrintHelp()
    {
        Console.WriteLine(
            """
            Fission.Serving.Reporter - aggregate serving load-generator JSON reports

            Expected layout:
              <input>/<engine>/<workload>/<run>.json

            Usage:
              dotnet run --project benchmarks/Fission.Serving.Reporter/Fission.Serving.Reporter.csproj -c Release -- \
                --input artifacts/serving \
                --markdown artifacts/serving/report.md \
                --csv artifacts/serving/report.csv \
                --baseline fission

            Options:
              --input <directory>       Required result root
              --markdown <path>         Optional Markdown report output
              --csv <path>              Optional CSV report output
              --baseline <engine>       Optional engine label for matched percentage deltas
              --help                    Show this help
            """);
    }
}

internal sealed record LabeledRun(
    string Engine,
    string Workload,
    BenchmarkRun Report);

internal sealed record BenchmarkRun(
    string Endpoint,
    string Model,
    int TotalRequests,
    int SuccessfulRequests,
    int FailedRequests,
    int Concurrency,
    int MaxTokens,
    double RequestsPerSecond,
    double OutputTokensPerSecond,
    bool AllTokenCountsFromUsage,
    MetricSummary Ttft,
    MetricSummary Tpot,
    MetricSummary E2e)
{
    public static BenchmarkRun Load(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(path));
        var root = document.RootElement;

        return new BenchmarkRun(
            RequiredString(root, "endpoint", path),
            RequiredString(root, "model", path),
            RequiredInt(root, "total_requests", path),
            RequiredInt(root, "successful_requests", path),
            RequiredInt(root, "failed_requests", path),
            RequiredInt(root, "concurrency", path),
            RequiredInt(root, "max_tokens", path),
            RequiredDouble(root, "requests_per_second", path),
            RequiredDouble(root, "output_tokens_per_second", path),
            RequiredBool(root, "all_token_counts_from_usage", path),
            MetricSummary.Load(root, "ttft", path),
            MetricSummary.Load(root, "tpot", path),
            MetricSummary.Load(root, "e2e", path));
    }

    private static string RequiredString(JsonElement root, string name, string path) =>
        root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String
            ? value.GetString()!
            : throw Invalid(path, name);

    private static int RequiredInt(JsonElement root, string name, string path) =>
        root.TryGetProperty(name, out var value) && value.TryGetInt32(out var parsed)
            ? parsed
            : throw Invalid(path, name);

    private static double RequiredDouble(JsonElement root, string name, string path) =>
        root.TryGetProperty(name, out var value) && value.TryGetDouble(out var parsed)
            ? parsed
            : throw Invalid(path, name);

    private static bool RequiredBool(JsonElement root, string name, string path) =>
        root.TryGetProperty(name, out var value) &&
        (value.ValueKind == JsonValueKind.True || value.ValueKind == JsonValueKind.False)
            ? value.GetBoolean()
            : throw Invalid(path, name);

    private static InvalidDataException Invalid(string path, string property) =>
        new($"Report '{path}' is missing or has invalid '{property}'.");
}

internal sealed record MetricSummary(double? P50, double? P95, double? P99, double? Mean)
{
    public static MetricSummary Load(JsonElement root, string name, string path)
    {
        if (!root.TryGetProperty(name, out var metric) ||
            metric.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException($"Report '{path}' is missing metric '{name}'.");
        }

        return new MetricSummary(
            OptionalDouble(metric, "p50"),
            OptionalDouble(metric, "p95"),
            OptionalDouble(metric, "p99"),
            OptionalDouble(metric, "mean"));
    }

    private static double? OptionalDouble(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out var value) || value.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        return value.TryGetDouble(out var parsed)
            ? parsed
            : throw new InvalidDataException($"Metric property '{name}' is not numeric.");
    }
}

internal readonly record struct SummaryKey(
    string Engine,
    string Workload,
    string Endpoint,
    string Model,
    int Concurrency,
    int MaxTokens);

internal sealed record BenchmarkSummary(
    SummaryKey Key,
    int Runs,
    int SuccessfulRequests,
    int FailedRequests,
    int ExactUsageRuns,
    double RequestsPerSecondMean,
    double RequestsPerSecondStdDev,
    double OutputTokensPerSecondMean,
    double? TtftP50Mean,
    double? TtftP95Mean,
    double? TpotP50Mean,
    double? TpotP95Mean,
    double? E2eP95Mean)
{
    public static BenchmarkSummary Create(
        SummaryKey key,
        IReadOnlyList<BenchmarkRun> runs)
    {
        return new BenchmarkSummary(
            key,
            runs.Count,
            runs.Sum(static run => run.SuccessfulRequests),
            runs.Sum(static run => run.FailedRequests),
            runs.Count(static run => run.AllTokenCountsFromUsage),
            Mean(runs.Select(static run => run.RequestsPerSecond))!.Value,
            StdDev(runs.Select(static run => run.RequestsPerSecond)),
            Mean(runs.Select(static run => run.OutputTokensPerSecond))!.Value,
            Mean(runs.Select(static run => run.Ttft.P50)),
            Mean(runs.Select(static run => run.Ttft.P95)),
            Mean(runs.Select(static run => run.Tpot.P50)),
            Mean(runs.Select(static run => run.Tpot.P95)),
            Mean(runs.Select(static run => run.E2e.P95)));
    }

    private static double? Mean(IEnumerable<double?> values)
    {
        var materialized = values.Where(static value => value.HasValue)
            .Select(static value => value!.Value)
            .ToArray();

        return materialized.Length == 0 ? null : materialized.Average();
    }

    private static double? Mean(IEnumerable<double> values)
    {
        var materialized = values.ToArray();
        return materialized.Length == 0 ? null : materialized.Average();
    }

    private static double StdDev(IEnumerable<double> values)
    {
        var materialized = values.ToArray();
        if (materialized.Length <= 1)
        {
            return 0;
        }

        var mean = materialized.Average();
        var variance = materialized.Sum(value => Math.Pow(value - mean, 2)) / materialized.Length;
        return Math.Sqrt(variance);
    }
}

internal sealed record SummaryComparison(
    BenchmarkSummary Summary,
    BenchmarkSummary? Baseline);
