using System.Diagnostics;
using System.Globalization;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;

try
{
    var options = BenchmarkOptions.Parse(args);
    if (options.ShowHelp)
    {
        BenchmarkOptions.PrintHelp();
        return 0;
    }

    using var http = new HttpClient
    {
        Timeout = Timeout.InfiniteTimeSpan
    };
    http.DefaultRequestHeaders.UserAgent.ParseAdd("Fission.Serving.LoadGen/1.0");

    if (!string.IsNullOrWhiteSpace(options.ApiKey))
    {
        http.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Bearer", options.ApiKey);
    }

    Console.WriteLine(
        $"Target: {options.BaseUrl} | endpoint={options.Endpoint.ToString().ToLowerInvariant()} | " +
        $"requests={options.Requests} | concurrency={options.Concurrency} | max_tokens={options.MaxTokens}");

    if (options.WarmupRequests > 0)
    {
        Console.WriteLine($"Warmup: {options.WarmupRequests} request(s)");
        var warmup = await RunBatchAsync(
            http,
            options,
            options.WarmupRequests,
            Math.Min(options.Concurrency, options.WarmupRequests),
            CancellationToken.None);

        var warmupFailures = warmup.Measurements.Count(static item => !item.Success);
        if (warmupFailures > 0)
        {
            Console.Error.WriteLine(
                $"Warmup failed: {warmupFailures}/{warmup.Measurements.Length} request(s) failed.");
            PrintFailures(warmup.Measurements);
            return 2;
        }
    }

    Console.WriteLine("Measured run starting...");
    var measured = await RunBatchAsync(
        http,
        options,
        options.Requests,
        options.Concurrency,
        CancellationToken.None);

    var report = BenchmarkReport.Create(options, measured);
    PrintReport(report, measured.Measurements);

    if (!string.IsNullOrWhiteSpace(options.OutputPath))
    {
        var outputPath = Path.GetFullPath(options.OutputPath);
        var directory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        await File.WriteAllTextAsync(
            outputPath,
            JsonSerializer.Serialize(report, JsonOptions.Indented));
        Console.WriteLine($"JSON: {outputPath}");
    }

    return report.FailedRequests == 0 ? 0 : 3;
}
catch (OperationCanceledException)
{
    Console.Error.WriteLine("Benchmark cancelled.");
    return 130;
}
catch (Exception exception)
{
    Console.Error.WriteLine(exception.Message);
    return 2;
}

static async Task<BatchResult> RunBatchAsync(
    HttpClient http,
    BenchmarkOptions options,
    int requestCount,
    int concurrency,
    CancellationToken cancellationToken)
{
    var measurements = new RequestMeasurement[requestCount];
    var started = Stopwatch.GetTimestamp();

    await Parallel.ForEachAsync(
        Enumerable.Range(0, requestCount),
        new ParallelOptions
        {
            MaxDegreeOfParallelism = concurrency,
            CancellationToken = cancellationToken
        },
        async (index, token) =>
        {
            measurements[index] = await RunRequestAsync(http, options, token)
                .ConfigureAwait(false);
        }).ConfigureAwait(false);

    return new BatchResult(
        measurements,
        Stopwatch.GetElapsedTime(started));
}

static async ValueTask<RequestMeasurement> RunRequestAsync(
    HttpClient http,
    BenchmarkOptions options,
    CancellationToken benchmarkToken)
{
    using var timeout = CancellationTokenSource.CreateLinkedTokenSource(benchmarkToken);
    timeout.CancelAfter(TimeSpan.FromSeconds(options.TimeoutSeconds));

    var endpoint = options.Endpoint == EndpointKind.Chat
        ? "/v1/chat/completions"
        : "/v1/completions";
    var uri = new Uri(options.BaseUrl, endpoint);

    object payload = options.Endpoint == EndpointKind.Chat
        ? new
        {
            model = options.Model,
            messages = new[]
            {
                new { role = "user", content = options.Prompt }
            },
            max_completion_tokens = options.MaxTokens,
            stream = true,
            stream_options = new { include_usage = true }
        }
        : new
        {
            model = options.Model,
            prompt = options.Prompt,
            max_tokens = options.MaxTokens,
            stream = true,
            stream_options = new { include_usage = true }
        };

    using var request = new HttpRequestMessage(HttpMethod.Post, uri)
    {
        Content = JsonContent.Create(payload)
    };

    var started = Stopwatch.GetTimestamp();

    try
    {
        using var response = await http.SendAsync(
            request,
            HttpCompletionOption.ResponseHeadersRead,
            timeout.Token).ConfigureAwait(false);

        if (!response.IsSuccessStatusCode)
        {
            var body = await response.Content.ReadAsStringAsync(timeout.Token)
                .ConfigureAwait(false);
            return RequestMeasurement.Failed(
                (int)response.StatusCode,
                Stopwatch.GetElapsedTime(started).TotalMilliseconds,
                $"HTTP {(int)response.StatusCode}: {Trim(body, 400)}");
        }

        await using var stream = await response.Content.ReadAsStreamAsync(timeout.Token)
            .ConfigureAwait(false);
        using var reader = new StreamReader(stream);

        double? ttftMs = null;
        var contentChunks = 0;
        int? usageCompletionTokens = null;
        var sawDone = false;

        while (await reader.ReadLineAsync(timeout.Token).ConfigureAwait(false) is { } line)
        {
            if (!line.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            var data = line[5..].Trim();
            if (data.Length == 0)
            {
                continue;
            }

            if (string.Equals(data, "[DONE]", StringComparison.Ordinal))
            {
                sawDone = true;
                break;
            }

            using var document = JsonDocument.Parse(data);
            var root = document.RootElement;

            if (TryGetCompletionTokens(root, out var reported))
            {
                usageCompletionTokens = reported;
            }

            var content = ExtractContent(root, options.Endpoint);
            if (!string.IsNullOrEmpty(content))
            {
                contentChunks++;
                ttftMs ??= Stopwatch.GetElapsedTime(started).TotalMilliseconds;
            }
        }

        var e2eMs = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
        if (!sawDone)
        {
            return RequestMeasurement.Failed(
                (int)response.StatusCode,
                e2eMs,
                "SSE stream ended before data: [DONE].");
        }

        var completionTokens = usageCompletionTokens ?? contentChunks;
        return new RequestMeasurement(
            Success: true,
            StatusCode: (int)response.StatusCode,
            E2eMs: e2eMs,
            TtftMs: ttftMs,
            CompletionTokens: completionTokens,
            UsageReported: usageCompletionTokens.HasValue,
            Error: null);
    }
    catch (OperationCanceledException) when (!benchmarkToken.IsCancellationRequested)
    {
        return RequestMeasurement.Failed(
            0,
            Stopwatch.GetElapsedTime(started).TotalMilliseconds,
            $"Request timed out after {options.TimeoutSeconds}s.");
    }
    catch (Exception exception)
    {
        return RequestMeasurement.Failed(
            0,
            Stopwatch.GetElapsedTime(started).TotalMilliseconds,
            $"{exception.GetType().Name}: {exception.Message}");
    }
}

static string? ExtractContent(JsonElement root, EndpointKind endpoint)
{
    if (!root.TryGetProperty("choices", out var choices) ||
        choices.ValueKind != JsonValueKind.Array ||
        choices.GetArrayLength() == 0)
    {
        return null;
    }

    var choice = choices[0];
    if (endpoint == EndpointKind.Completions)
    {
        return choice.TryGetProperty("text", out var text) &&
               text.ValueKind == JsonValueKind.String
            ? text.GetString()
            : null;
    }

    if (!choice.TryGetProperty("delta", out var delta) ||
        delta.ValueKind != JsonValueKind.Object)
    {
        return null;
    }

    return delta.TryGetProperty("content", out var content) &&
           content.ValueKind == JsonValueKind.String
        ? content.GetString()
        : null;
}

static bool TryGetCompletionTokens(JsonElement root, out int completionTokens)
{
    completionTokens = 0;
    return root.TryGetProperty("usage", out var usage) &&
           usage.ValueKind == JsonValueKind.Object &&
           usage.TryGetProperty("completion_tokens", out var value) &&
           value.TryGetInt32(out completionTokens);
}

static void PrintReport(BenchmarkReport report, IReadOnlyList<RequestMeasurement> measurements)
{
    Console.WriteLine();
    Console.WriteLine("Fission serving benchmark");
    Console.WriteLine(
        $"completed={report.SuccessfulRequests}/{report.TotalRequests} " +
        $"failed={report.FailedRequests} wall={report.WallSeconds:F3}s");
    Console.WriteLine(
        $"request_throughput={report.RequestsPerSecond:F2} req/s " +
        $"output_throughput={report.OutputTokensPerSecond:F2} tok/s " +
        $"usage_reported={report.UsageReportedRequests}/{report.SuccessfulRequests}");
    Console.WriteLine();
    Console.WriteLine("metric\tp50\tp95\tp99\tmean");
    PrintMetric("TTFT ms", report.Ttft);
    PrintMetric("TPOT ms", report.Tpot);
    PrintMetric("E2E ms", report.E2e);

    if (!report.AllTokenCountsFromUsage)
    {
        Console.WriteLine();
        Console.WriteLine(
            "Note: one or more responses omitted usage.completion_tokens; " +
            "those requests use non-empty SSE content chunks as an approximate token count.");
    }

    if (report.FailedRequests > 0)
    {
        Console.WriteLine();
        PrintFailures(measurements);
    }
}

static void PrintMetric(string name, MetricSummary metric) =>
    Console.WriteLine(
        $"{name}\t{Format(metric.P50)}\t{Format(metric.P95)}\t{Format(metric.P99)}\t{Format(metric.Mean)}");

static string Format(double? value) =>
    value.HasValue
        ? value.Value.ToString("F2", CultureInfo.InvariantCulture)
        : "n/a";

static void PrintFailures(IEnumerable<RequestMeasurement> measurements)
{
    foreach (var failure in measurements.Where(static item => !item.Success).Take(5))
    {
        Console.Error.WriteLine($"  failure: {failure.Error}");
    }
}

static string Trim(string value, int maxLength) =>
    value.Length <= maxLength
        ? value
        : value[..maxLength] + "...";

internal enum EndpointKind
{
    Completions,
    Chat
}

internal sealed record BenchmarkOptions(
    Uri BaseUrl,
    string Model,
    int Requests,
    int Concurrency,
    int MaxTokens,
    int WarmupRequests,
    int TimeoutSeconds,
    string Prompt,
    EndpointKind Endpoint,
    string? OutputPath,
    string? ApiKey,
    bool ShowHelp)
{
    public static BenchmarkOptions Parse(string[] args)
    {
        if (args.Any(static arg => arg is "-h" or "--help"))
        {
            return Defaults() with { ShowHelp = true };
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

        var defaults = Defaults();
        var baseUrl = new Uri(Get(values, "--url", defaults.BaseUrl.ToString()), UriKind.Absolute);
        var model = Get(values, "--model", defaults.Model);
        var requests = PositiveInt(values, "--requests", defaults.Requests);
        var concurrency = PositiveInt(values, "--concurrency", defaults.Concurrency);
        var maxTokens = PositiveInt(values, "--max-tokens", defaults.MaxTokens);
        var warmup = NonNegativeInt(values, "--warmup", defaults.WarmupRequests);
        var timeout = PositiveInt(values, "--timeout-seconds", defaults.TimeoutSeconds);
        var output = GetOptional(values, "--output");
        var apiKey = GetOptional(values, "--api-key") ??
                     Environment.GetEnvironmentVariable("OPENAI_API_KEY");

        var endpointText = Get(values, "--endpoint", "completions");
        var endpoint = endpointText.ToLowerInvariant() switch
        {
            "completions" => EndpointKind.Completions,
            "chat" => EndpointKind.Chat,
            _ => throw new ArgumentException(
                "--endpoint must be either 'completions' or 'chat'.")
        };

        var promptFile = GetOptional(values, "--prompt-file");
        var prompt = promptFile is null
            ? Get(values, "--prompt", defaults.Prompt)
            : File.ReadAllText(promptFile);

        if (string.IsNullOrWhiteSpace(model))
        {
            throw new ArgumentException("--model cannot be empty.");
        }

        if (string.IsNullOrWhiteSpace(prompt))
        {
            throw new ArgumentException("Prompt cannot be empty.");
        }

        return new BenchmarkOptions(
            NormalizeBaseUrl(baseUrl),
            model,
            requests,
            concurrency,
            maxTokens,
            warmup,
            timeout,
            prompt,
            endpoint,
            output,
            apiKey,
            ShowHelp: false);
    }

    public static void PrintHelp()
    {
        Console.WriteLine(
            """
            Fission.Serving.LoadGen - OpenAI-compatible streaming serving benchmark

            Usage:
              dotnet run --project benchmarks/Fission.Serving.LoadGen/Fission.Serving.LoadGen.csproj -c Release -- [options]

            Options:
              --url <url>               Base URL. Default: http://127.0.0.1:8000
              --model <name>            Model name. Default: benchmark
              --endpoint <kind>         completions | chat. Default: completions
              --requests <n>            Measured requests. Default: 100
              --concurrency <n>         Maximum in-flight requests. Default: 16
              --max-tokens <n>          Requested output tokens. Default: 64
              --warmup <n>              Warmup requests. Default: 4
              --timeout-seconds <n>     Per-request timeout. Default: 300
              --prompt <text>           Inline prompt
              --prompt-file <path>      Read prompt from UTF-8 file
              --api-key <key>           Bearer token; otherwise OPENAI_API_KEY is used
              --output <path>           Write JSON report
              --help                    Show this help

            Metrics:
              TTFT = request start to first non-empty streamed content
              TPOT = (E2E - TTFT) / (completion_tokens - 1)
              E2E  = request start to data: [DONE]
            """);
    }

    private static BenchmarkOptions Defaults() =>
        new(
            new Uri("http://127.0.0.1:8000/"),
            "benchmark",
            Requests: 100,
            Concurrency: 16,
            MaxTokens: 64,
            WarmupRequests: 4,
            TimeoutSeconds: 300,
            Prompt: "Explain why continuous batching improves LLM serving throughput.",
            EndpointKind.Completions,
            OutputPath: null,
            ApiKey: null,
            ShowHelp: false);

    private static Uri NormalizeBaseUrl(Uri value)
    {
        var text = value.ToString();
        return new Uri(text.EndsWith("/", StringComparison.Ordinal) ? text : text + "/");
    }

    private static string Get(
        IReadOnlyDictionary<string, string> values,
        string key,
        string fallback) =>
        values.TryGetValue(key, out var value) ? value : fallback;

    private static string? GetOptional(
        IReadOnlyDictionary<string, string> values,
        string key) =>
        values.TryGetValue(key, out var value) ? value : null;

    private static int PositiveInt(
        IReadOnlyDictionary<string, string> values,
        string key,
        int fallback)
    {
        var value = ParseInt(values, key, fallback);
        return value > 0
            ? value
            : throw new ArgumentOutOfRangeException(key, "Value must be positive.");
    }

    private static int NonNegativeInt(
        IReadOnlyDictionary<string, string> values,
        string key,
        int fallback)
    {
        var value = ParseInt(values, key, fallback);
        return value >= 0
            ? value
            : throw new ArgumentOutOfRangeException(key, "Value cannot be negative.");
    }

    private static int ParseInt(
        IReadOnlyDictionary<string, string> values,
        string key,
        int fallback)
    {
        if (!values.TryGetValue(key, out var text))
        {
            return fallback;
        }

        return int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var value)
            ? value
            : throw new ArgumentException($"'{text}' is not a valid integer for {key}.");
    }
}

internal readonly record struct RequestMeasurement(
    bool Success,
    int StatusCode,
    double E2eMs,
    double? TtftMs,
    int CompletionTokens,
    bool UsageReported,
    string? Error)
{
    public static RequestMeasurement Failed(
        int statusCode,
        double e2eMs,
        string error) =>
        new(
            Success: false,
            StatusCode: statusCode,
            E2eMs: e2eMs,
            TtftMs: null,
            CompletionTokens: 0,
            UsageReported: false,
            Error: error);
}

internal readonly record struct BatchResult(
    RequestMeasurement[] Measurements,
    TimeSpan WallTime);

internal sealed record MetricSummary(
    double? P50,
    double? P95,
    double? P99,
    double? Mean)
{
    public static MetricSummary From(IEnumerable<double> values)
    {
        var sorted = values.Order().ToArray();
        if (sorted.Length == 0)
        {
            return new MetricSummary(null, null, null, null);
        }

        return new MetricSummary(
            Percentile(sorted, 0.50),
            Percentile(sorted, 0.95),
            Percentile(sorted, 0.99),
            sorted.Average());
    }

    private static double Percentile(IReadOnlyList<double> sorted, double percentile)
    {
        if (sorted.Count == 1)
        {
            return sorted[0];
        }

        var position = percentile * (sorted.Count - 1);
        var lower = (int)Math.Floor(position);
        var upper = (int)Math.Ceiling(position);
        if (lower == upper)
        {
            return sorted[lower];
        }

        var fraction = position - lower;
        return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
    }
}

internal sealed record BenchmarkReport(
    DateTimeOffset GeneratedAt,
    string BaseUrl,
    string Endpoint,
    string Model,
    int TotalRequests,
    int SuccessfulRequests,
    int FailedRequests,
    int Concurrency,
    int MaxTokens,
    double WallSeconds,
    double RequestsPerSecond,
    double OutputTokensPerSecond,
    long CompletionTokens,
    int UsageReportedRequests,
    bool AllTokenCountsFromUsage,
    MetricSummary Ttft,
    MetricSummary Tpot,
    MetricSummary E2e)
{
    public static BenchmarkReport Create(
        BenchmarkOptions options,
        BatchResult batch)
    {
        var successful = batch.Measurements
            .Where(static item => item.Success)
            .ToArray();
        var completionTokens = successful.Sum(static item => (long)item.CompletionTokens);
        var wallSeconds = batch.WallTime.TotalSeconds;
        var usageReported = successful.Count(static item => item.UsageReported);

        var tpot = successful
            .Where(static item =>
                item.TtftMs.HasValue &&
                item.CompletionTokens > 1 &&
                item.E2eMs >= item.TtftMs.Value)
            .Select(static item =>
                (item.E2eMs - item.TtftMs!.Value) / (item.CompletionTokens - 1));

        return new BenchmarkReport(
            DateTimeOffset.UtcNow,
            options.BaseUrl.ToString(),
            options.Endpoint.ToString().ToLowerInvariant(),
            options.Model,
            batch.Measurements.Length,
            successful.Length,
            batch.Measurements.Length - successful.Length,
            options.Concurrency,
            options.MaxTokens,
            wallSeconds,
            wallSeconds > 0 ? successful.Length / wallSeconds : 0,
            wallSeconds > 0 ? completionTokens / wallSeconds : 0,
            completionTokens,
            usageReported,
            successful.Length > 0 && usageReported == successful.Length,
            MetricSummary.From(
                successful
                    .Where(static item => item.TtftMs.HasValue)
                    .Select(static item => item.TtftMs!.Value)),
            MetricSummary.From(tpot),
            MetricSummary.From(successful.Select(static item => item.E2eMs)));
    }
}

internal static class JsonOptions
{
    public static readonly JsonSerializerOptions Indented = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };
}
