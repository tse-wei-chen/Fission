using System.Diagnostics;
using System.Text;
using Fission.Abstractions;
using Fission.Engine;
using Microsoft.Extensions.Configuration;

namespace Fission.Server;

public sealed record StartupInferenceProbeResult(
    ModelId ModelId,
    int PromptTokenCount,
    int GeneratedTokenCount,
    InferenceFinishReason FinishReason,
    TimeSpan Elapsed,
    string DecodedText);

/// <summary>
/// Optional startup inference probe that exercises the configured tokenizer,
/// worker, scheduler/runtime, backend, and request-scoped decoder before the
/// HTTP server begins accepting traffic.
/// </summary>
public static class ServerStartupProbe
{
    public static async ValueTask<StartupInferenceProbeResult?> RunAsync(
        IConfiguration configuration,
        InferenceWorker worker,
        ITextTokenCodec codec,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(configuration);
        ArgumentNullException.ThrowIfNull(worker);
        ArgumentNullException.ThrowIfNull(codec);

        if (!ReadBoolean(configuration, "Fission:StartupProbeEnabled", fallback: false))
        {
            return null;
        }

        var prompt = ReadRequired(configuration, "Fission:StartupProbePrompt");
        var modelId = ReadOptional(configuration, "Fission:StartupProbeModelId")
            ?? ReadRequired(configuration, "Fission:ModelId");
        var maxNewTokens = ReadPositiveInt(
            configuration,
            "Fission:StartupProbeMaxTokens",
            fallback: 1);
        var timeoutSeconds = ReadPositiveInt(
            configuration,
            "Fission:StartupProbeTimeoutSeconds",
            fallback: 60);

        var promptTokens = codec.EncodePrompt(prompt);
        if (promptTokens.Length == 0)
        {
            throw new InvalidOperationException(
                "The startup inference probe prompt encoded to zero tokens.");
        }

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(TimeSpan.FromSeconds(timeoutSeconds));

        var started = Stopwatch.GetTimestamp();
        InferenceStream? stream = null;

        try
        {
            stream = await worker.SubmitAsync(
                new ModelId(modelId),
                promptTokens,
                maxNewTokens,
                cancellationToken: timeout.Token).ConfigureAwait(false);

            var snapshot = await stream.Completion
                .WaitAsync(timeout.Token)
                .ConfigureAwait(false);

            if (!snapshot.IsCompleted || snapshot.FinishReason is not { } finishReason)
            {
                throw new InvalidOperationException(
                    "The startup inference probe completed without a terminal inference reason.");
            }

            if (snapshot.GeneratedTokens.Count == 0)
            {
                throw new InvalidOperationException(
                    "The startup inference probe completed without generating a token.");
            }

            using var decoder = codec.CreateDecoder();
            var decoded = new StringBuilder();
            foreach (var tokenId in snapshot.GeneratedTokens)
            {
                decoded.Append(decoder.Append(tokenId));
            }

            decoded.Append(decoder.Complete());

            return new StartupInferenceProbeResult(
                snapshot.ModelId,
                snapshot.PromptTokenCount,
                snapshot.GeneratedTokens.Count,
                finishReason,
                Stopwatch.GetElapsedTime(started),
                decoded.ToString());
        }
        catch (OperationCanceledException exception)
            when (!cancellationToken.IsCancellationRequested && timeout.IsCancellationRequested)
        {
            if (stream is not null && !stream.Completion.IsCompleted)
            {
                _ = TryCancelAsync(stream);
            }

            throw new TimeoutException(
                $"Startup inference probe exceeded {timeoutSeconds} second(s).",
                exception);
        }
    }

    private static async Task TryCancelAsync(InferenceStream stream)
    {
        try
        {
            await stream.CancelAsync().ConfigureAwait(false);
        }
        catch
        {
            // Startup is already failing. Best-effort cancellation only ensures
            // an admitted probe does not outlive a recoverable timeout.
        }
    }

    private static string ReadRequired(
        IConfiguration configuration,
        string key) =>
        ReadOptional(configuration, key) ??
        throw new InvalidOperationException(
            $"Configuration value '{key}' is required when the startup inference probe is enabled.");

    private static string? ReadOptional(
        IConfiguration configuration,
        string key)
    {
        var value = configuration[key];
        return string.IsNullOrWhiteSpace(value) ? null : value.Trim();
    }

    private static bool ReadBoolean(
        IConfiguration configuration,
        string key,
        bool fallback)
    {
        var value = configuration[key];
        if (string.IsNullOrWhiteSpace(value))
        {
            return fallback;
        }

        if (bool.TryParse(value, out var parsed))
        {
            return parsed;
        }

        throw new InvalidOperationException(
            $"Configuration value '{key}' must be 'true' or 'false'.");
    }

    private static int ReadPositiveInt(
        IConfiguration configuration,
        string key,
        int fallback)
    {
        var value = configuration[key];
        if (string.IsNullOrWhiteSpace(value))
        {
            return fallback;
        }

        if (int.TryParse(value, out var parsed) && parsed > 0)
        {
            return parsed;
        }

        throw new InvalidOperationException(
            $"Configuration value '{key}' must be a positive integer.");
    }
}
