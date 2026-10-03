using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Fission.Abstractions;
using Fission.Engine;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;

namespace Fission.Server;

public static class ServerControlEndpoints
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    public const string ControlTokenHeader = "X-Fission-Control-Token";
    public const string ShutdownPath = "/internal/control/shutdown";
    public const string GeneratePath = "/internal/control/generate";

    public static void Map(
        WebApplication app,
        IConfiguration configuration,
        InferenceWorker worker,
        ITextTokenCodec codec)
    {
        ArgumentNullException.ThrowIfNull(app);
        ArgumentNullException.ThrowIfNull(configuration);
        ArgumentNullException.ThrowIfNull(worker);
        ArgumentNullException.ThrowIfNull(codec);

        var expectedToken = configuration["Fission:ControlToken"];
        if (string.IsNullOrWhiteSpace(expectedToken))
        {
            return;
        }

        expectedToken = expectedToken.Trim();
        app.MapPost(
            ShutdownPath,
            (HttpContext context, IHostApplicationLifetime lifetime) =>
            {
                if (!IsAuthorized(context, expectedToken))
                {
                    return Results.Unauthorized();
                }

                context.Response.OnCompleted(() =>
                {
                    lifetime.StopApplication();
                    return Task.CompletedTask;
                });

                return Results.Accepted(
                    ShutdownPath,
                    new { status = "stopping" });
            });

        app.MapPost(
            GeneratePath,
            async (HttpContext context, ControlGenerateRequest request) =>
            {
                if (!IsAuthorized(context, expectedToken))
                {
                    return Results.Unauthorized();
                }

                if (string.IsNullOrWhiteSpace(request.Model))
                {
                    return Results.BadRequest(new { error = "model is required." });
                }
                if (request.MaxTokens <= 0)
                {
                    return Results.BadRequest(new { error = "max_tokens must be positive." });
                }

                int[] promptTokens;
                try
                {
                    promptTokens = codec.EncodePrompt(request.Prompt ?? string.Empty);
                }
                catch (ArgumentException exception)
                {
                    return Results.BadRequest(new { error = exception.Message });
                }

                var stream = await worker.SubmitAsync(
                    new ModelId(request.Model),
                    promptTokens,
                    request.MaxTokens,
                    request.Priority,
                    cancellationToken: context.RequestAborted).ConfigureAwait(false);

                var text = new StringBuilder();
                using var decoder = codec.CreateDecoder();
                var completed = false;
                try
                {
                    await foreach (var token in stream.ReadTokensAsync(context.RequestAborted))
                    {
                        text.Append(decoder.Append(token));
                    }
                    text.Append(decoder.Complete());

                    var snapshot = await stream.Completion.WaitAsync(context.RequestAborted)
                        .ConfigureAwait(false);
                    completed = true;

                    return Results.Json(
                        new
                        {
                            model = request.Model,
                            prompt_tokens = promptTokens.Length,
                            token_ids = snapshot.GeneratedTokens.ToArray(),
                            text = text.ToString(),
                            finish_reason = FinishReason(snapshot),
                            completion_tokens = snapshot.GeneratedTokens.Count
                        },
                        JsonOptions);
                }
                finally
                {
                    if (!completed && !stream.Completion.IsCompleted)
                    {
                        try
                        {
                            await stream.CancelAsync(CancellationToken.None).ConfigureAwait(false);
                        }
                        catch
                        {
                            // Best-effort cleanup for an abandoned control request.
                        }
                    }
                }
            });
    }

    private static bool IsAuthorized(HttpContext context, string expectedToken)
    {
        var providedToken = context.Request.Headers[ControlTokenHeader].ToString();
        return TokensEqual(expectedToken, providedToken);
    }

    private static string FinishReason(InferenceRequestSnapshot snapshot) =>
        snapshot.FinishReason switch
        {
            InferenceFinishReason.Stop => "stop",
            InferenceFinishReason.Length => "length",
            InferenceFinishReason.Cancelled => "cancelled",
            _ => throw new InvalidOperationException(
                $"Completed request {snapshot.SequenceId} has no finish reason.")
        };

    private static bool TokensEqual(
        string expected,
        string provided)
    {
        if (string.IsNullOrEmpty(provided))
        {
            return false;
        }

        var expectedBytes = Encoding.UTF8.GetBytes(expected);
        var providedBytes = Encoding.UTF8.GetBytes(provided);
        return expectedBytes.Length == providedBytes.Length &&
            CryptographicOperations.FixedTimeEquals(
                expectedBytes,
                providedBytes);
    }
}

public sealed record ControlGenerateRequest(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("prompt")] string? Prompt,
    [property: JsonPropertyName("max_tokens")] int MaxTokens,
    [property: JsonPropertyName("priority")] int Priority = 0);
