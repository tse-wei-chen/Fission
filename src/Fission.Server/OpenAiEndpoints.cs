using System.Text;
using System.Text.Json;
using Fission.Abstractions;
using Fission.Engine;

namespace Fission.Server;

public static class OpenAiEndpoints
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    public static void Map(
        WebApplication app,
        InferenceWorker worker,
        ITextTokenCodec codec)
    {
        ArgumentNullException.ThrowIfNull(app);
        ArgumentNullException.ThrowIfNull(worker);
        ArgumentNullException.ThrowIfNull(codec);

        app.MapGet("/healthz", () => Results.Json(new { status = "ok" }, JsonOptions));
        app.MapPost(
            "/v1/completions",
            (HttpContext context, OpenAiCompletionRequest request) =>
                HandleCompletionAsync(context, request, worker, codec));
        app.MapPost(
            "/v1/chat/completions",
            (HttpContext context, OpenAiChatCompletionRequest request) =>
                HandleChatCompletionAsync(context, request, worker, codec));
    }

    private static async Task HandleCompletionAsync(
        HttpContext context,
        OpenAiCompletionRequest request,
        InferenceWorker worker,
        ITextTokenCodec codec)
    {
        if (!TryValidateModelAndLimit(
                request.Model,
                request.MaxTokens,
                out var maxTokens,
                out var error))
        {
            await WriteBadRequestAsync(context, error).ConfigureAwait(false);
            return;
        }

        int[] promptTokens;
        try
        {
            promptTokens = codec.EncodePrompt(request.Prompt);
        }
        catch (ArgumentException exception)
        {
            await WriteBadRequestAsync(context, exception.Message).ConfigureAwait(false);
            return;
        }

        var stream = await worker.SubmitAsync(
            new ModelId(request.Model),
            promptTokens,
            maxTokens,
            request.Priority ?? 0,
            cancellationToken: context.RequestAborted).ConfigureAwait(false);

        if (request.Stream)
        {
            await StreamCompletionAsync(
                context,
                request.Model,
                promptTokens.Length,
                stream,
                codec).ConfigureAwait(false);
            return;
        }

        await WriteCompletionAsync(
            context,
            request.Model,
            promptTokens.Length,
            stream,
            codec).ConfigureAwait(false);
    }

    private static async Task HandleChatCompletionAsync(
        HttpContext context,
        OpenAiChatCompletionRequest request,
        InferenceWorker worker,
        ITextTokenCodec codec)
    {
        var requestedLimit = request.MaxCompletionTokens ?? request.MaxTokens;
        if (!TryValidateModelAndLimit(
                request.Model,
                requestedLimit,
                out var maxTokens,
                out var error))
        {
            await WriteBadRequestAsync(context, error).ConfigureAwait(false);
            return;
        }

        int[] promptTokens;
        try
        {
            promptTokens = codec.EncodeChat(request.Messages);
        }
        catch (ArgumentException exception)
        {
            await WriteBadRequestAsync(context, exception.Message).ConfigureAwait(false);
            return;
        }

        var stream = await worker.SubmitAsync(
            new ModelId(request.Model),
            promptTokens,
            maxTokens,
            request.Priority ?? 0,
            cancellationToken: context.RequestAborted).ConfigureAwait(false);

        if (request.Stream)
        {
            await StreamChatCompletionAsync(
                context,
                request.Model,
                promptTokens.Length,
                stream,
                codec).ConfigureAwait(false);
            return;
        }

        await WriteChatCompletionAsync(
            context,
            request.Model,
            promptTokens.Length,
            stream,
            codec).ConfigureAwait(false);
    }

    private static async Task WriteCompletionAsync(
        HttpContext context,
        string model,
        int promptTokenCount,
        InferenceStream stream,
        ITextTokenCodec codec)
    {
        var text = new StringBuilder();
        var completed = false;

        try
        {
            await foreach (var token in stream.ReadTokensAsync(context.RequestAborted))
            {
                text.Append(codec.DecodeToken(token));
            }

            var snapshot = await stream.Completion.WaitAsync(context.RequestAborted)
                .ConfigureAwait(false);
            completed = true;

            await Results.Json(
                new
                {
                    id = CompletionId("cmpl", snapshot),
                    @object = "text_completion",
                    created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    model,
                    choices = new[]
                    {
                        new
                        {
                            text = text.ToString(),
                            index = 0,
                            logprobs = (object?)null,
                            finish_reason = FinishReason(snapshot)
                        }
                    },
                    usage = Usage(promptTokenCount, snapshot.GeneratedTokens.Count)
                },
                JsonOptions).ExecuteAsync(context).ConfigureAwait(false);
        }
        finally
        {
            await CancelIfAbandonedAsync(stream, completed).ConfigureAwait(false);
        }
    }

    private static async Task WriteChatCompletionAsync(
        HttpContext context,
        string model,
        int promptTokenCount,
        InferenceStream stream,
        ITextTokenCodec codec)
    {
        var text = new StringBuilder();
        var completed = false;

        try
        {
            await foreach (var token in stream.ReadTokensAsync(context.RequestAborted))
            {
                text.Append(codec.DecodeToken(token));
            }

            var snapshot = await stream.Completion.WaitAsync(context.RequestAborted)
                .ConfigureAwait(false);
            completed = true;

            await Results.Json(
                new
                {
                    id = CompletionId("chatcmpl", snapshot),
                    @object = "chat.completion",
                    created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    model,
                    choices = new[]
                    {
                        new
                        {
                            index = 0,
                            message = new { role = "assistant", content = text.ToString() },
                            finish_reason = FinishReason(snapshot)
                        }
                    },
                    usage = Usage(promptTokenCount, snapshot.GeneratedTokens.Count)
                },
                JsonOptions).ExecuteAsync(context).ConfigureAwait(false);
        }
        finally
        {
            await CancelIfAbandonedAsync(stream, completed).ConfigureAwait(false);
        }
    }

    private static async Task StreamCompletionAsync(
        HttpContext context,
        string model,
        int promptTokenCount,
        InferenceStream stream,
        ITextTokenCodec codec)
    {
        PrepareSse(context.Response);
        var id = $"cmpl-{stream.SequenceId.Value:N}";
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var completed = false;

        try
        {
            await foreach (var token in stream.ReadTokensAsync(context.RequestAborted))
            {
                await WriteSseAsync(
                    context.Response,
                    new
                    {
                        id,
                        @object = "text_completion",
                        created,
                        model,
                        choices = new[]
                        {
                            new
                            {
                                text = codec.DecodeToken(token),
                                index = 0,
                                logprobs = (object?)null,
                                finish_reason = (string?)null
                            }
                        }
                    },
                    context.RequestAborted).ConfigureAwait(false);
            }

            var snapshot = await stream.Completion.WaitAsync(context.RequestAborted)
                .ConfigureAwait(false);

            await WriteSseAsync(
                context.Response,
                new
                {
                    id,
                    @object = "text_completion",
                    created,
                    model,
                    choices = new[]
                    {
                        new
                        {
                            text = string.Empty,
                            index = 0,
                            logprobs = (object?)null,
                            finish_reason = FinishReason(snapshot)
                        }
                    },
                    usage = Usage(promptTokenCount, snapshot.GeneratedTokens.Count)
                },
                context.RequestAborted).ConfigureAwait(false);
            await WriteDoneAsync(context.Response, context.RequestAborted).ConfigureAwait(false);
            completed = true;
        }
        finally
        {
            await CancelIfAbandonedAsync(stream, completed).ConfigureAwait(false);
        }
    }

    private static async Task StreamChatCompletionAsync(
        HttpContext context,
        string model,
        int promptTokenCount,
        InferenceStream stream,
        ITextTokenCodec codec)
    {
        PrepareSse(context.Response);
        var id = $"chatcmpl-{stream.SequenceId.Value:N}";
        var created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var completed = false;

        try
        {
            await WriteSseAsync(
                context.Response,
                new
                {
                    id,
                    @object = "chat.completion.chunk",
                    created,
                    model,
                    choices = new[]
                    {
                        new
                        {
                            index = 0,
                            delta = new { role = "assistant" },
                            finish_reason = (string?)null
                        }
                    }
                },
                context.RequestAborted).ConfigureAwait(false);

            await foreach (var token in stream.ReadTokensAsync(context.RequestAborted))
            {
                await WriteSseAsync(
                    context.Response,
                    new
                    {
                        id,
                        @object = "chat.completion.chunk",
                        created,
                        model,
                        choices = new[]
                        {
                            new
                            {
                                index = 0,
                                delta = new { content = codec.DecodeToken(token) },
                                finish_reason = (string?)null
                            }
                        }
                    },
                    context.RequestAborted).ConfigureAwait(false);
            }

            var snapshot = await stream.Completion.WaitAsync(context.RequestAborted)
                .ConfigureAwait(false);
            await WriteSseAsync(
                context.Response,
                new
                {
                    id,
                    @object = "chat.completion.chunk",
                    created,
                    model,
                    choices = new[]
                    {
                        new
                        {
                            index = 0,
                            delta = new { },
                            finish_reason = FinishReason(snapshot)
                        }
                    },
                    usage = Usage(promptTokenCount, snapshot.GeneratedTokens.Count)
                },
                context.RequestAborted).ConfigureAwait(false);
            await WriteDoneAsync(context.Response, context.RequestAborted).ConfigureAwait(false);
            completed = true;
        }
        finally
        {
            await CancelIfAbandonedAsync(stream, completed).ConfigureAwait(false);
        }
    }

    private static bool TryValidateModelAndLimit(
        string model,
        int? requestedLimit,
        out int maxTokens,
        out string error)
    {
        if (string.IsNullOrWhiteSpace(model))
        {
            maxTokens = 0;
            error = "model is required.";
            return false;
        }

        maxTokens = requestedLimit ?? 128;
        if (maxTokens <= 0)
        {
            error = "max_tokens/max_completion_tokens must be positive.";
            return false;
        }

        error = string.Empty;
        return true;
    }

    private static async ValueTask CancelIfAbandonedAsync(
        InferenceStream stream,
        bool completed)
    {
        if (completed || stream.Completion.IsCompleted)
        {
            return;
        }

        try
        {
            await stream.CancelAsync(CancellationToken.None).ConfigureAwait(false);
        }
        catch
        {
            // Preserve the original HTTP/transport exception. Cancellation is a
            // best-effort cleanup path and the worker owns final failure fan-out.
        }
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

    private static object Usage(int promptTokens, int completionTokens) => new
    {
        prompt_tokens = promptTokens,
        completion_tokens = completionTokens,
        total_tokens = checked(promptTokens + completionTokens)
    };

    private static string CompletionId(
        string prefix,
        InferenceRequestSnapshot snapshot) =>
        $"{prefix}-{snapshot.SequenceId.Value:N}";

    private static void PrepareSse(HttpResponse response)
    {
        response.StatusCode = StatusCodes.Status200OK;
        response.ContentType = "text/event-stream";
        response.Headers.CacheControl = "no-cache";
        response.Headers.Connection = "keep-alive";
    }

    private static async ValueTask WriteSseAsync(
        HttpResponse response,
        object payload,
        CancellationToken cancellationToken)
    {
        var json = JsonSerializer.Serialize(payload, JsonOptions);
        await response.WriteAsync($"data: {json}\n\n", cancellationToken)
            .ConfigureAwait(false);
        await response.Body.FlushAsync(cancellationToken).ConfigureAwait(false);
    }

    private static async ValueTask WriteDoneAsync(
        HttpResponse response,
        CancellationToken cancellationToken)
    {
        await response.WriteAsync("data: [DONE]\n\n", cancellationToken)
            .ConfigureAwait(false);
        await response.Body.FlushAsync(cancellationToken).ConfigureAwait(false);
    }

    private static Task WriteBadRequestAsync(HttpContext context, string message) =>
        Results.Json(
            new
            {
                error = new
                {
                    message,
                    type = "invalid_request_error"
                }
            },
            JsonOptions,
            statusCode: StatusCodes.Status400BadRequest).ExecuteAsync(context);
}
