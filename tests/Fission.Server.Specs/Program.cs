using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Abstractions.Scheduling;
using Fission.Engine;
using Fission.Runtime.Execution;
using Fission.Runtime.Kv;
using Fission.Scheduler;
using Fission.Server;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

static async Task WaitUntilAsync(Func<bool> predicate, TimeSpan timeout)
{
    var deadline = DateTimeOffset.UtcNow + timeout;
    while (!predicate())
    {
        if (DateTimeOffset.UtcNow >= deadline)
        {
            throw new TimeoutException("Timed out waiting for serving state convergence.");
        }

        await Task.Delay(20);
    }
}

static TException RequireThrows<TException>(
    Action action,
    string message)
    where TException : Exception
{
    try
    {
        action();
    }
    catch (TException exception)
    {
        return exception;
    }

    throw new InvalidOperationException(message);
}

static async Task ValidateTextCodecCompositionAsync()
{
    var defaults = new ConfigurationManager();
    using (var codec = ServerTextTokenCodecFactory.Create(defaults))
    {
        Require(
            codec is DeterministicTextTokenCodec,
            "Server tokenizer composition must default to the deterministic codec.");
    }

    var unsupported = new ConfigurationManager();
    unsupported["Fission:Tokenizer"] = "not-a-tokenizer";
    RequireThrows<InvalidOperationException>(
        () => ServerTextTokenCodecFactory.Create(unsupported),
        "Unknown server tokenizers must fail during composition.");

    var missingTokenizer = new ConfigurationManager();
    missingTokenizer["Fission:Tokenizer"] = "huggingface";
    missingTokenizer["Fission:TokenizerPath"] = Path.Combine(
        Path.GetTempPath(),
        $"missing-tokenizer-{Guid.NewGuid():N}.json");
    RequireThrows<FileNotFoundException>(
        () => ServerTextTokenCodecFactory.Create(missingTokenizer),
        "The Hugging Face codec must reject a missing tokenizer before server startup.");

    var tokenizerPath = Path.Combine(
        Path.GetTempPath(),
        $"fission-tokenizer-{Guid.NewGuid():N}.json");

    const string tokenizerJson =
        """
        {
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [
            {
              "id": 1,
              "content": "<|im_start|>",
              "single_word": false,
              "lstrip": false,
              "rstrip": false,
              "normalized": false,
              "special": true
            },
            {
              "id": 2,
              "content": "<|im_end|>",
              "single_word": false,
              "lstrip": false,
              "rstrip": false,
              "normalized": false,
              "special": true
            },
            {
              "id": 8,
              "content": "<|begin_of_text|>",
              "single_word": false,
              "lstrip": false,
              "rstrip": false,
              "normalized": false,
              "special": true
            },
            {
              "id": 9,
              "content": "<|start_header_id|>",
              "single_word": false,
              "lstrip": false,
              "rstrip": false,
              "normalized": false,
              "special": true
            },
            {
              "id": 10,
              "content": "<|end_header_id|>",
              "single_word": false,
              "lstrip": false,
              "rstrip": false,
              "normalized": false,
              "special": true
            },
            {
              "id": 11,
              "content": "<|eot_id|>",
              "single_word": false,
              "lstrip": false,
              "rstrip": false,
              "normalized": false,
              "special": true
            }
          ],
          "normalizer": null,
          "pre_tokenizer": {
            "type": "Whitespace"
          },
          "post_processor": null,
          "decoder": null,
          "model": {
            "type": "WordLevel",
            "vocab": {
              "[UNK]": 0,
              "<|im_start|>": 1,
              "<|im_end|>": 2,
              "user": 3,
              "assistant": 4,
              "hello": 5,
              "world": 6,
              "system": 7,
              "<|begin_of_text|>": 8,
              "<|start_header_id|>": 9,
              "<|end_header_id|>": 10,
              "<|eot_id|>": 11,
              "developer": 12,
              "tool": 13
            },
            "unk_token": "[UNK]"
          }
        }
        """;

    await File.WriteAllTextAsync(tokenizerPath, tokenizerJson);

    try
    {
        var qwen = new ConfigurationManager();
        qwen["Fission:Tokenizer"] = "huggingface";
        qwen["Fission:TokenizerPath"] = tokenizerPath;
        qwen["Fission:ChatTemplate"] = "qwen2";
        qwen["Fission:AddPromptSpecialTokens"] = "false";

        using (var codec = ServerTextTokenCodecFactory.Create(qwen))
        {
            Require(
                codec.EncodePrompt("hello world").SequenceEqual([5, 6]),
                "Hugging Face prompt encoding must use tokenizer.json ids.");

            var chatTokens = codec.EncodeChat(
            [
                new OpenAiChatMessage("user", "hello")
            ]);
            Require(
                chatTokens.SequenceEqual([1, 3, 5, 2, 1, 4]),
                "Qwen2 chat rendering must preserve added control tokens as single ids.");

            using var decoder = codec.CreateDecoder();
            Require(
                decoder.Append(5) == "hello",
                "Streaming tokenizer decode must emit a stable first token.");
            Require(
                decoder.Append(6) == " world",
                "Streaming tokenizer decode must preserve cross-token spacing.");
            Require(
                decoder.Complete() == string.Empty,
                "Streaming tokenizer decode must not duplicate already emitted text.");
        }

        var llama = new ConfigurationManager();
        llama["Fission:Tokenizer"] = "huggingface";
        llama["Fission:TokenizerPath"] = tokenizerPath;
        llama["Fission:ChatTemplate"] = "llama3";
        llama["Fission:AddPromptSpecialTokens"] = "false";

        using (var codec = ServerTextTokenCodecFactory.Create(llama))
        {
            var chatTokens = codec.EncodeChat(
            [
                new OpenAiChatMessage("system", "hello"),
                new OpenAiChatMessage("user", "world")
            ]);
            Require(
                chatTokens.SequenceEqual(
                [
                    8,
                    9, 7, 10, 5, 11,
                    9, 3, 10, 6, 11,
                    9, 4, 10
                ]),
                "Llama 3 chat rendering must preserve header and turn control tokens.");
        }

        var noTemplate = new ConfigurationManager();
        noTemplate["Fission:Tokenizer"] = "huggingface";
        noTemplate["Fission:TokenizerPath"] = tokenizerPath;
        using (var codec = ServerTextTokenCodecFactory.Create(noTemplate))
        {
            RequireThrows<ArgumentException>(
                () => codec.EncodeChat([new OpenAiChatMessage("user", "hello")]),
                "Chat requests must fail clearly when no model chat template is configured.");
        }
    }
    finally
    {
        File.Delete(tokenizerPath);
    }
}

static async Task ValidateBackendCompositionAsync()
{
    var device = new DeviceId("cpu:composition-spec");

    var defaults = new ConfigurationManager();
    await using (var backend = ServerBackendFactory.Create(defaults, device))
    {
        Require(
            backend.Name == "deterministic",
            "Server backend composition must default to the deterministic backend.");
        Require(
            backend.Device == device,
            "Deterministic backend composition must preserve the configured device.");
    }

    var unsupported = new ConfigurationManager();
    unsupported["Fission:Backend"] = "not-a-backend";
    RequireThrows<InvalidOperationException>(
        () => ServerBackendFactory.Create(unsupported, device),
        "Unknown server backends must fail during composition.");

    var missingModel = new ConfigurationManager();
    missingModel["Fission:Backend"] = "onnx";
    missingModel["Fission:ModelPath"] = Path.Combine(
        Path.GetTempPath(),
        $"missing-fission-{Guid.NewGuid():N}.onnx");
    var missingException = RequireThrows<FileNotFoundException>(
        () => ServerBackendFactory.Create(missingModel, device),
        "The ONNX backend must reject a missing model before server startup.");
    Require(
        missingException.FileName is not null,
        "Missing-model failure should include the resolved model path.");

    var modelPath = Path.Combine(
        Path.GetTempPath(),
        $"fission-composition-{Guid.NewGuid():N}.onnx");
    await File.WriteAllBytesAsync(modelPath, [0]);

    try
    {
        var onnx = new ConfigurationManager();
        onnx["Fission:Backend"] = "onnx";
        onnx["Fission:ModelPath"] = modelPath;
        onnx["Fission:ModelId"] = "composition-spec-model";
        onnx["Fission:NumHiddenLayers"] = "2";
        onnx["Fission:NumKvHeads"] = "2";
        onnx["Fission:HeadDim"] = "4";
        onnx["Fission:VocabularySize"] = "32";
        onnx["Fission:EosTokenIds"] = "2, 3";

        await using var backend = ServerBackendFactory.Create(onnx, device);
        Require(
            backend.Name == "onnxruntime/decoder/optimum-legacy-fp32-greedy",
            "ONNX composition must build the Optimum legacy decoder backend.");
        Require(
            backend.Device == device,
            "ONNX backend composition must preserve the configured device.");

        onnx["Fission:VocabularySize"] = "0";
        RequireThrows<InvalidOperationException>(
            () => ServerBackendFactory.Create(onnx, device),
            "Invalid ONNX geometry must fail during server composition.");
    }
    finally
    {
        File.Delete(modelPath);
    }
}

await ValidateBackendCompositionAsync();
await ValidateTextCodecCompositionAsync();

using var deterministicDecoder = new DeterministicTextTokenCodec().CreateDecoder();
Require(
    deterministicDecoder.Append(7) == "<7>" &&
    deterministicDecoder.Complete() == string.Empty,
    "Deterministic text decoding must preserve the existing token rendering contract.");

var device = new DeviceId("cpu:server-spec");
var kvPool = new KvPagePool(capacity: 1024, tokensPerPage: 4);
await using var deviceExecutor = await ContinuousBatchExecutor.CreateAsync(
    new SlowBackend(device, TimeSpan.FromMilliseconds(15)),
    capacity: 128,
    maxBatchSize: 8);
using var runtime = new ExecutionPlanExecutor(deviceExecutor, kvPagePool: kvPool);
using var engine = new InferenceEngine(
    runtime,
    new SchedulingKernel(),
    new InferenceEngineOptions(
        MaxBatchTokens: 16,
        MaxBatchSequences: 4,
        Scheduling: new SchedulingPolicyOptions(
            DecodeTokenReserve: 2,
            MaxPrefillChunkTokens: 8,
            DeadlineUrgencyWindow: TimeSpan.FromMilliseconds(50))));
await using var worker = new InferenceWorker(
    engine,
    new InferenceWorkerOptions(AdmissionCapacity: 16));

var builder = WebApplication.CreateBuilder();
builder.WebHost.UseUrls("http://127.0.0.1:0");
var app = builder.Build();
using var servingCodec = new BufferedSpecTextTokenCodec();
OpenAiEndpoints.Map(app, worker, servingCodec);
await app.StartAsync();

var server = app.Services.GetRequiredService<IServer>();
var addresses = server.Features.Get<IServerAddressesFeature>()?.Addresses;
Require(addresses is { Count: > 0 }, "Kestrel must expose its bound ephemeral address.");
var address = addresses!.Single();
using var client = new HttpClient { BaseAddress = new Uri(address) };

var health = await client.GetAsync("/healthz");
Require(health.StatusCode == HttpStatusCode.OK, "Health endpoint must return HTTP 200.");

var completionResponse = await client.PostAsJsonAsync(
    "/v1/completions",
    new
    {
        model = "server-spec-model",
        prompt = "hello",
        max_tokens = 3,
        stream = false
    });
Require(completionResponse.StatusCode == HttpStatusCode.OK, "Non-stream completion must return HTTP 200.");
using (var completionJson = JsonDocument.Parse(await completionResponse.Content.ReadAsStringAsync()))
{
    var root = completionJson.RootElement;
    Require(root.GetProperty("object").GetString() == "text_completion", "Completion object type must match OpenAI transport shape.");
    Require(root.GetProperty("choices")[0].GetProperty("finish_reason").GetString() == "length", "max_tokens completion must finish with length.");
    Require(root.GetProperty("usage").GetProperty("completion_tokens").GetInt32() == 3, "Completion usage must report generated token count.");
    Require(
        root.GetProperty("choices")[0].GetProperty("text").GetString()?.EndsWith("[done]", StringComparison.Ordinal) == true,
        "Non-stream completion must include text flushed by the request-scoped decoder.");
}
Require(runtime.SequenceCount == 0 && kvPool.AllocatedPages == 0, "Non-stream completion must release sequence and KV ownership.");

using var chatRequest = new HttpRequestMessage(HttpMethod.Post, "/v1/chat/completions")
{
    Content = JsonContent.Create(new
    {
        model = "server-spec-model",
        messages = new[]
        {
            new { role = "user", content = "hello from chat" }
        },
        max_completion_tokens = 2,
        stream = true
    })
};
using var chatResponse = await client.SendAsync(
    chatRequest,
    HttpCompletionOption.ResponseHeadersRead);
Require(chatResponse.StatusCode == HttpStatusCode.OK, "Streaming chat completion must return HTTP 200.");
Require(
    chatResponse.Content.Headers.ContentType?.MediaType == "text/event-stream",
    "Streaming chat completion must use text/event-stream.");

var chatData = new List<string>();
await using (var chatBody = await chatResponse.Content.ReadAsStreamAsync())
using (var reader = new StreamReader(chatBody))
{
    while (await reader.ReadLineAsync() is { } line)
    {
        if (!line.StartsWith("data: ", StringComparison.Ordinal))
        {
            continue;
        }

        var payload = line[6..];
        chatData.Add(payload);
        if (payload == "[DONE]")
        {
            break;
        }
    }
}
Require(chatData.Count >= 5, "Chat SSE must include role, decoded text chunks, terminal chunk, and [DONE].");
Require(chatData[^1] == "[DONE]", "Chat SSE must terminate with [DONE].");

var chatContentChunks = chatData
    .Where(static payload => payload != "[DONE]")
    .Select(static payload => JsonDocument.Parse(payload))
    .Where(static document =>
    {
        var delta = document.RootElement.GetProperty("choices")[0].GetProperty("delta");
        return delta.TryGetProperty("content", out _);
    })
    .ToArray();
try
{
    Require(
        chatContentChunks.Length == 2,
        "Contextual chat decoding must suppress empty intermediate chunks and emit one buffered chunk plus final decoder flush.");
    Require(
        chatContentChunks.All(static document =>
            !string.IsNullOrEmpty(
                document.RootElement.GetProperty("choices")[0]
                    .GetProperty("delta")
                    .GetProperty("content")
                    .GetString())),
        "Streaming transport must not emit empty decoded text chunks.");
}
finally
{
    foreach (var document in chatContentChunks)
    {
        document.Dispose();
    }
}

var terminalChat = chatData[^2];
using (var terminalJson = JsonDocument.Parse(terminalChat))
{
    Require(
        terminalJson.RootElement.GetProperty("choices")[0].GetProperty("finish_reason").GetString() == "length",
        "Streaming chat terminal chunk must report length finish reason.");
}
Require(runtime.SequenceCount == 0 && kvPool.AllocatedPages == 0, "Streaming chat completion must release sequence and KV ownership.");

using var cancelledRequest = new HttpRequestMessage(HttpMethod.Post, "/v1/completions")
{
    Content = JsonContent.Create(new
    {
        model = "server-spec-model",
        prompt = "disconnect me",
        max_tokens = 200,
        stream = true
    })
};
var cancelledResponse = await client.SendAsync(
    cancelledRequest,
    HttpCompletionOption.ResponseHeadersRead);
Require(cancelledResponse.StatusCode == HttpStatusCode.OK, "Disconnect test must establish an SSE response.");
await using (var cancelledBody = await cancelledResponse.Content.ReadAsStreamAsync())
using (var reader = new StreamReader(cancelledBody))
{
    string? firstData = null;
    while (firstData is null && await reader.ReadLineAsync() is { } line)
    {
        if (line.StartsWith("data: {", StringComparison.Ordinal))
        {
            firstData = line;
        }
    }

    Require(firstData is not null, "Disconnect test must observe at least one SSE data chunk before aborting.");
    Require(runtime.SequenceCount > 0, "Disconnect test request must still own runtime state after its first streamed token.");
}
cancelledResponse.Dispose();

await WaitUntilAsync(
    () => runtime.SequenceCount == 0 &&
          kvPool.AllocatedPages == 0 &&
          engine.ActiveRequestCount == 0,
    TimeSpan.FromSeconds(5));
Require(engine.ActiveRequestCount == 0, "Client disconnect must cancel the engine request.");

await app.StopAsync();

Console.WriteLine(
    $"Fission server specs passed: chatEvents={chatData.Count}, sequences={runtime.SequenceCount}, kv={kvPool.AllocatedPages}/{kvPool.Capacity}.");

sealed class BufferedSpecTextTokenCodec : ITextTokenCodec
{
    private readonly DeterministicTextTokenCodec _promptCodec = new();

    public int[] EncodePrompt(string prompt) => _promptCodec.EncodePrompt(prompt);

    public int[] EncodeChat(IReadOnlyList<OpenAiChatMessage> messages) =>
        _promptCodec.EncodeChat(messages);

    public ITextTokenDecoder CreateDecoder() => new Decoder();

    public void Dispose() => _promptCodec.Dispose();

    private sealed class Decoder : ITextTokenDecoder
    {
        private int? _pending;

        public string Append(int tokenId)
        {
            if (_pending is null)
            {
                _pending = tokenId;
                return string.Empty;
            }

            var text = $"[{_pending.Value},{tokenId}]";
            _pending = null;
            return text;
        }

        public string Complete()
        {
            var trailing = _pending is { } tokenId
                ? $"[{tokenId}]"
                : string.Empty;
            _pending = null;
            return trailing + "[done]";
        }

        public void Dispose() { }
    }
}

sealed class SlowBackend : IInferenceBackend
{
    private readonly TimeSpan _delay;
    private bool _initialized;

    public SlowBackend(DeviceId device, TimeSpan delay)
    {
        Device = device;
        _delay = delay;
    }

    public string Name => "slow-spec";
    public DeviceId Device { get; }

    public ValueTask InitializeAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        _initialized = true;
        return ValueTask.CompletedTask;
    }

    public async ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
        PrefillBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        await Task.Delay(_delay, cancellationToken);
        return batch.Items
            .Select(static item => new BackendStepResult(item.SequenceId, item.Tokens.Length))
            .ToArray();
    }

    public async ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default)
    {
        EnsureInitialized();
        await Task.Delay(_delay, cancellationToken);
        return batch.Items
            .Select(static item => new BackendStepResult(item.SequenceId, 1_000 + item.Position))
            .ToArray();
    }

    public ValueTask DisposeAsync()
    {
        _initialized = false;
        return ValueTask.CompletedTask;
    }

    private void EnsureInitialized()
    {
        if (!_initialized)
        {
            throw new InvalidOperationException("Slow backend has not been initialized.");
        }
    }
}
