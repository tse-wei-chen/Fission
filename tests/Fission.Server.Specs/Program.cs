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

static async Task ValidateBackendCompositionAsync()
{
    var device = new DeviceId("cpu:composition-spec");

    var defaults = new ConfigurationManager();
    Require(
        ServerBackendFactory.ResolveDevice(defaults) == new DeviceId("cpu:0"),
        "Default server device must remain cpu:0.");

    var cudaDevice = new ConfigurationManager();
    cudaDevice["Fission:Backend"] = "onnx";
    cudaDevice["Fission:ExecutionProvider"] = "cuda";
    cudaDevice["Fission:CudaDeviceId"] = "3";
    Require(
        ServerBackendFactory.ResolveDevice(cudaDevice) == new DeviceId("cuda:3"),
        "CUDA composition must derive its default logical device from the CUDA ordinal.");

    cudaDevice["Fission:Device"] = "gpu:primary";
    Require(
        ServerBackendFactory.ResolveDevice(cudaDevice) == new DeviceId("gpu:primary"),
        "An explicit logical device must override provider-derived identity.");

    var invalidProvider = new ConfigurationManager();
    invalidProvider["Fission:Backend"] = "onnx";
    invalidProvider["Fission:ExecutionProvider"] = "metal";
    RequireThrows<InvalidOperationException>(
        () => ServerBackendFactory.ResolveDevice(invalidProvider),
        "Unsupported ONNX execution providers must fail during device resolution.");

    var invalidCudaDevice = new ConfigurationManager();
    invalidCudaDevice["Fission:Backend"] = "onnx";
    invalidCudaDevice["Fission:ExecutionProvider"] = "cuda";
    invalidCudaDevice["Fission:CudaDeviceId"] = "-1";
    RequireThrows<InvalidOperationException>(
        () => ServerBackendFactory.ResolveDevice(invalidCudaDevice),
        "Negative CUDA device ordinals must fail before native composition.");

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
            "CPU ONNX composition must build the Optimum legacy decoder backend.");
        Require(
            backend.Device == device,
            "ONNX backend composition must preserve the configured device.");

        onnx["Fission:ExecutionProvider"] = "cuda";
        onnx["Fission:CudaDeviceId"] = "-1";
        RequireThrows<InvalidOperationException>(
            () => ServerBackendFactory.Create(onnx, new DeviceId("cuda:0")),
            "CUDA composition must validate the device ordinal before loading CUDA Runtime.");

        onnx["Fission:CudaDeviceId"] = "0";
        onnx["Fission:CudaPoolMaxRetainedBytes"] = "-1";
        RequireThrows<InvalidOperationException>(
            () => ServerBackendFactory.Create(onnx, new DeviceId("cuda:0")),
            "CUDA composition must validate pool retention before loading CUDA Runtime.");

        onnx["Fission:CudaPoolMaxRetainedBytes"] = "0";
        onnx["Fission:CudaRuntimeLibraryPath"] = Path.Combine(
            Path.GetTempPath(),
            $"missing-cudart-{Guid.NewGuid():N}");
        RequireThrows<DllNotFoundException>(
            () => ServerBackendFactory.Create(onnx, new DeviceId("cuda:0")),
            "CUDA composition must fail before serving when its configured CUDA Runtime cannot be loaded.");

        onnx["Fission:ExecutionProvider"] = null;
        onnx["Fission:OnnxExecutionProvider"] = "cpu";
        onnx["Fission:CudaRuntimeLibraryPath"] = null;
        onnx["Fission:CudaPoolMaxRetainedBytes"] = null;
        onnx["Fission:VocabularySize"] = "0";
        RequireThrows<InvalidOperationException>(
            () => ServerBackendFactory.Create(onnx, device),
            "Invalid ONNX geometry must fail during server composition.");
    }
    finally
    {
        File.Delete(modelPath);
    }

    var failedBackend = new FailingInitializeBackend(device);
    try
    {
        _ = await ContinuousBatchExecutor.CreateAsync(
            failedBackend,
            capacity: 1,
            maxBatchSize: 1);
        throw new InvalidOperationException(
            "Backend initialization failure spec unexpectedly created an executor.");
    }
    catch (InvalidOperationException exception)
        when (exception.Message == FailingInitializeBackend.FailureMessage)
    {
    }

    Require(
        failedBackend.DisposeCount == 1,
        "ContinuousBatchExecutor.CreateAsync must dispose a backend whose initialization fails.");
}

static async Task ValidateTokenizerCompositionAsync()
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
        "Unknown tokenizer composition must fail during startup.");

    var chatTokenizerPath = Path.Combine(
        Path.GetTempPath(),
        $"fission-chat-tokenizer-{Guid.NewGuid():N}.json");
    var byteTokenizerPath = Path.Combine(
        Path.GetTempPath(),
        $"fission-byte-tokenizer-{Guid.NewGuid():N}.json");

    await File.WriteAllTextAsync(
        chatTokenizerPath,
        """
        {
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [
            {
              "id": 6,
              "content": "<|im_start|>",
              "single_word": false,
              "lstrip": false,
              "rstrip": false,
              "normalized": false,
              "special": true
            },
            {
              "id": 7,
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
          "pre_tokenizer": { "type": "Whitespace" },
          "post_processor": null,
          "decoder": null,
          "model": {
            "type": "WordLevel",
            "vocab": {
              "[UNK]": 0,
              "system": 1,
              "user": 2,
              "assistant": 3,
              "hello": 4,
              "world": 5
            },
            "unk_token": "[UNK]"
          }
        }
        """);

    await File.WriteAllTextAsync(
        byteTokenizerPath,
        """
        {
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [],
          "normalizer": null,
          "pre_tokenizer": null,
          "post_processor": null,
          "decoder": { "type": "ByteFallback" },
          "model": {
            "type": "WordLevel",
            "vocab": {
              "[UNK]": 0,
              "<0xC3>": 1,
              "<0xA9>": 2
            },
            "unk_token": "[UNK]"
          }
        }
        """);

    try
    {
        var configured = new ConfigurationManager();
        configured["Fission:Tokenizer"] = "huggingface";
        configured["Fission:TokenizerPath"] = chatTokenizerPath;
        configured["Fission:ChatTemplate"] = "qwen2";
        configured["Fission:AddPromptSpecialTokens"] = "false";

        using (var codec = ServerTextTokenCodecFactory.Create(configured))
        {
            Require(
                codec.EncodePrompt("hello").SequenceEqual([4]),
                "Hugging Face prompt encoding must use tokenizer.json vocabulary.");

            var chatTokens = codec.EncodeChat(
            [
                new OpenAiChatMessage("user", "hello")
            ]);
            Require(
                chatTokens.SequenceEqual([6, 2, 4, 7, 6, 3]),
                "Qwen2/ChatML rendering must encode model special tokens and the assistant generation prefix.");
        }

        var llama = new ConfigurationManager();
        llama["Fission:Tokenizer"] = "huggingface";
        llama["Fission:TokenizerPath"] = chatTokenizerPath;
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
                    9, 1, 10, 4, 11,
                    9, 2, 10, 5, 11,
                    9, 3, 10
                ]),
                "Llama 3 rendering must encode BOS, header, EOT, and assistant-prefix control tokens.");
        }

        using (var codec = new HuggingFaceTextTokenCodec(byteTokenizerPath))
        using (var decoder = codec.CreateDecoder())
        {
            Require(
                decoder.Append(1) == string.Empty,
                "Byte-fallback streaming decode must buffer an incomplete UTF-8 sequence.");
            Require(
                decoder.Append(2) == "é",
                "Byte-fallback streaming decode must emit the completed UTF-8 fragment.");
            Require(
                decoder.Complete() == string.Empty,
                "A fully emitted byte-fallback stream must not duplicate text at completion.");
        }

        using (var codec = new HuggingFaceTextTokenCodec(byteTokenizerPath))
        using (var decoder = codec.CreateDecoder())
        {
            Require(
                decoder.Append(1) == string.Empty &&
                decoder.Complete() == "�",
                "Decoder completion must flush a truncated byte-fallback sequence consistently with full decode.");
        }

        var noChatTemplate = new ConfigurationManager();
        noChatTemplate["Fission:Tokenizer"] = "huggingface";
        noChatTemplate["Fission:TokenizerPath"] = chatTokenizerPath;

        using (var codec = ServerTextTokenCodecFactory.Create(noChatTemplate))
        {
            RequireThrows<ArgumentException>(
                () => codec.EncodeChat([new OpenAiChatMessage("user", "hello")]),
                "Chat encoding must fail explicitly when no model chat template is configured.");
        }

        var invalidRole = new ConfigurationManager();
        invalidRole["Fission:Tokenizer"] = "huggingface";
        invalidRole["Fission:TokenizerPath"] = chatTokenizerPath;
        invalidRole["Fission:ChatTemplate"] = "qwen2";
        using (var codec = ServerTextTokenCodecFactory.Create(invalidRole))
        {
            RequireThrows<ArgumentException>(
                () => codec.EncodeChat([new OpenAiChatMessage("not-a-role", "hello")]),
                "Production chat templates must reject unknown role names rather than embedding arbitrary control roles.");
        }
    }
    finally
    {
        File.Delete(chatTokenizerPath);
        File.Delete(byteTokenizerPath);
    }
}

await ValidateBackendCompositionAsync();
await ValidateTokenizerCompositionAsync();

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
OpenAiEndpoints.Map(app, worker, new BufferedSpecTextTokenCodec());
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

sealed class FailingInitializeBackend : IInferenceBackend
{
    public const string FailureMessage = "intentional initialize failure";
    private int _disposeCount;

    public FailingInitializeBackend(DeviceId device)
    {
        Device = device;
    }

    public string Name => "failing-initialize-spec";
    public DeviceId Device { get; }
    public int DisposeCount => Volatile.Read(ref _disposeCount);

    public ValueTask InitializeAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        throw new InvalidOperationException(FailureMessage);
    }

    public ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
        PrefillBatch batch,
        CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
        DecodeBatch batch,
        CancellationToken cancellationToken = default) =>
        throw new NotSupportedException();

    public ValueTask DisposeAsync()
    {
        Interlocked.Increment(ref _disposeCount);
        return ValueTask.CompletedTask;
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
