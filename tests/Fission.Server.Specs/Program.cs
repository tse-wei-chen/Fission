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
OpenAiEndpoints.Map(app, worker, new DeterministicTextTokenCodec());
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
Require(chatData.Count >= 5, "Chat SSE must include role, token chunks, terminal chunk, and [DONE].");
Require(chatData[^1] == "[DONE]", "Chat SSE must terminate with [DONE].");
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
