using Fission.Abstractions;
using Fission.Abstractions.Scheduling;
using Fission.Engine;
using Fission.Runtime.Backends;
using Fission.Runtime.Execution;
using Fission.Runtime.Kv;
using Fission.Scheduler;
using Fission.Server;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

var device = ServerBackendFactory.ResolveDevice(builder.Configuration);
var kvPages = ReadPositiveInt(builder.Configuration["Fission:KvPages"], 16_384);
var tokensPerPage = ReadPositiveInt(builder.Configuration["Fission:TokensPerKvPage"], 16);
var maxBatchTokens = ReadPositiveInt(builder.Configuration["Fission:MaxBatchTokens"], 2_048);
var maxBatchSequences = ReadPositiveInt(builder.Configuration["Fission:MaxBatchSequences"], 128);
var maxPrefillChunk = ReadPositiveInt(builder.Configuration["Fission:MaxPrefillChunkTokens"], 512);
var admissionCapacity = ReadPositiveInt(builder.Configuration["Fission:AdmissionCapacity"], 1_024);

using var textCodec = ServerTextTokenCodecFactory.Create(builder.Configuration);

var backend = ServerBackendFactory.Create(builder.Configuration, device);
await using var deviceExecutor = await ContinuousBatchExecutor.CreateAsync(
    backend,
    capacity: admissionCapacity,
    maxBatchSize: maxBatchSequences);
using var runtime = new ExecutionPlanExecutor(
    deviceExecutor,
    kvPagePool: new KvPagePool(kvPages, tokensPerPage));
using var engine = new InferenceEngine(
    runtime,
    new SchedulingKernel(),
    new InferenceEngineOptions(
        maxBatchTokens,
        maxBatchSequences,
        new SchedulingPolicyOptions(
            DecodeTokenReserve: Math.Min(maxBatchSequences, maxBatchTokens),
            MaxPrefillChunkTokens: maxPrefillChunk,
            DeadlineUrgencyWindow: TimeSpan.FromMilliseconds(50))));
await using var worker = new InferenceWorker(
    engine,
    new InferenceWorkerOptions(admissionCapacity));

var startupProbe = await ServerStartupProbe.RunAsync(
    builder.Configuration,
    worker,
    textCodec);
if (startupProbe is not null)
{
    app.Logger.LogInformation(
        "Startup inference probe succeeded for model {ModelId}: promptTokens={PromptTokens}, generatedTokens={GeneratedTokens}, finishReason={FinishReason}, elapsedMs={ElapsedMs:F1}.",
        startupProbe.ModelId.Value,
        startupProbe.PromptTokenCount,
        startupProbe.GeneratedTokenCount,
        startupProbe.FinishReason,
        startupProbe.Elapsed.TotalMilliseconds);

    if (startupProbe.ExitAfterSuccess)
    {
        app.Logger.LogInformation(
            "Startup inference probe one-shot mode completed; exiting before HTTP serving starts.");
        return;
    }
}

OpenAiEndpoints.Map(app, worker, textCodec);
ServerControlEndpoints.Map(app, builder.Configuration, worker, textCodec);
await app.RunAsync();

static int ReadPositiveInt(string? value, int fallback)
{
    if (int.TryParse(value, out var parsed) && parsed > 0)
    {
        return parsed;
    }

    return fallback;
}
