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

var device = new DeviceId(builder.Configuration["Fission:Device"] ?? "cpu:0");
var kvPages = ReadPositiveInt(builder.Configuration["Fission:KvPages"], 16_384);
var tokensPerPage = ReadPositiveInt(builder.Configuration["Fission:TokensPerKvPage"], 16);
var maxBatchTokens = ReadPositiveInt(builder.Configuration["Fission:MaxBatchTokens"], 2_048);
var maxBatchSequences = ReadPositiveInt(builder.Configuration["Fission:MaxBatchSequences"], 128);
var maxPrefillChunk = ReadPositiveInt(builder.Configuration["Fission:MaxPrefillChunkTokens"], 512);
var admissionCapacity = ReadPositiveInt(builder.Configuration["Fission:AdmissionCapacity"], 1_024);

await using var deviceExecutor = await ContinuousBatchExecutor.CreateAsync(
    new DeterministicBackend(device),
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

OpenAiEndpoints.Map(app, worker, new DeterministicTextTokenCodec());
await app.RunAsync();

static int ReadPositiveInt(string? value, int fallback)
{
    if (int.TryParse(value, out var parsed) && parsed > 0)
    {
        return parsed;
    }

    return fallback;
}
