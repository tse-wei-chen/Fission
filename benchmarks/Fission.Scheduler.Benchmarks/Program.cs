using System.Diagnostics;
using Fission.Abstractions;
using Fission.Abstractions.Scheduling;
using Fission.Scheduler;

const int defaultIterations = 20_000;
var iterations = args.Length == 0
    ? defaultIterations
    : int.Parse(args[0], System.Globalization.CultureInfo.InvariantCulture);

if (iterations <= 0)
{
    throw new ArgumentOutOfRangeException(nameof(iterations), "Iteration count must be positive.");
}

var now = new DateTimeOffset(2026, 10, 2, 0, 0, 0, TimeSpan.Zero);
ISchedulingKernel kernel = new SchedulingKernel();

var scenarios = new[]
{
    CreateScenario("single-decode", now, 1, 1),
    CreateScenario("mixed-32-single-device", now, 32, 1),
    CreateScenario("mixed-128-four-device", now, 128, 4)
};

Console.WriteLine($"Fission scheduler microbenchmark ({iterations:N0} measured iterations)");
Console.WriteLine("scenario\tns/op\tbytes/op\tchecksum");

foreach (var scenario in scenarios)
{
    RunWarmup(kernel, scenario, now, Math.Min(iterations, 5_000));
    var result = Measure(kernel, scenario, now, iterations);

    Console.WriteLine(
        $"{scenario.Name}\t{result.NanosecondsPerOperation:F1}\t{result.BytesPerOperation:F1}\t{result.Checksum}");
}

static Scenario CreateScenario(
    string name,
    DateTimeOffset now,
    int candidateCount,
    int deviceCount)
{
    var devices = new DeviceId[deviceCount];
    for (var index = 0; index < devices.Length; index++)
    {
        devices[index] = new DeviceId($"cuda:{index}");
    }

    var candidates = new SchedulingCandidate[candidateCount];
    for (var index = 0; index < candidates.Length; index++)
    {
        var isDecode = index % 4 == 0;
        var deadline = index % 7 == 0
            ? now.AddMilliseconds(20 + index)
            : (DateTimeOffset?)null;
        var device = devices[index % devices.Length];

        candidates[index] = new SchedulingCandidate(
            new SequenceId(CreateDeterministicGuid(index + 1)),
            isDecode ? SchedulingPhase.Decoding : SchedulingPhase.Prefilling,
            deadline,
            now.AddMilliseconds(-index),
            isDecode ? 1 : 128 + (index % 5) * 32,
            64 + index * 3,
            16,
            index % 8,
            2_048,
            device);
    }

    var deviceMemory = new SchedulingDeviceMemoryBudget[deviceCount];
    var deviceSequences = new SchedulingDeviceSequenceBudget[deviceCount];
    for (var index = 0; index < deviceCount; index++)
    {
        deviceMemory[index] = new SchedulingDeviceMemoryBudget(devices[index], 1L << 40);
        deviceSequences[index] = new SchedulingDeviceSequenceBudget(
            devices[index],
            Math.Max(1, 64 / deviceCount));
    }

    var budget = new SchedulingBudget(
        MaxBatchTokens: candidateCount <= 1 ? 8 : 8_192,
        AvailableKvPages: 16_384,
        MaxBatchSequences: Math.Min(candidateCount, 64),
        AvailableKvBytes: 1L << 42,
        DeviceMemory: deviceMemory,
        DeviceSequences: deviceSequences);

    var policy = new SchedulingPolicyOptions(
        DecodeTokenReserve: Math.Min(candidateCount, 16),
        MaxPrefillChunkTokens: 256,
        DeadlineUrgencyWindow: TimeSpan.FromMilliseconds(50));

    return new Scenario(name, candidates, budget, policy);
}

static Guid CreateDeterministicGuid(int value)
{
    Span<byte> bytes = stackalloc byte[16];
    BitConverter.TryWriteBytes(bytes, value);
    BitConverter.TryWriteBytes(bytes[8..], unchecked(value * 0x5f3759df));
    return new Guid(bytes);
}

static void RunWarmup(
    ISchedulingKernel kernel,
    Scenario scenario,
    DateTimeOffset now,
    int iterations)
{
    var checksum = 0;

    for (var iteration = 0; iteration < iterations; iteration++)
    {
        var result = kernel.Schedule(
            Guid.Empty,
            now,
            scenario.Budget,
            scenario.Policy,
            scenario.Candidates);

        checksum ^= Consume(result);
    }

    GC.KeepAlive(checksum);
}

static Measurement Measure(
    ISchedulingKernel kernel,
    Scenario scenario,
    DateTimeOffset now,
    int iterations)
{
    GC.Collect();
    GC.WaitForPendingFinalizers();
    GC.Collect();

    var checksum = 0;
    var allocatedBefore = GC.GetAllocatedBytesForCurrentThread();
    var started = Stopwatch.GetTimestamp();

    for (var iteration = 0; iteration < iterations; iteration++)
    {
        var result = kernel.Schedule(
            Guid.Empty,
            now,
            scenario.Budget,
            scenario.Policy,
            scenario.Candidates);

        checksum = unchecked(checksum + Consume(result));
    }

    var elapsedTicks = Stopwatch.GetTimestamp() - started;
    var allocatedBytes = GC.GetAllocatedBytesForCurrentThread() - allocatedBefore;

    var elapsedSeconds = elapsedTicks / (double)Stopwatch.Frequency;
    var nanosecondsPerOperation = elapsedSeconds * 1_000_000_000d / iterations;
    var bytesPerOperation = allocatedBytes / (double)iterations;

    return new Measurement(nanosecondsPerOperation, bytesPerOperation, checksum);
}

static int Consume(SchedulingKernelResult result)
{
    var checksum = result.Batch.ConsumedTokens;
    checksum = unchecked(checksum * 31 + result.Batch.Items.Count);
    checksum = unchecked(checksum * 31 + result.Deferred.Count);
    checksum = unchecked(checksum * 31 + result.Rejected.Count);
    return checksum;
}

internal sealed record Scenario(
    string Name,
    IReadOnlyList<SchedulingCandidate> Candidates,
    SchedulingBudget Budget,
    SchedulingPolicyOptions Policy);

internal readonly record struct Measurement(
    double NanosecondsPerOperation,
    double BytesPerOperation,
    int Checksum);
