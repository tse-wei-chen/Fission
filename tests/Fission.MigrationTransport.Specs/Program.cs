using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

static SequenceMigrationTransportCapability Capability(
    string id,
    SequenceMigrationTransportKind kind,
    long maxBytes,
    long bandwidth,
    int latencyMicros,
    int preference = 0) =>
    new(
        id,
        kind,
        maxBytes,
        bandwidth,
        TimeSpan.FromTicks(latencyMicros * 10L),
        preference);

const long GiB = 1024L * 1024 * 1024;
var planner = new SequenceMigrationTransportPlanner();

var source = new[]
{
    Capability(
        "cuda-p2p",
        SequenceMigrationTransportKind.DirectDevice,
        2 * GiB,
        50_000_000_000,
        5,
        preference: 10),
    Capability(
        "pinned-host",
        SequenceMigrationTransportKind.HostStaging,
        0,
        12_000_000_000,
        30,
        preference: 5),
    Capability(
        "nixl-rdma",
        SequenceMigrationTransportKind.RemoteMemory,
        0,
        25_000_000_000,
        20,
        preference: 8)
};

var target = new[]
{
    Capability(
        "cuda-p2p",
        SequenceMigrationTransportKind.DirectDevice,
        4 * GiB,
        40_000_000_000,
        8,
        preference: 10),
    Capability(
        "pinned-host",
        SequenceMigrationTransportKind.HostStaging,
        0,
        10_000_000_000,
        40,
        preference: 5)
};

var direct = planner.Plan(GiB, source, target);
Require(direct.TransportId == "cuda-p2p", "Planner must select the fastest mutually supported direct path.");
Require(direct.Kind == SequenceMigrationTransportKind.DirectDevice, "Direct path kind is incorrect.");
Require(direct.EffectiveMaxTransferBytes == 2 * GiB, "Planner must intersect peer transfer-size limits.");
Require(direct.EffectiveBandwidthBytesPerSecond == 40_000_000_000, "Planner must use the slower peer bandwidth estimate.");
Require(direct.EffectiveFixedLatency == TimeSpan.FromTicks(80), "Planner must use the larger peer fixed latency.");
Require(direct.EstimatedDuration > direct.EffectiveFixedLatency, "Estimated duration must include payload transfer time.");

var staged = planner.Plan(3 * GiB, source, target);
Require(staged.TransportId == "pinned-host", "Oversized direct transfers must fall back to a compatible staging path.");
Require(staged.EffectiveMaxTransferBytes == long.MaxValue, "Zero capability limit must mean no backend-imposed transfer-size cap.");

var noPath = false;
try
{
    _ = planner.Plan(
        GiB,
        new[]
        {
            Capability("source-only", SequenceMigrationTransportKind.DirectDevice, 0, 10_000_000_000, 1)
        },
        new[]
        {
            Capability("target-only", SequenceMigrationTransportKind.DirectDevice, 0, 10_000_000_000, 1)
        });
}
catch (InvalidOperationException exception)
    when (exception.Message.Contains("No mutually supported", StringComparison.Ordinal))
{
    noPath = true;
}
Require(noPath, "Planner must reject peers with no compatible transport id.");

var kindMismatch = false;
try
{
    _ = planner.Plan(
        1024,
        new[]
        {
            Capability("shared", SequenceMigrationTransportKind.SharedMemory, 0, 1_000_000, 1)
        },
        new[]
        {
            Capability("shared", SequenceMigrationTransportKind.RemoteMemory, 0, 1_000_000, 1)
        });
}
catch (InvalidOperationException)
{
    kindMismatch = true;
}
Require(kindMismatch, "Matching transport ids with different kinds must not be treated as compatible.");

var preferred = planner.Plan(
    1_000,
    new[]
    {
        Capability("a", SequenceMigrationTransportKind.HostStaging, 0, 1_000_000, 10, preference: 1),
        Capability("b", SequenceMigrationTransportKind.HostStaging, 0, 1_000_000, 10, preference: 9)
    },
    new[]
    {
        Capability("a", SequenceMigrationTransportKind.HostStaging, 0, 1_000_000, 10, preference: 1),
        Capability("b", SequenceMigrationTransportKind.HostStaging, 0, 1_000_000, 10, preference: 9)
    });
Require(preferred.TransportId == "b", "Preference must deterministically break equal-duration transport ties.");

var duplicateRejected = false;
try
{
    _ = planner.Plan(
        1_000,
        new[]
        {
            Capability("dup", SequenceMigrationTransportKind.HostStaging, 0, 1_000_000, 1),
            Capability("dup", SequenceMigrationTransportKind.HostStaging, 0, 1_000_000, 1)
        },
        target);
}
catch (InvalidOperationException exception)
    when (exception.Message.Contains("more than once", StringComparison.Ordinal))
{
    duplicateRejected = true;
}
Require(duplicateRejected, "Duplicate capability ids on one backend must be rejected.");

using (var admission = new SequenceMigrationAdmissionController(
    maxInflightBytes: 100,
    maxConcurrentTransfers: 2))
{
    var first = await admission.AcquireAsync(70);
    Require(admission.InflightBytes == 70, "First migration lease must reserve estimated bytes.");
    Require(admission.ActiveTransfers == 1, "First migration lease must reserve one active transfer.");

    var blocked = admission.AcquireAsync(40).AsTask();
    await Task.Yield();
    Require(!blocked.IsCompleted, "Migration must wait when its bytes would exceed the in-flight budget.");

    first.Dispose();
    using var second = await blocked;
    Require(admission.InflightBytes == 40, "Waiting migration must acquire bytes after prior lease release.");
    Require(admission.ActiveTransfers == 1, "Waiting migration must become active after admission.");
}

using (var admission = new SequenceMigrationAdmissionController(
    maxInflightBytes: 1_000,
    maxConcurrentTransfers: 1))
{
    var first = await admission.AcquireAsync(100);
    var blockedBySlot = admission.AcquireAsync(100).AsTask();
    await Task.Yield();
    Require(!blockedBySlot.IsCompleted, "Concurrent-transfer limit must block additional migrations.");
    first.Dispose();
    using var second = await blockedBySlot;
    Require(admission.ActiveTransfers == 1, "Second migration must enter after the transfer slot is released.");
}

using (var admission = new SequenceMigrationAdmissionController(
    maxInflightBytes: 100,
    maxConcurrentTransfers: 2))
{
    var first = await admission.AcquireAsync(100);
    using var cancellation = new CancellationTokenSource();
    var blocked = admission.AcquireAsync(50, cancellation.Token).AsTask();
    await Task.Yield();
    cancellation.Cancel();

    var cancelled = false;
    try
    {
        _ = await blocked;
    }
    catch (OperationCanceledException)
    {
        cancelled = true;
    }

    Require(cancelled, "Blocked migration admission must observe caller cancellation.");
    first.Dispose();
    using var afterCancellation = await admission.AcquireAsync(100);
    Require(admission.ActiveTransfers == 1, "Cancelled waiter must release its transfer slot.");
}

using (var admission = new SequenceMigrationAdmissionController(
    maxInflightBytes: 100,
    maxConcurrentTransfers: 1))
{
    var rejected = false;
    try
    {
        _ = await admission.AcquireAsync(101);
    }
    catch (InvalidOperationException exception)
        when (exception.Message.Contains("at most 100", StringComparison.Ordinal))
    {
        rejected = true;
    }

    Require(rejected, "Migration larger than the total in-flight budget must fail immediately.");
}

Console.WriteLine("Fission migration transport planner/admission specs passed.");
