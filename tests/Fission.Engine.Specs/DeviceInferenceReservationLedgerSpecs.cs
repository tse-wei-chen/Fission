using System.Runtime.CompilerServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;

internal static class DeviceInferenceReservationLedgerSpecs
{
    private static Task? _runTask;

    [ModuleInitializer]
    internal static void Start()
    {
        _runTask = Task.Run(RunAsync);
        AppDomain.CurrentDomain.ProcessExit += static (_, _) => CompleteBeforeExit();
    }

    private static void CompleteBeforeExit()
    {
        var runTask = _runTask;
        if (runTask is null)
        {
            return;
        }

        try
        {
            runTask.GetAwaiter().GetResult();
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"Device inference reservation ledger spec failed: {exception}");
            Environment.ExitCode = 1;
        }
    }

    private static async Task RunAsync()
    {
        var deviceA = new DeviceId("cpu:inference-reservation-a");
        var deviceB = new DeviceId("cpu:inference-reservation-b");
        await using var actorA = await ContinuousBatchExecutor.CreateAsync(
            new ReservationProbeBackend(deviceA),
            capacity: 2,
            maxBatchSize: 8);
        await using var actorB = await ContinuousBatchExecutor.CreateAsync(
            new ReservationProbeBackend(deviceB),
            capacity: 3,
            maxBatchSize: 1);
        using var runtime = new ExecutionPlanExecutor(
            new ExecutionDeviceRegistry(actorA, actorB));

        Require(
            runtime.GetDeviceInferenceReservationState().Reservations.Count == 0,
            "Fresh runtime must not report inference reservations.");

        Require(
            runtime.TryReserveDeviceInference(
                new[] { new RuntimeDeviceInferenceReservationRequest(deviceA, 2) },
                out var fullA),
            "A reservation equal to the device inference-credit capacity must succeed.");

        var fullASnapshot = runtime.GetDeviceInferenceReservationState(new[] { deviceA });
        Require(
            fullASnapshot.Reservations.Count == 1 &&
            fullASnapshot.Reservations[0] ==
                new RuntimeDeviceInferenceReservationSnapshot(deviceA, 2, 0),
            "Reservation snapshot must expose reserved item count and initial release version.");

        Require(
            !runtime.TryReserveDeviceInference(
                new[] { new RuntimeDeviceInferenceReservationRequest(deviceA, 1) },
                out var blockedA),
            "A reservation beyond one device's item-credit capacity must fail without blocking.");
        blockedA.Dispose();

        Require(
            !runtime.TryReserveDeviceInference(
                new[]
                {
                    new RuntimeDeviceInferenceReservationRequest(deviceA, 1),
                    new RuntimeDeviceInferenceReservationRequest(deviceB, 1)
                },
                out var atomicFailure),
            "A multi-device reservation must fail when any participating device lacks capacity.");
        atomicFailure.Dispose();
        Require(
            runtime.GetDeviceInferenceReservationState(new[] { deviceB })
                .Reservations.Count == 0,
            "Failed multi-device reservation must not partially charge an otherwise-available device.");

        Require(
            runtime.TryReserveDeviceInference(
                new[] { new RuntimeDeviceInferenceReservationRequest(deviceB, 3) },
                out var fullB),
            "A disjoint device must reserve its own inference credits independently.");
        var bothFull = runtime.GetDeviceInferenceReservationState();
        Require(
            bothFull.Reservations.Count == 2 &&
            bothFull.Reservations[0].Device == deviceA &&
            bothFull.Reservations[0].Items == 2 &&
            bothFull.Reservations[1].Device == deviceB &&
            bothFull.Reservations[1].Items == 3,
            "All-device reservation snapshot must retain independent per-device accounting in device order.");

        fullA.Dispose();
        Require(
            runtime.GetDeviceInferenceReservationState(new[] { deviceA })
                .Reservations.Count == 0,
            "Releasing the last lease must remove active reservation bytes from the snapshot.");

        Require(
            runtime.TryReserveDeviceInference(
                new[] { new RuntimeDeviceInferenceReservationRequest(deviceA, 1) },
                out var afterReleaseA),
            "A released device must become reservable again.");
        var afterReleaseSnapshot = runtime.GetDeviceInferenceReservationState(new[] { deviceA });
        Require(
            afterReleaseSnapshot.Reservations.Count == 1 &&
            afterReleaseSnapshot.Reservations[0].ReleaseVersion == 1,
            "Release version must advance and persist across an empty-ledger interval.");
        afterReleaseA.Dispose();

        Require(
            runtime.TryReserveDeviceInference(
                new[]
                {
                    new RuntimeDeviceInferenceReservationRequest(deviceA, 1),
                    new RuntimeDeviceInferenceReservationRequest(deviceA, 1)
                },
                out var duplicateA),
            "Duplicate reservation requests for one device must normalize before capacity validation.");
        var duplicateSnapshot = runtime.GetDeviceInferenceReservationState(new[] { deviceA });
        Require(
            duplicateSnapshot.Reservations.Count == 1 &&
            duplicateSnapshot.Reservations[0].Items == 2 &&
            duplicateSnapshot.Reservations[0].ReleaseVersion == 2,
            "Normalized duplicate requests must charge the summed item count exactly once.");
        duplicateA.Dispose();
        fullB.Dispose();

        var unknownRejected = false;
        try
        {
            _ = runtime.TryReserveDeviceInference(
                new[]
                {
                    new RuntimeDeviceInferenceReservationRequest(
                        new DeviceId("cpu:inference-reservation-missing"),
                        1)
                },
                out var unknownLease);
            unknownLease.Dispose();
        }
        catch (KeyNotFoundException)
        {
            unknownRejected = true;
        }

        Require(
            unknownRejected,
            "Inference reservation must reject unregistered physical devices.");

        using var start = new ManualResetEventSlim(false);
        Task<(bool Success, IDisposable Lease)> CompeteAsync() => Task.Run(() =>
        {
            start.Wait();
            var success = runtime.TryReserveDeviceInference(
                new[] { new RuntimeDeviceInferenceReservationRequest(deviceA, 2) },
                out var lease);
            return (success, lease);
        });

        var first = CompeteAsync();
        var second = CompeteAsync();
        start.Set();
        var competitors = await Task.WhenAll(first, second)
            .WaitAsync(TimeSpan.FromSeconds(5));

        Require(
            competitors.Count(static result => result.Success) == 1,
            "Concurrent full-capacity reservations on the same device must admit exactly one caller.");
        foreach (var competitor in competitors)
        {
            competitor.Lease.Dispose();
        }

        Require(
            runtime.GetDeviceInferenceReservationState().Reservations.Count == 0,
            "All reservation leases must return the ledger to an empty state.");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class ReservationProbeBackend : IInferenceBackend
    {
        private bool _initialized;

        public ReservationProbeBackend(DeviceId device)
        {
            Device = device;
        }

        public string Name => "inference-reservation-probe";
        public DeviceId Device { get; }

        public ValueTask InitializeAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _initialized = true;
            return ValueTask.CompletedTask;
        }

        public ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
            PrefillBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
                batch.Items.Select(static item => new BackendStepResult(item.SequenceId, 1)).ToArray());
        }

        public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
            DecodeBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
                batch.Items.Select(static item => new BackendStepResult(item.SequenceId, 2)).ToArray());
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
                throw new InvalidOperationException("Inference reservation probe is not initialized.");
            }
        }
    }
}
