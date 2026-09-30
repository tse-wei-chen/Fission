using System.Runtime.CompilerServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;

internal static class DeviceMemoryAdmissionGateSpecs
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
            Console.Error.WriteLine(
                $"Device-memory admission gate spec failed: {exception}");
            Environment.ExitCode = 1;
        }
    }

    private static async Task RunAsync()
    {
        await using var device = await ContinuousBatchExecutor.CreateAsync(
            new GateProbeBackend(new DeviceId("cpu:admission-gate-probe")),
            capacity: 4,
            maxBatchSize: 2);
        using var runtime = new ExecutionPlanExecutor(device);

        var gpu0 = new DeviceId("cuda:gate-0");
        var gpu1 = new DeviceId("cuda:gate-1");

        var gpu0Lease = await runtime.EnterDeviceMemoryAdmissionAsync([gpu0]);
        try
        {
            // A disjoint physical device must no longer queue behind the held GPU0
            // admission critical section.
            using var gpu1Lease = await runtime
                .EnterDeviceMemoryAdmissionAsync([gpu1])
                .AsTask()
                .WaitAsync(TimeSpan.FromSeconds(5));

            var sameDeviceWait = runtime
                .EnterDeviceMemoryAdmissionAsync([gpu0])
                .AsTask();
            await RequirePendingAsync(
                sameDeviceWait,
                "Admission on the same physical device must remain serialized.");

            gpu0Lease.Dispose();
            using var sameDeviceLease = await sameDeviceWait
                .WaitAsync(TimeSpan.FromSeconds(5));
        }
        finally
        {
            gpu0Lease.Dispose();
        }

        // Duplicate input devices must normalize to one gate instead of trying to
        // acquire the same semaphore twice.
        using (await runtime
                   .EnterDeviceMemoryAdmissionAsync([gpu0, gpu0])
                   .AsTask()
                   .WaitAsync(TimeSpan.FromSeconds(5)))
        {
        }

        // Force cancellation after a multi-device acquire has obtained GPU0 but is
        // blocked on GPU1. The failed acquire must release GPU0 before propagating
        // cancellation, otherwise future admission deadlocks that device forever.
        var gpu1Blocker = await runtime.EnterDeviceMemoryAdmissionAsync([gpu1]);
        try
        {
            using var cancellation = new CancellationTokenSource(
                TimeSpan.FromMilliseconds(150));
            var cancelledAcquire = runtime
                .EnterDeviceMemoryAdmissionAsync([gpu1, gpu0], cancellation.Token)
                .AsTask();

            var cancelled = false;
            try
            {
                using var unexpected = await cancelledAcquire
                    .WaitAsync(TimeSpan.FromSeconds(5));
            }
            catch (OperationCanceledException)
            {
                cancelled = true;
            }

            Require(
                cancelled,
                "A multi-device admission blocked on its second gate must honor cancellation.");

            using var recoveredGpu0 = await runtime
                .EnterDeviceMemoryAdmissionAsync([gpu0])
                .AsTask()
                .WaitAsync(TimeSpan.FromSeconds(5));
        }
        finally
        {
            gpu1Blocker.Dispose();
        }

        // Reverse caller order must canonicalize to the same stable lock order.
        // Holding both gates makes the second acquire wait; once released, it must
        // complete without a lock-order deadlock.
        var reverseOrderLease = await runtime
            .EnterDeviceMemoryAdmissionAsync([gpu1, gpu0])
            .AsTask()
            .WaitAsync(TimeSpan.FromSeconds(5));
        var canonicalOrderWait = runtime
            .EnterDeviceMemoryAdmissionAsync([gpu0, gpu1])
            .AsTask();
        try
        {
            await RequirePendingAsync(
                canonicalOrderWait,
                "Overlapping multi-device admissions must serialize their shared gates.");
        }
        finally
        {
            reverseOrderLease.Dispose();
        }

        using var canonicalOrderLease = await canonicalOrderWait
            .WaitAsync(TimeSpan.FromSeconds(5));

        await VerifyReservationReleaseSignalsAsync(runtime, gpu0, gpu1);
    }

    private static async Task VerifyReservationReleaseSignalsAsync(
        ExecutionPlanExecutor runtime,
        DeviceId gpu0,
        DeviceId gpu1)
    {
        IDisposable? gpu0Reservation = runtime.ReserveDeviceMemory(
            [new RuntimeDeviceMemoryReservationRequest(gpu0, 64)]);
        IDisposable? gpu1Reservation = runtime.ReserveDeviceMemory(
            [new RuntimeDeviceMemoryReservationRequest(gpu1, 64)]);

        try
        {
            var state = runtime.GetDeviceMemoryReservationState();
            var gpu0Snapshot = state.Reservations.Single(
                reservation => reservation.Device == gpu0);
            var gpu1Snapshot = state.Reservations.Single(
                reservation => reservation.Device == gpu1);

            var gpu0Wait = runtime.WaitForDeviceMemoryReservationReleaseAsync(
                [new RuntimeDeviceMemoryReservationVersion(
                    gpu0,
                    gpu0Snapshot.ReleaseVersion)]);

            gpu1Reservation.Dispose();
            gpu1Reservation = null;
            await RequirePendingAsync(
                gpu0Wait,
                "Releasing GPU1 reservation must not wake a waiter scoped only to GPU0.");

            gpu0Reservation.Dispose();
            gpu0Reservation = null;
            await gpu0Wait.WaitAsync(TimeSpan.FromSeconds(5));

            Require(
                gpu1Snapshot.ReleaseVersion == 0,
                "Independent reservation versions should begin at zero for each device.");
        }
        finally
        {
            gpu1Reservation?.Dispose();
            gpu0Reservation?.Dispose();
        }

        // If release wins the race before the waiter attaches, the version mismatch
        // must complete the wait immediately rather than miss the wakeup.
        var preReleased = runtime.ReserveDeviceMemory(
            [new RuntimeDeviceMemoryReservationRequest(gpu0, 32)]);
        var preReleaseState = runtime.GetDeviceMemoryReservationState();
        var preReleaseSnapshot = preReleaseState.Reservations.Single(
            reservation => reservation.Device == gpu0);
        preReleased.Dispose();
        await runtime.WaitForDeviceMemoryReservationReleaseAsync(
                [new RuntimeDeviceMemoryReservationVersion(
                    gpu0,
                    preReleaseSnapshot.ReleaseVersion)])
            .WaitAsync(TimeSpan.FromSeconds(5));

        // A cycle blocked by multiple physical devices should retry when any one of
        // those relevant reservations changes, while still ignoring other devices.
        IDisposable? multiGpu0 = runtime.ReserveDeviceMemory(
            [new RuntimeDeviceMemoryReservationRequest(gpu0, 16)]);
        IDisposable? multiGpu1 = runtime.ReserveDeviceMemory(
            [new RuntimeDeviceMemoryReservationRequest(gpu1, 16)]);
        try
        {
            var state = runtime.GetDeviceMemoryReservationState();
            var observed = state.Reservations
                .Where(reservation => reservation.Device == gpu0 || reservation.Device == gpu1)
                .Select(static reservation => new RuntimeDeviceMemoryReservationVersion(
                    reservation.Device,
                    reservation.ReleaseVersion))
                .ToArray();
            var anyRelevantWait = runtime.WaitForDeviceMemoryReservationReleaseAsync(observed);

            multiGpu1.Dispose();
            multiGpu1 = null;
            await anyRelevantWait.WaitAsync(TimeSpan.FromSeconds(5));
        }
        finally
        {
            multiGpu1?.Dispose();
            multiGpu0?.Dispose();
        }
    }

    private static async Task RequirePendingAsync(
        Task task,
        string message)
    {
        var completed = await Task.WhenAny(
            task,
            Task.Delay(TimeSpan.FromMilliseconds(100)));
        Require(!ReferenceEquals(completed, task), message);
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class GateProbeBackend : IInferenceBackend
    {
        private bool _initialized;

        public GateProbeBackend(DeviceId device)
        {
            Device = device;
        }

        public string Name => "admission-gate-probe";
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
                batch.Items
                    .Select(static item => new BackendStepResult(item.SequenceId, 1))
                    .ToArray());
        }

        public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
            DecodeBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
                batch.Items
                    .Select(static item => new BackendStepResult(item.SequenceId, 2))
                    .ToArray());
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
                throw new InvalidOperationException("Gate probe backend is not initialized.");
            }
        }
    }
}
