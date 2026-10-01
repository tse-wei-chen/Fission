using System.Runtime.CompilerServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;

internal static class DeviceInferenceReservationSignalSpecs
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
                $"Device inference reservation signal spec failed: {exception}");
            Environment.ExitCode = 1;
        }
    }

    private static async Task RunAsync()
    {
        var deviceA = new DeviceId("cpu:inference-signal-a");
        var deviceB = new DeviceId("cpu:inference-signal-b");
        await using var actorA = await ContinuousBatchExecutor.CreateAsync(
            new SignalProbeBackend(deviceA),
            capacity: 2,
            maxBatchSize: 1);
        await using var actorB = await ContinuousBatchExecutor.CreateAsync(
            new SignalProbeBackend(deviceB),
            capacity: 2,
            maxBatchSize: 1);
        using var runtime = new ExecutionPlanExecutor(
            new ExecutionDeviceRegistry(actorA, actorB));

        Require(
            runtime.TryReserveDeviceInference(
                new[]
                {
                    new RuntimeDeviceInferenceReservationRequest(deviceA, 1),
                    new RuntimeDeviceInferenceReservationRequest(deviceB, 1)
                },
                out var lease),
            "Signal setup must reserve one inference item on each device.");

        var state = runtime.GetDeviceInferenceReservationState();
        var reservationA = state.Reservations.Single(item => item.Device == deviceA);
        var reservationB = state.Reservations.Single(item => item.Device == deviceB);
        var waitA = runtime.WaitForDeviceInferenceReservationReleaseAsync(
            new[]
            {
                new RuntimeDeviceInferenceReservationVersion(
                    deviceA,
                    reservationA.ReleaseVersion)
            });

        lease.Release(deviceB);
        await RequirePendingAsync(
            waitA,
            "A release on an unrelated physical device must not wake a device-scoped inference waiter.");

        lease.Release(deviceA);
        await waitA.WaitAsync(TimeSpan.FromSeconds(5));

        // Capture a version while a reservation is live, release it before attaching
        // the waiter, then verify the version mismatch completes immediately instead
        // of losing the wakeup.
        Require(
            runtime.TryReserveDeviceInference(
                new[] { new RuntimeDeviceInferenceReservationRequest(deviceA, 1) },
                out var missedWakeupLease),
            "Missed-wakeup setup reservation must succeed.");
        var beforeRelease = runtime.GetDeviceInferenceReservationState(new[] { deviceA })
            .Reservations.Single();
        missedWakeupLease.Release(deviceA);

        var lateWait = runtime.WaitForDeviceInferenceReservationReleaseAsync(
            new[]
            {
                new RuntimeDeviceInferenceReservationVersion(
                    deviceA,
                    beforeRelease.ReleaseVersion)
            });
        await lateWait.WaitAsync(TimeSpan.FromSeconds(5));

        // Waiting on multiple devices is satisfied by the first relevant release.
        Require(
            runtime.TryReserveDeviceInference(
                new[]
                {
                    new RuntimeDeviceInferenceReservationRequest(deviceA, 1),
                    new RuntimeDeviceInferenceReservationRequest(deviceB, 1)
                },
                out var multiLease),
            "Multi-device signal setup must reserve both devices.");
        var multiState = runtime.GetDeviceInferenceReservationState();
        var multiWait = runtime.WaitForDeviceInferenceReservationReleaseAsync(
            multiState.Reservations
                .Select(static item => new RuntimeDeviceInferenceReservationVersion(
                    item.Device,
                    item.ReleaseVersion))
                .ToArray());
        multiLease.Release(deviceB);
        await multiWait.WaitAsync(TimeSpan.FromSeconds(5));
        multiLease.Release(deviceA);

        var conflictingRejected = false;
        try
        {
            _ = runtime.WaitForDeviceInferenceReservationReleaseAsync(
                new[]
                {
                    new RuntimeDeviceInferenceReservationVersion(deviceA, 1),
                    new RuntimeDeviceInferenceReservationVersion(deviceA, 2)
                });
        }
        catch (InvalidOperationException)
        {
            conflictingRejected = true;
        }

        Require(
            conflictingRejected,
            "Conflicting observed release versions for one device must be rejected.");
        lease.Dispose();
        missedWakeupLease.Dispose();
        multiLease.Dispose();
    }

    private static async Task RequirePendingAsync(Task task, string message)
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

    private sealed class SignalProbeBackend : IInferenceBackend
    {
        private bool _initialized;

        public SignalProbeBackend(DeviceId device)
        {
            Device = device;
        }

        public string Name => "inference-reservation-signal-probe";
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
                throw new InvalidOperationException("Signal probe backend is not initialized.");
            }
        }
    }
}
