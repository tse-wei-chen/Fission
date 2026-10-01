using System.Runtime.CompilerServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;

internal static class ReservationWaiterDisposalSpecs
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
            Console.Error.WriteLine($"Reservation waiter disposal spec failed: {exception}");
            Environment.ExitCode = 1;
        }
    }

    private static async Task RunAsync()
    {
        var deviceId = new DeviceId("cpu:reservation-waiter-disposal");
        await using var device = await ContinuousBatchExecutor.CreateAsync(
            new DisposalProbeBackend(deviceId),
            capacity: 1,
            maxBatchSize: 1);
        var runtime = new ExecutionPlanExecutor(device);

        var memoryLease = runtime.ReserveDeviceMemory(
            new[] { new RuntimeDeviceMemoryReservationRequest(deviceId, 64) });
        Require(
            runtime.TryReserveDeviceInference(
                new[] { new RuntimeDeviceInferenceReservationRequest(deviceId, 1) },
                out var inferenceLease),
            "Inference reservation setup must succeed before disposal.");

        var memoryState = runtime.GetDeviceMemoryReservationState(new[] { deviceId });
        var inferenceState = runtime.GetDeviceInferenceReservationState(new[] { deviceId });
        Require(
            memoryState.Reservations.Count == 1 && inferenceState.Reservations.Count == 1,
            "Both reservation ledgers must be live before attaching disposal waiters.");

        var memoryWait = runtime.WaitForDeviceMemoryReservationReleaseAsync(
            new[]
            {
                new RuntimeDeviceMemoryReservationVersion(
                    deviceId,
                    memoryState.Reservations[0].ReleaseVersion)
            });
        var inferenceWait = runtime.WaitForDeviceInferenceReservationReleaseAsync(
            new[]
            {
                new RuntimeDeviceInferenceReservationVersion(
                    deviceId,
                    inferenceState.Reservations[0].ReleaseVersion)
            });

        await RequirePendingAsync(
            memoryWait,
            "Physical-memory reservation waiter must be pending before runtime disposal.");
        await RequirePendingAsync(
            inferenceWait,
            "Inference reservation waiter must be pending before runtime disposal.");

        runtime.Dispose();

        await RequireDisposedAsync(
            memoryWait,
            "Runtime disposal must fault physical-memory reservation waiters.");
        await RequireDisposedAsync(
            inferenceWait,
            "Runtime disposal must fault inference reservation waiters.");

        memoryLease.Dispose();
        inferenceLease.Dispose();
    }

    private static async Task RequirePendingAsync(Task task, string message)
    {
        var completed = await Task.WhenAny(
            task,
            Task.Delay(TimeSpan.FromMilliseconds(100)));
        Require(!ReferenceEquals(completed, task), message);
    }

    private static async Task RequireDisposedAsync(Task task, string message)
    {
        try
        {
            await task.WaitAsync(TimeSpan.FromSeconds(5));
        }
        catch (ObjectDisposedException)
        {
            return;
        }

        throw new InvalidOperationException(message);
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class DisposalProbeBackend : IInferenceBackend
    {
        private bool _initialized;

        public DisposalProbeBackend(DeviceId device)
        {
            Device = device;
        }

        public string Name => "reservation-waiter-disposal-probe";
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
                throw new InvalidOperationException("Disposal probe backend is not initialized.");
            }
        }
    }
}
