using System.Runtime.CompilerServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;

internal static class DeviceExecutionCapacitySpecs
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
            Console.Error.WriteLine($"Device execution capacity spec failed: {exception}");
            Environment.ExitCode = 1;
        }
    }

    private static async Task RunAsync()
    {
        var highCreditBackend = new CapacityFeedbackBackend(
            new DeviceId("cpu:capacity-feedback-z"));
        var wideBatchBackend = new CapacityFeedbackBackend(
            new DeviceId("cpu:capacity-feedback-a"));

        await using var highCredit = await ContinuousBatchExecutor.CreateAsync(
            highCreditBackend,
            capacity: 4,
            maxBatchSize: 1);
        await using var wideBatch = await ContinuousBatchExecutor.CreateAsync(
            wideBatchBackend,
            capacity: 2,
            maxBatchSize: 8);
        using var runtime = new ExecutionPlanExecutor(
            new ExecutionDeviceRegistry(highCredit, wideBatch));

        var all = runtime.GetDeviceExecutionCapacities();
        Require(all.Count == 2, "All-device execution capacity query must return both physical actors.");
        Require(
            all[0] == new RuntimeDeviceExecutionCapacity(wideBatchBackend.Device, 2, 8) &&
            all[1] == new RuntimeDeviceExecutionCapacity(highCreditBackend.Device, 4, 1),
            "Execution capacity query must preserve independent credit and backend-batch dimensions and sort by device id.");

        var scoped = runtime.GetDeviceExecutionCapacities(
            new[] { highCreditBackend.Device, highCreditBackend.Device });
        Require(
            scoped.Count == 1 &&
            scoped[0] == new RuntimeDeviceExecutionCapacity(highCreditBackend.Device, 4, 1),
            "Scoped execution capacity query must normalize duplicate physical device ids.");

        var single = runtime.GetDeviceExecutionCapacity(wideBatchBackend.Device);
        Require(
            single == new RuntimeDeviceExecutionCapacity(wideBatchBackend.Device, 2, 8),
            "Single-device execution capacity query must return that actor's exact dimensions.");

        Require(
            runtime.GetExecutionCapacity().MaxInferenceItems == 2,
            "Legacy global execution capacity must remain the minimum inference-credit capacity across registered actors.");

        var unknownRejected = false;
        try
        {
            _ = runtime.GetDeviceExecutionCapacity(new DeviceId("cpu:capacity-feedback-missing"));
        }
        catch (KeyNotFoundException)
        {
            unknownRejected = true;
        }

        Require(
            unknownRejected,
            "Execution capacity feedback must reject unregistered physical devices instead of inventing capacity.");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class CapacityFeedbackBackend : IInferenceBackend
    {
        private bool _initialized;

        public CapacityFeedbackBackend(DeviceId device)
        {
            Device = device;
        }

        public string Name => "capacity-feedback";
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
                throw new InvalidOperationException("Capacity feedback backend is not initialized.");
            }
        }
    }
}
