using System.Runtime.CompilerServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;

internal static class DeviceMemoryScopedObservationSpecs
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
                $"Device-memory scoped observation spec failed: {exception}");
            Environment.ExitCode = 1;
        }
    }

    private static async Task RunAsync()
    {
        var firstId = new DeviceId("cpu:scoped-pressure-0");
        var secondId = new DeviceId("cpu:scoped-pressure-1");
        var firstBackend = new CountingPressureBackend(firstId, reservedBytes: 96);
        var secondBackend = new CountingPressureBackend(secondId, reservedBytes: 160);

        await using var firstDevice = await ContinuousBatchExecutor.CreateAsync(
            firstBackend,
            capacity: 4,
            maxBatchSize: 2);
        await using var secondDevice = await ContinuousBatchExecutor.CreateAsync(
            secondBackend,
            capacity: 4,
            maxBatchSize: 2);
        using var runtime = new ExecutionPlanExecutor(
            new ExecutionDeviceRegistry(firstDevice, secondDevice));

        var scoped = runtime.GetDeviceMemoryPressureCore([secondId, secondId]);
        Require(
            scoped.Count == 1 &&
            scoped[0].Device == secondId &&
            scoped[0].ReservedBytes == 160,
            "A scoped pressure query must return only the requested physical device and normalize duplicates.");
        Require(
            firstBackend.PressureQueries == 0 && secondBackend.PressureQueries == 1,
            "A scoped pressure query must not invoke unrelated device actors.");

        var all = runtime.GetDeviceMemoryPressure();
        Require(
            all.Count == 2 &&
            all[0].Device == firstId &&
            all[1].Device == secondId,
            "The public all-device pressure API must preserve deterministic physical-device ordering.");
        Require(
            firstBackend.PressureQueries == 1 && secondBackend.PressureQueries == 2,
            "The public all-device pressure API must continue querying every registered actor.");

        var unknownFailed = false;
        try
        {
            _ = runtime.GetDeviceMemoryPressureCore(
                [new DeviceId("cpu:scoped-pressure-missing")]);
        }
        catch (KeyNotFoundException)
        {
            unknownFailed = true;
        }

        Require(
            unknownFailed,
            "Scoped physical pressure observation must reject unregistered device ids instead of silently falling back to the default actor.");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class CountingPressureBackend :
        IInferenceBackend,
        IInferenceDeviceMemoryPressureSource
    {
        private readonly long _reservedBytes;
        private bool _initialized;
        private int _pressureQueries;

        public CountingPressureBackend(DeviceId device, long reservedBytes)
        {
            Device = device;
            _reservedBytes = reservedBytes;
        }

        public string Name => "counting-pressure-backend";
        public DeviceId Device { get; }
        public int PressureQueries => Volatile.Read(ref _pressureQueries);

        public bool TryGetDeviceMemoryPressure(
            out InferenceDeviceMemoryPressure pressure)
        {
            EnsureInitialized();
            Interlocked.Increment(ref _pressureQueries);
            pressure = new InferenceDeviceMemoryPressure(
                ActiveBytes: _reservedBytes,
                ReclaimableBytes: 0,
                ReservedBytes: _reservedBytes,
                PeakReservedBytes: _reservedBytes);
            return true;
        }

        public ValueTask InitializeAsync(CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _initialized = true;
            return ValueTask.CompletedTask;
        }

        public ValueTask<IReadOnlyList<BackendStepResult>> PrefillAsync(
            PrefillBatch batch,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException("Inference is not used by the scoped pressure spec.");

        public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
            DecodeBatch batch,
            CancellationToken cancellationToken = default) =>
            throw new NotSupportedException("Inference is not used by the scoped pressure spec.");

        public ValueTask DisposeAsync()
        {
            _initialized = false;
            return ValueTask.CompletedTask;
        }

        private void EnsureInitialized()
        {
            if (!_initialized)
            {
                throw new InvalidOperationException(
                    "Counting pressure backend is not initialized.");
            }
        }
    }
}
