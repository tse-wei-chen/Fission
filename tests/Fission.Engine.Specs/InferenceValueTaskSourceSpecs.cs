using System.Runtime.CompilerServices;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;

internal static class InferenceValueTaskSourceSpecs
{
    private const string DecodeFailureMessage = "decode failure after prefill completion";
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
            Console.Error.WriteLine($"Inference value-task source spec failed: {exception}");
            Environment.ExitCode = 1;
        }
    }

    private static async Task RunAsync()
    {
        var backend = new PrefillThenFailDecodeBackend(
            new DeviceId("cpu:value-task-terminal"));
        var executor = await ContinuousBatchExecutor.CreateAsync(
            backend,
            capacity: 2,
            maxBatchSize: 2);
        using var submission = ContinuousBatchExecutor.BeginAtomicSubmission(2);

        var model = new ModelId("value-task-terminal-model");
        var prefillSequence = SequenceId.New();
        var decodeSequence = SequenceId.New();

        Task<BackendStepResult> prefill;
        using (submission.EnterSlot(0))
        {
            prefill = executor.SubmitPrefillAsync(
                    new PrefillItem(
                        prefillSequence,
                        model,
                        new ReadOnlyMemory<int>(new[] { 7 })))
                .AsTask();
        }

        Task<BackendStepResult> decode;
        using (submission.EnterSlot(1))
        {
            decode = executor.SubmitDecodeAsync(
                    new DecodeItem(decodeSequence, model, Position: 1))
                .AsTask();
        }

        var prefillResult = await prefill.WaitAsync(TimeSpan.FromSeconds(5));
        Require(
            prefillResult.SequenceId == prefillSequence &&
            prefillResult.TokenId == 701,
            "A successful inference item must retain its first terminal result when later envelope cleanup re-signals failure.");

        var decodeFailed = false;
        try
        {
            await decode.WaitAsync(TimeSpan.FromSeconds(5));
        }
        catch (InvalidOperationException exception) when (
            exception.Message == DecodeFailureMessage)
        {
            decodeFailed = true;
        }

        Require(
            decodeFailed,
            "The later failing inference item must preserve the backend failure while the earlier item remains successful.");

        var disposeFailed = false;
        try
        {
            await executor.DisposeAsync();
        }
        catch (InvalidOperationException exception) when (
            exception.Message == DecodeFailureMessage)
        {
            disposeFailed = true;
        }

        Require(
            disposeFailed,
            "The actor must continue to surface the original backend failure during disposal.");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class PrefillThenFailDecodeBackend(DeviceId device) : IInferenceBackend
    {
        private bool _initialized;

        public string Name => "prefill-then-fail-decode";
        public DeviceId Device { get; } = device;

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
            if (batch.Items.Count != 1)
            {
                throw new InvalidOperationException(
                    $"Expected one prefill item, received {batch.Items.Count}.");
            }

            return ValueTask.FromResult<IReadOnlyList<BackendStepResult>>(
                new[]
                {
                    new BackendStepResult(batch.Items[0].SequenceId, TokenId: 701)
                });
        }

        public ValueTask<IReadOnlyList<BackendStepResult>> DecodeAsync(
            DecodeBatch batch,
            CancellationToken cancellationToken = default)
        {
            EnsureInitialized();
            cancellationToken.ThrowIfCancellationRequested();
            throw new InvalidOperationException(DecodeFailureMessage);
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
                throw new InvalidOperationException("Backend is not initialized.");
            }
        }
    }
}
