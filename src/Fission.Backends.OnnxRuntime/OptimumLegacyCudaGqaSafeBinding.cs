using Fission.Abstractions.Execution;
using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Correctness policy for CUDA GroupQueryAttention exports whose continuation
/// prefill kernel requires batch size one when both a non-empty past and more
/// than one new token are supplied.
///
/// Initial prefill and one-token continuation remain batchable. Multi-token
/// continuation cohorts are executed as singleton CUDA runs. Their independently
/// produced KV frontiers are intentionally left device-resident; the wrapped
/// <see cref="OptimumLegacyCudaGatheringBinding"/> gathers those rows D2D on the
/// next batched decode, after which normal dense zero-copy decode batching resumes.
/// </summary>
public sealed class OptimumLegacyCudaGqaSafeBinding :
    IDecoderOrtAsyncBatchPrefillModelBinding,
    IDecoderOrtAsyncBatchModelBinding,
    IDecoderOrtChunkedPrefillModelBinding,
    IDecoderOrtCudaResidentStateBinding
{
    private readonly OptimumLegacyCudaGatheringBinding _inner;
    private long _continuationSingletonRunCount;
    private int _disposed;

    public OptimumLegacyCudaGqaSafeBinding(
        OptimumLegacyCudaGatheringBinding inner)
    {
        ArgumentNullException.ThrowIfNull(inner);
        _inner = inner;
    }

    public string Name => $"{_inner.Name}-gqa-safe";
    public OnnxSessionContract SessionContract => _inner.SessionContract;
    public DecoderOrtGeometry Geometry => _inner.Geometry;
    public int DeviceId => _inner.DeviceId;
    public string CudaResidentStateFormatId => _inner.CudaResidentStateFormatId;
    public int OrtRunCount => _inner.OrtRunCount;
    public int CudaPastReuseCount => _inner.CudaPastReuseCount;
    public long GatheredBatchCount => _inner.GatheredBatchCount;
    public long GatheredBytes => _inner.GatheredBytes;
    public long SingletonFallbackRunCount => _inner.SingletonFallbackRunCount;
    public long GqaContinuationSingletonRunCount =>
        Interlocked.Read(ref _continuationSingletonRunCount);

    public ValueTask InitializeAsync(
        InferenceSession session,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        return _inner.InitializeAsync(session, cancellationToken);
    }

    public DecoderOrtCudaResidentStateLease AcquireCudaResidentState(
        DecoderOrtState state,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        return _inner.AcquireCudaResidentState(state, cancellationToken);
    }

    public DecoderOrtStepResult ExecutePrefill(
        InferenceSession session,
        PrefillItem item,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        return _inner.ExecutePrefill(session, item, cancellationToken);
    }

    public DecoderOrtStepResult ExecutePrefillChunk(
        InferenceSession session,
        PrefillItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        return _inner.ExecutePrefillChunk(
            session,
            item,
            priorState,
            cancellationToken);
    }

    public DecoderOrtStepResult ExecuteDecode(
        InferenceSession session,
        DecodeItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        return _inner.ExecuteDecode(
            session,
            item,
            priorState,
            cancellationToken);
    }

    public async ValueTask<IReadOnlyList<DecoderOrtStepResult>> ExecutePrefillBatchAsync(
        InferenceSession session,
        IReadOnlyList<PrefillItem> items,
        IReadOnlyList<DecoderOrtState?> priorStates,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(items);
        ArgumentNullException.ThrowIfNull(priorStates);
        cancellationToken.ThrowIfCancellationRequested();

        if (items.Count != priorStates.Count)
        {
            throw new ArgumentException(
                $"Prefill batch contains {items.Count} items but {priorStates.Count} prior states.",
                nameof(priorStates));
        }

        if (items.Count == 0)
        {
            return Array.Empty<DecoderOrtStepResult>();
        }

        var cohorts = new SortedDictionary<(int Position, int SequenceLength), List<int>>();
        for (var index = 0; index < items.Count; index++)
        {
            var position = items[index].Position ?? priorStates[index]?.Position ?? 0;
            var sequenceLength = items[index].Tokens.Length;
            var key = (position, sequenceLength);
            if (!cohorts.TryGetValue(key, out var cohort))
            {
                cohort = new List<int>();
                cohorts.Add(key, cohort);
            }

            cohort.Add(index);
        }

        var results = new DecoderOrtStepResult[items.Count];
        var producedStates = new List<DecoderOrtState>(items.Count);
        try
        {
            foreach (var ((position, sequenceLength), cohort) in cohorts)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var requiresSingletonGqaContinuation =
                    cohort.Count > 1 &&
                    position > 0 &&
                    sequenceLength > 1;

                if (requiresSingletonGqaContinuation)
                {
                    foreach (var itemIndex in cohort)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        var priorState = priorStates[itemIndex] ??
                            throw new InvalidOperationException(
                                "CUDA GQA continuation prefill requires an existing decoder state.");
                        var produced = _inner.ExecutePrefillChunk(
                            session,
                            items[itemIndex],
                            priorState,
                            cancellationToken);
                        results[itemIndex] = produced;
                        producedStates.Add(produced.State);
                        Interlocked.Increment(ref _continuationSingletonRunCount);
                    }

                    continue;
                }

                var cohortItems = cohort.Select(index => items[index]).ToArray();
                var cohortPrior = cohort.Select(index => priorStates[index]).ToArray();
                var producedBatch = await _inner.ExecutePrefillBatchAsync(
                        session,
                        cohortItems,
                        cohortPrior,
                        cancellationToken)
                    .ConfigureAwait(false);
                AssignResults(
                    results,
                    producedStates,
                    cohort,
                    producedBatch);
            }

            return results;
        }
        catch
        {
            DisposeProducedStates(producedStates);
            throw;
        }
    }

    public ValueTask<IReadOnlyList<DecoderOrtStepResult>> ExecuteDecodeBatchAsync(
        InferenceSession session,
        IReadOnlyList<DecodeItem> items,
        IReadOnlyList<DecoderOrtState> priorStates,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        return _inner.ExecuteDecodeBatchAsync(
            session,
            items,
            priorStates,
            cancellationToken);
    }

    private static void AssignResults(
        DecoderOrtStepResult[] destination,
        List<DecoderOrtState> producedStates,
        IReadOnlyList<int> cohort,
        IReadOnlyList<DecoderOrtStepResult> produced)
    {
        if (produced.Count != cohort.Count)
        {
            DisposeResultStates(produced);
            throw new InvalidOperationException(
                $"CUDA GQA safety binding returned {produced.Count} results for {cohort.Count} cohort items.");
        }

        var uniqueStates = new HashSet<DecoderOrtState>(ReferenceEqualityComparer.Instance);
        for (var index = 0; index < produced.Count; index++)
        {
            var result = produced[index];
            if (result is null || result.State is null)
            {
                DisposeResultStates(produced);
                throw new InvalidOperationException(
                    "CUDA GQA safety binding received a null decoder result/state.");
            }

            if (!uniqueStates.Add(result.State))
            {
                DisposeResultStates(produced);
                throw new InvalidOperationException(
                    "CUDA GQA safety binding received one physical state for multiple rows.");
            }
        }

        for (var index = 0; index < cohort.Count; index++)
        {
            var result = produced[index];
            destination[cohort[index]] = result;
            producedStates.Add(result.State);
        }
    }

    private static void DisposeResultStates(
        IReadOnlyList<DecoderOrtStepResult> results)
    {
        var seen = new HashSet<DecoderOrtState>(ReferenceEqualityComparer.Instance);
        for (var index = results.Count - 1; index >= 0; index--)
        {
            var state = results[index]?.State;
            if (state is not null && seen.Add(state))
            {
                state.Dispose();
            }
        }
    }

    private static void DisposeProducedStates(List<DecoderOrtState> states)
    {
        var seen = new HashSet<DecoderOrtState>(ReferenceEqualityComparer.Instance);
        for (var index = states.Count - 1; index >= 0; index--)
        {
            if (seen.Add(states[index]))
            {
                states[index].Dispose();
            }
        }
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _inner.Dispose();
    }
}
