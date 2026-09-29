using System.Buffers;
using Fission.Abstractions.Execution;
using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Async continuous-batching policy over <see cref="OptimumLegacyCudaFloatDecoderBinding"/>.
/// Complete dense CUDA cohorts and singleton states keep the direct zero-copy path.
/// Unsupported multi-row layouts are gathered device-to-device into one temporary
/// dense cohort and then executed by the existing CUDA binding as one ORT batch.
/// The allocator and copy engine remain caller-owned; this wrapper owns its inner binding.
/// </summary>
public sealed class OptimumLegacyCudaGatheringBinding :
    IDecoderOrtAsyncBatchPrefillModelBinding,
    IDecoderOrtAsyncBatchModelBinding,
    IDecoderOrtChunkedPrefillModelBinding,
    IDecoderOrtCudaResidentStateBinding
{
    private readonly OptimumLegacyDecoderProfile _profile;
    private readonly CudaDeviceMemoryAllocator _allocator;
    private readonly CudaDeviceBoundAsyncCopyEngine _copyEngine;
    private readonly OptimumLegacyCudaFloatDecoderBinding _inner;
    private long _gatheredBatchCount;
    private long _gatheredBytes;
    private long _singletonFallbackRunCount;
    private int _disposed;

    public OptimumLegacyCudaGatheringBinding(
        OptimumLegacyDecoderProfile profile,
        CudaDeviceMemoryAllocator allocator,
        CudaDeviceBoundAsyncCopyEngine copyEngine,
        IEnumerable<int>? eosTokenIds = null,
        ArrayPool<float>? scratchFloatPool = null,
        ArrayPool<long>? scratchLongPool = null)
        : this(
            profile,
            allocator,
            copyEngine,
            new OptimumLegacyCudaFloatDecoderBinding(
                profile,
                allocator,
                eosTokenIds,
                scratchFloatPool,
                scratchLongPool))
    {
    }

    internal OptimumLegacyCudaGatheringBinding(
        OptimumLegacyDecoderProfile profile,
        CudaDeviceMemoryAllocator allocator,
        CudaDeviceBoundAsyncCopyEngine copyEngine,
        OptimumLegacyCudaFloatDecoderBinding inner)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentNullException.ThrowIfNull(allocator);
        ArgumentNullException.ThrowIfNull(copyEngine);
        ArgumentNullException.ThrowIfNull(inner);

        if (allocator.DeviceId != copyEngine.DeviceId || allocator.DeviceId != inner.DeviceId)
        {
            throw new ArgumentException(
                "CUDA gather binding allocator, copy engine, and inner binding must target the same device.",
                nameof(copyEngine));
        }

        if (profile.Geometry != inner.Geometry)
        {
            throw new ArgumentException(
                "CUDA gather binding profile geometry must match the inner binding geometry.",
                nameof(inner));
        }

        _profile = profile;
        _allocator = allocator;
        _copyEngine = copyEngine;
        _inner = inner;
    }

    public string Name => $"{_inner.Name}-d2d-gather";
    public OnnxSessionContract SessionContract => _inner.SessionContract;
    public DecoderOrtGeometry Geometry => _inner.Geometry;
    public int DeviceId => _inner.DeviceId;
    public string CudaResidentStateFormatId => _inner.CudaResidentStateFormatId;
    public int OrtRunCount => _inner.OrtRunCount;
    public int CudaPastReuseCount => _inner.CudaPastReuseCount;
    public long GatheredBatchCount => Interlocked.Read(ref _gatheredBatchCount);
    public long GatheredBytes => Interlocked.Read(ref _gatheredBytes);
    public long SingletonFallbackRunCount => Interlocked.Read(ref _singletonFallbackRunCount);

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
        return _inner.ExecutePrefillChunk(session, item, priorState, cancellationToken);
    }

    public DecoderOrtStepResult ExecuteDecode(
        InferenceSession session,
        DecodeItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        return _inner.ExecuteDecode(session, item, priorState, cancellationToken);
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
            var key = (position, items[index].Tokens.Length);
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
            foreach (var ((position, _), cohort) in cohorts)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var cohortItems = cohort.Select(index => items[index]).ToArray();
                var cohortPrior = cohort.Select(index => priorStates[index]).ToArray();

                IReadOnlyList<DecoderOrtStepResult> produced;
                if (cohort.Count == 1 || CanReuseDirectly(cohortPrior, position))
                {
                    produced = _inner.ExecutePrefillBatch(
                        session,
                        cohortItems,
                        cohortPrior,
                        cancellationToken);
                }
                else if (position > 0 && cohortPrior.All(static state => state is not null))
                {
                    produced = await ExecuteGatheredPrefillCohortAsync(
                            session,
                            cohortItems,
                            cohortPrior.Select(static state => state!).ToArray(),
                            cancellationToken)
                        .ConfigureAwait(false);
                }
                else
                {
                    produced = ExecutePrefillSingletonFallback(
                        session,
                        cohortItems,
                        cohortPrior,
                        cancellationToken);
                }

                AssignResults(results, producedStates, cohort, produced);
            }

            return results;
        }
        catch
        {
            DisposeProducedStates(producedStates);
            throw;
        }
    }

    public async ValueTask<IReadOnlyList<DecoderOrtStepResult>> ExecuteDecodeBatchAsync(
        InferenceSession session,
        IReadOnlyList<DecodeItem> items,
        IReadOnlyList<DecoderOrtState> priorStates,
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
                $"Decode batch contains {items.Count} items but {priorStates.Count} prior states.",
                nameof(priorStates));
        }

        if (items.Count == 0)
        {
            return Array.Empty<DecoderOrtStepResult>();
        }

        var cohorts = new SortedDictionary<int, List<int>>();
        for (var index = 0; index < items.Count; index++)
        {
            if (!cohorts.TryGetValue(items[index].Position, out var cohort))
            {
                cohort = new List<int>();
                cohorts.Add(items[index].Position, cohort);
            }

            cohort.Add(index);
        }

        var results = new DecoderOrtStepResult[items.Count];
        var producedStates = new List<DecoderOrtState>(items.Count);
        try
        {
            foreach (var (position, cohort) in cohorts)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var cohortItems = cohort.Select(index => items[index]).ToArray();
                var cohortPrior = cohort.Select(index => priorStates[index]).ToArray();
                var nullablePrior = cohortPrior
                    .Select(static state => (DecoderOrtState?)state)
                    .ToArray();

                IReadOnlyList<DecoderOrtStepResult> produced;
                if (cohort.Count == 1 || CanReuseDirectly(nullablePrior, position))
                {
                    produced = _inner.ExecuteDecodeBatch(
                        session,
                        cohortItems,
                        cohortPrior,
                        cancellationToken);
                }
                else if (position > 0)
                {
                    produced = await ExecuteGatheredDecodeCohortAsync(
                            session,
                            cohortItems,
                            cohortPrior,
                            cancellationToken)
                        .ConfigureAwait(false);
                }
                else
                {
                    produced = ExecuteDecodeSingletonFallback(
                        session,
                        cohortItems,
                        cohortPrior,
                        cancellationToken);
                }

                AssignResults(results, producedStates, cohort, produced);
            }

            return results;
        }
        catch
        {
            DisposeProducedStates(producedStates);
            throw;
        }
    }

    private async ValueTask<IReadOnlyList<DecoderOrtStepResult>> ExecuteGatheredPrefillCohortAsync(
        InferenceSession session,
        IReadOnlyList<PrefillItem> items,
        IReadOnlyList<DecoderOrtState> sourceStates,
        CancellationToken cancellationToken)
    {
        using var gathered = await OptimumLegacyCudaGatheredPastKvBatch.CreateAsync(
                _profile,
                sourceStates,
                CudaResidentStateFormatId,
                _allocator,
                _copyEngine,
                cancellationToken)
            .ConfigureAwait(false);

        var gatheredStates = CreateGatheredStates(gathered, sourceStates);
        try
        {
            var produced = _inner.ExecutePrefillBatch(
                session,
                items,
                gatheredStates.Select(static state => (DecoderOrtState?)state).ToArray(),
                cancellationToken);
            RecordGather(gathered);
            return produced;
        }
        finally
        {
            DisposeStates(gatheredStates);
        }
    }

    private async ValueTask<IReadOnlyList<DecoderOrtStepResult>> ExecuteGatheredDecodeCohortAsync(
        InferenceSession session,
        IReadOnlyList<DecodeItem> items,
        IReadOnlyList<DecoderOrtState> sourceStates,
        CancellationToken cancellationToken)
    {
        using var gathered = await OptimumLegacyCudaGatheredPastKvBatch.CreateAsync(
                _profile,
                sourceStates,
                CudaResidentStateFormatId,
                _allocator,
                _copyEngine,
                cancellationToken)
            .ConfigureAwait(false);

        var gatheredStates = CreateGatheredStates(gathered, sourceStates);
        try
        {
            var produced = _inner.ExecuteDecodeBatch(
                session,
                items,
                gatheredStates,
                cancellationToken);
            RecordGather(gathered);
            return produced;
        }
        finally
        {
            DisposeStates(gatheredStates);
        }
    }

    private static DecoderOrtState[] CreateGatheredStates(
        OptimumLegacyCudaGatheredPastKvBatch gathered,
        IReadOnlyList<DecoderOrtState> sourceStates)
    {
        var states = new DecoderOrtState[sourceStates.Count];
        var produced = 0;
        try
        {
            for (; produced < states.Length; produced++)
            {
                states[produced] = gathered.CreateRowState(
                    produced,
                    sourceStates[produced].NextTokenId);
            }

            return states;
        }
        catch
        {
            for (var index = produced - 1; index >= 0; index--)
            {
                states[index].Dispose();
            }

            throw;
        }
    }

    private IReadOnlyList<DecoderOrtStepResult> ExecutePrefillSingletonFallback(
        InferenceSession session,
        IReadOnlyList<PrefillItem> items,
        IReadOnlyList<DecoderOrtState?> priorStates,
        CancellationToken cancellationToken)
    {
        var results = new DecoderOrtStepResult[items.Count];
        var produced = 0;
        try
        {
            for (; produced < items.Count; produced++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var prior = priorStates[produced];
                results[produced] = prior is null
                    ? _inner.ExecutePrefill(session, items[produced], cancellationToken)
                    : _inner.ExecutePrefillChunk(
                        session,
                        items[produced],
                        prior,
                        cancellationToken);
                Interlocked.Increment(ref _singletonFallbackRunCount);
            }

            return results;
        }
        catch
        {
            DisposeResultStates(results, produced);
            throw;
        }
    }

    private IReadOnlyList<DecoderOrtStepResult> ExecuteDecodeSingletonFallback(
        InferenceSession session,
        IReadOnlyList<DecodeItem> items,
        IReadOnlyList<DecoderOrtState> priorStates,
        CancellationToken cancellationToken)
    {
        var results = new DecoderOrtStepResult[items.Count];
        var produced = 0;
        try
        {
            for (; produced < items.Count; produced++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                results[produced] = _inner.ExecuteDecode(
                    session,
                    items[produced],
                    priorStates[produced],
                    cancellationToken);
                Interlocked.Increment(ref _singletonFallbackRunCount);
            }

            return results;
        }
        catch
        {
            DisposeResultStates(results, produced);
            throw;
        }
    }

    private static bool CanReuseDirectly(
        IReadOnlyList<DecoderOrtState?> priorStates,
        int position)
    {
        if (priorStates.Count <= 1)
        {
            return true;
        }

        if (position == 0 && priorStates.All(static state => state is null))
        {
            return true;
        }

        var first = priorStates[0];
        if (first is null ||
            first.IsDisposed ||
            first.Position != position ||
            !first.TryGetCudaCohortSlice(out var firstSlice))
        {
            return false;
        }

        var arena = firstSlice.Arena;
        if (arena.Position != position || arena.BatchSize != priorStates.Count)
        {
            return false;
        }

        var rows = new bool[priorStates.Count];
        foreach (var state in priorStates)
        {
            if (state is null ||
                state.IsDisposed ||
                state.Position != position ||
                !state.TryGetCudaCohortSlice(out var slice) ||
                !ReferenceEquals(slice.Arena, arena) ||
                slice.Row < 0 ||
                slice.Row >= rows.Length ||
                rows[slice.Row])
            {
                return false;
            }

            rows[slice.Row] = true;
        }

        return rows.All(static present => present);
    }

    private void RecordGather(OptimumLegacyCudaGatheredPastKvBatch gathered)
    {
        Interlocked.Increment(ref _gatheredBatchCount);
        Interlocked.Add(ref _gatheredBytes, gathered.CopiedBytes);
    }

    private static void AssignResults(
        DecoderOrtStepResult[] destination,
        List<DecoderOrtState> producedStates,
        IReadOnlyList<int> cohort,
        IReadOnlyList<DecoderOrtStepResult> produced)
    {
        if (produced.Count != cohort.Count)
        {
            DisposeResultStates(produced, produced.Count);
            throw new InvalidOperationException(
                $"CUDA gather binding returned {produced.Count} results for {cohort.Count} cohort items.");
        }

        var uniqueStates = new HashSet<DecoderOrtState>(ReferenceEqualityComparer.Instance);
        for (var index = 0; index < produced.Count; index++)
        {
            var result = produced[index];
            if (result is null || result.State is null)
            {
                DisposeResultStates(produced, produced.Count);
                throw new InvalidOperationException(
                    "CUDA gather binding inner decoder returned a null result/state.");
            }

            if (!uniqueStates.Add(result.State))
            {
                DisposeResultStates(produced, produced.Count);
                throw new InvalidOperationException(
                    "CUDA gather binding inner decoder returned the same physical state for multiple rows.");
            }
        }

        for (var index = 0; index < produced.Count; index++)
        {
            var result = produced[index];
            destination[cohort[index]] = result;
            producedStates.Add(result.State);
        }
    }

    private static void DisposeResultStates(
        IReadOnlyList<DecoderOrtStepResult> results,
        int count)
    {
        var seen = new HashSet<DecoderOrtState>(ReferenceEqualityComparer.Instance);
        for (var index = count - 1; index >= 0; index--)
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

    private static void DisposeStates(IReadOnlyList<DecoderOrtState> states)
    {
        for (var index = states.Count - 1; index >= 0; index--)
        {
            states[index].Dispose();
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
