using System.Buffers;
using Fission.Abstractions.Execution;
using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Correctness-first batching policy over <see cref="OptimumLegacyCudaFloatDecoderBinding"/>.
///
/// Dense complete CUDA cohorts remain batched and zero-copy. Multi-row layouts
/// that would require a device gather (subset rows, duplicate/forked rows, or
/// mixed arenas) are split into singleton CUDA executions instead of falling back
/// through host KV packing. This preserves device residency and correctness while
/// making the throughput penalty explicit through <see cref="SingletonFallbackRunCount"/>.
/// </summary>
public sealed class OptimumLegacyCudaSplitFallbackBinding :
    IDecoderOrtBatchPrefillModelBinding,
    IDecoderOrtBatchModelBinding,
    IDecoderOrtChunkedPrefillModelBinding,
    IDecoderOrtCudaResidentStateBinding
{
    private readonly OptimumLegacyCudaFloatDecoderBinding _inner;
    private int _singletonFallbackRunCount;
    private int _disposed;

    public OptimumLegacyCudaSplitFallbackBinding(
        OptimumLegacyDecoderProfile profile,
        CudaDeviceMemoryAllocator allocator,
        IEnumerable<int>? eosTokenIds = null,
        ArrayPool<float>? scratchFloatPool = null,
        ArrayPool<long>? scratchLongPool = null)
        : this(new OptimumLegacyCudaFloatDecoderBinding(
            profile,
            allocator,
            eosTokenIds,
            scratchFloatPool,
            scratchLongPool))
    {
    }

    internal OptimumLegacyCudaSplitFallbackBinding(
        OptimumLegacyCudaFloatDecoderBinding inner)
    {
        ArgumentNullException.ThrowIfNull(inner);
        _inner = inner;
    }

    public string Name => $"{_inner.Name}-split-fallback";
    public OnnxSessionContract SessionContract => _inner.SessionContract;
    public DecoderOrtGeometry Geometry => _inner.Geometry;
    public int DeviceId => _inner.DeviceId;
    public string CudaResidentStateFormatId => _inner.CudaResidentStateFormatId;
    public int OrtRunCount => _inner.OrtRunCount;
    public int CudaPastReuseCount => _inner.CudaPastReuseCount;
    public int SingletonFallbackRunCount => Volatile.Read(ref _singletonFallbackRunCount);

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

    public IReadOnlyList<DecoderOrtStepResult> ExecutePrefillBatch(
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
            if (!cohorts.TryGetValue(key, out var indices))
            {
                indices = new List<int>();
                cohorts.Add(key, indices);
            }

            indices.Add(index);
        }

        var results = new DecoderOrtStepResult[items.Count];
        var producedStates = new List<DecoderOrtState>(items.Count);
        try
        {
            foreach (var ((position, _), cohort) in cohorts)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var canBatch = position == 0
                    ? cohort.All(index => priorStates[index] is null)
                    : CanReuseDenseCudaCohort(priorStates, cohort, position);

                if (cohort.Count == 1 || canBatch)
                {
                    var cohortItems = cohort.Select(index => items[index]).ToArray();
                    var cohortStates = cohort.Select(index => priorStates[index]).ToArray();
                    var produced = _inner.ExecutePrefillBatch(
                        session,
                        cohortItems,
                        cohortStates,
                        cancellationToken);
                    AssignResults(
                        results,
                        producedStates,
                        cohort,
                        produced);
                    continue;
                }

                foreach (var itemIndex in cohort)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var prior = priorStates[itemIndex];
                    var produced = prior is null
                        ? _inner.ExecutePrefill(
                            session,
                            items[itemIndex],
                            cancellationToken)
                        : _inner.ExecutePrefillChunk(
                            session,
                            items[itemIndex],
                            prior,
                            cancellationToken);
                    results[itemIndex] = produced;
                    producedStates.Add(produced.State);
                    Interlocked.Increment(ref _singletonFallbackRunCount);
                }
            }

            return results;
        }
        catch
        {
            DisposeProducedStates(producedStates);
            throw;
        }
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

    public IReadOnlyList<DecoderOrtStepResult> ExecuteDecodeBatch(
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
            if (!cohorts.TryGetValue(items[index].Position, out var indices))
            {
                indices = new List<int>();
                cohorts.Add(items[index].Position, indices);
            }

            indices.Add(index);
        }

        var results = new DecoderOrtStepResult[items.Count];
        var producedStates = new List<DecoderOrtState>(items.Count);
        try
        {
            foreach (var (position, cohort) in cohorts)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (cohort.Count == 1 ||
                    CanReuseDenseCudaCohort(priorStates, cohort, position))
                {
                    var cohortItems = cohort.Select(index => items[index]).ToArray();
                    var cohortStates = cohort.Select(index => priorStates[index]).ToArray();
                    var produced = _inner.ExecuteDecodeBatch(
                        session,
                        cohortItems,
                        cohortStates,
                        cancellationToken);
                    AssignResults(
                        results,
                        producedStates,
                        cohort,
                        produced);
                    continue;
                }

                foreach (var itemIndex in cohort)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var produced = _inner.ExecuteDecode(
                        session,
                        items[itemIndex],
                        priorStates[itemIndex],
                        cancellationToken);
                    results[itemIndex] = produced;
                    producedStates.Add(produced.State);
                    Interlocked.Increment(ref _singletonFallbackRunCount);
                }
            }

            return results;
        }
        catch
        {
            DisposeProducedStates(producedStates);
            throw;
        }
    }

    private static bool CanReuseDenseCudaCohort(
        IReadOnlyList<DecoderOrtState?> priorStates,
        IReadOnlyList<int> cohort,
        int position)
    {
        if (cohort.Count <= 1)
        {
            return true;
        }

        var first = priorStates[cohort[0]];
        if (first is null ||
            first.IsDisposed ||
            first.Position != position ||
            !first.TryGetCudaCohortSlice(out var firstSlice))
        {
            return false;
        }

        var arena = firstSlice.Arena;
        if (arena.Position != position || arena.BatchSize != cohort.Count)
        {
            return false;
        }

        var rows = new bool[cohort.Count];
        foreach (var itemIndex in cohort)
        {
            var state = priorStates[itemIndex];
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
                $"CUDA split fallback inner binding returned {produced.Count} results for {cohort.Count} items.");
        }

        // Validate the entire returned collection before transferring any state
        // into the wrapper's cross-cohort ownership tracker. That keeps handoff
        // atomic even if a buggy/custom inner binding violates its result contract.
        for (var index = 0; index < produced.Count; index++)
        {
            if (produced[index] is null || produced[index].State is null)
            {
                DisposeResultStates(produced);
                throw new InvalidOperationException(
                    "CUDA split fallback inner binding returned a null decoder result/state.");
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
        foreach (var result in results)
        {
            if (result?.State is { } state && seen.Add(state))
            {
                state.Dispose();
            }
        }
    }

    private static void DisposeProducedStates(List<DecoderOrtState> states)
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
