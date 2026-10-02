using System.Buffers;
using System.Collections.ObjectModel;
using Fission.Abstractions.Execution;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Fission.Backends.OnnxRuntime;

internal delegate void CallerOwnedOrtRunInvoker(
    InferenceSession session,
    RunOptions runOptions,
    IReadOnlyCollection<string> inputNames,
    IReadOnlyCollection<OrtValue> inputValues,
    IReadOnlyCollection<string> outputNames,
    IReadOnlyCollection<OrtValue> outputValues);

/// <summary>
/// CUDA-resident FP32 Optimum legacy decoder-with-past binding.
///
/// Scalar token/position/mask inputs and logits remain host-backed. Past and
/// present KV stay CUDA-resident: complete multi-row CUDA cohorts are reused
/// zero-copy, while singleton CUDA states (including migration imports) are
/// retained through the generic resident-state capability and rebound directly.
/// Unsupported multi-row subset/reordered layouts fail explicitly until a
/// device-side gather path exists.
/// </summary>
public sealed class OptimumLegacyCudaFloatDecoderBinding :
    IDecoderOrtBatchPrefillModelBinding,
    IDecoderOrtBatchModelBinding,
    IDecoderOrtChunkedPrefillModelBinding,
    IDecoderOrtCudaResidentStateBinding
{
    private readonly OptimumLegacyDecoderProfile _profile;
    private readonly CudaDeviceMemoryAllocator _allocator;
    private readonly CallerOwnedOrtRunInvoker _runInvoker;
    private readonly HashSet<int> _eosTokenIds;
    private readonly ArrayPool<float> _scratchFloatPool;
    private readonly ArrayPool<long> _scratchLongPool;
    private readonly PinnedFloatBufferPool? _decodeLogitsHostPool;
    private int _ortRunCount;
    private int _cudaPastReuseCount;
    private long _scratchFloatRentCount;
    private long _scratchLongRentCount;
    private long _pageLockedDecodeLogitsRentCount;
    private int _disposed;

    public OptimumLegacyCudaFloatDecoderBinding(
        OptimumLegacyDecoderProfile profile,
        CudaDeviceMemoryAllocator allocator,
        IEnumerable<int>? eosTokenIds = null,
        ArrayPool<float>? scratchFloatPool = null,
        ArrayPool<long>? scratchLongPool = null,
        IHostStagingFloatBufferAllocator? decodeLogitsHostAllocator = null,
        PinnedHostStagingPoolOptions? decodeLogitsHostPoolOptions = null)
        : this(
            profile,
            allocator,
            CallerOwnedOrtRun.Execute,
            eosTokenIds,
            scratchFloatPool,
            scratchLongPool,
            decodeLogitsHostAllocator,
            decodeLogitsHostPoolOptions)
    {
    }

    internal OptimumLegacyCudaFloatDecoderBinding(
        OptimumLegacyDecoderProfile profile,
        CudaDeviceMemoryAllocator allocator,
        CallerOwnedOrtRunInvoker runInvoker,
        IEnumerable<int>? eosTokenIds = null,
        ArrayPool<float>? scratchFloatPool = null,
        ArrayPool<long>? scratchLongPool = null,
        IHostStagingFloatBufferAllocator? decodeLogitsHostAllocator = null,
        PinnedHostStagingPoolOptions? decodeLogitsHostPoolOptions = null)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentNullException.ThrowIfNull(allocator);
        ArgumentNullException.ThrowIfNull(runInvoker);

        if (profile.Geometry.KvElementType != TensorElementType.Float)
        {
            throw new ArgumentException(
                "The CUDA FP32 decoder binding requires Float KV tensors.",
                nameof(profile));
        }

        if (profile.Contract.LogitsElementType != TensorElementType.Float)
        {
            throw new ArgumentException(
                "The CUDA FP32 decoder binding requires Float logits.",
                nameof(profile));
        }

        if (string.IsNullOrWhiteSpace(profile.Contract.AttentionMask) ||
            string.IsNullOrWhiteSpace(profile.Contract.PositionIds) ||
            !profile.Contract.UsesPastKeyValues)
        {
            throw new ArgumentException(
                "The CUDA Optimum FP32 binding requires attention_mask, position_ids, and past/present KV tensors.",
                nameof(profile));
        }

        if ((profile.Contract.AdditionalInputs?.Count ?? 0) != 0 ||
            (profile.Contract.AdditionalOutputs?.Count ?? 0) != 0)
        {
            throw new ArgumentException(
                "Additional model-specific tensors are not supported by the CUDA Optimum FP32 binding.",
                nameof(profile));
        }

        _profile = profile;
        _allocator = allocator;
        _runInvoker = runInvoker;
        _eosTokenIds = eosTokenIds is null
            ? new HashSet<int>()
            : new HashSet<int>(eosTokenIds);
        _scratchFloatPool = scratchFloatPool ?? ArrayPool<float>.Create();
        _scratchLongPool = scratchLongPool ?? ArrayPool<long>.Create();
        _decodeLogitsHostPool = decodeLogitsHostAllocator is null
            ? null
            : new PinnedFloatBufferPool(
                decodeLogitsHostPoolOptions ??
                    new PinnedHostStagingPoolOptions
                    {
                        MaxRetainedBuffersPerLength = 2,
                        MaxRetainedBytes = 64L * 1024L * 1024L,
                        ClearOnReturn = false
                    },
                decodeLogitsHostAllocator);

        if (_eosTokenIds.Any(static token => token < 0))
        {
            throw new ArgumentOutOfRangeException(
                nameof(eosTokenIds),
                "EOS token ids cannot be negative.");
        }

        CudaResidentStateFormatId =
            $"optimum-legacy:fp32:kv4d:l{Geometry.NumHiddenLayers}:h{Geometry.NumKvHeads}:d{Geometry.HeadDim}:cuda-v1";
    }

    public string Name => "optimum-legacy-cuda-fp32-greedy";
    public OnnxSessionContract SessionContract => _profile.SessionContract;
    public DecoderOrtGeometry Geometry => _profile.Geometry;
    public int DeviceId => _allocator.DeviceId;
    public string CudaResidentStateFormatId { get; }
    public int OrtRunCount => Volatile.Read(ref _ortRunCount);
    public int CudaPastReuseCount => Volatile.Read(ref _cudaPastReuseCount);
    public long ScratchFloatRentCount => Interlocked.Read(ref _scratchFloatRentCount);
    public long ScratchLongRentCount => Interlocked.Read(ref _scratchLongRentCount);
    public bool PageLockedDecodeLogitsEnabled => _decodeLogitsHostPool is not null;
    public long PageLockedDecodeLogitsRentCount =>
        Interlocked.Read(ref _pageLockedDecodeLogitsRentCount);

    public DecoderOrtCudaResidentStateLease AcquireCudaResidentState(
        DecoderOrtState state,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(state);
        cancellationToken.ThrowIfCancellationRequested();
        var lease = state.AcquireCudaResidentState(CudaResidentStateFormatId);
        if (lease.DeviceId != DeviceId)
        {
            lease.Dispose();
            throw new InvalidOperationException(
                $"CUDA decoder state belongs to device {lease.DeviceId}, but this binding targets device {DeviceId}.");
        }

        return lease;
    }

    public DecoderOrtStepResult ExecutePrefill(
        InferenceSession session,
        PrefillItem item,
        CancellationToken cancellationToken = default)
    {
        var results = ExecutePrefillBatch(
            session,
            new[] { item },
            new DecoderOrtState?[] { null },
            cancellationToken);
        return results[0];
    }

    public DecoderOrtStepResult ExecutePrefillChunk(
        InferenceSession session,
        PrefillItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(priorState);
        var results = ExecutePrefillBatch(
            session,
            new[] { item },
            new DecoderOrtState?[] { priorState },
            cancellationToken);
        return results[0];
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

        var normalizedItems = new PrefillItem[items.Count];
        var cohorts = new SortedDictionary<(int Position, int SequenceLength), List<int>>();
        for (var index = 0; index < items.Count; index++)
        {
            var startPosition = ValidatePrefillState(items[index], priorStates[index]);
            var normalized = items[index] with { Position = startPosition };
            normalizedItems[index] = normalized;

            var key = (startPosition, normalized.Tokens.Length);
            if (!cohorts.TryGetValue(key, out var indices))
            {
                indices = new List<int>();
                cohorts.Add(key, indices);
            }

            indices.Add(index);
        }

        var results = new DecoderOrtStepResult[items.Count];
        var committedStates = new List<DecoderOrtState>(items.Count);
        try
        {
            foreach (var cohort in cohorts.Values)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var produced = ExecutePrefillCohort(
                    session,
                    normalizedItems,
                    priorStates,
                    cohort,
                    cancellationToken);

                for (var cohortIndex = 0; cohortIndex < cohort.Count; cohortIndex++)
                {
                    var itemIndex = cohort[cohortIndex];
                    results[itemIndex] = produced[cohortIndex];
                    committedStates.Add(produced[cohortIndex].State);
                }
            }

            return results;
        }
        catch
        {
            for (var index = committedStates.Count - 1; index >= 0; index--)
            {
                committedStates[index].Dispose();
            }

            throw;
        }
    }

    public DecoderOrtStepResult ExecuteDecode(
        InferenceSession session,
        DecodeItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(priorState);
        var results = ExecuteDecodeBatch(
            session,
            new[] { item },
            new[] { priorState },
            cancellationToken);
        return results[0];
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

        var nullablePriorStates = new DecoderOrtState?[priorStates.Count];
        var cohorts = new SortedDictionary<int, List<int>>();
        for (var index = 0; index < items.Count; index++)
        {
            var priorState = priorStates[index];
            _ = ValidateDecodeState(items[index], priorState);
            nullablePriorStates[index] = priorState;

            if (!cohorts.TryGetValue(items[index].Position, out var indices))
            {
                indices = new List<int>();
                cohorts.Add(items[index].Position, indices);
            }

            indices.Add(index);
        }

        var results = new DecoderOrtStepResult[items.Count];
        var committedStates = new List<DecoderOrtState>(items.Count);
        try
        {
            foreach (var cohort in cohorts.Values)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var produced = ExecuteDecodeCohort(
                    session,
                    items,
                    nullablePriorStates,
                    cohort,
                    cancellationToken);

                for (var cohortIndex = 0; cohortIndex < cohort.Count; cohortIndex++)
                {
                    var itemIndex = cohort[cohortIndex];
                    results[itemIndex] = produced[cohortIndex];
                    committedStates.Add(produced[cohortIndex].State);
                }
            }

            return results;
        }
        catch
        {
            for (var index = committedStates.Count - 1; index >= 0; index--)
            {
                committedStates[index].Dispose();
            }

            throw;
        }
    }

    private DecoderOrtStepResult[] ExecutePrefillCohort(
        InferenceSession session,
        IReadOnlyList<PrefillItem> items,
        IReadOnlyList<DecoderOrtState?> priorStates,
        IReadOnlyList<int> cohort,
        CancellationToken cancellationToken)
    {
        if (cohort.Count == 0)
        {
            return Array.Empty<DecoderOrtStepResult>();
        }

        var first = items[cohort[0]];
        var pastSequenceLength = first.Position
            ?? throw new InvalidOperationException("Prefill cohort items must have a normalized position.");
        var sequenceLength = first.Tokens.Length;
        var execution = BuildCudaCohortExecution(
            priorStates,
            cohort,
            pastSequenceLength,
            Geometry.NumHiddenLayers);
        var scratchLeases = new List<IDisposable>(4);
        var handedOff = false;

        try
        {
            var batchSize = cohort.Count;
            var inputIdsLease = RentLongScratch(
                checked(batchSize * sequenceLength),
                scratchLeases);
            var positionIdsLease = RentLongScratch(
                checked(batchSize * sequenceLength),
                scratchLeases);

            for (var row = 0; row < batchSize; row++)
            {
                var itemIndex = execution.ItemIndicesByRow[row];
                var item = items[itemIndex];
                if (item.Position != pastSequenceLength ||
                    item.Tokens.Length != sequenceLength)
                {
                    throw new InvalidOperationException(
                        "Prefill cohort contains more than one past position or chunk length.");
                }

                var offset = checked(row * sequenceLength);
                CopyPromptTokens(
                    item.Tokens.Span,
                    inputIdsLease.Span.Slice(offset, sequenceLength));
                var positions = positionIdsLease.Span.Slice(offset, sequenceLength);
                for (var token = 0; token < sequenceLength; token++)
                {
                    positions[token] = checked((long)pastSequenceLength + token);
                }
            }

            handedOff = true;
            return ExecuteCohort(
                session,
                priorStates,
                execution,
                pastSequenceLength,
                sequenceLength,
                inputIdsLease.Memory,
                positionIdsLease.Memory,
                scratchLeases,
                useDecodeLogitsHostPool: false,
                cancellationToken);
        }
        finally
        {
            if (!handedOff)
            {
                DisposeLeases(scratchLeases);
            }
        }
    }

    private DecoderOrtStepResult[] ExecuteDecodeCohort(
        InferenceSession session,
        IReadOnlyList<DecodeItem> items,
        IReadOnlyList<DecoderOrtState?> priorStates,
        IReadOnlyList<int> cohort,
        CancellationToken cancellationToken)
    {
        if (cohort.Count == 0)
        {
            return Array.Empty<DecoderOrtStepResult>();
        }

        var pastSequenceLength = items[cohort[0]].Position;
        const int sequenceLength = 1;
        var execution = BuildCudaCohortExecution(
            priorStates,
            cohort,
            pastSequenceLength,
            Geometry.NumHiddenLayers);
        var scratchLeases = new List<IDisposable>(4);
        var handedOff = false;

        try
        {
            var batchSize = cohort.Count;
            var inputIdsLease = RentLongScratch(batchSize, scratchLeases);
            var positionIdsLease = RentLongScratch(batchSize, scratchLeases);

            for (var row = 0; row < batchSize; row++)
            {
                var itemIndex = execution.ItemIndicesByRow[row];
                var item = items[itemIndex];
                var priorState = priorStates[itemIndex]
                    ?? throw new InvalidOperationException("Decode cohort is missing a prior decoder state.");

                if (item.Position != pastSequenceLength)
                {
                    throw new InvalidOperationException(
                        "Decode cohort contains more than one past sequence length.");
                }

                inputIdsLease.Span[row] = ValidateDecodeState(item, priorState);
                positionIdsLease.Span[row] = item.Position;
            }

            handedOff = true;
            return ExecuteCohort(
                session,
                priorStates,
                execution,
                pastSequenceLength,
                sequenceLength,
                inputIdsLease.Memory,
                positionIdsLease.Memory,
                scratchLeases,
                useDecodeLogitsHostPool: true,
                cancellationToken);
        }
        finally
        {
            if (!handedOff)
            {
                DisposeLeases(scratchLeases);
            }
        }
    }

    private DecoderOrtStepResult[] ExecuteCohort(
        InferenceSession session,
        IReadOnlyList<DecoderOrtState?> priorStates,
        CohortExecution execution,
        int pastSequenceLength,
        int sequenceLength,
        Memory<long> inputIds,
        Memory<long> positionIds,
        List<IDisposable> scratchLeases,
        bool useDecodeLogitsHostPool,
        CancellationToken cancellationToken)
    {
        var geometry = Geometry;
        var contract = _profile.Contract;
        var batchSize = execution.ItemIndicesByRow.Length;
        var expectedInputLength = checked(batchSize * sequenceLength);
        if (inputIds.Length != expectedInputLength ||
            positionIds.Length != expectedInputLength)
        {
            throw new InvalidOperationException(
                "Cohort input buffers do not match the requested batch/sequence geometry.");
        }

        var inputNames = new List<string>(3 + geometry.NumHiddenLayers * 2);
        var inputValues = new List<OrtValue>(inputNames.Capacity);
        var ownedInputs = new List<OrtValue>(3 + geometry.NumHiddenLayers * 2);
        var outputNames = new List<string>(1 + geometry.NumHiddenLayers * 2);
        var outputValues = new List<OrtValue>(outputNames.Capacity);
        var ownedOutputs = new List<OrtValue>(1);
        var producedStates = new List<DecoderOrtState>(batchSize);
        IDisposable? pastOwner = null;
        OptimumLegacyCudaPresentKvBatch? present = null;

        try
        {
            var inputIdsValue = OrtValue.CreateTensorValueFromMemory(
                OrtMemoryInfo.DefaultInstance,
                inputIds,
                geometry.GetInputIdsShape(batchSize, sequenceLength));
            ownedInputs.Add(inputIdsValue);
            inputNames.Add(contract.InputIds);
            inputValues.Add(inputIdsValue);

            var attentionMaskShape = geometry.GetAttentionMaskShape(
                batchSize,
                pastSequenceLength,
                sequenceLength);
            var attentionMaskLease = RentLongScratch(
                CheckedTensorLength(attentionMaskShape),
                scratchLeases);
            attentionMaskLease.Span.Fill(1L);
            var attentionMaskValue = OrtValue.CreateTensorValueFromMemory(
                OrtMemoryInfo.DefaultInstance,
                attentionMaskLease.Memory,
                attentionMaskShape);
            ownedInputs.Add(attentionMaskValue);
            inputNames.Add(contract.AttentionMask!);
            inputValues.Add(attentionMaskValue);

            var positionIdsValue = OrtValue.CreateTensorValueFromMemory(
                OrtMemoryInfo.DefaultInstance,
                positionIds,
                geometry.GetPositionIdsShape(batchSize, sequenceLength));
            ownedInputs.Add(positionIdsValue);
            inputNames.Add(contract.PositionIds!);
            inputValues.Add(positionIdsValue);

            if (pastSequenceLength == 0)
            {
                var batchedPastShape = geometry.GetPastKvShape(
                    batchSize,
                    pastSequenceLength);
                for (var layer = 0; layer < geometry.NumHiddenLayers; layer++)
                {
                    inputNames.Add(DecoderOnlyOnnxContract.ExpandLayerName(
                        contract.PastKeyNames!,
                        layer));
                    inputNames.Add(DecoderOnlyOnnxContract.ExpandLayerName(
                        contract.PastValueNames!,
                        layer));

                    var key = OrtValue.CreateTensorValueFromMemory(
                        Array.Empty<float>(),
                        batchedPastShape);
                    var value = OrtValue.CreateTensorValueFromMemory(
                        Array.Empty<float>(),
                        batchedPastShape);
                    ownedInputs.Add(key);
                    ownedInputs.Add(value);
                    inputValues.Add(key);
                    inputValues.Add(value);
                }
            }
            else
            {
                var orderedStates = new DecoderOrtState[batchSize];
                for (var row = 0; row < batchSize; row++)
                {
                    orderedStates[row] = priorStates[execution.ItemIndicesByRow[row]]
                        ?? throw new InvalidOperationException(
                            "A non-empty CUDA past cohort is missing a prior decoder state.");
                }

                if (batchSize == 1)
                {
                    var singleton = new SingletonCudaPastKvBatch(
                        _profile,
                        orderedStates[0],
                        CudaResidentStateFormatId,
                        DeviceId);
                    singleton.AppendInputs(inputNames, inputValues);
                    pastOwner = singleton;
                }
                else
                {
                    var dense = new OptimumLegacyCudaPastKvBatch(
                        _profile,
                        orderedStates);
                    if (dense.DeviceId != DeviceId)
                    {
                        dense.Dispose();
                        throw new InvalidOperationException(
                            $"CUDA past-KV cohort belongs to device {dense.DeviceId}, but this binding targets device {DeviceId}.");
                    }

                    dense.AppendInputs(inputNames, inputValues);
                    pastOwner = dense;
                }

                Interlocked.Increment(ref _cudaPastReuseCount);
            }

            var logitsShape = geometry.GetLogitsShape(batchSize, sequenceLength);
            var logitsLength = CheckedTensorLength(logitsShape);
            Memory<float> logitsMemory;
            if (useDecodeLogitsHostPool && _decodeLogitsHostPool is not null)
            {
                var logitsLease = _decodeLogitsHostPool.Rent(logitsLength);
                scratchLeases.Add(logitsLease);
                logitsMemory = logitsLease.Memory;
                Interlocked.Increment(ref _pageLockedDecodeLogitsRentCount);
            }
            else
            {
                var logitsLease = RentFloatScratch(
                    logitsLength,
                    scratchLeases);
                logitsMemory = logitsLease.Memory;
            }

            var logitsValue = OrtValue.CreateTensorValueFromMemory(
                OrtMemoryInfo.DefaultInstance,
                logitsMemory,
                logitsShape);
            ownedOutputs.Add(logitsValue);
            outputNames.Add(contract.Logits);
            outputValues.Add(logitsValue);

            present = new OptimumLegacyCudaPresentKvBatch(
                _profile,
                batchSize,
                pastSequenceLength,
                sequenceLength,
                _allocator);
            present.AppendOutputs(outputNames, outputValues);

            cancellationToken.ThrowIfCancellationRequested();
            using (var runOptions = new RunOptions())
            {
                _runInvoker(
                    session,
                    runOptions,
                    inputNames,
                    inputValues,
                    outputNames,
                    outputValues);
            }
            Interlocked.Increment(ref _ortRunCount);

            var perSequenceLogitsLength = CheckedTensorLength(
                geometry.GetLogitsShape(batchSize: 1, sequenceLength));
            var results = new DecoderOrtStepResult[batchSize];
            for (var row = 0; row < batchSize; row++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var logitsOffset = checked(row * perSequenceLogitsLength);
                var tokenId = OptimumLegacyFloatDecoderBinding.GreedySampleLastPosition(
                    logitsMemory.Span.Slice(logitsOffset, perSequenceLogitsLength),
                    sequenceLength,
                    geometry.VocabularySize);
                var state = present.CreateRowState(row, tokenId);
                producedStates.Add(state);
                results[execution.ResultIndicesByRow[row]] = new DecoderOrtStepResult(
                    tokenId,
                    state,
                    _eosTokenIds.Contains(tokenId));
            }

            return results;
        }
        catch
        {
            for (var index = producedStates.Count - 1; index >= 0; index--)
            {
                producedStates[index].Dispose();
            }

            throw;
        }
        finally
        {
            present?.Dispose();
            pastOwner?.Dispose();

            for (var index = ownedOutputs.Count - 1; index >= 0; index--)
            {
                ownedOutputs[index].Dispose();
            }

            for (var index = ownedInputs.Count - 1; index >= 0; index--)
            {
                ownedInputs[index].Dispose();
            }

            DisposeLeases(scratchLeases);
        }
    }

    private int ValidatePrefillState(
        PrefillItem item,
        DecoderOrtState? priorState)
    {
        if (item.Tokens.IsEmpty)
        {
            throw new InvalidOperationException(
                "Decoder prefill requires at least one prompt token.");
        }

        ValidatePromptTokens(item.Tokens.Span);
        var startPosition = item.Position ?? priorState?.Position ?? 0;
        if (startPosition < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(item),
                "Decoder prefill position cannot be negative.");
        }

        if (priorState is null)
        {
            if (startPosition != 0)
            {
                throw new InvalidOperationException(
                    $"Initial decoder prefill must start at position 0, not {startPosition}.");
            }

            return startPosition;
        }

        ValidatePriorState(priorState, startPosition, "Prefill continuation");
        return startPosition;
    }

    private int ValidateDecodeState(
        DecodeItem item,
        DecoderOrtState priorState)
    {
        ValidatePriorState(priorState, item.Position, "Decode");
        if (priorState.NextTokenId is not { } nextInputToken)
        {
            throw new InvalidOperationException(
                "Decoder state is missing its next-token frontier.");
        }

        return nextInputToken;
    }

    private void ValidatePriorState(
        DecoderOrtState priorState,
        int expectedPosition,
        string operation)
    {
        ArgumentNullException.ThrowIfNull(priorState);
        if (priorState.IsDisposed)
        {
            throw new ObjectDisposedException(
                nameof(priorState),
                $"{operation} cannot consume a disposed decoder state.");
        }

        if (priorState.LayerCount != Geometry.NumHiddenLayers)
        {
            throw new InvalidOperationException(
                $"Decoder state has {priorState.LayerCount} KV layers; geometry requires {Geometry.NumHiddenLayers}.");
        }

        if (priorState.Position != expectedPosition)
        {
            throw new InvalidOperationException(
                $"Decoder state position {priorState.Position} does not match requested position {expectedPosition}.");
        }
    }

    private static CohortExecution BuildCudaCohortExecution(
        IReadOnlyList<DecoderOrtState?> priorStates,
        IReadOnlyList<int> cohort,
        int pastSequenceLength,
        int layerCount)
    {
        var identity = CreateIdentityCohortExecution(cohort);
        if (pastSequenceLength == 0 || cohort.Count == 1)
        {
            return identity;
        }

        var firstState = priorStates[cohort[0]];
        if (firstState is null ||
            !firstState.TryGetCudaCohortSlice(out var firstSlice))
        {
            throw CreateUnsupportedCudaCohortException();
        }

        var arena = firstSlice.Arena;
        if (arena.Position != pastSequenceLength ||
            arena.BatchSize != cohort.Count ||
            arena.LayerCount != layerCount)
        {
            throw CreateUnsupportedCudaCohortException();
        }

        var itemsByRow = new int[cohort.Count];
        var resultsByRow = new int[cohort.Count];
        Array.Fill(itemsByRow, -1);

        for (var cohortIndex = 0; cohortIndex < cohort.Count; cohortIndex++)
        {
            var itemIndex = cohort[cohortIndex];
            var state = priorStates[itemIndex];
            if (state is null ||
                !state.TryGetCudaCohortSlice(out var slice) ||
                !ReferenceEquals(slice.Arena, arena) ||
                slice.Row < 0 ||
                slice.Row >= cohort.Count ||
                itemsByRow[slice.Row] != -1)
            {
                throw CreateUnsupportedCudaCohortException();
            }

            itemsByRow[slice.Row] = itemIndex;
            resultsByRow[slice.Row] = cohortIndex;
        }

        if (itemsByRow.Any(static item => item < 0))
        {
            throw CreateUnsupportedCudaCohortException();
        }

        return new CohortExecution(itemsByRow, resultsByRow);
    }

    private static NotSupportedException CreateUnsupportedCudaCohortException() =>
        new(
            "CUDA Optimum execution currently requires a singleton resident state or the complete dense rows of one CUDA cohort arena. " +
            "Subset, duplicate-row, mixed-arena and device-gather layouts are not silently host-packed.");

    private static CohortExecution CreateIdentityCohortExecution(
        IReadOnlyList<int> cohort)
    {
        var itemIndices = new int[cohort.Count];
        var resultIndices = new int[cohort.Count];
        for (var index = 0; index < cohort.Count; index++)
        {
            itemIndices[index] = cohort[index];
            resultIndices[index] = index;
        }

        return new CohortExecution(itemIndices, resultIndices);
    }

    private PooledArrayLease<float> RentFloatScratch(
        int length,
        List<IDisposable> leases)
    {
        var lease = new PooledArrayLease<float>(_scratchFloatPool, length);
        leases.Add(lease);
        Interlocked.Increment(ref _scratchFloatRentCount);
        return lease;
    }

    private PooledArrayLease<long> RentLongScratch(
        int length,
        List<IDisposable> leases)
    {
        var lease = new PooledArrayLease<long>(_scratchLongPool, length);
        leases.Add(lease);
        Interlocked.Increment(ref _scratchLongRentCount);
        return lease;
    }

    private static void DisposeLeases(List<IDisposable> leases)
    {
        for (var index = leases.Count - 1; index >= 0; index--)
        {
            leases[index].Dispose();
        }

        leases.Clear();
    }

    private static void ValidatePromptTokens(ReadOnlySpan<int> tokens)
    {
        if (tokens.Length == 0)
        {
            throw new InvalidOperationException(
                "Decoder prefill requires at least one prompt token.");
        }

        foreach (var token in tokens)
        {
            if (token < 0)
            {
                throw new InvalidOperationException(
                    "Decoder input token ids cannot be negative.");
            }
        }
    }

    private static void CopyPromptTokens(
        ReadOnlySpan<int> source,
        Span<long> destination)
    {
        if (source.Length != destination.Length)
        {
            throw new ArgumentException(
                "Prompt token source and destination lengths must match.");
        }

        for (var index = 0; index < source.Length; index++)
        {
            destination[index] = source[index];
        }
    }

    private static int CheckedTensorLength(IReadOnlyList<long> shape)
    {
        long count = 1;
        foreach (var dimension in shape)
        {
            if (dimension < 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(shape),
                    "Runtime tensor shapes cannot contain negative dimensions.");
            }

            count = checked(count * dimension);
        }

        if (count > int.MaxValue)
        {
            throw new NotSupportedException(
                $"Managed tensor contains {count} elements, exceeding the current array-backed scratch allocator limit.");
        }

        return checked((int)count);
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        _decodeLogitsHostPool?.Dispose();
    }

    private readonly record struct CohortExecution(
        int[] ItemIndicesByRow,
        int[] ResultIndicesByRow);

    private sealed class SingletonCudaPastKvBatch : IDisposable
    {
        private readonly DecoderOrtCudaResidentStateLease _lease;
        private readonly OrtMemoryInfo _memoryInfo;
        private readonly OrtValue[] _inputValues;
        private readonly ReadOnlyCollection<string> _inputNamesView;
        private int _disposed;

        public SingletonCudaPastKvBatch(
            OptimumLegacyDecoderProfile profile,
            DecoderOrtState state,
            string formatId,
            int expectedDeviceId)
        {
            ArgumentNullException.ThrowIfNull(profile);
            ArgumentNullException.ThrowIfNull(state);
            ArgumentException.ThrowIfNullOrWhiteSpace(formatId);
            ArgumentOutOfRangeException.ThrowIfNegative(expectedDeviceId);

            _lease = state.AcquireCudaResidentState(formatId);
            OrtMemoryInfo? memoryInfo = null;
            var produced = 0;
            try
            {
                if (_lease.DeviceId != expectedDeviceId)
                {
                    throw new InvalidOperationException(
                        $"CUDA singleton past state belongs to device {_lease.DeviceId}, but execution targets device {expectedDeviceId}.");
                }

                var geometry = profile.Geometry;
                if (_lease.LayerCount != geometry.NumHiddenLayers ||
                    _lease.Position != state.Position)
                {
                    throw new InvalidOperationException(
                        "CUDA singleton resident lease does not match the decoder state frontier.");
                }

                var expectedShape = geometry.GetPastKvShape(1, state.Position);
                memoryInfo = new OrtMemoryInfo(
                    "Cuda",
                    OrtAllocatorType.DeviceAllocator,
                    expectedDeviceId,
                    OrtMemType.Default);

                var inputNames = new string[checked(geometry.NumHiddenLayers * 2)];
                _inputValues = new OrtValue[inputNames.Length];
                for (var layer = 0; layer < geometry.NumHiddenLayers; layer++)
                {
                    var resident = _lease.GetLayer(layer);
                    ValidateResidentTensor(resident.Key, expectedShape, expectedDeviceId, layer, "key");
                    ValidateResidentTensor(resident.Value, expectedShape, expectedDeviceId, layer, "value");
                    var offset = checked(layer * 2);
                    inputNames[offset] = DecoderOnlyOnnxContract.ExpandLayerName(
                        profile.Contract.PastKeyNames!,
                        layer);
                    inputNames[offset + 1] = DecoderOnlyOnnxContract.ExpandLayerName(
                        profile.Contract.PastValueNames!,
                        layer);

                    _inputValues[offset] = OrtValue.CreateTensorValueWithData(
                        memoryInfo,
                        TensorElementType.Float,
                        expectedShape,
                        resident.Key.DevicePointer,
                        resident.Key.ByteLength);
                    produced++;
                    _inputValues[offset + 1] = OrtValue.CreateTensorValueWithData(
                        memoryInfo,
                        TensorElementType.Float,
                        expectedShape,
                        resident.Value.DevicePointer,
                        resident.Value.ByteLength);
                    produced++;
                }

                _inputNamesView = Array.AsReadOnly(inputNames);
                _memoryInfo = memoryInfo;
                memoryInfo = null;
            }
            catch
            {
                if (_inputValues is not null)
                {
                    for (var index = produced - 1; index >= 0; index--)
                    {
                        _inputValues[index].Dispose();
                    }
                }

                memoryInfo?.Dispose();
                _lease.Dispose();
                throw;
            }
        }

        public void AppendInputs(
            ICollection<string> inputNames,
            ICollection<OrtValue> inputValues)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            for (var index = 0; index < _inputValues.Length; index++)
            {
                inputNames.Add(_inputNamesView[index]);
                inputValues.Add(_inputValues[index]);
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
            {
                return;
            }

            for (var index = _inputValues.Length - 1; index >= 0; index--)
            {
                _inputValues[index].Dispose();
            }

            _memoryInfo.Dispose();
            _lease.Dispose();
        }

        private static void ValidateResidentTensor(
            CudaDeviceTensorView tensor,
            IReadOnlyList<long> expectedShape,
            int expectedDeviceId,
            int layer,
            string slot)
        {
            if (tensor.DeviceId != expectedDeviceId ||
                tensor.ElementType != TensorElementType.Float ||
                !tensor.Shape.SequenceEqual(expectedShape))
            {
                throw new InvalidOperationException(
                    $"CUDA singleton past layer {layer} {slot} does not match the expected FP32 device tensor geometry.");
            }
        }
    }
}
