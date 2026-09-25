using System.Buffers;
using Fission.Abstractions.Execution;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Concrete causal-LM binding for the Optimum legacy decoder-with-past tensor
/// convention, using CPU FP32 KV/logits buffers and greedy sampling.
/// </summary>
public sealed class OptimumLegacyFloatDecoderBinding :
    IDecoderOrtBatchModelBinding,
    IDecoderOrtChunkedPrefillModelBinding
{
    private readonly OptimumLegacyDecoderProfile _profile;
    private readonly HashSet<int> _eosTokenIds;
    private readonly ArrayPool<float> _cohortBufferPool;
    private readonly ArrayPool<float> _scratchFloatPool;
    private readonly ArrayPool<long> _scratchLongPool;
    private int _ortRunCount;
    private int _pastKvPackCount;
    private int _pastKvArenaReuseCount;
    private long _pastKvCopiedElementCount;
    private long _scratchFloatRentCount;
    private long _scratchLongRentCount;
    private int _disposed;

    public OptimumLegacyFloatDecoderBinding(
        OptimumLegacyDecoderProfile profile,
        IEnumerable<int>? eosTokenIds = null,
        ArrayPool<float>? cohortBufferPool = null,
        ArrayPool<float>? scratchFloatPool = null,
        ArrayPool<long>? scratchLongPool = null)
    {
        ArgumentNullException.ThrowIfNull(profile);
        if (profile.Geometry.KvElementType != TensorElementType.Float)
        {
            throw new ArgumentException(
                "The FP32 decoder binding requires Float KV tensors.",
                nameof(profile));
        }

        if (profile.Contract.LogitsElementType != TensorElementType.Float)
        {
            throw new ArgumentException(
                "The FP32 decoder binding requires Float logits.",
                nameof(profile));
        }

        if (string.IsNullOrWhiteSpace(profile.Contract.AttentionMask) ||
            string.IsNullOrWhiteSpace(profile.Contract.PositionIds) ||
            !profile.Contract.UsesPastKeyValues)
        {
            throw new ArgumentException(
                "The Optimum FP32 binding requires attention_mask, position_ids, and past/present KV tensors.",
                nameof(profile));
        }

        if ((profile.Contract.AdditionalInputs?.Count ?? 0) != 0 ||
            (profile.Contract.AdditionalOutputs?.Count ?? 0) != 0)
        {
            throw new ArgumentException(
                "Additional model-specific tensors are not supported by the Optimum FP32 binding.",
                nameof(profile));
        }

        _profile = profile;
        _eosTokenIds = eosTokenIds is null
            ? new HashSet<int>()
            : new HashSet<int>(eosTokenIds);
        _cohortBufferPool = cohortBufferPool ?? ArrayPool<float>.Create();
        _scratchFloatPool = scratchFloatPool ?? ArrayPool<float>.Create();
        _scratchLongPool = scratchLongPool ?? ArrayPool<long>.Create();

        if (_eosTokenIds.Any(static token => token < 0))
        {
            throw new ArgumentOutOfRangeException(
                nameof(eosTokenIds),
                "EOS token ids cannot be negative.");
        }
    }

    public string Name => "optimum-legacy-fp32-greedy";
    public OnnxSessionContract SessionContract => _profile.SessionContract;
    public DecoderOrtGeometry Geometry => _profile.Geometry;
    public int OrtRunCount => Volatile.Read(ref _ortRunCount);
    public int PastKvPackCount => Volatile.Read(ref _pastKvPackCount);
    public int PastKvArenaReuseCount => Volatile.Read(ref _pastKvArenaReuseCount);
    public long PastKvCopiedElementCount => Interlocked.Read(ref _pastKvCopiedElementCount);
    public long ScratchFloatRentCount => Interlocked.Read(ref _scratchFloatRentCount);
    public long ScratchLongRentCount => Interlocked.Read(ref _scratchLongRentCount);

    public DecoderOrtStepResult ExecutePrefill(
        InferenceSession session,
        PrefillItem item,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(session);
        cancellationToken.ThrowIfCancellationRequested();

        var startPosition = item.Position ?? 0;
        if (startPosition != 0)
        {
            throw new InvalidOperationException(
                $"Initial decoder prefill must start at position 0, not {startPosition}.");
        }

        var inputIds = ConvertPromptTokens(item.Tokens);
        var positions = CreatePositions(startPosition, inputIds.Length);
        return ExecuteStep(
            session,
            inputIds,
            positions,
            priorState: null,
            pastSequenceLength: 0,
            cancellationToken);
    }

    public DecoderOrtStepResult ExecutePrefillChunk(
        InferenceSession session,
        PrefillItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(priorState);
        cancellationToken.ThrowIfCancellationRequested();

        if (priorState.IsDisposed)
        {
            throw new ObjectDisposedException(
                nameof(priorState),
                "Prefill continuation cannot consume a disposed decoder state.");
        }

        if (priorState.LayerCount != Geometry.NumHiddenLayers)
        {
            throw new InvalidOperationException(
                $"Decoder state has {priorState.LayerCount} KV layers; geometry requires {Geometry.NumHiddenLayers}.");
        }

        var startPosition = item.Position ?? priorState.Position;
        if (startPosition != priorState.Position)
        {
            throw new InvalidOperationException(
                $"Decoder prefill continuation position {startPosition} does not match prior state position {priorState.Position}.");
        }

        var inputIds = ConvertPromptTokens(item.Tokens);
        var positions = CreatePositions(startPosition, inputIds.Length);
        return ExecuteStep(
            session,
            inputIds,
            positions,
            priorState,
            startPosition,
            cancellationToken);
    }

    public DecoderOrtStepResult ExecuteDecode(
        InferenceSession session,
        DecodeItem item,
        DecoderOrtState priorState,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(session);
        ArgumentNullException.ThrowIfNull(priorState);
        cancellationToken.ThrowIfCancellationRequested();

        var nextInputToken = ValidateDecodeState(item, priorState);
        return ExecuteStep(
            session,
            new long[] { nextInputToken },
            new long[] { item.Position },
            priorState,
            item.Position,
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
            _ = ValidateDecodeState(items[index], priorStates[index]);
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

    private DecoderOrtStepResult[] ExecuteDecodeCohort(
        InferenceSession session,
        IReadOnlyList<DecodeItem> items,
        IReadOnlyList<DecoderOrtState> priorStates,
        IReadOnlyList<int> cohort,
        CancellationToken cancellationToken)
    {
        if (cohort.Count == 0)
        {
            return Array.Empty<DecoderOrtStepResult>();
        }

        var geometry = Geometry;
        var contract = _profile.Contract;
        var batchSize = cohort.Count;
        const int sequenceLength = 1;
        var pastSequenceLength = items[cohort[0]].Position;
        var execution = BuildCohortExecution(
            priorStates,
            cohort,
            pastSequenceLength,
            geometry.NumHiddenLayers);
        var inputNames = new List<string>(3 + geometry.NumHiddenLayers * 2);
        var inputValues = new List<OrtValue>(inputNames.Capacity);
        var ownedInputs = new List<OrtValue>();
        var outputNames = new List<string>(1 + geometry.NumHiddenLayers * 2);
        var outputValues = new List<OrtValue>(outputNames.Capacity);
        var ownedOutputs = new List<OrtValue>(outputNames.Capacity);
        var scratchLeases = new List<IDisposable>(4 + geometry.NumHiddenLayers * 2);
        var producedStates = new List<DecoderOrtState>(batchSize);
        DecoderOrtCohortArena? retainedInputArena = null;
        DecoderOrtCohortArena? nextArena = null;

        try
        {
            var inputIdsLease = RentLongScratch(batchSize, scratchLeases);
            var positionIdsLease = RentLongScratch(batchSize, scratchLeases);
            var inputIds = inputIdsLease.Span;
            var positionIds = positionIdsLease.Span;
            for (var row = 0; row < batchSize; row++)
            {
                var itemIndex = execution.ItemIndicesByRow[row];
                var item = items[itemIndex];
                var priorState = priorStates[itemIndex];
                if (item.Position != pastSequenceLength)
                {
                    throw new InvalidOperationException(
                        "Decode cohort contains more than one past sequence length.");
                }

                inputIds[row] = ValidateDecodeState(item, priorState);
                positionIds[row] = item.Position;
            }

            var inputIdsShape = geometry.GetInputIdsShape(batchSize, sequenceLength);
            var inputIdsValue = OrtValue.CreateTensorValueFromMemory(
                OrtMemoryInfo.DefaultInstance,
                inputIdsLease.Memory,
                inputIdsShape);
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
                positionIdsLease.Memory,
                geometry.GetPositionIdsShape(batchSize, sequenceLength));
            ownedInputs.Add(positionIdsValue);
            inputNames.Add(contract.PositionIds!);
            inputValues.Add(positionIdsValue);

            var perSequencePastShape = geometry.GetPastKvShape(
                batchSize: 1,
                pastSequenceLength);
            var perSequencePastLength = CheckedTensorLength(perSequencePastShape);
            var batchedPastShape = geometry.GetPastKvShape(
                batchSize,
                pastSequenceLength);
            var batchedPastLength = CheckedTensorLength(batchedPastShape);

            if (execution.ReusableArena is { } reusableArena)
            {
                reusableArena.Retain();
                retainedInputArena = reusableArena;

                if (reusableArena.LogicalBufferLength != batchedPastLength)
                {
                    throw new InvalidOperationException(
                        "Reusable cohort arena does not match the configured decoder geometry.");
                }

                for (var layer = 0; layer < geometry.NumHiddenLayers; layer++)
                {
                    inputNames.Add(DecoderOnlyOnnxContract.ExpandLayerName(
                        contract.PastKeyNames!,
                        layer));
                    inputNames.Add(DecoderOnlyOnnxContract.ExpandLayerName(
                        contract.PastValueNames!,
                        layer));

                    var keyValue = OrtValue.CreateTensorValueFromMemory(
                        OrtMemoryInfo.DefaultInstance,
                        reusableArena.GetKeyMemory(layer),
                        batchedPastShape);
                    var valueValue = OrtValue.CreateTensorValueFromMemory(
                        OrtMemoryInfo.DefaultInstance,
                        reusableArena.GetValueMemory(layer),
                        batchedPastShape);
                    ownedInputs.Add(keyValue);
                    ownedInputs.Add(valueValue);
                    inputValues.Add(keyValue);
                    inputValues.Add(valueValue);
                }

                Interlocked.Increment(ref _pastKvArenaReuseCount);
            }
            else
            {
                for (var layer = 0; layer < geometry.NumHiddenLayers; layer++)
                {
                    inputNames.Add(DecoderOnlyOnnxContract.ExpandLayerName(
                        contract.PastKeyNames!,
                        layer));
                    inputNames.Add(DecoderOnlyOnnxContract.ExpandLayerName(
                        contract.PastValueNames!,
                        layer));

                    var keyLease = RentFloatScratch(batchedPastLength, scratchLeases);
                    var valueLease = RentFloatScratch(batchedPastLength, scratchLeases);

                    for (var row = 0; row < batchSize; row++)
                    {
                        var priorLayer = priorStates[execution.ItemIndicesByRow[row]].GetLayer(layer);
                        var priorKey = priorLayer.Key.GetTensorDataAsSpan<float>();
                        var priorValue = priorLayer.Value.GetTensorDataAsSpan<float>();
                        if (priorKey.Length != perSequencePastLength ||
                            priorValue.Length != perSequencePastLength)
                        {
                            throw new InvalidOperationException(
                                $"Decoder state layer {layer} has a KV payload size that does not match " +
                                $"position {pastSequenceLength} and the configured geometry.");
                        }

                        var offset = checked(row * perSequencePastLength);
                        priorKey.CopyTo(keyLease.Span.Slice(offset, perSequencePastLength));
                        priorValue.CopyTo(valueLease.Span.Slice(offset, perSequencePastLength));
                    }

                    var keyValue = OrtValue.CreateTensorValueFromMemory(
                        OrtMemoryInfo.DefaultInstance,
                        keyLease.Memory,
                        batchedPastShape);
                    var valueValue = OrtValue.CreateTensorValueFromMemory(
                        OrtMemoryInfo.DefaultInstance,
                        valueLease.Memory,
                        batchedPastShape);
                    ownedInputs.Add(keyValue);
                    ownedInputs.Add(valueValue);
                    inputValues.Add(keyValue);
                    inputValues.Add(valueValue);
                }

                Interlocked.Increment(ref _pastKvPackCount);
                Interlocked.Add(
                    ref _pastKvCopiedElementCount,
                    checked((long)batchSize * perSequencePastLength * geometry.NumHiddenLayers * 2L));
            }

            var logitsShape = geometry.GetLogitsShape(batchSize, sequenceLength);
            var logitsLease = RentFloatScratch(
                CheckedTensorLength(logitsShape),
                scratchLeases);
            var logitsValue = OrtValue.CreateTensorValueFromMemory(
                OrtMemoryInfo.DefaultInstance,
                logitsLease.Memory,
                logitsShape);
            ownedOutputs.Add(logitsValue);
            outputNames.Add(contract.Logits);
            outputValues.Add(logitsValue);

            var nextPosition = checked(pastSequenceLength + sequenceLength);
            var presentShape = geometry.GetPresentKvShape(
                batchSize,
                pastSequenceLength,
                sequenceLength);
            var presentLength = CheckedTensorLength(presentShape);
            nextArena = new DecoderOrtCohortArena(
                nextPosition,
                batchSize,
                geometry.NumHiddenLayers,
                presentLength,
                _cohortBufferPool);

            for (var layer = 0; layer < geometry.NumHiddenLayers; layer++)
            {
                var key = OrtValue.CreateTensorValueFromMemory(
                    OrtMemoryInfo.DefaultInstance,
                    nextArena.GetKeyMemory(layer),
                    presentShape);
                var value = OrtValue.CreateTensorValueFromMemory(
                    OrtMemoryInfo.DefaultInstance,
                    nextArena.GetValueMemory(layer),
                    presentShape);
                ownedOutputs.Add(key);
                ownedOutputs.Add(value);
                outputNames.Add(DecoderOnlyOnnxContract.ExpandLayerName(
                    contract.PresentKeyNames!,
                    layer));
                outputNames.Add(DecoderOnlyOnnxContract.ExpandLayerName(
                    contract.PresentValueNames!,
                    layer));
                outputValues.Add(key);
                outputValues.Add(value);
            }

            cancellationToken.ThrowIfCancellationRequested();
            using (var runOptions = new RunOptions())
            {
                CallerOwnedOrtRun.Execute(
                    session,
                    runOptions,
                    inputNames,
                    inputValues,
                    outputNames,
                    outputValues);
            }
            Interlocked.Increment(ref _ortRunCount);

            var perSequencePresentShape = geometry.GetPresentKvShape(
                batchSize: 1,
                pastSequenceLength,
                sequenceLength);
            var perSequencePresentLength = CheckedTensorLength(perSequencePresentShape);
            var perSequenceLogitsLength = CheckedTensorLength(
                geometry.GetLogitsShape(batchSize: 1, sequenceLength));
            var results = new DecoderOrtStepResult[batchSize];

            for (var row = 0; row < batchSize; row++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var rowOwnedValues = new List<OrtValue>(geometry.NumHiddenLayers * 2);
                try
                {
                    var nextLayers = new DecoderOrtLayerState[geometry.NumHiddenLayers];
                    for (var layer = 0; layer < geometry.NumHiddenLayers; layer++)
                    {
                        var offset = checked(row * perSequencePresentLength);
                        var key = CreateTensorSlice(
                            nextArena.GetKeyMemory(layer),
                            offset,
                            perSequencePresentLength,
                            perSequencePresentShape);
                        var value = CreateTensorSlice(
                            nextArena.GetValueMemory(layer),
                            offset,
                            perSequencePresentLength,
                            perSequencePresentShape);
                        rowOwnedValues.Add(key);
                        rowOwnedValues.Add(value);
                        nextLayers[layer] = new DecoderOrtLayerState(key, value);
                    }

                    var logitsOffset = checked(row * perSequenceLogitsLength);
                    var tokenId = GreedySampleLastPosition(
                        logitsLease.Span.Slice(logitsOffset, perSequenceLogitsLength),
                        sequenceLength,
                        geometry.VocabularySize);
                    var state = new DecoderOrtState(
                        nextPosition,
                        nextLayers,
                        tokenId,
                        new DecoderOrtCohortSlice(nextArena, row));
                    rowOwnedValues.Clear();
                    producedStates.Add(state);
                    results[execution.ResultIndicesByRow[row]] = new DecoderOrtStepResult(
                        tokenId,
                        state,
                        _eosTokenIds.Contains(tokenId));
                }
                catch
                {
                    for (var index = rowOwnedValues.Count - 1; index >= 0; index--)
                    {
                        rowOwnedValues[index].Dispose();
                    }

                    throw;
                }
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
            for (var index = ownedOutputs.Count - 1; index >= 0; index--)
            {
                ownedOutputs[index].Dispose();
            }

            for (var index = ownedInputs.Count - 1; index >= 0; index--)
            {
                ownedInputs[index].Dispose();
            }

            for (var index = scratchLeases.Count - 1; index >= 0; index--)
            {
                scratchLeases[index].Dispose();
            }

            retainedInputArena?.Release();
            nextArena?.Release();
        }
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

    private static CohortExecution BuildCohortExecution(
        IReadOnlyList<DecoderOrtState> priorStates,
        IReadOnlyList<int> cohort,
        int pastSequenceLength,
        int layerCount)
    {
        var batchSize = cohort.Count;
        var identityItems = new int[batchSize];
        var identityResults = new int[batchSize];
        for (var index = 0; index < batchSize; index++)
        {
            identityItems[index] = cohort[index];
            identityResults[index] = index;
        }

        if (!priorStates[cohort[0]].TryGetCohortSlice(out var firstSlice))
        {
            return new CohortExecution(identityItems, identityResults, ReusableArena: null);
        }

        var arena = firstSlice.Arena;
        if (arena.Position != pastSequenceLength ||
            arena.BatchSize != batchSize ||
            arena.LayerCount != layerCount)
        {
            return new CohortExecution(identityItems, identityResults, ReusableArena: null);
        }

        var itemsByRow = new int[batchSize];
        var resultsByRow = new int[batchSize];
        Array.Fill(itemsByRow, -1);

        for (var cohortIndex = 0; cohortIndex < batchSize; cohortIndex++)
        {
            var itemIndex = cohort[cohortIndex];
            if (!priorStates[itemIndex].TryGetCohortSlice(out var slice) ||
                !ReferenceEquals(slice.Arena, arena) ||
                slice.Row < 0 ||
                slice.Row >= batchSize ||
                itemsByRow[slice.Row] != -1)
            {
                return new CohortExecution(identityItems, identityResults, ReusableArena: null);
            }

            itemsByRow[slice.Row] = itemIndex;
            resultsByRow[slice.Row] = cohortIndex;
        }

        for (var row = 0; row < batchSize; row++)
        {
            if (itemsByRow[row] == -1)
            {
                return new CohortExecution(identityItems, identityResults, ReusableArena: null);
            }
        }

        return new CohortExecution(itemsByRow, resultsByRow, arena);
    }

    private static OrtValue CreateTensorSlice(
        Memory<float> buffer,
        int offset,
        int length,
        long[] shape)
    {
        return OrtValue.CreateTensorValueFromMemory(
            OrtMemoryInfo.DefaultInstance,
            buffer.Slice(offset, length),
            shape);
    }

    private int ValidateDecodeState(
        DecodeItem item,
        DecoderOrtState priorState)
    {
        if (priorState.IsDisposed)
        {
            throw new ObjectDisposedException(
                nameof(priorState),
                "Decode cannot consume a disposed decoder state.");
        }

        if (priorState.LayerCount != Geometry.NumHiddenLayers)
        {
            throw new InvalidOperationException(
                $"Decoder state has {priorState.LayerCount} KV layers; geometry requires {Geometry.NumHiddenLayers}.");
        }

        if (priorState.Position != item.Position)
        {
            throw new InvalidOperationException(
                $"Decoder state position {priorState.Position} does not match requested position {item.Position}.");
        }

        if (priorState.NextTokenId is not { } nextInputToken)
        {
            throw new InvalidOperationException(
                "Decoder state is missing its next-token frontier.");
        }

        return nextInputToken;
    }

    private static long[] ConvertPromptTokens(ReadOnlyMemory<int> tokens)
    {
        if (tokens.Length == 0)
        {
            throw new InvalidOperationException(
                "Decoder prefill requires at least one prompt token.");
        }

        var inputIds = new long[tokens.Length];
        var source = tokens.Span;
        for (var index = 0; index < source.Length; index++)
        {
            if (source[index] < 0)
            {
                throw new InvalidOperationException(
                    "Decoder input token ids cannot be negative.");
            }

            inputIds[index] = source[index];
        }

        return inputIds;
    }

    private static long[] CreatePositions(int startPosition, int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(startPosition);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);

        var positions = new long[count];
        for (var index = 0; index < positions.Length; index++)
        {
            positions[index] = checked(startPosition + index);
        }

        return positions;
    }

    private DecoderOrtStepResult ExecuteStep(
        InferenceSession session,
        long[] inputIds,
        long[] positionIds,
        DecoderOrtState? priorState,
        int pastSequenceLength,
        CancellationToken cancellationToken)
    {
        var sequenceLength = inputIds.Length;
        var geometry = Geometry;
        var contract = _profile.Contract;
        var inputNames = new List<string>(3 + geometry.NumHiddenLayers * 2);
        var inputValues = new List<OrtValue>(inputNames.Capacity);
        var ownedInputs = new List<OrtValue>();
        var outputNames = new List<string>(1 + geometry.NumHiddenLayers * 2);
        var outputValues = new List<OrtValue>(outputNames.Capacity);
        var ownedOutputs = new List<OrtValue>(outputNames.Capacity);
        var stateOwnsKv = false;

        try
        {
            var inputIdsValue = OrtValue.CreateTensorValueFromMemory(
                inputIds,
                geometry.GetInputIdsShape(batchSize: 1, sequenceLength));
            ownedInputs.Add(inputIdsValue);
            inputNames.Add(contract.InputIds);
            inputValues.Add(inputIdsValue);

            var attentionMask = new long[checked(pastSequenceLength + sequenceLength)];
            Array.Fill(attentionMask, 1L);
            var attentionMaskValue = OrtValue.CreateTensorValueFromMemory(
                attentionMask,
                geometry.GetAttentionMaskShape(
                    batchSize: 1,
                    pastSequenceLength,
                    sequenceLength));
            ownedInputs.Add(attentionMaskValue);
            inputNames.Add(contract.AttentionMask!);
            inputValues.Add(attentionMaskValue);

            var positionIdsValue = OrtValue.CreateTensorValueFromMemory(
                positionIds,
                geometry.GetPositionIdsShape(batchSize: 1, sequenceLength));
            ownedInputs.Add(positionIdsValue);
            inputNames.Add(contract.PositionIds!);
            inputValues.Add(positionIdsValue);

            for (var layer = 0; layer < geometry.NumHiddenLayers; layer++)
            {
                inputNames.Add(DecoderOnlyOnnxContract.ExpandLayerName(
                    contract.PastKeyNames!,
                    layer));
                inputNames.Add(DecoderOnlyOnnxContract.ExpandLayerName(
                    contract.PastValueNames!,
                    layer));

                if (priorState is null)
                {
                    var emptyShape = geometry.GetPastKvShape(
                        batchSize: 1,
                        pastSequenceLength: 0);
                    var key = OrtValue.CreateTensorValueFromMemory(
                        Array.Empty<float>(),
                        emptyShape);
                    var value = OrtValue.CreateTensorValueFromMemory(
                        Array.Empty<float>(),
                        emptyShape);
                    ownedInputs.Add(key);
                    ownedInputs.Add(value);
                    inputValues.Add(key);
                    inputValues.Add(value);
                }
                else
                {
                    var priorLayer = priorState.GetLayer(layer);
                    inputValues.Add(priorLayer.Key);
                    inputValues.Add(priorLayer.Value);
                }
            }

            var logitsShape = geometry.GetLogitsShape(batchSize: 1, sequenceLength);
            var logits = new float[CheckedTensorLength(logitsShape)];
            var logitsValue = OrtValue.CreateTensorValueFromMemory(
                logits,
                logitsShape);
            ownedOutputs.Add(logitsValue);
            outputNames.Add(contract.Logits);
            outputValues.Add(logitsValue);

            var nextLayers = new DecoderOrtLayerState[geometry.NumHiddenLayers];
            var presentShape = geometry.GetPresentKvShape(
                batchSize: 1,
                pastSequenceLength,
                sequenceLength);
            var presentLength = CheckedTensorLength(presentShape);

            for (var layer = 0; layer < geometry.NumHiddenLayers; layer++)
            {
                var keyBuffer = new float[presentLength];
                var valueBuffer = new float[presentLength];
                var key = OrtValue.CreateTensorValueFromMemory(
                    keyBuffer,
                    presentShape);
                var value = OrtValue.CreateTensorValueFromMemory(
                    valueBuffer,
                    presentShape);
                ownedOutputs.Add(key);
                ownedOutputs.Add(value);
                outputNames.Add(DecoderOnlyOnnxContract.ExpandLayerName(
                    contract.PresentKeyNames!,
                    layer));
                outputNames.Add(DecoderOnlyOnnxContract.ExpandLayerName(
                    contract.PresentValueNames!,
                    layer));
                outputValues.Add(key);
                outputValues.Add(value);
                nextLayers[layer] = new DecoderOrtLayerState(key, value);
            }

            cancellationToken.ThrowIfCancellationRequested();
            using (var runOptions = new RunOptions())
            {
                CallerOwnedOrtRun.Execute(
                    session,
                    runOptions,
                    inputNames,
                    inputValues,
                    outputNames,
                    outputValues);
            }
            Interlocked.Increment(ref _ortRunCount);

            var tokenId = GreedySampleLastPosition(
                logits,
                sequenceLength,
                geometry.VocabularySize);
            var state = new DecoderOrtState(
                checked(pastSequenceLength + sequenceLength),
                nextLayers,
                nextTokenId: tokenId);
            stateOwnsKv = true;

            logitsValue.Dispose();
            ownedOutputs.Remove(logitsValue);
            ownedOutputs.Clear();

            return new DecoderOrtStepResult(
                tokenId,
                state,
                _eosTokenIds.Contains(tokenId));
        }
        catch
        {
            if (!stateOwnsKv)
            {
                for (var index = ownedOutputs.Count - 1; index >= 0; index--)
                {
                    ownedOutputs[index].Dispose();
                }
            }

            throw;
        }
        finally
        {
            for (var index = ownedInputs.Count - 1; index >= 0; index--)
            {
                ownedInputs[index].Dispose();
            }
        }
    }

    public static int GreedySampleLastPosition(
        ReadOnlySpan<float> logits,
        int sequenceLength,
        int vocabularySize)
    {
        if (sequenceLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sequenceLength));
        }

        if (vocabularySize <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        }

        var expected = checked(sequenceLength * vocabularySize);
        if (logits.Length != expected)
        {
            throw new ArgumentException(
                $"Logit buffer contains {logits.Length} values; expected {expected}.",
                nameof(logits));
        }

        var offset = checked((sequenceLength - 1) * vocabularySize);
        var bestToken = 0;
        var bestLogit = logits[offset];
        for (var token = 1; token < vocabularySize; token++)
        {
            var candidate = logits[offset + token];
            if (candidate > bestLogit)
            {
                bestLogit = candidate;
                bestToken = token;
            }
        }

        return bestToken;
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
                $"Managed FP32 tensor contains {count} elements, exceeding the current array-backed allocator limit.");
        }

        return checked((int)count);
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        Interlocked.Exchange(ref _disposed, 1);
    }

    private readonly record struct CohortExecution(
        int[] ItemIndicesByRow,
        int[] ResultIndicesByRow,
        DecoderOrtCohortArena? ReusableArena);
}
