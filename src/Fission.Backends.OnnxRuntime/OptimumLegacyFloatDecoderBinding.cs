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
    IDecoderOrtBatchPrefillModelBinding,
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
        var execution = BuildCohortExecution(
            priorStates,
            cohort,
            pastSequenceLength,
            Geometry.NumHiddenLayers);
        var scratchLeases = new List<IDisposable>(4 + Geometry.NumHiddenLayers * 2);
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
        var execution = BuildCohortExecution(
            priorStates,
            cohort,
            pastSequenceLength,
            Geometry.NumHiddenLayers);
        var scratchLeases = new List<IDisposable>(4 + Geometry.NumHiddenLayers * 2);
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
        var ownedInputs = new List<OrtValue>();
        var outputNames = new List<string>(1 + geometry.NumHiddenLayers * 2);
        var outputValues = new List<OrtValue>(outputNames.Capacity);
        var ownedOutputs = new List<OrtValue>(outputNames.Capacity);
        var producedStates = new List<DecoderOrtState>(batchSize);
        DecoderOrtCohortArena? retainedInputArena = null;
        DecoderOrtCohortArena? nextArena = null;

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

            var batchedPastShape = geometry.GetPastKvShape(
                batchSize,
                pastSequenceLength);
            var batchedPastLength = CheckedTensorLength(batchedPastShape);

            if (pastSequenceLength == 0)
            {
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
            else if (execution.ReusableArena is { } reusableArena)
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
                var perSequencePastShape = geometry.GetPastKvShape(
                    batchSize: 1,
                    pastSequenceLength);
                var perSequencePastLength = CheckedTensorLength(perSequencePastShape);

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
                        var priorState = priorStates[execution.ItemIndicesByRow[row]]
                            ?? throw new InvalidOperationException(
                                "A non-empty past cohort is missing a prior decoder state.");
                        var priorLayer = priorState.GetLayer(layer);
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
                    checked((long)batchSize *
                            perSequencePastLength *
                            geometry.NumHiddenLayers *
                            2L));
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

            DisposeLeases(scratchLeases);
            retainedInputArena?.Release();
            nextArena?.Release();
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

        ValidatePriorState(
            priorState,
            startPosition,
            "Prefill continuation");
        return startPosition;
    }

    private int ValidateDecodeState(
        DecodeItem item,
        DecoderOrtState priorState)
    {
        ValidatePriorState(
            priorState,
            item.Position,
            "Decode");

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
        if (priorState.IsDisposed)
        {
            throw new ObjectDisposedException(
                nameof(priorState),
                $"{operation} cannot consume a disposed decoder state.");
        }

        if (priorState.LayerCount != Geometry.NumHiddenLayers)
        {
            throw new InvalidOperationException(
                $"Decoder state has {priorState.LayerCount} KV layers; " +
                $"geometry requires {Geometry.NumHiddenLayers}.");
        }

        if (priorState.Position != expectedPosition)
        {
            throw new InvalidOperationException(
                $"Decoder state position {priorState.Position} does not match " +
                $"requested position {expectedPosition}.");
        }
    }

    private static CohortExecution BuildCohortExecution(
        IReadOnlyList<DecoderOrtState?> priorStates,
        IReadOnlyList<int> cohort,
        int pastSequenceLength,
        int layerCount)
    {
        var identity = CreateIdentityCohortExecution(cohort);
        if (pastSequenceLength == 0)
        {
            return identity;
        }

        var firstState = priorStates[cohort[0]];
        if (firstState is null ||
            !firstState.TryGetCohortSlice(out var firstSlice))
        {
            return identity;
        }

        var arena = firstSlice.Arena;
        if (arena.Position != pastSequenceLength ||
            arena.BatchSize != cohort.Count ||
            arena.LayerCount != layerCount)
        {
            return identity;
        }

        var itemsByRow = new int[cohort.Count];
        var resultsByRow = new int[cohort.Count];
        Array.Fill(itemsByRow, -1);

        for (var cohortIndex = 0; cohortIndex < cohort.Count; cohortIndex++)
        {
            var itemIndex = cohort[cohortIndex];
            var state = priorStates[itemIndex];
            if (state is null ||
                !state.TryGetCohortSlice(out var slice) ||
                !ReferenceEquals(slice.Arena, arena) ||
                slice.Row < 0 ||
                slice.Row >= cohort.Count ||
                itemsByRow[slice.Row] != -1)
            {
                return identity;
            }

            itemsByRow[slice.Row] = itemIndex;
            resultsByRow[slice.Row] = cohortIndex;
        }

        for (var row = 0; row < cohort.Count; row++)
        {
            if (itemsByRow[row] == -1)
            {
                return identity;
            }
        }

        return new CohortExecution(itemsByRow, resultsByRow, arena);
    }

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

        return new CohortExecution(itemIndices, resultIndices, ReusableArena: null);
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
