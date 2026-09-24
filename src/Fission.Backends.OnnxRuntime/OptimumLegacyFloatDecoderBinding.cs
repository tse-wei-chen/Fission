using Fission.Abstractions.Execution;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// First concrete causal-LM binding for the Optimum legacy decoder-with-past
/// tensor convention, using CPU FP32 KV/logits buffers and greedy sampling.
///
/// Prefill is executed by supplying zero-length past KV tensors. Therefore the
/// bound ONNX graph must accept an empty past sequence on its with-past path.
/// Graphs that require a distinct no-past prefill session need a dual-session
/// binding and are intentionally outside this first implementation.
/// </summary>
public sealed class OptimumLegacyFloatDecoderBinding : IDecoderOrtModelBinding
{
    private readonly OptimumLegacyDecoderProfile _profile;
    private readonly HashSet<int> _eosTokenIds;
    private int _disposed;

    public OptimumLegacyFloatDecoderBinding(
        OptimumLegacyDecoderProfile profile,
        IEnumerable<int>? eosTokenIds = null)
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
                "Additional model-specific tensors are not supported by the first Optimum FP32 binding.",
                nameof(profile));
        }

        _profile = profile;
        _eosTokenIds = eosTokenIds is null
            ? new HashSet<int>()
            : new HashSet<int>(eosTokenIds);

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

    public DecoderOrtStepResult ExecutePrefill(
        InferenceSession session,
        PrefillItem item,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(session);
        cancellationToken.ThrowIfCancellationRequested();

        if (item.Tokens.Length == 0)
        {
            throw new InvalidOperationException(
                "Decoder prefill requires at least one prompt token.");
        }

        var inputIds = Array.ConvertAll(
            item.Tokens.Span.ToArray(),
            static token => (long)token);
        if (inputIds.Any(static token => token < 0))
        {
            throw new InvalidOperationException(
                "Decoder input token ids cannot be negative.");
        }

        var positions = new long[inputIds.Length];
        for (var index = 0; index < positions.Length; index++)
        {
            positions[index] = index;
        }

        return ExecuteStep(
            session,
            inputIds,
            positions,
            priorState: null,
            pastSequenceLength: 0,
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

        return ExecuteStep(
            session,
            new long[] { nextInputToken },
            new long[] { item.Position },
            priorState,
            item.Position,
            cancellationToken);
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

            var logits = new float[CheckedTensorLength(
                geometry.GetLogitsShape(batchSize: 1, sequenceLength))];
            var logitsValue = OrtValue.CreateTensorValueFromMemory(
                logits,
                geometry.GetLogitsShape(batchSize: 1, sequenceLength));
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

            var tokenId = GreedySampleLastPosition(
                logits,
                sequenceLength,
                geometry.VocabularySize);
            var state = new DecoderOrtState(
                checked(pastSequenceLength + sequenceLength),
                nextLayers,
                nextTokenId: tokenId);
            stateOwnsKv = true;

            // Logits are step-local; KV outputs have transferred into state.
            logitsValue.Dispose();
            ownedOutputs.Remove(logitsValue);
            for (var index = ownedOutputs.Count - 1; index >= 0; index--)
            {
                // Every remaining output is now owned by DecoderOrtState.
                ownedOutputs.RemoveAt(index);
            }

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
}
