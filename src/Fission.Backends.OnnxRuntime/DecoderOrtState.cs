using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

internal interface IDecoderOrtOwnedLifetimeAnchor
{
    void Release();
}

/// <summary>
/// Internal capability implemented by a state lifetime owner that can provide an
/// independently retained CUDA resident-state lease. This is used for CUDA states
/// whose allocations are not cohort-backed, such as H2D migration imports.
/// </summary>
internal interface IDecoderOrtCudaResidentStateSource
{
    DecoderOrtCudaResidentStateLease AcquireCudaResidentState(
        string formatId,
        DecoderOrtState state);
}

public readonly record struct DecoderOrtLayerState(
    OrtValue Key,
    OrtValue Value);

/// <summary>
/// Immutable owned physical decoder state for one version.
///
/// Besides KV tensors, the state may retain NextTokenId: the sampled token that
/// must be fed as input to the next one-token decode step. Keeping that token in
/// the same immutable object as KV means snapshot/fork/restore moves the complete
/// causal frontier rather than only the cache payload.
///
/// Construction transfers ownership of every layer Key/Value OrtValue to this
/// instance only after the complete payload validates. The same OrtValue instance
/// cannot appear in more than one owned slot. DecoderStateStore may then share the
/// state object across snapshots/branches and the OrtValues are disposed exactly
/// once when the final state owner disappears.
/// </summary>
public sealed class DecoderOrtState : IDisposable
{
    private readonly DecoderOrtLayerState[] _layers;
    private DecoderOrtCohortSlice? _cohortSlice;
    private DecoderOrtCudaCohortSlice? _cudaCohortSlice;
    private object? _lifetimeAnchor;
    private int _disposed;

    public DecoderOrtState(
        int position,
        IReadOnlyList<DecoderOrtLayerState> layers,
        int? nextTokenId = null)
        : this(
            position,
            layers,
            nextTokenId,
            cohortSlice: null,
            cudaCohortSlice: null,
            lifetimeAnchor: null)
    {
    }

    internal DecoderOrtState(
        int position,
        IReadOnlyList<DecoderOrtLayerState> layers,
        int? nextTokenId,
        DecoderOrtCohortSlice cohortSlice)
        : this(
            position,
            layers,
            nextTokenId,
            (DecoderOrtCohortSlice?)cohortSlice,
            cudaCohortSlice: null,
            lifetimeAnchor: null)
    {
    }

    internal DecoderOrtState(
        int position,
        IReadOnlyList<DecoderOrtLayerState> layers,
        int? nextTokenId,
        DecoderOrtCudaCohortSlice cudaCohortSlice)
        : this(
            position,
            layers,
            nextTokenId,
            cohortSlice: null,
            (DecoderOrtCudaCohortSlice?)cudaCohortSlice,
            lifetimeAnchor: null)
    {
    }

    /// <summary>
    /// Creates a state whose OrtValues are views over memory owned by an external
    /// managed object. The anchor is retained until after all OrtValues are
    /// disposed so host-staging buffers cannot be reclaimed early. Ordinary
    /// caller-supplied anchors are retained only; Fission-owned host-staging leases
    /// additionally receive a Release callback during state disposal.
    /// </summary>
    public DecoderOrtState(
        int position,
        IReadOnlyList<DecoderOrtLayerState> layers,
        int? nextTokenId,
        object lifetimeAnchor)
        : this(
            position,
            layers,
            nextTokenId,
            cohortSlice: null,
            cudaCohortSlice: null,
            lifetimeAnchor: lifetimeAnchor)
    {
        ArgumentNullException.ThrowIfNull(lifetimeAnchor);
    }

    private DecoderOrtState(
        int position,
        IReadOnlyList<DecoderOrtLayerState> layers,
        int? nextTokenId,
        DecoderOrtCohortSlice? cohortSlice,
        DecoderOrtCudaCohortSlice? cudaCohortSlice,
        object? lifetimeAnchor)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(position);
        if (nextTokenId is < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(nextTokenId));
        }

        ArgumentNullException.ThrowIfNull(layers);
        if (layers.Count == 0)
        {
            throw new ArgumentException(
                "Decoder KV state must contain at least one layer.",
                nameof(layers));
        }

        if (cohortSlice is not null && cudaCohortSlice is not null)
        {
            throw new ArgumentException(
                "Decoder state cannot belong to CPU and CUDA cohort arenas simultaneously.");
        }

        var owned = new HashSet<OrtValue>(ReferenceEqualityComparer.Instance);
        var validated = new DecoderOrtLayerState[layers.Count];

        for (var index = 0; index < layers.Count; index++)
        {
            var layer = layers[index];
            ArgumentNullException.ThrowIfNull(layer.Key);
            ArgumentNullException.ThrowIfNull(layer.Value);

            if (!layer.Key.IsTensor)
            {
                throw new ArgumentException(
                    $"Decoder layer {index} key OrtValue is not a tensor.",
                    nameof(layers));
            }

            if (!layer.Value.IsTensor)
            {
                throw new ArgumentException(
                    $"Decoder layer {index} value OrtValue is not a tensor.",
                    nameof(layers));
            }

            if (!owned.Add(layer.Key))
            {
                throw new ArgumentException(
                    $"Decoder layer {index} key OrtValue is already owned by another KV slot.",
                    nameof(layers));
            }

            if (!owned.Add(layer.Value))
            {
                throw new ArgumentException(
                    $"Decoder layer {index} value OrtValue is already owned by another KV slot.",
                    nameof(layers));
            }

            validated[index] = layer;
        }

        if (cohortSlice is { } slice)
        {
            ValidateCohortSlice(
                slice.Arena,
                slice.Row,
                position,
                layers.Count,
                nameof(cohortSlice));
            slice.Arena.Retain();
        }

        if (cudaCohortSlice is { } cudaSlice)
        {
            ValidateCudaCohortSlice(
                cudaSlice.Arena,
                cudaSlice.Row,
                position,
                layers.Count,
                nameof(cudaCohortSlice));
            cudaSlice.Arena.Retain();
        }

        Position = position;
        NextTokenId = nextTokenId;
        _layers = validated;
        _cohortSlice = cohortSlice;
        _cudaCohortSlice = cudaCohortSlice;
        _lifetimeAnchor = lifetimeAnchor;
    }

    public int Position { get; }
    public int? NextTokenId { get; }
    public int LayerCount => _layers.Length;
    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    public DecoderOrtLayerState GetLayer(int layer)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        ArgumentOutOfRangeException.ThrowIfNegative(layer);
        if (layer >= _layers.Length)
        {
            throw new ArgumentOutOfRangeException(nameof(layer));
        }

        return _layers[layer];
    }

    internal bool TryGetCohortSlice(out DecoderOrtCohortSlice slice)
    {
        if (!IsDisposed && _cohortSlice is { } current)
        {
            slice = current;
            return true;
        }

        slice = default;
        return false;
    }

    internal bool TryGetCudaCohortSlice(out DecoderOrtCudaCohortSlice slice)
    {
        if (!IsDisposed && _cudaCohortSlice is { } current)
        {
            slice = current;
            return true;
        }

        slice = default;
        return false;
    }

    /// <summary>
    /// Acquires an independently retained CUDA resident-state lease regardless of
    /// whether this state is backed by a cohort row or by a standalone CUDA
    /// lifetime owner such as H2D migration import.
    /// </summary>
    internal DecoderOrtCudaResidentStateLease AcquireCudaResidentState(
        string formatId)
    {
        ObjectDisposedException.ThrowIf(IsDisposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(formatId);

        if (_cudaCohortSlice is { } cudaSlice)
        {
            return cudaSlice.Arena.AcquireResidentState(
                formatId,
                this,
                cudaSlice.Row);
        }

        if (_lifetimeAnchor is IDecoderOrtCudaResidentStateSource source)
        {
            return source.AcquireCudaResidentState(formatId, this);
        }

        throw new InvalidOperationException(
            "Decoder state does not expose a retained CUDA resident-state source.");
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        for (var index = _layers.Length - 1; index >= 0; index--)
        {
            _layers[index].Value.Dispose();
            _layers[index].Key.Dispose();
        }

        if (_cohortSlice is { } slice)
        {
            _cohortSlice = null;
            slice.Arena.Release();
        }

        if (_cudaCohortSlice is { } cudaSlice)
        {
            _cudaCohortSlice = null;
            cudaSlice.Arena.Release();
        }

        var lifetimeAnchor = Interlocked.Exchange(ref _lifetimeAnchor, null);
        if (lifetimeAnchor is IDecoderOrtOwnedLifetimeAnchor ownedAnchor)
        {
            ownedAnchor.Release();
        }
    }

    private static void ValidateCohortSlice(
        DecoderOrtCohortArena arena,
        int row,
        int position,
        int layerCount,
        string parameterName)
    {
        ArgumentNullException.ThrowIfNull(arena);
        if (arena.Position != position)
        {
            throw new ArgumentException(
                "Decoder cohort arena position must match the state position.",
                parameterName);
        }

        if (arena.LayerCount != layerCount)
        {
            throw new ArgumentException(
                "Decoder cohort arena layer count must match the state payload.",
                parameterName);
        }

        if (row < 0 || row >= arena.BatchSize)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                "Decoder cohort row is outside the arena batch.");
        }
    }

    private static void ValidateCudaCohortSlice(
        CudaDecoderOrtCohortArena arena,
        int row,
        int position,
        int layerCount,
        string parameterName)
    {
        ArgumentNullException.ThrowIfNull(arena);
        if (arena.Position != position)
        {
            throw new ArgumentException(
                "CUDA decoder cohort arena position must match the state position.",
                parameterName);
        }

        if (arena.LayerCount != layerCount)
        {
            throw new ArgumentException(
                "CUDA decoder cohort arena layer count must match the state payload.",
                parameterName);
        }

        if (row < 0 || row >= arena.BatchSize)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                "CUDA decoder cohort row is outside the arena batch.");
        }
    }
}
