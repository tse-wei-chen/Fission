using Fission.Abstractions.Execution;

namespace Fission.Runtime.Execution;

/// <summary>
/// Aggregates independent physical device-memory sources into one pressure and
/// reclaim capability. This lets a backend report allocator caches together with
/// model weights, CUDA Graph residency, workspaces, or other device allocations
/// without folding those bytes into logical KV accounting.
/// </summary>
public sealed class CompositeInferenceDeviceMemoryController :
    IInferenceDeviceMemoryPressureSource,
    IInferenceDeviceMemoryReclaimer
{
    private readonly IInferenceDeviceMemoryPressureSource[] _sources;

    public CompositeInferenceDeviceMemoryController(
        params IInferenceDeviceMemoryPressureSource[] sources)
    {
        ArgumentNullException.ThrowIfNull(sources);
        if (sources.Length == 0)
        {
            throw new ArgumentException(
                "At least one device-memory pressure source is required.",
                nameof(sources));
        }

        _sources = new IInferenceDeviceMemoryPressureSource[sources.Length];
        for (var index = 0; index < sources.Length; index++)
        {
            var source = sources[index] ?? throw new ArgumentException(
                "Device-memory pressure sources cannot contain null entries.",
                nameof(sources));

            for (var prior = 0; prior < index; prior++)
            {
                if (ReferenceEquals(_sources[prior], source))
                {
                    throw new ArgumentException(
                        "The same device-memory pressure source cannot be aggregated more than once.",
                        nameof(sources));
                }
            }

            _sources[index] = source;
        }
    }

    public bool TryGetDeviceMemoryPressure(
        out InferenceDeviceMemoryPressure pressure)
    {
        if (!TryCaptureComponents(out var components))
        {
            pressure = default;
            return false;
        }

        pressure = Aggregate(components);
        return true;
    }

    public async ValueTask<InferenceDeviceMemoryReclaimResult> ReclaimDeviceMemoryAsync(
        long targetReclaimableBytes,
        CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(targetReclaimableBytes);
        cancellationToken.ThrowIfCancellationRequested();

        if (!TryCaptureComponents(out var components))
        {
            throw new InvalidOperationException(
                "Composite device-memory pressure became unavailable before reclaim.");
        }

        var before = Aggregate(components);
        if (before.ReclaimableBytes <= targetReclaimableBytes)
        {
            return new InferenceDeviceMemoryReclaimResult(
                ReleasedBytes: 0,
                ReclaimableBytes: before.ReclaimableBytes,
                ReservedBytes: before.ReservedBytes);
        }

        var remainingReclaimable = before.ReclaimableBytes;
        long releasedBytes = 0;

        foreach (var component in components
                     .Where(static component =>
                         component.Pressure.ReclaimableBytes > 0 &&
                         component.Source is IInferenceDeviceMemoryReclaimer)
                     .OrderByDescending(static component =>
                         component.Pressure.ReclaimableBytes)
                     .ThenBy(static component => component.Index))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (remainingReclaimable <= targetReclaimableBytes)
            {
                break;
            }

            var releaseNeeded = checked(
                remainingReclaimable - targetReclaimableBytes);
            var componentTarget = Math.Max(
                0L,
                component.Pressure.ReclaimableBytes - releaseNeeded);
            var reclaimer = (IInferenceDeviceMemoryReclaimer)component.Source;
            var result = await reclaimer.ReclaimDeviceMemoryAsync(
                    componentTarget,
                    cancellationToken)
                .ConfigureAwait(false);

            ValidateReclaimResult(result);
            remainingReclaimable = checked(
                remainingReclaimable -
                component.Pressure.ReclaimableBytes +
                result.ReclaimableBytes);
            releasedBytes = checked(releasedBytes + result.ReleasedBytes);
        }

        if (!TryCaptureComponents(out var afterComponents))
        {
            throw new InvalidOperationException(
                "Composite device-memory pressure became unavailable after reclaim.");
        }

        var after = Aggregate(afterComponents);
        return new InferenceDeviceMemoryReclaimResult(
            ReleasedBytes: releasedBytes,
            ReclaimableBytes: after.ReclaimableBytes,
            ReservedBytes: after.ReservedBytes);
    }

    private bool TryCaptureComponents(out MemoryComponent[] components)
    {
        components = new MemoryComponent[_sources.Length];
        for (var index = 0; index < _sources.Length; index++)
        {
            var source = _sources[index];
            if (!source.TryGetDeviceMemoryPressure(out var pressure))
            {
                components = Array.Empty<MemoryComponent>();
                return false;
            }

            ValidatePressure(pressure);
            components[index] = new MemoryComponent(index, source, pressure);
        }

        return true;
    }

    private static InferenceDeviceMemoryPressure Aggregate(
        IReadOnlyList<MemoryComponent> components)
    {
        long activeBytes = 0;
        long reclaimableBytes = 0;
        long reservedBytes = 0;
        long peakReservedBytes = 0;

        foreach (var component in components)
        {
            activeBytes = checked(activeBytes + component.Pressure.ActiveBytes);
            reclaimableBytes = checked(
                reclaimableBytes + component.Pressure.ReclaimableBytes);
            reservedBytes = checked(
                reservedBytes + component.Pressure.ReservedBytes);
            peakReservedBytes = checked(
                peakReservedBytes + component.Pressure.PeakReservedBytes);
        }

        return new InferenceDeviceMemoryPressure(
            activeBytes,
            reclaimableBytes,
            reservedBytes,
            peakReservedBytes);
    }

    private static void ValidatePressure(InferenceDeviceMemoryPressure pressure)
    {
        if (pressure.ActiveBytes < 0 ||
            pressure.ReclaimableBytes < 0 ||
            pressure.ReservedBytes < 0 ||
            pressure.PeakReservedBytes < 0)
        {
            throw new InvalidOperationException(
                "A composite device-memory source reported negative byte counts.");
        }

        if (checked(pressure.ActiveBytes + pressure.ReclaimableBytes) !=
            pressure.ReservedBytes)
        {
            throw new InvalidOperationException(
                "A composite device-memory source must satisfy reserved = active + reclaimable bytes.");
        }

        if (pressure.PeakReservedBytes < pressure.ReservedBytes)
        {
            throw new InvalidOperationException(
                "A composite device-memory source peak cannot be smaller than current reserved bytes.");
        }
    }

    private static void ValidateReclaimResult(
        InferenceDeviceMemoryReclaimResult result)
    {
        if (result.ReleasedBytes < 0 ||
            result.ReclaimableBytes < 0 ||
            result.ReservedBytes < 0)
        {
            throw new InvalidOperationException(
                "A composite device-memory reclaimer returned negative byte counts.");
        }

        if (result.ReclaimableBytes > result.ReservedBytes)
        {
            throw new InvalidOperationException(
                "A composite device-memory reclaimer returned reclaimable bytes larger than reserved bytes.");
        }
    }

    private readonly record struct MemoryComponent(
        int Index,
        IInferenceDeviceMemoryPressureSource Source,
        InferenceDeviceMemoryPressure Pressure);
}
