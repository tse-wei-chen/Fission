using Fission.Abstractions.Execution;

namespace Fission.Runtime.Execution;

/// <summary>
/// Deterministically selects one mutually supported migration transport.
/// Selection minimizes the conservative estimated completion time, then prefers
/// higher backend preference, then TransportId ordinal for stable tie-breaking.
/// </summary>
public sealed class SequenceMigrationTransportPlanner
{
    public SequenceMigrationTransportPlan Plan(
        long estimatedBytes,
        IReadOnlyList<SequenceMigrationTransportCapability> sourceCapabilities,
        IReadOnlyList<SequenceMigrationTransportCapability> targetCapabilities)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(estimatedBytes);
        ArgumentNullException.ThrowIfNull(sourceCapabilities);
        ArgumentNullException.ThrowIfNull(targetCapabilities);

        var source = Normalize(sourceCapabilities, "source");
        var target = Normalize(targetCapabilities, "target");
        SequenceMigrationTransportPlan? best = null;

        foreach (var (transportId, sourceCapability) in source)
        {
            if (!target.TryGetValue(transportId, out var targetCapability) ||
                targetCapability.Kind != sourceCapability.Kind)
            {
                continue;
            }

            var effectiveMaxTransferBytes = IntersectMaxTransferBytes(
                sourceCapability.MaxTransferBytes,
                targetCapability.MaxTransferBytes);
            if (effectiveMaxTransferBytes != long.MaxValue && estimatedBytes > effectiveMaxTransferBytes)
            {
                continue;
            }

            var bandwidth = Math.Min(
                sourceCapability.EstimatedBandwidthBytesPerSecond,
                targetCapability.EstimatedBandwidthBytesPerSecond);
            var fixedLatency = sourceCapability.EstimatedFixedLatency >= targetCapability.EstimatedFixedLatency
                ? sourceCapability.EstimatedFixedLatency
                : targetCapability.EstimatedFixedLatency;
            var preference = Math.Min(sourceCapability.Preference, targetCapability.Preference);
            var estimatedDuration = EstimateDuration(estimatedBytes, bandwidth, fixedLatency);

            var candidate = new SequenceMigrationTransportPlan(
                transportId,
                sourceCapability.Kind,
                estimatedBytes,
                effectiveMaxTransferBytes,
                bandwidth,
                fixedLatency,
                estimatedDuration,
                preference);

            if (best is null || Compare(candidate, best) < 0)
            {
                best = candidate;
            }
        }

        return best ?? throw new InvalidOperationException(
            $"No mutually supported migration transport can carry {estimatedBytes} byte(s).");
    }

    private static Dictionary<string, SequenceMigrationTransportCapability> Normalize(
        IReadOnlyList<SequenceMigrationTransportCapability> capabilities,
        string side)
    {
        var normalized = new Dictionary<string, SequenceMigrationTransportCapability>(
            StringComparer.Ordinal);

        foreach (var capability in capabilities)
        {
            ArgumentNullException.ThrowIfNull(capability);
            if (string.IsNullOrWhiteSpace(capability.TransportId))
            {
                throw new InvalidOperationException(
                    $"Migration transport capability on {side} has an empty TransportId.");
            }

            ArgumentOutOfRangeException.ThrowIfNegative(capability.MaxTransferBytes);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(
                capability.EstimatedBandwidthBytesPerSecond);
            if (capability.EstimatedFixedLatency < TimeSpan.Zero)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(capability.EstimatedFixedLatency),
                    "Migration transport fixed latency cannot be negative.");
            }

            if (!normalized.TryAdd(capability.TransportId, capability))
            {
                throw new InvalidOperationException(
                    $"Migration transport '{capability.TransportId}' is advertised more than once by the {side} backend.");
            }
        }

        return normalized;
    }

    private static long IntersectMaxTransferBytes(long source, long target)
    {
        var sourceLimit = source == 0 ? long.MaxValue : source;
        var targetLimit = target == 0 ? long.MaxValue : target;
        return Math.Min(sourceLimit, targetLimit);
    }

    private static TimeSpan EstimateDuration(
        long estimatedBytes,
        long bandwidthBytesPerSecond,
        TimeSpan fixedLatency)
    {
        var transferSeconds = (double)estimatedBytes / bandwidthBytesPerSecond;
        return fixedLatency + TimeSpan.FromSeconds(transferSeconds);
    }

    private static int Compare(
        SequenceMigrationTransportPlan left,
        SequenceMigrationTransportPlan right)
    {
        var duration = left.EstimatedDuration.CompareTo(right.EstimatedDuration);
        if (duration != 0)
        {
            return duration;
        }

        var preference = right.Preference.CompareTo(left.Preference);
        if (preference != 0)
        {
            return preference;
        }

        return StringComparer.Ordinal.Compare(left.TransportId, right.TransportId);
    }
}
