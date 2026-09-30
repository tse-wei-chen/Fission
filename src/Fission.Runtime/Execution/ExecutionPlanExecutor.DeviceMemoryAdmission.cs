using Fission.Abstractions;

namespace Fission.Runtime.Execution;

internal readonly record struct RuntimeDeviceMemoryReservationRequest(
    DeviceId Device,
    long Bytes);

internal readonly record struct RuntimeDeviceMemoryReservationSnapshot(
    DeviceId Device,
    long Bytes);

public sealed partial class ExecutionPlanExecutor
{
    private readonly SemaphoreSlim _deviceMemoryAdmissionGate = new(1, 1);
    private readonly object _deviceMemoryReservationGate = new();
    private readonly Dictionary<DeviceId, long> _deviceMemoryReservations = new();

    /// <summary>
    /// Serializes pressure observation, optional reclaim, scheduling, and transient
    /// reservation across InferenceEngine instances that share this runtime. The
    /// gate is released before backend execution; the returned reservation lease
    /// keeps admitted transient bytes charged until execution completes.
    /// </summary>
    internal async ValueTask<IDisposable> EnterDeviceMemoryAdmissionAsync(
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        await _deviceMemoryAdmissionGate.WaitAsync(cancellationToken)
            .ConfigureAwait(false);

        if (Volatile.Read(ref _disposed) != 0)
        {
            _deviceMemoryAdmissionGate.Release();
            throw new ObjectDisposedException(nameof(ExecutionPlanExecutor));
        }

        return new AdmissionGateLease(_deviceMemoryAdmissionGate);
    }

    internal IReadOnlyList<RuntimeDeviceMemoryReservationSnapshot>
        GetDeviceMemoryReservations()
    {
        lock (_deviceMemoryReservationGate)
        {
            return _deviceMemoryReservations
                .OrderBy(static pair => pair.Key.Value, StringComparer.Ordinal)
                .Select(static pair => new RuntimeDeviceMemoryReservationSnapshot(
                    pair.Key,
                    pair.Value))
                .ToArray();
        }
    }

    internal IDisposable ReserveDeviceMemory(
        IReadOnlyList<RuntimeDeviceMemoryReservationRequest> requests)
    {
        ArgumentNullException.ThrowIfNull(requests);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        var normalized = new Dictionary<DeviceId, long>();
        foreach (var request in requests)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(request.Bytes);
            if (request.Bytes == 0)
            {
                continue;
            }

            normalized.TryGetValue(request.Device, out var existing);
            normalized[request.Device] = checked(existing + request.Bytes);
        }

        if (normalized.Count == 0)
        {
            return EmptyReservationLease.Instance;
        }

        lock (_deviceMemoryReservationGate)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            foreach (var (device, bytes) in normalized)
            {
                _deviceMemoryReservations.TryGetValue(device, out var existing);
                _deviceMemoryReservations[device] = checked(existing + bytes);
            }
        }

        return new DeviceMemoryReservationLease(this, normalized);
    }

    private void ReleaseDeviceMemoryReservations(
        IReadOnlyDictionary<DeviceId, long> reservations)
    {
        lock (_deviceMemoryReservationGate)
        {
            foreach (var (device, bytes) in reservations)
            {
                if (!_deviceMemoryReservations.TryGetValue(device, out var existing) ||
                    existing < bytes)
                {
                    throw new InvalidOperationException(
                        $"Device-memory reservation ledger underflow for {device}: " +
                        $"reserved={existing}, releasing={bytes}.");
                }

                var remaining = existing - bytes;
                if (remaining == 0)
                {
                    _deviceMemoryReservations.Remove(device);
                }
                else
                {
                    _deviceMemoryReservations[device] = remaining;
                }
            }
        }
    }

    private sealed class AdmissionGateLease : IDisposable
    {
        private SemaphoreSlim? _gate;

        public AdmissionGateLease(SemaphoreSlim gate)
        {
            _gate = gate;
        }

        public void Dispose() =>
            Interlocked.Exchange(ref _gate, null)?.Release();
    }

    private sealed class DeviceMemoryReservationLease : IDisposable
    {
        private ExecutionPlanExecutor? _owner;
        private readonly IReadOnlyDictionary<DeviceId, long> _reservations;

        public DeviceMemoryReservationLease(
            ExecutionPlanExecutor owner,
            IReadOnlyDictionary<DeviceId, long> reservations)
        {
            _owner = owner;
            _reservations = reservations;
        }

        public void Dispose()
        {
            Interlocked.Exchange(ref _owner, null)?
                .ReleaseDeviceMemoryReservations(_reservations);
        }
    }

    private sealed class EmptyReservationLease : IDisposable
    {
        public static EmptyReservationLease Instance { get; } = new();
        public void Dispose()
        {
        }
    }
}
