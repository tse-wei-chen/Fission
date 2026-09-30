using System.Collections.Concurrent;
using Fission.Abstractions;

namespace Fission.Runtime.Execution;

internal readonly record struct RuntimeDeviceMemoryReservationRequest(
    DeviceId Device,
    long Bytes);

internal readonly record struct RuntimeDeviceMemoryReservationSnapshot(
    DeviceId Device,
    long Bytes);

internal sealed record RuntimeDeviceMemoryReservationState(
    long ReleaseVersion,
    IReadOnlyList<RuntimeDeviceMemoryReservationSnapshot> Reservations);

public sealed partial class ExecutionPlanExecutor
{
    private readonly ConcurrentDictionary<DeviceId, SemaphoreSlim>
        _deviceMemoryAdmissionGates = new();
    private readonly object _deviceMemoryReservationGate = new();
    private readonly Dictionary<DeviceId, long> _deviceMemoryReservations = new();
    private long _deviceMemoryReservationReleaseVersion;
    private TaskCompletionSource<long> _deviceMemoryReservationReleased =
        CreateReservationReleaseSignal();

    /// <summary>
    /// Serializes pressure observation, optional reclaim, scheduling, and transient
    /// reservation only across engines whose scheduling candidate sets touch the
    /// same physical execution device. Multi-device admissions acquire gates in
    /// stable DeviceId order, preventing lock-order deadlocks while allowing
    /// disjoint GPUs to perform admission concurrently. Gates are released before
    /// backend execution; the returned reservation lease keeps admitted transient
    /// bytes charged until execution completes.
    /// </summary>
    internal async ValueTask<IDisposable> EnterDeviceMemoryAdmissionAsync(
        IReadOnlyList<DeviceId> devices,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(devices);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        var normalized = devices
            .Distinct()
            .OrderBy(static device => device.Value, StringComparer.Ordinal)
            .ToArray();
        if (normalized.Length == 0)
        {
            return EmptyAdmissionGateLease.Instance;
        }

        var acquired = new SemaphoreSlim[normalized.Length];
        var acquiredCount = 0;
        try
        {
            foreach (var device in normalized)
            {
                var gate = _deviceMemoryAdmissionGates.GetOrAdd(
                    device,
                    static _ => new SemaphoreSlim(1, 1));
                await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
                acquired[acquiredCount++] = gate;
            }

            if (Volatile.Read(ref _disposed) != 0)
            {
                throw new ObjectDisposedException(nameof(ExecutionPlanExecutor));
            }

            return new AdmissionGateLease(acquired);
        }
        catch
        {
            ReleaseAdmissionGates(acquired, acquiredCount);
            throw;
        }
    }

    internal IReadOnlyList<RuntimeDeviceMemoryReservationSnapshot>
        GetDeviceMemoryReservations() =>
        GetDeviceMemoryReservationState().Reservations;

    internal RuntimeDeviceMemoryReservationState GetDeviceMemoryReservationState()
    {
        lock (_deviceMemoryReservationGate)
        {
            return new RuntimeDeviceMemoryReservationState(
                _deviceMemoryReservationReleaseVersion,
                _deviceMemoryReservations
                    .OrderBy(static pair => pair.Key.Value, StringComparer.Ordinal)
                    .Select(static pair => new RuntimeDeviceMemoryReservationSnapshot(
                        pair.Key,
                        pair.Value))
                    .ToArray());
        }
    }

    internal Task WaitForDeviceMemoryReservationReleaseAsync(
        long observedReleaseVersion,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        Task<long> signal;
        lock (_deviceMemoryReservationGate)
        {
            if (_deviceMemoryReservationReleaseVersion != observedReleaseVersion)
            {
                return Task.CompletedTask;
            }

            signal = _deviceMemoryReservationReleased.Task;
        }

        return signal.WaitAsync(cancellationToken);
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
        TaskCompletionSource<long> releasedSignal;
        long releaseVersion;

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

            releaseVersion = checked(_deviceMemoryReservationReleaseVersion + 1);
            _deviceMemoryReservationReleaseVersion = releaseVersion;
            releasedSignal = _deviceMemoryReservationReleased;
            _deviceMemoryReservationReleased = CreateReservationReleaseSignal();
        }

        releasedSignal.TrySetResult(releaseVersion);
    }

    private static TaskCompletionSource<long> CreateReservationReleaseSignal() =>
        new(TaskCreationOptions.RunContinuationsAsynchronously);

    private static void ReleaseAdmissionGates(
        IReadOnlyList<SemaphoreSlim> gates,
        int count)
    {
        for (var index = count - 1; index >= 0; index--)
        {
            gates[index].Release();
        }
    }

    private sealed class AdmissionGateLease : IDisposable
    {
        private SemaphoreSlim[]? _gates;

        public AdmissionGateLease(SemaphoreSlim[] gates)
        {
            _gates = gates;
        }

        public void Dispose()
        {
            var gates = Interlocked.Exchange(ref _gates, null);
            if (gates is not null)
            {
                ReleaseAdmissionGates(gates, gates.Length);
            }
        }
    }

    private sealed class EmptyAdmissionGateLease : IDisposable
    {
        public static EmptyAdmissionGateLease Instance { get; } = new();
        public void Dispose()
        {
        }
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
