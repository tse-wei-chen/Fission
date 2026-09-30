using System.Collections.Concurrent;
using Fission.Abstractions;

namespace Fission.Runtime.Execution;

internal readonly record struct RuntimeDeviceMemoryReservationRequest(
    DeviceId Device,
    long Bytes);

internal readonly record struct RuntimeDeviceMemoryReservationSnapshot(
    DeviceId Device,
    long Bytes,
    long ReleaseVersion);

internal readonly record struct RuntimeDeviceMemoryReservationVersion(
    DeviceId Device,
    long ReleaseVersion);

internal sealed record RuntimeDeviceMemoryReservationState(
    IReadOnlyList<RuntimeDeviceMemoryReservationSnapshot> Reservations);

public sealed partial class ExecutionPlanExecutor
{
    private readonly ConcurrentDictionary<DeviceId, SemaphoreSlim>
        _deviceMemoryAdmissionGates = new();
    private readonly object _deviceMemoryReservationGate = new();
    private readonly Dictionary<DeviceId, long> _deviceMemoryReservations = new();
    private readonly Dictionary<DeviceId, long>
        _deviceMemoryReservationReleaseVersions = new();
    private readonly Dictionary<DeviceId, TaskCompletionSource<long>>
        _deviceMemoryReservationReleased = new();

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
            var reservations = _deviceMemoryReservations
                .OrderBy(static pair => pair.Key.Value, StringComparer.Ordinal)
                .Select(pair => new RuntimeDeviceMemoryReservationSnapshot(
                    pair.Key,
                    pair.Value,
                    GetDeviceMemoryReservationReleaseVersionLocked(pair.Key)))
                .ToArray();
            return new RuntimeDeviceMemoryReservationState(reservations);
        }
    }

    /// <summary>
    /// Waits until any observed physical device releases reservation bytes. Each
    /// observed version is device-scoped, so releases on unrelated GPUs do not
    /// wake this waiter. Version comparison and signal capture share the ledger
    /// lock, preserving the missed-wakeup protection used by admission retries.
    /// </summary>
    internal Task WaitForDeviceMemoryReservationReleaseAsync(
        IReadOnlyList<RuntimeDeviceMemoryReservationVersion> observedVersions,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(observedVersions);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        if (observedVersions.Count == 0)
        {
            return Task.CompletedTask;
        }

        Task<long>[] signals;
        lock (_deviceMemoryReservationGate)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

            var normalized = new Dictionary<DeviceId, long>();
            foreach (var observed in observedVersions)
            {
                if (normalized.TryGetValue(observed.Device, out var existing) &&
                    existing != observed.ReleaseVersion)
                {
                    throw new InvalidOperationException(
                        $"Conflicting reservation release versions were supplied for {observed.Device}: " +
                        $"{existing} and {observed.ReleaseVersion}.");
                }

                normalized[observed.Device] = observed.ReleaseVersion;
            }

            foreach (var (device, observedVersion) in normalized)
            {
                if (GetDeviceMemoryReservationReleaseVersionLocked(device) !=
                    observedVersion)
                {
                    return Task.CompletedTask;
                }
            }

            signals = normalized
                .OrderBy(static pair => pair.Key.Value, StringComparer.Ordinal)
                .Select(pair => GetOrCreateReservationReleaseSignalLocked(pair.Key).Task)
                .ToArray();
        }

        return WaitForAnyReservationReleaseAsync(signals, cancellationToken);
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
        var releasedSignals = new List<(TaskCompletionSource<long> Signal, long Version)>();

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

                var releaseVersion = checked(
                    GetDeviceMemoryReservationReleaseVersionLocked(device) + 1);
                _deviceMemoryReservationReleaseVersions[device] = releaseVersion;

                if (_deviceMemoryReservationReleased.Remove(device, out var signal))
                {
                    releasedSignals.Add((signal, releaseVersion));
                }
            }
        }

        foreach (var (signal, version) in releasedSignals)
        {
            signal.TrySetResult(version);
        }
    }

    private long GetDeviceMemoryReservationReleaseVersionLocked(DeviceId device) =>
        _deviceMemoryReservationReleaseVersions.TryGetValue(device, out var version)
            ? version
            : 0L;

    private TaskCompletionSource<long> GetOrCreateReservationReleaseSignalLocked(
        DeviceId device)
    {
        if (_deviceMemoryReservationReleased.TryGetValue(device, out var signal))
        {
            return signal;
        }

        signal = CreateReservationReleaseSignal();
        _deviceMemoryReservationReleased.Add(device, signal);
        return signal;
    }

    private static async Task WaitForAnyReservationReleaseAsync(
        IReadOnlyList<Task<long>> signals,
        CancellationToken cancellationToken)
    {
        var completed = await Task.WhenAny(signals)
            .WaitAsync(cancellationToken)
            .ConfigureAwait(false);
        await completed.ConfigureAwait(false);
    }

    private void DisposeDeviceMemoryReservationWaiters()
    {
        TaskCompletionSource<long>[] signals;
        lock (_deviceMemoryReservationGate)
        {
            signals = _deviceMemoryReservationReleased.Values.ToArray();
            _deviceMemoryReservationReleased.Clear();
        }

        foreach (var signal in signals)
        {
            signal.TrySetException(
                new ObjectDisposedException(nameof(ExecutionPlanExecutor)));
        }
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
