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

internal interface IRuntimeDeviceMemoryReservationLease : IDisposable
{
    void Release(DeviceId device);
}

public sealed partial class ExecutionPlanExecutor
{
    private readonly ConcurrentDictionary<DeviceId, SemaphoreSlim>
        _deviceMemoryAdmissionGates = new();
    private readonly ConcurrentDictionary<DeviceId, DeviceMemoryReservationLedger>
        _deviceMemoryReservationLedgers = new();

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

        var normalized = NormalizeDevices(devices);
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

    internal IReadOnlyList<RuntimeDeviceMemoryReservationSnapshot>
        GetDeviceMemoryReservations(IReadOnlyList<DeviceId> devices) =>
        GetDeviceMemoryReservationState(devices).Reservations;

    internal RuntimeDeviceMemoryReservationState GetDeviceMemoryReservationState() =>
        SnapshotDeviceMemoryReservationState(
            _deviceMemoryReservationLedgers.Values
                .OrderBy(static ledger => ledger.Device.Value, StringComparer.Ordinal)
                .ToArray());

    internal RuntimeDeviceMemoryReservationState GetDeviceMemoryReservationState(
        IReadOnlyList<DeviceId> devices)
    {
        ArgumentNullException.ThrowIfNull(devices);
        var ledgers = NormalizeDevices(devices)
            .Select(device => _deviceMemoryReservationLedgers.TryGetValue(device, out var ledger)
                ? ledger
                : null)
            .Where(static ledger => ledger is not null)
            .Cast<DeviceMemoryReservationLedger>()
            .ToArray();
        return SnapshotDeviceMemoryReservationState(ledgers);
    }

    private static RuntimeDeviceMemoryReservationState SnapshotDeviceMemoryReservationState(
        IReadOnlyList<DeviceMemoryReservationLedger> ledgers)
    {
        var reservations = new List<RuntimeDeviceMemoryReservationSnapshot>(ledgers.Count);
        foreach (var ledger in ledgers)
        {
            lock (ledger.Gate)
            {
                if (ledger.ReservedBytes == 0)
                {
                    continue;
                }

                reservations.Add(new RuntimeDeviceMemoryReservationSnapshot(
                    ledger.Device,
                    ledger.ReservedBytes,
                    ledger.ReleaseVersion));
            }
        }

        return new RuntimeDeviceMemoryReservationState(reservations);
    }

    /// <summary>
    /// Waits until any observed physical device releases reservation bytes. Each
    /// observed version is device-scoped, so releases on unrelated GPUs do not
    /// wake this waiter. Version comparison and signal capture are serialized by
    /// that device's ledger lock, preserving missed-wakeup protection without a
    /// runtime-wide reservation lock.
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

        var signals = new Task<long>[normalized.Count];
        var index = 0;
        foreach (var (device, observedVersion) in normalized
                     .OrderBy(static pair => pair.Key.Value, StringComparer.Ordinal))
        {
            var ledger = GetOrCreateDeviceMemoryReservationLedger(device);
            lock (ledger.Gate)
            {
                ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
                if (ledger.ReleaseVersion != observedVersion)
                {
                    return Task.CompletedTask;
                }

                ledger.ReleaseSignal ??= CreateReservationReleaseSignal();
                signals[index++] = ledger.ReleaseSignal.Task;
            }
        }

        return WaitForAnyReservationReleaseAsync(signals, cancellationToken);
    }

    internal IDisposable ReserveDeviceMemory(
        IReadOnlyList<RuntimeDeviceMemoryReservationRequest> requests)
    {
        var normalized = ReserveDeviceMemoryCore(requests);
        return normalized.Count == 0
            ? EmptyReservationLease.Instance
            : new DeviceMemoryReservationLease(this, normalized);
    }

    /// <summary>
    /// Atomically charges all requested physical devices, then returns one logical
    /// lease whose individual device reservations may be released independently as
    /// their execution groups reach a terminal state. Disposing the lease releases
    /// any devices that have not already completed.
    /// </summary>
    internal IRuntimeDeviceMemoryReservationLease ReserveDeviceMemoryByDevice(
        IReadOnlyList<RuntimeDeviceMemoryReservationRequest> requests)
    {
        var normalized = ReserveDeviceMemoryCore(requests);
        return new DeviceMemoryReservationLeaseSet(this, normalized);
    }

    private Dictionary<DeviceId, long> ReserveDeviceMemoryCore(
        IReadOnlyList<RuntimeDeviceMemoryReservationRequest> requests)
    {
        ArgumentNullException.ThrowIfNull(requests);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        var normalized = NormalizeReservationRequests(requests);
        if (normalized.Count == 0)
        {
            return normalized;
        }

        var entries = normalized
            .OrderBy(static pair => pair.Key.Value, StringComparer.Ordinal)
            .Select(pair => new ReservationLedgerMutation(
                GetOrCreateDeviceMemoryReservationLedger(pair.Key),
                pair.Value))
            .ToArray();

        using (EnterReservationLedgerLocks(entries))
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

            var updatedBytes = new long[entries.Length];
            for (var index = 0; index < entries.Length; index++)
            {
                updatedBytes[index] = checked(
                    entries[index].Ledger.ReservedBytes + entries[index].Bytes);
            }

            for (var index = 0; index < entries.Length; index++)
            {
                entries[index].Ledger.ReservedBytes = updatedBytes[index];
            }
        }

        return normalized;
    }

    private void ReleaseDeviceMemoryReservations(
        IReadOnlyDictionary<DeviceId, long> reservations)
    {
        var entries = reservations
            .OrderBy(static pair => pair.Key.Value, StringComparer.Ordinal)
            .Select(pair => new ReservationLedgerMutation(
                GetExistingDeviceMemoryReservationLedger(pair.Key),
                pair.Value))
            .ToArray();
        var releasedSignals = new List<(TaskCompletionSource<long> Signal, long Version)>(
            entries.Length);

        using (EnterReservationLedgerLocks(entries))
        {
            var remainingBytes = new long[entries.Length];
            var releaseVersions = new long[entries.Length];
            for (var index = 0; index < entries.Length; index++)
            {
                var entry = entries[index];
                if (entry.Ledger.ReservedBytes < entry.Bytes)
                {
                    throw new InvalidOperationException(
                        $"Device-memory reservation ledger underflow for {entry.Ledger.Device}: " +
                        $"reserved={entry.Ledger.ReservedBytes}, releasing={entry.Bytes}.");
                }

                remainingBytes[index] = entry.Ledger.ReservedBytes - entry.Bytes;
                releaseVersions[index] = checked(entry.Ledger.ReleaseVersion + 1);
            }

            for (var index = 0; index < entries.Length; index++)
            {
                var ledger = entries[index].Ledger;
                ledger.ReservedBytes = remainingBytes[index];
                ledger.ReleaseVersion = releaseVersions[index];

                if (ledger.ReleaseSignal is { } signal)
                {
                    ledger.ReleaseSignal = null;
                    releasedSignals.Add((signal, releaseVersions[index]));
                }
            }
        }

        foreach (var (signal, version) in releasedSignals)
        {
            signal.TrySetResult(version);
        }
    }

    private static Dictionary<DeviceId, long> NormalizeReservationRequests(
        IReadOnlyList<RuntimeDeviceMemoryReservationRequest> requests)
    {
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

        return normalized;
    }

    private DeviceMemoryReservationLedger GetOrCreateDeviceMemoryReservationLedger(
        DeviceId device) =>
        _deviceMemoryReservationLedgers.GetOrAdd(
            device,
            static key => new DeviceMemoryReservationLedger(key));

    private DeviceMemoryReservationLedger GetExistingDeviceMemoryReservationLedger(
        DeviceId device) =>
        _deviceMemoryReservationLedgers.TryGetValue(device, out var ledger)
            ? ledger
            : throw new InvalidOperationException(
                $"Device-memory reservation ledger does not exist for {device}.");

    private static IDisposable EnterReservationLedgerLocks(
        IReadOnlyList<ReservationLedgerMutation> entries)
    {
        var ledgers = entries
            .Select(static entry => entry.Ledger)
            .Distinct()
            .OrderBy(static ledger => ledger.Device.Value, StringComparer.Ordinal)
            .ToArray();
        var acquiredCount = 0;
        try
        {
            foreach (var ledger in ledgers)
            {
                Monitor.Enter(ledger.Gate);
                acquiredCount++;
            }

            return new ReservationLedgerLockLease(ledgers);
        }
        catch
        {
            ReleaseReservationLedgerLocks(ledgers, acquiredCount);
            throw;
        }
    }

    private static void ReleaseReservationLedgerLocks(
        IReadOnlyList<DeviceMemoryReservationLedger> ledgers,
        int count)
    {
        for (var index = count - 1; index >= 0; index--)
        {
            Monitor.Exit(ledgers[index].Gate);
        }
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
        var signals = new List<TaskCompletionSource<long>>();
        foreach (var ledger in _deviceMemoryReservationLedgers.Values)
        {
            lock (ledger.Gate)
            {
                if (ledger.ReleaseSignal is not { } signal)
                {
                    continue;
                }

                ledger.ReleaseSignal = null;
                signals.Add(signal);
            }
        }

        foreach (var signal in signals)
        {
            signal.TrySetException(
                new ObjectDisposedException(nameof(ExecutionPlanExecutor)));
        }
    }

    private static TaskCompletionSource<long> CreateReservationReleaseSignal() =>
        new(TaskCreationOptions.RunContinuationsAsynchronously);

    private static DeviceId[] NormalizeDevices(IReadOnlyList<DeviceId> devices) =>
        devices
            .Distinct()
            .OrderBy(static device => device.Value, StringComparer.Ordinal)
            .ToArray();

    private static void ReleaseAdmissionGates(
        IReadOnlyList<SemaphoreSlim> gates,
        int count)
    {
        for (var index = count - 1; index >= 0; index--)
        {
            gates[index].Release();
        }
    }

    private sealed class DeviceMemoryReservationLedger
    {
        public DeviceMemoryReservationLedger(DeviceId device)
        {
            Device = device;
        }

        public DeviceId Device { get; }
        public object Gate { get; } = new();
        public long ReservedBytes { get; set; }
        public long ReleaseVersion { get; set; }
        public TaskCompletionSource<long>? ReleaseSignal { get; set; }
    }

    private readonly record struct ReservationLedgerMutation(
        DeviceMemoryReservationLedger Ledger,
        long Bytes);

    private sealed class ReservationLedgerLockLease : IDisposable
    {
        private DeviceMemoryReservationLedger[]? _ledgers;

        public ReservationLedgerLockLease(DeviceMemoryReservationLedger[] ledgers)
        {
            _ledgers = ledgers;
        }

        public void Dispose()
        {
            var ledgers = Interlocked.Exchange(ref _ledgers, null);
            if (ledgers is not null)
            {
                ReleaseReservationLedgerLocks(ledgers, ledgers.Length);
            }
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

    private sealed class DeviceMemoryReservationLeaseSet :
        IRuntimeDeviceMemoryReservationLease
    {
        private readonly object _gate = new();
        private ExecutionPlanExecutor? _owner;
        private Dictionary<DeviceId, long>? _remaining;

        public DeviceMemoryReservationLeaseSet(
            ExecutionPlanExecutor owner,
            IReadOnlyDictionary<DeviceId, long> reservations)
        {
            if (reservations.Count == 0)
            {
                return;
            }

            _owner = owner;
            _remaining = reservations.ToDictionary(
                static pair => pair.Key,
                static pair => pair.Value);
        }

        public void Release(DeviceId device)
        {
            lock (_gate)
            {
                if (_owner is null ||
                    _remaining is null ||
                    !_remaining.TryGetValue(device, out var bytes))
                {
                    return;
                }

                _owner.ReleaseDeviceMemoryReservations(
                    new Dictionary<DeviceId, long>
                    {
                        [device] = bytes
                    });
                _remaining.Remove(device);
                if (_remaining.Count == 0)
                {
                    _remaining = null;
                    _owner = null;
                }
            }
        }

        public void Dispose()
        {
            lock (_gate)
            {
                if (_owner is null || _remaining is null)
                {
                    return;
                }

                _owner.ReleaseDeviceMemoryReservations(_remaining);
                _remaining = null;
                _owner = null;
            }
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
