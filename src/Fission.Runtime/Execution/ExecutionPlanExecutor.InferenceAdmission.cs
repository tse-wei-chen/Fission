using System.Collections.Concurrent;
using Fission.Abstractions;

namespace Fission.Runtime.Execution;

internal readonly record struct RuntimeDeviceInferenceReservationRequest(
    DeviceId Device,
    int Items);

internal readonly record struct RuntimeDeviceInferenceReservationSnapshot(
    DeviceId Device,
    int Items,
    long ReleaseVersion);

internal readonly record struct RuntimeDeviceInferenceReservationVersion(
    DeviceId Device,
    long ReleaseVersion);

internal sealed record RuntimeDeviceInferenceReservationState(
    IReadOnlyList<RuntimeDeviceInferenceReservationSnapshot> Reservations);

internal interface IRuntimeDeviceInferenceReservationLease : IDisposable
{
    void Release(DeviceId device);
}

/// <summary>
/// Runtime-scoped reservation ledger for inference item credits that have been
/// admitted by a scheduler but are not represented in the device actor's credit
/// gate yet. This closes the shared-Engine admission race without treating a
/// racy SemaphoreSlim availability snapshot as authoritative.
/// </summary>
public sealed partial class ExecutionPlanExecutor
{
    private static readonly RuntimeDeviceInferenceReservationState EmptyDeviceInferenceReservationState =
        new(Array.Empty<RuntimeDeviceInferenceReservationSnapshot>());

    private readonly ConcurrentDictionary<DeviceId, DeviceInferenceReservationLedger>
        _deviceInferenceReservationLedgers = new();

    internal RuntimeDeviceInferenceReservationState GetDeviceInferenceReservationState()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return SnapshotDeviceInferenceReservationState(
            _deviceInferenceReservationLedgers.Values
                .OrderBy(static ledger => ledger.Device.Value, StringComparer.Ordinal)
                .ToArray());
    }

    internal RuntimeDeviceInferenceReservationState GetDeviceInferenceReservationState(
        IReadOnlyList<DeviceId> devices)
    {
        ArgumentNullException.ThrowIfNull(devices);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        var normalized = NormalizeDevices(devices);
        List<RuntimeDeviceInferenceReservationSnapshot>? reservations = null;

        foreach (var device in normalized)
        {
            if (!_deviceInferenceReservationLedgers.TryGetValue(device, out var ledger))
            {
                continue;
            }

            lock (ledger.Gate)
            {
                if (ledger.ReservedItems == 0)
                {
                    continue;
                }

                (reservations ??= new List<RuntimeDeviceInferenceReservationSnapshot>(normalized.Length))
                    .Add(new RuntimeDeviceInferenceReservationSnapshot(
                        ledger.Device,
                        ledger.ReservedItems,
                        ledger.ReleaseVersion));
            }
        }

        return reservations is null
            ? EmptyDeviceInferenceReservationState
            : new RuntimeDeviceInferenceReservationState(reservations);
    }

    /// <summary>
    /// Waits until any observed physical device releases inference-item
    /// reservations. Version comparison and signal capture happen under that
    /// device's ledger lock, so a release immediately before waiter attachment is
    /// observed as a version mismatch instead of being lost.
    /// </summary>
    internal Task WaitForDeviceInferenceReservationReleaseAsync(
        IReadOnlyList<RuntimeDeviceInferenceReservationVersion> observedVersions,
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
                    $"Conflicting inference reservation release versions were supplied for {observed.Device}: " +
                    $"{existing} and {observed.ReleaseVersion}.");
            }

            normalized[observed.Device] = observed.ReleaseVersion;
        }

        var signals = new Task<long>[normalized.Count];
        var index = 0;
        foreach (var (device, observedVersion) in normalized
                     .OrderBy(static pair => pair.Key.Value, StringComparer.Ordinal))
        {
            var ledger = GetOrCreateDeviceInferenceReservationLedger(device);
            lock (ledger.Gate)
            {
                ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
                if (ledger.ReleaseVersion != observedVersion)
                {
                    return Task.CompletedTask;
                }

                ledger.ReleaseSignal ??= CreateInferenceReservationReleaseSignal();
                signals[index++] = ledger.ReleaseSignal.Task;
            }
        }

        return WaitForAnyInferenceReservationReleaseAsync(signals, cancellationToken);
    }

    internal bool TryReserveDeviceInference(
        IReadOnlyList<RuntimeDeviceInferenceReservationRequest> requests,
        out IRuntimeDeviceInferenceReservationLease reservation)
    {
        ArgumentNullException.ThrowIfNull(requests);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        if (requests.Count == 1)
        {
            var request = requests[0];
            ArgumentOutOfRangeException.ThrowIfNegative(request.Items);
            if (request.Items == 0)
            {
                reservation = EmptyInferenceReservationLease.Instance;
                return true;
            }

            var capacity = GetDeviceExecutionCapacityCore(request.Device)
                .InferenceCreditCapacity;
            var ledger = GetOrCreateDeviceInferenceReservationLedger(request.Device);
            lock (ledger.Gate)
            {
                ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

                var updated = checked(ledger.ReservedItems + request.Items);
                if (updated > capacity)
                {
                    reservation = EmptyInferenceReservationLease.Instance;
                    return false;
                }

                ledger.ReservedItems = updated;
            }

            reservation = new SingleDeviceInferenceReservationLease(
                this,
                request.Device,
                request.Items);
            return true;
        }

        var normalized = NormalizeInferenceReservationRequests(requests);
        if (normalized.Count == 0)
        {
            reservation = EmptyInferenceReservationLease.Instance;
            return true;
        }

        var entries = normalized
            .OrderBy(static pair => pair.Key.Value, StringComparer.Ordinal)
            .Select(pair =>
            {
                var capacity = GetDeviceExecutionCapacityCore(pair.Key)
                    .InferenceCreditCapacity;
                return new InferenceReservationLedgerMutation(
                    GetOrCreateDeviceInferenceReservationLedger(pair.Key),
                    pair.Value,
                    capacity);
            })
            .ToArray();

        using (EnterInferenceReservationLedgerLocks(entries))
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

            var updatedItems = new int[entries.Length];
            for (var index = 0; index < entries.Length; index++)
            {
                var entry = entries[index];
                var updated = checked(entry.Ledger.ReservedItems + entry.Items);
                if (updated > entry.Capacity)
                {
                    reservation = EmptyInferenceReservationLease.Instance;
                    return false;
                }

                updatedItems[index] = updated;
            }

            for (var index = 0; index < entries.Length; index++)
            {
                entries[index].Ledger.ReservedItems = updatedItems[index];
            }
        }

        reservation = new DeviceInferenceReservationLeaseSet(this, normalized);
        return true;
    }

    private static RuntimeDeviceInferenceReservationState
        SnapshotDeviceInferenceReservationState(
            IReadOnlyList<DeviceInferenceReservationLedger> ledgers)
    {
        List<RuntimeDeviceInferenceReservationSnapshot>? reservations = null;
        foreach (var ledger in ledgers)
        {
            lock (ledger.Gate)
            {
                if (ledger.ReservedItems == 0)
                {
                    continue;
                }

                (reservations ??= new List<RuntimeDeviceInferenceReservationSnapshot>(ledgers.Count))
                    .Add(new RuntimeDeviceInferenceReservationSnapshot(
                        ledger.Device,
                        ledger.ReservedItems,
                        ledger.ReleaseVersion));
            }
        }

        return reservations is null
            ? EmptyDeviceInferenceReservationState
            : new RuntimeDeviceInferenceReservationState(reservations);
    }

    private static Dictionary<DeviceId, int> NormalizeInferenceReservationRequests(
        IReadOnlyList<RuntimeDeviceInferenceReservationRequest> requests)
    {
        var normalized = new Dictionary<DeviceId, int>();
        foreach (var request in requests)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(request.Items);
            if (request.Items == 0)
            {
                continue;
            }

            normalized.TryGetValue(request.Device, out var existing);
            normalized[request.Device] = checked(existing + request.Items);
        }

        return normalized;
    }

    private DeviceInferenceReservationLedger GetOrCreateDeviceInferenceReservationLedger(
        DeviceId device) =>
        _deviceInferenceReservationLedgers.GetOrAdd(
            device,
            static key => new DeviceInferenceReservationLedger(key));

    private DeviceInferenceReservationLedger GetExistingDeviceInferenceReservationLedger(
        DeviceId device) =>
        _deviceInferenceReservationLedgers.TryGetValue(device, out var ledger)
            ? ledger
            : throw new InvalidOperationException(
                $"Device inference reservation ledger does not exist for {device}.");

    private static IDisposable EnterInferenceReservationLedgerLocks(
        IReadOnlyList<InferenceReservationLedgerMutation> entries)
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

            return new InferenceReservationLedgerLockLease(ledgers);
        }
        catch
        {
            ReleaseInferenceReservationLedgerLocks(ledgers, acquiredCount);
            throw;
        }
    }

    private static void ReleaseInferenceReservationLedgerLocks(
        IReadOnlyList<DeviceInferenceReservationLedger> ledgers,
        int count)
    {
        for (var index = count - 1; index >= 0; index--)
        {
            Monitor.Exit(ledgers[index].Gate);
        }
    }

    private static async Task WaitForAnyInferenceReservationReleaseAsync(
        IReadOnlyList<Task<long>> signals,
        CancellationToken cancellationToken)
    {
        var completed = await Task.WhenAny(signals)
            .WaitAsync(cancellationToken)
            .ConfigureAwait(false);
        await completed.ConfigureAwait(false);
    }

    private void ReleaseDeviceInferenceReservation(DeviceId device, int items)
    {
        var ledger = GetExistingDeviceInferenceReservationLedger(device);
        TaskCompletionSource<long>? releasedSignal;
        long releaseVersion;

        lock (ledger.Gate)
        {
            if (ledger.ReservedItems < items)
            {
                throw new InvalidOperationException(
                    $"Device inference reservation ledger underflow for {ledger.Device}: " +
                    $"reserved={ledger.ReservedItems}, releasing={items}.");
            }

            ledger.ReservedItems -= items;
            releaseVersion = checked(ledger.ReleaseVersion + 1);
            ledger.ReleaseVersion = releaseVersion;
            releasedSignal = ledger.ReleaseSignal;
            ledger.ReleaseSignal = null;
        }

        releasedSignal?.TrySetResult(releaseVersion);
    }

    private void ReleaseDeviceInferenceReservations(
        IReadOnlyDictionary<DeviceId, int> reservations)
    {
        var entries = reservations
            .OrderBy(static pair => pair.Key.Value, StringComparer.Ordinal)
            .Select(pair => new InferenceReservationLedgerMutation(
                GetExistingDeviceInferenceReservationLedger(pair.Key),
                pair.Value,
                int.MaxValue))
            .ToArray();
        var releasedSignals = new List<(TaskCompletionSource<long> Signal, long Version)>(
            entries.Length);

        using (EnterInferenceReservationLedgerLocks(entries))
        {
            var remainingItems = new int[entries.Length];
            var releaseVersions = new long[entries.Length];
            for (var index = 0; index < entries.Length; index++)
            {
                var entry = entries[index];
                if (entry.Ledger.ReservedItems < entry.Items)
                {
                    throw new InvalidOperationException(
                        $"Device inference reservation ledger underflow for {entry.Ledger.Device}: " +
                        $"reserved={entry.Ledger.ReservedItems}, releasing={entry.Items}.");
                }

                remainingItems[index] = entry.Ledger.ReservedItems - entry.Items;
                releaseVersions[index] = checked(entry.Ledger.ReleaseVersion + 1);
            }

            for (var index = 0; index < entries.Length; index++)
            {
                var ledger = entries[index].Ledger;
                ledger.ReservedItems = remainingItems[index];
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

    private void DisposeDeviceInferenceReservationWaiters()
    {
        var signals = new List<TaskCompletionSource<long>>();
        foreach (var ledger in _deviceInferenceReservationLedgers.Values)
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

    private static TaskCompletionSource<long> CreateInferenceReservationReleaseSignal() =>
        new(TaskCreationOptions.RunContinuationsAsynchronously);

    private sealed class DeviceInferenceReservationLedger(DeviceId device)
    {
        public DeviceId Device { get; } = device;
        public object Gate { get; } = new();
        public int ReservedItems { get; set; }
        public long ReleaseVersion { get; set; }
        public TaskCompletionSource<long>? ReleaseSignal { get; set; }
    }

    private readonly record struct InferenceReservationLedgerMutation(
        DeviceInferenceReservationLedger Ledger,
        int Items,
        int Capacity);

    private sealed class InferenceReservationLedgerLockLease : IDisposable
    {
        private DeviceInferenceReservationLedger[]? _ledgers;

        public InferenceReservationLedgerLockLease(
            DeviceInferenceReservationLedger[] ledgers)
        {
            _ledgers = ledgers;
        }

        public void Dispose()
        {
            var ledgers = Interlocked.Exchange(ref _ledgers, null);
            if (ledgers is not null)
            {
                ReleaseInferenceReservationLedgerLocks(ledgers, ledgers.Length);
            }
        }
    }

    private sealed class SingleDeviceInferenceReservationLease :
        IRuntimeDeviceInferenceReservationLease
    {
        private ExecutionPlanExecutor? _owner;
        private readonly DeviceId _device;
        private readonly int _items;

        public SingleDeviceInferenceReservationLease(
            ExecutionPlanExecutor owner,
            DeviceId device,
            int items)
        {
            _owner = owner;
            _device = device;
            _items = items;
        }

        public void Release(DeviceId device)
        {
            if (!device.Equals(_device))
            {
                return;
            }

            ReleaseCore();
        }

        public void Dispose() => ReleaseCore();

        private void ReleaseCore()
        {
            Interlocked.Exchange(ref _owner, null)?
                .ReleaseDeviceInferenceReservation(_device, _items);
        }
    }

    private sealed class DeviceInferenceReservationLeaseSet :
        IRuntimeDeviceInferenceReservationLease
    {
        private readonly object _gate = new();
        private ExecutionPlanExecutor? _owner;
        private Dictionary<DeviceId, int>? _remaining;

        public DeviceInferenceReservationLeaseSet(
            ExecutionPlanExecutor owner,
            Dictionary<DeviceId, int> reservations)
        {
            if (reservations.Count == 0)
            {
                return;
            }

            _owner = owner;
            _remaining = reservations;
        }

        public void Release(DeviceId device)
        {
            lock (_gate)
            {
                if (_owner is null ||
                    _remaining is null ||
                    !_remaining.TryGetValue(device, out var items))
                {
                    return;
                }

                _owner.ReleaseDeviceInferenceReservation(device, items);
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

                _owner.ReleaseDeviceInferenceReservations(_remaining);
                _remaining = null;
                _owner = null;
            }
        }
    }

    private sealed class EmptyInferenceReservationLease :
        IRuntimeDeviceInferenceReservationLease
    {
        public static EmptyInferenceReservationLease Instance { get; } = new();
        public void Release(DeviceId device)
        {
        }

        public void Dispose()
        {
        }
    }
}
