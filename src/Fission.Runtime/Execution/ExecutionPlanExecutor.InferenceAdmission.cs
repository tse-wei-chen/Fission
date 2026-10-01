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

/// <summary>
/// Runtime-scoped reservation ledger for inference item credits that have been
/// admitted by a scheduler but are not represented in the device actor's credit
/// gate yet. This closes the shared-Engine admission race without treating a
/// racy SemaphoreSlim availability snapshot as authoritative.
/// </summary>
public sealed partial class ExecutionPlanExecutor
{
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

        var ledgers = devices
            .Distinct()
            .OrderBy(static device => device.Value, StringComparer.Ordinal)
            .Select(device => _deviceInferenceReservationLedgers.TryGetValue(device, out var ledger)
                ? ledger
                : null)
            .Where(static ledger => ledger is not null)
            .Cast<DeviceInferenceReservationLedger>()
            .ToArray();
        return SnapshotDeviceInferenceReservationState(ledgers);
    }

    internal bool TryReserveDeviceInference(
        IReadOnlyList<RuntimeDeviceInferenceReservationRequest> requests,
        out IDisposable reservation)
    {
        ArgumentNullException.ThrowIfNull(requests);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

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

        reservation = new DeviceInferenceReservationLease(this, normalized);
        return true;
    }

    private static RuntimeDeviceInferenceReservationState
        SnapshotDeviceInferenceReservationState(
            IReadOnlyList<DeviceInferenceReservationLedger> ledgers)
    {
        var reservations = new List<RuntimeDeviceInferenceReservationSnapshot>(ledgers.Count);
        foreach (var ledger in ledgers)
        {
            lock (ledger.Gate)
            {
                if (ledger.ReservedItems == 0)
                {
                    continue;
                }

                reservations.Add(new RuntimeDeviceInferenceReservationSnapshot(
                    ledger.Device,
                    ledger.ReservedItems,
                    ledger.ReleaseVersion));
            }
        }

        return new RuntimeDeviceInferenceReservationState(reservations);
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
                entries[index].Ledger.ReservedItems = remainingItems[index];
                entries[index].Ledger.ReleaseVersion = releaseVersions[index];
            }
        }
    }

    private sealed class DeviceInferenceReservationLedger(DeviceId device)
    {
        public DeviceId Device { get; } = device;
        public object Gate { get; } = new();
        public int ReservedItems { get; set; }
        public long ReleaseVersion { get; set; }
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

    private sealed class DeviceInferenceReservationLease : IDisposable
    {
        private ExecutionPlanExecutor? _owner;
        private readonly IReadOnlyDictionary<DeviceId, int> _reservations;

        public DeviceInferenceReservationLease(
            ExecutionPlanExecutor owner,
            IReadOnlyDictionary<DeviceId, int> reservations)
        {
            _owner = owner;
            _reservations = reservations;
        }

        public void Dispose()
        {
            Interlocked.Exchange(ref _owner, null)?
                .ReleaseDeviceInferenceReservations(_reservations);
        }
    }

    private sealed class EmptyInferenceReservationLease : IDisposable
    {
        public static EmptyInferenceReservationLease Instance { get; } = new();
        public void Dispose()
        {
        }
    }
}
