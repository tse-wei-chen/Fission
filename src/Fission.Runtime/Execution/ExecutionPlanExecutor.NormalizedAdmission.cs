using Fission.Abstractions;

namespace Fission.Runtime.Execution;

public sealed partial class ExecutionPlanExecutor
{
    internal async ValueTask<IDisposable> EnterDeviceMemoryAdmissionAsync(
        DeviceId[] devices,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(devices);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        if (!IsNormalizedAdmissionDeviceArray(devices))
        {
            return await EnterDeviceMemoryAdmissionAsync(
                    (IReadOnlyList<DeviceId>)devices,
                    cancellationToken)
                .ConfigureAwait(false);
        }

        if (devices.Length == 0)
        {
            return EmptyAdmissionGateLease.Instance;
        }

        var acquired = new SemaphoreSlim[devices.Length];
        var acquiredCount = 0;
        try
        {
            foreach (var device in devices)
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

    internal RuntimeDeviceMemoryReservationState GetDeviceMemoryReservationState(
        DeviceId[] devices)
    {
        ArgumentNullException.ThrowIfNull(devices);

        if (!IsNormalizedAdmissionDeviceArray(devices))
        {
            return GetDeviceMemoryReservationState((IReadOnlyList<DeviceId>)devices);
        }

        List<RuntimeDeviceMemoryReservationSnapshot>? reservations = null;

        foreach (var device in devices)
        {
            if (!_deviceMemoryReservationLedgers.TryGetValue(device, out var ledger))
            {
                continue;
            }

            lock (ledger.Gate)
            {
                if (ledger.ReservedBytes == 0)
                {
                    continue;
                }

                (reservations ??= new List<RuntimeDeviceMemoryReservationSnapshot>(devices.Length))
                    .Add(new RuntimeDeviceMemoryReservationSnapshot(
                        ledger.Device,
                        ledger.ReservedBytes,
                        ledger.ReleaseVersion));
            }
        }

        return reservations is null
            ? EmptyDeviceMemoryReservationState
            : new RuntimeDeviceMemoryReservationState(reservations);
    }

    internal IReadOnlyList<RuntimeDeviceMemoryReservationSnapshot>
        GetDeviceMemoryReservations(DeviceId[] devices) =>
        GetDeviceMemoryReservationState(devices).Reservations;

    internal RuntimeDeviceInferenceReservationState GetDeviceInferenceReservationState(
        DeviceId[] devices)
    {
        ArgumentNullException.ThrowIfNull(devices);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

        if (!IsNormalizedAdmissionDeviceArray(devices))
        {
            return GetDeviceInferenceReservationState((IReadOnlyList<DeviceId>)devices);
        }

        List<RuntimeDeviceInferenceReservationSnapshot>? reservations = null;

        foreach (var device in devices)
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

                (reservations ??= new List<RuntimeDeviceInferenceReservationSnapshot>(devices.Length))
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

    internal IReadOnlyList<RuntimeDeviceMemoryPressure> GetDeviceMemoryPressureCore(
        DeviceId[] devices)
    {
        ArgumentNullException.ThrowIfNull(devices);

        return IsNormalizedAdmissionDeviceArray(devices)
            ? _devices.GetDeviceMemoryPressureNormalized(devices)
            : GetDeviceMemoryPressureCore((IReadOnlyList<DeviceId>)devices);
    }

    private static bool IsNormalizedAdmissionDeviceArray(DeviceId[] devices)
    {
        for (var index = 1; index < devices.Length; index++)
        {
            if (StringComparer.Ordinal.Compare(
                    devices[index - 1].Value,
                    devices[index].Value) >= 0)
            {
                return false;
            }
        }

        return true;
    }
}
