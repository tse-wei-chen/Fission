using Fission.Abstractions;
using Fission.Abstractions.Scheduling;
using Fission.Runtime.Execution;

namespace Fission.Engine;

public sealed partial class InferenceEngine
{
    private DeviceMemoryBudgetSnapshot GetDeviceMemoryBudgets(DeviceId[] devices)
    {
        if (_options.MaxDeviceBytes is not { } maxDeviceBytes)
        {
            throw new InvalidOperationException(
                "Device-memory budgets require MaxDeviceBytes to be configured.");
        }

        var pressures = _runtime.GetDeviceMemoryPressureCore(devices);
        var reservationState = _runtime.GetDeviceMemoryReservationState(devices);
        var budgets = new SchedulingDeviceMemoryBudget[devices.Length];

        for (var index = 0; index < devices.Length; index++)
        {
            var device = devices[index];
            if (!TryGetDeviceMemoryPressure(pressures, device, out var pressure))
            {
                throw new InvalidOperationException(
                    $"MaxDeviceBytes requires physical memory pressure from execution device {device}.");
            }

            var physicalAvailableBytes = pressure.ReservedBytes >= maxDeviceBytes
                ? 0L
                : maxDeviceBytes - pressure.ReservedBytes;
            var outstandingReservationBytes = GetReservedDeviceMemoryBytes(
                reservationState.Reservations,
                device);
            var availableBytes = outstandingReservationBytes >= physicalAvailableBytes
                ? 0L
                : physicalAvailableBytes - outstandingReservationBytes;
            budgets[index] = new SchedulingDeviceMemoryBudget(device, availableBytes);
        }

        return new DeviceMemoryBudgetSnapshot(budgets, reservationState);
    }
}
