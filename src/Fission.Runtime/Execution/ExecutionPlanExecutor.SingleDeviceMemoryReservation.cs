using Fission.Abstractions;

namespace Fission.Runtime.Execution;

public sealed partial class ExecutionPlanExecutor
{
    private IRuntimeDeviceMemoryReservationLease ReserveSingleDeviceMemoryByDevice(
        RuntimeDeviceMemoryReservationRequest request)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentOutOfRangeException.ThrowIfNegative(request.Bytes);
        if (request.Bytes == 0)
        {
            return EmptyDeviceMemoryReservationLease.Instance;
        }

        var ledger = GetOrCreateDeviceMemoryReservationLedger(request.Device);
        lock (ledger.Gate)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
            ledger.ReservedBytes = checked(ledger.ReservedBytes + request.Bytes);
        }

        return new SingleDeviceMemoryReservationLease(
            this,
            request.Device,
            request.Bytes);
    }

    private sealed class SingleDeviceMemoryReservationLease :
        IRuntimeDeviceMemoryReservationLease
    {
        private ExecutionPlanExecutor? _owner;
        private readonly DeviceId _device;
        private readonly long _bytes;

        public SingleDeviceMemoryReservationLease(
            ExecutionPlanExecutor owner,
            DeviceId device,
            long bytes)
        {
            _owner = owner;
            _device = device;
            _bytes = bytes;
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
                .ReleaseDeviceMemoryReservation(_device, _bytes);
        }
    }

    private sealed class EmptyDeviceMemoryReservationLease :
        IRuntimeDeviceMemoryReservationLease
    {
        public static EmptyDeviceMemoryReservationLease Instance { get; } = new();

        public void Release(DeviceId device)
        {
        }

        public void Dispose()
        {
        }
    }
}
