using System.Runtime.CompilerServices;
using Fission.Abstractions.Execution;
using Fission.Runtime.Execution;

internal static class CompositeDeviceMemorySpecs
{
    [ModuleInitializer]
    internal static void Run()
    {
        var fixedResidency = new PressureOnlySource(
            activeBytes: 100,
            reclaimableBytes: 50,
            peakReservedBytes: 200);
        var largeCache = new ReclaimableSource(
            activeBytes: 20,
            reclaimableBytes: 300,
            peakReservedBytes: 400);
        var smallCache = new ReclaimableSource(
            activeBytes: 30,
            reclaimableBytes: 100,
            peakReservedBytes: 200);

        var composite = new CompositeInferenceDeviceMemoryController(
            fixedResidency,
            largeCache,
            smallCache);

        Require(
            composite.TryGetDeviceMemoryPressure(out var initial),
            "Composite pressure should be available when every component reports pressure.");
        Require(
            initial.ActiveBytes == 150 &&
            initial.ReclaimableBytes == 450 &&
            initial.ReservedBytes == 600 &&
            initial.PeakReservedBytes == 800,
            "Composite pressure must sum active, reclaimable, reserved, and component peak bytes.");

        var partial = composite.ReclaimDeviceMemoryAsync(targetReclaimableBytes: 300)
            .AsTask()
            .GetAwaiter()
            .GetResult();
        Require(
            partial.ReleasedBytes == 150 &&
            partial.ReclaimableBytes == 300 &&
            partial.ReservedBytes == 450,
            "Composite partial reclaim must stop at the requested aggregate reclaimable target.");
        Require(
            largeCache.ReclaimCalls == 1 &&
            largeCache.LastTargetReclaimableBytes == 150 &&
            smallCache.ReclaimCalls == 0,
            "Composite reclaim should drain the largest reclaimable component first and avoid unnecessary reclaimers.");

        var full = composite.ReclaimDeviceMemoryAsync(targetReclaimableBytes: 0)
            .AsTask()
            .GetAwaiter()
            .GetResult();
        Require(
            full.ReleasedBytes == 250 &&
            full.ReclaimableBytes == 50 &&
            full.ReservedBytes == 200,
            "Composite reclaim is best effort: reclaimable residency without a reclaimer must remain charged.");
        Require(
            largeCache.ReclaimCalls == 2 && smallCache.ReclaimCalls == 1,
            "Full composite reclaim must visit every reclaimable component that can release memory.");

        var unavailable = new CompositeInferenceDeviceMemoryController(
            new UnavailablePressureSource(),
            largeCache);
        Require(
            !unavailable.TryGetDeviceMemoryPressure(out _),
            "Composite pressure must return unavailable instead of undercounting when any component is unavailable.");

        var invalid = new CompositeInferenceDeviceMemoryController(
            new InvalidPressureSource());
        var rejectedInvalidPressure = false;
        try
        {
            _ = invalid.TryGetDeviceMemoryPressure(out _);
        }
        catch (InvalidOperationException)
        {
            rejectedInvalidPressure = true;
        }

        Require(
            rejectedInvalidPressure,
            "Composite pressure must reject a component that violates physical-memory accounting invariants.");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class PressureOnlySource : IInferenceDeviceMemoryPressureSource
    {
        private readonly InferenceDeviceMemoryPressure _pressure;

        public PressureOnlySource(
            long activeBytes,
            long reclaimableBytes,
            long peakReservedBytes)
        {
            _pressure = new InferenceDeviceMemoryPressure(
                ActiveBytes: activeBytes,
                ReclaimableBytes: reclaimableBytes,
                ReservedBytes: checked(activeBytes + reclaimableBytes),
                PeakReservedBytes: peakReservedBytes);
        }

        public bool TryGetDeviceMemoryPressure(
            out InferenceDeviceMemoryPressure pressure)
        {
            pressure = _pressure;
            return true;
        }
    }

    private sealed class ReclaimableSource :
        IInferenceDeviceMemoryPressureSource,
        IInferenceDeviceMemoryReclaimer
    {
        private readonly long _activeBytes;
        private readonly long _peakReservedBytes;
        private long _reclaimableBytes;

        public ReclaimableSource(
            long activeBytes,
            long reclaimableBytes,
            long peakReservedBytes)
        {
            _activeBytes = activeBytes;
            _reclaimableBytes = reclaimableBytes;
            _peakReservedBytes = peakReservedBytes;
        }

        public int ReclaimCalls { get; private set; }
        public long LastTargetReclaimableBytes { get; private set; } = -1;

        public bool TryGetDeviceMemoryPressure(
            out InferenceDeviceMemoryPressure pressure)
        {
            pressure = new InferenceDeviceMemoryPressure(
                ActiveBytes: _activeBytes,
                ReclaimableBytes: _reclaimableBytes,
                ReservedBytes: checked(_activeBytes + _reclaimableBytes),
                PeakReservedBytes: _peakReservedBytes);
            return true;
        }

        public ValueTask<InferenceDeviceMemoryReclaimResult> ReclaimDeviceMemoryAsync(
            long targetReclaimableBytes,
            CancellationToken cancellationToken = default)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(targetReclaimableBytes);
            cancellationToken.ThrowIfCancellationRequested();

            var retainedBytes = Math.Min(_reclaimableBytes, targetReclaimableBytes);
            var releasedBytes = _reclaimableBytes - retainedBytes;
            _reclaimableBytes = retainedBytes;
            ReclaimCalls++;
            LastTargetReclaimableBytes = targetReclaimableBytes;

            return ValueTask.FromResult(new InferenceDeviceMemoryReclaimResult(
                ReleasedBytes: releasedBytes,
                ReclaimableBytes: retainedBytes,
                ReservedBytes: checked(_activeBytes + retainedBytes)));
        }
    }

    private sealed class UnavailablePressureSource :
        IInferenceDeviceMemoryPressureSource
    {
        public bool TryGetDeviceMemoryPressure(
            out InferenceDeviceMemoryPressure pressure)
        {
            pressure = default;
            return false;
        }
    }

    private sealed class InvalidPressureSource :
        IInferenceDeviceMemoryPressureSource
    {
        public bool TryGetDeviceMemoryPressure(
            out InferenceDeviceMemoryPressure pressure)
        {
            pressure = new InferenceDeviceMemoryPressure(
                ActiveBytes: 1,
                ReclaimableBytes: 1,
                ReservedBytes: 1,
                PeakReservedBytes: 1);
            return true;
        }
    }
}
