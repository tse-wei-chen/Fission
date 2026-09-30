using System.Runtime.CompilerServices;
using Fission.Abstractions.Execution;
using Fission.Backends.OnnxRuntime;

internal static class CudaDeviceMemoryPressureMonitorSpecs
{
    [ModuleInitializer]
    internal static void Run()
    {
        var cuda = new FakeCudaDeviceMemoryInfoApi
        {
            CurrentDevice = 0,
            FreeBytes = 400,
            TotalBytes = 1_000
        };
        var pool = new FakeKnownReclaimableSource(
            cuda,
            activeBytes: 100,
            reclaimableBytes: 200);
        var monitor = new CudaDeviceMemoryPressureMonitor(
            cuda,
            new CudaDeviceMemoryPressureMonitorOptions { DeviceId = 2 },
            pool);

        Require(
            monitor.TryGetDeviceMemoryPressure(out var first),
            "CUDA device-wide monitor must report pressure when cudaMemGetInfo succeeds.");
        Require(
            first.ActiveBytes == 400 &&
            first.ReclaimableBytes == 200 &&
            first.ReservedBytes == 600 &&
            first.PeakReservedBytes == 600,
            "Device-wide used memory must include all residency while known pool bytes are only reclassified as reclaimable.");
        Require(
            cuda.SetDeviceHistory.SequenceEqual(new[] { 2, 0 }) && cuda.CurrentDevice == 0,
            "cudaMemGetInfo must run on the configured device and restore the caller's previous CUDA device.");

        cuda.SetDeviceHistory.Clear();
        cuda.FreeBytes = 500;
        _ = monitor.TryGetDeviceMemoryPressure(out var lowerPressure);
        Require(
            lowerPressure.ActiveBytes == 300 &&
            lowerPressure.ReclaimableBytes == 200 &&
            lowerPressure.ReservedBytes == 500 &&
            lowerPressure.PeakReservedBytes == 600,
            "Device-wide pressure must preserve the high-water mark when current used bytes fall.");

        var reclaim = monitor.ReclaimDeviceMemoryAsync(targetReclaimableBytes: 50)
            .AsTask()
            .GetAwaiter()
            .GetResult();
        Require(
            pool.ReclaimCalls == 1 && pool.LastTargetReclaimableBytes == 50,
            "Device-wide monitor must forward reclaim targets to the same source used for reclaimable classification.");
        Require(
            reclaim.ReleasedBytes == 150 &&
            reclaim.ReclaimableBytes == 50 &&
            reclaim.ReservedBytes == 350,
            "Device-wide reclaim must re-sample cudaMemGetInfo after release instead of returning pool-only residency.");
        Require(
            monitor.TryGetDeviceMemoryPressure(out var afterReclaim) &&
            afterReclaim.ActiveBytes == 300 &&
            afterReclaim.ReclaimableBytes == 50 &&
            afterReclaim.ReservedBytes == 350 &&
            afterReclaim.PeakReservedBytes == 600,
            "Post-reclaim device pressure must keep non-pool device usage charged as active bytes.");

        pool.IsAvailable = false;
        Require(
            monitor.TryGetDeviceMemoryPressure(out var classifierUnavailable) &&
            classifierUnavailable.ActiveBytes == 350 &&
            classifierUnavailable.ReclaimableBytes == 0 &&
            classifierUnavailable.ReservedBytes == 350,
            "Unavailable optional classifier must make all device usage non-reclaimable rather than hiding residency.");

        pool.IsAvailable = true;
        pool.SetReclaimableBytes(100);
        cuda.FreeBytes = 950;
        Require(
            monitor.TryGetDeviceMemoryPressure(out var clamped) &&
            clamped.ActiveBytes == 0 &&
            clamped.ReclaimableBytes == 50 &&
            clamped.ReservedBytes == 50,
            "Known reclaimable classification must be clamped to the device-wide used-byte sample to tolerate snapshot races safely.");

        cuda.FreeBytes = 1_001;
        var rejectedImpossibleSample = false;
        try
        {
            _ = monitor.TryGetDeviceMemoryPressure(out _);
        }
        catch (InvalidOperationException)
        {
            rejectedImpossibleSample = true;
        }

        Require(
            rejectedImpossibleSample,
            "cudaMemGetInfo samples with free bytes larger than total bytes must be rejected.");

        cuda.FreeBytes = 400;
        cuda.MemGetInfoResult = 17;
        cuda.SetDeviceHistory.Clear();
        CudaRuntimeException? cudaFailure = null;
        try
        {
            _ = monitor.TryGetDeviceMemoryPressure(out _);
        }
        catch (CudaRuntimeException exception)
        {
            cudaFailure = exception;
        }

        Require(
            cudaFailure is not null &&
            cudaFailure.Operation == "cudaMemGetInfo" &&
            cudaFailure.ErrorCode == 17,
            "Native cudaMemGetInfo failures must preserve the CUDA operation and error code.");
        Require(
            cuda.CurrentDevice == 0 && cuda.SetDeviceHistory.SequenceEqual(new[] { 2, 0 }),
            "CUDA device scope must still restore the previous device when cudaMemGetInfo returns an error.");
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private sealed class FakeKnownReclaimableSource :
        IInferenceDeviceMemoryPressureSource,
        IInferenceDeviceMemoryReclaimer
    {
        private readonly FakeCudaDeviceMemoryInfoApi _cuda;
        private readonly long _activeBytes;
        private long _reclaimableBytes;
        private long _peakReservedBytes;

        public FakeKnownReclaimableSource(
            FakeCudaDeviceMemoryInfoApi cuda,
            long activeBytes,
            long reclaimableBytes)
        {
            _cuda = cuda;
            _activeBytes = activeBytes;
            _reclaimableBytes = reclaimableBytes;
            _peakReservedBytes = checked(activeBytes + reclaimableBytes);
        }

        public bool IsAvailable { get; set; } = true;
        public int ReclaimCalls { get; private set; }
        public long LastTargetReclaimableBytes { get; private set; } = -1;

        public void SetReclaimableBytes(long reclaimableBytes)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(reclaimableBytes);
            _reclaimableBytes = reclaimableBytes;
            _peakReservedBytes = Math.Max(
                _peakReservedBytes,
                checked(_activeBytes + reclaimableBytes));
        }

        public bool TryGetDeviceMemoryPressure(
            out InferenceDeviceMemoryPressure pressure)
        {
            if (!IsAvailable)
            {
                pressure = default;
                return false;
            }

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
            _cuda.FreeBytes = checked(_cuda.FreeBytes + (nuint)releasedBytes);
            ReclaimCalls++;
            LastTargetReclaimableBytes = targetReclaimableBytes;

            return ValueTask.FromResult(new InferenceDeviceMemoryReclaimResult(
                ReleasedBytes: releasedBytes,
                ReclaimableBytes: retainedBytes,
                ReservedBytes: checked(_activeBytes + retainedBytes)));
        }
    }

    private sealed class FakeCudaDeviceMemoryInfoApi : ICudaDeviceMemoryInfoApi
    {
        public int CurrentDevice { get; set; }
        public nuint FreeBytes { get; set; }
        public nuint TotalBytes { get; set; }
        public int MemGetInfoResult { get; set; }
        public List<int> SetDeviceHistory { get; } = new();

        public int MemGetInfo(out nuint freeBytes, out nuint totalBytes)
        {
            freeBytes = FreeBytes;
            totalBytes = TotalBytes;
            return MemGetInfoResult;
        }

        public int GetDevice(out int deviceId)
        {
            deviceId = CurrentDevice;
            return 0;
        }

        public int SetDevice(int deviceId)
        {
            SetDeviceHistory.Add(deviceId);
            CurrentDevice = deviceId;
            return 0;
        }

        public string? GetErrorString(int errorCode) => $"fake cuda error {errorCode}";
    }
}
