using Fission.Abstractions.Execution;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Backend-facing capability adapter over a pooled CUDA allocator. The allocator
/// remains caller-owned; this controller exposes only pressure and idle-cache
/// reclaim semantics to the generic runtime.
/// </summary>
public sealed class CudaDeviceMemoryPoolController :
    IInferenceDeviceMemoryPressureSource,
    IInferenceDeviceMemoryReclaimer
{
    private readonly CudaPooledDeviceMemoryAllocator _pool;

    public CudaDeviceMemoryPoolController(CudaPooledDeviceMemoryAllocator pool)
    {
        ArgumentNullException.ThrowIfNull(pool);
        _pool = pool;
    }

    public bool TryGetDeviceMemoryPressure(
        out InferenceDeviceMemoryPressure pressure) =>
        _pool.TryGetDeviceMemoryPressure(out pressure);

    public ValueTask<InferenceDeviceMemoryReclaimResult> ReclaimDeviceMemoryAsync(
        long targetReclaimableBytes,
        CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(targetReclaimableBytes);
        cancellationToken.ThrowIfCancellationRequested();

        var trim = _pool.TrimRetained(targetReclaimableBytes);
        var statistics = _pool.Statistics;
        return ValueTask.FromResult(new InferenceDeviceMemoryReclaimResult(
            ReleasedBytes: trim.ReleasedBytes,
            ReclaimableBytes: statistics.RetainedBytes,
            ReservedBytes: statistics.ReservedBytes));
    }
}
