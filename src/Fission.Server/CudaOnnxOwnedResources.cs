using Fission.Backends.OnnxRuntime;

namespace Fission.Server;

internal sealed class CudaOnnxOwnedResources : IDisposable
{
    private CudaDeviceBoundAsyncCopyEngine? _copyEngine;
    private IDisposable? _deviceMemory;
    private int _disposed;

    public CudaOnnxOwnedResources(
        CudaDeviceBoundAsyncCopyEngine copyEngine,
        IDisposable deviceMemory)
    {
        ArgumentNullException.ThrowIfNull(copyEngine);
        ArgumentNullException.ThrowIfNull(deviceMemory);
        _copyEngine = copyEngine;
        _deviceMemory = deviceMemory;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        List<Exception>? failures = null;
        var copyEngine = Interlocked.Exchange(ref _copyEngine, null);
        if (copyEngine is not null)
        {
            try
            {
                copyEngine.DisposeAsync().AsTask().GetAwaiter().GetResult();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }

        var deviceMemory = Interlocked.Exchange(ref _deviceMemory, null);
        if (deviceMemory is not null)
        {
            try
            {
                deviceMemory.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }

        if (failures is { Count: 1 })
        {
            throw failures[0];
        }

        if (failures is { Count: > 1 })
        {
            throw new AggregateException(
                "One or more CUDA serving resources failed to dispose.",
                failures);
        }
    }
}
