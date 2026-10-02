using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Execution-provider session factories used by host composition. Keeping these
/// in the backend project prevents serving layers from taking a direct dependency
/// on ONNX Runtime provider APIs.
/// </summary>
public static class OnnxRuntimeSessionOptions
{
    public static Func<SessionOptions> Cuda(int deviceId)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceId);

        return () =>
        {
            var options = new SessionOptions();
            try
            {
                options.AppendExecutionProvider_CUDA(deviceId);
                return options;
            }
            catch
            {
                options.Dispose();
                throw;
            }
        };
    }
}
