using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

/// <summary>
/// Execution-provider session factories used by host composition. Keeping these
/// in the backend project prevents serving layers from taking a direct dependency
/// on ONNX Runtime provider APIs.
/// </summary>
public static class OnnxRuntimeSessionOptions
{
    public static Func<SessionOptions> Cuda(
        int deviceId,
        string? profileOutputPathPrefix = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceId);
        if (profileOutputPathPrefix is not null &&
            string.IsNullOrWhiteSpace(profileOutputPathPrefix))
        {
            throw new ArgumentException(
                "ONNX Runtime profile path prefix must be non-empty when specified.",
                nameof(profileOutputPathPrefix));
        }

        return () =>
        {
            var options = new SessionOptions();
            try
            {
                options.AppendExecutionProvider_CUDA(deviceId);
                if (profileOutputPathPrefix is not null)
                {
                    options.ProfileOutputPathPrefix = profileOutputPathPrefix;
                    options.EnableProfiling = true;
                }

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
