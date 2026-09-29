using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

internal static class OptimumLegacyCudaFloatDecoderBindingExtensions
{
    public static ValueTask InitializeAsync(
        this OptimumLegacyCudaFloatDecoderBinding binding,
        InferenceSession session,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(binding);
        ArgumentNullException.ThrowIfNull(session);
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.CompletedTask;
    }
}
