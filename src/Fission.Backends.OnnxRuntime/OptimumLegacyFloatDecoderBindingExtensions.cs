using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

internal static class OptimumLegacyFloatDecoderBindingExtensions
{
    /// <summary>
    /// Bridges the default IDecoderOrtModelBinding lifecycle member for decorators
    /// that hold the concrete Optimum binding type. Default interface members are
    /// callable through the interface, but are not surfaced as concrete members.
    /// </summary>
    public static ValueTask InitializeAsync(
        this OptimumLegacyFloatDecoderBinding binding,
        InferenceSession session,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(binding);
        return ((IDecoderOrtModelBinding)binding).InitializeAsync(
            session,
            cancellationToken);
    }
}
