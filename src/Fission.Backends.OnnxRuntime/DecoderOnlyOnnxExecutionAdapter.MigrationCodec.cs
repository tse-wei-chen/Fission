namespace Fission.Backends.OnnxRuntime;

public sealed partial class DecoderOnlyOnnxExecutionAdapter
{
    private readonly IDecoderOrtHostStagingCodec? _migrationCodecOverride;

    /// <summary>
    /// Creates a decoder adapter whose model-execution binding and migration codec
    /// are separate components. The codec is caller-owned and must remain alive for
    /// the adapter lifetime. Keeping it separate preserves optional execution
    /// capabilities (batch decode, batch prefill, chunked prefill) on the original
    /// binding instead of hiding them behind a migration decorator.
    /// </summary>
    public DecoderOnlyOnnxExecutionAdapter(
        IDecoderOrtModelBinding binding,
        IDecoderOrtHostStagingCodec migrationCodec)
        : this(binding)
    {
        ArgumentNullException.ThrowIfNull(migrationCodec);
        _migrationCodecOverride = migrationCodec;
    }

    private IDecoderOrtHostStagingCodec? HostStagingCodec =>
        _migrationCodecOverride ?? _binding as IDecoderOrtHostStagingCodec;
}
