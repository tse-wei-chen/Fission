using Microsoft.ML.OnnxRuntime;

namespace Fission.Backends.OnnxRuntime;

public abstract record OnnxRuntimeModelSource
{
    private OnnxRuntimeModelSource()
    {
    }

    public static OnnxRuntimeModelSource FromFile(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return new FileModelSource(path);
    }

    public static OnnxRuntimeModelSource FromBytes(ReadOnlyMemory<byte> model)
    {
        if (model.IsEmpty)
        {
            throw new ArgumentException("ONNX model bytes cannot be empty.", nameof(model));
        }

        return new ByteModelSource(model.ToArray());
    }

    internal abstract InferenceSession CreateSession(SessionOptions options);

    private sealed record FileModelSource(string Path) : OnnxRuntimeModelSource
    {
        internal override InferenceSession CreateSession(SessionOptions options) =>
            new(Path, options);
    }

    private sealed record ByteModelSource(byte[] Model) : OnnxRuntimeModelSource
    {
        internal override InferenceSession CreateSession(SessionOptions options) =>
            new(Model, options);
    }
}
