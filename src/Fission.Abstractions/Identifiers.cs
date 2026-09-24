namespace Fission.Abstractions;

public readonly record struct SequenceId(Guid Value)
{
    public static SequenceId New() => new(Guid.NewGuid());
    public override string ToString() => Value.ToString("N");
}

public readonly record struct KvSnapshotId(Guid Value)
{
    public static KvSnapshotId New() => new(Guid.NewGuid());
    public override string ToString() => Value.ToString("N");
}

public readonly record struct ModelId(string Value)
{
    public override string ToString() => Value;
}

public readonly record struct DeviceId(string Value)
{
    public override string ToString() => Value;
}
