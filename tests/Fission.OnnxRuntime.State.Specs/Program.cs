using Fission.Abstractions;
using Fission.Backends.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;

static void Require(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

var parent = SequenceId.New();
var branchA = SequenceId.New();
var branchB = SequenceId.New();
var snapshot = KvSnapshotId.New();

var shared = new TrackingState("shared-v0");
var diverged = new TrackingState("branch-a-v1");

using (var store = new DecoderStateStore<TrackingState>())
{
    store.AddSequence(parent, shared);
    store.SnapshotSequence(parent, snapshot);
    store.ForkSequence(parent, new[] { branchA, branchB });

    Require(store.SequenceCount == 3, "Parent plus two branches must be registered.");
    Require(store.SnapshotCount == 1, "One decoder snapshot must be registered.");
    Require(ReferenceEquals(store.GetSequence(parent), shared), "Parent must retain the original state version.");
    Require(ReferenceEquals(store.GetSequence(branchA), shared), "Fork must share the parent state version without copying.");
    Require(ReferenceEquals(store.GetSequence(branchB), shared), "Sibling fork must share the parent state version without copying.");

    store.ReplaceSequence(branchA, diverged);
    Require(ReferenceEquals(store.GetSequence(branchA), diverged), "A branch decode must install a new immutable state version.");
    Require(ReferenceEquals(store.GetSequence(parent), shared), "Parent must not change when a branch diverges.");
    Require(ReferenceEquals(store.GetSequence(branchB), shared), "Sibling must not change when a branch diverges.");
    Require(shared.DisposeCount == 0, "Shared state must remain alive while parent/sibling/snapshot still reference it.");
    Require(diverged.DisposeCount == 0, "Diverged state must remain alive while branch A owns it.");

    store.RestoreSequence(branchA, snapshot);
    Require(ReferenceEquals(store.GetSequence(branchA), shared), "Restore must repoint the sequence to the snapshot state version.");
    Require(diverged.DisposeCount == 1, "Restoring away from an unshared state must dispose it exactly once.");

    Require(store.ReleaseSnapshot(snapshot), "Snapshot release must succeed.");
    Require(store.SnapshotCount == 0, "Snapshot map must be empty after release.");
    Require(shared.DisposeCount == 0, "Releasing one shared owner must not dispose the state while sequences remain.");

    Require(store.ReleaseSequence(parent), "Parent release must succeed.");
    Require(store.ReleaseSequence(branchB), "Sibling release must succeed.");
    Require(shared.DisposeCount == 0, "The last remaining branch must keep the shared state alive.");
    Require(store.ReleaseSequence(branchA), "Restored branch release must succeed.");
    Require(shared.DisposeCount == 1, "Shared state must dispose exactly once when its final owner is released.");
    Require(store.SequenceCount == 0, "All sequence references must be gone.");

    var orphan = new TrackingState("orphan");
    var missingReplaceRejected = false;
    try
    {
        store.ReplaceSequence(SequenceId.New(), orphan);
    }
    catch (KeyNotFoundException)
    {
        missingReplaceRejected = true;
    }

    Require(missingReplaceRejected, "Replacing a missing sequence must fail.");
    Require(orphan.DisposeCount == 0, "A failed replace must leave ownership with the caller.");
    orphan.Dispose();
    Require(orphan.DisposeCount == 1, "Caller must still be able to dispose state after failed ownership transfer.");

    var duplicateParent = SequenceId.New();
    var duplicateState = new TrackingState("duplicate-parent");
    store.AddSequence(duplicateParent, duplicateState);
    var duplicateBranch = SequenceId.New();
    var duplicateForkRejected = false;
    try
    {
        store.ForkSequence(duplicateParent, new[] { duplicateBranch, duplicateBranch });
    }
    catch (InvalidOperationException)
    {
        duplicateForkRejected = true;
    }

    Require(duplicateForkRejected, "Duplicate branch ids must be rejected before ownership changes.");
    Require(!store.TryGetSequence(duplicateBranch, out _), "Rejected fork must not publish partial branch state.");
    Require(duplicateState.DisposeCount == 0, "Rejected fork must not disturb the parent reference.");
    Require(store.ReleaseSequence(duplicateParent), "Duplicate-test parent release must succeed.");
    Require(duplicateState.DisposeCount == 1, "Duplicate-test parent state must dispose exactly once.");

    // Store disposal is the shutdown fallback: all sequence and snapshot owning
    // references are released, but a shared payload still disposes only once.
    var shutdownSequence = SequenceId.New();
    var shutdownBranch = SequenceId.New();
    var shutdownSnapshot = KvSnapshotId.New();
    var shutdownState = new TrackingState("shutdown-shared");
    store.AddSequence(shutdownSequence, shutdownState);
    store.SnapshotSequence(shutdownSequence, shutdownSnapshot);
    store.ForkSequence(shutdownSequence, new[] { shutdownBranch });
    Require(shutdownState.DisposeCount == 0, "Shutdown state must be alive before store disposal.");
}

Require(shared.DisposeCount == 1, "Already released shared state must not be disposed again by store shutdown.");
Require(diverged.DisposeCount == 1, "Already released divergent state must not be disposed again by store shutdown.");

// Real OrtValue ownership spec. The immutable payload owns two KV layers, while
// DecoderStateStore shares the payload object between sequence/snapshot/branch.
var ortParent = SequenceId.New();
var ortBranch = SequenceId.New();
var ortSnapshot = KvSnapshotId.New();
var key0Buffer = new[] { 1f, 2f };
var value0Buffer = new[] { 3f, 4f };
var key1Buffer = new[] { 5f, 6f };
var value1Buffer = new[] { 7f, 8f };
var key0 = OrtValue.CreateTensorValueFromMemory(key0Buffer, new long[] { 1, 1, 1, 2 });
var value0 = OrtValue.CreateTensorValueFromMemory(value0Buffer, new long[] { 1, 1, 1, 2 });
var key1 = OrtValue.CreateTensorValueFromMemory(key1Buffer, new long[] { 1, 1, 1, 2 });
var value1 = OrtValue.CreateTensorValueFromMemory(value1Buffer, new long[] { 1, 1, 1, 2 });
var ortState = new DecoderOrtState(
    position: 4,
    new[]
    {
        new DecoderOrtLayerState(key0, value0),
        new DecoderOrtLayerState(key1, value1)
    });

Require(ortState.Position == 4 && ortState.LayerCount == 2, "Owned ORT decoder state must expose position and layer count.");
Require(ortState.GetLayer(0).Key.GetTensorDataAsSpan<float>()[1] == 2f, "Layer key OrtValue must remain readable while state is alive.");
Require(ortState.GetLayer(1).Value.GetTensorDataAsSpan<float>()[1] == 8f, "Layer value OrtValue must remain readable while state is alive.");

using (var ortStore = new DecoderStateStore<DecoderOrtState>())
{
    ortStore.AddSequence(ortParent, ortState);
    ortStore.SnapshotSequence(ortParent, ortSnapshot);
    ortStore.ForkSequence(ortParent, new[] { ortBranch });
    Require(ReferenceEquals(ortStore.GetSequence(ortParent), ortState), "ORT parent must own the registered physical state.");
    Require(ReferenceEquals(ortStore.GetSequence(ortBranch), ortState), "ORT fork must share the same immutable physical state object.");

    Require(ortStore.ReleaseSequence(ortParent), "ORT parent release must succeed.");
    Require(ortStore.ReleaseSnapshot(ortSnapshot), "ORT snapshot release must succeed.");
    Require(!ortState.IsDisposed, "Remaining ORT branch owner must keep all native KV values alive.");
    Require(ortState.GetLayer(0).Value.GetTensorDataAsSpan<float>()[0] == 3f, "Shared KV OrtValue must still be readable before final owner release.");

    Require(ortStore.ReleaseSequence(ortBranch), "ORT branch release must succeed.");
    Require(ortState.IsDisposed, "Final ORT state owner release must dispose the physical KV payload.");
}

var disposedAccessRejected = false;
try
{
    _ = ortState.GetLayer(0);
}
catch (ObjectDisposedException)
{
    disposedAccessRejected = true;
}
Require(disposedAccessRejected, "Disposed ORT decoder state must reject layer access.");

// Constructor validation is pre-ownership. If the same OrtValue is supplied to
// two owned slots, construction fails and the caller remains responsible for it.
var duplicateBuffer = new[] { 9f };
var duplicateOrtValue = OrtValue.CreateTensorValueFromMemory(
    duplicateBuffer,
    new long[] { 1, 1, 1, 1 });
var duplicateOwnershipRejected = false;
try
{
    _ = new DecoderOrtState(
        position: 0,
        new[] { new DecoderOrtLayerState(duplicateOrtValue, duplicateOrtValue) });
}
catch (ArgumentException)
{
    duplicateOwnershipRejected = true;
}
Require(duplicateOwnershipRejected, "DecoderOrtState must reject duplicate OrtValue ownership slots.");
// Caller still owns the OrtValue because construction never succeeded.
duplicateOrtValue.Dispose();

Console.WriteLine(
    $"Fission ONNX decoder state specs passed: sharedDisposes={shared.DisposeCount}, " +
    $"divergedDisposes={diverged.DisposeCount}, ortLayers={ortState.LayerCount}, ortDisposed={ortState.IsDisposed}.");

sealed class TrackingState : IDisposable
{
    private int _disposeCount;

    public TrackingState(string name)
    {
        Name = name;
    }

    public string Name { get; }
    public int DisposeCount => Volatile.Read(ref _disposeCount);

    public void Dispose()
    {
        if (Interlocked.Increment(ref _disposeCount) != 1)
        {
            throw new InvalidOperationException(
                $"State '{Name}' was disposed more than once.");
        }
    }
}
