using System.Collections.Concurrent;
using Fission.Abstractions.Execution;

namespace Fission.Runtime.Execution;

public sealed partial class ContinuousBatchExecutor
{
    private readonly ConcurrentQueue<PendingPrefill> _prefillWorkPool = new();
    private readonly ConcurrentQueue<PendingDecode> _decodeWorkPool = new();
    private int _pooledPrefillCount;
    private int _pooledDecodeCount;
    private long _inferenceWorkCreatedCount;

    internal long InferenceWorkCreatedCount =>
        Volatile.Read(ref _inferenceWorkCreatedCount);

    internal (int Prefill, int Decode) InferenceWorkPoolCounts =>
        (Volatile.Read(ref _pooledPrefillCount), Volatile.Read(ref _pooledDecodeCount));

    private PendingPrefill RentPrefill(PrefillItem item)
    {
        if (!_prefillWorkPool.TryDequeue(out var work))
        {
            work = new PendingPrefill();
            Interlocked.Increment(ref _inferenceWorkCreatedCount);
        }
        else
        {
            Interlocked.Decrement(ref _pooledPrefillCount);
        }

        work.Initialize(this, item);
        return work;
    }

    private PendingDecode RentDecode(DecodeItem item)
    {
        if (!_decodeWorkPool.TryDequeue(out var work))
        {
            work = new PendingDecode();
            Interlocked.Increment(ref _inferenceWorkCreatedCount);
        }
        else
        {
            Interlocked.Decrement(ref _pooledDecodeCount);
        }

        work.Initialize(this, item);
        return work;
    }

    private void ReturnInferenceWorkToPool(PendingInference work)
    {
        switch (work)
        {
            case PendingPrefill prefill:
                prefill.ClearItemForPool();
                ReturnPrefillToPool(prefill);
                break;

            case PendingDecode decode:
                decode.ClearItemForPool();
                ReturnDecodeToPool(decode);
                break;

            default:
                throw new InvalidOperationException(
                    $"Unsupported pooled inference work type {work.GetType().Name}.");
        }
    }

    private void ReturnPrefillToPool(PendingPrefill work)
    {
        var count = Interlocked.Increment(ref _pooledPrefillCount);
        if (count <= InferenceCapacity)
        {
            _prefillWorkPool.Enqueue(work);
            return;
        }

        Interlocked.Decrement(ref _pooledPrefillCount);
    }

    private void ReturnDecodeToPool(PendingDecode work)
    {
        var count = Interlocked.Increment(ref _pooledDecodeCount);
        if (count <= InferenceCapacity)
        {
            _decodeWorkPool.Enqueue(work);
            return;
        }

        Interlocked.Decrement(ref _pooledDecodeCount);
    }
}
