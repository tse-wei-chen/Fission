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
        (_prefillWorkPool.Count, _decodeWorkPool.Count);

    private PendingPrefill RentPrefill(PrefillItem item)
    {
        var work = TryRentPrefill() ?? CreatePrefill();
        work.Initialize(this, item);
        return work;
    }

    private PendingDecode RentDecode(DecodeItem item)
    {
        var work = TryRentDecode() ?? CreateDecode();
        work.Initialize(this, item);
        return work;
    }

    private PendingPrefill? TryRentPrefill()
    {
        var spinner = new SpinWait();
        while (true)
        {
            if (_prefillWorkPool.TryDequeue(out var work))
            {
                Interlocked.Decrement(ref _pooledPrefillCount);
                return work;
            }

            if (Volatile.Read(ref _pooledPrefillCount) == 0 || spinner.NextSpinWillYield)
            {
                return null;
            }

            // Returners reserve the bounded pool count before publishing into the
            // concurrent queue. Spin only through that very small publication gap;
            // if the owner was descheduled, allocate rather than blocking submitters.
            spinner.SpinOnce();
        }
    }

    private PendingDecode? TryRentDecode()
    {
        var spinner = new SpinWait();
        while (true)
        {
            if (_decodeWorkPool.TryDequeue(out var work))
            {
                Interlocked.Decrement(ref _pooledDecodeCount);
                return work;
            }

            if (Volatile.Read(ref _pooledDecodeCount) == 0 || spinner.NextSpinWillYield)
            {
                return null;
            }

            spinner.SpinOnce();
        }
    }

    private PendingPrefill CreatePrefill()
    {
        Interlocked.Increment(ref _inferenceWorkCreatedCount);
        return new PendingPrefill();
    }

    private PendingDecode CreateDecode()
    {
        Interlocked.Increment(ref _inferenceWorkCreatedCount);
        return new PendingDecode();
    }

    private void ReturnInferenceWorkToPool(PendingInference work)
    {
        switch (work)
        {
            case PendingPrefill prefill:
                prefill.ClearItemForPool();
                if (TryReservePoolSlot(ref _pooledPrefillCount))
                {
                    _prefillWorkPool.Enqueue(prefill);
                }
                break;

            case PendingDecode decode:
                decode.ClearItemForPool();
                if (TryReservePoolSlot(ref _pooledDecodeCount))
                {
                    _decodeWorkPool.Enqueue(decode);
                }
                break;

            default:
                throw new InvalidOperationException(
                    $"Unsupported pooled inference work type {work.GetType().Name}.");
        }
    }

    private bool TryReservePoolSlot(ref int count)
    {
        while (true)
        {
            var current = Volatile.Read(ref count);
            if (current >= InferenceCapacity)
            {
                return false;
            }

            if (Interlocked.CompareExchange(ref count, current + 1, current) == current)
            {
                return true;
            }
        }
    }
}
