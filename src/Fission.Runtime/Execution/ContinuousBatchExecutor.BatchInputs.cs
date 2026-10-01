using System.Collections;
using Fission.Abstractions.Execution;

namespace Fission.Runtime.Execution;

public sealed partial class ContinuousBatchExecutor
{
    private ReusableBatchItems<PrefillItem>? _prefillBatchItems;
    private ReusableBatchItems<DecodeItem>? _decodeBatchItems;
    private PrefillBatch? _prefillBatch;
    private DecodeBatch? _decodeBatch;

    private PrefillBatch PreparePrefillBatch(
        IReadOnlyList<PendingInference> segment,
        int start,
        int count)
    {
        var items = _prefillBatchItems ??=
            new ReusableBatchItems<PrefillItem>(_maxBatchSize);
        var batch = _prefillBatch ??= new PrefillBatch(items);

        items.Begin(count);
        for (var offset = 0; offset < count; offset++)
        {
            items.Set(
                offset,
                ((PendingPrefill)segment[start + offset]).Item);
        }

        return batch;
    }

    private DecodeBatch PrepareDecodeBatch(
        IReadOnlyList<PendingInference> segment,
        int start,
        int count)
    {
        var items = _decodeBatchItems ??=
            new ReusableBatchItems<DecodeItem>(_maxBatchSize);
        var batch = _decodeBatch ??= new DecodeBatch(items);

        items.Begin(count);
        for (var offset = 0; offset < count; offset++)
        {
            items.Set(
                offset,
                ((PendingDecode)segment[start + offset]).Item);
        }

        return batch;
    }

    private void ClearPrefillBatch() => _prefillBatchItems?.Clear();

    private void ClearDecodeBatch() => _decodeBatchItems?.Clear();

    /// <summary>
    /// Actor-owned mutable storage presented to a backend as IReadOnlyList. The
    /// list is borrowed only for the duration of one backend ValueTask and is
    /// cleared before the next homogeneous micro-batch is prepared.
    /// </summary>
    private sealed class ReusableBatchItems<T> : IReadOnlyList<T>
    {
        private readonly T[] _buffer;
        private int _count;

        public ReusableBatchItems(int capacity)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
            _buffer = new T[capacity];
        }

        public int Count => _count;

        public T this[int index]
        {
            get
            {
                if ((uint)index >= (uint)_count)
                {
                    throw new ArgumentOutOfRangeException(nameof(index));
                }

                return _buffer[index];
            }
        }

        public void Begin(int count)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
            if (count > _buffer.Length)
            {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            if (_count != 0)
            {
                throw new InvalidOperationException(
                    "Reusable backend batch storage was prepared before the prior batch completed.");
            }

            _count = count;
        }

        public void Set(int index, T item)
        {
            if ((uint)index >= (uint)_count)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            _buffer[index] = item;
        }

        public void Clear()
        {
            Array.Clear(_buffer, 0, _count);
            _count = 0;
        }

        public Enumerator GetEnumerator() => new(this);

        IEnumerator<T> IEnumerable<T>.GetEnumerator() => GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public struct Enumerator : IEnumerator<T>
        {
            private readonly ReusableBatchItems<T> _owner;
            private int _index;

            internal Enumerator(ReusableBatchItems<T> owner)
            {
                _owner = owner;
                _index = -1;
            }

            public T Current => _owner._buffer[_index];

            object? IEnumerator.Current => Current;

            public bool MoveNext()
            {
                var next = _index + 1;
                if (next >= _owner._count)
                {
                    return false;
                }

                _index = next;
                return true;
            }

            public void Reset() => throw new NotSupportedException();

            public void Dispose()
            {
            }
        }
    }
}
