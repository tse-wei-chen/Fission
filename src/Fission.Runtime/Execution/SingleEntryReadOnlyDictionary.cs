using System.Collections;

namespace Fission.Runtime.Execution;

/// <summary>
/// Minimal one-entry IReadOnlyDictionary used where callers only need keyed
/// lookup but a full hash table would allocate bucket and entry arrays.
/// </summary>
internal sealed class SingleEntryReadOnlyDictionary<TKey, TValue> :
    IReadOnlyDictionary<TKey, TValue>
    where TKey : notnull
{
    private readonly TKey _key;
    private readonly TValue _value;

    internal SingleEntryReadOnlyDictionary(TKey key, TValue value)
    {
        _key = key;
        _value = value;
    }

    public int Count => 1;

    public IEnumerable<TKey> Keys
    {
        get
        {
            yield return _key;
        }
    }

    public IEnumerable<TValue> Values
    {
        get
        {
            yield return _value;
        }
    }

    public TValue this[TKey key] =>
        TryGetValue(key, out var value)
            ? value
            : throw new KeyNotFoundException();

    public bool ContainsKey(TKey key) =>
        EqualityComparer<TKey>.Default.Equals(_key, key);

    public bool TryGetValue(TKey key, out TValue value)
    {
        if (ContainsKey(key))
        {
            value = _value;
            return true;
        }

        value = default!;
        return false;
    }

    public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
    {
        yield return new KeyValuePair<TKey, TValue>(_key, _value);
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
