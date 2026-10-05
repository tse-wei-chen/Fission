# Next milestone

Promote forked transactional KV branches into the normal continuous-batching lifecycle: let branch sequences enter engine scheduling as first-class runnable work, preserve their snapshot/fork ownership across admission and cancellation, and define the backend hook needed to materialize or lazily copy the physical KV tail when metadata-level COW is committed.
