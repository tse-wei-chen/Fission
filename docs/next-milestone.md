# Next milestone

Define the backend-facing physical KV copy-on-write materialization contract. Engine-level forked branches now re-enter normal continuous batching as first-class runnable requests, and the logical KV page table already identifies shared partial-tail write pressure.

The next step is to carry that logical COW event across the execution boundary so stateful backends can materialize the writable tail before the first divergent token. The contract should let ONNX Runtime / CUDA adapters lazily clone only the live tail state (or remap an equivalent backend-owned page), preserve transaction and rollback semantics on failure, and expose enough physical-memory evidence for admission/qualification tests without leaking runtime refcounts or page IDs into the scheduler.
