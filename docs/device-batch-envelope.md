# Device Batch Envelope

Scheduler-selected inference work is submitted to the device actor as one ordered envelope per scheduling batch. The envelope preserves scheduler item order and is executed as contiguous prefill/decode segments. Items inside the envelope are not interleaved with unrelated queue work.

The runtime still commits metadata after each backend segment completes. This is deliberate: a mixed prefill/decode envelope is not treated as an all-or-nothing transaction because a later backend segment may fail after an earlier segment has already mutated backend state.
