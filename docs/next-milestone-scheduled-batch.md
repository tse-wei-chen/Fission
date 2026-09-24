# Next milestone after scheduled batches

After scheduler decisions can drive runtime work, the next execution milestone is physical KV-page reservation and chunked-prefill admission.

Planned direction:

- reserve/release KV pages against `KvPageGrant`
- split long prefills into scheduler-visible chunks
- feed actual runtime pressure back into F# scheduling inputs
- trace batch composition and resource grants for replay
- later compare alternative scheduling policies against the same recorded workload
