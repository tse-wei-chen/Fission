# Next milestone

Implement the backend-specific physical KV tail materialization behind the new `InferenceKvWriteIntent` contract.

Runtime and scheduler semantics are now connected end-to-end: partial-tail sharing contributes a COW page to admission, the live runtime revalidates that demand before dispatch, and both generic and continuous-batching inference items carry the matching physical write geometry into the backend boundary without exposing runtime page ids or refcounts.

The next step is to make the ONNX Runtime / CUDA state path consume that intent with a genuinely page-local strategy where possible: lazily clone or remap only the live shared tail before the first divergent token, preserve the existing immutable causal frontier on allocation/copy failure, and add physical-memory evidence showing that fork divergence does not eagerly duplicate the whole sequence. Qualification should cover FP32 and FP16 CUDA paths and verify that subsequent writes to the now-private tail do not repeat materialization.
