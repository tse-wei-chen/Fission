# Fission

Fission is an experimental high-concurrency LLM inference runtime for .NET.

The project explores a typed, stateful inference architecture where:

- **F#** expresses inference plans, scheduling policy, and optimization passes.
- **C#** hosts the runtime, sequence lifecycle, memory/KV management, and serving hot paths.
- Native GPU backends can be integrated behind stable runtime abstractions.

The long-term direction is an inference operating system with first-class support for forkable KV state, snapshots, migration, scheduling contracts, and replayable execution plans.

> Status: early bootstrap.
