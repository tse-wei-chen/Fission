namespace Fission.Scheduler

open System
open Fission.Abstractions

type SequencePhase =
    | Waiting
    | Prefilling
    | Decoding
    | Suspended
    | Finished
    | Cancelled

type ReadySequence =
    { SequenceId: SequenceId
      Phase: SequencePhase
      Deadline: DateTimeOffset option
      KvPages: int
      Priority: int }

type ResourceBudget =
    { MaxBatchTokens: int
      AvailableKvPages: int }

type SchedulingDecision =
    { Selected: ReadySequence list
      Deferred: ReadySequence list
      ConsumedTokens: int }
