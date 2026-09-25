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
      EnqueuedAt: DateTimeOffset
      TokenDemand: int
      Position: int
      TokensPerKvPage: int
      Priority: int
      KvBytesPerToken: int64 }

type ResourceBudget =
    { MaxBatchTokens: int
      AvailableKvPages: int
      MaxBatchSequences: int
      AvailableKvBytes: int64 }

type SchedulingPolicy =
    { DecodeTokenReserve: int
      MaxPrefillChunkTokens: int
      DeadlineUrgencyWindow: TimeSpan }

type AdmissionRejectionReason =
    | InvalidTokenDemand
    | InvalidPosition
    | InvalidKvPageSize
    | InvalidDecodeQuantum
    | InvalidKvPageDemand
    | InvalidKvBytesPerToken
    | TokenDemandExceedsBatchCapacity
    | KvDemandExceedsCapacity

type DeferredReason =
    | NotRunnable
    | TokenBudget
    | KvBudget
    | KvByteBudget
    | TransientKvByteBudget
    | BatchSequenceBudget

type ScheduledSequence =
    { Sequence: ReadySequence
      TokenGrant: int
      KvPageGrant: int
      KvByteGrant: int64
      TransientKvByteGrant: int64 }

type DeferredSequence =
    { Sequence: ReadySequence
      Reason: DeferredReason }

type RejectedSequence =
    { Sequence: ReadySequence
      Reason: AdmissionRejectionReason }

type SchedulingDecision =
    { Selected: ScheduledSequence list
      Deferred: DeferredSequence list
      Rejected: RejectedSequence list
      ConsumedTokens: int
      ConsumedKvPages: int
      ConsumedKvBytes: int64
      ConsumedTransientKvBytes: int64 }
