namespace Fission.Scheduler

open Fission.Abstractions.Scheduling

/// C#-friendly adapter that keeps policy implementation in F# while exposing
/// stable DTO contracts to the orchestration/data-plane layers.
type SchedulingKernel() =
    let toPhase (phase: SchedulingPhase) =
        match phase with
        | SchedulingPhase.Waiting -> Waiting
        | SchedulingPhase.Prefilling -> Prefilling
        | SchedulingPhase.Decoding -> Decoding
        | SchedulingPhase.Suspended -> Suspended
        | SchedulingPhase.Finished -> Finished
        | SchedulingPhase.Cancelled -> Cancelled
        | _ -> invalidArg "phase" $"Unsupported scheduling phase {phase}."

    let toCandidate (candidate: SchedulingCandidate) : ReadySequence =
        { SequenceId = candidate.SequenceId
          Phase = toPhase candidate.Phase
          Deadline =
            if candidate.Deadline.HasValue then
                Some candidate.Deadline.Value
            else
                None
          EnqueuedAt = candidate.EnqueuedAt
          TokenDemand = candidate.TokenDemand
          Position = candidate.Position
          TokensPerKvPage = candidate.TokensPerKvPage
          Priority = candidate.Priority
          KvBytesPerToken = candidate.KvBytesPerToken }

    let toDeferralReason reason =
        match reason with
        | NotRunnable -> SchedulingDeferralReason.NotRunnable
        | TokenBudget -> SchedulingDeferralReason.TokenBudget
        | KvBudget -> SchedulingDeferralReason.KvBudget
        | KvByteBudget -> SchedulingDeferralReason.KvByteBudget
        | BatchSequenceBudget -> SchedulingDeferralReason.BatchSequenceBudget

    let toRejectionReason reason =
        match reason with
        | InvalidTokenDemand -> SchedulingRejectionReason.InvalidTokenDemand
        | InvalidPosition -> SchedulingRejectionReason.InvalidPosition
        | InvalidKvPageSize -> SchedulingRejectionReason.InvalidKvPageSize
        | InvalidDecodeQuantum -> SchedulingRejectionReason.InvalidDecodeQuantum
        | InvalidKvPageDemand -> SchedulingRejectionReason.InvalidKvPageDemand
        | InvalidKvBytesPerToken -> SchedulingRejectionReason.InvalidKvBytesPerToken
        | TokenDemandExceedsBatchCapacity -> SchedulingRejectionReason.TokenDemandExceedsBatchCapacity
        | KvDemandExceedsCapacity -> SchedulingRejectionReason.KvDemandExceedsCapacity

    interface ISchedulingKernel with
        member _.Schedule(scheduleId, now, budget, policy, candidates) =
            let resourceBudget : ResourceBudget =
                { MaxBatchTokens = budget.MaxBatchTokens
                  AvailableKvPages = budget.AvailableKvPages
                  MaxBatchSequences = budget.MaxBatchSequences
                  AvailableKvBytes = budget.AvailableKvBytes }

            let schedulingPolicy : SchedulingPolicy =
                { DecodeTokenReserve = policy.DecodeTokenReserve
                  MaxPrefillChunkTokens = policy.MaxPrefillChunkTokens
                  DeadlineUrgencyWindow = policy.DeadlineUrgencyWindow }

            let decision =
                candidates
                |> Seq.map toCandidate
                |> Seq.toList
                |> Scheduler.scheduleAt now resourceBudget schedulingPolicy

            let batch = ScheduleCompiler.compile scheduleId decision

            let deferred =
                decision.Deferred
                |> List.map (fun item ->
                    SchedulingDeferral(
                        item.Sequence.SequenceId,
                        toDeferralReason item.Reason))
                |> List.toArray

            let rejected =
                decision.Rejected
                |> List.map (fun item ->
                    SchedulingRejection(
                        item.Sequence.SequenceId,
                        toRejectionReason item.Reason))
                |> List.toArray

            SchedulingKernelResult(batch, deferred, rejected)
