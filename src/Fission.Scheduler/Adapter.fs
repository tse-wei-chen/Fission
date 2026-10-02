namespace Fission.Scheduler

open Fission.Abstractions.Scheduling

/// C#-friendly adapter that keeps policy implementation in F# while exposing
/// stable DTO contracts to the orchestration/data-plane layers.
type SchedulingKernel() =
    let mapReadOnlyList
        (mapping: 'T -> 'U)
        (items: System.Collections.Generic.IReadOnlyList<'T>)
        =
        let mutable mapped = []
        for index = items.Count - 1 downto 0 do
            mapped <- mapping items[index] :: mapped
        mapped

    let mapReadOnlyListToArray
        (mapping: 'T -> 'U)
        (items: System.Collections.Generic.IReadOnlyList<'T>)
        =
        if items.Count = 0 then
            Array.empty<'U>
        else
            let mapped = Array.zeroCreate<'U> items.Count
            for index = 0 to items.Count - 1 do
                mapped[index] <- mapping items[index]
            mapped

    let mapListToArray
        (mapping: 'T -> 'U)
        (items: 'T list)
        =
        match items with
        | [] -> Array.empty<'U>
        | _ ->
            let mapped = Array.zeroCreate<'U> (List.length items)

            let rec fill index remaining =
                match remaining with
                | [] -> mapped
                | item :: tail ->
                    mapped[index] <- mapping item
                    fill (index + 1) tail

            fill 0 items

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
          KvBytesPerToken = candidate.KvBytesPerToken
          ExecutionDevice =
            if candidate.ExecutionDevice.HasValue then
                ValueSome candidate.ExecutionDevice.Value
            else
                ValueNone }

    let toDeferralReason reason =
        match reason with
        | NotRunnable -> SchedulingDeferralReason.NotRunnable
        | TokenBudget -> SchedulingDeferralReason.TokenBudget
        | KvBudget -> SchedulingDeferralReason.KvBudget
        | KvByteBudget -> SchedulingDeferralReason.KvByteBudget
        | TransientKvByteBudget -> SchedulingDeferralReason.TransientKvByteBudget
        | DeviceMemoryBudget -> SchedulingDeferralReason.DeviceMemoryBudget
        | BatchSequenceBudget -> SchedulingDeferralReason.BatchSequenceBudget
        | DeviceSequenceBudget -> SchedulingDeferralReason.DeviceSequenceBudget

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
            let availableDeviceBytes =
                match budget.DeviceMemory with
                | null -> []
                | deviceMemory ->
                    mapReadOnlyList
                        (fun (item: SchedulingDeviceMemoryBudget) ->
                            struct (item.Device, item.AvailableBytes))
                        deviceMemory

            let maxDeviceSequences =
                match budget.DeviceSequences with
                | null -> []
                | deviceSequences ->
                    mapReadOnlyList
                        (fun (item: SchedulingDeviceSequenceBudget) ->
                            struct (item.Device, item.MaxSequences))
                        deviceSequences

            let resourceBudget : ResourceBudget =
                { MaxBatchTokens = budget.MaxBatchTokens
                  AvailableKvPages = budget.AvailableKvPages
                  MaxBatchSequences = budget.MaxBatchSequences
                  AvailableKvBytes = budget.AvailableKvBytes
                  AvailableDeviceBytes = availableDeviceBytes
                  MaxDeviceSequences = maxDeviceSequences }

            let schedulingPolicy : SchedulingPolicy =
                { DecodeTokenReserve = policy.DecodeTokenReserve
                  MaxPrefillChunkTokens = policy.MaxPrefillChunkTokens
                  DeadlineUrgencyWindow = policy.DeadlineUrgencyWindow }

            let decision =
                candidates
                |> mapReadOnlyListToArray toCandidate
                |> Scheduler.scheduleArrayAt now resourceBudget schedulingPolicy

            let batch = ScheduleCompiler.compile scheduleId decision

            let deferred =
                decision.Deferred
                |> mapListToArray (fun (item: DeferredSequence) ->
                    SchedulingDeferral(
                        item.Sequence.SequenceId,
                        toDeferralReason item.Reason))

            let rejected =
                decision.Rejected
                |> mapListToArray (fun (item: RejectedSequence) ->
                    SchedulingRejection(
                        item.Sequence.SequenceId,
                        toRejectionReason item.Reason))

            SchedulingKernelResult(batch, deferred, rejected)
