namespace Fission.Scheduler

open System
open Fission.Abstractions.Scheduling

[<RequireQualifiedAccess>]
module ScheduleCompiler =
    let private compileItems (selected: ScheduledSequence list) =
        let items = Array.zeroCreate<ScheduledWorkItem> (List.length selected)

        let rec fill index remaining =
            match remaining with
            | [] -> items
            | selectedItem :: tail ->
                let kind, completesPrefill =
                    match selectedItem.Sequence.Phase with
                    | Prefilling ->
                        ScheduledWorkKind.Prefill,
                        selectedItem.TokenGrant >= selectedItem.Sequence.TokenDemand
                    | Decoding -> ScheduledWorkKind.Decode, false
                    | phase -> invalidOp $"Cannot compile non-runnable phase {phase}."

                items[index] <-
                    ScheduledWorkItem(
                        selectedItem.Sequence.SequenceId,
                        kind,
                        selectedItem.TokenGrant,
                        selectedItem.KvPageGrant,
                        selectedItem.Sequence.Priority,
                        completesPrefill,
                        selectedItem.KvByteGrant,
                        selectedItem.TransientKvByteGrant)

                fill (index + 1) tail

        fill 0 selected

    let compile (scheduleId: Guid) (decision: SchedulingDecision) =
        let items = compileItems decision.Selected

        ScheduledBatch(
            scheduleId,
            items,
            decision.ConsumedTokens,
            decision.ConsumedKvPages,
            decision.ConsumedKvBytes,
            decision.ConsumedTransientKvBytes)

    let compileNew (decision: SchedulingDecision) =
        compile (Guid.NewGuid()) decision
