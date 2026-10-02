namespace Fission.Scheduler

open System
open Fission.Abstractions.Scheduling

[<RequireQualifiedAccess>]
module ScheduleCompiler =
    let private toWorkItem (selectedItem: ScheduledSequence) =
        let kind, completesPrefill =
            match selectedItem.Sequence.Phase with
            | SchedulingPhase.Prefilling ->
                ScheduledWorkKind.Prefill,
                selectedItem.TokenGrant >= selectedItem.Sequence.TokenDemand
            | SchedulingPhase.Decoding -> ScheduledWorkKind.Decode, false
            | phase -> invalidOp $"Cannot compile non-runnable phase {phase}."

        ScheduledWorkItem(
            selectedItem.Sequence.SequenceId,
            kind,
            selectedItem.TokenGrant,
            selectedItem.KvPageGrant,
            selectedItem.Sequence.Priority,
            completesPrefill,
            selectedItem.KvByteGrant,
            selectedItem.TransientKvByteGrant)

    let private compileItems (selected: ScheduledSequence list) =
        let items = Array.zeroCreate<ScheduledWorkItem> (List.length selected)

        let rec fill index remaining =
            match remaining with
            | [] -> items
            | (selectedItem: ScheduledSequence) :: tail ->
                items[index] <- toWorkItem selectedItem
                fill (index + 1) tail

        fill 0 selected

    let private compileReverseItems (selectedRev: ScheduledSequence list) =
        match selectedRev with
        | [] -> Array.empty<ScheduledWorkItem>
        | _ ->
            let items =
                Array.zeroCreate<ScheduledWorkItem> (List.length selectedRev)

            let rec fill index remaining =
                match remaining with
                | [] -> items
                | (selectedItem: ScheduledSequence) :: tail ->
                    items[index] <- toWorkItem selectedItem
                    fill (index - 1) tail

            fill (items.Length - 1) selectedRev

    let compile (scheduleId: Guid) (decision: SchedulingDecision) =
        let items = compileItems decision.Selected

        ScheduledBatch(
            scheduleId,
            items,
            decision.ConsumedTokens,
            decision.ConsumedKvPages,
            decision.ConsumedKvBytes,
            decision.ConsumedTransientKvBytes)

    let internal compileRaw
        (scheduleId: Guid)
        (decision: RawSchedulingDecision)
        =
        let items = compileReverseItems decision.SelectedRev

        ScheduledBatch(
            scheduleId,
            items,
            decision.ConsumedTokens,
            decision.ConsumedKvPages,
            decision.ConsumedKvBytes,
            decision.ConsumedTransientKvBytes)

    let compileNew (decision: SchedulingDecision) =
        compile (Guid.NewGuid()) decision
