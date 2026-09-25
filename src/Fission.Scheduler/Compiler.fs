namespace Fission.Scheduler

open System
open Fission.Abstractions.Scheduling

[<RequireQualifiedAccess>]
module ScheduleCompiler =
    let compile (scheduleId: Guid) (decision: SchedulingDecision) =
        let items =
            decision.Selected
            |> List.map (fun selected ->
                let kind, completesPrefill =
                    match selected.Sequence.Phase with
                    | Prefilling ->
                        ScheduledWorkKind.Prefill,
                        selected.TokenGrant >= selected.Sequence.TokenDemand
                    | Decoding -> ScheduledWorkKind.Decode, false
                    | phase -> invalidOp $"Cannot compile non-runnable phase {phase}."

                ScheduledWorkItem(
                    selected.Sequence.SequenceId,
                    kind,
                    selected.TokenGrant,
                    selected.KvPageGrant,
                    selected.Sequence.Priority,
                    completesPrefill,
                    selected.KvByteGrant,
                    selected.TransientKvByteGrant))
            |> List.toArray

        ScheduledBatch(
            scheduleId,
            items,
            decision.ConsumedTokens,
            decision.ConsumedKvPages,
            decision.ConsumedKvBytes,
            decision.ConsumedTransientKvBytes)

    let compileNew (decision: SchedulingDecision) =
        compile (Guid.NewGuid()) decision
