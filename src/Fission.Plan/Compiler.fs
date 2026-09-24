namespace Fission.Plan

open Fission.Abstractions.Execution

[<RequireQualifiedAccess>]
module ExecutionPlanCompiler =
    let private compileOp (op: InferOp) : ExecutionStep =
        match op with
        | Prefill (sequence, model, tokenCount) ->
            PrefillExecutionStep(sequence, model, tokenCount) :> ExecutionStep
        | Decode (sequence, maxTokens) ->
            DecodeExecutionStep(sequence, maxTokens) :> ExecutionStep
        | ForkKv (sequence, branches) ->
            ForkKvExecutionStep(sequence, branches) :> ExecutionStep
        | SnapshotKv sequence ->
            SnapshotKvExecutionStep(sequence) :> ExecutionStep
        | RestoreKv (sequence, snapshot) ->
            RestoreKvExecutionStep(sequence, snapshot) :> ExecutionStep
        | MigrateKv (sequence, target) ->
            MigrateKvExecutionStep(sequence, target) :> ExecutionStep

    let compile (plan: InferencePlan) =
        let steps = plan.Ops |> List.map compileOp |> List.toArray
        CompiledExecutionPlan(plan.PlanId, plan.Contract.Priority, steps)
