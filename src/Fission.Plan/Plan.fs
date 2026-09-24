namespace Fission.Plan

open System
open Fission.Abstractions

type SchedulingClass =
    | Interactive
    | Standard
    | Background

type InferenceContract =
    { MaxTtft: TimeSpan option
      MaxTpot: TimeSpan option
      Priority: int
      Class: SchedulingClass }

type InferOp =
    | Prefill of sequence: SequenceId * model: ModelId * tokenCount: int
    | PrefillChunk of sequence: SequenceId * model: ModelId * tokenCount: int * completesPrefill: bool
    | Decode of sequence: SequenceId * maxTokens: int
    | ForkKv of sequence: SequenceId * branches: int
    | SnapshotKv of sequence: SequenceId
    | RestoreKv of sequence: SequenceId * snapshot: KvSnapshotId
    | MigrateKv of sequence: SequenceId * target: DeviceId

type InferencePlan =
    { PlanId: Guid
      Contract: InferenceContract
      Ops: InferOp list }

[<RequireQualifiedAccess>]
module InferencePlan =
    let create contract ops =
        { PlanId = Guid.NewGuid()
          Contract = contract
          Ops = ops }
