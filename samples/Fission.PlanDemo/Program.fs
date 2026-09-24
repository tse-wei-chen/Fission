open System
open System.Collections.Generic
open System.Threading
open Fission.Abstractions
open Fission.Plan
open Fission.Runtime.Backends
open Fission.Runtime.Execution

let run () =
    task {
        let model = ModelId("demo-model")
        let device = DeviceId("cpu:0")

        let! deviceExecutor =
            ContinuousBatchExecutor.CreateAsync(
                DeterministicBackend(device),
                128,
                8,
                CancellationToken.None).AsTask()

        use planExecutor = new ExecutionPlanExecutor(deviceExecutor)

        let sequence = SequenceId.New()
        let contract =
            { MaxTtft = None
              MaxTpot = None
              Priority = 100
              Class = SchedulingClass.Interactive }

        let logicalPlan =
            InferencePlan.create contract
                [ Prefill(sequence, model, 4)
                  SnapshotKv sequence
                  ForkKv(sequence, 2)
                  Decode(sequence, 3) ]

        let compiled = ExecutionPlanCompiler.compile logicalPlan
        let tokenBindings = Dictionary<SequenceId, ReadOnlyMemory<int>>()
        tokenBindings.Add(sequence, ReadOnlyMemory<int>([| 1; 2; 3; 4 |]))
        let bindings = ExecutionBindings(tokenBindings)

        let! result = planExecutor.ExecuteAsync(compiled, bindings).AsTask()

        if result.BackendResults.Count <> 4 then
            failwithf "Expected 4 backend results, got %d." result.BackendResults.Count

        if result.Snapshots.Count <> 1 then
            failwithf "Expected 1 snapshot, got %d." result.Snapshots.Count

        if result.Forks.Count <> 1 || result.Forks[0].Branches.Count <> 2 then
            failwith "Expected one fork operation with two branches."

        if planExecutor.SequenceCount <> 3 then
            failwithf "Expected parent plus two branches, got %d sequences." planExecutor.SequenceCount

        do! deviceExecutor.DisposeAsync().AsTask()

        printfn "Fission plan smoke run completed: plan=%O, backendResults=%d, sequences=%d"
            result.PlanId
            result.BackendResults.Count
            planExecutor.SequenceCount
    }

run().GetAwaiter().GetResult()
