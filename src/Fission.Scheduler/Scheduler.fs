namespace Fission.Scheduler

open System

[<RequireQualifiedAccess>]
module Scheduler =
    let private tokenCost sequence =
        match sequence.Phase with
        | Prefilling -> 256
        | Decoding -> 1
        | _ -> 0

    let private deadlineTicks sequence =
        sequence.Deadline
        |> Option.map _.UtcTicks
        |> Option.defaultValue Int64.MaxValue

    let private orderingKey sequence =
        (-sequence.Priority, deadlineTicks sequence)

    let schedule budget sequences =
        let candidates =
            sequences
            |> List.filter (fun sequence ->
                sequence.Phase = Prefilling || sequence.Phase = Decoding)
            |> List.sortBy orderingKey

        let folder (selected, deferred, usedTokens, usedKvPages) sequence =
            let cost = tokenCost sequence
            let fitsTokens = usedTokens + cost <= budget.MaxBatchTokens
            let fitsKv = usedKvPages + sequence.KvPages <= budget.AvailableKvPages

            if fitsTokens && fitsKv then
                (sequence :: selected, deferred, usedTokens + cost, usedKvPages + sequence.KvPages)
            else
                (selected, sequence :: deferred, usedTokens, usedKvPages)

        let selected, deferred, usedTokens, _ =
            candidates
            |> List.fold folder ([], [], 0, 0)

        { Selected = List.rev selected
          Deferred = List.rev deferred
          ConsumedTokens = usedTokens }
