namespace Fission.Scheduler

open System

[<RequireQualifiedAccess>]
module Scheduler =
    type private SelectionState =
        { SelectedRev: ScheduledSequence list
          DeferredRev: DeferredSequence list
          UsedTokens: int
          UsedKvPages: int }

    let private isRunnable sequence =
        sequence.Phase = Prefilling || sequence.Phase = Decoding

    let private deadlineTicks sequence =
        sequence.Deadline
        |> Option.map _.UtcTicks
        |> Option.defaultValue Int64.MaxValue

    let private isUrgent now policy sequence =
        match sequence.Deadline with
        | Some deadline -> deadline <= now + policy.DeadlineUrgencyWindow
        | None -> false

    let private compareReady now policy left right =
        let urgentLeft = isUrgent now policy left
        let urgentRight = isUrgent now policy right

        let first = compare (if urgentLeft then 0 else 1) (if urgentRight then 0 else 1)
        if first <> 0 then
            first
        elif urgentLeft && urgentRight then
            let byDeadline = compare (deadlineTicks left) (deadlineTicks right)
            if byDeadline <> 0 then byDeadline
            else
                let byPriority = compare right.Priority left.Priority
                if byPriority <> 0 then byPriority
                else
                    let byArrival = compare left.EnqueuedAt.UtcTicks right.EnqueuedAt.UtcTicks
                    if byArrival <> 0 then byArrival
                    else compare left.SequenceId.Value right.SequenceId.Value
        else
            let byPriority = compare right.Priority left.Priority
            if byPriority <> 0 then byPriority
            else
                let byDeadline = compare (deadlineTicks left) (deadlineTicks right)
                if byDeadline <> 0 then byDeadline
                else
                    let byArrival = compare left.EnqueuedAt.UtcTicks right.EnqueuedAt.UtcTicks
                    if byArrival <> 0 then byArrival
                    else compare left.SequenceId.Value right.SequenceId.Value

    let private classifyAdmission budget sequence =
        if not (isRunnable sequence) then
            Choice2Of3 { Sequence = sequence; Reason = NotRunnable }
        elif sequence.TokenDemand <= 0 then
            Choice3Of3 { Sequence = sequence; Reason = InvalidTokenDemand }
        elif sequence.KvPageDemand < 0 then
            Choice3Of3 { Sequence = sequence; Reason = InvalidKvPageDemand }
        elif sequence.Phase = Decoding && sequence.TokenDemand <> 1 then
            Choice3Of3 { Sequence = sequence; Reason = InvalidDecodeQuantum }
        elif sequence.TokenDemand > budget.MaxBatchTokens then
            Choice3Of3 { Sequence = sequence; Reason = TokenDemandExceedsBatchCapacity }
        elif sequence.KvPageDemand > budget.AvailableKvPages then
            Choice3Of3 { Sequence = sequence; Reason = KvDemandExceedsCapacity }
        else
            Choice1Of3 sequence

    let private trySelect budget state sequence =
        let reason =
            if List.length state.SelectedRev >= budget.MaxBatchSequences then
                Some BatchSequenceBudget
            elif state.UsedTokens + sequence.TokenDemand > budget.MaxBatchTokens then
                Some TokenBudget
            elif state.UsedKvPages + sequence.KvPageDemand > budget.AvailableKvPages then
                Some KvBudget
            else
                None

        match reason with
        | Some deferredReason ->
            { state with
                DeferredRev = { Sequence = sequence; Reason = deferredReason } :: state.DeferredRev },
            false
        | None ->
            { state with
                SelectedRev =
                    { Sequence = sequence
                      TokenGrant = sequence.TokenDemand
                      KvPageGrant = sequence.KvPageDemand }
                    :: state.SelectedRev
                UsedTokens = state.UsedTokens + sequence.TokenDemand
                UsedKvPages = state.UsedKvPages + sequence.KvPageDemand },
            true

    let private reserveDecodeTokens budget policy orderedDecodes state =
        let reserveTarget = min policy.DecodeTokenReserve budget.MaxBatchTokens

        let rec loop current remaining =
            if current.UsedTokens >= reserveTarget then
                current, remaining
            else
                match remaining with
                | [] -> current, []
                | sequence :: tail ->
                    let next, _ = trySelect budget current sequence
                    loop next tail

        loop state orderedDecodes

    let scheduleAt now budget policy sequences =
        if budget.MaxBatchTokens < 0 then invalidArg (nameof budget.MaxBatchTokens) "MaxBatchTokens cannot be negative."
        if budget.AvailableKvPages < 0 then invalidArg (nameof budget.AvailableKvPages) "AvailableKvPages cannot be negative."
        if budget.MaxBatchSequences < 0 then invalidArg (nameof budget.MaxBatchSequences) "MaxBatchSequences cannot be negative."
        if policy.DecodeTokenReserve < 0 then invalidArg (nameof policy.DecodeTokenReserve) "DecodeTokenReserve cannot be negative."
        if policy.DeadlineUrgencyWindow < TimeSpan.Zero then invalidArg (nameof policy.DeadlineUrgencyWindow) "DeadlineUrgencyWindow cannot be negative."

        let admittedRev, deferredRev, rejectedRev =
            sequences
            |> List.fold (fun (admitted, deferred, rejected) sequence ->
                match classifyAdmission budget sequence with
                | Choice1Of3 candidate -> candidate :: admitted, deferred, rejected
                | Choice2Of3 deferredItem -> admitted, deferredItem :: deferred, rejected
                | Choice3Of3 rejectedItem -> admitted, deferred, rejectedItem :: rejected)
                ([], [], [])

        let admitted = List.rev admittedRev
        let initiallyDeferred = List.rev deferredRev
        let rejected = List.rev rejectedRev

        let decodes, prefills =
            admitted
            |> List.partition (fun sequence -> sequence.Phase = Decoding)

        let orderedDecodes = decodes |> List.sortWith (compareReady now policy)
        let orderedPrefills = prefills |> List.sortWith (compareReady now policy)

        let initialState =
            { SelectedRev = []
              DeferredRev = []
              UsedTokens = 0
              UsedKvPages = 0 }

        let afterReserve, remainingDecodes =
            reserveDecodeTokens budget policy orderedDecodes initialState

        let remainingCandidates =
            remainingDecodes @ orderedPrefills
            |> List.sortWith (compareReady now policy)

        let finalState =
            remainingCandidates
            |> List.fold (fun state sequence -> fst (trySelect budget state sequence)) afterReserve

        { Selected = List.rev finalState.SelectedRev
          Deferred = initiallyDeferred @ List.rev finalState.DeferredRev
          Rejected = rejected
          ConsumedTokens = finalState.UsedTokens
          ConsumedKvPages = finalState.UsedKvPages }

    let schedule budget policy sequences =
        scheduleAt DateTimeOffset.UtcNow budget policy sequences
