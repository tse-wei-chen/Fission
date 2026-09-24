namespace Fission.Scheduler

open System

[<RequireQualifiedAccess>]
module Scheduler =
    type private AdmissionResult =
        | Admitted of ReadySequence
        | DeferredAdmission of DeferredSequence
        | RejectedAdmission of RejectedSequence

    type private SelectionState =
        { SelectedRev: ScheduledSequence list
          DeferredRev: DeferredSequence list
          SelectedCount: int
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
            DeferredAdmission { Sequence = sequence; Reason = NotRunnable }
        elif sequence.TokenDemand <= 0 then
            RejectedAdmission { Sequence = sequence; Reason = InvalidTokenDemand }
        elif sequence.KvPageDemand < 0 then
            RejectedAdmission { Sequence = sequence; Reason = InvalidKvPageDemand }
        elif sequence.Phase = Decoding && sequence.TokenDemand <> 1 then
            RejectedAdmission { Sequence = sequence; Reason = InvalidDecodeQuantum }
        elif sequence.TokenDemand > budget.MaxBatchTokens then
            RejectedAdmission { Sequence = sequence; Reason = TokenDemandExceedsBatchCapacity }
        elif sequence.KvPageDemand > budget.AvailableKvPages then
            RejectedAdmission { Sequence = sequence; Reason = KvDemandExceedsCapacity }
        else
            Admitted sequence

    let private trySelect budget state sequence =
        let reason =
            if state.SelectedCount >= budget.MaxBatchSequences then
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
                SelectedCount = state.SelectedCount + 1
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
        if budget.MaxBatchTokens < 0 then invalidArg "MaxBatchTokens" "MaxBatchTokens cannot be negative."
        if budget.AvailableKvPages < 0 then invalidArg "AvailableKvPages" "AvailableKvPages cannot be negative."
        if budget.MaxBatchSequences < 0 then invalidArg "MaxBatchSequences" "MaxBatchSequences cannot be negative."
        if policy.DecodeTokenReserve < 0 then invalidArg "DecodeTokenReserve" "DecodeTokenReserve cannot be negative."
        if policy.DeadlineUrgencyWindow < TimeSpan.Zero then invalidArg "DeadlineUrgencyWindow" "DeadlineUrgencyWindow cannot be negative."

        let admittedRev, deferredRev, rejectedRev =
            sequences
            |> List.fold (fun (admitted, deferred, rejected) sequence ->
                match classifyAdmission budget sequence with
                | Admitted candidate -> candidate :: admitted, deferred, rejected
                | DeferredAdmission deferredItem -> admitted, deferredItem :: deferred, rejected
                | RejectedAdmission rejectedItem -> admitted, deferred, rejectedItem :: rejected)
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
              SelectedCount = 0
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
