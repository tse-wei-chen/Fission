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
          UsedKvPages: int
          UsedKvBytes: int64
          UsedTransientKvBytes: int64 }

    let private isRunnable (sequence: ReadySequence) =
        sequence.Phase = Prefilling || sequence.Phase = Decoding

    let private deadlineTicks (sequence: ReadySequence) =
        sequence.Deadline
        |> Option.map _.UtcTicks
        |> Option.defaultValue Int64.MaxValue

    let private isUrgent (now: DateTimeOffset) (policy: SchedulingPolicy) (sequence: ReadySequence) =
        match sequence.Deadline with
        | Some deadline -> deadline <= now.Add(policy.DeadlineUrgencyWindow)
        | None -> false

    let private compareReady
        (now: DateTimeOffset)
        (policy: SchedulingPolicy)
        (left: ReadySequence)
        (right: ReadySequence)
        =
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

    let private pagesForTokens tokensPerPage (tokenCount: int64) =
        if tokenCount <= 0L then
            0L
        else
            ((tokenCount - 1L) / int64 tokensPerPage) + 1L

    let private kvPagesForGrant (sequence: ReadySequence) tokenGrant =
        let before = pagesForTokens sequence.TokensPerKvPage (int64 sequence.Position)
        let afterPosition = int64 sequence.Position + int64 tokenGrant
        let after = pagesForTokens sequence.TokensPerKvPage afterPosition
        let pageDelta = after - before
        if pageDelta > int64 Int32.MaxValue then
            invalidOp "KV page grant exceeds Int32 capacity."
        int pageDelta

    let private tokensWritableWithKvPages (sequence: ReadySequence) availablePages =
        let currentPages = pagesForTokens sequence.TokensPerKvPage (int64 sequence.Position)
        let capacityPages = currentPages + int64 availablePages
        let capacityTokens = capacityPages * int64 sequence.TokensPerKvPage
        let writable = max 0L (capacityTokens - int64 sequence.Position)
        if writable > int64 Int32.MaxValue then Int32.MaxValue else int writable

    // Retained KV accounting only charges the incremental tokens that survive the
    // step after the immutable prior state is released.
    let private tokensWritableWithKvBytes (sequence: ReadySequence) (availableBytes: int64) =
        if sequence.KvBytesPerToken = 0L then
            Int32.MaxValue
        elif availableBytes <= 0L then
            0
        else
            let writable = availableBytes / sequence.KvBytesPerToken
            if writable > int64 Int32.MaxValue then Int32.MaxValue else int writable

    // Dense immutable decoder execution temporarily owns both the prior frontier
    // (already included in retained device usage) and the full successor frontier.
    // The successor therefore has to fit entirely inside the remaining physical
    // byte slack. This is deliberately conservative for future paged/COW backends.
    let private tokensWritableWithTransientKvBytes (sequence: ReadySequence) (availableBytes: int64) =
        if sequence.KvBytesPerToken = 0L then
            Int32.MaxValue
        elif availableBytes <= 0L then
            0
        else
            let successorTokenCapacity = availableBytes / sequence.KvBytesPerToken
            let writable = max 0L (successorTokenCapacity - int64 sequence.Position)
            if writable > int64 Int32.MaxValue then Int32.MaxValue else int writable

    let private transientKvBytesForGrant (sequence: ReadySequence) tokenGrant =
        if sequence.KvBytesPerToken = 0L then
            0L
        else
            (int64 sequence.Position + int64 tokenGrant) * sequence.KvBytesPerToken

    let private classifyAdmission (sequence: ReadySequence) =
        if not (isRunnable sequence) then
            DeferredAdmission { Sequence = sequence; Reason = NotRunnable }
        elif sequence.TokenDemand <= 0 then
            RejectedAdmission { Sequence = sequence; Reason = InvalidTokenDemand }
        elif sequence.Position < 0 then
            RejectedAdmission { Sequence = sequence; Reason = InvalidPosition }
        elif sequence.TokensPerKvPage <= 0 then
            RejectedAdmission { Sequence = sequence; Reason = InvalidKvPageSize }
        elif sequence.KvBytesPerToken < 0L then
            RejectedAdmission { Sequence = sequence; Reason = InvalidKvBytesPerToken }
        elif sequence.Phase = Decoding && sequence.TokenDemand <> 1 then
            RejectedAdmission { Sequence = sequence; Reason = InvalidDecodeQuantum }
        else
            Admitted sequence

    let private trySelect
        (budget: ResourceBudget)
        (policy: SchedulingPolicy)
        (state: SelectionState)
        (sequence: ReadySequence)
        =
        if state.SelectedCount >= budget.MaxBatchSequences then
            { state with
                DeferredRev = { Sequence = sequence; Reason = BatchSequenceBudget } :: state.DeferredRev },
            false
        else
            let availableTokens = budget.MaxBatchTokens - state.UsedTokens
            if availableTokens <= 0 then
                { state with
                    DeferredRev = { Sequence = sequence; Reason = TokenBudget } :: state.DeferredRev },
                false
            else
                let desiredTokens =
                    if sequence.Phase = Decoding then
                        1
                    else
                        min sequence.TokenDemand policy.MaxPrefillChunkTokens

                let availableKvPages = budget.AvailableKvPages - state.UsedKvPages
                let kvTokenCapacity = tokensWritableWithKvPages sequence availableKvPages
                let availableKvBytes = budget.AvailableKvBytes - state.UsedKvBytes
                let kvByteTokenCapacity = tokensWritableWithKvBytes sequence availableKvBytes
                let availableTransientKvBytes = budget.AvailableKvBytes - state.UsedTransientKvBytes
                let transientKvByteTokenCapacity =
                    tokensWritableWithTransientKvBytes sequence availableTransientKvBytes
                let tokenGrant =
                    min
                        desiredTokens
                        (min
                            availableTokens
                            (min kvTokenCapacity (min kvByteTokenCapacity transientKvByteTokenCapacity)))

                if tokenGrant <= 0 then
                    let reason =
                        if kvTokenCapacity <= 0 then KvBudget
                        elif kvByteTokenCapacity <= 0 then KvByteBudget
                        elif transientKvByteTokenCapacity <= 0 then TransientKvByteBudget
                        else TokenBudget
                    { state with
                        DeferredRev = { Sequence = sequence; Reason = reason } :: state.DeferredRev },
                    false
                else
                    let kvPageGrant = kvPagesForGrant sequence tokenGrant
                    let kvByteGrant = int64 tokenGrant * sequence.KvBytesPerToken
                    let transientKvByteGrant = transientKvBytesForGrant sequence tokenGrant
                    { state with
                        SelectedRev =
                            { Sequence = sequence
                              TokenGrant = tokenGrant
                              KvPageGrant = kvPageGrant
                              KvByteGrant = kvByteGrant
                              TransientKvByteGrant = transientKvByteGrant }
                            :: state.SelectedRev
                        SelectedCount = state.SelectedCount + 1
                        UsedTokens = state.UsedTokens + tokenGrant
                        UsedKvPages = state.UsedKvPages + kvPageGrant
                        UsedKvBytes = state.UsedKvBytes + kvByteGrant
                        UsedTransientKvBytes = state.UsedTransientKvBytes + transientKvByteGrant },
                    true

    let private reserveDecodeTokens
        (budget: ResourceBudget)
        (policy: SchedulingPolicy)
        (orderedDecodes: ReadySequence list)
        (state: SelectionState)
        =
        let reserveTarget = min policy.DecodeTokenReserve budget.MaxBatchTokens

        let rec loop current remaining =
            if current.UsedTokens >= reserveTarget then
                current, remaining
            else
                match remaining with
                | [] -> current, []
                | sequence :: tail ->
                    let next, _ = trySelect budget policy current sequence
                    loop next tail

        loop state orderedDecodes

    let scheduleAt
        (now: DateTimeOffset)
        (budget: ResourceBudget)
        (policy: SchedulingPolicy)
        (sequences: ReadySequence list)
        =
        if budget.MaxBatchTokens < 0 then invalidArg "MaxBatchTokens" "MaxBatchTokens cannot be negative."
        if budget.AvailableKvPages < 0 then invalidArg "AvailableKvPages" "AvailableKvPages cannot be negative."
        if budget.MaxBatchSequences < 0 then invalidArg "MaxBatchSequences" "MaxBatchSequences cannot be negative."
        if budget.AvailableKvBytes < 0L then invalidArg "AvailableKvBytes" "AvailableKvBytes cannot be negative."
        if policy.DecodeTokenReserve < 0 then invalidArg "DecodeTokenReserve" "DecodeTokenReserve cannot be negative."
        if policy.MaxPrefillChunkTokens <= 0 then invalidArg "MaxPrefillChunkTokens" "MaxPrefillChunkTokens must be positive."
        if policy.DeadlineUrgencyWindow < TimeSpan.Zero then invalidArg "DeadlineUrgencyWindow" "DeadlineUrgencyWindow cannot be negative."

        let admittedRev, deferredRev, rejectedRev =
            sequences
            |> List.fold (fun (admitted, deferred, rejected) sequence ->
                match classifyAdmission sequence with
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
              UsedKvPages = 0
              UsedKvBytes = 0L
              UsedTransientKvBytes = 0L }

        let afterReserve, remainingDecodes =
            reserveDecodeTokens budget policy orderedDecodes initialState

        let remainingCandidates =
            remainingDecodes @ orderedPrefills
            |> List.sortWith (compareReady now policy)

        let finalState =
            remainingCandidates
            |> List.fold (fun state sequence -> fst (trySelect budget policy state sequence)) afterReserve

        { Selected = List.rev finalState.SelectedRev
          Deferred = initiallyDeferred @ List.rev finalState.DeferredRev
          Rejected = rejected
          ConsumedTokens = finalState.UsedTokens
          ConsumedKvPages = finalState.UsedKvPages
          ConsumedKvBytes = finalState.UsedKvBytes
          ConsumedTransientKvBytes = finalState.UsedTransientKvBytes }

    let schedule (budget: ResourceBudget) (policy: SchedulingPolicy) (sequences: ReadySequence list) =
        scheduleAt DateTimeOffset.UtcNow budget policy sequences
