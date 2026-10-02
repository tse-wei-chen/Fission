namespace Fission.Scheduler

open System

[<RequireQualifiedAccess>]
module Scheduler =
    [<Struct>]
    type private AdmissionResult =
        | Admitted of candidate: ReadySequence
        | DeferredAdmission of deferredItem: DeferredSequence
        | RejectedAdmission of rejectedItem: RejectedSequence

    [<Struct>]
    type private DeviceUsage =
        { Device: Fission.Abstractions.DeviceId
          TransientBytes: int64
          Sequences: int }

    [<Struct>]
    type private SelectionState =
        { SelectedRev: ScheduledSequence list
          DeferredRev: DeferredSequence list
          SelectedCount: int
          UsedTokens: int
          UsedKvPages: int
          UsedKvBytes: int64
          UsedTransientKvBytes: int64
          FirstUsedDevice: DeviceUsage voption
          AdditionalUsedDevices: DeviceUsage list }

    let rec private containsDevice
        (device: Fission.Abstractions.DeviceId)
        (budgets: struct (Fission.Abstractions.DeviceId * 'T) list)
        =
        match budgets with
        | [] -> false
        | struct (candidate, _) :: tail ->
            candidate = device || containsDevice device tail

    let rec private hasDuplicateDevice
        (budgets: struct (Fission.Abstractions.DeviceId * 'T) list)
        =
        match budgets with
        | [] | [_] -> false
        | struct (device, _) :: tail ->
            containsDevice device tail || hasDuplicateDevice tail

    let private isRunnable (sequence: ReadySequence) =
        sequence.Phase = Prefilling || sequence.Phase = Decoding

    let private deadlineTicks (sequence: ReadySequence) =
        if sequence.DeadlineUtcTicks >= 0L then
            sequence.DeadlineUtcTicks
        else
            Int64.MaxValue

    let private compareReady
        (urgencyCutoffTicks: int64)
        (left: ReadySequence)
        (right: ReadySequence)
        =
        let urgentLeft =
            left.DeadlineUtcTicks >= 0L
            && left.DeadlineUtcTicks <= urgencyCutoffTicks
        let urgentRight =
            right.DeadlineUtcTicks >= 0L
            && right.DeadlineUtcTicks <= urgencyCutoffTicks

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
    // and the full successor frontier. The same successor-frontier estimate is
    // used both for logical transient-KV policy and for per-device physical
    // headroom, while the two budgets remain independently observable.
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

    let rec private tryFindDeviceValue
        (device: Fission.Abstractions.DeviceId)
        (values: struct (Fission.Abstractions.DeviceId * 'T) list)
        =
        match values with
        | [] -> ValueNone
        | struct (candidate, value) :: tail ->
            if candidate = device then
                ValueSome value
            else
                tryFindDeviceValue device tail

    let private tryFindDeviceBudget (budget: ResourceBudget) device =
        tryFindDeviceValue device budget.AvailableDeviceBytes

    let private tryFindDeviceSequenceBudget (budget: ResourceBudget) device =
        tryFindDeviceValue device budget.MaxDeviceSequences

    let rec private tryFindAdditionalDeviceUsage
        (device: Fission.Abstractions.DeviceId)
        (usages: DeviceUsage list)
        =
        match usages with
        | [] -> ValueNone
        | usage :: tail ->
            if usage.Device = device then
                ValueSome usage
            else
                tryFindAdditionalDeviceUsage device tail

    let private tryFindDeviceUsage
        (device: Fission.Abstractions.DeviceId)
        (state: SelectionState)
        =
        match state.FirstUsedDevice with
        | ValueSome usage when usage.Device = device -> ValueSome usage
        | _ -> tryFindAdditionalDeviceUsage device state.AdditionalUsedDevices

    let private usedDeviceBytes usage =
        match usage with
        | ValueSome current -> current.TransientBytes
        | ValueNone -> 0L

    let private usedDeviceSequences usage =
        match usage with
        | ValueSome current -> current.Sequences
        | ValueNone -> 0

    let rec private removeAdditionalDeviceUsage
        (device: Fission.Abstractions.DeviceId)
        (usages: DeviceUsage list)
        =
        match usages with
        | [] -> struct (ValueNone, [])
        | current :: tail when current.Device = device ->
            struct (ValueSome current, tail)
        | head :: tail ->
            let struct (current, withoutDevice) =
                removeAdditionalDeviceUsage device tail
            struct (current, head :: withoutDevice)

    let private updatedDeviceUsage device byteGrant sequenceGrant current =
        let struct (currentBytes, currentSequences) =
            match current with
            | ValueSome usage -> struct (usage.TransientBytes, usage.Sequences)
            | ValueNone -> struct (0L, 0)
        { Device = device
          TransientBytes = currentBytes + byteGrant
          Sequences = currentSequences + sequenceGrant }

    let private addDeviceUsage
        device
        byteGrant
        sequenceGrant
        (state: SelectionState)
        =
        match state.FirstUsedDevice with
        | ValueNone ->
            struct (
                ValueSome (updatedDeviceUsage device byteGrant sequenceGrant ValueNone),
                state.AdditionalUsedDevices)
        | ValueSome first when first.Device = device ->
            struct (
                ValueSome (updatedDeviceUsage device byteGrant sequenceGrant (ValueSome first)),
                state.AdditionalUsedDevices)
        | ValueSome first ->
            let struct (current, withoutDevice) =
                removeAdditionalDeviceUsage device state.AdditionalUsedDevices
            struct (
                ValueSome (updatedDeviceUsage device byteGrant sequenceGrant current),
                first :: withoutDevice)

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
                DeferredRev = { Sequence = sequence; Reason = BatchSequenceBudget } :: state.DeferredRev }
        else
            let deviceSequenceBudget =
                match sequence.ExecutionDevice with
                | ValueSome device -> tryFindDeviceSequenceBudget budget device
                | ValueNone -> ValueNone
            let deviceMemoryBudget =
                match sequence.ExecutionDevice with
                | ValueSome device -> tryFindDeviceBudget budget device
                | ValueNone -> ValueNone
            let tracksDeviceMemory =
                match deviceMemoryBudget with
                | ValueSome _ -> true
                | ValueNone -> false
            let tracksDeviceSequences =
                match deviceSequenceBudget with
                | ValueSome _ -> true
                | ValueNone -> false
            let currentDeviceUsage =
                match sequence.ExecutionDevice with
                | ValueSome device when tracksDeviceMemory || tracksDeviceSequences ->
                    tryFindDeviceUsage device state
                | _ -> ValueNone

            let deviceSequenceCapacityReached =
                match deviceSequenceBudget with
                | ValueSome maxSequences ->
                    usedDeviceSequences currentDeviceUsage >= maxSequences
                | ValueNone -> false

            if deviceSequenceCapacityReached then
                { state with
                    DeferredRev = { Sequence = sequence; Reason = DeviceSequenceBudget } :: state.DeferredRev }
            else
                let availableTokens = budget.MaxBatchTokens - state.UsedTokens
                if availableTokens <= 0 then
                    { state with
                        DeferredRev = { Sequence = sequence; Reason = TokenBudget } :: state.DeferredRev }
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
                    let deviceMemoryTokenCapacity =
                        match deviceMemoryBudget with
                        | ValueSome availableBytes ->
                            let remainingBytes = availableBytes - usedDeviceBytes currentDeviceUsage
                            tokensWritableWithTransientKvBytes sequence remainingBytes
                        | ValueNone -> Int32.MaxValue
                    let tokenGrant =
                        min
                            desiredTokens
                            (min
                                availableTokens
                                (min
                                    kvTokenCapacity
                                    (min kvByteTokenCapacity (min transientKvByteTokenCapacity deviceMemoryTokenCapacity))))

                    if tokenGrant <= 0 then
                        let reason =
                            if kvTokenCapacity <= 0 then KvBudget
                            elif kvByteTokenCapacity <= 0 then KvByteBudget
                            elif transientKvByteTokenCapacity <= 0 then TransientKvByteBudget
                            elif deviceMemoryTokenCapacity <= 0 then DeviceMemoryBudget
                            else TokenBudget
                        { state with
                            DeferredRev = { Sequence = sequence; Reason = reason } :: state.DeferredRev }
                    else
                        let kvPageGrant = kvPagesForGrant sequence tokenGrant
                        let kvByteGrant = int64 tokenGrant * sequence.KvBytesPerToken
                        let transientKvByteGrant = transientKvBytesForGrant sequence tokenGrant
                        let struct (nextFirstUsedDevice, nextAdditionalUsedDevices) =
                            match sequence.ExecutionDevice with
                            | ValueSome device when tracksDeviceMemory || tracksDeviceSequences ->
                                addDeviceUsage
                                    device
                                    (if tracksDeviceMemory then transientKvByteGrant else 0L)
                                    (if tracksDeviceSequences then 1 else 0)
                                    state
                            | _ ->
                                struct (state.FirstUsedDevice, state.AdditionalUsedDevices)
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
                            UsedTransientKvBytes = state.UsedTransientKvBytes + transientKvByteGrant
                            FirstUsedDevice = nextFirstUsedDevice
                            AdditionalUsedDevices = nextAdditionalUsedDevices }

    let private reserveDecodeTokens
        (budget: ResourceBudget)
        (policy: SchedulingPolicy)
        (orderedDecodes: ReadySequence list)
        (state: SelectionState)
        =
        let reserveTarget = min policy.DecodeTokenReserve budget.MaxBatchTokens

        let rec loop current remaining =
            if current.UsedTokens >= reserveTarget then
                struct (current, remaining)
            else
                match remaining with
                | [] -> struct (current, [])
                | sequence :: tail ->
                    let next = trySelect budget policy current sequence
                    loop next tail

        loop state orderedDecodes

    let private reserveDecodeArrayTokens
        (budget: ResourceBudget)
        (policy: SchedulingPolicy)
        (orderedDecodes: ReadySequence array)
        (state: SelectionState)
        =
        let reserveTarget = min policy.DecodeTokenReserve budget.MaxBatchTokens
        let mutable current = state
        let mutable index = 0

        while
            current.UsedTokens < reserveTarget
            && index < orderedDecodes.Length
            do
            current <-
                trySelect
                    budget
                    policy
                    current
                    orderedDecodes[index]
            index <- index + 1

        struct (current, index)

    let private selectMergedCandidates
        (budget: ResourceBudget)
        (policy: SchedulingPolicy)
        (compareCandidates: ReadySequence -> ReadySequence -> int)
        (initialState: SelectionState)
        (orderedDecodes: ReadySequence list)
        (orderedPrefills: ReadySequence array)
        =
        let rec loop state decodes prefillIndex =
            if prefillIndex >= orderedPrefills.Length then
                match decodes with
                | [] -> state
                | sequence :: tail ->
                    loop
                        (trySelect budget policy state sequence)
                        tail
                        prefillIndex
            else
                match decodes with
                | [] ->
                    let next =
                        trySelect
                            budget
                            policy
                            state
                            orderedPrefills[prefillIndex]
                    loop next [] (prefillIndex + 1)
                | decode :: decodeTail ->
                    let prefill = orderedPrefills[prefillIndex]
                    if compareCandidates decode prefill <= 0 then
                        loop
                            (trySelect budget policy state decode)
                            decodeTail
                            prefillIndex
                    else
                        loop
                            (trySelect budget policy state prefill)
                            decodes
                            (prefillIndex + 1)

        loop initialState orderedDecodes 0

    let private selectMergedCandidateArrays
        (budget: ResourceBudget)
        (policy: SchedulingPolicy)
        (compareCandidates: ReadySequence -> ReadySequence -> int)
        (initialState: SelectionState)
        (orderedDecodes: ReadySequence array)
        (decodeStartIndex: int)
        (orderedPrefills: ReadySequence array)
        =
        let mutable state = initialState
        let mutable decodeIndex = decodeStartIndex
        let mutable prefillIndex = 0

        while
            decodeIndex < orderedDecodes.Length
            || prefillIndex < orderedPrefills.Length
            do
            let useDecode =
                if decodeIndex >= orderedDecodes.Length then
                    false
                elif prefillIndex >= orderedPrefills.Length then
                    true
                else
                    compareCandidates
                        orderedDecodes[decodeIndex]
                        orderedPrefills[prefillIndex]
                    <= 0

            if useDecode then
                state <-
                    trySelect
                        budget
                        policy
                        state
                        orderedDecodes[decodeIndex]
                decodeIndex <- decodeIndex + 1
            else
                state <-
                    trySelect
                        budget
                        policy
                        state
                        orderedPrefills[prefillIndex]
                prefillIndex <- prefillIndex + 1

        state

    let private reserveDecodeWorkspaceTokens
        (budget: ResourceBudget)
        (policy: SchedulingPolicy)
        (orderedCandidates: ReadySequence array)
        (decodeCount: int)
        (state: SelectionState)
        =
        let reserveTarget = min policy.DecodeTokenReserve budget.MaxBatchTokens
        let mutable current = state
        let mutable index = 0

        while
            current.UsedTokens < reserveTarget
            && index < decodeCount
            do
            current <-
                trySelect
                    budget
                    policy
                    current
                    orderedCandidates[index]
            index <- index + 1

        struct (current, index)

    let private selectMergedCandidateWorkspace
        (budget: ResourceBudget)
        (policy: SchedulingPolicy)
        (compareCandidates: ReadySequence -> ReadySequence -> int)
        (initialState: SelectionState)
        (orderedCandidates: ReadySequence array)
        (decodeStartIndex: int)
        (decodeCount: int)
        (prefillStartIndex: int)
        =
        let mutable state = initialState
        let mutable decodeIndex = decodeStartIndex
        let mutable prefillIndex = prefillStartIndex

        while
            decodeIndex < decodeCount
            || prefillIndex < orderedCandidates.Length
            do
            let useDecode =
                if decodeIndex >= decodeCount then
                    false
                elif prefillIndex >= orderedCandidates.Length then
                    true
                else
                    compareCandidates
                        orderedCandidates[decodeIndex]
                        orderedCandidates[prefillIndex]
                    <= 0

            if useDecode then
                state <-
                    trySelect
                        budget
                        policy
                        state
                        orderedCandidates[decodeIndex]
                decodeIndex <- decodeIndex + 1
            else
                state <-
                    trySelect
                        budget
                        policy
                        state
                        orderedCandidates[prefillIndex]
                prefillIndex <- prefillIndex + 1

        state

    let private validateScheduleInputs
        (budget: ResourceBudget)
        (policy: SchedulingPolicy)
        =
        if budget.MaxBatchTokens < 0 then invalidArg "MaxBatchTokens" "MaxBatchTokens cannot be negative."
        if budget.AvailableKvPages < 0 then invalidArg "AvailableKvPages" "AvailableKvPages cannot be negative."
        if budget.MaxBatchSequences < 0 then invalidArg "MaxBatchSequences" "MaxBatchSequences cannot be negative."
        if budget.AvailableKvBytes < 0L then invalidArg "AvailableKvBytes" "AvailableKvBytes cannot be negative."
        if budget.AvailableDeviceBytes |> List.exists (fun struct (_, availableBytes) -> availableBytes < 0L) then
            invalidArg "AvailableDeviceBytes" "Available device bytes cannot be negative."
        if hasDuplicateDevice budget.AvailableDeviceBytes then
            invalidArg "AvailableDeviceBytes" "Each execution device may appear only once in the device-memory budget."
        if budget.MaxDeviceSequences |> List.exists (fun struct (_, maxSequences) -> maxSequences < 0) then
            invalidArg "MaxDeviceSequences" "Per-device sequence capacity cannot be negative."
        if hasDuplicateDevice budget.MaxDeviceSequences then
            invalidArg "MaxDeviceSequences" "Each execution device may appear only once in the per-device sequence budget."
        if policy.DecodeTokenReserve < 0 then invalidArg "DecodeTokenReserve" "DecodeTokenReserve cannot be negative."
        if policy.MaxPrefillChunkTokens <= 0 then invalidArg "MaxPrefillChunkTokens" "MaxPrefillChunkTokens must be positive."
        if policy.DeadlineUrgencyWindow < TimeSpan.Zero then invalidArg "DeadlineUrgencyWindow" "DeadlineUrgencyWindow cannot be negative."

    let private classifyList (sequences: ReadySequence list) =
        sequences
        |> List.fold (fun struct (decodes, prefills, deferred, rejected) sequence ->
            match classifyAdmission sequence with
            | Admitted candidate when candidate.Phase = Decoding ->
                struct (candidate :: decodes, prefills, deferred, rejected)
            | Admitted candidate ->
                struct (decodes, candidate :: prefills, deferred, rejected)
            | DeferredAdmission deferredItem ->
                struct (decodes, prefills, deferredItem :: deferred, rejected)
            | RejectedAdmission rejectedItem ->
                struct (decodes, prefills, deferred, rejectedItem :: rejected))
            (struct ([], [], [], []))

    let private classifyArrayInOrder (sequences: ReadySequence array) =
        let mutable decodes = []
        let mutable prefills = []
        let mutable deferred = []
        let mutable rejected = []

        for index = sequences.Length - 1 downto 0 do
            match classifyAdmission sequences[index] with
            | Admitted candidate when candidate.Phase = Decoding ->
                decodes <- candidate :: decodes
            | Admitted candidate ->
                prefills <- candidate :: prefills
            | DeferredAdmission deferredItem ->
                deferred <- deferredItem :: deferred
            | RejectedAdmission rejectedItem ->
                rejected <- rejectedItem :: rejected

        struct (decodes, prefills, deferred, rejected)

    let private classifyMappedReadOnlyInOrder
        (mapping: 'T -> ReadySequence)
        (sequences: System.Collections.Generic.IReadOnlyList<'T>)
        =
        let mutable decodes = []
        let mutable prefills = []
        let mutable deferred = []
        let mutable rejected = []

        for index = sequences.Count - 1 downto 0 do
            let sequence = mapping sequences[index]
            match classifyAdmission sequence with
            | Admitted candidate when candidate.Phase = Decoding ->
                decodes <- candidate :: decodes
            | Admitted candidate ->
                prefills <- candidate :: prefills
            | DeferredAdmission deferredItem ->
                deferred <- deferredItem :: deferred
            | RejectedAdmission rejectedItem ->
                rejected <- rejectedItem :: rejected

        struct (decodes, prefills, deferred, rejected)

    let private classifyMappedReadOnlyWorkspace
        (mapping: 'T -> ReadySequence)
        (sequences: System.Collections.Generic.IReadOnlyList<'T>)
        =
        let workspace = Array.zeroCreate<ReadySequence> sequences.Count
        let mutable decodeCount = 0
        let mutable prefillStartIndex = sequences.Count
        let mutable deferred = []
        let mutable rejected = []

        for index = sequences.Count - 1 downto 0 do
            let sequence = mapping sequences[index]
            match classifyAdmission sequence with
            | Admitted candidate when candidate.Phase = Decoding ->
                workspace[decodeCount] <- candidate
                decodeCount <- decodeCount + 1
            | Admitted candidate ->
                prefillStartIndex <- prefillStartIndex - 1
                workspace[prefillStartIndex] <- candidate
            | DeferredAdmission deferredItem ->
                deferred <- deferredItem :: deferred
            | RejectedAdmission rejectedItem ->
                rejected <- rejectedItem :: rejected

        struct (
            workspace,
            decodeCount,
            prefillStartIndex,
            deferred,
            rejected)

    let private schedulePartitionedAt
        (now: DateTimeOffset)
        (budget: ResourceBudget)
        (policy: SchedulingPolicy)
        decodes
        prefills
        initiallyDeferred
        rejected
        =
        let urgencyCutoffTicks = now.Add(policy.DeadlineUrgencyWindow).UtcTicks
        let compareCandidates = compareReady urgencyCutoffTicks

        let initialState =
            { SelectedRev = []
              DeferredRev = []
              SelectedCount = 0
              UsedTokens = 0
              UsedKvPages = 0
              UsedKvBytes = 0L
              UsedTransientKvBytes = 0L
              FirstUsedDevice = ValueNone
              AdditionalUsedDevices = [] }

        let orderedPrefills =
            match prefills with
            | [] -> Array.empty
            | _ ->
                let items = List.toArray prefills
                if items.Length > 1 then
                    Array.sortInPlaceWith compareCandidates items
                items

        let finalState =
            match decodes with
            | [] ->
                selectMergedCandidates
                    budget
                    policy
                    compareCandidates
                    initialState
                    []
                    orderedPrefills
            | [_] ->
                let struct (afterReserve, remainingDecodes) =
                    reserveDecodeTokens
                        budget
                        policy
                        decodes
                        initialState
                selectMergedCandidates
                    budget
                    policy
                    compareCandidates
                    afterReserve
                    remainingDecodes
                    orderedPrefills
            | _ ->
                let orderedDecodes = List.toArray decodes
                Array.sortInPlaceWith compareCandidates orderedDecodes
                let struct (afterReserve, decodeStartIndex) =
                    reserveDecodeArrayTokens
                        budget
                        policy
                        orderedDecodes
                        initialState
                selectMergedCandidateArrays
                    budget
                    policy
                    compareCandidates
                    afterReserve
                    orderedDecodes
                    decodeStartIndex
                    orderedPrefills

        { SelectedRev = finalState.SelectedRev
          InitiallyDeferred = initiallyDeferred
          DeferredRev = finalState.DeferredRev
          Rejected = rejected
          ConsumedTokens = finalState.UsedTokens
          ConsumedKvPages = finalState.UsedKvPages
          ConsumedKvBytes = finalState.UsedKvBytes
          ConsumedTransientKvBytes = finalState.UsedTransientKvBytes }

    let private normalizeDecision (decision: RawSchedulingDecision) =
        { Selected = List.rev decision.SelectedRev
          Deferred =
            decision.InitiallyDeferred
            @ List.rev decision.DeferredRev
          Rejected = decision.Rejected
          ConsumedTokens = decision.ConsumedTokens
          ConsumedKvPages = decision.ConsumedKvPages
          ConsumedKvBytes = decision.ConsumedKvBytes
          ConsumedTransientKvBytes = decision.ConsumedTransientKvBytes }

    let scheduleAt
        (now: DateTimeOffset)
        (budget: ResourceBudget)
        (policy: SchedulingPolicy)
        (sequences: ReadySequence list)
        =
        validateScheduleInputs budget policy
        let struct (decodesRev, prefillsRev, deferredRev, rejectedRev) =
            classifyList sequences

        schedulePartitionedAt
            now
            budget
            policy
            (List.rev decodesRev)
            (List.rev prefillsRev)
            (List.rev deferredRev)
            (List.rev rejectedRev)
        |> normalizeDecision

    let scheduleArrayAt
        (now: DateTimeOffset)
        (budget: ResourceBudget)
        (policy: SchedulingPolicy)
        (sequences: ReadySequence array)
        =
        validateScheduleInputs budget policy
        let struct (decodes, prefills, deferred, rejected) =
            classifyArrayInOrder sequences

        schedulePartitionedAt
            now
            budget
            policy
            decodes
            prefills
            deferred
            rejected
        |> normalizeDecision

    let internal scheduleMappedReadOnlyRawAt
        (now: DateTimeOffset)
        (budget: ResourceBudget)
        (policy: SchedulingPolicy)
        (mapping: 'T -> ReadySequence)
        (sequences: System.Collections.Generic.IReadOnlyList<'T>)
        =
        validateScheduleInputs budget policy

        if sequences.Count <= 1 then
            let struct (decodes, prefills, deferred, rejected) =
                classifyMappedReadOnlyInOrder mapping sequences

            schedulePartitionedAt
                now
                budget
                policy
                decodes
                prefills
                deferred
                rejected
        else
            let struct (
                workspace,
                decodeCount,
                prefillStartIndex,
                deferred,
                rejected) =
                classifyMappedReadOnlyWorkspace mapping sequences

            let urgencyCutoffTicks =
                now.Add(policy.DeadlineUrgencyWindow).UtcTicks
            let compareCandidates = compareReady urgencyCutoffTicks
            let comparer =
                System.Collections.Generic.Comparer<ReadySequence>.Create(
                    System.Comparison<ReadySequence>(
                        fun left right ->
                            compareCandidates left right))

            if decodeCount > 1 then
                System.Array.Sort<ReadySequence>(
                    workspace,
                    0,
                    decodeCount,
                    comparer)

            let prefillCount = workspace.Length - prefillStartIndex
            if prefillCount > 1 then
                System.Array.Sort<ReadySequence>(
                    workspace,
                    prefillStartIndex,
                    prefillCount,
                    comparer)

            let initialState =
                { SelectedRev = []
                  DeferredRev = []
                  SelectedCount = 0
                  UsedTokens = 0
                  UsedKvPages = 0
                  UsedKvBytes = 0L
                  UsedTransientKvBytes = 0L
                  FirstUsedDevice = ValueNone
                  AdditionalUsedDevices = [] }

            let struct (afterReserve, decodeStartIndex) =
                reserveDecodeWorkspaceTokens
                    budget
                    policy
                    workspace
                    decodeCount
                    initialState

            let finalState =
                selectMergedCandidateWorkspace
                    budget
                    policy
                    compareCandidates
                    afterReserve
                    workspace
                    decodeStartIndex
                    decodeCount
                    prefillStartIndex

            { SelectedRev = finalState.SelectedRev
              InitiallyDeferred = deferred
              DeferredRev = finalState.DeferredRev
              Rejected = rejected
              ConsumedTokens = finalState.UsedTokens
              ConsumedKvPages = finalState.UsedKvPages
              ConsumedKvBytes = finalState.UsedKvBytes
              ConsumedTransientKvBytes = finalState.UsedTransientKvBytes }

    let scheduleMappedReadOnlyAt
        (now: DateTimeOffset)
        (budget: ResourceBudget)
        (policy: SchedulingPolicy)
        (mapping: 'T -> ReadySequence)
        (sequences: System.Collections.Generic.IReadOnlyList<'T>)
        =
        scheduleMappedReadOnlyRawAt
            now
            budget
            policy
            mapping
            sequences
        |> normalizeDecision

    let schedule (budget: ResourceBudget) (policy: SchedulingPolicy) (sequences: ReadySequence list) =
        scheduleAt DateTimeOffset.UtcNow budget policy sequences
