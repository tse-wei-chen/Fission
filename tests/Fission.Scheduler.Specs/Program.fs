open System
open Fission.Abstractions
open Fission.Abstractions.Scheduling
open Fission.Scheduler

let require condition message =
    if not condition then
        invalidOp message

let sid (value: string) =
    SequenceId(Guid.Parse(value))

let now = DateTimeOffset(2026, 9, 24, 12, 0, 0, TimeSpan.Zero)
let blockSize = 4

let mk sequenceId phase priority deadline enqueuedAt tokenDemand position tokensPerKvPage =
    { SequenceId = sequenceId
      Phase = phase
      Deadline = deadline
      EnqueuedAt = enqueuedAt
      TokenDemand = tokenDemand
      Position = position
      TokensPerKvPage = tokensPerKvPage
      Priority = priority
      KvBytesPerToken = 0L }

let policy =
    { DecodeTokenReserve = 2
      MaxPrefillChunkTokens = 4
      DeadlineUrgencyWindow = TimeSpan.FromMilliseconds(50.0) }

let budget =
    { MaxBatchTokens = 8
      AvailableKvPages = 2
      MaxBatchSequences = 3
      AvailableKvBytes = Int64.MaxValue }

let decodeA =
    mk
        (sid "00000000-0000-0000-0000-000000000001")
        Decoding
        0
        None
        (now - TimeSpan.FromMilliseconds(4.0))
        1
        4
        blockSize

let decodeB =
    mk
        (sid "00000000-0000-0000-0000-000000000002")
        Decoding
        0
        None
        (now - TimeSpan.FromMilliseconds(3.0))
        1
        3
        blockSize

let urgentPrefill =
    mk
        (sid "00000000-0000-0000-0000-000000000003")
        Prefilling
        -10
        (Some(now + TimeSpan.FromMilliseconds(10.0)))
        (now - TimeSpan.FromMilliseconds(2.0))
        10
        0
        blockSize

let highPriorityPrefill =
    mk
        (sid "00000000-0000-0000-0000-000000000004")
        Prefilling
        100
        None
        (now - TimeSpan.FromMilliseconds(1.0))
        4
        0
        blockSize

let mixedDecision =
    Scheduler.scheduleAt now budget policy [ highPriorityPrefill; urgentPrefill; decodeB; decodeA ]

let selectedIds =
    mixedDecision.Selected
    |> List.map (fun item -> item.Sequence.SequenceId)

require
    (selectedIds = [ decodeA.SequenceId; decodeB.SequenceId; urgentPrefill.SequenceId ])
    "Decode reserve should protect two decode quanta before SLO-aware mixed selection."

require (mixedDecision.Selected[0].KvPageGrant = 1) "Decode at a page boundary must reserve a new KV page."
require (mixedDecision.Selected[1].KvPageGrant = 0) "Decode inside an existing KV page must not reserve another page."
require (mixedDecision.Selected[2].TokenGrant = 4) "Long prefill must be chunked by MaxPrefillChunkTokens."
require (mixedDecision.Selected[2].KvPageGrant = 1) "Four prefill tokens at block size four must reserve one page."
require (mixedDecision.ConsumedTokens = 6) "Mixed scheduling token accounting is incorrect."
require (mixedDecision.ConsumedKvPages = 2) "Mixed scheduling KV accounting is incorrect."
require (mixedDecision.ConsumedKvBytes = 0L) "Unknown physical KV cost must not invent retained byte consumption."
require (mixedDecision.ConsumedTransientKvBytes = 0L) "Unknown physical KV cost must not invent transient byte consumption."
require (mixedDecision.Deferred.Length = 1) "Expected one deferred prefill after batch sequence capacity is reached."
require
    (mixedDecision.Deferred.Head.Sequence.SequenceId = highPriorityPrefill.SequenceId
     && mixedDecision.Deferred.Head.Reason = BatchSequenceBudget)
    "The non-urgent prefill should be deferred by batch sequence capacity."

let scheduleId = Guid.Parse("11111111-1111-1111-1111-111111111111")
let compiledBatch = ScheduleCompiler.compile scheduleId mixedDecision

require (compiledBatch.ScheduleId = scheduleId) "Compiled schedule id must be preserved."
require (compiledBatch.Items.Count = mixedDecision.Selected.Length) "Compiled work item count must match the selected decision."
require (compiledBatch.ConsumedTokens = mixedDecision.ConsumedTokens) "Compiled token accounting must match the decision."
require (compiledBatch.ConsumedKvPages = mixedDecision.ConsumedKvPages) "Compiled KV accounting must match the decision."
require (compiledBatch.ConsumedKvBytes = mixedDecision.ConsumedKvBytes) "Compiled retained KV byte accounting must match the decision."
require (compiledBatch.ConsumedTransientKvBytes = mixedDecision.ConsumedTransientKvBytes) "Compiled transient KV byte accounting must match the decision."
require (compiledBatch.Items[0].Kind = ScheduledWorkKind.Decode) "First reserved decode must compile as decode work."
require (compiledBatch.Items[1].Kind = ScheduledWorkKind.Decode) "Second reserved decode must compile as decode work."
require (compiledBatch.Items[2].Kind = ScheduledWorkKind.Prefill) "Urgent prefill must compile as prefill work."
require (compiledBatch.Items[2].TokenGrant = 4) "Compiled prefill token grant must be preserved."
require (not compiledBatch.Items[2].CompletesPrefill) "A 4-token grant from 10 remaining tokens must remain a partial prefill."

let admissionPolicy =
    { DecodeTokenReserve = 0
      MaxPrefillChunkTokens = 4
      DeadlineUrgencyWindow = TimeSpan.FromMilliseconds(50.0) }

let invalidToken =
    mk (sid "00000000-0000-0000-0000-000000000010") Prefilling 0 None now 0 0 blockSize

let invalidPosition =
    mk (sid "00000000-0000-0000-0000-000000000011") Prefilling 0 None now 1 -1 blockSize

let invalidPageSize =
    mk (sid "00000000-0000-0000-0000-000000000012") Prefilling 0 None now 1 0 0

let invalidDecodeQuantum =
    mk (sid "00000000-0000-0000-0000-000000000013") Decoding 0 None now 2 0 blockSize

let invalidByteCost =
    { mk (sid "00000000-0000-0000-0000-000000000019") Prefilling 0 None now 1 0 blockSize with
        KvBytesPerToken = -1L }

let waiting =
    mk (sid "00000000-0000-0000-0000-000000000014") Waiting 0 None now 1 0 blockSize

let admissionDecision =
    Scheduler.scheduleAt
        now
        budget
        admissionPolicy
        [ invalidToken; invalidPosition; invalidPageSize; invalidDecodeQuantum; invalidByteCost; waiting ]

require admissionDecision.Selected.IsEmpty "Invalid/non-runnable work must not be selected."
require (admissionDecision.Rejected.Length = 5) "Expected five admission rejections."
require (admissionDecision.Deferred.Length = 1) "Expected one non-runnable deferral."
require (admissionDecision.Deferred.Head.Reason = NotRunnable) "Waiting work must be deferred as non-runnable."
require
    (admissionDecision.Rejected |> List.exists (fun item -> item.Reason = InvalidKvBytesPerToken))
    "Negative physical KV byte cost must be rejected."

let longPrefill =
    mk (sid "00000000-0000-0000-0000-000000000015") Prefilling 0 None now 100 0 blockSize

let longDecision =
    Scheduler.scheduleAt now { budget with MaxBatchSequences = 1 } admissionPolicy [ longPrefill ]

require (longDecision.Rejected.IsEmpty) "Long prefills must be chunked instead of rejected for exceeding batch token capacity."
require (longDecision.Selected.Head.TokenGrant = 4) "Long prefill must receive only one configured chunk."

let pressurePolicy =
    { admissionPolicy with MaxPrefillChunkTokens = 16 }

let pressureBudget =
    { MaxBatchTokens = 16
      AvailableKvPages = 1
      MaxBatchSequences = 1
      AvailableKvBytes = Int64.MaxValue }

let pressurePrefill =
    mk (sid "00000000-0000-0000-0000-000000000016") Prefilling 0 None now 16 0 blockSize

let pressureDecision =
    Scheduler.scheduleAt now pressureBudget pressurePolicy [ pressurePrefill ]

require (pressureDecision.Selected.Head.TokenGrant = 4) "KV pressure must shrink a large prefill chunk to one writable page."
require (pressureDecision.Selected.Head.KvPageGrant = 1) "KV-pressure-limited chunk must account for one page."

let bytePressurePrefill =
    { mk (sid "00000000-0000-0000-0000-000000000023") Prefilling 0 None now 16 0 blockSize with
        KvBytesPerToken = 128L }

let bytePressureBudget =
    { MaxBatchTokens = 16
      AvailableKvPages = 16
      MaxBatchSequences = 1
      AvailableKvBytes = 512L }

let bytePressureDecision =
    Scheduler.scheduleAt now bytePressureBudget pressurePolicy [ bytePressurePrefill ]

require (bytePressureDecision.Selected.Head.TokenGrant = 4) "Physical KV bytes must shrink prefill to the writable token count."
require (bytePressureDecision.Selected.Head.KvByteGrant = 512L) "Selected work must carry its retained physical KV byte grant."
require (bytePressureDecision.Selected.Head.TransientKvByteGrant = 512L) "Initial prefill successor frontier should reserve the same 512 bytes."
require (bytePressureDecision.ConsumedKvBytes = 512L) "Decision must account consumed retained physical KV bytes."
require (bytePressureDecision.ConsumedTransientKvBytes = 512L) "Decision must account consumed transient successor bytes."
let bytePressureCompiled = ScheduleCompiler.compileNew bytePressureDecision
require (bytePressureCompiled.Items[0].KvByteGrant = 512L) "Compiled work must preserve the retained physical KV byte grant."
require (bytePressureCompiled.Items[0].TransientKvByteGrant = 512L) "Compiled work must preserve transient successor byte grant."
require (bytePressureCompiled.ConsumedKvBytes = 512L) "Compiled batch must preserve retained physical KV byte accounting."
require (bytePressureCompiled.ConsumedTransientKvBytes = 512L) "Compiled batch must preserve transient physical KV byte accounting."

let noByteBudget =
    { bytePressureBudget with AvailableKvBytes = 0L }
let byteBlockedDecision =
    Scheduler.scheduleAt now noByteBudget pressurePolicy [ bytePressurePrefill ]
require
    (byteBlockedDecision.Selected.IsEmpty && byteBlockedDecision.Deferred.Head.Reason = KvByteBudget)
    "A modeled decoder must defer on retained KV byte exhaustion even when page capacity remains."

// Immutable-state overlap: incremental retained growth can fit while the complete
// successor frontier cannot coexist with the prior state. Position 3 plus one new
// token produces a 4-token successor (512 bytes) and is admissible. Position 4
// would produce a 5-token successor (640 bytes) and must defer despite requiring
// only 128 retained bytes after the old state is released.
let transientFits =
    { mk (sid "00000000-0000-0000-0000-000000000024") Decoding 0 None now 1 3 blockSize with
        KvBytesPerToken = 128L }
let transientBlocked =
    { mk (sid "00000000-0000-0000-0000-000000000025") Decoding 0 None now 1 4 blockSize with
        KvBytesPerToken = 128L }
let transientBudget =
    { MaxBatchTokens = 1
      AvailableKvPages = 16
      MaxBatchSequences = 1
      AvailableKvBytes = 512L }

let transientFitsDecision =
    Scheduler.scheduleAt now transientBudget admissionPolicy [ transientFits ]
require (transientFitsDecision.Selected.Length = 1) "A successor frontier equal to physical byte slack must be admitted."
require (transientFitsDecision.Selected.Head.KvByteGrant = 128L) "Decode retained growth should remain one token."
require (transientFitsDecision.Selected.Head.TransientKvByteGrant = 512L) "Decode must reserve the complete four-token successor frontier."

let transientBlockedDecision =
    Scheduler.scheduleAt now transientBudget admissionPolicy [ transientBlocked ]
require transientBlockedDecision.Selected.IsEmpty "A successor frontier larger than physical byte slack must not be selected."
require
    (transientBlockedDecision.Deferred.Head.Reason = TransientKvByteBudget)
    "Immutable successor overlap must report TransientKvByteBudget rather than incremental retained pressure."

let noPageBudget =
    { MaxBatchTokens = 4
      AvailableKvPages = 0
      MaxBatchSequences = 1
      AvailableKvBytes = Int64.MaxValue }

let decodeInsidePage =
    mk (sid "00000000-0000-0000-0000-000000000017") Decoding 0 None now 1 3 blockSize

let decodeAtBoundary =
    mk (sid "00000000-0000-0000-0000-000000000018") Decoding 0 None now 1 4 blockSize

let insideDecision = Scheduler.scheduleAt now noPageBudget admissionPolicy [ decodeInsidePage ]
let boundaryDecision = Scheduler.scheduleAt now noPageBudget admissionPolicy [ decodeAtBoundary ]
require (insideDecision.Selected.Length = 1 && insideDecision.ConsumedKvPages = 0) "Existing page slack must allow decode without free pages."
require (boundaryDecision.Selected.IsEmpty && boundaryDecision.Deferred.Head.Reason = KvBudget) "Decode at a full-page boundary must wait when no KV page is available."

let oneSlotBudget =
    { MaxBatchTokens = 4
      AvailableKvPages = 1
      MaxBatchSequences = 1
      AvailableKvBytes = Int64.MaxValue }

let urgentLowPriority =
    mk
        (sid "00000000-0000-0000-0000-000000000020")
        Prefilling
        1
        (Some(now + TimeSpan.FromMilliseconds(10.0)))
        now
        4
        0
        blockSize

let nonUrgentHighPriority =
    mk
        (sid "00000000-0000-0000-0000-000000000021")
        Prefilling
        100
        None
        now
        4
        0
        blockSize

let urgentDecision =
    Scheduler.scheduleAt now oneSlotBudget admissionPolicy [ nonUrgentHighPriority; urgentLowPriority ]

require
    (urgentDecision.Selected.Head.Sequence.SequenceId = urgentLowPriority.SequenceId)
    "A deadline inside the urgency window must outrank non-urgent priority."

let urgentCompiled = ScheduleCompiler.compileNew urgentDecision
require urgentCompiled.Items[0].CompletesPrefill "A grant that consumes the full prefill demand must be marked final."

let distantDeadline =
    { urgentLowPriority with
        SequenceId = sid "00000000-0000-0000-0000-000000000022"
        Deadline = Some(now + TimeSpan.FromSeconds(10.0)) }

let nonUrgentDecision =
    Scheduler.scheduleAt now oneSlotBudget admissionPolicy [ distantDeadline; nonUrgentHighPriority ]

require
    (nonUrgentDecision.Selected.Head.Sequence.SequenceId = nonUrgentHighPriority.SequenceId)
    "Outside the urgency window, priority should dominate deadline order."

let tieA =
    mk
        (sid "00000000-0000-0000-0000-000000000030")
        Prefilling
        5
        None
        now
        4
        0
        blockSize

let tieB =
    { tieA with SequenceId = sid "00000000-0000-0000-0000-000000000031" }

let deterministicLeft =
    Scheduler.scheduleAt now oneSlotBudget admissionPolicy [ tieB; tieA ]

let deterministicRight =
    Scheduler.scheduleAt now oneSlotBudget admissionPolicy [ tieA; tieB ]

require
    (deterministicLeft.Selected.Head.Sequence.SequenceId = tieA.SequenceId
     && deterministicRight.Selected.Head.Sequence.SequenceId = tieA.SequenceId)
    "Stable sequence-id tie breaking must make scheduling independent of input order."

printfn
    "Fission scheduler specs passed: selected=%d tokens=%d kvPages=%d kvBytes=%d transientKvBytes=%d rejected=%d bytePressureGrant=%d"
    mixedDecision.Selected.Length
    mixedDecision.ConsumedTokens
    mixedDecision.ConsumedKvPages
    bytePressureDecision.ConsumedKvBytes
    transientFitsDecision.ConsumedTransientKvBytes
    admissionDecision.Rejected.Length
    bytePressureDecision.Selected.Head.TokenGrant
