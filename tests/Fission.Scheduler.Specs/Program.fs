open System
open Fission.Abstractions
open Fission.Scheduler

let require condition message =
    if not condition then
        invalidOp message

let sid (value: string) =
    SequenceId(Guid.Parse(value))

let now = DateTimeOffset(2026, 9, 24, 12, 0, 0, TimeSpan.Zero)

let mk sequenceId phase priority deadline enqueuedAt tokenDemand kvPageDemand =
    { SequenceId = sequenceId
      Phase = phase
      Deadline = deadline
      EnqueuedAt = enqueuedAt
      TokenDemand = tokenDemand
      KvPageDemand = kvPageDemand
      Priority = priority }

let policy =
    { DecodeTokenReserve = 2
      DeadlineUrgencyWindow = TimeSpan.FromMilliseconds(50.0) }

let budget =
    { MaxBatchTokens = 8
      AvailableKvPages = 4
      MaxBatchSequences = 3 }

let decodeA =
    mk
        (sid "00000000-0000-0000-0000-000000000001")
        Decoding
        0
        None
        (now - TimeSpan.FromMilliseconds(4.0))
        1
        1

let decodeB =
    mk
        (sid "00000000-0000-0000-0000-000000000002")
        Decoding
        0
        None
        (now - TimeSpan.FromMilliseconds(3.0))
        1
        1

let urgentPrefill =
    mk
        (sid "00000000-0000-0000-0000-000000000003")
        Prefilling
        -10
        (Some(now + TimeSpan.FromMilliseconds(10.0)))
        (now - TimeSpan.FromMilliseconds(2.0))
        4
        1

let highPriorityPrefill =
    mk
        (sid "00000000-0000-0000-0000-000000000004")
        Prefilling
        100
        None
        (now - TimeSpan.FromMilliseconds(1.0))
        4
        1

let mixedDecision =
    Scheduler.scheduleAt now budget policy [ highPriorityPrefill; urgentPrefill; decodeB; decodeA ]

let selectedIds =
    mixedDecision.Selected
    |> List.map (fun item -> item.Sequence.SequenceId)

require
    (selectedIds = [ decodeA.SequenceId; decodeB.SequenceId; urgentPrefill.SequenceId ])
    "Decode reserve should protect two decode quanta before SLO-aware mixed selection."

require (mixedDecision.ConsumedTokens = 6) "Mixed scheduling token accounting is incorrect."
require (mixedDecision.ConsumedKvPages = 3) "Mixed scheduling KV accounting is incorrect."
require (mixedDecision.Deferred.Length = 1) "Expected one deferred prefill after batch sequence capacity is reached."
require
    (mixedDecision.Deferred.Head.Sequence.SequenceId = highPriorityPrefill.SequenceId
     && mixedDecision.Deferred.Head.Reason = BatchSequenceBudget)
    "The non-urgent prefill should be deferred by batch sequence capacity."

let admissionPolicy =
    { DecodeTokenReserve = 0
      DeadlineUrgencyWindow = TimeSpan.FromMilliseconds(50.0) }

let invalidToken =
    mk (sid "00000000-0000-0000-0000-000000000010") Prefilling 0 None now 0 0

let invalidKv =
    mk (sid "00000000-0000-0000-0000-000000000011") Prefilling 0 None now 1 -1

let invalidDecodeQuantum =
    mk (sid "00000000-0000-0000-0000-000000000012") Decoding 0 None now 2 0

let oversizedToken =
    mk (sid "00000000-0000-0000-0000-000000000013") Prefilling 0 None now 9 0

let oversizedKv =
    mk (sid "00000000-0000-0000-0000-000000000014") Prefilling 0 None now 1 5

let waiting =
    mk (sid "00000000-0000-0000-0000-000000000015") Waiting 0 None now 1 0

let admissionDecision =
    Scheduler.scheduleAt
        now
        budget
        admissionPolicy
        [ invalidToken; invalidKv; invalidDecodeQuantum; oversizedToken; oversizedKv; waiting ]

require admissionDecision.Selected.IsEmpty "Invalid/non-runnable work must not be selected."
require (admissionDecision.Rejected.Length = 5) "Expected five admission rejections."
require (admissionDecision.Deferred.Length = 1) "Expected one non-runnable deferral."
require (admissionDecision.Deferred.Head.Reason = NotRunnable) "Waiting work must be deferred as non-runnable."

let oneSlotBudget =
    { MaxBatchTokens = 4
      AvailableKvPages = 4
      MaxBatchSequences = 1 }

let urgentLowPriority =
    mk
        (sid "00000000-0000-0000-0000-000000000020")
        Prefilling
        1
        (Some(now + TimeSpan.FromMilliseconds(10.0)))
        now
        4
        1

let nonUrgentHighPriority =
    mk
        (sid "00000000-0000-0000-0000-000000000021")
        Prefilling
        100
        None
        now
        4
        1

let urgentDecision =
    Scheduler.scheduleAt now oneSlotBudget admissionPolicy [ nonUrgentHighPriority; urgentLowPriority ]

require
    (urgentDecision.Selected.Head.Sequence.SequenceId = urgentLowPriority.SequenceId)
    "A deadline inside the urgency window must outrank non-urgent priority."

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
    "Fission scheduler specs passed: selected=%d tokens=%d kv=%d rejected=%d"
    mixedDecision.Selected.Length
    mixedDecision.ConsumedTokens
    mixedDecision.ConsumedKvPages
    admissionDecision.Rejected.Length
