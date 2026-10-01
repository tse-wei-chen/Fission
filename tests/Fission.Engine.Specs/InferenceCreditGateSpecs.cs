using System.Runtime.CompilerServices;
using Fission.Runtime.Execution;

internal static class InferenceCreditGateSpecs
{
    private static Task? _runTask;

    [ModuleInitializer]
    internal static void Start()
    {
        _runTask = Task.Run(RunAsync);
        AppDomain.CurrentDomain.ProcessExit += static (_, _) => CompleteBeforeExit();
    }

    private static void CompleteBeforeExit()
    {
        var runTask = _runTask;
        if (runTask is null)
        {
            return;
        }

        try
        {
            runTask.GetAwaiter().GetResult();
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine($"Inference credit gate spec failed: {exception}");
            Environment.ExitCode = 1;
        }
    }

    private static async Task RunAsync()
    {
        await VerifyWeightedAcquireAsync();
        await VerifyFifoCancellationAsync();
        await VerifyDisposalAsync();
        VerifyOversizedAcquireRejected();
    }

    private static async Task VerifyWeightedAcquireAsync()
    {
        using var gate = new InferenceCreditGate(capacity: 3);
        var first = await gate.AcquireAsync(2);

        var blocked = gate.AcquireAsync(2).AsTask();
        await RequirePendingAsync(
            blocked,
            "A two-credit waiter must remain pending while only one credit is available.");

        first.Dispose();
        using var second = await blocked.WaitAsync(TimeSpan.FromSeconds(5));

        var finalCredit = gate.AcquireAsync(1);
        Require(
            finalCredit.IsCompletedSuccessfully,
            "The remaining credit should stay available after one weighted waiter is granted atomically.");
        using var third = await finalCredit;
    }

    private static async Task VerifyFifoCancellationAsync()
    {
        using var gate = new InferenceCreditGate(capacity: 3);
        using var held = await gate.AcquireAsync(2);
        using var headCancellation = new CancellationTokenSource();

        var head = gate.AcquireAsync(2, headCancellation.Token).AsTask();
        var follower = gate.AcquireAsync(1).AsTask();

        await RequirePendingAsync(
            head,
            "The FIFO head must wait when its full weight is unavailable.");
        await RequirePendingAsync(
            follower,
            "A smaller later waiter must not bypass a blocked weighted FIFO head.");

        headCancellation.Cancel();
        await RequireCanceledAsync(
            head,
            "Canceling the FIFO head must cancel that acquisition.");

        using var followerLease = await follower.WaitAsync(TimeSpan.FromSeconds(5));
        Require(
            !follower.IsFaulted && !follower.IsCanceled,
            "Canceling a blocked head must immediately re-evaluate and grant an eligible follower from already-available credits.");
    }

    private static async Task VerifyDisposalAsync()
    {
        var gate = new InferenceCreditGate(capacity: 1);
        var held = await gate.AcquireAsync(1);
        var pending = gate.AcquireAsync(1).AsTask();

        await RequirePendingAsync(
            pending,
            "The disposal regression requires one queued credit waiter.");

        gate.Dispose();
        await RequireDisposedAsync(
            pending,
            "Gate disposal must fault queued credit waiters instead of leaving them pending.");

        // Active work may terminate after actor disposal has started. Returning an
        // already-issued lease must therefore remain harmless and idempotent.
        held.Dispose();
        held.Dispose();
        gate.Dispose();

        var rejected = false;
        try
        {
            _ = gate.AcquireAsync(1);
        }
        catch (ObjectDisposedException)
        {
            rejected = true;
        }

        Require(rejected, "New acquisitions must fail after gate disposal.");
    }

    private static void VerifyOversizedAcquireRejected()
    {
        using var gate = new InferenceCreditGate(capacity: 2);
        var rejected = false;

        try
        {
            _ = gate.AcquireAsync(3);
        }
        catch (InvalidOperationException exception) when (
            exception.Message.Contains("configured capacity is 2", StringComparison.Ordinal))
        {
            rejected = true;
        }

        Require(
            rejected,
            "A weighted acquisition larger than total gate capacity must fail immediately.");
    }

    private static async Task RequirePendingAsync(Task task, string message)
    {
        var completed = await Task.WhenAny(
            task,
            Task.Delay(TimeSpan.FromMilliseconds(100)));
        Require(!ReferenceEquals(completed, task), message);
    }

    private static async Task RequireCanceledAsync(Task task, string message)
    {
        try
        {
            await task.WaitAsync(TimeSpan.FromSeconds(5));
        }
        catch (OperationCanceledException)
        {
            return;
        }

        throw new InvalidOperationException(message);
    }

    private static async Task RequireDisposedAsync(Task task, string message)
    {
        try
        {
            await task.WaitAsync(TimeSpan.FromSeconds(5));
        }
        catch (ObjectDisposedException)
        {
            return;
        }

        throw new InvalidOperationException(message);
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }
}
