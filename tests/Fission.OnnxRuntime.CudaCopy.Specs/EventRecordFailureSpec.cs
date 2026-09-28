using System.Runtime.CompilerServices;
using Fission.Backends.OnnxRuntime;

internal static class EventRecordFailureSpec
{
    [ModuleInitializer]
    internal static void Run()
    {
        var cuda = new EventRecordFailureCudaApi();
        var engine = new CudaAsyncCopyEngine(
            cuda,
            new CudaAsyncCopyEngineOptions
            {
                StreamCount = 1,
                CompletionPollInterval = TimeSpan.FromMilliseconds(1)
            });

        var source = System.Runtime.InteropServices.Marshal.AllocHGlobal(4);
        var destination = System.Runtime.InteropServices.Marshal.AllocHGlobal(4);
        try
        {
            CudaRuntimeException? observed = null;
            try
            {
                engine.CopyAsync(
                        destination,
                        source,
                        4,
                        CudaMemcpyKind.HostToHost)
                    .AsTask()
                    .GetAwaiter()
                    .GetResult();
            }
            catch (CudaRuntimeException exception)
            {
                observed = exception;
            }

            if (observed is null ||
                observed.Operation != "cudaEventRecord" ||
                observed.ErrorCode != 99)
            {
                throw new InvalidOperationException(
                    "cudaEventRecord failure must preserve the native operation and error code.");
            }

            if (cuda.StreamSynchronizeCalls < 1)
            {
                throw new InvalidOperationException(
                    "An already-submitted CUDA copy must synchronize before stream reuse when event recording fails.");
            }

            if (cuda.EventDestroyCalls != 1)
            {
                throw new InvalidOperationException(
                    "Event-record failure must destroy the unusable completion event exactly once.");
            }

            if (engine.ActiveCopies != 0)
            {
                throw new InvalidOperationException(
                    "Event-record failure cleanup must return stream admission.");
            }
        }
        finally
        {
            System.Runtime.InteropServices.Marshal.FreeHGlobal(source);
            System.Runtime.InteropServices.Marshal.FreeHGlobal(destination);
            engine.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }

        if (cuda.StreamDestroyCalls != 1)
        {
            throw new InvalidOperationException(
                "Engine disposal must still destroy the reusable stream after event-record failure cleanup.");
        }
    }

    private sealed class EventRecordFailureCudaApi : ICudaAsyncCopyApi
    {
        private nint _stream;
        private nint _event;

        public int StreamSynchronizeCalls { get; private set; }
        public int EventDestroyCalls { get; private set; }
        public int StreamDestroyCalls { get; private set; }

        public int StreamCreateWithFlags(out nint stream, uint flags)
        {
            stream = _stream = (nint)1;
            return 0;
        }

        public int StreamDestroy(nint stream)
        {
            StreamDestroyCalls++;
            return stream == _stream ? 0 : 17;
        }

        public int StreamSynchronize(nint stream)
        {
            StreamSynchronizeCalls++;
            return stream == _stream ? 0 : 17;
        }

        public int MemcpyAsync(
            nint destination,
            nint source,
            nuint byteLength,
            CudaMemcpyKind kind,
            nint stream) =>
            stream == _stream ? 0 : 17;

        public int EventCreateWithFlags(out nint completionEvent, uint flags)
        {
            completionEvent = _event = (nint)2;
            return 0;
        }

        public int EventRecord(nint completionEvent, nint stream) => 99;

        public int EventQuery(nint completionEvent) =>
            CudaAsyncCopyEngine.CudaErrorNotReady;

        public int EventDestroy(nint completionEvent)
        {
            EventDestroyCalls++;
            return completionEvent == _event ? 0 : 17;
        }

        public string? GetErrorString(int errorCode) =>
            $"fake-cuda-error-{errorCode}";
    }
}
