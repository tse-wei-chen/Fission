using System.Globalization;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Backends.OnnxRuntime;
using Fission.Runtime.Backends;
using Microsoft.Extensions.Configuration;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Fission.Server;

/// <summary>
/// Builds the inference backend selected by server configuration.
/// Deterministic and CPU ONNX remain hardware-independent defaults; CUDA is
/// opt-in and fails during startup when its native provider/runtime is unavailable.
/// </summary>
public static class ServerBackendFactory
{
    private const long DefaultCudaPoolRetainedBytes = 256L * 1024L * 1024L;
    private const int DefaultCudaPoolRetainedBuffersPerSize = 8;

    public static DeviceId ResolveDevice(IConfiguration configuration)
    {
        ArgumentNullException.ThrowIfNull(configuration);

        var explicitDevice = configuration["Fission:Device"];
        if (!string.IsNullOrWhiteSpace(explicitDevice))
        {
            return new DeviceId(explicitDevice.Trim());
        }

        var backend = (configuration["Fission:Backend"] ?? "deterministic").Trim();
        if (backend.Equals("onnx", StringComparison.OrdinalIgnoreCase) ||
            backend.Equals("onnxruntime", StringComparison.OrdinalIgnoreCase))
        {
            var provider = ReadExecutionProvider(configuration);
            if (provider == OnnxExecutionProvider.Cuda)
            {
                return new DeviceId($"cuda:{ReadCudaDeviceId(configuration)}");
            }
        }

        return new DeviceId("cpu:0");
    }

    public static IInferenceBackend Create(
        IConfiguration configuration,
        DeviceId device)
    {
        ArgumentNullException.ThrowIfNull(configuration);

        var backend = (configuration["Fission:Backend"] ?? "deterministic").Trim();
        return backend.ToLowerInvariant() switch
        {
            "" or "deterministic" => new DeterministicBackend(device),
            "onnx" or "onnxruntime" => CreateOnnxRuntime(configuration, device),
            _ => throw new InvalidOperationException(
                $"Unsupported Fission backend '{backend}'. Expected 'deterministic' or 'onnx'.")
        };
    }

    private static IInferenceBackend CreateOnnxRuntime(
        IConfiguration configuration,
        DeviceId device)
    {
        var modelPath = Path.GetFullPath(ReadRequired(configuration, "Fission:ModelPath"));
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException(
                $"Configured ONNX model does not exist: {modelPath}",
                modelPath);
        }

        var modelId = ReadRequired(configuration, "Fission:ModelId");
        var provider = ReadExecutionProvider(configuration);
        var modelElementType = ReadModelElementType(configuration);
        var sampledTokenIdsOutput = ReadOptional(
            configuration,
            "Fission:SampledTokenIdsOutput");
        if (sampledTokenIdsOutput is not null &&
            provider != OnnxExecutionProvider.Cuda)
        {
            throw new InvalidOperationException(
                "Fission:SampledTokenIdsOutput is currently supported only with the CUDA execution provider.");
        }

        if (modelElementType == TensorElementType.Float16)
        {
            if (provider != OnnxExecutionProvider.Cuda)
            {
                throw new InvalidOperationException(
                    "Fission:ModelPrecision=fp16 is currently supported only with the CUDA execution provider.");
            }

            if (sampledTokenIdsOutput is null)
            {
                throw new InvalidOperationException(
                    "Fission:ModelPrecision=fp16 currently requires Fission:SampledTokenIdsOutput.");
            }

            if (ReadBoolean(
                    configuration,
                    "Fission:CudaPageLockedDecodeLogits",
                    fallback: false))
            {
                throw new InvalidOperationException(
                    "Fission:CudaPageLockedDecodeLogits is not used by the graph-sampled FP16 CUDA path.");
            }
        }

        var profile = OptimumLegacyDecoderProfile.CreateLlamaLike(
            numHiddenLayers: ReadPositiveInt(configuration, "Fission:NumHiddenLayers"),
            numKvHeads: ReadPositiveInt(configuration, "Fission:NumKvHeads"),
            headDim: ReadPositiveInt(configuration, "Fission:HeadDim"),
            vocabularySize: ReadPositiveInt(configuration, "Fission:VocabularySize"),
            kvElementType: modelElementType,
            logitsElementType: modelElementType,
            sampledTokenIdsOutput: sampledTokenIdsOutput);
        var eosTokenIds = ReadTokenIds(configuration, "Fission:EosTokenIds");

        return provider switch
        {
            OnnxExecutionProvider.Cpu =>
                CreateCpuOnnxRuntime(
                    modelPath,
                    modelId,
                    device,
                    profile,
                    eosTokenIds),
            OnnxExecutionProvider.Cuda =>
                CreateCudaOnnxRuntime(
                    configuration,
                    modelPath,
                    modelId,
                    device,
                    profile,
                    eosTokenIds),
            _ => throw new InvalidOperationException(
                "Unsupported ONNX execution-provider state.")
        };
    }

    private static IInferenceBackend CreateCpuOnnxRuntime(
        string modelPath,
        string modelId,
        DeviceId device,
        OptimumLegacyDecoderProfile profile,
        int[] eosTokenIds)
    {
        var binding = new OptimumLegacyFloatDecoderBinding(profile, eosTokenIds);
        var adapter = new DecoderOnlyOnnxExecutionAdapter(binding);

        try
        {
            return new OnnxRuntimeBackend(
                new OnnxRuntimeBackendOptions(
                    new ModelId(modelId),
                    device,
                    OnnxRuntimeModelSource.FromFile(modelPath),
                    SessionContract: profile.SessionContract),
                adapter);
        }
        catch
        {
            adapter.Dispose();
            throw;
        }
    }

    private static IInferenceBackend CreateCudaOnnxRuntime(
        IConfiguration configuration,
        string modelPath,
        string modelId,
        DeviceId device,
        OptimumLegacyDecoderProfile profile,
        int[] eosTokenIds)
    {
        var cudaDeviceId = ReadCudaDeviceId(configuration);
        var runtimeLibraryPath = ReadOptional(configuration, "Fission:CudaRuntimeLibraryPath");
        var ortProfileOutputPathPrefix = ReadOptional(
            configuration,
            "Fission:OrtProfileOutputPathPrefix");
        var pageLockedDecodeLogits = ReadBoolean(
            configuration,
            "Fission:CudaPageLockedDecodeLogits",
            fallback: false);
        var pool = new CudaPooledDeviceMemoryAllocator(
            new CudaDeviceMemoryPoolOptions
            {
                MaxRetainedBytes = ReadNonNegativeLong(
                    configuration,
                    "Fission:CudaPoolMaxRetainedBytes",
                    DefaultCudaPoolRetainedBytes),
                MaxRetainedBuffersPerSize = ReadNonNegativeInt(
                    configuration,
                    "Fission:CudaPoolMaxRetainedBuffersPerSize",
                    DefaultCudaPoolRetainedBuffersPerSize)
            },
            new CudaDeviceMemoryAllocatorOptions
            {
                DeviceId = cudaDeviceId,
                RuntimeLibraryPath = runtimeLibraryPath
            });

        DecoderOnlyOnnxExecutionAdapter? adapter = null;
        OptimumLegacyCudaGqaSafeBinding? binding = null;
        CudaDeviceBoundAsyncCopyEngine? copyEngine = null;
        CudaOnnxOwnedResources? ownedResources = null;
        try
        {
            var poolController = new CudaDeviceMemoryPoolController(pool);
            var pressureMonitor = new CudaDeviceMemoryPressureMonitor(
                new CudaDeviceMemoryPressureMonitorOptions
                {
                    DeviceId = cudaDeviceId,
                    RuntimeLibraryPath = runtimeLibraryPath
                },
                poolController);

            IHostStagingFloatBufferAllocator? decodeLogitsHostAllocator = null;
            if (pageLockedDecodeLogits)
            {
                decodeLogitsHostAllocator =
                    new CudaPageLockedHostStagingFloatBufferAllocator(
                        new CudaPageLockedHostStagingAllocatorOptions
                        {
                            RuntimeLibraryPath = runtimeLibraryPath
                        });
            }

            var innerBinding = new OptimumLegacyCudaFloatDecoderBinding(
                profile,
                pool,
                eosTokenIds,
                decodeLogitsHostAllocator: decodeLogitsHostAllocator);
            copyEngine = new CudaDeviceBoundAsyncCopyEngine(
                cudaDeviceId,
                new CudaAsyncCopyEngineOptions
                {
                    RuntimeLibraryPath = runtimeLibraryPath
                });
            var gatheringBinding = new OptimumLegacyCudaGatheringBinding(
                profile,
                pool,
                copyEngine,
                innerBinding);
            binding = new OptimumLegacyCudaGqaSafeBinding(gatheringBinding);
            adapter = new DecoderOnlyOnnxExecutionAdapter(binding);
            ownedResources = new CudaOnnxOwnedResources(copyEngine, pool);

            return new OnnxRuntimeBackend(
                new OnnxRuntimeBackendOptions(
                    new ModelId(modelId),
                    device,
                    OnnxRuntimeModelSource.FromFile(modelPath),
                    SessionContract: profile.SessionContract,
                    DeviceMemoryPressureSource: pressureMonitor,
                    DeviceMemoryReclaimer: pressureMonitor),
                adapter,
                OnnxRuntimeSessionOptions.Cuda(
                    cudaDeviceId,
                    ortProfileOutputPathPrefix),
                ownedResource: ownedResources);
        }
        catch
        {
            adapter?.Dispose();
            if (adapter is null)
            {
                binding?.Dispose();
            }

            if (ownedResources is not null)
            {
                ownedResources.Dispose();
            }
            else
            {
                if (copyEngine is not null)
                {
                    copyEngine.DisposeAsync().AsTask().GetAwaiter().GetResult();
                }

                pool.Dispose();
            }

            throw;
        }
    }

    private static OnnxExecutionProvider ReadExecutionProvider(
        IConfiguration configuration)
    {
        var configured = configuration["Fission:ExecutionProvider"];
        if (string.IsNullOrWhiteSpace(configured))
        {
            configured = configuration["Fission:OnnxExecutionProvider"];
        }

        var value = (configured ?? "cpu").Trim();
        return value.ToLowerInvariant() switch
        {
            "" or "cpu" => OnnxExecutionProvider.Cpu,
            "cuda" or "gpu" => OnnxExecutionProvider.Cuda,
            _ => throw new InvalidOperationException(
                $"Unsupported Fission execution provider '{value}'. Expected 'cpu' or 'cuda'.")
        };
    }

    private static int ReadCudaDeviceId(IConfiguration configuration) =>
        ReadNonNegativeInt(configuration, "Fission:CudaDeviceId", fallback: 0);

    private static TensorElementType ReadModelElementType(
        IConfiguration configuration)
    {
        var value = (configuration["Fission:ModelPrecision"] ?? "fp32")
            .Trim()
            .ToLowerInvariant();
        return value switch
        {
            "" or "fp32" or "float" or "float32" => TensorElementType.Float,
            "fp16" or "half" or "float16" => TensorElementType.Float16,
            _ => throw new InvalidOperationException(
                $"Unsupported Fission model precision '{value}'. Expected 'fp32' or 'fp16'.")
        };
    }

    private static bool ReadBoolean(
        IConfiguration configuration,
        string key,
        bool fallback)
    {
        var value = configuration[key];
        if (string.IsNullOrWhiteSpace(value))
        {
            return fallback;
        }

        if (bool.TryParse(value, out var parsed))
        {
            return parsed;
        }

        throw new InvalidOperationException(
            $"Configuration value '{key}' must be 'true' or 'false'.");
    }

    private static string ReadRequired(
        IConfiguration configuration,
        string key)
    {
        var value = configuration[key];
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new InvalidOperationException(
                $"Configuration value '{key}' is required for the ONNX backend.");
        }

        return value.Trim();
    }

    private static string? ReadOptional(
        IConfiguration configuration,
        string key)
    {
        var value = configuration[key];
        return string.IsNullOrWhiteSpace(value) ? null : value.Trim();
    }

    private static int ReadPositiveInt(
        IConfiguration configuration,
        string key)
    {
        var value = ReadRequired(configuration, key);
        if (!int.TryParse(
                value,
                NumberStyles.Integer,
                CultureInfo.InvariantCulture,
                out var parsed) ||
            parsed <= 0)
        {
            throw new InvalidOperationException(
                $"Configuration value '{key}' must be a positive integer.");
        }

        return parsed;
    }

    private static int ReadNonNegativeInt(
        IConfiguration configuration,
        string key,
        int fallback)
    {
        var value = configuration[key];
        if (string.IsNullOrWhiteSpace(value))
        {
            return fallback;
        }

        if (!int.TryParse(
                value,
                NumberStyles.Integer,
                CultureInfo.InvariantCulture,
                out var parsed) ||
            parsed < 0)
        {
            throw new InvalidOperationException(
                $"Configuration value '{key}' must be a non-negative integer.");
        }

        return parsed;
    }

    private static long ReadNonNegativeLong(
        IConfiguration configuration,
        string key,
        long fallback)
    {
        var value = configuration[key];
        if (string.IsNullOrWhiteSpace(value))
        {
            return fallback;
        }

        if (!long.TryParse(
                value,
                NumberStyles.Integer,
                CultureInfo.InvariantCulture,
                out var parsed) ||
            parsed < 0)
        {
            throw new InvalidOperationException(
                $"Configuration value '{key}' must be a non-negative integer.");
        }

        return parsed;
    }

    private static int[] ReadTokenIds(
        IConfiguration configuration,
        string key)
    {
        var value = configuration[key];
        if (string.IsNullOrWhiteSpace(value))
        {
            return Array.Empty<int>();
        }

        var parts = value.Split(
            [',', ';', ' '],
            StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var tokenIds = new int[parts.Length];

        for (var index = 0; index < parts.Length; index++)
        {
            if (!int.TryParse(
                    parts[index],
                    NumberStyles.Integer,
                    CultureInfo.InvariantCulture,
                    out var tokenId) ||
                tokenId < 0)
            {
                throw new InvalidOperationException(
                    $"Configuration value '{key}' contains invalid token id '{parts[index]}'.");
            }

            tokenIds[index] = tokenId;
        }

        return tokenIds;
    }

    private enum OnnxExecutionProvider
    {
        Cpu,
        Cuda
    }
}
