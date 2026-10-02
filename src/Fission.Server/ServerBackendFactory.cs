using System.Globalization;
using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Backends.OnnxRuntime;
using Fission.Runtime.Backends;
using Microsoft.Extensions.Configuration;
using Microsoft.ML.OnnxRuntime;

namespace Fission.Server;

/// <summary>
/// Builds the inference backend selected by server configuration.
/// Deterministic remains the default for tests and zero-model smoke runs.
/// </summary>
public static class ServerBackendFactory
{
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
        var profile = OptimumLegacyDecoderProfile.CreateLlamaLike(
            numHiddenLayers: ReadPositiveInt(configuration, "Fission:NumHiddenLayers"),
            numKvHeads: ReadPositiveInt(configuration, "Fission:NumKvHeads"),
            headDim: ReadPositiveInt(configuration, "Fission:HeadDim"),
            vocabularySize: ReadPositiveInt(configuration, "Fission:VocabularySize"));
        var eosTokenIds = ReadTokenIds(configuration, "Fission:EosTokenIds");
        var executionProvider =
            (configuration["Fission:OnnxExecutionProvider"] ?? "cpu").Trim();

        return executionProvider.ToLowerInvariant() switch
        {
            "" or "cpu" => CreateCpuOnnxRuntime(
                modelPath,
                modelId,
                device,
                profile,
                eosTokenIds),
            "cuda" => CreateCudaOnnxRuntime(
                configuration,
                modelPath,
                modelId,
                device,
                profile,
                eosTokenIds),
            _ => throw new InvalidOperationException(
                $"Unsupported ONNX Runtime execution provider '{executionProvider}'. " +
                "Expected 'cpu' or 'cuda'.")
        };
    }

    private static IInferenceBackend CreateCpuOnnxRuntime(
        string modelPath,
        string modelId,
        DeviceId device,
        OptimumLegacyDecoderProfile profile,
        IReadOnlyCollection<int> eosTokenIds)
    {
        var binding = new OptimumLegacyFloatDecoderBinding(
            profile,
            eosTokenIds);
        var adapter = new DecoderOnlyOnnxExecutionAdapter(binding);

        try
        {
            return new OnnxRuntimeBackend(
                CreateBackendOptions(modelPath, modelId, device, profile),
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
        IReadOnlyCollection<int> eosTokenIds)
    {
        var cudaDeviceId = ReadNonNegativeInt(
            configuration,
            "Fission:CudaDeviceId",
            fallback: 0);
        var runtimeLibraryPath = ReadOptional(
            configuration,
            "Fission:CudaRuntimeLibraryPath");

        var allocator = new CudaDeviceMemoryAllocator(
            new CudaDeviceMemoryAllocatorOptions
            {
                DeviceId = cudaDeviceId,
                RuntimeLibraryPath = runtimeLibraryPath
            });
        var binding = new OptimumLegacyCudaFloatDecoderBinding(
            profile,
            allocator,
            eosTokenIds);
        var adapter = new DecoderOnlyOnnxExecutionAdapter(binding);

        try
        {
            return new OnnxRuntimeBackend(
                CreateBackendOptions(modelPath, modelId, device, profile),
                adapter,
                sessionOptionsFactory: () => CreateCudaSessionOptions(cudaDeviceId));
        }
        catch
        {
            adapter.Dispose();
            throw;
        }
    }

    private static OnnxRuntimeBackendOptions CreateBackendOptions(
        string modelPath,
        string modelId,
        DeviceId device,
        OptimumLegacyDecoderProfile profile) =>
        new(
            new ModelId(modelId),
            device,
            OnnxRuntimeModelSource.FromFile(modelPath),
            SessionContract: profile.SessionContract);

    private static SessionOptions CreateCudaSessionOptions(int deviceId)
    {
        var options = new SessionOptions();
        try
        {
            options.AppendExecutionProvider_CUDA(deviceId);
            return options;
        }
        catch
        {
            options.Dispose();
            throw;
        }
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
        return string.IsNullOrWhiteSpace(value)
            ? null
            : value.Trim();
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
}
