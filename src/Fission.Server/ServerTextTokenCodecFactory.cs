using Microsoft.Extensions.Configuration;

namespace Fission.Server;

public static class ServerTextTokenCodecFactory
{
    public static ITextTokenCodec Create(IConfiguration configuration)
    {
        ArgumentNullException.ThrowIfNull(configuration);

        var tokenizer = (configuration["Fission:Tokenizer"] ?? "deterministic").Trim();
        return tokenizer.ToLowerInvariant() switch
        {
            "" or "deterministic" => new DeterministicTextTokenCodec(),
            "huggingface" or "hf" => CreateHuggingFace(configuration),
            _ => throw new InvalidOperationException(
                $"Unsupported Fission tokenizer '{tokenizer}'. " +
                "Expected 'deterministic' or 'huggingface'.")
        };
    }

    private static ITextTokenCodec CreateHuggingFace(IConfiguration configuration)
    {
        var tokenizerPath = Path.GetFullPath(
            ReadRequired(configuration, "Fission:TokenizerPath"));
        if (!File.Exists(tokenizerPath))
        {
            throw new FileNotFoundException(
                $"Configured Hugging Face tokenizer does not exist: {tokenizerPath}",
                tokenizerPath);
        }

        var chatTemplate = ParseChatTemplate(configuration["Fission:ChatTemplate"]);
        var addPromptSpecialTokens = ReadBool(
            configuration,
            "Fission:AddPromptSpecialTokens",
            fallback: true);
        var skipSpecialTokensOnDecode = ReadBool(
            configuration,
            "Fission:SkipSpecialTokensOnDecode",
            fallback: true);

        return new HuggingFaceTextTokenCodec(
            tokenizerPath,
            chatTemplate,
            addPromptSpecialTokens,
            skipSpecialTokensOnDecode);
    }

    private static HuggingFaceChatTemplate ParseChatTemplate(string? value)
    {
        var normalized = (value ?? "none").Trim();
        return normalized.ToLowerInvariant() switch
        {
            "" or "none" => HuggingFaceChatTemplate.None,
            "qwen2" or "qwen" => HuggingFaceChatTemplate.Qwen2,
            "llama3" or "llama-3" or "llama" => HuggingFaceChatTemplate.Llama3,
            _ => throw new InvalidOperationException(
                $"Unsupported Fission chat template '{value}'. " +
                "Expected 'none', 'qwen2', or 'llama3'.")
        };
    }

    private static bool ReadBool(
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
            $"Configuration value '{key}' must be true or false.");
    }

    private static string ReadRequired(
        IConfiguration configuration,
        string key)
    {
        var value = configuration[key];
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new InvalidOperationException(
                $"Configuration value '{key}' is required for the Hugging Face tokenizer.");
        }

        return value.Trim();
    }
}
