using Microsoft.Extensions.Configuration;

namespace Fission.Server;

public static class ServerTextTokenCodecFactory
{
    public static ITextTokenCodec Create(IConfiguration configuration)
    {
        ArgumentNullException.ThrowIfNull(configuration);

        var kind = (configuration["Fission:Tokenizer"] ?? "deterministic").Trim();
        return kind.ToLowerInvariant() switch
        {
            "" or "deterministic" => new DeterministicTextTokenCodec(),
            "huggingface" or "hf" => CreateHuggingFace(configuration),
            _ => throw new InvalidOperationException(
                $"Unsupported Fission tokenizer '{kind}'. Expected 'deterministic' or 'huggingface'.")
        };
    }

    private static ITextTokenCodec CreateHuggingFace(IConfiguration configuration)
    {
        var tokenizerPath = ReadRequired(configuration, "Fission:TokenizerPath");
        var chatTemplate = ParseChatTemplate(configuration["Fission:ChatTemplate"]);
        var addPromptSpecialTokens = ReadBoolean(
            configuration,
            "Fission:AddPromptSpecialTokens",
            aliasKey: "Fission:AddSpecialTokens",
            fallback: true);
        var skipSpecialTokensOnDecode = ReadBoolean(
            configuration,
            "Fission:SkipSpecialTokensOnDecode",
            aliasKey: "Fission:SkipSpecialTokens",
            fallback: true);

        return new HuggingFaceTextTokenCodec(
            tokenizerPath,
            chatTemplate,
            addPromptSpecialTokens,
            skipSpecialTokensOnDecode);
    }

    private static HuggingFaceChatTemplate ParseChatTemplate(string? value)
    {
        if (string.IsNullOrWhiteSpace(value) ||
            value.Equals("none", StringComparison.OrdinalIgnoreCase))
        {
            return HuggingFaceChatTemplate.None;
        }

        return value.Trim().ToLowerInvariant() switch
        {
            "chatml" or "qwen2" or "qwen" => HuggingFaceChatTemplate.ChatMl,
            "llama3" or "llama-3" or "llama" => HuggingFaceChatTemplate.Llama3,
            _ => throw new InvalidOperationException(
                $"Unsupported Fission chat template '{value}'. Expected 'chatml/qwen2', 'llama3', or 'none'.")
        };
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

    private static bool ReadBoolean(
        IConfiguration configuration,
        string key,
        string aliasKey,
        bool fallback)
    {
        var value = configuration[key];
        var effectiveKey = key;

        if (string.IsNullOrWhiteSpace(value))
        {
            value = configuration[aliasKey];
            effectiveKey = aliasKey;
        }

        if (string.IsNullOrWhiteSpace(value))
        {
            return fallback;
        }

        if (bool.TryParse(value, out var parsed))
        {
            return parsed;
        }

        throw new InvalidOperationException(
            $"Configuration value '{effectiveKey}' must be 'true' or 'false'.");
    }
}
