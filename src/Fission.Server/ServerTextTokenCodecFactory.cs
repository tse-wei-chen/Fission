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
        var addSpecialTokens = ReadBoolean(
            configuration,
            "Fission:AddSpecialTokens",
            fallback: true);
        var skipSpecialTokens = ReadBoolean(
            configuration,
            "Fission:SkipSpecialTokens",
            fallback: true);

        return new HuggingFaceTextTokenCodec(
            tokenizerPath,
            chatTemplate,
            addSpecialTokens,
            skipSpecialTokens);
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
            "chatml" => HuggingFaceChatTemplate.ChatMl,
            "llama3" or "llama-3" => HuggingFaceChatTemplate.Llama3,
            _ => throw new InvalidOperationException(
                $"Unsupported Fission chat template '{value}'. Expected 'chatml', 'llama3', or 'none'.")
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
}
