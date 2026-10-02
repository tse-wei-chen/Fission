using System.Text;
using HuggingFaceTokenizer = Tokenizers.HuggingFace.Tokenizer.Tokenizer;

namespace Fission.Server;

public enum HuggingFaceChatTemplate
{
    None = 0,
    Qwen2 = 1,
    Llama3 = 2
}

/// <summary>
/// Production text codec backed by a Hugging Face tokenizer.json file.
/// One immutable native tokenizer instance is shared by request-local decoder state.
/// </summary>
public sealed class HuggingFaceTextTokenCodec : ITextTokenCodec
{
    private readonly HuggingFaceTokenizer _tokenizer;
    private readonly HuggingFaceChatTemplate _chatTemplate;
    private readonly bool _addPromptSpecialTokens;
    private readonly bool _skipSpecialTokensOnDecode;
    private bool _disposed;

    public HuggingFaceTextTokenCodec(
        string tokenizerPath,
        HuggingFaceChatTemplate chatTemplate,
        bool addPromptSpecialTokens = true,
        bool skipSpecialTokensOnDecode = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tokenizerPath);

        var fullPath = Path.GetFullPath(tokenizerPath);
        if (!File.Exists(fullPath))
        {
            throw new FileNotFoundException(
                $"Configured Hugging Face tokenizer does not exist: {fullPath}",
                fullPath);
        }

        _tokenizer = HuggingFaceTokenizer.FromFile(fullPath);
        // false means added special tokens are extracted and replaced by their ids.
        // true would pass strings such as <|im_start|> through the normal model.
        _tokenizer.SetEncodeSpecialTokens(false);
        _chatTemplate = chatTemplate;
        _addPromptSpecialTokens = addPromptSpecialTokens;
        _skipSpecialTokensOnDecode = skipSpecialTokensOnDecode;
    }

    public int[] EncodePrompt(string prompt)
    {
        ThrowIfDisposed();
        ArgumentException.ThrowIfNullOrWhiteSpace(prompt);
        return Encode(prompt, _addPromptSpecialTokens);
    }

    public int[] EncodeChat(IReadOnlyList<OpenAiChatMessage> messages)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(messages);
        if (messages.Count == 0)
        {
            throw new ArgumentException(
                "At least one chat message is required.",
                nameof(messages));
        }

        if (_chatTemplate == HuggingFaceChatTemplate.None)
        {
            throw new ArgumentException(
                "Chat serving requires Fission:ChatTemplate to be 'qwen2' or 'llama3'.",
                nameof(messages));
        }

        var prompt = RenderChatPrompt(messages, _chatTemplate);
        // The chat template already carries its own control/special tokens.
        return Encode(prompt, addSpecialTokens: false);
    }

    public ITextTokenDecoder CreateDecoder()
    {
        ThrowIfDisposed();
        return new Decoder(_tokenizer, _skipSpecialTokensOnDecode);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _tokenizer.Dispose();
    }

    internal static string RenderChatPrompt(
        IReadOnlyList<OpenAiChatMessage> messages,
        HuggingFaceChatTemplate template)
    {
        var builder = new StringBuilder();

        if (template == HuggingFaceChatTemplate.Llama3)
        {
            builder.Append("<|begin_of_text|>");
        }

        foreach (var message in messages)
        {
            var role = ValidateRole(message.Role);
            ArgumentException.ThrowIfNullOrWhiteSpace(message.Content);

            switch (template)
            {
                case HuggingFaceChatTemplate.Qwen2:
                    builder.Append("<|im_start|>")
                        .Append(role)
                        .Append('\n')
                        .Append(message.Content)
                        .Append("<|im_end|>\n");
                    break;

                case HuggingFaceChatTemplate.Llama3:
                    builder.Append("<|start_header_id|>")
                        .Append(role)
                        .Append("<|end_header_id|>\n\n")
                        .Append(message.Content)
                        .Append("<|eot_id|>");
                    break;

                default:
                    throw new ArgumentOutOfRangeException(
                        nameof(template),
                        template,
                        "A concrete chat template is required.");
            }
        }

        switch (template)
        {
            case HuggingFaceChatTemplate.Qwen2:
                builder.Append("<|im_start|>assistant\n");
                break;
            case HuggingFaceChatTemplate.Llama3:
                builder.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
                break;
        }

        return builder.ToString();
    }

    private int[] Encode(string text, bool addSpecialTokens)
    {
        var encoding = _tokenizer.Encode(text, addSpecialTokens).Single();
        var ids = new int[encoding.Ids.Count];

        for (var index = 0; index < ids.Length; index++)
        {
            ids[index] = checked((int)encoding.Ids[index]);
        }

        return ids;
    }

    private static string ValidateRole(string role)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(role);

        return role switch
        {
            "system" or "developer" or "user" or "assistant" or "tool" => role,
            _ => throw new ArgumentException(
                $"Unsupported chat role '{role}'.",
                nameof(role))
        };
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    /// <summary>
    /// Managed reproduction of Hugging Face tokenizers 0.23 DecodeStream state.
    /// The public .NET wrapper exposes Decode but not DecodeStream.
    /// </summary>
    private sealed class Decoder(
        HuggingFaceTokenizer tokenizer,
        bool skipSpecialTokens) : ITextTokenDecoder
    {
        private List<uint>? _ids = [];
        private string _prefix = string.Empty;
        private int _prefixIndex;
        private bool _completed;

        public string Append(int tokenId)
        {
            var ids = RequireActive();
            ArgumentOutOfRangeException.ThrowIfNegative(tokenId);

            if (_prefix.Length == 0 && ids.Count != 0)
            {
                var newPrefix = Decode(ids);
                if (!EndsWithReplacementCharacter(newPrefix))
                {
                    _prefix = newPrefix;
                    _prefixIndex = ids.Count;
                }
            }

            ids.Add((uint)tokenId);
            var decoded = Decode(ids);

            if (decoded.Length <= _prefix.Length ||
                EndsWithReplacementCharacter(decoded))
            {
                return string.Empty;
            }

            if (!decoded.StartsWith(_prefix, StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    $"Tokenizer streaming decode no longer extends its stable prefix. " +
                    $"Token id: {tokenId}; expected prefix: '{_prefix}'; decoded: '{decoded}'.");
            }

            var newText = decoded[_prefix.Length..];
            var newPrefixIndex = ids.Count - _prefixIndex;

            if (_prefixIndex != 0)
            {
                ids.RemoveRange(0, _prefixIndex);
            }

            _prefix = Decode(ids);
            _prefixIndex = newPrefixIndex;
            return newText;
        }

        public string Complete()
        {
            var ids = RequireActive();
            if (_completed)
            {
                return string.Empty;
            }

            _completed = true;
            var decoded = Decode(ids);
            if (!decoded.StartsWith(_prefix, StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    "Tokenizer final decode no longer extends its stable prefix.");
            }

            return decoded[_prefix.Length..];
        }

        public void Dispose()
        {
            _completed = true;
            _ids = null;
            _prefix = string.Empty;
            _prefixIndex = 0;
        }

        private string Decode(IEnumerable<uint> ids) =>
            tokenizer.Decode(ids, skipSpecialTokens);

        private List<uint> RequireActive()
        {
            ObjectDisposedException.ThrowIf(_ids is null, this);
            if (_completed)
            {
                throw new InvalidOperationException(
                    "The request-scoped tokenizer decoder is already complete.");
            }

            return _ids;
        }

        private static bool EndsWithReplacementCharacter(string text) =>
            text.EndsWith('\uFFFD');
    }
}
