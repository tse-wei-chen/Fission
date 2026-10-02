using System.Text;
using HfTokenizer = Tokenizers.HuggingFace.Tokenizer.Tokenizer;

namespace Fission.Server;

public enum HuggingFaceChatTemplate
{
    None = 0,
    ChatMl = 1,
    Llama3 = 2
}

/// <summary>
/// Production text codec backed by a local Hugging Face tokenizer.json.
///
/// One immutable tokenizer instance is shared across requests. Encoding and decoding
/// call the native tokenizer through read-only operations; request-specific streaming
/// decode state lives in <see cref="Decoder"/>.
/// </summary>
public sealed class HuggingFaceTextTokenCodec : ITextTokenCodec
{
    private readonly HfTokenizer _tokenizer;
    private readonly HuggingFaceChatTemplate _chatTemplate;
    private readonly bool _addSpecialTokens;
    private readonly bool _skipSpecialTokens;
    private bool _disposed;

    public HuggingFaceTextTokenCodec(
        string tokenizerPath,
        HuggingFaceChatTemplate chatTemplate = HuggingFaceChatTemplate.None,
        bool addSpecialTokens = true,
        bool skipSpecialTokens = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(tokenizerPath);

        TokenizerPath = Path.GetFullPath(tokenizerPath);
        if (!File.Exists(TokenizerPath))
        {
            throw new FileNotFoundException(
                $"Configured tokenizer does not exist: {TokenizerPath}",
                TokenizerPath);
        }

        _tokenizer = HfTokenizer.FromFile(TokenizerPath);
        _chatTemplate = chatTemplate;
        _addSpecialTokens = addSpecialTokens;
        _skipSpecialTokens = skipSpecialTokens;
    }

    public string TokenizerPath { get; }

    public int[] EncodePrompt(string prompt)
    {
        ThrowIfDisposed();
        ArgumentException.ThrowIfNullOrWhiteSpace(prompt);
        return Encode(prompt, _addSpecialTokens);
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

        var rendered = _chatTemplate switch
        {
            HuggingFaceChatTemplate.ChatMl => RenderChatMl(messages),
            HuggingFaceChatTemplate.Llama3 => RenderLlama3(messages),
            _ => throw new ArgumentException(
                "Chat requests require Fission:ChatTemplate=chatml or llama3.")
        };

        // The built-in chat templates explicitly carry their model special tokens,
        // so do not also invoke a tokenizer post-processor that could add another BOS.
        return Encode(rendered, addSpecialTokens: false);
    }

    public ITextTokenDecoder CreateDecoder()
    {
        ThrowIfDisposed();
        return new Decoder(_tokenizer, _skipSpecialTokens);
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

    private int[] Encode(string text, bool addSpecialTokens)
    {
        var encoding = _tokenizer.Encode(text, addSpecialTokens).Single();
        var result = new int[encoding.Ids.Count];

        for (var index = 0; index < result.Length; index++)
        {
            result[index] = checked((int)encoding.Ids[index]);
        }

        return result;
    }

    private static string RenderChatMl(IReadOnlyList<OpenAiChatMessage> messages)
    {
        var builder = new StringBuilder();

        foreach (var message in messages)
        {
            ValidateMessage(message);
            builder.Append("<|im_start|>")
                .Append(message.Role)
                .Append('\n')
                .Append(message.Content)
                .Append("<|im_end|>\n");
        }

        builder.Append("<|im_start|>assistant\n");
        return builder.ToString();
    }

    private static string RenderLlama3(IReadOnlyList<OpenAiChatMessage> messages)
    {
        var builder = new StringBuilder("<|begin_of_text|>");

        foreach (var message in messages)
        {
            ValidateMessage(message);
            builder.Append("<|start_header_id|>")
                .Append(message.Role)
                .Append("<|end_header_id|>\n\n")
                .Append(message.Content)
                .Append("<|eot_id|>");
        }

        builder.Append("<|start_header_id|>assistant<|end_header_id|>\n\n");
        return builder.ToString();
    }

    private static void ValidateMessage(OpenAiChatMessage message)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(message.Role);
        ArgumentException.ThrowIfNullOrWhiteSpace(message.Content);
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }

    /// <summary>
    /// C# port of the state machine used by Hugging Face tokenizers 0.23.x
    /// DecodeStream. It retains only the token ids required to preserve decoder
    /// context, rather than repeatedly decoding the complete generation history.
    /// </summary>
    private sealed class Decoder(
        HfTokenizer tokenizer,
        bool skipSpecialTokens) : ITextTokenDecoder
    {
        private readonly List<uint> _ids = [];
        private string _prefix = string.Empty;
        private int _prefixIndex;
        private bool _completed;
        private bool _disposed;

        public string Append(int tokenId)
        {
            ThrowIfUnavailable();
            ArgumentOutOfRangeException.ThrowIfNegative(tokenId);

            if (_prefix.Length == 0 && _ids.Count != 0)
            {
                var newPrefix = Decode();
                if (!EndsWithReplacementCharacter(newPrefix))
                {
                    _prefix = newPrefix;
                    _prefixIndex = _ids.Count;
                }
            }

            _ids.Add(checked((uint)tokenId));
            var decoded = Decode();

            if (decoded.Length <= _prefix.Length ||
                EndsWithReplacementCharacter(decoded))
            {
                return string.Empty;
            }

            if (!decoded.StartsWith(_prefix, StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    $"Tokenizer streaming decode lost prefix continuity for token {tokenId}. " +
                    $"Expected prefix '{_prefix}', decoded '{decoded}'.");
            }

            var newText = decoded[_prefix.Length..];
            var newPrefixIndex = _ids.Count - _prefixIndex;

            if (_prefixIndex != 0)
            {
                _ids.RemoveRange(0, _prefixIndex);
            }

            _prefix = Decode();
            _prefixIndex = newPrefixIndex;
            return newText;
        }

        public string Complete()
        {
            ThrowIfUnavailable();
            _completed = true;

            if (_ids.Count == 0)
            {
                return string.Empty;
            }

            var decoded = Decode();
            if (_prefix.Length == 0)
            {
                return decoded;
            }

            if (!decoded.StartsWith(_prefix, StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    $"Tokenizer streaming decode lost prefix continuity while completing. " +
                    $"Expected prefix '{_prefix}', decoded '{decoded}'.");
            }

            return decoded[_prefix.Length..];
        }

        public void Dispose()
        {
            _disposed = true;
            _ids.Clear();
            _prefix = string.Empty;
            _prefixIndex = 0;
        }

        private string Decode() =>
            tokenizer.Decode(_ids, skipSpecialTokens);

        private static bool EndsWithReplacementCharacter(string value) =>
            value.EndsWith('\uFFFD');

        private void ThrowIfUnavailable()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_completed)
            {
                throw new InvalidOperationException(
                    "The request-scoped tokenizer decoder has already completed.");
            }
        }
    }
}
