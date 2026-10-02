using System.Text;

namespace Fission.Server;

public interface ITextTokenDecoder : IDisposable
{
    /// <summary>
    /// Appends one generated token and returns text that is stable enough to emit.
    /// A contextual tokenizer may return an empty string until later token ids
    /// complete a byte sequence or otherwise make the decoded suffix stable.
    /// </summary>
    string Append(int tokenId);

    /// <summary>
    /// Completes the request-scoped decode and returns any remaining text.
    /// </summary>
    string Complete();
}

public interface ITextTokenCodec : IDisposable
{
    int[] EncodePrompt(string prompt);
    int[] EncodeChat(IReadOnlyList<OpenAiChatMessage> messages);
    ITextTokenDecoder CreateDecoder();
}

/// <summary>
/// Transport-only codec used by the deterministic backend and serving specs.
/// Real model integrations should replace this with the model tokenizer/chat
/// template without changing the HTTP/SSE surface.
/// </summary>
public sealed class DeterministicTextTokenCodec : ITextTokenCodec
{
    public int[] EncodePrompt(string prompt)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(prompt);
        return Encoding.UTF8.GetBytes(prompt).Select(static value => (int)value).ToArray();
    }

    public int[] EncodeChat(IReadOnlyList<OpenAiChatMessage> messages)
    {
        ArgumentNullException.ThrowIfNull(messages);
        if (messages.Count == 0)
        {
            throw new ArgumentException("At least one chat message is required.", nameof(messages));
        }

        var builder = new StringBuilder();
        foreach (var message in messages)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(message.Role);
            ArgumentException.ThrowIfNullOrWhiteSpace(message.Content);
            builder.Append("<|").Append(message.Role).Append("|>\n")
                .Append(message.Content).Append('\n');
        }

        builder.Append("<|assistant|>\n");
        return EncodePrompt(builder.ToString());
    }

    public ITextTokenDecoder CreateDecoder() => new Decoder();

    public void Dispose() { }

    private sealed class Decoder : ITextTokenDecoder
    {
        public string Append(int tokenId) => $"<{tokenId}>";
        public string Complete() => string.Empty;
        public void Dispose() { }
    }
}
