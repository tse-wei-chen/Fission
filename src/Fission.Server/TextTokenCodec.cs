using System.Text;

namespace Fission.Server;

public interface ITextTokenCodec
{
    int[] EncodePrompt(string prompt);
    int[] EncodeChat(IReadOnlyList<OpenAiChatMessage> messages);
    string DecodeToken(int tokenId);
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

    public string DecodeToken(int tokenId) => $"<{tokenId}>";
}
