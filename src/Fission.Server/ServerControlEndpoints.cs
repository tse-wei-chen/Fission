using System.Security.Cryptography;
using System.Text;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;

namespace Fission.Server;

public static class ServerControlEndpoints
{
    public const string ControlTokenHeader = "X-Fission-Control-Token";
    public const string ShutdownPath = "/internal/control/shutdown";

    public static void Map(
        WebApplication app,
        IConfiguration configuration)
    {
        ArgumentNullException.ThrowIfNull(app);
        ArgumentNullException.ThrowIfNull(configuration);

        var expectedToken = configuration["Fission:ControlToken"];
        if (string.IsNullOrWhiteSpace(expectedToken))
        {
            return;
        }

        expectedToken = expectedToken.Trim();
        app.MapPost(
            ShutdownPath,
            (HttpContext context, IHostApplicationLifetime lifetime) =>
            {
                var providedToken =
                    context.Request.Headers[ControlTokenHeader].ToString();
                if (!TokensEqual(expectedToken, providedToken))
                {
                    return Results.Unauthorized();
                }

                context.Response.OnCompleted(() =>
                {
                    lifetime.StopApplication();
                    return Task.CompletedTask;
                });

                return Results.Accepted(
                    ShutdownPath,
                    new { status = "stopping" });
            });
    }

    private static bool TokensEqual(
        string expected,
        string provided)
    {
        if (string.IsNullOrEmpty(provided))
        {
            return false;
        }

        var expectedBytes = Encoding.UTF8.GetBytes(expected);
        var providedBytes = Encoding.UTF8.GetBytes(provided);
        return expectedBytes.Length == providedBytes.Length &&
            CryptographicOperations.FixedTimeEquals(
                expectedBytes,
                providedBytes);
    }
}
