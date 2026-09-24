using Fission.Abstractions;
using Fission.Abstractions.Execution;
using Fission.Runtime.Backends;
using Fission.Runtime.Execution;

var model = new ModelId("demo-model");
var device = new DeviceId("cpu:0");

await using var executor = await ContinuousBatchExecutor.CreateAsync(
    new DeterministicBackend(device),
    capacity: 1024,
    maxBatchSize: 32);

var sequences = Enumerable.Range(0, 128)
    .Select(_ => SequenceId.New())
    .ToArray();

var prefillTasks = sequences
    .Select(sequence => executor.SubmitPrefillAsync(
        new PrefillItem(sequence, model, new[] { 1, 2, 3, 4 })).AsTask())
    .ToArray();

var prefillResults = await Task.WhenAll(prefillTasks);

var decodeTasks = prefillResults
    .Select((result, position) => executor.SubmitDecodeAsync(
        new DecodeItem(result.SequenceId, model, position + 4)).AsTask())
    .ToArray();

var decodeResults = await Task.WhenAll(decodeTasks);

if (decodeResults.Length != sequences.Length || decodeResults.Any(static result => result.TokenId < 0))
{
    throw new InvalidOperationException("Fission smoke execution failed.");
}

Console.WriteLine($"Fission smoke run completed: {decodeResults.Length} concurrent sequences on {device}.");
