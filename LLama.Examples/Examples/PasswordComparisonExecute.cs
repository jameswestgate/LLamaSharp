using LLama.Common;
using LLama.Native;
using LLama.Sampling;
using System.Diagnostics;
using System.Text;

namespace LLama.Examples.Examples
{
    public class PasswordComparisonExecute
    {
        public static async Task Run()
        {
            var stopwatch = new Stopwatch();
            var modelPath = UserSettings.GetModelPath();
            var promptTemplate = (await File.ReadAllTextAsync("Assets/password-tokenise.txt")).Trim();

            var parameters = new ModelParams(modelPath)
            {
                GpuLayerCount = 0,
                ContextSize = 2048,
            };
            stopwatch.Start();

            using var model = await LLamaWeights.LoadFromFileAsync(parameters);
            var executor = new StatelessExecutor(model, parameters);

            var inferenceParams = new InferenceParams
            {
                SamplingPipeline = new DefaultSamplingPipeline { Temperature = 0.0f },
                AntiPrompts = new List<string> { "Question:", "#", "Question: ", ".\n" },
                MaxTokens = 600
            };

            stopwatch.Stop();
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"Model loaded in {stopwatch.ElapsedMilliseconds} ms.");

            Console.Write("Enter first password: ");
            Console.ForegroundColor = ConsoleColor.White;
            var password1 = Console.ReadLine();

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write("Enter second password: ");
            Console.ForegroundColor = ConsoleColor.White;
            var password2 = Console.ReadLine();

            stopwatch.Restart();
            var result1 = await GetModelOutput(executor, inferenceParams, promptTemplate.Replace("{password}", password1));
            stopwatch.Stop();

            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"First password semantic tokens: ");
            Console.ForegroundColor = ConsoleColor.Blue;
            Console.Write(result1);
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine($" ({stopwatch.ElapsedMilliseconds} ms)");
            Console.ForegroundColor = ConsoleColor.Gray;

            stopwatch.Restart();
            var result2 = await GetModelOutput(executor, inferenceParams, promptTemplate.Replace("{password}", password2));
            stopwatch.Stop();

            Console.ForegroundColor = ConsoleColor.White;
            Console.Write($"Second password semantic tokens: ");
            Console.ForegroundColor = ConsoleColor.Blue;
            Console.Write(result2);
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine($" ({stopwatch.ElapsedMilliseconds} ms)");
            Console.ForegroundColor = ConsoleColor.Gray;

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("\nGenerating embeddings and calculating similarity...");
            Console.ForegroundColor = ConsoleColor.White;

            var embeddingModel = "all-MiniLM-L12-v2.Q8_0.gguf";
            var embeddingPath = Path.Combine(Path.GetDirectoryName(modelPath), embeddingModel);

            var embedParams = new ModelParams(embeddingPath)
            {
                PoolingType = LLamaPoolingType.Mean
            };

            using var weights = await LLamaWeights.LoadFromFileAsync(embedParams);
            var embedder = new LLamaEmbedder(weights, embedParams);

            using var context = new LLamaContext(weights, embedParams);

            //Write out the embeddings
            var tokens1 = context.Tokenize(result1, true);
            var tokens2 = context.Tokenize(result2, true);

            var decoder = new StreamingTokenDecoder(context);
            var tokens = new List<IReadOnlyList<LLamaToken>> { tokens1, tokens2 };

            //Write out the text representation of each token
            foreach (var list in tokens)
            {
                var output = new List<string>();

                foreach (var token in list)
                {
                    decoder.Add(token);
                    output.Add(decoder.Read().Trim());
                }

                Console.ForegroundColor = ConsoleColor.White;
                Console.Write($"Got embeddings: ");
                Console.ForegroundColor = ConsoleColor.Blue;
                Console.WriteLine($"{string.Join(' ', output)}");
                Console.ForegroundColor = ConsoleColor.Gray;
            }

            var embeddingVec1 = (await embedder.GetEmbeddings(result1)).Single();
            var embeddingVec2 = (await embedder.GetEmbeddings(result2)).Single();

            var similarity = CosineSimilarity([.. embeddingVec1], [.. embeddingVec2]);

            Console.ForegroundColor = ConsoleColor.White;
            Console.Write("Cosine similarity between the password outputs: ");
            Console.ForegroundColor = ConsoleColor.Blue;
            Console.WriteLine($"{similarity:F4}");
            Console.ForegroundColor = ConsoleColor.Gray;
        }

        //ChatGPT modified version that times out after 30 seconds
        private static async Task<string?> GetModelOutput(StatelessExecutor executor, InferenceParams inferenceParams, string prompt)
        {
            var resultBuilder = new StringBuilder();

            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(20));

            try
            {
                await foreach (var text in executor.InferAsync(prompt, inferenceParams, cts.Token))
                {
                    resultBuilder.Append(text);
                }

                // Sometimes, there is a hallucination separated by a CRLF, so we trim the result
                var result = resultBuilder.ToString().Trim();

                if (result.Contains('\n'))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"Hallucination detected.");
                    Console.ForegroundColor = ConsoleColor.Gray;

                    var splits = result.Split('\n', StringSplitOptions.RemoveEmptyEntries);
                    result = splits[1];
                }

                // Sometimes it hallicinates from the prompt examples e.g. 
                //First password (@) semantic tokens: Result: 12345 abcde (10744 ms)
                if (result.StartsWith("Result:"))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"Prompt hallucination artifact detected.");
                    Console.ForegroundColor = ConsoleColor.Gray;

                    result = result.Substring("Result:".Length);
                }

                return result;
            }
            catch (OperationCanceledException)
            {
                // Timeout occurred
                Console.WriteLine("Inference timed out after 30 seconds.");
                throw;
            }
        }



        private static float CosineSimilarity(float[] vec1, float[] vec2)
        {
            float dot = 0f, normA = 0f, normB = 0f;

            for (var i = 0; i < vec1.Length; i++)
            {
                dot += vec1[i] * vec2[i];
                normA += vec1[i] * vec1[i];
                normB += vec2[i] * vec2[i];
            }

            return dot / (float)(Math.Sqrt(normA) * Math.Sqrt(normB));
        }
    }
}
