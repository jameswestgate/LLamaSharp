using DocumentFormat.OpenXml.InkML;
using LLama.Common;
using LLama.Examples.Extensions;
using LLama.Native;
using LLama.Sampling;
using System.Text;

namespace LLama.Examples.Examples
{
    public class PasswordComparisonExecute
    {
        public static async Task Run()
        {
            var modelPath = UserSettings.GetModelPath();
            var prompt = (await File.ReadAllTextAsync("Assets/password-tokenise.txt")).Trim();

            var parameters = new ModelParams(modelPath)
            {
                // Ensure GPU layers are disabled — this forces CPU-only execution
                GpuLayerCount = 0,

                // Optional: set context size and other performance-related settings
                ContextSize = 2048,
            };

            using var model = await LLamaWeights.LoadFromFileAsync(parameters);
            var ex = new StatelessExecutor(model, parameters);

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Loading the generative model and running initial prompt. ");
            Console.ForegroundColor = ConsoleColor.White;

            var inferenceParams = new InferenceParams
            {
                SamplingPipeline = new DefaultSamplingPipeline
                {
                    Temperature = 0.0f
                },

                AntiPrompts = new List<string> { "Question:", "#", "Question: ", ".\n" },
                MaxTokens = 600
            };

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Enter your password:");
            Console.ForegroundColor = ConsoleColor.White;

            Console.ForegroundColor = ConsoleColor.Green;
            var password = Console.ReadLine();
            Console.ForegroundColor = ConsoleColor.White;

            prompt = prompt.Replace("{password}", password);

            var result = new StringBuilder();

            await foreach (var text in ex.InferAsync(prompt, inferenceParams).Spinner())
            {
                Console.Write(text);
                result.Append(text);
            }

            var comparisons = File.ReadAllLines("Assets/password-comparisons.txt").ToList();

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Loading the embedding model and running comparisons ...");
            Console.ForegroundColor = ConsoleColor.White;

            //DEtermine the path to the embedding model given the model path of the generative model
            var embeddingModel = "all-MiniLM-L12-v2.Q8_0.gguf";

            // Replace the filename in modelPath with myFile
            var newPath = Path.Combine(Path.GetDirectoryName(modelPath), embeddingModel);

            Console.ForegroundColor = ConsoleColor.DarkGray;
            var @params = new ModelParams(newPath)
            {
                // Embedding models can return one embedding per token, or all of them can be combined ("pooled") into
                // one single embedding. Setting PoolingType to "Mean" will combine all of the embeddings using mean average.
                PoolingType = LLamaPoolingType.Mean,
            };

            using var weights = await LLamaWeights.LoadFromFileAsync(@params);
            var embedder = new LLamaEmbedder(weights, @params);

            // Initialize tokenizer from weights and model params
            using var context = new LLamaContext(weights, @params);

            // Get embeddings, because we are using mean above, we should only get exacty one embedding vector for each sentence.
            Console.ForegroundColor = ConsoleColor.Gray;
            var embedding1 = (await embedder.GetEmbeddings(result.ToString())).Single();
            var tokens1 = context.Tokenize(result.ToString(), true);

            foreach (var comparison in comparisons)
            {
                var compare = comparison.Split(':').ElementAtOrDefault(1) ?? string.Empty;
                var embedding2 = (await embedder.GetEmbeddings(compare)).Single();

                // Compute cosine similarity
                var similarity = CosineSimilarity(embedding1.ToArray(), embedding2.ToArray());

                Console.ForegroundColor = ConsoleColor.Gray;

                // Tokenize sentences
        
                var tokens2 = context.Tokenize(compare, true);

                var decoder = new StreamingTokenDecoder(context);
                var tokens = new List<IReadOnlyList<LLamaToken>> { tokens1, tokens2 };

                //Write out the text representation of each token
                var flag = false;

                foreach (var list in tokens)
                {
                    foreach (var token in list)
                    {
                        if (flag) Console.Write("|");
                        decoder.Add(token);
                        Console.Write(decoder.Read());
                        flag = true;
                    }
                    flag = false;
                    Console.WriteLine();
                }

                Console.ForegroundColor = ConsoleColor.Blue;
                Console.WriteLine($"Cosine similarity: {similarity:F4}.");
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
