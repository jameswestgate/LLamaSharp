using LLama.Common;
using LLama.Native;

namespace LLama.Examples.Examples
{
    public class CompareEmbeddings
    {
        public static async Task Run()
        {
            string modelPath = UserSettings.GetModelPath();

            Console.ForegroundColor = ConsoleColor.DarkGray;
            var @params = new ModelParams(modelPath)
            {
                // Embedding models can return one embedding per token, or all of them can be combined ("pooled") into
                // one single embedding. Setting PoolingType to "Mean" will combine all of the embeddings using mean average.
                PoolingType = LLamaPoolingType.Mean,
            };
            
            using var weights = await LLamaWeights.LoadFromFileAsync(@params);
            var embedder = new LLamaEmbedder(weights, @params);

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine(
                """
                This example embeddings from two text prompts.
                Embeddings are vectors that represent information like words, images, or concepts.
                These vector capture important relationships between those objects, 
                like how similar words are in meaning or how close images are visually.
                """);

            while (true)
            {
                Console.ForegroundColor = ConsoleColor.White;
                Console.Write("Please input your first text: ");
                Console.ForegroundColor = ConsoleColor.Green;
                var sentence1 = Console.ReadLine();
                Console.ForegroundColor = ConsoleColor.White;

                Console.ForegroundColor = ConsoleColor.White;
                Console.Write("Please input your second text: ");
                Console.ForegroundColor = ConsoleColor.Green;
                var sentence2 = Console.ReadLine();

                // Get embeddings, because we are using mean above, we should only get exacty one embedding vector for each sentence.
                Console.ForegroundColor = ConsoleColor.Gray;
                var embedding1 = (await embedder.GetEmbeddings(sentence1)).Single();
                var embedding2 = (await embedder.GetEmbeddings(sentence2)).Single();

                // Compute cosine similarity
                var similarity = CosineSimilarity(embedding1.ToArray(), embedding2.ToArray());

                Console.ForegroundColor = ConsoleColor.Gray;

                // Initialize tokenizer from weights and model params
                using var context = new LLamaContext(weights, @params);

                // Tokenize sentences
                var tokens1 = context.Tokenize(sentence1, true);
                var tokens2 = context.Tokenize(sentence2, true);

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
                Console.WriteLine($"Cosine similarity: {similarity:F4}. Press any key.");
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
