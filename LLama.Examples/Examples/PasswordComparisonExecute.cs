using LLama.Common;
using LLama.Examples.Extensions;
using LLama.Sampling;

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
            var ex = new StatelessExecutor(model, parameters)
            {
                ApplyTemplate = false,
                SystemMessage = "You are a helpful bot."
            };

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Loading model and running initial prompt");
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

            prompt = prompt.Replace("{password}", "HelloWorld1974");

            await foreach (var text in ex.InferAsync(prompt, inferenceParams).Spinner())
            {
                Console.Write(text);
            }
            
        }
    }
}
