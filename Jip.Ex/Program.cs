using Jip.Ne;

public static class Program
{
    static void Main()
    {
        string basePath = @"C:\Users\bapti\Desktop\MNIST";
        
        string imagePath = Path.Combine(basePath, "train-images.idx3-ubyte");

        if (!File.Exists(imagePath))
        {
            Console.WriteLine($"Le fichier {imagePath} n'existe pas !");
            return;
        }

        var images = MnistReader.ReadImages(Path.Combine(basePath, "train-images.idx3-ubyte"));
        var labels = MnistReader.ReadLabels(Path.Combine(basePath, "train-labels.idx1-ubyte"));

        var nn = new NeuralNetwork(28 * 28, 64, 10);

        int epochs = 3;
        double learningRate = 0.01;

        for (int e = 0; e < epochs; e++)
        {
            double lossSum = 0;
            int correct = 0;

            for (int i = 0; i < images.Length; i++)
            {
                var input = Helpers.Normalize(images[i]);
                var target = Helpers.OneHot(labels[i]);

                var (_, output) = nn.Forward(input);
                lossSum += nn.CrossEntropyLoss(output, target);

                int predicted = ArgMax(output);
                if (predicted == labels[i])
                    correct++;

                nn.Train(input, target, learningRate);

                if (i % 1000 == 0)
                    Console.WriteLine($"Epoch {e + 1}, Sample {i}, Loss: {lossSum / (i + 1):F4}, Accuracy: {(double)correct / (i + 1):P}");
            }
        }

        Console.WriteLine("Training terminé.");

        TestRandomSamples(nn, images, labels);
    }

    static int ArgMax(double[] array)
    {
        int idx = 0;
        double max = array[0];
        for (int i = 1; i < array.Length; i++)
        {
            if (array[i] > max)
            {
                max = array[i];
                idx = i;
            }
        }
        return idx;
    }
    static void TestRandomSamples(NeuralNetwork nn, byte[][] images, byte[] labels, int sampleCount = 10)
    {
        var rnd = new Random();

        Console.WriteLine("\nTest de 10 images aléatoires après entraînement :");
        for (int i = 0; i < sampleCount; i++)
        {
            int idx = rnd.Next(images.Length);
            var input = Helpers.Normalize(images[idx]);
            var (_, output) = nn.Forward(input);
            int predicted = ArgMax(output);
            byte actual = labels[idx];

            Console.WriteLine($"Image #{idx}: Prédit = {predicted}, Réel = {actual}");
        }
    }
}