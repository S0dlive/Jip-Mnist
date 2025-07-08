namespace Jip.Ne;

using System;

public class NeuralNetwork
{
    int inputSize;
    int hiddenSize;
    int outputSize;

    double[,] W1; 
    double[] B1;
    double[,] W2; 
    double[] B2;

    Random rnd;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
    {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        rnd = new Random();

        W1 = new double[inputSize, hiddenSize];
        B1 = new double[hiddenSize];
        W2 = new double[hiddenSize, outputSize];
        B2 = new double[outputSize];

        InitWeights(W1);
        InitWeights(W2);
        InitBiases(B1);
        InitBiases(B2);
    }

    private void InitWeights(double[,] weights)
    {
        int rows = weights.GetLength(0);
        int cols = weights.GetLength(1);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                weights[i, j] = rnd.NextDouble() * 0.1 - 0.05;
    }

    private void InitBiases(double[] biases)
    {
        for (int i = 0; i < biases.Length; i++)
            biases[i] = 0;
    }

    private double[] ReLU(double[] x)
    {
        var res = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
            res[i] = Math.Max(0, x[i]);
        return res;
    }

    private double[] ReLUDerivative(double[] x)
    {
        var res = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
            res[i] = x[i] > 0 ? 1 : 0;
        return res;
    }
    private double[] Softmax(double[] x)
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < x.Length; i++)
            if (x[i] > max) max = x[i];

        double sum = 0;
        double[] exp = new double[x.Length];
        for (int i = 0; i < x.Length; i++)
        {
            exp[i] = Math.Exp(x[i] - max);
            sum += exp[i];
        }

        for (int i = 0; i < x.Length; i++)
            exp[i] /= sum;

        return exp;
    }
    
    public (double[] hidden, double[] output) Forward(double[] input)
    {
        double[] hiddenRaw = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++)
        {
            double sum = B1[j];
            for (int i = 0; i < inputSize; i++)
                sum += input[i] * W1[i, j];
            hiddenRaw[j] = sum;
        }

        var hidden = ReLU(hiddenRaw);

        double[] outputRaw = new double[outputSize];
        for (int k = 0; k < outputSize; k++)
        {
            double sum = B2[k];
            for (int j = 0; j < hiddenSize; j++)
                sum += hidden[j] * W2[j, k];
            outputRaw[k] = sum;
        }

        var output = Softmax(outputRaw);
        return (hidden, output);
    }

    public double CrossEntropyLoss(double[] predicted, double[] target)
    {
        double loss = 0;
        for (int i = 0; i < predicted.Length; i++)
        {
            loss -= target[i] * Math.Log(predicted[i] + 1e-15);
        }
        return loss;
    }

    public void Train(double[] input, double[] target, double learningRate)
    {
        var (hidden, output) = Forward(input);
        
        double[] deltaOutput = new double[outputSize];
        for (int i = 0; i < outputSize; i++)
        {
            deltaOutput[i] = output[i] - target[i]; 
        }

        double[] deltaHidden = new double[hiddenSize];
        var dRelu = ReLUDerivative(hidden);
        for (int j = 0; j < hiddenSize; j++)
        {
            double sum = 0;
            for (int k = 0; k < outputSize; k++)
            {
                sum += deltaOutput[k] * W2[j, k];
            }
            deltaHidden[j] = sum * dRelu[j];
        }

        for (int j = 0; j < hiddenSize; j++)
        {
            for (int k = 0; k < outputSize; k++)
            {
                W2[j, k] -= learningRate * deltaOutput[k] * hidden[j];
            }
        }

        for (int k = 0; k < outputSize; k++)
            B2[k] -= learningRate * deltaOutput[k];
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                W1[i, j] -= learningRate * deltaHidden[j] * input[i];
            }
        }

        for (int j = 0; j < hiddenSize; j++)
            B1[j] -= learningRate * deltaHidden[j];
    }
}
