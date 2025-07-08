namespace Jip.Ne;

public static class Helpers
{
    public static double[] Normalize(byte[] pixels)
    {
        var result = new double[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
            result[i] = pixels[i] / 255.0;
        return result;
    }

    public static double[] OneHot(int label, int size = 10)
    {
        var result = new double[size];
        result[label] = 1.0;
        return result;
    }
}
