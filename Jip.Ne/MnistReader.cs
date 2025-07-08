using System;
using System.IO;

public static class MnistReader
{
    public static byte[][] ReadImages(string path)
    {
        using var fs = new FileStream(path, FileMode.Open);
        using var br = new BinaryReader(fs);

        int magic = ReadInt32BigEndian(br);
        int count = ReadInt32BigEndian(br);
        int rows = ReadInt32BigEndian(br);
        int cols = ReadInt32BigEndian(br);

        var images = new byte[count][];

        for (int i = 0; i < count; i++)
        {
            var data = br.ReadBytes(rows * cols);
            images[i] = data;
        }

        return images;
    }

    public static byte[] ReadLabels(string path)
    {
        using var fs = new FileStream(path, FileMode.Open);
        using var br = new BinaryReader(fs);

        int magic = ReadInt32BigEndian(br);
        int count = ReadInt32BigEndian(br);

        var labels = br.ReadBytes(count);
        return labels;
    }

    private static int ReadInt32BigEndian(BinaryReader br)
    {
        var bytes = br.ReadBytes(4);
        if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
        return BitConverter.ToInt32(bytes, 0);
    }
}
