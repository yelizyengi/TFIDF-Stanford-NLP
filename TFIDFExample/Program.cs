using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TFIDFExample
{
    class Program
    {
        public static void Main(string[] args)
        {
            // Some example documents.
            //string[] documents = ReadAllTextFolder();
            
            // Apply TF*IDF to the documents and get the resulting vectors.
            //double[][] inputs = TFIDF.Transform(documents, 0);
            //inputs = TFIDF.Normalize(inputs);

        }

        public static string[] ReadAllTextFolder()
        {
            string[] wordList;
            string[] uri = { @"C:\Users\YELLOW\Downloads\aclImdb\train\neg\", 
                                        @"C:\Users\YELLOW\Downloads\aclImdb\train\pos\" };
            var countOfText = 1000;
            var index = 0;
            var count = countOfText * uri.Count();
            wordList = new string[count];
            for (int i = 0; i < uri.Count(); i++)
            {
                var index2 = 0;
                foreach (string file in Directory.EnumerateFiles(uri[i], "*.txt"))
                {
                    string contents = File.ReadAllText(file);
                    wordList[index] = contents;
                    index++;
                    index2++;
                    //labelofcomment.Add(i);
                    if (index2 >= countOfText) break;
                }
            }

            return wordList;
        }
    }
}
