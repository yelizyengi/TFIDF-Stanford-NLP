using EnglishStemmer;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using java.io;
using edu.stanford.nlp.process;
using edu.stanford.nlp.ling;
using edu.stanford.nlp.trees;
using edu.stanford.nlp.parser.lexparser;
using Console = System.Console;
using edu.stanford.nlp.tagger.maxent;
using java.util;
using Collections = System.Collections;

namespace TFIDFExample
{
    /// <summary>
    /// Copyright (c) 2013 Kory Becker http://www.primaryobjects.com/kory-becker.aspx
    /// 
    /// Permission is hereby granted, free of charge, to any person obtaining
    /// a copy of this software and associated documentation files (the
    /// "Software"), to deal in the Software without restriction, including
    /// without limitation the rights to use, copy, modify, merge, publish,
    /// distribute, sublicense, and/or sell copies of the Software, and to
    /// permit persons to whom the Software is furnished to do so, subject to
    /// the following conditions:
    /// 
    /// The above copyright notice and this permission notice shall be
    /// included in all copies or substantial portions of the Software.
    /// 
    /// Description:
    /// Performs a TF*IDF (Term Frequency * Inverse Document Frequency) transformation on an array of documents.
    /// Each document string is transformed into an array of doubles, cooresponding to their associated TF*IDF values.
    /// 
    /// Usage:
    /// string[] documents = LoadYourDocuments();
    ///
    /// double[][] inputs = TFIDF.Transform(documents);
    /// inputs = TFIDF.Normalize(inputs);
    /// 
    /// </summary>
    public static class TFIDF
    {

        /// <summary>
        /// Document vocabulary, containing each word's IDF value.
        /// </summary> @"C:\Users\YELLOW\Downloads\aclImdb\train\pos\" 
        private static Dictionary<string, double> _vocabularyIDF        = new Dictionary<string, double>();
        public static List<int> countOfEmptyTFIDF                       = new List<int>();
        public static List<int> LabelOfContent                          = new List<int>();
        public static List<string> vocabulary                           = new List<string>();
        public static List<string> uriComment                           = new List<string>();
        public static List<string> tagList                              = new List<string> { "JJ", "RB", "RBR", "RBS", "NN", "NNS", "VB", "VBD", "VBG", "VBN" };
        public static string[] wordList;
        public static MaxentTagger tagger;

        public static string[] ReadAllTextFolder()
        {
            // URL Düzelt
            string[] uri = { @"C:\Users\userr\Downloads\movies", @"C:\Users\YELLOW\Desktop\tokens\pos\" };
            var countOfText = 700;
            var index       = 0;
            var count       = countOfText * uri.Count();
            wordList        = new string[count];

            for (int i = 0; i < uri.Count(); i++)
            {
                var index2 = 0;
                var files = Directory.EnumerateFiles(uri[i], "*.txt");
                foreach (string file in  files)
                {
                    string contents     = System.IO.File.ReadAllText(file);
                    wordList[index] = contents;
                    index++;
                    index2++;
                    LabelOfContent.Add(i);
                    uriComment.Add(file);
                    if (index2 >= countOfText) break;
                }
            }
            //DeleteUnUsePOSTag(wordList);
            return wordList;
        }

        private static void initialLanguagePattern()
        {
            //URL Düzelt
            var jarRoot         = @"J:\ToltecSoft.WebCrawler";
            var modelsDirectory = jarRoot + @"\models";
            tagger              = new MaxentTagger(modelsDirectory + @"\english-left3words-distsim.tagger");
        }

        private static List<string> DeleteUnUsePOSTag(List<string> parts)
        {
            //System.IO.File.AppendAllText("J:\\ToltecSoft.WebCrawler\\taggedSentence.txt", Environment.NewLine);
            
            List<string> taggedPart = new List<string>();
            var sentences = MaxentTagger.tokenizeText(new java.io.StringReader(string.Join(" ", parts))).toArray();
            foreach (ArrayList sentence in sentences)
            {
                var taggedSentence = tagger.tagSentence(sentence).toArray();
                
                foreach (TaggedWord item in taggedSentence)
                {
                    //System.IO.File.AppendAllText("J:\\ToltecSoft.WebCrawler\\taggedSentence.txt", item.ToString());
                    if(tagList.Contains(item.tag())) taggedPart.Add(item.word());
                }             
            }
            return taggedPart;
        }

        static string ConvertStringArrayToString(string[] array)
        {
            //
            // Concatenate all the elements into a StringBuilder.
            //
            StringBuilder builder = new StringBuilder();
            foreach (string value in array)
            {
                builder.Append(value);
                builder.Append(' ');
            }
            return builder.ToString();
        }
        /// <summary>
        /// Transforms a list of documents into their associated TF*IDF values.
        /// If a vocabulary does not yet exist, one will be created, based upon the documents' words.
        /// </summary>
        /// <param name="documents">string[]</param>
        /// <param name="vocabularyThreshold">Minimum number of occurences of the term within all documents</param>
        /// <returns>double[][]</returns>
        public static List<List<double>> Transform()
        {
            int vocabularyThreshold = 0;
            List<List<string>> stemmedDocs;
            List<string> vocabulary;
            string[] documents = ReadAllTextFolder();
            // Get the vocabulary and stem the documents at the same time.
            vocabulary = GetVocabulary(documents, out stemmedDocs, vocabularyThreshold);

            if (_vocabularyIDF.Count == 0)
            {
                // Calculate the IDF for each vocabulary term.
                foreach (var term in vocabulary)
                {
                    double numberOfDocsContainingTerm = stemmedDocs.Where(d => d.Contains(term)).Count();
                   // System.IO.File.AppendAllText("J:\\ToltecSoft.WebCrawler\\ToltecSoft.WebCrawler\\Negative.txt", term.ToString() + " : "+ numberOfDocsContainingTerm.ToString() + Environment.NewLine);
                    _vocabularyIDF[term] = Math.Log((double)stemmedDocs.Count / ((double)1 + numberOfDocsContainingTerm));
                }
            }

            // Transform each document into a vector of tfidf values.
            return TransformToTFIDFVectors(stemmedDocs, _vocabularyIDF);
        }

        /// <summary>
        /// Converts a list of stemmed documents (lists of stemmed words) and their associated vocabulary + idf values, into an array of TF*IDF values.
        /// </summary>
        /// <param name="stemmedDocs">List of List of string</param>
        /// <param name="vocabularyIDF">Dictionary of string, double (term, IDF)</param>
        /// <returns>double[][]</returns>
        private static List<List<double>> TransformToTFIDFVectors(List<List<string>> stemmedDocs, Dictionary<string, double> vocabularyIDF)
        {
            // Transform each document into a vector of tfidf values.
            List<List<double>> vectors = new List<List<double>>();
            foreach (var doc in stemmedDocs)
            {
                List<double> vector = new List<double>();
                var countOfEmpty = 0;
                foreach (var vocab in vocabularyIDF)
                {
                    // Term frequency = count how many times the term appears in this document.
                    double tf = doc.Where(d => d == vocab.Key).Count();
                    double tfidf = tf * vocab.Value;
                    tfidf = Math.Round(tfidf, 6);
                    if (tfidf == 0.0) countOfEmpty++;
                    vector.Add(tfidf);
                }
                countOfEmptyTFIDF.Add(countOfEmpty);
                vectors.Add(vector);
            }

            return vectors;
        }

        /// <summary>
        /// Normalizes a TF*IDF array of vectors using L2-Norm.
        /// Xi = Xi / Sqrt(X0^2 + X1^2 + .. + Xn^2)
        /// </summary>
        /// <param name="vectors">double[][]</param>
        /// <returns>double[][]</returns>
        public static List<List<double>> Normalize(List<List<double>> vectors)
        {
            // Normalize the vectors using L2-Norm.
            List<List<double>> normalizedVectors = new List<List<double>>();
            foreach (var vector in vectors)
            {
                var normalized = Normalize(vector);
                normalizedVectors.Add(normalized);
            }

            return normalizedVectors;
        }

        /// <summary>
        /// Normalizes a TF*IDF vector using L2-Norm.
        /// Xi = Xi / Sqrt(X0^2 + X1^2 + .. + Xn^2)
        /// </summary>
        /// <param name="vectors">double[][]</param>
        /// <returns>double[][]</returns>
        public static List<double> Normalize(List<double> vector)
        {
            List<double> result = new List<double>();

            double sumSquared = 0;
            foreach (var value in vector)
            {
                sumSquared += value * value;
            }

            double SqrtSumSquared = Math.Sqrt(sumSquared);

            foreach (var value in vector)
            {
                // L2-norm: Xi = Xi / Sqrt(X0^2 + X1^2 + .. + Xn^2)
                result.Add(Math.Round((value / SqrtSumSquared), 4));
            }

            return result;
        }

        /// <summary>
        /// Saves the TFIDF vocabulary to disk.
        /// </summary>
        /// <param name="filePath">File path</param>
        public static void Save(string filePath = "vocabulary.dat")
        {
            // Save result to disk.
            using (FileStream fs = new FileStream(filePath, FileMode.Create))
            {
                BinaryFormatter formatter = new BinaryFormatter();
                formatter.Serialize(fs, _vocabularyIDF);
            }
        }

        /// <summary>
        /// Loads the TFIDF vocabulary from disk.
        /// </summary>
        /// <param name="filePath">File path</param>
        public static void Load(string filePath = "vocabulary.dat")
        {
            // Load from disk.
            using (FileStream fs = new FileStream(filePath, FileMode.Open))
            {
                BinaryFormatter formatter = new BinaryFormatter();
                _vocabularyIDF = (Dictionary<string, double>)formatter.Deserialize(fs);
            }
        }

        #region Private Helpers

        /// <summary>
        /// Parses and tokenizes a list of documents, returning a vocabulary of words.
        /// </summary>
        /// <param name="docs">string[]</param>
        /// <param name="stemmedDocs">List of List of string</param>
        /// <returns>Vocabulary (list of strings)</returns>
        private static List<string> GetVocabulary(string[] docs, out List<List<string>> stemmedDocs, int vocabularyThreshold)
        {
            
            Dictionary<string, int> wordCountList = new Dictionary<string, int>();
            stemmedDocs = new List<List<string>>();

            int docIndex = 0;
            initialLanguagePattern();
            foreach (var doc in docs)
            {
                List<string> stemmedDoc = new List<string>();

                docIndex++;

                if (docIndex % 100 == 0)
                {
                    System.Console.WriteLine("Processing " + docIndex + "/" + docs.Length);
                }

                string[] parts2 = Tokenize(doc);                
                List<string> words = new List<string>();
                foreach (string part in parts2)
                {
                    // Strip non-alphanumeric characters.
                    string stripped = Regex.Replace(part, "[^a-zA-Z0-9]", "");

                    if (!StopWords.stopWordsList.Contains(stripped.ToLower()))
                    {
                        try
                        {
                            var english = new EnglishWord(stripped);
                            string stem = english.Stem;
                            words.Add(stem);

                            if (stem.Length > 0)
                            {
                                // Build the word count list.
                                if (wordCountList.ContainsKey(stem))
                                {
                                    wordCountList[stem]++;
                                }
                                else
                                {
                                    wordCountList.Add(stem, 0);
                                }

                                stemmedDoc.Add(stem);
                            }
                        }
                        catch
                        {
                        }
                    }
                }
                
                stemmedDocs.Add(DeleteUnUsePOSTag(stemmedDoc));
            }
            
            // Get the top words.
            var vocabList = wordCountList.Where(w => w.Value >= vocabularyThreshold);
            foreach (var item in vocabList)
            {
                vocabulary.Add(item.Key);
                //System.IO.File.AppendAllText("J:\\ToltecSoft.WebCrawler\\ToltecSoft.WebCrawler\\wordList.txt", item.Key + "," + Environment.NewLine); 
            }

            return vocabulary;
        }

        /// <summary>
        /// Tokenizes a string, returning its list of words.
        /// </summary>
        /// <param name="text">string</param>
        /// <returns>string[]</returns>
        private static string[] Tokenize(string text)
        {
            // Strip all HTML.
            text = Regex.Replace(text, "<[^<>]+>", "");

            // Strip numbers.
            text = Regex.Replace(text, "[0-9]+", "number");

            // Strip urls.
            text = Regex.Replace(text, @"(http|https)://[^\s]*", "httpaddr");

            // Strip email addresses.
            text = Regex.Replace(text, @"[^\s]+@[^\s]+", "emailaddr");

            // Strip dollar sign.
            text = Regex.Replace(text, "[$]+", "dollar");

            // Strip usernames.
            text = Regex.Replace(text, @"@[^\s]+", "username");

            // Tokenize and also get rid of any punctuation
            return text.Split(" @$/#.-:&*+=[]?!(){},''\">_<;%\\".ToCharArray());
        }

        #endregion
    }
}
