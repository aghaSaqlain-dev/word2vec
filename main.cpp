#include <iostream>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <limits>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <omp.h>

using namespace std;

// Cosine similarity between two vectors
double cosineSimilarity(const vector<double> &vec1, const vector<double> &vec2)
{
    double dotProduct = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (size_t i = 0; i < vec1.size(); i++)
    {
        dotProduct += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    if (norm1 == 0.0 || norm2 == 0.0)
        return 0.0;
    return dotProduct / (sqrt(norm1) * sqrt(norm2));
}

// Find the most similar word from the context matrix
string findMostSimilarWord(const vector<double> &wordEmbedding,
                           const vector<vector<double>> &contextMatrix,
                           const vector<string> &vocabulary)
{
    double maxSimilarity = -1.0;
    size_t bestWordIndex = 0;
    for (size_t i = 0; i < vocabulary.size(); i++)
    {
        vector<double> contextVector(contextMatrix.size());
        for (size_t j = 0; j < contextMatrix.size(); j++)
            contextVector[j] = contextMatrix[j][i];
        double similarity = cosineSimilarity(wordEmbedding, contextVector);
        if (similarity > maxSimilarity)
        {
            maxSimilarity = similarity;
            bestWordIndex = i;
        }
    }
    return vocabulary[bestWordIndex];
}

// Fill a matrix with random values
void fillMatrixWithRandomValues(vector<vector<double>> &matrix, int rows, int cols)
{
    srand(static_cast<unsigned>(time(0)));
    matrix.resize(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix[i][j] = static_cast<double>(rand()) / RAND_MAX;
}

// Efficiently build vocabulary from words
vector<string> buildVocabulary(const vector<string> &words)
{
    unordered_set<string> vocabSet(words.begin(), words.end());
    return vector<string>(vocabSet.begin(), vocabSet.end());
}

// Print matrix with vocabulary labels
void printMatrix(const vector<vector<double>> &matrix, const vector<string> &vocabulary, const string &title)
{
    cout << title << endl;
    for (size_t i = 0; i < matrix.size(); i++)
    {
        if (!vocabulary.empty())
            cout << vocabulary[i] << ": ";
        for (const auto &val : matrix[i])
            cout << val << " ";
        cout << endl;
    }
    cout << "------------------------" << endl;
}

// Print context matrix
void printContextMatrix(const vector<vector<double>> &matrix, const string &title)
{
    cout << title << endl;
    for (const auto &row : matrix)
    {
        for (const auto &val : row)
            cout << val << " ";
        cout << endl;
    }
    cout << "------------------------" << endl;
}

// Training loop
void trainModel(
    vector<vector<double>> &embeddingMatrix,
    vector<vector<double>> &contextMatrix,
    const vector<string> &words,
    const vector<string> &vocabulary,
    const unordered_map<string, size_t> &wordToIndex,
    int numEpochs,
    double learningRate)
{
    for (int epoch = 0; epoch < numEpochs; epoch++)
    {
        double totalLoss = 0.0;

        // Prepare global gradients
        vector<vector<double>> contextMatrixGrad(contextMatrix.size(), vector<double>(contextMatrix[0].size(), 0.0));
        vector<vector<double>> embeddingMatrixGrad(embeddingMatrix.size(), vector<double>(embeddingMatrix[0].size(), 0.0));

        int nThreads = 1;
        #pragma omp parallel
        {
            #pragma omp single
            nThreads = omp_get_num_threads();
        }

        // Thread-local gradient buffers
        vector<vector<vector<double>>> threadContextGrads(nThreads, vector<vector<double>>(contextMatrix.size(), vector<double>(contextMatrix[0].size(), 0.0)));
        vector<vector<vector<double>>> threadEmbeddingGrads(nThreads, vector<vector<double>>(embeddingMatrix.size(), vector<double>(embeddingMatrix[0].size(), 0.0)));
        vector<double> threadLosses(nThreads, 0.0);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto &localContextGrad = threadContextGrads[tid];
            auto &localEmbeddingGrad = threadEmbeddingGrads[tid];
            double &localLoss = threadLosses[tid];

            #pragma omp for schedule(static)
            for (size_t i = 0; i < words.size(); ++i)
            {
                vector<double> oneHot(vocabulary.size(), 0.0);
                auto it = wordToIndex.find(words[i]);
                if (it == wordToIndex.end())
                    continue;
                size_t currentWordIndex = it->second;
                oneHot[currentWordIndex] = 1.0;

                // Forward pass
                vector<double> embeddingResult(embeddingMatrix[0].size(), 0.0);
                for (size_t k = 0; k < embeddingMatrix[0].size(); ++k)
                    for (size_t l = 0; l < vocabulary.size(); ++l)
                        embeddingResult[k] += oneHot[l] * embeddingMatrix[l][k];

                vector<double> contextResult(vocabulary.size(), 0.0);
                for (size_t k = 0; k < vocabulary.size(); ++k)
                    for (size_t l = 0; l < embeddingResult.size(); ++l)
                        contextResult[k] += embeddingResult[l] * contextMatrix[l][k];

               vector<double> softmaxResult(vocabulary.size(), 0.0);
double maxVal = *max_element(contextResult.begin(), contextResult.end());
double sumExp = 0.0;
for (const auto &val : contextResult)
    sumExp += exp(val - maxVal);
for (size_t k = 0; k < contextResult.size(); ++k)
    softmaxResult[k] = exp(contextResult[k] - maxVal) / sumExp;
                // Target: next word
                if (i + 1 < words.size())
                {
                    auto targetIt = wordToIndex.find(words[i + 1]);
                    if (targetIt == wordToIndex.end())
                        continue;
                    size_t targetWordIndex = targetIt->second;

                    const double EPS = 1e-10;
double crossEntropyLoss = -log(softmaxResult[targetWordIndex] + EPS);
                    localLoss += crossEntropyLoss;

                    // Backpropagation
                    vector<double> softmaxGradient = softmaxResult;
                    softmaxGradient[targetWordIndex] -= 1.0;

                    // Gradients
                    for (size_t d = 0; d < embeddingResult.size(); d++)
                        for (size_t w = 0; w < vocabulary.size(); w++)
                            localContextGrad[d][w] += embeddingResult[d] * softmaxGradient[w];

                    for (size_t d = 0; d < embeddingResult.size(); d++)
                        for (size_t w = 0; w < vocabulary.size(); w++)
                            localEmbeddingGrad[currentWordIndex][d] += softmaxGradient[w] * contextMatrix[d][w];
                }
            }
        }

        // Sum thread-local gradients into global gradients
        for (int t = 0; t < nThreads; ++t)
        {
            for (size_t d = 0; d < contextMatrix.size(); d++)
                for (size_t w = 0; w < contextMatrix[0].size(); w++)
                    contextMatrixGrad[d][w] += threadContextGrads[t][d][w];

            for (size_t i = 0; i < embeddingMatrix.size(); i++)
                for (size_t d = 0; d < embeddingMatrix[0].size(); d++)
                    embeddingMatrixGrad[i][d] += threadEmbeddingGrads[t][i][d];

            totalLoss += threadLosses[t];
        }

        // Update weights (outside parallel region)
        for (size_t d = 0; d < contextMatrix.size(); d++)
            for (size_t w = 0; w < contextMatrix[0].size(); w++)
                contextMatrix[d][w] -= learningRate * contextMatrixGrad[d][w];

        for (size_t i = 0; i < embeddingMatrix.size(); i++)
            for (size_t d = 0; d < embeddingMatrix[0].size(); d++)
                embeddingMatrix[i][d] -= learningRate * embeddingMatrixGrad[i][d];

        if (epoch % 10 == 0 || epoch == numEpochs - 1)
            cout << "Epoch " << epoch << ", Average Loss: " << totalLoss / words.size() << endl;
    }
}

// Interactive prediction loop
void interactivePrediction(
    const vector<vector<double>> &embeddingMatrix,
    const vector<vector<double>> &contextMatrix,
    const vector<string> &vocabulary,
    const unordered_map<string, size_t> &wordToIndex)
{
    cout << "Word Prediction Mode (enter 'exit' to quit)" << endl;
    cout << "------------------------" << endl;
    string inputWord;
    while (true)
    {
        cout << "Enter a word: ";
        cin >> inputWord;
        if (inputWord == "exit")
            break;
        auto wordIt = wordToIndex.find(inputWord);
        if (wordIt == wordToIndex.end())
        {
            cout << "Word not in vocabulary. Please try another word." << endl;
            continue;
        }
        size_t wordIndex = wordIt->second;
        vector<double> wordEmbedding = embeddingMatrix[wordIndex];

        // Compute similarities for all words
        vector<pair<string, double>> similarities;
        for (size_t i = 0; i < vocabulary.size(); i++)
        {
            vector<double> contextVector(contextMatrix.size());
            for (size_t j = 0; j < contextMatrix.size(); j++)
                contextVector[j] = contextMatrix[j][i];
            double similarity = cosineSimilarity(wordEmbedding, contextVector);
            similarities.emplace_back(vocabulary[i], similarity);
        }

        // Sort by similarity in descending order
        sort(similarities.begin(), similarities.end(),
             [](const pair<string, double> &a, const pair<string, double> &b)
             {
                 return a.second > b.second;
             });

        // Output top 3 predictions
        cout << "Top 3 predictions:" << endl;
        for (size_t i = 0; i < min(size_t(3), similarities.size()); i++)
            cout << similarities[i].first << " (" << similarities[i].second * 100 << "% similarity)" << endl;
    }
}

int main()
{
    // Hyperparameters
    const int NUM_EPOCHS = 100;
    const double LEARNING_RATE = 0.1;

    // Read corpus from a text file
    vector<string> words;
    ifstream file("output.txt");
    if (file.is_open())
    {
        string word;
        while (file >> word)
            words.push_back(word);
        file.close();
    }
    else
    {
        cerr << "Error: Unable to open file 'output.txt'" << endl;
        return 1;
    }

    // Build vocabulary / remove duplicates
    vector<string> vocabulary = buildVocabulary(words);

    // Build fast word-to-index map
    unordered_map<string, size_t> wordToIndex;
    for (size_t i = 0; i < vocabulary.size(); ++i)
        wordToIndex[vocabulary[i]] = i;

    // Initialize matrices
    vector<vector<double>> embeddingMatrix;
    fillMatrixWithRandomValues(embeddingMatrix, vocabulary.size(), 3);
    vector<vector<double>> contextMatrix;
    fillMatrixWithRandomValues(contextMatrix, 3, vocabulary.size());

    // Print initial matrices (optional, comment out for large vocab)
    // printMatrix(embeddingMatrix, vocabulary, "Embedding Matrix:");
    // printContextMatrix(contextMatrix, "Context Matrix:");

    // Train model
    trainModel(embeddingMatrix, contextMatrix, words, vocabulary, wordToIndex, NUM_EPOCHS, LEARNING_RATE);

    // Print final matrices (optional, comment out for large vocab)
    // printMatrix(embeddingMatrix, vocabulary, "Embedding Matrix:");
    // printContextMatrix(contextMatrix, "Context Matrix:");

    // Interactive prediction
    interactivePrediction(embeddingMatrix, contextMatrix, vocabulary, wordToIndex);

    return 0;
}